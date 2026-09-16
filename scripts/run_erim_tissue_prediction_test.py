#!/usr/bin/env python3
"""Out-of-sample causal test: can a pre-campaign tissue basket model predict the ERIM coupon lottery?

Protocol (fixed before any prediction was inspected)
----------------------------------------------------
Model.  The additive energy basket model restricted to one category. Items are 11 tissue
brands; a household-week basket S (possibly empty) has probability
    p(S) proportional to exp(sum_{j in S} u_hjt - rho_|S|),  rho_0 = rho_1 = 0,
    u_hjt = b_j + alpha_h + beta_hj - tau * dev_jt,
where dev_jt is the Sioux Falls chain log-price deviation of brand j (retail file only).
The empty basket is included so that category incidence is modeled. A coupon of face
value v held by the household lowers the Cottonelle price it faces:
    u_hCt += -tau * (log(max(pbar_t - v, 0.1 pbar_t)) - log pbar_t).
A dropped coupon is noticed with probability q, retained each week with probability r
and, when Cottonelle is bought while it is held, redeemed with probability pi (and then
consumed). The holding state is latent and integrated out by a forward recursion.

Training data.  All Sioux Falls panel households, weeks 198514-198532: before any
cell-specific coupon. The only test coupon in that window (code 17, $1.00, from 198529)
went to every arm and identifies q, r and pi. The cell file is never read by the fit or
by the prediction.

Prediction.  Every experimental household is simulated under all three coupon schedules
over weeks 198533-198625, using its observed shopping weeks and observed shelf prices.
The predicted effect of a schedule is the household-average difference from control.

Pre-registered criteria.
  C1  For arms A and B, the predicted effect on Cottonelle purchase-weeks is positive and
      lies inside the observed randomization 95% interval.
  C2  For arms A and B, the predicted effect on tissue purchase-weeks lies inside the
      observed 95% interval.
Secondary (reported, not pass/fail): other-brand purchase-weeks, arm-specific coupon
redemptions, level calibration of the control arm, and a sensitivity in which arm B also
receives the control $0.70 drops.

Amendment (added after an arm-blind level check, before any arm comparison was viewed).
Pooled over all experimental households, Cottonelle purchase-weeks per shopping week rose
from 0.043 in the fit window to 0.072 in the test window, while the primary model predicts
about 0.03: the brand was growing and the model has no trend. Because predicted coupon
effects scale with the base rate, a secondary "level-anchored" prediction is also
reported. In each 4-week block it adds a Cottonelle shift and a common tissue shift to the
utilities, chosen so that the equal mixture of the three schedules reproduces the pooled
observed Cottonelle and tissue purchase-weeks of all experimental households. Anchoring
uses pooled outcomes only, never arm labels. C1 and C2 are evaluated for both predictions;
the primary verdict remains the pre-registered one.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts/version4"))

from external_choice_data import read_erim_purchase, read_erim_retail  # noqa: E402
from run_erim_coupon_experiment import (  # noqa: E402
    RAW, analyse_window, read_assignment,
)

BRANDS = ("NRTHRN", "CHRMN", "CTNL", "FAM", "WHT", "BANNER", "SF", "CTL", "GENERIC", "SCOTT", "OTHER")
CTNL = BRANDS.index("CTNL")
FIT_WEEKS = (198514, 198532)
TEST_WEEKS = (198533, 198625)
COUPON_VALUE = {17: 1.00, 1: 1.00, 20: 1.00, 19: 0.70, 35: 0.70, 36: 0.75, 37: 0.70, 38: 0.50}
DROP_WEEK = {17: 198529, 1: 198533, 20: 198533, 19: 198537, 35: 198548, 36: 198549,
             37: 198605, 38: 198606}
FIT_COUPON = 17
SCHEDULES = {
    "control": (19, 35, 38),
    "A": (1, 19, 35, 37, 38),
    "B": (20, 36, 38),
}
SENSITIVITY_SCHEDULES = {"B_with_control_drops": (20, 19, 35, 36, 38)}
EPS = 1e-6
NEG = -1e4  # finite log(0): -inf breaks gradients through logaddexp/where
torch.set_default_dtype(torch.float64)


def erim_weeks(low: int, high: int) -> list[int]:
    weeks, week = [], low
    while week <= high:
        weeks.append(week)
        year, number = divmod(week, 100)
        week = (year + 1) * 100 + 1 if number >= 52 else week + 1
    return weeks


# --------------------------------------------------------------------------- data
def brand_of(description: pd.Series) -> pd.Series:
    first = description.str.split().str[0]
    return first.where(first.isin(BRANDS), "OTHER")


def load_panel(weeks: list[int]) -> dict:
    products = pd.read_fwf(RAW / "tissue_f6.dat", widths=[13, 30], names=["upc", "description"], dtype=str)
    brand_by_upc = dict(zip(products.upc, brand_of(products.description)))

    purchases = read_erim_purchase(RAW / "tissue_f1.dat")
    purchases = purchases[(purchases.market == 1) & purchases.week.isin(weeks)].copy()
    purchases["brand"] = purchases.upc.astype(str).str.zfill(13).map(brand_by_upc).fillna("OTHER")

    shopping = pd.read_fwf(RAW / "tissue_f7.dat", widths=[8, 6, 2, 12], dtype=str,
                           names=["hh", "week", "visits", "dollars"])
    shopping = shopping[shopping.hh.str[:2] == "01"]
    shopping = pd.DataFrame({"household": shopping.hh.astype(int), "week": shopping.week.astype(int)})
    shopping = shopping[shopping.week.isin(weeks)]

    households = np.array(sorted(set(shopping.household) | set(purchases.household)))
    h_index = {h: i for i, h in enumerate(households)}
    w_index = {w: i for i, w in enumerate(weeks)}
    H, T, J = len(households), len(weeks), len(BRANDS)
    opportunity = np.zeros((H, T), dtype=bool)
    opportunity[shopping.household.map(h_index), shopping.week.map(w_index)] = True
    basket = np.zeros((H, T, J), dtype=bool)
    hi = purchases.household.map(h_index).to_numpy()
    ti = purchases.week.map(w_index).to_numpy()
    ji = purchases.brand.map({b: j for j, b in enumerate(BRANDS)}).to_numpy()
    basket[hi, ti, ji] = True
    added = int((basket.any(2) & ~opportunity).sum())
    opportunity |= basket.any(2)
    redeemed = {}
    for code in COUPON_VALUE:
        flag = np.zeros((H, T), dtype=bool)
        rows = purchases[(purchases.special_coupon_code == code) & (purchases.brand == "CTNL")]
        flag[rows.household.map(h_index), rows.week.map(w_index)] = True
        redeemed[code] = flag
    return {"households": households, "weeks": weeks, "opportunity": opportunity,
            "basket": basket, "redeemed": redeemed, "purchase_weeks_without_shopping_record": added}


def load_prices(weeks: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Chain log-price deviations per brand-week and the Cottonelle shelf price level."""
    products = pd.read_fwf(RAW / "tissue_f6.dat", widths=[13, 30], names=["upc", "description"], dtype=str)
    brand_by_upc = dict(zip(products.upc, brand_of(products.description)))
    retail = read_erim_retail(RAW / "tissue_f10.dat")
    retail = retail[(retail.market == 1) & retail.week.isin(weeks)].copy()
    retail["upc_text"] = retail.upc.astype(str).str.zfill(13)
    retail["brand"] = retail.upc_text.map(brand_by_upc).fillna("OTHER")
    cell = retail.groupby(["upc_text", "week"]).agg(units=("units", "sum"), revenue=("revenue", "sum"),
                                                   brand=("brand", "first"))
    cell["log_price"] = np.log(cell.revenue / cell.units)
    grid = cell.log_price.unstack("week").reindex(columns=weeks).ffill(axis=1).bfill(axis=1)
    fit_columns = [w for w in weeks if FIT_WEEKS[0] <= w <= FIT_WEEKS[1]]
    reference = grid[fit_columns].mean(axis=1)
    weight = cell.reset_index()
    weight = weight[weight.week.isin(fit_columns)].groupby("upc_text").units.sum().reindex(grid.index).fillna(0)
    brand = cell.reset_index().groupby("upc_text").brand.first().reindex(grid.index)
    dev = np.zeros((len(BRANDS), len(weeks)))
    level = None
    for j, name in enumerate(BRANDS):
        members = brand.index[(brand == name) & (weight > 0)]
        if len(members) == 0:
            continue
        w = weight[members].to_numpy() / weight[members].sum()
        dev[j] = w @ (grid.loc[members].to_numpy() - reference[members].to_numpy()[:, None])
        if j == CTNL:
            level = np.exp(w @ grid.loc[members].to_numpy())
    return dev, level


# --------------------------------------------------------------------------- model
def log_esp(u: torch.Tensor, K: int) -> torch.Tensor:
    """log elementary symmetric polynomials e_0..e_K of exp(u) over the last axis."""
    shape = u.shape[:-1]
    out = torch.full((*shape, K + 1), NEG)
    out[..., 0] = 0.0
    for j in range(u.shape[-1]):
        shifted = out[..., :-1] + u[..., j:j + 1]
        out = torch.cat([out[..., :1], torch.logaddexp(out[..., 1:], shifted)], dim=-1)
    return out


class TissueModel(torch.nn.Module):
    def __init__(self, n_households: int, K: int):
        super().__init__()
        J = len(BRANDS)
        self.K = K
        self.b = torch.nn.Parameter(torch.zeros(J))
        self.alpha = torch.nn.Parameter(torch.zeros(n_households))
        self.beta = torch.nn.Parameter(torch.zeros(n_households, J))
        self.rho_free = torch.nn.Parameter(torch.full((K - 1,), 2.0))
        self.tau_raw = torch.nn.Parameter(torch.tensor(0.5))
        self.q_raw = torch.nn.Parameter(torch.tensor(-1.0))
        self.r_raw = torch.nn.Parameter(torch.tensor(1.0))
        self.pi_raw = torch.nn.Parameter(torch.tensor(1.0))
        self.fixed_tau = None

    def rho(self) -> torch.Tensor:
        return torch.cat([torch.zeros(2), self.rho_free])

    def tau(self) -> torch.Tensor:
        if self.fixed_tau is not None:
            return torch.tensor(self.fixed_tau)
        return torch.nn.functional.softplus(self.tau_raw)

    def exposure(self) -> dict:
        return {"q": torch.sigmoid(self.q_raw), "r": torch.sigmoid(self.r_raw), "pi": torch.sigmoid(self.pi_raw)}

    def utilities(self, dev: torch.Tensor) -> torch.Tensor:
        """[H, T, J] base utilities."""
        return (self.b + self.alpha[:, None] + self.beta)[:, None, :] - self.tau() * dev.T[None, :, :]

    def coupon_shift(self, level: torch.Tensor, value: float | torch.Tensor) -> torch.Tensor:
        paid = torch.clamp(level - value, min=0.1 * level)
        return -self.tau() * (torch.log(paid) - torch.log(level))

    def basket_log_prob(self, u: torch.Tensor, basket: torch.Tensor) -> torch.Tensor:
        size = basket.sum(-1).long().clamp(max=self.K)
        energy = (u * basket).sum(-1) - self.rho()[size]
        log_z = torch.logsumexp(log_esp(u, self.K) - self.rho(), dim=-1)
        return energy - log_z


def household_log_likelihood(model: TissueModel, data: dict, dev: torch.Tensor, level: torch.Tensor,
                             held_out: torch.Tensor | None = None, return_filter: bool = False):
    """Forward recursion over the latent holding state of the fit-window coupon."""
    u = model.utilities(dev)
    basket = data["basket_t"]
    base = model.basket_log_prob(u, basket)
    shifted = u.clone()
    shifted[..., CTNL] = shifted[..., CTNL] + model.coupon_shift(level, COUPON_VALUE[FIT_COUPON])[None, :]
    coupon = model.basket_log_prob(shifted, basket)
    observed = data["opportunity_t"] if held_out is None else data["opportunity_t"] & ~held_out
    bought = basket[..., CTNL].bool()
    flag = data["redeemed_t"][FIT_COUPON] & bought
    ex = model.exposure()
    log_q, log_1q = torch.log(ex["q"]), torch.log1p(-ex["q"])
    log_r, log_1r = torch.log(ex["r"]), torch.log1p(-ex["r"])
    log_pi, log_1pi = torch.log(ex["pi"]), torch.log1p(-ex["pi"])
    H, T = base.shape
    not_held = torch.zeros(H)
    held = torch.full((H,), NEG)
    drop = data["weeks"].index(DROP_WEEK[FIT_COUPON])
    impossible = np.log(EPS)
    for t in range(T):
        if t == drop:
            total = torch.logaddexp(not_held, held)
            held, not_held = total + log_q, total + log_1q
        elif t > drop:
            held, not_held = held + log_r, torch.logaddexp(not_held, held + log_1r)
        obs = observed[:, t]
        e_not = base[:, t] + torch.where(flag[:, t], impossible, 0.0)
        redeem = held + coupon[:, t] + log_pi
        keep = held + coupon[:, t] + torch.where(bought[:, t], log_1pi, 0.0)
        new_not = torch.where(flag[:, t], torch.logaddexp(not_held + e_not, redeem), not_held + e_not)
        new_held = torch.where(flag[:, t], torch.full_like(keep, NEG), keep)
        if t < drop:
            new_not, new_held = not_held + base[:, t], held
        not_held = torch.where(obs, new_not, not_held)
        held = torch.where(obs, new_held, held)
    total = torch.logaddexp(not_held, held)
    if return_filter:
        return total, torch.exp(held - total)
    return total


def held_out_log_likelihood(model: TissueModel, data: dict, dev: torch.Tensor, level: torch.Tensor,
                            held_out: torch.Tensor) -> float:
    with torch.no_grad():
        full = household_log_likelihood(model, data, dev, level)
        train = household_log_likelihood(model, data, dev, level, held_out)
    return float((full - train).sum() / held_out.sum())


def fit(model: TissueModel, data: dict, dev: torch.Tensor, level: torch.Tensor, ridge: float,
        weights: torch.Tensor | None = None, held_out: torch.Tensor | None = None,
        max_iter: int = 300) -> float:
    optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=max_iter,
                                  line_search_fn="strong_wolfe", tolerance_grad=1e-7)
    H = model.alpha.shape[0]
    w = torch.ones(H) if weights is None else weights

    def closure():
        optimizer.zero_grad()
        ll = household_log_likelihood(model, data, dev, level, held_out)
        penalty = 0.5 * ridge * (model.beta ** 2).sum() + 0.5 * (ridge / 10) * (model.alpha ** 2).sum()
        loss = -(w * ll).sum() / w.sum() + penalty / H
        loss.backward()
        return loss

    return float(optimizer.step(closure).detach())


# --------------------------------------------------------------------------- prediction
def ctnl_marginals(others: np.ndarray, u_c: np.ndarray, rho: np.ndarray, K: int):
    """P(Cottonelle in S), P(S nonempty), P(S has a non-Cottonelle brand).

    others: [H, J-1] utilities of the other brands; u_c: [..., H] Cottonelle utility.
    """
    log_e = log_esp(torch.from_numpy(others), K).numpy()
    without = np.logaddexp.reduce(log_e - rho, axis=1)            # baskets without Cottonelle
    with_c = np.logaddexp.reduce(log_e[:, :K] - rho[1:], axis=1)  # with it, divided by exp(u_C)
    log_z = np.logaddexp(without, u_c + with_c)
    p_c = np.exp(u_c + with_c - log_z)
    p_any = 1.0 - np.exp(-log_z)
    p_other = 1.0 - np.exp(np.logaddexp(0.0, u_c) - log_z)
    return p_c, p_any, p_other


def simulate(model: TissueModel, data: dict, dev: np.ndarray, level: np.ndarray, members: np.ndarray,
             initial_hold: np.ndarray, schedule: tuple[int, ...], replicates: int, seed: int,
             offset: np.ndarray | None = None) -> dict:
    """Rao-Blackwellized expected outcomes for each household under one coupon schedule."""
    rng = np.random.default_rng(seed)  # common random numbers across schedules
    with torch.no_grad():
        tau = float(model.tau())
        ex = {k: float(v) for k, v in model.exposure().items()}
        rho = model.rho().numpy()
        static = (model.b + model.alpha[:, None] + model.beta).numpy()[members]
    K = model.K
    H, T = len(members), len(data["weeks"])
    codes = sorted(set(schedule) | {FIT_COUPON})
    values = np.array([COUPON_VALUE[c] for c in codes])
    held = np.zeros((replicates, H, len(codes)), dtype=bool)
    held[:, :, codes.index(FIT_COUPON)] = rng.random((replicates, H)) < initial_hold[None, :]
    opportunity = data["opportunity"][members]
    out = {name: np.zeros(H) for name in ("ctnl_weeks", "tissue_weeks", "other_weeks")}
    out["ctnl_by_week"] = np.zeros(T)
    out["tissue_by_week"] = np.zeros(T)
    redemptions = {c: np.zeros(H) for c in codes}
    for t, week in enumerate(data["weeks"]):
        held &= rng.random(held.shape) < ex["r"]  # one retention step per week
        for c in schedule:
            if DROP_WEEK[c] == week:
                held[:, :, codes.index(c)] |= rng.random((replicates, H)) < ex["q"]
        draws = rng.random((replicates, H))
        redeem_draws = rng.random((replicates, H))
        if not opportunity[:, t].any():
            continue
        u = static - tau * dev[:, t][None, :]
        if offset is not None:
            u = u + offset[t, 1]
            u[:, CTNL] += offset[t, 0]
        best = np.where(held.any(2), (held * values).max(2), 0.0)
        paid = np.maximum(level[t] - best, 0.1 * level[t])
        u_c = u[:, CTNL][None, :] - tau * (np.log(paid) - np.log(level[t]))
        p_c, p_any, p_other = ctnl_marginals(np.delete(u, CTNL, axis=1), u_c, rho, K)
        active = opportunity[:, t][None, :]
        out["ctnl_weeks"] += (p_c * active).mean(0)
        out["tissue_weeks"] += (p_any * active).mean(0)
        out["other_weeks"] += (p_other * active).mean(0)
        out["ctnl_by_week"][t] = (p_c * active).mean(0).sum()
        out["tissue_by_week"][t] = (p_any * active).mean(0).sum()
        top = np.argmax(held * values, axis=2)
        has = held.any(2) & active
        for i, c in enumerate(codes):
            redemptions[c] += (p_c * ex["pi"] * (has & (top == i))).mean(0)
        use = has & (draws < p_c) & (redeem_draws < ex["pi"])
        r_idx, h_idx = np.nonzero(use)
        held[r_idx, h_idx, top[r_idx, h_idx]] = False
    out["test_coupon_redemptions"] = sum(redemptions[c] for c in codes if c not in (FIT_COUPON, 38))
    out["redemptions_by_code"] = redemptions
    return out


# --------------------------------------------------------------------------- driver
def to_tensors(data: dict) -> dict:
    data["basket_t"] = torch.from_numpy(data["basket"]).double()
    data["opportunity_t"] = torch.from_numpy(data["opportunity"])
    data["redeemed_t"] = {c: torch.from_numpy(v) for c, v in data["redeemed"].items()}
    return data


def run_fit(args, fit_data, dev_t, level_t, fixed_tau=None, weights=None, init=None):
    K = int(fit_data["basket"].sum(2).max())
    model = TissueModel(len(fit_data["households"]), K)
    if init is not None:
        model.load_state_dict(init)
    model.fixed_tau = fixed_tau
    for _ in range(args.lbfgs_rounds):
        loss = fit(model, fit_data, dev_t, level_t, args.ridge, weights=weights)
    return model, loss


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/erim_tissue_prediction_test")
    parser.add_argument("--ridge-grid", type=float, nargs="+", default=[1.0, 3.0, 10.0, 30.0])
    parser.add_argument("--lbfgs-rounds", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=4999)
    parser.add_argument("--plug-in-tau", type=float, default=1.4928,
                        help="1986-87 retail-only basket model tissue sensitivity (sensitivity run)")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--anchor-iterations", type=int, default=8)
    parser.add_argument("--anchor-replicates", type=int, default=50)
    parser.add_argument("--predict-only", action="store_true",
                        help="stop after freezing predictions, before reading the cell assignment")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    started = time.time()
    report = {"protocol": __doc__, "fit_weeks": FIT_WEEKS, "test_weeks": TEST_WEEKS, "brands": BRANDS,
              "schedules": SCHEDULES, "coupon_value": COUPON_VALUE, "drop_week": DROP_WEEK}

    # ---- fit on pre-campaign weeks (no cell information) ----
    fit_weeks = erim_weeks(*FIT_WEEKS)
    fit_data = to_tensors(load_panel(fit_weeks))
    all_weeks = erim_weeks(FIT_WEEKS[0], TEST_WEEKS[1])
    dev_all, level_all = load_prices(all_weeks)
    n_fit = len(fit_weeks)
    dev_t = torch.from_numpy(dev_all[:, :n_fit])
    level_t = torch.from_numpy(level_all[:n_fit])
    report["fit_data"] = {
        "households": int(len(fit_data["households"])),
        "opportunities": int(fit_data["opportunity"].sum()),
        "nonempty_baskets": int(fit_data["basket"].any(2).sum()),
        "ctnl_purchase_weeks": int(fit_data["basket"][..., CTNL].sum()),
        "fit_coupon_redemptions": int(fit_data["redeemed"][FIT_COUPON].sum()),
        "purchase_weeks_without_shopping_record": fit_data["purchase_weeks_without_shopping_record"],
        "price_deviation_sd_by_brand": dict(zip(BRANDS, dev_all[:, :n_fit].std(1).round(4).tolist())),
    }
    print(json.dumps(report["fit_data"], indent=1), flush=True)

    rng = torch.Generator().manual_seed(args.seed)
    held_out = (torch.rand(fit_data["opportunity_t"].shape, generator=rng) < 0.2) & fit_data["opportunity_t"]
    selection = {}
    for ridge in args.ridge_grid:
        model = TissueModel(len(fit_data["households"]), int(fit_data["basket"].sum(2).max()))
        for _ in range(args.lbfgs_rounds):
            fit(model, fit_data, dev_t, level_t, ridge, held_out=held_out)
        selection[ridge] = held_out_log_likelihood(model, fit_data, dev_t, level_t, held_out)
        print(f"ridge {ridge}: held-out log-lik per opportunity {selection[ridge]:.5f}", flush=True)
    args.ridge = max(selection, key=selection.get)
    report["ridge_selection"] = {"held_out_log_likelihood_per_opportunity": selection, "selected": args.ridge}

    model, loss = run_fit(args, fit_data, dev_t, level_t)
    with torch.no_grad():
        _, hold = household_log_likelihood(model, fit_data, dev_t, level_t, return_filter=True)
    params = {"tau": float(model.tau()), **{k: float(v) for k, v in model.exposure().items()},
              "rho": model.rho().tolist(), "b": dict(zip(BRANDS, model.b.tolist())), "loss": loss}
    report["fit"] = params
    print("fit", json.dumps(params), flush=True)

    # ---- predict every experimental household under every schedule ----
    test_weeks = erim_weeks(*TEST_WEEKS)
    test_data = load_panel(test_weeks)
    experiment = pd.read_fwf(RAW / "tissue_f4.dat", widths=[8, 2], names=["hh", "cell"], dtype=str)
    experimental = np.array(sorted(experiment.hh.astype(int)))   # household list only; cells unused
    fit_index = {h: i for i, h in enumerate(fit_data["households"])}
    test_index = {h: i for i, h in enumerate(test_data["households"])}
    # households absent from the fit window get population effects (zero random effects)
    extended = TissueModel(len(experimental), model.K)
    state = model.state_dict()
    rows = [fit_index.get(h, -1) for h in experimental]
    with torch.no_grad():
        for name, value in state.items():
            if name in ("alpha", "beta"):
                gathered = value[torch.tensor([max(r, 0) for r in rows])].clone()
                gathered[torch.tensor([r < 0 for r in rows])] = 0.0
                getattr(extended, name).copy_(gathered)
            else:
                getattr(extended, name).copy_(value)
    extended.fixed_tau = model.fixed_tau
    initial_hold = np.array([float(hold[r]) if r >= 0 else 0.0 for r in rows])
    test_panel = {"weeks": test_weeks,
                  "opportunity": np.zeros((len(experimental), len(test_weeks)), dtype=bool)}
    present = np.array([test_index.get(h, -1) for h in experimental])
    test_panel["opportunity"][present >= 0] = test_data["opportunity"][present[present >= 0]]
    dev_test = dev_all[:, n_fit:]
    level_test = level_all[n_fit:]
    members = np.arange(len(experimental))

    def predict(net: TissueModel, schedules: dict, seed: int, offset=None, replicates=None) -> dict:
        return {arm: simulate(net, test_panel, dev_test, level_test, members, initial_hold,
                              schedule, replicates or args.replicates, seed, offset)
                for arm, schedule in schedules.items()}

    # arm-blind pooled test-window levels of the experimental households (amendment)
    pooled_basket = np.zeros((len(experimental), len(test_weeks), len(BRANDS)), dtype=bool)
    pooled_basket[present >= 0] = test_data["basket"][present[present >= 0]]
    block = np.arange(len(test_weeks)) // 4
    pooled_ctnl = np.bincount(block, pooled_basket[..., CTNL].sum(0))
    pooled_tissue = np.bincount(block, pooled_basket.any(2).sum(0))

    def anchor(net: TissueModel, seed: int, iterations: int) -> np.ndarray:
        offset = np.zeros((len(test_weeks), 2))
        for _ in range(iterations):
            sims = predict(net, SCHEDULES, seed, offset, replicates=args.anchor_replicates)
            ctnl = np.mean([np.bincount(block, s["ctnl_by_week"]) for s in sims.values()], axis=0)
            tissue = np.mean([np.bincount(block, s["tissue_by_week"]) for s in sims.values()], axis=0)
            offset[:, 1] += np.log(pooled_tissue / tissue)[block]
            offset[:, 0] += np.log(pooled_ctnl / ctnl)[block] - np.log(pooled_tissue / tissue)[block]
        return offset

    def effects(sim: dict) -> dict:
        out = {}
        for arm in [a for a in sim if a != "control"]:
            out[arm] = {k: float((sim[arm][k] - sim["control"][k]).mean())
                        for k in ("ctnl_weeks", "tissue_weeks", "other_weeks", "test_coupon_redemptions")}
        out["control_level"] = {k: float(sim["control"][k].mean())
                                for k in ("ctnl_weeks", "tissue_weeks", "other_weeks", "test_coupon_redemptions")}
        return out

    all_schedules = {**SCHEDULES, **SENSITIVITY_SCHEDULES}
    point = predict(extended, all_schedules, args.seed)
    report["prediction"] = effects(point)
    per_code = {arm: {str(c): float(v.mean()) for c, v in point[arm]["redemptions_by_code"].items()}
                for arm in point}
    report["prediction"]["redemptions_by_code"] = per_code
    offset = anchor(extended, args.seed, args.anchor_iterations)
    anchored = predict(extended, all_schedules, args.seed, offset)
    report["prediction_level_anchored"] = effects(anchored)
    report["prediction_level_anchored"]["offsets_by_block"] = offset[::4].tolist()
    n_exp = len(experimental)
    report["prediction_level_anchored"]["pooled_fit_check"] = {
        "observed_ctnl_weeks_per_household": float(pooled_ctnl.sum() / n_exp),
        "mixture_ctnl_weeks_per_household": float(np.mean([anchored[a]["ctnl_by_week"].sum() for a in SCHEDULES]) / n_exp),
        "observed_tissue_weeks_per_household": float(pooled_tissue.sum() / n_exp),
        "mixture_tissue_weeks_per_household": float(np.mean([anchored[a]["tissue_by_week"].sum() for a in SCHEDULES]) / n_exp)}
    print("level-anchored", json.dumps(report["prediction_level_anchored"], indent=1), flush=True)
    print("prediction", json.dumps(report["prediction"], indent=1), flush=True)

    # Bayesian-bootstrap parameter uncertainty (household weights), warm-started refits
    boot = []
    for b in range(args.bootstrap):
        weights = torch.distributions.Exponential(1.0).sample((len(fit_data["households"]),))
        bm, _ = run_fit(args, fit_data, dev_t, level_t, weights=weights, init=model.state_dict())
        with torch.no_grad():
            for name, value in bm.state_dict().items():
                if name in ("alpha", "beta"):
                    gathered = value[torch.tensor([max(r, 0) for r in rows])].clone()
                    gathered[torch.tensor([r < 0 for r in rows])] = 0.0
                    getattr(extended, name).copy_(gathered)
                else:
                    getattr(extended, name).copy_(value)
        eff = effects(predict(extended, SCHEDULES, args.seed + 1000 + b))
        boot_offset = anchor(extended, args.seed + 1000 + b, args.anchor_iterations)
        eff["anchored"] = effects(predict(extended, SCHEDULES, args.seed + 1000 + b, boot_offset))
        eff["tau"] = float(bm.tau()); eff.update({k: float(v) for k, v in bm.exposure().items()})
        boot.append(eff)
        print(f"bootstrap {b + 1}/{args.bootstrap}: A ctnl {eff['A']['ctnl_weeks']:+.3f} "
              f"B ctnl {eff['B']['ctnl_weeks']:+.3f} tau {eff['tau']:.3f}", flush=True)
    if boot:
        report["prediction_interval_90"] = {
            arm: {k: [float(np.quantile([e[arm][k] for e in boot], 0.05)),
                      float(np.quantile([e[arm][k] for e in boot], 0.95))]
                  for k in ("ctnl_weeks", "tissue_weeks", "other_weeks", "test_coupon_redemptions")}
            for arm in ("A", "B")}
        report["prediction_level_anchored_interval_90"] = {
            arm: {k: [float(np.quantile([e["anchored"][arm][k] for e in boot], 0.05)),
                      float(np.quantile([e["anchored"][arm][k] for e in boot], 0.95))]
                  for k in ("ctnl_weeks", "tissue_weeks", "other_weeks", "test_coupon_redemptions")}
            for arm in ("A", "B")}
        report["bootstrap_parameters"] = {k: [float(np.quantile([e[k] for e in boot], 0.05)),
                                              float(np.quantile([e[k] for e in boot], 0.95))]
                                          for k in ("tau", "q", "r", "pi")}

    # Sensitivity: tau fixed at the 1986-87 basket model value, other parameters refit
    plug, _ = run_fit(args, fit_data, dev_t, level_t, fixed_tau=args.plug_in_tau, init=model.state_dict())
    with torch.no_grad():
        for name, value in plug.state_dict().items():
            if name in ("alpha", "beta"):
                gathered = value[torch.tensor([max(r, 0) for r in rows])].clone()
                gathered[torch.tensor([r < 0 for r in rows])] = 0.0
                getattr(extended, name).copy_(gathered)
            else:
                getattr(extended, name).copy_(value)
    extended.fixed_tau = args.plug_in_tau
    report["sensitivity_plug_in_tau"] = {"tau": args.plug_in_tau,
                                         **{k: float(v) for k, v in plug.exposure().items()},
                                         "effects": effects(predict(extended, SCHEDULES, args.seed))}
    (args.output / "prediction.json").write_text(json.dumps(report, indent=2, default=str))
    print("predictions frozen to", args.output / "prediction.json", flush=True)
    if args.predict_only:
        return

    # ---- only now read the assignment and compare with the randomized outcomes ----
    assignment = read_assignment().set_index("household")
    observed = pd.DataFrame(index=assignment.index)
    t_index = {h: i for i, h in enumerate(test_data["households"])}
    rows_t = observed.index.map(lambda h: t_index.get(h, -1)).to_numpy()
    basket = np.zeros((len(observed), len(test_weeks), len(BRANDS)), dtype=bool)
    basket[rows_t >= 0] = test_data["basket"][rows_t[rows_t >= 0]]
    observed["ctnl_weeks"] = basket[..., CTNL].sum(1)
    observed["tissue_weeks"] = basket.any(2).sum(1)
    observed["other_weeks"] = np.delete(basket, CTNL, axis=2).any(2).sum(1)
    red = np.zeros(len(observed))
    for code in (1, 19, 20, 35, 36, 37):
        flags = np.zeros((len(observed), len(test_weeks)), dtype=bool)
        flags[rows_t >= 0] = test_data["redeemed"][code][rows_t[rows_t >= 0]]
        red += flags.sum(1)
    observed["test_coupon_redemptions"] = red
    frame = assignment.join(observed)
    columns = ["ctnl_weeks", "tissue_weeks", "other_weeks", "test_coupon_redemptions"]
    measured = analyse_window(frame, columns, args.permutations, args.seed)
    comparison, verdict = {}, {}
    for arm in ("A", "B"):
        comparison[arm] = {}
        for k in columns:
            m = measured[k][f"{arm}_minus_control"]
            lo_band, hi_band = m["randomization_95_null_band"]
            interval = [m["difference"] - hi_band, m["difference"] - lo_band]
            predicted = report["prediction"][arm][k]
            comparison[arm][k] = {"observed": m["difference"], "observed_95": interval,
                                  "p": m["randomization_p_two_sided"], "predicted": predicted,
                                  "predicted_90": report.get("prediction_interval_90", {}).get(arm, {}).get(k),
                                  "plug_in_tau_predicted": report["sensitivity_plug_in_tau"]["effects"][arm][k],
                                  "level_anchored_predicted": report["prediction_level_anchored"][arm][k],
                                  "level_anchored_90": report.get("prediction_level_anchored_interval_90", {}).get(arm, {}).get(k),
                                  "level_anchored_inside": bool(interval[0] <= report["prediction_level_anchored"][arm][k] <= interval[1]),
                                  "inside": bool(interval[0] <= predicted <= interval[1])}
        verdict[f"C1_{arm}"] = bool(comparison[arm]["ctnl_weeks"]["inside"]
                                    and report["prediction"][arm]["ctnl_weeks"] > 0)
        verdict[f"C2_{arm}"] = comparison[arm]["tissue_weeks"]["inside"]
        verdict[f"level_anchored_C1_{arm}"] = bool(comparison[arm]["ctnl_weeks"]["level_anchored_inside"]
                                                   and report["prediction_level_anchored"][arm]["ctnl_weeks"] > 0)
        verdict[f"level_anchored_C2_{arm}"] = comparison[arm]["tissue_weeks"]["level_anchored_inside"]
    report["observed_control_level"] = {k: measured[k]["mean_by_arm"]["control"] for k in columns}
    report["comparison"] = comparison
    report["verdict"] = verdict
    report["runtime_seconds"] = round(time.time() - started, 1)
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"comparison": comparison, "verdict": verdict,
                      "observed_control_level": report["observed_control_level"],
                      "predicted_control_level": report["prediction"]["control_level"]}, indent=1))


if __name__ == "__main__":
    main()
