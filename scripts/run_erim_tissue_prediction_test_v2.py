#!/usr/bin/env python3
"""Re-test of the tissue basket model against the ERIM coupon lottery, with three fixes.

Disclosure
----------
Version 1 (`run_erim_tissue_prediction_test.py`) was blind. This version was designed after
its results and the experiment's outcomes were known, so it is not a blind test. To limit
that, (i) every fix is motivated by a mechanism found in pre-campaign or store data, (ii) no
parameter is fitted or tuned on any test-period panel outcome, (iii) no level anchoring is
used, and (iv) the criteria are stricter and include every quantity version 1 got wrong.

Fixes
-----
1. Launch availability and awareness. Cottonelle launched in Sioux Falls during the fit
   window (first store sale 1985 week 21; 16 of 17 stores by week 30). Version 1 treated
   pre-launch weeks as refusals. Now, for household h in week t,
       u_hCt += log(max(a_ht, 0.02)) + theta * log(1 + s_t),
   where a_ht is h's pre-campaign share of purchase-days (all eight ERIM categories) at
   stores whose retail file already shows Cottonelle sales, and s_t is weeks since the
   chain launch. theta is fitted on the fit window and extrapolated.
2. Direct redemption. Holding a coupon of value v adds a separate "Cottonelle with coupon"
   alternative with weight x = kappa * v / pbar_t. Cottonelle's basket utility becomes
   u_C + log(1 + x); given Cottonelle is bought, the coupon is redeemed with probability
   x / (1 + x) and consumed. A held coupon raises purchasing only through its use.
3. Separate coupon response. kappa is its own parameter, not the shelf-price tau, and a
   zero-value coupon has no effect.

Fit window, latent holding state (noticed with q, retained weekly with r), prediction and
observed randomization inference are as in version 1.

Criteria (all must hold)
------------------------
  C1  Cottonelle purchase-weeks effect positive and inside the observed 95% interval (A, B).
  C2  Tissue purchase-weeks effect inside the observed 95% interval (A, B).
  C3  Arm-specific test-coupon redemption effect inside the observed 95% interval (A, B).
  C4  A minus B Cottonelle purchase-weeks inside the observed 95% interval.
  C5  Control-arm Cottonelle purchase-weeks within +/-20% of observed, and control-arm
      test-coupon redemptions within +/-25% of observed.
Secondary, never examined before: effects in the first (1985 wk 33 - 1986 wk 3) and second
(1986 wk 4 - 25) halves of the test window. Sensitivities: theta = 0; arm B also receives
the control $0.70 drops.
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
from run_erim_coupon_experiment import RAW, analyse_window, read_assignment  # noqa: E402
from run_erim_tissue_prediction_test import (  # noqa: E402
    BRANDS, COUPON_VALUE, CTNL, DROP_WEEK, FIT_COUPON, FIT_WEEKS, NEG, SCHEDULES,
    SENSITIVITY_SCHEDULES, TEST_WEEKS, ctnl_marginals, erim_weeks, load_panel, load_prices,
    log_esp, to_tensors,
)

COTTONELLE = ("0054000494000", "0054000494100", "0054000494200", "0054000496000")
LAUNCH_WEEK = 198521
AVAILABILITY_FLOOR = 0.02
CATEGORY_FILES = ("brownie/brownie_f1.dat", "ddinner/ddinner_f1.dat", "ketchup/ketchup_f1.dat",
                  "marg/marg_f1.dat", "pbutter/pbut_f1.dat", "sugar/sugar_f1.dat",
                  "tissue/tissue_f1.dat", "tuna/tuna_f1.dat")
OUTCOMES = ("ctnl_weeks", "tissue_weeks", "other_weeks", "test_coupon_redemptions")
LEVEL_TOLERANCE = {"ctnl_weeks": 0.20, "test_coupon_redemptions": 0.25}
torch.set_default_dtype(torch.float64)


# --------------------------------------------------------------------------- availability
def store_launch_weeks() -> dict[int, int]:
    retail = read_erim_retail(RAW / "tissue_f10.dat")
    retail = retail[(retail.market == 1) & retail.upc.astype(str).str.zfill(13).isin(COTTONELLE)]
    return retail[retail.units > 0].groupby("store").week.min().to_dict()


def household_store_weights() -> tuple[pd.DataFrame, pd.Series]:
    """Pre-campaign share of purchase-days per store, from all eight ERIM categories."""
    frames = []
    for name in CATEGORY_FILES:
        rows = read_erim_purchase(ROOT / "data/erim_basket/raw" / name)
        frames.append(rows[(rows.market == 1) & (rows.week <= FIT_WEEKS[1])][["household", "week", "dow", "store"]])
    visits = pd.concat(frames).drop_duplicates()
    weights = visits.groupby(["household", "store"]).size().unstack(fill_value=0)
    pooled = weights.sum(0) / weights.values.sum()
    return weights.div(weights.sum(1), axis=0), pooled


def availability(households: np.ndarray, weeks: list[int], weights: pd.DataFrame, pooled: pd.Series,
                 launch: dict[int, int]) -> np.ndarray:
    stores = list(pooled.index)
    carries = np.array([[launch.get(s, 10 ** 9) <= w for w in weeks] for s in stores], dtype=float)
    table = weights.reindex(columns=stores, fill_value=0.0).reindex(households)
    missing = table.isna().any(axis=1).to_numpy()
    matrix = table.fillna(0.0).to_numpy()
    matrix[missing] = pooled.to_numpy()
    return matrix @ carries  # [H, T]


def weeks_since_launch(weeks: list[int]) -> np.ndarray:
    all_weeks = erim_weeks(198501, max(weeks))
    index = {w: i for i, w in enumerate(all_weeks)}
    return np.array([max(0, index[w] - index[LAUNCH_WEEK]) for w in weeks], dtype=float)


# --------------------------------------------------------------------------- model
class TissueModelV2(torch.nn.Module):
    def __init__(self, n_households: int, K: int):
        super().__init__()
        J = len(BRANDS)
        self.K = K
        self.b = torch.nn.Parameter(torch.zeros(J))
        self.alpha = torch.nn.Parameter(torch.zeros(n_households))
        self.beta = torch.nn.Parameter(torch.zeros(n_households, J))
        self.rho_free = torch.nn.Parameter(torch.full((K - 1,), 2.0))
        self.tau_raw = torch.nn.Parameter(torch.tensor(0.5))
        self.theta = torch.nn.Parameter(torch.tensor(0.0))
        self.q_raw = torch.nn.Parameter(torch.tensor(0.0))
        self.r_raw = torch.nn.Parameter(torch.tensor(1.0))
        self.kappa_raw = torch.nn.Parameter(torch.tensor(1.0))
        self.fixed_theta = None

    def rho(self):
        return torch.cat([torch.zeros(2), self.rho_free])

    def tau(self):
        return torch.nn.functional.softplus(self.tau_raw)

    def theta_value(self):
        return self.theta if self.fixed_theta is None else torch.tensor(self.fixed_theta)

    def exposure(self):
        return {"q": torch.sigmoid(self.q_raw), "r": torch.sigmoid(self.r_raw),
                "kappa": torch.nn.functional.softplus(self.kappa_raw)}

    def utilities(self, dev, log_avail, since):
        u = (self.b + self.alpha[:, None] + self.beta)[:, None, :] - self.tau() * dev.T[None, :, :]
        shift = torch.zeros_like(u)
        shift[..., CTNL] = log_avail + self.theta_value() * torch.log1p(since)[None, :]
        return u + shift

    def coupon_odds(self, value, level):
        return self.exposure()["kappa"] * value / level

    def basket_log_prob(self, u, basket):
        size = basket.sum(-1).long().clamp(max=self.K)
        energy = (u * basket).sum(-1) - self.rho()[size]
        return energy - torch.logsumexp(log_esp(u, self.K) - self.rho(), dim=-1)


def log_likelihood(model, data, dev, level, log_avail, since, held_out=None, return_filter=False):
    u = model.utilities(dev, log_avail, since)
    basket = data["basket_t"]
    base = model.basket_log_prob(u, basket)
    x = model.coupon_odds(COUPON_VALUE[FIT_COUPON], level)          # [T]
    held_u = u.clone()
    held_u[..., CTNL] = held_u[..., CTNL] + torch.log1p(x)[None, :]
    held = model.basket_log_prob(held_u, basket)
    log_redeem = torch.log(x) - torch.log1p(x)
    log_keep = -torch.log1p(x)
    observed = data["opportunity_t"] if held_out is None else data["opportunity_t"] & ~held_out
    bought = basket[..., CTNL].bool()
    flag = data["redeemed_t"][FIT_COUPON] & bought
    ex = model.exposure()
    H, T = base.shape
    not_held = torch.zeros(H)
    holding = torch.full((H,), NEG)
    drop = data["weeks"].index(DROP_WEEK[FIT_COUPON])
    impossible = float(np.log(1e-6))
    for t in range(T):
        if t == drop:
            total = torch.logaddexp(not_held, holding)
            holding, not_held = total + torch.log(ex["q"]), total + torch.log1p(-ex["q"])
        elif t > drop:
            holding, not_held = (holding + torch.log(ex["r"]),
                                 torch.logaddexp(not_held, holding + torch.log1p(-ex["r"])))
        if t < drop:
            new_not, new_hold = not_held + base[:, t], holding
        else:
            e_not = base[:, t] + torch.where(flag[:, t], impossible, 0.0)
            redeem = holding + held[:, t] + log_redeem[t]
            keep = holding + held[:, t] + torch.where(bought[:, t], log_keep[t], 0.0)
            new_not = torch.where(flag[:, t], torch.logaddexp(not_held + e_not, redeem), not_held + e_not)
            new_hold = torch.where(flag[:, t], torch.full_like(keep, NEG), keep)
        obs = observed[:, t]
        not_held = torch.where(obs, new_not, not_held)
        holding = torch.where(obs, new_hold, holding)
    total = torch.logaddexp(not_held, holding)
    return (total, torch.exp(holding - total)) if return_filter else total


def fit(model, data, dev, level, log_avail, since, ridge, weights=None, held_out=None, rounds=3):
    H = model.alpha.shape[0]
    w = torch.ones(H) if weights is None else weights
    loss = None
    for _ in range(rounds):
        optimizer = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=300,
                                      line_search_fn="strong_wolfe", tolerance_grad=1e-7)

        def closure():
            optimizer.zero_grad()
            ll = log_likelihood(model, data, dev, level, log_avail, since, held_out)
            penalty = 0.5 * ridge * (model.beta ** 2).sum() + 0.05 * ridge * (model.alpha ** 2).sum()
            value = -(w * ll).sum() / w.sum() + penalty / H
            value.backward()
            return value

        loss = float(optimizer.step(closure).detach())
    return loss


# --------------------------------------------------------------------------- simulation
def simulate(model, weeks, opportunity, log_avail, since, dev, level, initial_hold, schedule,
             replicates, seed):
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        tau = float(model.tau())
        theta = float(model.theta_value())
        ex = {k: float(v) for k, v in model.exposure().items()}
        rho = model.rho().numpy()
        static = (model.b + model.alpha[:, None] + model.beta).numpy()
    K, (H, T) = model.K, opportunity.shape
    codes = sorted(set(schedule) | {FIT_COUPON})
    values = np.array([COUPON_VALUE[c] for c in codes])
    holding = np.zeros((replicates, H, len(codes)), dtype=bool)
    holding[:, :, codes.index(FIT_COUPON)] = rng.random((replicates, H)) < initial_hold[None, :]
    out = {k: np.zeros((H, T)) for k in OUTCOMES}
    for t, week in enumerate(weeks):
        holding &= rng.random(holding.shape) < ex["r"]
        for c in schedule:
            if DROP_WEEK[c] == week:
                holding[:, :, codes.index(c)] |= rng.random((replicates, H)) < ex["q"]
        buy_draw, redeem_draw = rng.random((replicates, H)), rng.random((replicates, H))
        active = opportunity[:, t][None, :]
        if not active.any():
            continue
        u = static - tau * dev[:, t][None, :]
        u[:, CTNL] += log_avail[:, t] + theta * np.log1p(since[t])
        best = np.where(holding.any(2), (holding * values).max(2), 0.0)
        x = ex["kappa"] * best / level[t]
        u_c = u[:, CTNL][None, :] + np.log1p(x)
        p_c, p_any, p_other = ctnl_marginals(np.delete(u, CTNL, axis=1), u_c, rho, K)
        redeem_share = x / (1.0 + x)
        top = np.argmax(holding * values, axis=2)
        has = holding.any(2) & active
        out["ctnl_weeks"][:, t] = (p_c * active).mean(0)
        out["tissue_weeks"][:, t] = (p_any * active).mean(0)
        out["other_weeks"][:, t] = (p_other * active).mean(0)
        arm_specific = np.isin(np.array(codes)[top], [c for c in schedule if c != 38])
        out["test_coupon_redemptions"][:, t] = (p_c * redeem_share * (has & arm_specific)).mean(0)
        use = has & (buy_draw < p_c) & (redeem_draw < redeem_share)
        r_idx, h_idx = np.nonzero(use)
        holding[r_idx, h_idx, top[r_idx, h_idx]] = False
    return out


# --------------------------------------------------------------------------- driver
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/erim_tissue_prediction_test_v2")
    parser.add_argument("--ridge-grid", type=float, nargs="+", default=[0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--bootstrap", type=int, default=20)
    parser.add_argument("--permutations", type=int, default=4999)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    started = time.time()
    report = {"protocol": __doc__}

    fit_weeks = erim_weeks(*FIT_WEEKS)
    test_weeks = erim_weeks(*TEST_WEEKS)
    fit_data = to_tensors(load_panel(fit_weeks))
    dev_all, level_all = load_prices(erim_weeks(FIT_WEEKS[0], TEST_WEEKS[1]))
    n_fit = len(fit_weeks)
    launch = store_launch_weeks()
    weights, pooled = household_store_weights()
    avail_fit = availability(fit_data["households"], fit_weeks, weights, pooled, launch)
    report["availability"] = {"store_launch_week": {int(k): int(v) for k, v in launch.items()},
                              "fit_mean_by_week": dict(zip(fit_weeks, avail_fit.mean(0).round(3).tolist()))}
    tensors = dict(dev=torch.from_numpy(dev_all[:, :n_fit]), level=torch.from_numpy(level_all[:n_fit]),
                   log_avail=torch.from_numpy(np.log(np.maximum(avail_fit, AVAILABILITY_FLOOR))),
                   since=torch.from_numpy(weeks_since_launch(fit_weeks)))
    K = int(fit_data["basket"].sum(2).max())
    H = len(fit_data["households"])

    held_out = (torch.rand(fit_data["opportunity_t"].shape, generator=torch.Generator().manual_seed(args.seed))
                < 0.2) & fit_data["opportunity_t"]
    selection = {}
    for ridge in args.ridge_grid:
        model = TissueModelV2(H, K)
        fit(model, fit_data, **tensors, ridge=ridge, held_out=held_out, rounds=args.rounds)
        with torch.no_grad():
            full = log_likelihood(model, fit_data, **tensors)
            train = log_likelihood(model, fit_data, **tensors, held_out=held_out)
        selection[ridge] = float((full - train).sum() / held_out.sum())
        print(f"ridge {ridge}: held-out log-lik per opportunity {selection[ridge]:.5f}", flush=True)
    ridge = max(selection, key=selection.get)
    report["ridge_selection"] = {"held_out": selection, "selected": ridge}

    model = TissueModelV2(H, K)
    report["fit_loss"] = fit(model, fit_data, **tensors, ridge=ridge, rounds=args.rounds)
    with torch.no_grad():
        _, hold = log_likelihood(model, fit_data, **tensors, return_filter=True)
    report["fit"] = {"tau": float(model.tau()), "theta": float(model.theta),
                     **{k: float(v) for k, v in model.exposure().items()},
                     "rho": model.rho().tolist(), "b": dict(zip(BRANDS, model.b.tolist()))}
    print("fit", json.dumps(report["fit"]), flush=True)

    # ---- prediction for every experimental household under every schedule ----
    test_data = load_panel(test_weeks)
    experiment = pd.read_fwf(RAW / "tissue_f4.dat", widths=[8, 2], names=["hh", "cell"], dtype=str)
    experimental = np.array(sorted(experiment.hh.astype(int)))  # household list only
    fit_rows = np.array([{h: i for i, h in enumerate(fit_data["households"])}.get(h, -1) for h in experimental])
    test_rows = np.array([{h: i for i, h in enumerate(test_data["households"])}.get(h, -1) for h in experimental])
    opportunity = np.zeros((len(experimental), len(test_weeks)), dtype=bool)
    opportunity[test_rows >= 0] = test_data["opportunity"][test_rows[test_rows >= 0]]
    log_avail_test = np.log(np.maximum(availability(experimental, test_weeks, weights, pooled, launch),
                                       AVAILABILITY_FLOOR))
    since_test = weeks_since_launch(test_weeks)
    dev_test, level_test = dev_all[:, n_fit:], level_all[n_fit:]
    initial_hold = np.where(fit_rows >= 0, hold.detach().numpy()[np.maximum(fit_rows, 0)], 0.0)
    half = np.array([w <= 198603 for w in test_weeks])

    def restrict(source: TissueModelV2) -> TissueModelV2:
        net = TissueModelV2(len(experimental), K)
        with torch.no_grad():
            for name, value in source.state_dict().items():
                if name in ("alpha", "beta"):
                    gathered = value[torch.from_numpy(np.maximum(fit_rows, 0))].clone()
                    gathered[torch.from_numpy(fit_rows < 0)] = 0.0
                    getattr(net, name).copy_(gathered)
                else:
                    getattr(net, name).copy_(value)
        net.fixed_theta = source.fixed_theta
        return net

    def predict(source, schedules, seed):
        net = restrict(source)
        sims = {arm: simulate(net, test_weeks, opportunity, log_avail_test, since_test, dev_test,
                              level_test, initial_hold, schedule, args.replicates, seed)
                for arm, schedule in schedules.items()}
        summary = {"control_level": {k: float(sims["control"][k].sum(1).mean()) for k in OUTCOMES}}
        for arm in [a for a in sims if a != "control"]:
            summary[arm] = {k: float((sims[arm][k] - sims["control"][k]).sum(1).mean()) for k in OUTCOMES}
            summary[arm]["by_half"] = {
                name: {k: float((sims[arm][k] - sims["control"][k])[:, mask].sum(1).mean()) for k in OUTCOMES}
                for name, mask in (("first", half), ("second", ~half))}
        if "A" in sims and "B" in sims:
            summary["A_minus_B"] = {k: float((sims["A"][k] - sims["B"][k]).sum(1).mean()) for k in OUTCOMES}
        return summary

    report["prediction"] = predict(model, {**SCHEDULES, **SENSITIVITY_SCHEDULES}, args.seed)
    print("prediction", json.dumps(report["prediction"], indent=1), flush=True)
    no_trend = TissueModelV2(H, K)
    no_trend.load_state_dict(model.state_dict())
    no_trend.fixed_theta = 0.0
    fit(no_trend, fit_data, **tensors, ridge=ridge, rounds=args.rounds)
    report["sensitivity_no_trend"] = {"fit": {**{k: float(v) for k, v in no_trend.exposure().items()},
                                              "tau": float(no_trend.tau())},
                                      "prediction": predict(no_trend, SCHEDULES, args.seed)}

    boot = []
    for b in range(args.bootstrap):
        bm = TissueModelV2(H, K)
        bm.load_state_dict(model.state_dict())
        w = torch.distributions.Exponential(1.0).sample((H,))
        fit(bm, fit_data, **tensors, ridge=ridge, weights=w, rounds=max(1, args.rounds - 1))
        pred = predict(bm, SCHEDULES, args.seed + 1000 + b)
        pred["parameters"] = {"tau": float(bm.tau()), "theta": float(bm.theta),
                              **{k: float(v) for k, v in bm.exposure().items()}}
        boot.append(pred)
        print(f"bootstrap {b + 1}/{args.bootstrap}: A ctnl {pred['A']['ctnl_weeks']:+.3f} "
              f"B ctnl {pred['B']['ctnl_weeks']:+.3f} theta {pred['parameters']['theta']:.3f} "
              f"kappa {pred['parameters']['kappa']:.3f}", flush=True)

    def band(values):
        return [float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))] if values else None

    report["prediction_interval_90"] = {
        **{arm: {k: band([p[arm][k] for p in boot]) for k in OUTCOMES} for arm in ("A", "B", "A_minus_B")},
        "control_level": {k: band([p["control_level"][k] for p in boot]) for k in OUTCOMES},
        "parameters": {k: band([p["parameters"][k] for p in boot]) for k in ("tau", "theta", "q", "r", "kappa")}}
    (args.output / "prediction.json").write_text(json.dumps(report, indent=2, default=str))
    print("predictions frozen to", args.output / "prediction.json", flush=True)
    if args.predict_only:
        return

    # ---- only now read the assignment ----
    assignment = read_assignment().set_index("household")
    rows = assignment.index.map({h: i for i, h in enumerate(test_data["households"])}.get).to_numpy()
    rows = np.array([-1 if r is None or pd.isna(r) else int(r) for r in rows])
    basket = np.zeros((len(assignment), len(test_weeks), len(BRANDS)), dtype=bool)
    basket[rows >= 0] = test_data["basket"][rows[rows >= 0]]
    redeemed = np.zeros((len(assignment), len(test_weeks)))
    for code in (1, 19, 20, 35, 36, 37):
        flags = np.zeros((len(assignment), len(test_weeks)), dtype=bool)
        flags[rows >= 0] = test_data["redeemed"][code][rows[rows >= 0]]
        redeemed += flags
    weekly = {"ctnl_weeks": basket[..., CTNL], "tissue_weeks": basket.any(2),
              "other_weeks": np.delete(basket, CTNL, axis=2).any(2), "test_coupon_redemptions": redeemed}
    measured = {}
    for name, mask in (("all", np.ones(len(test_weeks), bool)), ("first", half), ("second", ~half)):
        frame = assignment.copy()
        for k, v in weekly.items():
            frame[k] = v[:, mask].sum(1)
        measured[name] = analyse_window(frame, list(OUTCOMES), args.permutations, args.seed)

    def observed(name, k, contrast):
        m = measured[name][k][contrast]
        lo, hi = m["randomization_95_null_band"]
        return {"observed": m["difference"], "observed_95": [m["difference"] - hi, m["difference"] - lo],
                "p": m["randomization_p_two_sided"]}

    comparison = {}
    for arm, contrast in (("A", "A_minus_control"), ("B", "B_minus_control"), ("A_minus_B", "A_minus_B")):
        comparison[arm] = {}
        for k in OUTCOMES:
            entry = observed("all", k, contrast)
            entry.update(predicted=report["prediction"][arm][k],
                         predicted_90=report["prediction_interval_90"][arm][k],
                         predicted_no_trend=report["sensitivity_no_trend"]["prediction"][arm][k])
            entry["inside"] = bool(entry["observed_95"][0] <= entry["predicted"] <= entry["observed_95"][1])
            comparison[arm][k] = entry
    by_half = {arm: {half_name: {k: {**observed(half_name, k, f"{arm}_minus_control"),
                                    "predicted": report["prediction"][arm]["by_half"][half_name][k]}
                                for k in ("ctnl_weeks", "test_coupon_redemptions")}
                     for half_name in ("first", "second")}
               for arm in ("A", "B")}
    levels = {k: {"observed": measured["all"][k]["mean_by_arm"]["control"],
                  "predicted": report["prediction"]["control_level"][k]} for k in OUTCOMES}
    for k, entry in levels.items():
        entry["relative_error"] = entry["predicted"] / entry["observed"] - 1 if entry["observed"] else None
    verdict = {}
    for arm in ("A", "B"):
        verdict[f"C1_{arm}"] = comparison[arm]["ctnl_weeks"]["inside"] and report["prediction"][arm]["ctnl_weeks"] > 0
        verdict[f"C2_{arm}"] = comparison[arm]["tissue_weeks"]["inside"]
        verdict[f"C3_{arm}"] = comparison[arm]["test_coupon_redemptions"]["inside"]
    verdict["C4"] = comparison["A_minus_B"]["ctnl_weeks"]["inside"]
    verdict["C5"] = all(abs(levels[k]["relative_error"]) <= tol for k, tol in LEVEL_TOLERANCE.items())
    verdict["all_pass"] = all(verdict.values())
    report.update(comparison=comparison, by_half=by_half, control_levels=levels, verdict=verdict,
                  runtime_seconds=round(time.time() - started, 1))
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": verdict, "comparison": comparison, "control_levels": levels,
                      "by_half": by_half}, indent=1))


if __name__ == "__main__":
    main()
