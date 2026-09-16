#!/usr/bin/env python3
"""Third tissue-model test: add trial-and-repeat (state dependence) to the version-2 model.

Disclosure
----------
Designed after versions 1 and 2 and the experiment's outcomes were known; not blind. No
parameter is fitted on test-period panel outcomes and nothing else in the version-2 model
or criteria changes.

Motivation
----------
Version 2 fixed purchase levels but predicted almost no lift for arm B, whose observed lift
came early and was large relative to its extra redemptions (about 2.4 purchase-weeks per
redemption). Cottonelle was a new brand, so a coupon that causes a first purchase may raise
later purchasing. Version 2 had no purchase history.

Change
------
    u_hCt += lambda * tried_ht,   tried_ht = 1 if h bought Cottonelle before week t.
Cottonelle launched inside the fit window, so every household's history is observed from
zero and there is no initial-conditions problem. In simulation tried_ht evolves with the
simulated purchases; at the start of the test window it is the household's observed status.

Identification risk
-------------------
Persistent household taste for Cottonelle can masquerade as state dependence. The primary
model keeps the ridge household-brand effect for Cottonelle. Sensitivities: (a) lambda = 0,
which should approximately reproduce version 2; (b) no household Cottonelle taste, which
attributes all persistence to state dependence (an upper bound on lambda).

Criteria (unchanged from version 2; all must hold)
--------------------------------------------------
  C1  Cottonelle purchase-weeks effect positive and inside the observed 95% interval (A, B).
  C2  Tissue purchase-weeks effect inside the observed 95% interval (A, B).
  C3  Arm-specific test-coupon redemption effect inside the observed 95% interval (A, B).
  C4  A minus B Cottonelle purchase-weeks inside the observed 95% interval.
  C5  Control-arm Cottonelle purchase-weeks within +/-20% and control-arm test-coupon
      redemptions within +/-25% of observed.
Expectation stated before running: state dependence targets C1 for arm B. It is not
expected to fix C3 or the redemption part of C5, which version 2 traced to coupon delivery.
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

from run_erim_coupon_experiment import RAW, analyse_window, read_assignment  # noqa: E402
from run_erim_tissue_prediction_test import (  # noqa: E402
    BRANDS, COUPON_VALUE, CTNL, DROP_WEEK, FIT_COUPON, FIT_WEEKS, SCHEDULES,
    SENSITIVITY_SCHEDULES, TEST_WEEKS, ctnl_marginals, erim_weeks, load_panel, load_prices,
    to_tensors,
)
from run_erim_tissue_prediction_test_v2 import (  # noqa: E402
    AVAILABILITY_FLOOR, LEVEL_TOLERANCE, OUTCOMES, TissueModelV2, availability, fit,
    household_store_weights, log_likelihood, store_launch_weeks, weeks_since_launch,
)

torch.set_default_dtype(torch.float64)


class TissueModelV3(TissueModelV2):
    def __init__(self, n_households: int, K: int):
        super().__init__(n_households, K)
        self.lam = torch.nn.Parameter(torch.tensor(0.0))
        self.fixed_lam = None
        self.zero_ctnl_taste = False
        self.tried = None  # [H, T] purchase-history indicator used while fitting

    def lam_value(self):
        return self.lam if self.fixed_lam is None else torch.tensor(self.fixed_lam)

    def utilities(self, dev, log_avail, since):
        u = super().utilities(dev, log_avail, since)
        shift = torch.zeros_like(u)
        shift[..., CTNL] = self.lam_value() * self.tried
        if self.zero_ctnl_taste:
            shift[..., CTNL] = shift[..., CTNL] - self.beta[:, None, CTNL]
        return u + shift


def simulate(model, weeks, opportunity, log_avail, since, dev, level, initial_hold, initial_tried,
             schedule, replicates, seed):
    """Version-2 simulation plus a purchase-history state that evolves with simulated purchases."""
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        tau, theta, lam = float(model.tau()), float(model.theta_value()), float(model.lam_value())
        ex = {k: float(v) for k, v in model.exposure().items()}
        rho = model.rho().numpy()
        static = (model.b + model.alpha[:, None] + model.beta).numpy()
        if model.zero_ctnl_taste:
            static[:, CTNL] -= model.beta[:, CTNL].numpy()
    K, (H, T) = model.K, opportunity.shape
    codes = sorted(set(schedule) | {FIT_COUPON})
    values = np.array([COUPON_VALUE[c] for c in codes])
    holding = np.zeros((replicates, H, len(codes)), dtype=bool)
    holding[:, :, codes.index(FIT_COUPON)] = rng.random((replicates, H)) < initial_hold[None, :]
    tried = np.repeat(initial_tried[None, :], replicates, axis=0)
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
        u_c = u[:, CTNL][None, :] + lam * tried + np.log1p(x)
        p_c, p_any, p_other = ctnl_marginals(np.delete(u, CTNL, axis=1), u_c, rho, K)
        redeem_share = x / (1.0 + x)
        top = np.argmax(holding * values, axis=2)
        has = holding.any(2) & active
        out["ctnl_weeks"][:, t] = (p_c * active).mean(0)
        out["tissue_weeks"][:, t] = (p_any * active).mean(0)
        out["other_weeks"][:, t] = (p_other * active).mean(0)
        arm_specific = np.isin(np.array(codes)[top], [c for c in schedule if c != 38])
        out["test_coupon_redemptions"][:, t] = (p_c * redeem_share * (has & arm_specific)).mean(0)
        bought = active & (buy_draw < p_c)
        use = has & bought & (redeem_draw < redeem_share)
        r_idx, h_idx = np.nonzero(use)
        holding[r_idx, h_idx, top[r_idx, h_idx]] = False
        tried |= bought
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/erim_tissue_prediction_test_v3")
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
    ctnl_history = fit_data["basket"][..., CTNL]
    tried_fit = torch.from_numpy(np.cumsum(ctnl_history, axis=1) - ctnl_history > 0).double()

    def make_model(**options):
        model = TissueModelV3(H, K)
        model.tried = tried_fit
        for name, value in options.items():
            setattr(model, name, value)
        return model

    held_out = (torch.rand(fit_data["opportunity_t"].shape, generator=torch.Generator().manual_seed(args.seed))
                < 0.2) & fit_data["opportunity_t"]
    selection = {}
    for ridge in args.ridge_grid:
        model = make_model()
        fit(model, fit_data, **tensors, ridge=ridge, held_out=held_out, rounds=args.rounds)
        with torch.no_grad():
            full = log_likelihood(model, fit_data, **tensors)
            train = log_likelihood(model, fit_data, **tensors, held_out=held_out)
        selection[ridge] = float((full - train).sum() / held_out.sum())
        print(f"ridge {ridge}: held-out log-lik per opportunity {selection[ridge]:.5f}", flush=True)
    ridge = max(selection, key=selection.get)
    report["ridge_selection"] = {"held_out": selection, "selected": ridge}

    model = make_model()
    report["fit_loss"] = fit(model, fit_data, **tensors, ridge=ridge, rounds=args.rounds)
    with torch.no_grad():
        _, hold = log_likelihood(model, fit_data, **tensors, return_filter=True)
    report["fit"] = {"tau": float(model.tau()), "theta": float(model.theta), "lambda": float(model.lam),
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
    tried_end = ctnl_history.any(1)
    initial_tried = np.where(fit_rows >= 0, tried_end[np.maximum(fit_rows, 0)], False)
    report["initial_tried_share_experimental"] = float(initial_tried.mean())
    fit_check = simulate(model, fit_weeks, fit_data["opportunity"], tensors["log_avail"].numpy(),
                         tensors["since"].numpy(), dev_all[:, :n_fit], level_all[:n_fit],
                         np.zeros(H), np.zeros(H, dtype=bool), (FIT_COUPON,), 100, args.seed)
    report["fit_window_check"] = {
        str(w): {"observed_ctnl": int(ctnl_history[:, t].sum()),
                 "predicted_ctnl": float(fit_check["ctnl_weeks"][:, t].sum()),
                 "observed_redemptions": int(fit_data["redeemed"][FIT_COUPON][:, t].sum()),
                 "predicted_redemptions": float(fit_check["test_coupon_redemptions"][:, t].sum())}
        for t, w in enumerate(fit_weeks)}

    def restrict(source: TissueModelV3) -> TissueModelV3:
        net = TissueModelV3(len(experimental), K)
        with torch.no_grad():
            for name, value in source.state_dict().items():
                if name in ("alpha", "beta"):
                    gathered = value[torch.from_numpy(np.maximum(fit_rows, 0))].clone()
                    gathered[torch.from_numpy(fit_rows < 0)] = 0.0
                    getattr(net, name).copy_(gathered)
                else:
                    getattr(net, name).copy_(value)
        net.fixed_theta = source.fixed_theta
        net.fixed_lam = source.fixed_lam
        net.zero_ctnl_taste = source.zero_ctnl_taste
        return net

    def predict(source, schedules, seed):
        net = restrict(source)
        sims = {arm: simulate(net, test_weeks, opportunity, log_avail_test, since_test, dev_test,
                              level_test, initial_hold, initial_tried, schedule, args.replicates, seed)
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
    for label, options in (("no_habit", {"fixed_lam": 0.0}), ("no_household_ctnl_taste", {"zero_ctnl_taste": True})):
        variant = make_model(**options)
        variant.load_state_dict(model.state_dict())
        fit(variant, fit_data, **tensors, ridge=ridge, rounds=args.rounds)
        report[f"sensitivity_{label}"] = {
            "fit": {**{k: float(v) for k, v in variant.exposure().items()}, "tau": float(variant.tau()),
                    "lambda": float(variant.lam_value()), "theta": float(variant.theta)},
            "prediction": predict(variant, SCHEDULES, args.seed)}
        print(label, json.dumps(report[f"sensitivity_{label}"]["fit"]), flush=True)

    boot = []
    for b in range(args.bootstrap):
        bm = make_model()
        bm.load_state_dict(model.state_dict())
        w = torch.distributions.Exponential(1.0).sample((H,))
        fit(bm, fit_data, **tensors, ridge=ridge, weights=w, rounds=max(1, args.rounds - 1))
        pred = predict(bm, SCHEDULES, args.seed + 1000 + b)
        pred["parameters"] = {"tau": float(bm.tau()), "theta": float(bm.theta), "lambda": float(bm.lam),
                              **{k: float(v) for k, v in bm.exposure().items()}}
        boot.append(pred)
        print(f"bootstrap {b + 1}/{args.bootstrap}: A ctnl {pred['A']['ctnl_weeks']:+.3f} "
              f"B ctnl {pred['B']['ctnl_weeks']:+.3f} lambda {pred['parameters']['lambda']:.3f} "
              f"kappa {pred['parameters']['kappa']:.3f}", flush=True)

    def band(values):
        return [float(np.quantile(values, 0.05)), float(np.quantile(values, 0.95))] if values else None

    report["prediction_interval_90"] = {
        **{arm: {k: band([p[arm][k] for p in boot]) for k in OUTCOMES} for arm in ("A", "B", "A_minus_B")},
        "control_level": {k: band([p["control_level"][k] for p in boot]) for k in OUTCOMES},
        "parameters": {k: band([p["parameters"][k] for p in boot]) for k in ("tau", "theta", "lambda", "q", "r", "kappa")}}
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
                         predicted_no_habit=report["sensitivity_no_habit"]["prediction"][arm][k],
                         predicted_no_household_ctnl_taste=report["sensitivity_no_household_ctnl_taste"]["prediction"][arm][k])
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
