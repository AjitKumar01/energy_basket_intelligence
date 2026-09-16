#!/usr/bin/env python3
"""Cross-arm transport test: borrow coupon delivery from one arm, predict the other arm's lift.

Disclosure
----------
Designed after tissue-model versions 1-3 and the experiment's outcomes were known; not blind.

Motivation
----------
Versions 2 and 3 predicted normal purchasing well but under-predicted campaign coupon
redemptions about four-fold. They learned coupon reach and life from the only pre-campaign
coupon, which evidently was delivered differently from the campaign's cell-targeted coupons.
This test separates delivery from purchase response.

Design
------
* Purchase model: the version-2 tissue model (availability, awareness trend, direct
  redemption with coupon response kappa, shelf-price tau, household tastes), fitted only on
  pre-campaign weeks exactly as in version 2. Version 3's history term is omitted: it did not
  improve held-out fit or predictions.
* Campaign delivery: the arm-specific coupon codes {1, 19, 20, 35, 36, 37} get their own
  notice probability q_c and weekly retention r_c. Market-wide codes 17 and 38 keep the
  pre-campaign q and r.
* Calibration: q_c and r_c are chosen by minimising the Poisson deviance between observed and
  simulated redemptions, in 4-week blocks by coupon code, among the households of ONE arm
  (its own schedule codes only). No purchase outcome and no other arm is used.
* Transport: calibrated on arm A, predict arm B minus control; calibrated on arm B, predict
  arm A minus control. The control arm is never used for calibration.

Criteria (all must hold)
------------------------
  T1  Calibrated on A: predicted B-minus-control Cottonelle purchase-weeks positive and inside
      the observed 95% interval.
  T2  Calibrated on B: predicted A-minus-control Cottonelle purchase-weeks positive and inside
      the observed 95% interval.
  T3  In both directions, the held-out arm's redemption effect inside its observed interval.
  T4  In both directions, the held-out arm's tissue purchase-weeks effect inside its interval.
  T5  In both directions, control-arm Cottonelle purchase-weeks within +/-20% and control-arm
      campaign redemptions within +/-25% of observed.
Expectation stated before running: if delivery is the missing piece, T1-T3 pass. If they fail
with delivery borrowed, the model's purchase response to coupons is also wrong.
"""
from __future__ import annotations

import argparse
import itertools
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
    BRANDS, COUPON_VALUE, CTNL, DROP_WEEK, FIT_COUPON, FIT_WEEKS, SCHEDULES, TEST_WEEKS,
    ctnl_marginals, erim_weeks, load_panel, load_prices, to_tensors,
)
from run_erim_tissue_prediction_test_v2 import (  # noqa: E402
    AVAILABILITY_FLOOR, LEVEL_TOLERANCE, OUTCOMES, TissueModelV2, availability, fit,
    household_store_weights, log_likelihood, store_launch_weeks, weeks_since_launch,
)

CAMPAIGN_CODES = (1, 19, 20, 35, 36, 37)
ALL_CODES = tuple(sorted(COUPON_VALUE))
Q_GRID = tuple(np.round(np.arange(0.1, 1.01, 0.1), 2))
R_GRID = (0.6, 0.7, 0.8, 0.85, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99)
torch.set_default_dtype(torch.float64)


def simulate(model, weeks, opportunity, log_avail, since, dev, level, initial_hold, schedule,
             replicates, seed, delivery):
    """Version-2 simulation with per-code notice/retention; also returns redemptions by code.

    delivery = (q_c, r_c) for CAMPAIGN_CODES; other codes use the fitted pre-campaign values.
    """
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        tau, theta = float(model.tau()), float(model.theta_value())
        ex = {k: float(v) for k, v in model.exposure().items()}
        rho = model.rho().numpy()
        static = (model.b + model.alpha[:, None] + model.beta).numpy()
    K, (H, T) = model.K, opportunity.shape
    codes = sorted(set(schedule) | {FIT_COUPON})
    values = np.array([COUPON_VALUE[c] for c in codes])
    q = np.array([delivery[0] if c in CAMPAIGN_CODES else ex["q"] for c in codes])
    r = np.array([delivery[1] if c in CAMPAIGN_CODES else ex["r"] for c in codes])
    holding = np.zeros((replicates, H, len(codes)), dtype=bool)
    holding[:, :, codes.index(FIT_COUPON)] = rng.random((replicates, H)) < initial_hold[None, :]
    out = {k: np.zeros((H, T)) for k in OUTCOMES}
    by_code = np.zeros((H, T, len(ALL_CODES)))
    for t, week in enumerate(weeks):
        holding &= rng.random(holding.shape) < r[None, None, :]
        for c in schedule:
            if DROP_WEEK[c] == week:
                i = codes.index(c)
                holding[:, :, i] |= rng.random((replicates, H)) < q[i]
        buy_draw, redeem_draw = rng.random((replicates, H)), rng.random((replicates, H))
        active = opportunity[:, t][None, :]
        if not active.any():
            continue
        u = static - tau * dev[:, t][None, :]
        u[:, CTNL] += log_avail[:, t] + theta * np.log1p(since[t])
        best = np.where(holding.any(2), (holding * values).max(2), 0.0)
        x = ex["kappa"] * best / level[t]
        p_c, p_any, p_other = ctnl_marginals(np.delete(u, CTNL, axis=1), u[:, CTNL][None, :] + np.log1p(x), rho, K)
        share = x / (1.0 + x)
        top = np.argmax(holding * values, axis=2)
        has = holding.any(2) & active
        out["ctnl_weeks"][:, t] = (p_c * active).mean(0)
        out["tissue_weeks"][:, t] = (p_any * active).mean(0)
        out["other_weeks"][:, t] = (p_other * active).mean(0)
        expected = p_c * share * has
        for i, c in enumerate(codes):
            by_code[:, t, ALL_CODES.index(c)] = (expected * (top == i)).mean(0)
        campaign = np.isin(np.array(codes)[top], [c for c in schedule if c in CAMPAIGN_CODES])
        out["test_coupon_redemptions"][:, t] = (expected * campaign).mean(0)
        use = has & (buy_draw < p_c) & (redeem_draw < share)
        r_idx, h_idx = np.nonzero(use)
        holding[r_idx, h_idx, top[r_idx, h_idx]] = False
    out["redemptions_by_code"] = by_code
    return out


def poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    expected = np.maximum(expected, 1e-9)
    term = np.where(observed > 0, observed * np.log(np.maximum(observed, 1e-12) / expected), 0.0)
    return float(2.0 * np.sum(term - (observed - expected)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/erim_tissue_transport_test")
    parser.add_argument("--ridge", type=float, default=1.0,
                        help="version-2 selected value (selection is not repeated)")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--replicates", type=int, default=200)
    parser.add_argument("--calibration-replicates", type=int, default=30)
    parser.add_argument("--bootstrap", type=int, default=10)
    parser.add_argument("--permutations", type=int, default=4999)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--predict-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    started = time.time()
    report = {"protocol": __doc__, "ridge": args.ridge}

    # ---- pre-campaign purchase model (version 2) ----
    fit_weeks, test_weeks = erim_weeks(*FIT_WEEKS), erim_weeks(*TEST_WEEKS)
    fit_data = to_tensors(load_panel(fit_weeks))
    dev_all, level_all = load_prices(erim_weeks(FIT_WEEKS[0], TEST_WEEKS[1]))
    n_fit = len(fit_weeks)
    launch = store_launch_weeks()
    weights, pooled = household_store_weights()
    avail_fit = availability(fit_data["households"], fit_weeks, weights, pooled, launch)
    tensors = dict(dev=torch.from_numpy(dev_all[:, :n_fit]), level=torch.from_numpy(level_all[:n_fit]),
                   log_avail=torch.from_numpy(np.log(np.maximum(avail_fit, AVAILABILITY_FLOOR))),
                   since=torch.from_numpy(weeks_since_launch(fit_weeks)))
    K, H = int(fit_data["basket"].sum(2).max()), len(fit_data["households"])
    model = TissueModelV2(H, K)
    fit(model, fit_data, **tensors, ridge=args.ridge, rounds=args.rounds)
    report["fit"] = {"tau": float(model.tau()), "theta": float(model.theta),
                     **{k: float(v) for k, v in model.exposure().items()}}
    print("fit", json.dumps(report["fit"]), flush=True)

    # ---- experimental households, test-window inputs ----
    test_data = load_panel(test_weeks)
    assignment = read_assignment().set_index("household")
    experimental = assignment.index.to_numpy()
    fit_index = {h: i for i, h in enumerate(fit_data["households"])}
    test_index = {h: i for i, h in enumerate(test_data["households"])}
    fit_rows = np.array([fit_index.get(h, -1) for h in experimental])
    test_rows = np.array([test_index.get(h, -1) for h in experimental])
    opportunity = np.zeros((len(experimental), len(test_weeks)), dtype=bool)
    opportunity[test_rows >= 0] = test_data["opportunity"][test_rows[test_rows >= 0]]
    log_avail = np.log(np.maximum(availability(experimental, test_weeks, weights, pooled, launch), AVAILABILITY_FLOOR))
    since = weeks_since_launch(test_weeks)
    dev_test, level_test = dev_all[:, n_fit:], level_all[n_fit:]
    arm = assignment.arm.to_numpy()
    block = np.arange(len(test_weeks)) // 4

    # Observed redemptions by code, used ONLY for the calibration arm.
    redeemed = np.zeros((len(experimental), len(test_weeks), len(ALL_CODES)))
    for i, code in enumerate(ALL_CODES):
        redeemed[test_rows >= 0, :, i] = test_data["redeemed"][code][test_rows[test_rows >= 0]]

    hold_cache: dict[int, np.ndarray] = {}

    def restrict(source: TissueModelV2, members: np.ndarray) -> tuple[TissueModelV2, np.ndarray]:
        net = TissueModelV2(len(members), K)
        rows = fit_rows[members]
        with torch.no_grad():
            for name, value in source.state_dict().items():
                if name in ("alpha", "beta"):
                    gathered = value[torch.from_numpy(np.maximum(rows, 0))].clone()
                    gathered[torch.from_numpy(rows < 0)] = 0.0
                    getattr(net, name).copy_(gathered)
                else:
                    getattr(net, name).copy_(value)
            if id(source) not in hold_cache:
                hold_cache[id(source)] = log_likelihood(source, fit_data, **tensors, return_filter=True)[1].numpy()
        hold = hold_cache[id(source)]
        initial = np.where(rows >= 0, hold[np.maximum(rows, 0)], 0.0)
        return net, initial

    def run(source, members, schedule, replicates, seed, delivery):
        net, initial = restrict(source, members)
        return simulate(net, test_weeks, opportunity[members], log_avail[members], since, dev_test,
                        level_test, initial, schedule, replicates, seed, delivery)

    def calibrate(source, calibration_arm, seed, household_weight=None):
        members = np.flatnonzero(arm == calibration_arm)
        schedule = SCHEDULES[calibration_arm]
        code_columns = [ALL_CODES.index(c) for c in schedule if c in CAMPAIGN_CODES]
        w = np.ones(len(members)) if household_weight is None else household_weight[members]
        observed = np.stack([np.bincount(block, (w[:, None] * redeemed[members][:, :, j]).sum(0))
                             for j in code_columns])
        net, initial = restrict(source, members)

        def loss(q_c, r_c):
            sim = simulate(net, test_weeks, opportunity[members], log_avail[members], since, dev_test,
                           level_test, initial, schedule, args.calibration_replicates, seed, (q_c, r_c))
            expected = np.stack([np.bincount(block, (w[:, None] * sim["redemptions_by_code"][:, :, j]).sum(0))
                                 for j in code_columns])
            return poisson_deviance(observed, expected)

        grid = {(q, r): loss(q, r) for q, r in itertools.product(Q_GRID, R_GRID)}
        q0, r0 = min(grid, key=grid.get)
        fine_q = [v for v in np.round(q0 + np.array([-0.08, -0.04, 0.0, 0.04, 0.08]), 3) if 0.02 <= v <= 1.0]
        fine_r = [v for v in np.round(r0 + np.array([-0.02, -0.01, 0.0, 0.01, 0.02]), 3) if 0.5 <= v <= 0.995]
        grid.update({(q, r): loss(q, r) for q, r in itertools.product(fine_q, fine_r) if (q, r) not in grid})
        best = min(grid, key=grid.get)
        return {"q_c": float(best[0]), "r_c": float(best[1]), "deviance": grid[best],
                "observed_by_code_total": dict(zip([ALL_CODES[j] for j in code_columns],
                                                   observed.sum(1).round(1).tolist()))}

    def predict(source, delivery, seed, replicates):
        everyone = np.arange(len(experimental))
        sims = {a: run(source, everyone, SCHEDULES[a], replicates, seed, delivery) for a in SCHEDULES}
        summary = {"control_level": {k: float(sims["control"][k].sum(1).mean()) for k in OUTCOMES}}
        for a in ("A", "B"):
            summary[a] = {k: float((sims[a][k] - sims["control"][k]).sum(1).mean()) for k in OUTCOMES}
        summary["own_arm_redemptions_by_code"] = {
            a: {str(c): float(sims[a]["redemptions_by_code"][:, :, ALL_CODES.index(c)].sum(1).mean())
                for c in SCHEDULES[a]} for a in SCHEDULES}
        return summary

    directions = {"calibrate_A_predict_B": ("A", "B"), "calibrate_B_predict_A": ("B", "A")}
    report["calibration"], report["prediction"] = {}, {}
    for name, (source_arm, target_arm) in directions.items():
        cal = calibrate(model, source_arm, args.seed)
        report["calibration"][name] = cal
        report["prediction"][name] = predict(model, (cal["q_c"], cal["r_c"]), args.seed, args.replicates)
        print(name, json.dumps(cal), "->", target_arm,
              json.dumps(report["prediction"][name][target_arm]), flush=True)

    boot = {name: [] for name in directions}
    for b in range(args.bootstrap):
        bm = TissueModelV2(H, K)
        bm.load_state_dict(model.state_dict())
        fit(bm, fit_data, **tensors, ridge=args.ridge,
            weights=torch.distributions.Exponential(1.0).sample((H,)), rounds=max(1, args.rounds - 1))
        household_weight = np.random.default_rng(args.seed + b).exponential(size=len(experimental))
        for name, (source_arm, target_arm) in directions.items():
            cal = calibrate(bm, source_arm, args.seed + 1000 + b, household_weight)
            pred = predict(bm, (cal["q_c"], cal["r_c"]), args.seed + 1000 + b, args.replicates // 2)
            boot[name].append({"calibration": cal, "prediction": pred})
            print(f"bootstrap {b + 1}/{args.bootstrap} {name}: q_c {cal['q_c']:.2f} r_c {cal['r_c']:.3f} "
                  f"{target_arm} ctnl {pred[target_arm]['ctnl_weeks']:+.3f}", flush=True)
    band = lambda v: [float(np.quantile(v, 0.05)), float(np.quantile(v, 0.95))]  # noqa: E731
    report["interval_90"] = {
        name: {"q_c": band([e["calibration"]["q_c"] for e in entries]),
               "r_c": band([e["calibration"]["r_c"] for e in entries]),
               **{f"{a}_{k}": band([e["prediction"][a][k] for e in entries]) for a in ("A", "B") for k in OUTCOMES},
               **{f"control_{k}": band([e["prediction"]["control_level"][k] for e in entries]) for k in OUTCOMES}}
        for name, entries in boot.items() if entries}
    (args.output / "prediction.json").write_text(json.dumps(report, indent=2, default=str))
    print("predictions frozen to", args.output / "prediction.json", flush=True)
    if args.predict_only:
        return

    # ---- observed contrasts ----
    rows = test_rows
    basket = np.zeros((len(experimental), len(test_weeks), len(BRANDS)), dtype=bool)
    basket[rows >= 0] = test_data["basket"][rows[rows >= 0]]
    frame = assignment.copy()
    frame["ctnl_weeks"] = basket[..., CTNL].sum(1)
    frame["tissue_weeks"] = basket.any(2).sum(1)
    frame["other_weeks"] = np.delete(basket, CTNL, axis=2).any(2).sum(1)
    frame["test_coupon_redemptions"] = redeemed[:, :, [ALL_CODES.index(c) for c in CAMPAIGN_CODES]].sum((1, 2))
    measured = analyse_window(frame, list(OUTCOMES), args.permutations, args.seed)

    comparison, verdict = {}, {}
    for name, (source_arm, target_arm) in directions.items():
        pred = report["prediction"][name]
        entry = {}
        for k in OUTCOMES:
            m = measured[k][f"{target_arm}_minus_control"]
            lo, hi = m["randomization_95_null_band"]
            interval = [m["difference"] - hi, m["difference"] - lo]
            entry[k] = {"observed": m["difference"], "observed_95": interval, "p": m["randomization_p_two_sided"],
                        "predicted": pred[target_arm][k],
                        "predicted_90": report.get("interval_90", {}).get(name, {}).get(f"{target_arm}_{k}"),
                        "inside": bool(interval[0] <= pred[target_arm][k] <= interval[1])}
        levels = {k: {"observed": measured[k]["mean_by_arm"]["control"], "predicted": pred["control_level"][k]}
                  for k in OUTCOMES}
        for v in levels.values():
            v["relative_error"] = v["predicted"] / v["observed"] - 1 if v["observed"] else None
        comparison[name] = {"target_arm": target_arm, "effects": entry, "control_levels": levels,
                            "calibration_arm_fit": {
                                "observed_redemptions_per_household": measured["test_coupon_redemptions"]["mean_by_arm"][source_arm],
                                "calibrated_redemptions_by_code": pred["own_arm_redemptions_by_code"][source_arm]}}
        tag = "T1" if target_arm == "B" else "T2"
        verdict[tag] = entry["ctnl_weeks"]["inside"] and pred[target_arm]["ctnl_weeks"] > 0
        verdict[f"T3_{name}"] = entry["test_coupon_redemptions"]["inside"]
        verdict[f"T4_{name}"] = entry["tissue_weeks"]["inside"]
        verdict[f"T5_{name}"] = all(abs(levels[k]["relative_error"]) <= tol for k, tol in LEVEL_TOLERANCE.items())
    verdict["all_pass"] = all(verdict.values())
    report.update(comparison=comparison, verdict=verdict, runtime_seconds=round(time.time() - started, 1))
    (args.output / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps({"verdict": verdict, "comparison": comparison}, indent=1))


if __name__ == "__main__":
    main()
