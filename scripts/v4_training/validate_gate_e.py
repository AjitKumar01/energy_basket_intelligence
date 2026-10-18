#!/usr/bin/env python3
"""Validate Step-1 Gate E importance-sampling log Z estimator (is_log_z vs is_log_z_orig)
against exact enumeration across interaction strengths on 200 small-world contexts,
for baseline Version-4, M1 (free category-count potential f_c(n_c)), and M2 (M1 + presence-based s(S))."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hs import is_log_z, is_log_z_orig, log_zc  # noqa: E402
from world import Support, contexts, exact_log_z, make_truth  # noqa: E402

torch.set_default_dtype(torch.float64)


def run_mode(args, sup, hp, cw, use_f_extra=False, presence_phi=False):
    J, H, W, K, r, nmax = 18, 100, 8, 3, 3, 5
    results = []
    for strength in args.strengths:
        truth, week_price = make_truth(args.seed, strength, J=J, H=H, W=W, K=K, r=r, nmax=nmax,
                                       use_f_extra=use_f_extra, presence_phi=presence_phi)
        pp = week_price[torch.as_tensor(cw)]

        # Also verify exact beta=0 DP normalizer log_zc against brute-force enumeration at Phi=0
        if strength == args.strengths[0]:
            t_zero, _ = make_truth(args.seed, strength, J=J, H=H, W=W, K=K, r=r, nmax=nmax,
                                   use_f_extra=use_f_extra, presence_phi=presence_phi)
            with torch.no_grad():
                t_zero.p["phi"].zero_()
                if presence_phi:
                    t_zero.p["psi"].zero_()
                    t_zero.p["delta_phi"].zero_()
                ex0 = exact_log_z(t_zero, sup, hp, pp)
                dp0 = log_zc(t_zero.b(hp, pp), t_zero.part, t_zero.p["rho_c"], t_zero.rho0(),
                             f_extra=t_zero.f_extra(), cat_shift=None)
                dp_err = float((ex0 - dp0).abs().max())
                assert dp_err < 1e-12, f"DP log_zc mismatch vs enumeration at Phi=0: {dp_err}"

        exact = exact_log_z(truth, sup, hp, pp)

        est_o, ess_o = is_log_z_orig(
            truth, hp, pp, torch.Generator().manual_seed(args.seed + 12),
            burn=args.is_burn, components=args.is_components, thin=args.is_thin, draws=args.is_draws)
        err_o = (est_o - exact).abs()

        est_f, ess_f = is_log_z(
            truth, hp, pp, torch.Generator().manual_seed(args.seed + 12),
            burn=args.is_burn, components=args.is_components, thin=args.is_thin, draws=args.is_draws)
        err_f = (est_f - exact).abs()

        row = {
            "strength": float(strength),
            "orig": {
                "mean_abs_err": round(float(err_o.mean()), 5),
                "max_abs_err": round(float(err_o.max()), 5),
                "mean_ess": round(float(ess_o.mean()), 4),
                "min_ess": round(float(ess_o.min()), 4),
                "pass_gate_E": bool(err_o.mean() <= 0.01 and err_o.max() <= 0.05),
            },
            "fixed": {
                "mean_abs_err": round(float(err_f.mean()), 5),
                "max_abs_err": round(float(err_f.max()), 5),
                "mean_ess": round(float(ess_f.mean()), 4),
                "min_ess": round(float(ess_f.min()), 4),
                "pass_gate_E": bool(err_f.mean() <= 0.01 and err_f.max() <= 0.05),
            },
        }
        print(json.dumps({"mode": ("m2" if presence_phi else ("m1" if use_f_extra else "baseline")), **row}), flush=True)
        results.append(row)
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("baseline", "m1", "m2", "all"), default="baseline")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--strengths", type=float, nargs="+", default=[1.0, 3.0, 7.0, 25.0, 50.0, 75.0])
    p.add_argument("--panel", type=int, default=200)
    p.add_argument("--is-burn", type=int, default=200)
    p.add_argument("--is-components", type=int, default=256)
    p.add_argument("--is-thin", type=int, default=4)
    p.add_argument("--is-draws", type=int, default=2048)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("runs/stress/gate_e_validation.json"))
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    J, H, W, nmax = 18, 100, 8, 5
    sup = Support(J, nmax)
    ch, cw = contexts(H, W)
    panel = np.random.default_rng(args.seed + 4).choice(len(ch), size=args.panel, replace=False)
    hp = torch.as_tensor(ch[panel])
    cw_p = cw[panel]

    if args.mode == "baseline":
        results = run_mode(args, sup, hp, cw_p, use_f_extra=False, presence_phi=False)
        payload = {
            "panel_contexts": args.panel,
            "seed": args.seed,
            "is_burn": args.is_burn,
            "is_components": args.is_components,
            "is_draws": args.is_draws,
            "gate_E_thresholds": {"mean_abs_err_le": 0.01, "max_abs_err_le": 0.05},
            "results": results,
        }
    elif args.mode == "m1":
        results = run_mode(args, sup, hp, cw_p, use_f_extra=True, presence_phi=False)
        payload = {"mode": "m1", "results": results}
    elif args.mode == "m2":
        results = run_mode(args, sup, hp, cw_p, use_f_extra=True, presence_phi=True)
        payload = {"mode": "m2", "results": results}
    else:
        payload = {
            "panel_contexts": args.panel,
            "seed": args.seed,
            "is_burn": args.is_burn,
            "is_components": args.is_components,
            "is_draws": args.is_draws,
            "gate_E_thresholds": {"mean_abs_err_le": 0.01, "max_abs_err_le": 0.05},
            "baseline": run_mode(args, sup, hp, cw_p, use_f_extra=False, presence_phi=False),
            "m1": run_mode(args, sup, hp, cw_p, use_f_extra=True, presence_phi=False),
            "m2": run_mode(args, sup, hp, cw_p, use_f_extra=True, presence_phi=True),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
