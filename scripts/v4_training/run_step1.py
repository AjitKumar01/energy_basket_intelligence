#!/usr/bin/env python3
"""Step 1: does exact-gradient HS training recover Version-4 with strong interactions?

On exactly enumerable Version-4 worlds (strength = top eigenvalue of Phi Phi'):
  truth       the generating model
  exact MLE   reference only: enumeration, started at the truth (best this class can do)
  pseudo-lik  the starting point of the proposed training (exact conditionals, no Z)
  HS-SML      proposed: pseudo-likelihood start, then stochastic maximum likelihood with
              persistent HS block-Gibbs chains and Rao-Blackwellized exact-DP gradients
  plug-in     comparison: the same loop with z = s(observed basket) (fit_fast_v4.py's shortcut)

Pre-registered gates (configs/v4_training/step1_preregistration.json), on confirmation seeds:
  Gate E  importance-sampling log Z of the HS-SML fit vs exact, over a panel of test contexts:
          mean |error| <= 0.01, max <= 0.05 nats
  Gate T  exact test log-likelihood of HS-SML within 0.02 nats per basket of the truth, and
          correlation of the identified pair interaction (phi_i.phi_j - rho_c[same]) with the
          truth >= 0.9
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hs import V4, fit_pseudo_likelihood, fit_sml, is_log_z  # noqa: E402
from world import (Support, contexts, draw_trips, exact_log_z, exact_trip_ll, fit_exact_mle,  # noqa: E402
                   make_truth, net_pair_interaction, tensors)

torch.set_default_dtype(torch.float64)


def summary(d):
    return {"mean": float(d.mean()), "se": float(d.std(ddof=1) / np.sqrt(len(d)))}


def run(args, seed, strength, log):
    started = time.time()
    J, H, W, K, r, nmax = 18, args.households, args.weeks, 3, 3, 5
    truth, week_price = make_truth(seed, strength, J=J, H=H, W=W, K=K, r=r, nmax=nmax)
    sup = Support(J, nmax)
    rng = np.random.default_rng(seed + 1)
    train = draw_trips(truth, sup, week_price, H, W, args.train, rng)
    test = draw_trips(truth, sup, week_price, H, W, args.test, rng)
    X, h, price = tensors(train, sup, week_price)
    prec = {
        "theta": 1 / 0.8 ** 2,
        "alpha": 1 / 0.8 ** 2,
        "delta": 1 / 0.5 ** 2,
        "phi": args.phi_prec,
        "rho_c": args.rho_c_prec,
    }
    out = {"seed": seed, "strength": strength, "train": args.train, "test": args.test,
           "data_mean_size": float(X.sum(1).mean())}
    ll = {"truth": exact_trip_ll(truth, sup, test, week_price, H, W)}
    W_true = net_pair_interaction(truth)

    def evaluate(name, model):
        ll[name] = exact_trip_ll(model, sup, test, week_price, H, W)
        out.setdefault("arms", {})[name] = {
            "test_ll": float(ll[name].mean()),
            "truth_minus_arm": summary(ll["truth"] - ll[name]),
            "pair_interaction_corr": float(np.corrcoef(W_true, net_pair_interaction(model))[0, 1]),
            "phi_top_eig": float(torch.linalg.svdvals(model.p["phi"].detach())[0] ** 2)}
        log({"seed": seed, "strength": strength, "arm": name, **out["arms"][name]})

    if not args.skip_exact:
        ref = V4(J, H, K, r, truth.part, seed=seed)
        ref.load(truth.state())
        fit_exact_mle(ref, sup, train, week_price, H, W, prec)
        evaluate("exact MLE", ref)

    pl = V4(J, H, K, r, truth.part, seed=seed + 7)
    with torch.no_grad():
        rate = X.mean(0).clamp(1e-3, 1 - 1e-3)
        pl.p["lam"].copy_(torch.log(rate) - torch.log1p(-rate))
    fit_pseudo_likelihood(pl, X, h, price, prec)
    evaluate("pseudo-lik", pl)
    start = pl.state()

    test_X, test_h, test_p = tensors(test, sup, week_price)
    mon_rows = np.random.default_rng(seed + 3).choice(len(test["ctx"]), size=2000, replace=False)
    mon = {k: v[mon_rows] for k, v in test.items()}

    def monitor(model):          # diagnostic only (exact, never used for any decision in the fit)
        return {"test_ll_2000": float(exact_trip_ll(model, sup, mon, week_price, H, W).mean())}

    for name, plug in (("HS-SML", False), ("plug-in", True)):
        if name == "plug-in" and args.skip_plugin:
            continue
        m = V4(J, H, K, r, truth.part)
        m.load(start)
        t0 = time.time()
        fit_sml(m, X, h, price, prec, args.iters, args.lr, args.lr_final,
                torch.Generator().manual_seed(seed + 11), chains=args.chains, sweeps=args.sweeps,
                plug_in=plug, every=args.every, monitor=monitor,
                log=lambda row, n=name: log({"seed": seed, "strength": strength, "arm": n, **row}))
        out.setdefault("fit_seconds", {})[name] = time.time() - t0
        evaluate(name, m)
        if name == "HS-SML":
            sml = m

    # Gate E: IS log Z of the HS-SML fit on a panel of test contexts
    ch, cw = contexts(H, W)
    panel = np.random.default_rng(seed + 4).choice(len(ch), size=args.panel, replace=False)
    hp, pp = torch.as_tensor(ch[panel]), week_price[torch.as_tensor(cw[panel])]
    exact = exact_log_z(sml, sup, hp, pp)
    t0 = time.time()
    est, ess = is_log_z(sml, hp, pp, torch.Generator().manual_seed(seed + 12), burn=args.is_burn,
                        components=args.is_components, thin=args.is_thin, draws=args.is_draws)
    err = (est - exact).abs()
    out["gate_E"] = {"mean_abs_error": float(err.mean()), "max_abs_error": float(err.max()),
                     "mean_signed_error": float((est - exact).mean()),
                     "ess_fraction_mean": float(ess.mean()), "ess_fraction_min": float(ess.min()),
                     "seconds_per_context": (time.time() - t0) / len(panel),
                     "pass": bool(err.mean() <= 0.01 and err.max() <= 0.05)}
    a = out["arms"]["HS-SML"]
    out["gate_T"] = {"truth_minus_fit": a["truth_minus_arm"]["mean"],
                     "pair_interaction_corr": a["pair_interaction_corr"],
                     "pass": bool(a["truth_minus_arm"]["mean"] <= 0.02 and a["pair_interaction_corr"] >= 0.9)}
    out["seconds"] = time.time() - started
    log({"seed": seed, "strength": strength, "gate_E": out["gate_E"], "gate_T": out["gate_T"]})
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--seeds", type=int, nargs="+", required=True)
    p.add_argument("--strengths", type=float, nargs="+", default=[1.0, 3.0, 7.0])
    p.add_argument("--households", type=int, default=100)
    p.add_argument("--weeks", type=int, default=8)
    p.add_argument("--train", type=int, default=40000)
    p.add_argument("--test", type=int, default=40000)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--lr-final", type=float, default=0.001)
    p.add_argument("--phi-prec", type=float, default=2.0)
    p.add_argument("--rho-c-prec", type=float, default=0.2)
    p.add_argument("--chains", type=int, default=1)
    p.add_argument("--sweeps", type=int, default=1)
    p.add_argument("--every", type=int, default=250)
    p.add_argument("--is-thin", type=int, default=4)
    p.add_argument("--panel", type=int, default=200)
    p.add_argument("--is-burn", type=int, default=200)
    p.add_argument("--is-components", type=int, default=256)
    p.add_argument("--is-draws", type=int, default=2048)
    p.add_argument("--skip-exact", action="store_true")
    p.add_argument("--skip-plugin", action="store_true")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logf = open(args.output.with_suffix(".log"), "a")

    def log(row):
        line = json.dumps(row)
        print(line, flush=True)
        logf.write(line + "\n"); logf.flush()

    results = []
    for seed in args.seeds:
        for strength in args.strengths:
            results.append(run(args, seed, strength, log))
            args.output.write_text(json.dumps({"config": {k: str(v) for k, v in vars(args).items()},
                                               "results": results}, indent=1) + "\n")
    passed = all(r["gate_E"]["pass"] and r["gate_T"]["pass"] for r in results)
    log({"all_worlds_pass": passed})


if __name__ == "__main__":
    main()
