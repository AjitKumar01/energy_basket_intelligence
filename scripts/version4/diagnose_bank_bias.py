"""Does the refinement bank estimate the exact change in log Z between two checkpoints?

For training contexts, compare against the exact Smolyak/ESP normalizer:

  exact   dlogZ_c = log Z_child(c) - log Z_parent(c);  dE_obs_c from model.energy
  bank    log mean_m exp(E_child(S_m) - E_parent(S_m)) over baskets S_m drawn from the PARENT,
          for several bank settings (burn-in, warm start, draws);  dE_obs_c from basket_energy.

A bank that is at the parent's law gives mean(bank - exact) ~ 0 up to Monte Carlo error
(slightly negative from Jensen).  A systematic gap means the bank is not the parent's law.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from checkpoint_io import load_checkpoint
from data import build
from features import Features
from fit import Batcher
from fit_joint_refinement import choose_contexts, draw_all, observed_items_of
from joint_refinement import basket_energy, baskets_from_items, slot_table
from pipeline_support import install_quadrature, smolyak_rule

torch.set_default_dtype(torch.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--child", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contexts", type=int, default=256)
    p.add_argument("--level-offset", type=int, default=3)
    p.add_argument("--threads", type=int, default=10)
    p.add_argument("--seed", type=int, default=41001)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)

    data = build()
    parent, blob, meta = load_checkpoint(args.parent, data)
    child, _, _ = load_checkpoint(args.child, data)
    for m in (parent, child):
        m.requires_grad_(False)
    rank, nmax, J = int(blob["active_rank"]), int(meta["nmax"]), int(data["n_item"])
    batcher = Batcher(data, Features(J, int(data["n_store"]), include_recency=False), nmax,
                      include_recency=False)
    trips = choose_contexts(data, nmax, 10, 0, rng)
    trips = np.sort(rng.choice(trips, size=args.contexts, replace=False))
    ix, ctx, line, house, li, lt, lc, _lq = batcher.make(trips)
    full = (house, ctx)
    table = slot_table(ix, J)
    observed_items = observed_items_of(data, trips)
    observed = baskets_from_items(table, list(range(len(trips))), observed_items)
    rule = smolyak_rule(parent, rank, rank + args.level_offset)

    exact = {}
    for name, m in (("parent", parent), ("child", child)):
        m.house, m.ctx = full
        install_quadrature(m, rule)
        with torch.no_grad():
            exact[name] = {"E_obs": m.energy(li, lt, lc, ix.B, line), "log_Z": m.log_Z(ix, drop_empty=True),
                           "E_obs_bank_formula": basket_energy(m, ix, observed)}
        m.quad = None
    d_logz = exact["child"]["log_Z"] - exact["parent"]["log_Z"]
    d_obs = exact["child"]["E_obs"] - exact["parent"]["E_obs"]
    d_obs_bank = exact["child"]["E_obs_bank_formula"] - exact["parent"]["E_obs_bank_formula"]
    report = {"contexts": len(trips),
              "exact": {"mean_dlogZ": float(d_logz.mean()), "mean_dE_obs": float(d_obs.mean()),
                        "mean_loglik_gain": float((d_obs - d_logz).mean())},
              "observed_energy_formula_gap": {
                  "parent_max_abs": float((exact["parent"]["E_obs"] - exact["parent"]["E_obs_bank_formula"]).abs().max()),
                  "delta_mean": float((d_obs_bank - d_obs).mean())},
              "banks": []}
    print(json.dumps(report, indent=2), flush=True)

    settings = [("production: warm, burn 3, 4 chains x 4 x 2", dict(chains=4, draws_per_chain=4, burn=3, bpz=2, warm=True)),
                ("warm, burn 30", dict(chains=4, draws_per_chain=4, burn=30, bpz=2, warm=True)),
                ("cold, burn 30", dict(chains=4, draws_per_chain=4, burn=30, bpz=2, warm=False)),
                ("warm, burn 3, 16 chains x 4 x 2 (128 draws)", dict(chains=16, draws_per_chain=4, burn=3, bpz=2, warm=True))]
    for label, s in settings:
        bank = draw_all(parent, batcher, trips, table, s["chains"], s["draws_per_chain"], s["burn"], 128,
                        torch.Generator().manual_seed(args.seed + 1), full,
                        observed_items if s["warm"] else None, s["bpz"])
        draws = s["chains"] * s["draws_per_chain"] * s["bpz"]
        with torch.no_grad():
            for m in (parent, child):
                m.house, m.ctx = full
            dE = (basket_energy(child, ix, bank) - basket_energy(parent, ix, bank)).view(len(trips), draws)
            est = torch.logsumexp(dE, 1) - np.log(draws)
            size = torch.bincount(bank.basket, minlength=bank.n).double().view(len(trips), draws).mean(1)
        gap = est - d_logz
        entry = {"bank": label, "draws": draws, "mean_bank_dlogZ": float(est.mean()),
                 "mean_gap_bank_minus_exact": float(gap.mean()),
                 "gap_se": float(gap.std() / np.sqrt(len(gap))),
                 "bank_loglik_gain": float((d_obs_bank - est).mean()),
                 "bank_mean_size": float(size.mean())}
        report["banks"].append(entry)
        print(json.dumps(entry), flush=True)
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
