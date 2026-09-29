"""Serving latency of a frozen Version-4 checkpoint: exact normalizer and basket generation.

For one checkpoint and its bundle this measures, on held-out contexts:

  normalizer  log Z(context) at the evaluation level (Smolyak rule, ESP/category dynamic program),
              for one context at a time (live request) and in batches of 128 (offline scoring);
  generation  one exact basket for one context by the warm-started blocked Gibbs chain
              (burn sweeps, then one draw; S | z drawn by the compiled reverse sampler), and a
              batch of 64 baskets for one context (one chain per basket).

Usage (from scripts/version4, with ENERGY_MODEL_DATA_ROOT set to the bundle):
  python benchmark_serving_latency.py --checkpoint <final checkpoint> --level-offset 3 --output out.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from checkpoint_io import load_checkpoint
from data import build
from features import Features
from fit import Batcher
from joint_refinement import draw_bank_fast, slot_table
from pipeline_support import install_quadrature, smolyak_rule, supported_trips

torch.set_default_dtype(torch.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--level-offset", type=int, default=2)
    p.add_argument("--contexts", type=int, default=32, help="single-context requests timed")
    p.add_argument("--burn", type=int, default=3)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    data = build()
    model, blob, meta = load_checkpoint(args.checkpoint, data)
    model.requires_grad_(False)
    rank, nmax, J = int(blob["active_rank"]), int(meta["nmax"]), int(data["n_item"])
    batcher = Batcher(data, Features(J, int(data["n_store"]), include_recency=False), nmax,
                      include_recency=False)
    trips = supported_trips(data, 1, nmax)
    trips = np.sort(trips[rng.permutation(len(trips))[:max(128, args.contexts)]])
    level = rank + args.level_offset
    rule = smolyak_rule(model, rank, level)
    install_quadrature(model, rule)

    def log_z(batch):
        vix, vctx, _vline, vhouse, *_ = batcher.make(batch)
        model.house, model.ctx = vhouse, vctx
        with torch.no_grad():
            return model.log_Z(vix, drop_empty=True)

    log_z(trips[:1])                                             # warm-up
    single = []
    for t in trips[:args.contexts]:
        t0 = time.perf_counter(); log_z(trips[trips == t]); single.append(time.perf_counter() - t0)
    t0 = time.perf_counter(); log_z(trips[:128]); batch_seconds = time.perf_counter() - t0
    assortment = float(np.mean([batcher.make(trips[k:k + 1])[0].item.numel() for k in range(8)]))

    model.quad = None

    def generate(t, chains):
        ix_rep, ctx_rep, _lc, house_rep, *_ = batcher.make(np.repeat([t], chains))
        model.house, model.ctx = house_rep, ctx_rep
        table = slot_table(batcher.make(np.asarray([t]))[0], J)
        return draw_bank_fast(model, ix_rep, np.zeros(chains, dtype=np.int64), table, 1, args.burn,
                              generator)

    generate(trips[0], 1)                                        # compile and warm up
    one, sixty_four, sizes = [], [], []
    for t in trips[:args.contexts]:
        t0 = time.perf_counter(); bank = generate(t, 1); one.append(time.perf_counter() - t0)
        sizes.append(bank.slots.numel() / bank.n)
    for t in trips[:8]:
        t0 = time.perf_counter(); generate(t, 64); sixty_four.append(time.perf_counter() - t0)

    report = {"checkpoint": str(args.checkpoint), "products": J, "rank": rank, "level": level,
              "quadrature_nodes": int(len(rule[1])), "mean_assortment_per_context": assortment,
              "normalizer": {"single_context_ms_median": 1e3 * float(np.median(single)),
                             "single_context_ms_p95": 1e3 * float(np.quantile(single, 0.95)),
                             "batch128_ms_per_context": 1e3 * batch_seconds / 128},
              "generation": {"burn_sweeps": args.burn,
                             "one_basket_ms_median": 1e3 * float(np.median(one)),
                             "one_basket_ms_p95": 1e3 * float(np.quantile(one, 0.95)),
                             "sixty_four_baskets_ms_median": 1e3 * float(np.median(sixty_four)),
                             "mean_generated_size": float(np.mean(sizes))},
              "threads": args.threads}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
