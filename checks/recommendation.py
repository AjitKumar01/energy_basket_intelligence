#!/usr/bin/env python3
"""Recommendation check: hide one product of a held-out basket and rank every product the
store offers by the model's exact probability of being added to the rest of the basket.

Protocol (the same as the earlier pipeline's, so results compare directly):
    trips   --trips trips of the split, size within 1..nmax, in a seeded order (--seed);
            processed in groups of 24
    hidden  one product drawn uniformly from each basket with at least 2 products (seed + 17)
    rank    candidates = every offered product not already in the rest of the basket;
            score = E(rest + j) - E(rest), the exact log odds of adding j (the normaliser
            cancels); the hidden product's rank, ties sharing the average rank
    report  MRR, recall@5/10/20/100 and median rank, household-clustered standard errors;
            also a popularity ranking on the same cases

Version 4: the full change of energy (utilities, pair term, category and size terms, M4c).
Base model: utility - rho_c x (products of the category already in the basket); its size
term is the same for every candidate, so it does not change the ranking.

    recommendation.py --checkpoint CK [--initialization INIT] --split validation --out FILE
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from initialize import popularity_logits  # noqa: E402
from v4.bundle import data_root, load_bundle  # noqa: E402
from v4.checkpoints import batcher_for, freeze, load_model  # noqa: E402
from v4.training import conditional_log_odds  # noqa: E402


def midrank(score, position):
    """1-based rank of score[position], ties sharing the average rank."""
    target = score[position]
    return 1.0 + np.count_nonzero(score > target) + 0.5 * (np.count_nonzero(score == target) - 1)


def household_se(values, households):
    values, households = np.asarray(values, float), np.asarray(households)
    codes, idx = np.unique(households, return_inverse=True)
    resid = np.bincount(idx, values - values.mean())
    G = len(codes)
    return float(math.sqrt(G / max(G - 1, 1) * np.sum(resid ** 2)) / len(values))


def metrics(ranks, households):
    ranks = np.asarray(ranks, float)
    rr = 1.0 / ranks
    out = {"cases": int(len(ranks)), "mrr": float(rr.mean()), "mrr_se": household_se(rr, households),
           "median_rank": float(np.median(ranks)), "mean_rank": float(ranks.mean())}
    for k in (5, 10, 20, 100):
        out[f"recall_at_{k}"] = float((ranks <= k).mean())
    return out


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True, help="base or Version-4 checkpoint")
    p.add_argument("--initialization", type=Path, help="for checkpoints that do not record it")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--split", choices=("validation", "test"), default="validation")
    p.add_argument("--trips", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2560202)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    root = data_root(a.data_root)
    data = load_bundle(root)
    model, meta, parent, flags, _ = load_model(a.checkpoint, data, root, initialization=a.initialization)
    freeze(model)
    batcher = batcher_for(data, root, parent, recency=flags["recency"], purchase_history=flags["recency"])
    split = {"validation": 1, "test": 2}[a.split]
    population = np.flatnonzero((data["trip_split"] == split) & (data["trip_nlines"] <= model.nmax))
    trips = population[np.random.default_rng(a.seed).permutation(len(population))[:a.trips]]
    popularity = popularity_logits(data, np.flatnonzero(data["trip_split"] == 0)).numpy()
    rng = np.random.default_rng(a.seed + 17)
    ranks, pop_ranks, households = [], [], []
    for start in range(0, len(trips), 24):
        batch = batcher.make(trips[start:start + 24])
        batch.bind(model)
        ix = batch.ix
        cases = []                                   # (trip in batch, hidden product, rest)
        rest_mask = torch.zeros(len(ix.item), dtype=torch.bool)
        for b in range(ix.B):
            observed = torch.unique(batch.line_item[batch.line_trip == b])
            if observed.numel() < 2:
                continue
            hidden = int(observed[int(rng.integers(observed.numel()))])
            rest = observed[observed != hidden]
            rest_mask |= (ix.item_trip == b) & torch.isin(ix.item, rest)
            cases.append((b, hidden, rest))
        if model.rank > 0:
            score, _ = conditional_log_odds(model, ix, rest_mask)
        else:
            n_row = torch.zeros(ix.n_rows).index_add_(0, ix.row_of, rest_mask.double())
            score = model.slot_utilities(ix) - model.rho_c[ix.row_cat[ix.row_of]] * n_row[ix.row_of]
        for b, hidden, rest in cases:
            slots = torch.nonzero((ix.item_trip == b) & ~rest_mask, as_tuple=True)[0]
            position = torch.nonzero(ix.item[slots] == hidden, as_tuple=True)[0]
            if position.numel() != 1:
                continue
            pos = int(position[0])
            ranks.append(midrank(score[slots].numpy(), pos))
            pop_ranks.append(midrank(popularity[ix.item[slots].numpy()], pos))
            households.append(int(data["trip_user"][trips[start + b]]))
    report = {"checkpoint": str(a.checkpoint), "split": a.split, "rank": model.rank,
              "model": metrics(ranks, households), "popularity": metrics(pop_ranks, households),
              "per_case": {"ranks": ranks, "households": households}}
    a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"model": report["model"], "popularity": report["popularity"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
