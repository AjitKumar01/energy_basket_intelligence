#!/usr/bin/env python3
"""Choose a fixed set of held-out reference trips for scoring (scripts/evaluate.py).

Every model is scored on the same trips, so per-trip log-likelihoods can be compared in
pairs.  The trips are the first --trips of a seeded random order of the split's trips
whose basket size is within the model's support (1..nmax); the order is kept, because the
evaluator draws its random numbers batch by batch in this order.

Writes an .npz with trips, household (for household-clustered intervals) and lines (basket
sizes).  With --base-model, also the base model's exact per-trip log-likelihood
(``base_model``); the evaluator compares a Version-4 model against every such column.

    reference_trips.py --initialization INIT --out FILE [--base-model ADDITIVE] [--split test]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from v4.bundle import data_root, load_bundle  # noqa: E402
from v4.checkpoints import batcher_for, freeze, load_additive  # noqa: E402
from v4.exact import additive_log_size  # noqa: E402

torch.set_default_dtype(torch.float64)


@torch.no_grad()
def base_model_loglik(initialization, additive, data, root, trips, chunk=128):
    """Exact log p(S) of the base model (phi = 0) on each trip."""
    model, _, blob = load_additive(initialization, additive, data, root)
    freeze(model)
    batcher = batcher_for(data, root, blob)
    out = []
    for s in range(0, len(trips), chunk):
        batch = batcher.make(trips[s:s + chunk])
        batch.bind(model)
        energy = model.additive_energy(batch.line_item, batch.line_trip, batch.line_cat, batch.ix.B,
                                       batch.line_ctx)
        out.append((energy - torch.logsumexp(additive_log_size(model, batch.ix), -1)).numpy())
    return np.concatenate(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--initialization", type=Path, required=True, help="gives the support 1..nmax")
    p.add_argument("--data-root", type=Path)
    p.add_argument("--split", choices=("validation", "test"), default="test")
    p.add_argument("--trips", type=int, default=4096)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--base-model", type=Path, help="add this base model's exact per-trip log-likelihood")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    root = data_root(a.data_root)
    data = load_bundle(root)
    nmax = int(torch.load(a.initialization, map_location="cpu", weights_only=False)["metadata"]["nmax"])
    split = {"validation": 1, "test": 2}[a.split]
    population = np.flatnonzero((data["trip_split"] == split) & (data["trip_nlines"] <= nmax))
    trips = population[np.random.default_rng(a.seed).permutation(len(population))[:a.trips]]
    lines = (data["line_ptr"][trips + 1] - data["line_ptr"][trips]).astype(np.int32)
    columns = {"trips": trips, "household": data["trip_user"][trips], "lines": lines}
    if a.base_model is not None:
        columns["base_model"] = base_model_loglik(a.initialization, a.base_model, data, root, trips)
        print(f"[reference] base model mean log-likelihood {columns['base_model'].mean():.4f}", flush=True)
    np.savez_compressed(a.out, **columns)
    print(f"[reference] {len(trips)} {a.split} trips (of {len(population)}) -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
