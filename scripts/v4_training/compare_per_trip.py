#!/usr/bin/env python3
"""Paired per-trip comparison of a model against baselines on the same held-out trips.

Each file holds `trips` and `ll` (stress.py per-trip output), or is given as PATH:TRIPS_KEY:LL_KEY
(e.g. the chain's cached splits).  For every baseline: mean gain, its standard error and a
household-clustered 95% bootstrap interval, over all trips and over one-item and multi-item trips.
Households and basket sizes come from the bundle cache under $ENERGY_MODEL_DATA_ROOT.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np


def load(spec):
    path, *keys = spec.split(":")
    z = np.load(path)
    tk, lk = keys if keys else ("trips", "ll")
    return z[tk].astype(np.int64), z[lk].astype(np.float64)


def clustered_ci(delta, cluster, reps=2000, seed=0):
    ids, inv = np.unique(cluster, return_inverse=True)
    sums, counts = np.bincount(inv, delta), np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), (reps, len(ids)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return [round(float(np.quantile(means, 0.025)), 4), round(float(np.quantile(means, 0.975)), 4)]


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--new", required=True, help="NAME=SPEC of the model under test")
    p.add_argument("--base", nargs="+", required=True, help="NAME=SPEC baselines")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cache = np.load(Path(os.environ["ENERGY_MODEL_DATA_ROOT"]) / "basket_input/v3_index_affinity.npz")
    name, spec = a.new.split("=", 1)
    trips, ll = load(spec)
    house, size = cache["trip_user"][trips], cache["trip_nlines"][trips]
    masks = {"all": np.ones(len(trips), bool), "one_item": size == 1, "multi_item": size > 1}
    out = {"model": name, "trips": int(len(trips)), "mean_ll": round(float(ll.mean()), 4), "vs": {}}
    for item in a.base:
        bname, bspec = item.split("=", 1)
        bt, bll = load(bspec)
        if not np.array_equal(bt, trips):
            raise ValueError(f"{bname}: trips differ")
        res = {"baseline_mean_ll": round(float(bll.mean()), 4)}
        for k, m in masks.items():
            d = ll[m] - bll[m]
            res[k] = {"trips": int(m.sum()), "gain": round(float(d.mean()), 4),
                      "se": round(float(d.std(ddof=1) / np.sqrt(m.sum())), 4),
                      "household_clustered_95": clustered_ci(d, house[m])}
        out["vs"][bname] = res
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
