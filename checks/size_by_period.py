#!/usr/bin/env python3
"""Does the model's expected basket size follow the real one across calendar time?

For trips from the early and late quarters of the training weeks, the validation weeks and
the test weeks: the model's expected basket size E[N] under its own law, the real size, and
the number of products the household had ever bought before the trip (the purchase-history
feature that grows with time).

E[N] for a Version-4 model is an importance-sampling average of the exact E[N | z] over
z, with a proposal that is a mixture of N(s(S_k), I) over two sets of chain states: 32
from a chain started at the real basket (where real baskets sit) and 32 after a long run
(--long sweeps; where the chain drifts if the law has weight on big baskets).  So both
regions are covered and their weights decide.  A base model's E[N] is exact.

    size_by_period.py --checkpoint CK [--initialization INIT] --out FILE
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from v4.bundle import data_root, load_bundle, supported_trips  # noqa: E402
from v4.checkpoints import batcher_for, freeze, load_model  # noqa: E402
from v4.exact import incidence  # noqa: E402
from v4.sampling import gibbs  # noqa: E402


@torch.no_grad()
def expected_size(model, batch, gen, *, long, draws):
    ix = batch.ix
    B, tr = ix.B, ix.item_trip.numpy()
    iv = model.iv_shift(ix)
    if model.rank == 0:
        return np.bincount(tr, incidence(model, ix, model.slot_utilities(ix))[1].numpy(), B)
    X, centres = batch.observed.clone(), []
    for _ in range(32):
        X = gibbs(model, ix, X, gen)
        centres.append(model.basket_field(ix, X))
    X = gibbs(model, ix, X, gen, sweeps=long)
    for _ in range(32):
        X = gibbs(model, ix, X, gen, sweeps=4)
        centres.append(model.basket_field(ix, X))
    C = torch.stack(centres)                                     # [64, B, rank]
    log_w, sizes = [], []
    for _ in range(draws):
        pick = torch.randint(0, len(C), (B,), generator=gen)
        z = C[pick, torch.arange(B)] + torch.randn(B, model.rank, generator=gen)
        lz, p = incidence(model, ix, model.slot_utilities_given_z(ix, z), z=z, iv=iv)
        log_q = torch.logsumexp(-0.5 * ((z[None] - C) ** 2).sum(-1), 0) - math.log(len(C))
        log_w.append((-0.5 * (z * z).sum(1) + lz - log_q).numpy())
        sizes.append(np.bincount(tr, p.numpy(), B))
    log_w = np.array(log_w)
    w = np.exp(log_w - log_w.max(0))
    return (w / w.sum(0) * np.array(sizes)).sum(0)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--initialization", type=Path)
    p.add_argument("--additive", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--trips", type=int, default=48, help="trips per period")
    p.add_argument("--long", type=int, default=1000)
    p.add_argument("--draws", type=int, default=256)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    root = data_root(a.data_root)
    data = load_bundle(root)
    model, _, parent, flags, _ = load_model(a.checkpoint, data, root, initialization=a.initialization,
                                            additive=a.additive)
    freeze(model)
    batcher = batcher_for(data, root, parent, recency=flags["recency"], purchase_history=flags["recency"])
    weeks, sizes = data["trip_week"], np.diff(data["line_ptr"])
    rng, gen = np.random.default_rng(a.seed), torch.Generator().manual_seed(a.seed)
    train = supported_trips(data, 0, model.nmax)
    periods = [("train_early", train[weeks[train] < np.quantile(weeks[train], 0.25)]),
               ("train_late", train[weeks[train] > np.quantile(weeks[train], 0.75)]),
               ("validation", supported_trips(data, 1, model.nmax)),
               ("test", supported_trips(data, 2, model.nmax))]
    rows = []
    for name, pool in periods:
        trips = np.sort(rng.choice(pool, min(a.trips, len(pool)), replace=False))
        batch = batcher.make(trips)
        batch.bind(model)
        en = expected_size(model, batch, gen, long=a.long, draws=a.draws)
        ever = None
        if "rec" in batch.ctx:                                   # purchase history: [slots, 3], first = ever bought
            ever = float(np.bincount(batch.ix.item_trip.numpy(), batch.ctx["rec"][:, 0].numpy(), batch.ix.B).mean())
        row = {"period": name, "weeks": [int(weeks[trips].min()), int(weeks[trips].max())], "trips": len(trips),
               "model_expected_size": round(float(en.mean()), 2), "model_median": round(float(np.median(en)), 2),
               "real_size": round(float(sizes[trips].mean()), 2),
               "products_ever_bought": None if ever is None else round(ever, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)
    a.out.write_text(json.dumps({"checkpoint": str(a.checkpoint), "periods": rows}, indent=1) + "\n")


if __name__ == "__main__":
    main()
