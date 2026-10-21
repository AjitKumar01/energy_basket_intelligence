#!/usr/bin/env python3
"""How much probability would a fitted model put on baskets beyond the support caps?

A model may be fitted with at most R products from one category (--max-per-category).
This check evaluates the same parameters without that cap and reports, per trip, the
probability of baskets with more than R products from one category, and of baskets with
more than M categories present, both exactly from the dynamic program.

For a Version-4 model the probability is averaged over latent draws z from Gibbs chains,
reweighted from the capped to the uncapped law by Z_uncapped(z) / Z_capped(z) (THEORY.md 7.2).

    support_cap.py --checkpoint CK [--initialization INIT] --out FILE
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from v4.bundle import data_root, load_bundle, supported_trips  # noqa: E402
from v4.checkpoints import batcher_for, freeze, load_model  # noqa: E402
from v4.exact import _trip_grid, category_polynomials  # noqa: E402
from v4.kernels import CategoryProduct2D  # noqa: E402
from v4.sampling import draw_from_model, gibbs  # noqa: E402


@torch.no_grad()
def size_table(model, ix, u, z, iv):
    """log weight of baskets with n products in m categories [B, n, m], n, m = 0..nmax,
    with the model's size potential (on n, or on m with M3); the empty basket excluded."""
    log_g, scale = category_polynomials(model, ix, u, z=z, iv=iv)
    tilted, degree, slope = _trip_grid(model, ix, log_g)
    a = CategoryProduct2D.apply(tilted[0].contiguous(), degree.contiguous(), model.nmax)
    n = torch.arange(model.nmax + 1, dtype=a.dtype)
    a = a + slope[0, :, None, None] * n[None, :, None] + n[None, :, None] * scale[:, None, None]
    rho0 = model.rho_0()
    a = a - (rho0[None, None, :] if model.size_on_categories else rho0[None, :, None])
    a[:, 0, :] = -float("inf")
    return a


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--initialization", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--trips", type=int, default=256, help="validation trips")
    p.add_argument("--max-categories", type=int, nargs="+", default=[40, 60])
    p.add_argument("--draws", type=int, default=16, help="latent draws per trip (Version 4)")
    p.add_argument("--burn", type=int, default=100)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    root = data_root(a.data_root)
    data = load_bundle(root)
    model, meta, parent, flags, _ = load_model(a.checkpoint, data, root, initialization=a.initialization)
    freeze(model)
    batcher = batcher_for(data, root, parent, recency=flags["recency"], purchase_history=flags["recency"])
    R_cap = model.R
    trips = np.sort(np.random.default_rng(5).choice(supported_trips(data, 1, model.nmax), a.trips, replace=False))
    gen = torch.Generator().manual_seed(11)
    per_k, per_m = [], {M: [] for M in a.max_categories}
    for s in range(0, len(trips), 64):
        batch = batcher.make(trips[s:s + 64])
        batch.bind(model)
        ix = batch.ix
        if model.rank > 0:
            X = draw_from_model(model, ix, gen, a.burn)
            zs = []
            for _ in range(a.draws):
                X = gibbs(model, ix, X, gen)
                zs.append(model.basket_field(ix, X) + torch.randn(ix.B, model.rank, generator=gen))
        else:
            zs = [None]
        b = model.slot_utilities(ix)
        iv = model.iv_shift(ix, b_slot=b)
        lz_cap, lz_full, beyond_m = [], [], {M: [] for M in a.max_categories}
        for z in zs:
            u = model.slot_utilities_given_z(ix, z, b=b) if z is not None else b
            model.R = R_cap
            lz_cap.append(torch.logsumexp(size_table(model, ix, u, z, iv).reshape(ix.B, -1), -1))
            model.R = model.nmax
            full = size_table(model, ix, u, z, iv)
            lz = torch.logsumexp(full.reshape(ix.B, -1), -1)
            lz_full.append(lz)
            for M in a.max_categories:
                beyond_m[M].append((torch.logsumexp(full[:, :, M + 1:].reshape(ix.B, -1), -1) - lz).exp())
        model.R = R_cap
        lz_cap, lz_full = torch.stack(lz_cap, 1), torch.stack(lz_full, 1)
        w = torch.softmax(lz_full - lz_cap, 1)                 # capped draws -> uncapped law
        per_k.append((w * (1 - (lz_cap - lz_full).exp())).sum(1))
        for M in a.max_categories:
            per_m[M].append((w * torch.stack(beyond_m[M], 1)).sum(1))
    k = torch.cat(per_k)
    out = {"checkpoint": str(a.checkpoint), "rank": model.rank, "max_per_category": R_cap, "trips": len(trips),
           f"more_than_{R_cap}_in_one_category": {"mean": float(k.mean()), "max": float(k.max())}}
    for M in a.max_categories:
        v = torch.cat(per_m[M])
        out[f"more_than_{M}_categories"] = {"mean": float(v.mean()), "max": float(v.max())}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
