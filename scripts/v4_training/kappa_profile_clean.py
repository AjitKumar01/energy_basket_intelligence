#!/usr/bin/env python3
"""Paired profile of the global relative-price multiplier (softplus(price_kappa)) of an additive
checkpoint on a chosen price panel: exact basket log-likelihood per trip at each scale, reported
as the mean difference from scale 1 with a paired standard error, on training and validation trips.
With --discount-panel the grid is two-dimensional: --scales multiplies the regular-price
multiplier and --disc-scales the discount multiplier (both relative to the parent's kappa)."""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/version4"))
torch.set_default_dtype(torch.float64)
from data import build
from features import Features
from fit import Batcher
from fit_exact_additive import exact_loglik_and_size, load_model
from pipeline_support import supported_trips

p = argparse.ArgumentParser()
p.add_argument("--initialization", type=Path, required=True)
p.add_argument("--additive", type=Path, required=True)
p.add_argument("--store-price", action="store_true")
p.add_argument("--trips", type=int, default=4096)
p.add_argument("--scales", default="0.5,1,1.5,2,3,4")
p.add_argument("--discount-panel", type=Path)
p.add_argument("--disc-scales", default="1")
p.add_argument("--splits", default="0,1")
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(4)
data = build()
model, meta, _ = load_model(a.initialization, data)
model.load_state_dict(torch.load(a.additive, map_location="cpu", weights_only=False)["model"])
with torch.no_grad():
    model.phi.zero_()
model.eval()
nmax = int(meta["nmax"]); rec = not bool(meta.get("no_rec", False))
batcher = Batcher(data, Features(int(data["n_item"]), int(data["n_store"]), include_recency=rec,
                                 include_store_price=a.store_price,
                                 discount_panel=a.discount_panel or False), nmax, include_recency=rec)
k0 = float(torch.nn.functional.softplus(model.price_kappa))
inv = lambda y: y + np.log(-np.expm1(-y))
if a.discount_panel is not None:
    model.disc_kappa = torch.nn.Parameter(model.price_kappa.detach().clone())
grid = [(float(s), float(d)) for s in a.scales.split(",") for d in a.disc_scales.split(",")]
out = {"kappa": k0, "store_price": a.store_price, "discount_panel": str(a.discount_panel), "rows": []}
for split in map(int, a.splits.split(",")):
    pop = supported_trips(data, split, nmax)
    trips = np.sort(np.random.default_rng(5 + split).choice(pop, min(a.trips, len(pop)), replace=False))
    ll = {}
    for s, ds in grid:
        with torch.no_grad():
            model.price_kappa.fill_(inv(k0 * s))
            if a.discount_panel is not None:
                model.disc_kappa.fill_(inv(k0 * ds))
            vals = []
            for i in range(0, len(trips), 128):
                ix, ctx, line_ctx, house, li, lt, lc, _ = batcher.make(trips[i:i + 128])
                model.house, model.ctx = house, ctx
                vals.append(exact_loglik_and_size(model, ix, li, lt, lc, line_ctx)[0])
            ll[(s, ds)] = torch.cat(vals).numpy()
    for (s, ds), v in ll.items():
        d = v - ll[(1.0, 1.0)]
        row = {"split": ["train", "validation"][split], "scale": s, "disc_scale": ds, "kappa": k0 * s,
               "mean_ll": float(v.mean()), "diff_vs_1": float(d.mean()), "diff_se": float(d.std(ddof=1) / np.sqrt(len(d)))}
        out["rows"].append(row); print(json.dumps(row), flush=True)
with torch.no_grad():
    model.price_kappa.fill_(inv(k0))
a.output.write_text(json.dumps(out, indent=1))
