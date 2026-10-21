#!/usr/bin/env python3
"""Paired profile of one common discount slope c added to an additive checkpoint:
b_j -= c * dsc_j for every household and product (dsc = centred discount depth), everything
else fixed at the checkpoint.  Mean per-trip exact log-likelihood difference from c = 0."""
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
p.add_argument("--discount-panel", type=Path, required=True)
p.add_argument("--values", default="0,0.1,0.25,0.5,0.75,1,1.5")
p.add_argument("--splits", default="0,1")
p.add_argument("--trips", type=int, default=2048)
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
                                 include_store_price=False, discount_panel=a.discount_panel), nmax, include_recency=rec)
orig = model.b_at
c = {"v": 0.0}
model.b_at = lambda it, trip, ctx: orig(it, trip, ctx) - (c["v"] * ctx["dsc"] if ctx is not None and "dsc" in ctx else 0.0)
out = {"rows": []}
for split in map(int, a.splits.split(",")):
    trips = np.sort(np.random.default_rng(5 + split).choice(supported_trips(data, split, nmax), a.trips, replace=False))
    made = [batcher.make(trips[i:i + 128]) for i in range(0, len(trips), 128)]
    ll = {}
    for v in map(float, a.values.split(",")):
        c["v"] = v
        vals = []
        with torch.no_grad():
            for ix, ctx, lctx, house, li, lt, lc, _ in made:
                model.house, model.ctx = house, ctx
                vals.append(exact_loglik_and_size(model, ix, li, lt, lc, lctx)[0])
        ll[v] = torch.cat(vals).numpy()
    for v, x in ll.items():
        d = x - ll[0.0]
        row = {"split": ["train", "validation"][split], "c": v, "diff_vs_0": float(d.mean()),
               "diff_se": float(d.std(ddof=1) / np.sqrt(len(d)))}
        out["rows"].append(row); print(json.dumps(row), flush=True)
a.output.write_text(json.dumps(out, indent=1))
