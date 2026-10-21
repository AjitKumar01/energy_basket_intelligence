#!/usr/bin/env python3
"""Checks for the optional discount split of the price input (disc_kappa, ctx["dsc"]).

1. Nesting: with disc_kappa = price_kappa the exact basket log-likelihood equals the parent's
   (no discount panel) on the same trips.
2. Gradient: autograd d LL / d disc_kappa against a central finite difference, at the nested
   point and at a moved value.
3. Counterfactual context: changed_price_context with a discount change moves dsc and dlp by
   the given amounts and leaves dsc alone without one.
4. The discount term is the only change: utilities differ from the parent by exactly
   -g (softplus(disc_kappa) - kappa) dsc.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/version4"))
torch.set_default_dtype(torch.float64)
from data import build                                     # noqa: E402
from features import Features                              # noqa: E402
from fit import Batcher                                    # noqa: E402
from fit_exact_additive import exact_loglik_and_size, load_model  # noqa: E402
from pipeline_support import supported_trips               # noqa: E402
from price_response import changed_price_context           # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--initialization", type=Path, required=True)
p.add_argument("--additive", type=Path, required=True)
p.add_argument("--discount-panel", type=Path, required=True)
p.add_argument("--trips", type=int, default=256)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(4)
data = build()
model, meta, _ = load_model(a.initialization, data)
model.load_state_dict(torch.load(a.additive, map_location="cpu", weights_only=False)["model"])
with torch.no_grad():
    model.phi.zero_()
nmax = int(meta["nmax"]); rec = not bool(meta.get("no_rec", False))
mk = lambda panel: Batcher(data, Features(int(data["n_item"]), int(data["n_store"]), include_recency=rec,
                                          include_store_price=False, discount_panel=panel), nmax, include_recency=rec)
b0, b1 = mk(False), mk(a.discount_panel)
trips = np.sort(np.random.default_rng(3).choice(supported_trips(data, 0, nmax), a.trips, replace=False))
batches = [(b0.make(trips[i:i + 64]), b1.make(trips[i:i + 64])) for i in range(0, len(trips), 64)]


def ll(batch_set, which):
    out = []
    for pair in batch_set:
        ix, ctx, lctx, house, li, lt, lc, _ = pair[which]
        model.house, model.ctx = house, ctx
        out.append(exact_loglik_and_size(model, ix, li, lt, lc, lctx)[0])
    return torch.cat(out)


res = {}
with torch.no_grad():
    base = ll(batches, 0)
    model.disc_kappa = torch.nn.Parameter(model.price_kappa.detach().clone())
    nested = ll(batches, 1)
    model.disc_common = torch.nn.Parameter(torch.zeros(()))
    nested_common = ll(batches, 1)
res["nesting_max_abs_diff"] = float(max((nested - base).abs().max(), (nested_common - base).abs().max()))

# 4. the utility difference is exactly the discount term
ix, ctx, lctx, house, *_ = batches[0][1]
model.house = house
with torch.no_grad():
    model.disc_kappa.fill_(float(model.price_kappa) + 1.0)
    u1 = model.b_at(ix.item, ix.item_trip, ctx)
    u0 = model.b_at(ix.item, ix.item_trip, {k: v for k, v in ctx.items() if k != "dsc"})
    sp = torch.nn.functional.softplus
    model.disc_common.fill_(0.3)
    u1 = model.b_at(ix.item, ix.item_trip, ctx)
    expect = (-model.price_coefficients(ix.item, ix.item_trip) * (sp(model.disc_kappa) - sp(model.price_kappa))
              - 0.3) * ctx["dsc"]
res["utility_term_max_abs_err"] = float((u1 - u0 - expect).abs().max())
res["share_slots_with_discount"] = float((ctx["dsc"] < 0).double().mean())

# 2. gradients wrt disc_kappa and disc_common
res["gradient"] = []
for shift, common in ((0.0, 0.0), (5.0, 0.4)):
    with torch.no_grad():
        model.disc_kappa.copy_(model.price_kappa.detach() + shift)
        model.disc_common.fill_(common)
    model.zero_grad(set_to_none=True)
    v = ll(batches, 1).mean()
    v.backward()
    for prm in (model.disc_kappa, model.disc_common):
        g = float(prm.grad)
        h = 1e-4
        with torch.no_grad():
            k = prm.detach().clone()
            prm.copy_(k + h); vp = float(ll(batches, 1).mean())
            prm.copy_(k - h); vm = float(ll(batches, 1).mean())
            prm.copy_(k)
        fd = (vp - vm) / (2 * h)
        res["gradient"].append({"param": "disc_kappa" if prm is model.disc_kappa else "disc_common",
                                "disc_kappa_minus_kappa_raw": shift, "disc_common": common,
                                "autograd": g, "finite_difference": fd,
                                "rel_err": abs(g - fd) / max(abs(fd), 1e-12)})

# 3. counterfactual context
d = torch.full_like(ctx["dlp"], -0.1)
c1 = changed_price_context(ctx, ix.item_trip, d, discount_change=d)
c2 = changed_price_context(ctx, ix.item_trip, d)
res["cf_dsc_moves"] = float((c1["dsc"] - ctx["dsc"] - d).abs().max())
res["cf_dlp_moves"] = float((c1["dlp"] - ctx["dlp"] - d).abs().max())
res["cf_without_discount_change_keeps_dsc"] = float((c2["dsc"] - ctx["dsc"]).abs().max())

res["pass"] = bool(res["nesting_max_abs_diff"] < 1e-9 and res["utility_term_max_abs_err"] < 1e-12
                   and all(r["rel_err"] < 1e-4 for r in res["gradient"])
                   and res["cf_dsc_moves"] < 1e-12 and res["cf_dlp_moves"] < 1e-12
                   and res["cf_without_discount_change_keeps_dsc"] == 0.0)
a.output.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
