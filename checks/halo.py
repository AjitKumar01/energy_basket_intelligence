#!/usr/bin/env python3
"""Does a price change on one product spill over into the rest of the basket?  (Synthetic
worlds with known truth.)

For each of the world's price scenarios (one product's price x 0.8 or x 1.2), the change in
items per basket in the product's own category and in all *other* categories (the halo):

    truth        from the generator's re-simulation of the test weeks (common random numbers)
    model        on --trips test trips: exact for a base model; for Version 4 two ways --
                 "reweighted": latent draws at factual prices reweighted by Z_new(z) / Z_old(z)
                 (the law's own counterfactual, THEORY.md 7.2), and "z fixed": the same draws
                 kept with equal weight, i.e. each trip's latent state held fixed and only the
                 basket given z allowed to respond

    halo.py --checkpoint CK --world WORLD_DIR [--initialization INIT --additive ADD] --out FILE
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

from v4 import scenarios  # noqa: E402
from v4.bundle import data_root, load_bundle  # noqa: E402
from v4.checkpoints import batcher_for, freeze, load_model  # noqa: E402
from v4.exact import incidence  # noqa: E402
from v4.price import changed_price_context  # noqa: E402
from v4.sampling import draw_from_model, gibbs  # noqa: E402


PARTS = ("own_product", "own_category", "other_categories")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--world", type=Path, required=True)
    p.add_argument("--initialization", type=Path)
    p.add_argument("--additive", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--trips", type=int, default=400)
    p.add_argument("--draws", type=int, default=64)
    p.add_argument("--burn", type=int, default=200)
    p.add_argument("--thin", type=int, default=4)
    p.add_argument("--seed", type=int, default=41006)
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
    o = np.load(a.world / "observed.npz")
    truth = json.loads((a.world / "counterfactual_truth.json").read_text())
    J = int(data["n_item"])
    category = o["category"]                                # per modelled product
    original = o["original_item"]
    to_model = {int(v): i for i, v in enumerate(original)}
    rows = scenarios.test_rows(a.world, a.trips)
    gen = torch.Generator().manual_seed(a.seed)
    v4 = model.rank > 0

    # Factual pass: batches, latent draws and their factual log Z.
    batches = []
    for s in range(0, len(rows), 128):
        batch = batcher.make(rows[s:s + 128])
        batch.bind(model)
        zs = None
        if v4:
            X = draw_from_model(model, batch.ix, gen, a.burn)
            zs = []
            for k in range(a.draws * a.thin):
                X = gibbs(model, batch.ix, X, gen)
                if (k + 1) % a.thin == 0:
                    zs.append(model.basket_field(batch.ix, X) + torch.randn(batch.ix.B, model.rank, generator=gen))
        batches.append((batch, zs))

    def expected_items(ctx_change=None):
        """Expected purchases of every product per trip: {method: array[J]}."""
        out = {}
        for batch, zs in batches:
            ix = batch.ix
            ctx = batch.ctx if ctx_change is None else changed_price_context(batch.ctx, ix.item_trip, ctx_change(ix))
            batch.bind(model, ctx)
            if not v4:
                parts = {"exact": incidence(model, ix, model.slot_utilities(ix))[1]}
            else:
                iv = model.iv_shift(ix)
                res = [incidence(model, ix, model.slot_utilities_given_z(ix, z), z=z, iv=iv) for z in zs]
                lz, ps = torch.stack([r[0] for r in res], 1), torch.stack([r[1] for r in res], 1)
                parts = {"lz": lz, "p": ps}
            for name, value in parts.items():
                out.setdefault(name, []).append(value)
        batch.bind(model)
        return out

    def per_trip(p_list):
        inc = np.zeros(J)
        for (batch, _), p in zip(batches, p_list):
            np.add.at(inc, batch.ix.item.numpy(), p.numpy())
        return inc / len(rows)

    base = expected_items()
    results = []
    for sc in truth["scenarios"]:
        k = to_model.get(int(sc["product"]))
        if k is None:
            continue
        own_cat = category == category[k]
        t_change = np.zeros(J)
        change_all = np.asarray(sc["incidence_change_all_products"])
        t_change[[to_model[int(j)] for j in original]] = change_all[original]
        lm = math.log(sc["price_multiplier"])
        alt = expected_items(lambda ix: torch.where(ix.item == k, lm, 0.0))
        row = {"product": int(sc["product"]), "price_multiplier": sc["price_multiplier"],
               "truth": {"own_product": float(t_change[k]), "own_category": float(t_change[own_cat].sum()),
                         "other_categories": float(t_change[~own_cat].sum())}}
        if not v4:
            d = per_trip(alt["exact"]) - per_trip(base["exact"])
            row["model"] = {"own_product": float(d[k]), "own_category": float(d[own_cat].sum()),
                            "other_categories": float(d[~own_cat].sum())}
        else:
            factual = per_trip([p.mean(1) for p in base["p"]])           # equal weight at factual prices
            reweighted = per_trip([(pa * torch.softmax(la - lb, 1)[b.ix.item_trip]).sum(1)
                                   for (b, _), pa, la, lb in zip(batches, alt["p"], alt["lz"], base["lz"])])
            fixed = per_trip([pa.mean(1) for pa in alt["p"]])
            for name, value in (("reweighted", reweighted), ("z_fixed", fixed)):
                d = value - factual
                row[name] = {"own_product": float(d[k]), "own_category": float(d[own_cat].sum()),
                             "other_categories": float(d[~own_cat].sum())}
        results.append(row)
        print(json.dumps(row), flush=True)

    methods = [m for m in ("model", "reweighted", "z_fixed") if m in results[0]]
    summary = {}
    for mult in (0.8, 1.2):
        rs = [r for r in results if r["price_multiplier"] == mult]
        summary[f"x{mult}"] = {"scenarios": len(rs), "truth": {part: float(np.mean([r["truth"][part] for r in rs]))
                                                          for part in PARTS}}
        for m in methods:
            summary[f"x{mult}"][m] = {part: float(np.mean([r[m][part] for r in rs])) for part in PARTS}
    for m in methods:
        t = np.array([r["truth"]["other_categories"] for r in results])
        v = np.array([r[m]["other_categories"] for r in results])
        summary[f"{m}_other_categories_corr_with_truth"] = float(np.corrcoef(t, v)[0, 1])
    out = {"checkpoint": str(a.checkpoint), "trips": len(rows), "summary": summary, "scenarios": results}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
