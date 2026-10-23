#!/usr/bin/env python3
"""Real-data halo check: do displays on *other* commodities change how much shoppers buy of
a commodity?  And does the model reproduce the answer?

Display is recorded per (product, store, week), so stores can be compared within the same
week.  For store-weeks with at least --min-trips trips (training and validation weeks;
test is left closed), and each of the --commodities most-bought commodities k:

    y_skw = items of k per trip in store s, week w
          = a_sk + g_kw + b_own * own_k + b_halo * other_k + e

    own_k    products of k on display in the store-week, per 10
    other_k  products of every other commodity on display, per 100

with store and week fixed effects per commodity (two-way within transformation) and
standard errors clustered by store.  b_halo is the halo: the change in k's items per trip
when 100 more products elsewhere in the store are on display.  Summed over the commodities,
it is the change in all other items per trip.

With --checkpoint the same regression is run on the model's expected items per trip, from
up to --model-trips trips per store-week with their real households and features (exact
incidence for a base model; for Version 4 the latent state is drawn once per trip at the
factual features, see --draws).  So data and model are compared on identical terms.

    halo_real.py --out FILE [--checkpoint CK [--initialization INIT]]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from v4.bundle import data_root, load_bundle  # noqa: E402
from v4.checkpoints import batcher_for, freeze, load_model  # noqa: E402
from v4.exact import incidence  # noqa: E402
from v4.features import WEEK_STRIDE  # noqa: E402
from v4.sampling import draw_from_model, gibbs  # noqa: E402


def within(values, store, week, iters=50):
    """Remove store and week means (alternating projections; unbalanced panels)."""
    v = values.astype(float).copy()
    for _ in range(iters):
        v -= pd.Series(v).groupby(store).transform("mean").to_numpy()
        v -= pd.Series(v).groupby(week).transform("mean").to_numpy()
    return v


def regress(y, X, store, week):
    """Two-way fixed-effects OLS of y on the columns of X; store-clustered standard errors."""
    yt = within(y, store, week)
    Xt = np.column_stack([within(x, store, week) for x in X.T])
    used = Xt.std(0) > 1e-9                     # a column with no variation left (never displayed) is dropped
    beta, se = np.full(X.shape[1], np.nan), np.full(X.shape[1], np.nan)
    Xu = Xt[:, used]
    XtX_inv = np.linalg.inv(Xu.T @ Xu)
    b = XtX_inv @ Xu.T @ yt
    resid = yt - Xu @ b
    meat = np.zeros((Xu.shape[1], Xu.shape[1]))
    for s in np.unique(store):
        g = Xu[store == s].T @ resid[store == s]
        meat += np.outer(g, g)
    beta[used], se[used] = b, np.sqrt(np.diag(XtX_inv @ meat @ XtX_inv))
    return beta, se


@torch.no_grad()
def model_items(model, batcher, trips, commodity_of, K, draws, gen):
    """Expected items per trip of each of the K commodities (rows: trips)."""
    out = np.zeros((len(trips), K))
    for s in range(0, len(trips), 128):
        batch = batcher.make(trips[s:s + 128])
        batch.bind(model)
        ix = batch.ix
        if model.rank == 0:
            p = incidence(model, ix, model.slot_utilities(ix))[1]
        else:
            X, p = draw_from_model(model, ix, gen, 100), 0.0
            iv = model.iv_shift(ix)
            for _ in range(draws):
                X = gibbs(model, ix, X, gen)
                z = model.basket_field(ix, X) + torch.randn(ix.B, model.rank, generator=gen)
                p = p + incidence(model, ix, model.slot_utilities_given_z(ix, z), z=z, iv=iv)[1] / draws
        k = commodity_of[ix.item.numpy()]
        keep = k >= 0
        np.add.at(out[s:s + 128], (ix.item_trip.numpy()[keep], k[keep]), p.numpy()[keep])
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--initialization", type=Path)
    p.add_argument("--additive", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--commodities", type=int, default=60)
    p.add_argument("--min-trips", type=int, default=30)
    p.add_argument("--model-trips", type=int, default=24, help="trips per store-week for the model")
    p.add_argument("--draws", type=int, default=16, help="Version 4: latent draws per trip")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    root = data_root(a.data_root)
    data = load_bundle(root)
    S = int(data["n_store"])
    items = pd.read_parquet(root / "basket_input/items.parquet").sort_values("item_id").reset_index(drop=True)
    commodity = items.COMMODITY_DESC.to_numpy()

    # Trips of training and validation weeks, grouped by store-week.
    split, store, week = data["trip_split"], data["trip_store"].astype(np.int64), data["trip_week"].astype(np.int64)
    trips = np.flatnonzero(split <= 1)
    sw = store[trips] * WEEK_STRIDE + week[trips]
    counts = pd.Series(sw).value_counts()
    cells = np.sort(counts[counts >= a.min_trips].index.to_numpy())
    trips = trips[np.isin(sw, cells)]
    sw = store[trips] * WEEK_STRIDE + week[trips]
    cell = np.searchsorted(cells, sw)
    n_trips = np.bincount(cell, minlength=len(cells))

    # Observed items per trip of the top commodities.
    lp, li = data["line_ptr"], data["line_item"]
    line_trip = np.repeat(np.arange(len(lp) - 1), np.diff(lp))
    on = np.isin(line_trip, trips)
    line_cell = np.searchsorted(cells, store[line_trip[on]] * WEEK_STRIDE + week[line_trip[on]])
    line_com = commodity[li[on]]
    top = pd.Series(line_com).value_counts().index[:a.commodities].to_list()
    k_of = {c: i for i, c in enumerate(top)}
    commodity_of = np.array([k_of.get(c, -1) for c in commodity])
    lk = commodity_of[li[on]]
    y_data = np.zeros((len(cells), len(top)))
    np.add.at(y_data, (line_cell[lk >= 0], lk[lk >= 0]), 1.0)
    y_data /= n_trips[:, None]

    # Display per store-week: own commodity (per 10 products) and everything else (per 100).
    promo = np.load(root / "basket_input/promo.npz")
    keys, disp = promo["keys"], promo["disp"].astype(bool)
    pw = keys[disp] % WEEK_STRIDE
    ps = (keys[disp] // WEEK_STRIDE) % S
    pj = (keys[disp] // WEEK_STRIDE) // S
    pc = np.searchsorted(cells, ps * WEEK_STRIDE + pw)
    hit = (pc < len(cells)) & (cells[np.minimum(pc, len(cells) - 1)] == ps * WEEK_STRIDE + pw)
    displayed = np.zeros((len(cells), len(top)))
    pk = commodity_of[pj[hit]]
    np.add.at(displayed, (pc[hit][pk >= 0], pk[pk >= 0]), 1.0)
    total = np.bincount(pc[hit], minlength=len(cells)).astype(float)
    cs, cw = cells // WEEK_STRIDE, cells % WEEK_STRIDE

    def fit(y):
        rows = []
        for k, name in enumerate(top):
            X = np.column_stack([displayed[:, k] / 10.0, (total - displayed[:, k]) / 100.0])
            beta, se = regress(y[:, k], X, cs, cw)
            rows.append({"commodity": name, "mean_items_per_trip": float(y[:, k].mean()),
                         "own": float(beta[0]), "own_se": float(se[0]), "halo": float(beta[1]), "halo_se": float(se[1])})
        halo = np.array([r["halo"] for r in rows])
        halo_se = np.array([r["halo_se"] for r in rows])
        return {"commodities": rows,
                "halo_sum_items_per_trip_per_100_displayed": float(halo.sum()),
                "halo_sum_se_if_independent": float(np.sqrt((halo_se ** 2).sum())),
                "halo_mean_relative": float(np.mean(halo / np.array([r["mean_items_per_trip"] for r in rows]))),
                "own_mean_relative": float(np.nanmean([r["own"] / r["mean_items_per_trip"] for r in rows]))}

    report = {"store_weeks": len(cells), "stores": int(len(np.unique(cs))), "trips": int(len(trips)),
              "displayed_per_store_week_mean": float(total.mean()),
              "displayed_within_week_sd_median": float(pd.Series(total).groupby(cw).std().median()),
              "data": fit(y_data)}
    print(json.dumps({k: v for k, v in report["data"].items() if k != "commodities"}), flush=True)

    if a.checkpoint is not None:
        model, _, parent, flags, _ = load_model(a.checkpoint, data, root, initialization=a.initialization,
                                                additive=a.additive)
        freeze(model)
        batcher = batcher_for(data, root, parent, recency=flags["recency"], purchase_history=flags["recency"])
        rng, gen = np.random.default_rng(a.seed), torch.Generator().manual_seed(a.seed)
        chosen = np.concatenate([np.sort(rng.choice(trips[cell == c], min(a.model_trips, int(n_trips[c])), replace=False))
                                 for c in range(len(cells))])
        chosen = chosen[np.diff(lp)[chosen] <= model.nmax]
        per_trip = model_items(model, batcher, chosen, commodity_of, len(top), a.draws, gen)
        ccell = np.searchsorted(cells, store[chosen] * WEEK_STRIDE + week[chosen])
        y_model = np.zeros((len(cells), len(top)))
        np.add.at(y_model, ccell, per_trip)
        y_model /= np.bincount(ccell, minlength=len(cells)).clip(1)[:, None]
        report["model"] = {"checkpoint": str(a.checkpoint), "trips": int(len(chosen)), **fit(y_model)}
        print(json.dumps({k: v for k, v in report["model"].items() if k != "commodities"}), flush=True)
    a.out.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
