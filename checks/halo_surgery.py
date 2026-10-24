#!/usr/bin/env python3
"""In-memory checkpoint surgery for diagnosing the display and price halo in a base model.

Applies two exact, in-memory parameter edits to a fitted additive base checkpoint (phi = 0)
without refitting or altering model code:

1. ``linear_size``:
   Replaces the learned size potential rho_0(n) (for n = 1..nmax) with a linear potential
   anchored at the first step,

       rho_0_lin(n) = rho_0(1) + (n - 1) * [rho_0(2) - rho_0(1)].

   In the base model's normaliser Z = sum_{n=1..nmax} exp(-rho_0(n)) [y^n] prod_c G_c(y),
   concavity of rho_0(n) (decreasing marginal size penalties Delta rho_0(n) = rho_0(n) -
   rho_0(n-1)) couples categories positively: an increase in one category's utility shifts
   the basket into larger sizes n where marginal penalties are smaller, pulling in items
   from all other categories. A linear rho_0(n) removes this curvature so that, apart from
   the empty-basket truncation (n >= 1), categories factorise independently.

2. ``no_dlp_bar``:
   In v4/model.py the price utility is

       -g_hj * [dlp_bar + kappa * (dlp - dlp_bar)]
       - g_hj * (disc_kappa - kappa) * dsc - disc_common * dsc
     = -g_hj * kappa * dlp + g_hj * (kappa - 1) * dlp_bar
       - g_hj * (disc_kappa - kappa) * dsc - disc_common * dsc.

   When category c is discounted, the assortment-mean log price dlp_bar falls by
   (J_c / J) * |delta|, which shifts every non-discounted product j not in c by
   -g_hj * (kappa - 1) * (J_c / J) * |delta|. Setting kappa = 1 while scaling
   softplus(beta) by kappa_old (and softplus(disc_kappa) by 1 / kappa_old when present)
   keeps every product's own regular-price slope g_hj * kappa and discount slope
   g_hj * disc_kappa unchanged, while setting the dlp_bar cross-term (kappa - 1) to zero.

Evaluation modes:
- ``display``: runs the exact real-data display regression of checks/halo_real.py on the
  same store-weeks, sampled trips, and seed, writing halo_real_linear_size.json (and any
  other requested edits) and comparing against the factual data and as-fitted model.
- ``price``: takes the top-12 commodities by training revenue (as in sim/promotion_env.py)
  on the validation weeks and cuts each commodity's price and discount depth by 30%
  (log(1 - 0.30)), reporting own-commodity and rest-of-basket expected items per trip and
  expected revenue at paid prices, alongside the fitted regular-price and discount slopes.

Usage:
    python3 checks/halo_surgery.py \\
        --data-root /path/to/bundle_commodity \\
        --checkpoint /path/to/v3_pipeline_additive_best.pt \\
        --initialization /path/to/initialization.pt \\
        --discount-panel /path/to/log_discount.npy \\
        --out-dir runs/dunnhumby_surgery --threads 2
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.nn.functional import softplus  # noqa: E402

torch.set_default_dtype(torch.float64)

from v4.batches import Batcher  # noqa: E402
from v4.bundle import data_root, load_bundle  # noqa: E402
from v4.checkpoints import freeze, load_additive  # noqa: E402
from v4.exact import incidence  # noqa: E402
from v4.features import WEEK_STRIDE, Features  # noqa: E402
from v4.price import changed_price_context  # noqa: E402


# ---- in-memory checkpoint surgery -----------------------------------------------------------
@torch.no_grad()
def apply_edit(model, orig_state: dict, edit: str):
    """Restore model parameters from orig_state and apply ``edit`` in memory.

    edit in {"as_fitted", "linear_size", "no_dlp_bar", "both"}.
    """
    model.rho_0_free.copy_(orig_state["rho_0_free"])
    model.price_kappa.copy_(orig_state["price_kappa"])
    model.beta.copy_(orig_state["beta"])
    if model.disc_kappa is not None and "disc_kappa" in orig_state:
        model.disc_kappa.copy_(orig_state["disc_kappa"])

    if edit in ("linear_size", "both"):
        rho0 = orig_state["rho_0_free"]
        d1 = float(rho0[1] - rho0[0])
        n = torch.arange(1, len(rho0) + 1, dtype=rho0.dtype)
        model.rho_0_free.copy_(rho0[0] + (n - 1) * d1)

    if edit in ("no_dlp_bar", "both"):
        k_old = float(softplus(orig_state["price_kappa"]))
        model.price_kappa.fill_(math.log(math.expm1(1.0)))
        b_old = softplus(orig_state["beta"])
        model.beta.copy_(torch.log(torch.expm1(b_old * k_old)))
        if model.disc_kappa is not None and "disc_kappa" in orig_state:
            dk_old = float(softplus(orig_state["disc_kappa"]))
            dk_new = dk_old / k_old
            model.disc_kappa.fill_(math.log(math.expm1(dk_new)))


def slope_summary(model) -> dict:
    """Compute summary statistics of regular-price and discount slopes from a fitted model."""
    with torch.no_grad():
        kappa = float(softplus(model.price_kappa))
        disc_kappa = float(softplus(model.disc_kappa)) if model.disc_kappa is not None else kappa
        disc_common = float(model.disc_common) if model.disc_common is not None else 0.0
        g_h = softplus(model.gamma)
        b_j = softplus(model.beta)
        g_mat = g_h @ b_j.T
        g_mean = float(g_mat.mean())
        g_sd = float(g_mat.std())
        g_med = float(torch.median(g_mat))
        rho0 = model.rho_0_free.cpu().numpy()
        d_rho0 = np.diff(np.concatenate([[0.0], rho0]))
    return {
        "g_mean": g_mean,
        "g_sd": g_sd,
        "g_median": g_med,
        "kappa": kappa,
        "disc_kappa": disc_kappa,
        "disc_common": disc_common,
        "regular_price_slope_mean_g_kappa": g_mean * kappa,
        "discount_slope_mean_g_disc_kappa": g_mean * disc_kappa,
        "discount_slope_total_mean_g_disc_kappa_plus_common": g_mean * disc_kappa + disc_common,
        "regular_plus_discount_combined_cut_slope_mean": g_mean * disc_kappa + disc_common,
        "d_rho_0_first_15": [round(float(x), 4) for x in d_rho0[:15]],
    }


# ---- Question 1: real-data display halo check (checks/halo_real.py logic) -------------------
def within(values, store, week, iters=50):
    v = values.astype(float).copy()
    for _ in range(iters):
        v -= pd.Series(v).groupby(store).transform("mean").to_numpy()
        v -= pd.Series(v).groupby(week).transform("mean").to_numpy()
    return v


def regress(y, X, store, week):
    yt = within(y, store, week)
    Xt = np.column_stack([within(x, store, week) for x in X.T])
    used = Xt.std(0) > 1e-9
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
def model_items_display(model, batcher, trips, commodity_of, K):
    out = np.zeros((len(trips), K))
    for s in range(0, len(trips), 128):
        batch = batcher.make(trips[s:s + 128])
        batch.bind(model)
        ix = batch.ix
        p = incidence(model, ix, model.slot_utilities(ix))[1]
        k = commodity_of[ix.item.numpy()]
        keep = k >= 0
        np.add.at(out[s:s + 128], (ix.item_trip.numpy()[keep], k[keep]), p.numpy()[keep])
    return out


def own_summaries(data_rows, rows, max_rse=0.5):
    rse = np.array([r["own_se"] / r["mean_items_per_trip"] for r in data_rows])
    rel = np.array([r["own"] / r["mean_items_per_trip"] for r in rows])
    ok = np.isfinite(rse) & np.isfinite(rel)
    good = ok & (rse < max_rse)
    w = np.where(ok, 1.0 / np.where(ok, rse, 1.0) ** 2, 0.0)
    return {
        "own_mean_relative_well_measured": float(rel[good].mean()),
        "own_well_measured_commodities": int(good.sum()),
        "own_precision_weighted_relative": float(np.sum(w[ok] * rel[ok]) / w[ok].sum()),
    }


def run_display_halo_check(data, root, model, batcher, orig_state, args, out_dir: Path):
    S = int(data["n_store"])
    items = pd.read_parquet(root / "basket_input/items.parquet").sort_values("item_id").reset_index(drop=True)
    commodity = items.COMMODITY_DESC.to_numpy()

    split, store, week = data["trip_split"], data["trip_store"].astype(np.int64), data["trip_week"].astype(np.int64)
    trips = np.flatnonzero(split <= 1)
    sw = store[trips] * WEEK_STRIDE + week[trips]
    counts = pd.Series(sw).value_counts()
    cells = np.sort(counts[counts >= args.min_trips].index.to_numpy())
    trips = trips[np.isin(sw, cells)]
    sw = store[trips] * WEEK_STRIDE + week[trips]
    cell = np.searchsorted(cells, sw)
    n_trips = np.bincount(cell, minlength=len(cells))

    lp, li = data["line_ptr"], data["line_item"]
    line_trip = np.repeat(np.arange(len(lp) - 1), np.diff(lp))
    on = np.isin(line_trip, trips)
    line_cell = np.searchsorted(cells, store[line_trip[on]] * WEEK_STRIDE + week[line_trip[on]])
    line_com = commodity[li[on]]
    top = pd.Series(line_com).value_counts().index[:args.commodities].to_list()
    k_of = {c: i for i, c in enumerate(top)}
    commodity_of = np.array([k_of.get(c, -1) for c in commodity])
    lk = commodity_of[li[on]]
    y_data = np.zeros((len(cells), len(top)))
    np.add.at(y_data, (line_cell[lk >= 0], lk[lk >= 0]), 1.0)
    y_data /= n_trips[:, None]

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
            rows.append({
                "commodity": name,
                "mean_items_per_trip": float(y[:, k].mean()),
                "own": float(beta[0]),
                "own_se": float(se[0]),
                "halo": float(beta[1]),
                "halo_se": float(se[1]),
            })
        halo = np.array([r["halo"] for r in rows])
        halo_se = np.array([r["halo_se"] for r in rows])
        return {
            "commodities": rows,
            "halo_sum_items_per_trip_per_100_displayed": float(halo.sum()),
            "halo_sum_se_if_independent": float(np.sqrt((halo_se ** 2).sum())),
            "halo_mean_relative": float(np.mean(halo / np.array([r["mean_items_per_trip"] for r in rows]))),
            "own_mean_relative": float(np.nanmean([r["own"] / r["mean_items_per_trip"] for r in rows])),
        }

    data_res = fit(y_data)
    data_res.update(own_summaries(data_res["commodities"], data_res["commodities"]))

    rng = np.random.default_rng(args.seed)
    chosen = np.concatenate([
        np.sort(rng.choice(trips[cell == c], min(args.model_trips, int(n_trips[c])), replace=False))
        for c in range(len(cells))
    ])
    chosen = chosen[np.diff(lp)[chosen] <= model.nmax]
    ccell = np.searchsorted(cells, store[chosen] * WEEK_STRIDE + week[chosen])
    cell_counts = np.bincount(ccell, minlength=len(cells)).clip(1)[:, None]

    results = {}
    for edit in ("linear_size", "as_fitted"):
        t0 = time.time()
        apply_edit(model, orig_state, edit)
        per_trip = model_items_display(model, batcher, chosen, commodity_of, len(top))
        y_model = np.zeros((len(cells), len(top)))
        np.add.at(y_model, ccell, per_trip)
        y_model /= cell_counts
        m_res = {"checkpoint": str(args.checkpoint), "edit": edit, "trips": int(len(chosen)), **fit(y_model)}
        m_res.update(own_summaries(data_res["commodities"], m_res["commodities"]))
        m_res["mean_total_items_per_trip_top60"] = float(y_model.sum(1).mean())
        m_res["seconds"] = round(time.time() - t0, 1)

        report = {
            "store_weeks": len(cells),
            "stores": int(len(np.unique(cs))),
            "trips": int(len(trips)),
            "displayed_per_store_week_mean": float(total.mean()),
            "displayed_within_week_sd_median": float(pd.Series(total).groupby(cw).std().median()),
            "data": data_res,
            "model": m_res,
        }
        out_file = out_dir / f"halo_real_{edit}.json"
        out_file.write_text(json.dumps(report, indent=1) + "\n")
        results[edit] = {k: v for k, v in m_res.items() if k != "commodities"}
        print(f"[display:{edit}] wrote {out_file.name}: " + json.dumps(results[edit]), flush=True)

    apply_edit(model, orig_state, "as_fitted")
    return {"data": {k: v for k, v in data_res.items() if k != "commodities"}, "models": results}


# ---- Question 2: top-12 commodities 30% discount cut (PromotionEnv logic) -------------------
def weekly_prices(root, items, J, n_weeks):
    week = pd.read_parquet(root / "data/price_week.parquet", columns=["PRODUCT_ID", "WEEK_NO", "price"])
    week = week.merge(items[["PRODUCT_ID", "item_id"]], on="PRODUCT_ID")
    grid = np.full((J, n_weeks), np.nan)
    grid[week.item_id.to_numpy(), week.WEEK_NO.clip(upper=n_weeks - 1).to_numpy()] = week.price.to_numpy()
    return pd.DataFrame(grid).ffill(axis=1).bfill(axis=1).to_numpy()


def run_price_promotion_check(data, root, model, batcher, orig_state, args, out_dir: Path):
    J = int(data["n_item"])
    items = pd.read_parquet(root / "basket_input/items.parquet").sort_values("item_id").reset_index(drop=True)
    lines = pd.read_parquet(root / "basket_input/baskets.parquet", columns=["item_id", "units", "price", "split"])
    train = lines[lines.split == "train"]
    units = train.groupby("item_id").units.mean().reindex(range(J)).fillna(1.0).to_numpy()
    revenue = (train.units * train.price).groupby(items.COMMODITY_DESC.to_numpy()[train.item_id]).sum()
    categories = list(revenue.sort_values(ascending=False).index[:args.n_categories])
    item_category = np.full(J, -1)
    for k, c in enumerate(categories):
        item_category[(items.COMMODITY_DESC == c).to_numpy()] = k

    n_weeks = int(data["trip_week"].max()) + 1
    price_grid = weekly_prices(root, items, J, n_weeks)

    meta_split = data["trip_split"]
    weeks_all = data["trip_week"]
    weeks = sorted(int(w) for w in np.unique(weeks_all[meta_split == 1]))
    rng = np.random.default_rng(args.seed)
    nmax = model.nmax
    week_trips = {}
    for w in weeks:
        pool = np.flatnonzero((weeks_all == w) & (data["trip_nlines"] <= nmax))
        week_trips[w] = np.sort(rng.choice(pool, size=min(args.trips_per_week, len(pool)), replace=False))
    total_trips = sum(len(v) for v in week_trips.values())

    # Pre-build batches once for the sampled validation trips
    week_batches = {}
    for w in weeks:
        trips = week_trips[w]
        week_batches[w] = [batcher.make(trips[s:s + 128]) for s in range(0, len(trips), 128)]

    def simulate_action(depth_vec, dlp_moves=True, dsc_moves=True):
        """Return total and per-commodity expected items and paid revenue across validation weeks."""
        tot_items, tot_rev, reg_rev = 0.0, 0.0, 0.0
        cat_items = np.zeros(len(categories))
        cat_rev = np.zeros(len(categories))
        for w in weeks:
            for batch in week_batches[w]:
                ix = batch.ix
                slot_item = ix.item.numpy()
                k = item_category[slot_item]
                d = np.where(k >= 0, depth_vec[np.maximum(k, 0)], 0.0)
                if d.any():
                    change = torch.from_numpy(np.log1p(-d))
                    dlp_chg = change if dlp_moves else torch.zeros_like(change)
                    dsc_chg = change if dsc_moves else None
                    ctx = changed_price_context(batch.ctx, ix.item_trip, dlp_chg, discount_change=dsc_chg)
                else:
                    ctx = batch.ctx
                batch.bind(model, ctx)
                p = incidence(model, ix, model.slot_utilities(ix))[1].detach().numpy()
                reg_val = p * units[slot_item] * price_grid[slot_item, w]
                paid_val = reg_val * (1.0 - d)
                tot_items += float(p.sum())
                reg_rev += float(reg_val.sum())
                tot_rev += float(paid_val.sum())
                mask = k >= 0
                np.add.at(cat_items, k[mask], p[mask])
                np.add.at(cat_rev, k[mask], paid_val[mask])
        return {
            "total_items": tot_items,
            "items_per_trip": tot_items / total_trips,
            "total_revenue": tot_rev,
            "revenue_per_trip": tot_rev / total_trips,
            "regular_revenue": reg_rev,
            "cat_items": cat_items,
            "cat_items_per_trip": cat_items / total_trips,
            "cat_revenue": cat_rev,
            "cat_revenue_per_trip": cat_rev / total_trips,
        }

    K = len(categories)
    edits_out = {}
    for edit in ("as_fitted", "linear_size", "no_dlp_bar", "both"):
        t0 = time.time()
        apply_edit(model, orig_state, edit)
        base = simulate_action(np.zeros(K))
        per_commodity = []
        own_item_chg, rest_item_chg = [], []
        own_item_rel, rest_item_rel, rest_item_tot_rel = [], [], []
        own_rev_chg, rest_rev_chg = [], []
        own_rev_rel, rest_rev_rel, rest_rev_tot_rel = [], [], []

        for k, cname in enumerate(categories):
            depth = np.zeros(K)
            depth[k] = 0.30
            sim = simulate_action(depth)

            own_i0, own_i1 = float(base["cat_items_per_trip"][k]), float(sim["cat_items_per_trip"][k])
            rest_i0 = float(base["items_per_trip"] - own_i0)
            rest_i1 = float(sim["items_per_trip"] - own_i1)

            own_r0, own_r1 = float(base["cat_revenue_per_trip"][k]), float(sim["cat_revenue_per_trip"][k])
            rest_r0 = float(base["revenue_per_trip"] - own_r0)
            rest_r1 = float(sim["revenue_per_trip"] - own_r1)

            own_item_chg.append(own_i1 - own_i0)
            rest_item_chg.append(rest_i1 - rest_i0)
            own_item_rel.append(own_i1 / own_i0 - 1.0)
            rest_item_rel.append(rest_i1 / rest_i0 - 1.0)
            rest_item_tot_rel.append((rest_i1 - rest_i0) / float(base["items_per_trip"]))

            own_rev_chg.append(own_r1 - own_r0)
            rest_rev_chg.append(rest_r1 - rest_r0)
            own_rev_rel.append(own_r1 / own_r0 - 1.0)
            rest_rev_rel.append(rest_r1 / rest_r0 - 1.0)
            rest_rev_tot_rel.append((rest_r1 - rest_r0) / float(base["revenue_per_trip"]))

            per_commodity.append({
                "commodity": cname,
                "own_items_per_trip_base": own_i0,
                "own_items_per_trip_cut30": own_i1,
                "own_items_per_trip_diff": own_i1 - own_i0,
                "own_items_rel_change": own_i1 / own_i0 - 1.0,
                "rest_items_per_trip_base": rest_i0,
                "rest_items_per_trip_cut30": rest_i1,
                "rest_items_per_trip_diff": rest_i1 - rest_i0,
                "rest_items_rel_change": rest_i1 / rest_i0 - 1.0,
                "rest_items_share_of_total_change": (rest_i1 - rest_i0) / float(base["items_per_trip"]),
                "own_revenue_per_trip_base": own_r0,
                "own_revenue_per_trip_cut30": own_r1,
                "own_revenue_per_trip_diff": own_r1 - own_r0,
                "own_revenue_rel_change": own_r1 / own_r0 - 1.0,
                "rest_revenue_per_trip_base": rest_r0,
                "rest_revenue_per_trip_cut30": rest_r1,
                "rest_revenue_per_trip_diff": rest_r1 - rest_r0,
                "rest_revenue_rel_change": rest_r1 / rest_r0 - 1.0,
                "rest_revenue_share_of_total_change": (rest_r1 - rest_r0) / float(base["revenue_per_trip"]),
            })

        summary = {
            "edit": edit,
            "trips": int(total_trips),
            "weeks": weeks,
            "baseline_items_per_trip": float(base["items_per_trip"]),
            "baseline_revenue_per_trip": float(base["revenue_per_trip"]),
            "baseline_total_revenue": float(base["total_revenue"]),
            "mean_own_items_per_trip_base": float(np.mean([r["own_items_per_trip_base"] for r in per_commodity])),
            "mean_own_items_per_trip_cut30": float(np.mean([r["own_items_per_trip_cut30"] for r in per_commodity])),
            "mean_own_items_per_trip_diff": float(np.mean(own_item_chg)),
            "mean_own_items_rel_change": float(np.mean(own_item_rel)),
            "mean_rest_items_per_trip_base": float(np.mean([r["rest_items_per_trip_base"] for r in per_commodity])),
            "mean_rest_items_per_trip_cut30": float(np.mean([r["rest_items_per_trip_cut30"] for r in per_commodity])),
            "mean_rest_items_per_trip_diff": float(np.mean(rest_item_chg)),
            "mean_rest_items_rel_change": float(np.mean(rest_item_rel)),
            "mean_rest_items_share_of_total_change": float(np.mean(rest_item_tot_rel)),
            "mean_own_revenue_per_trip_base": float(np.mean([r["own_revenue_per_trip_base"] for r in per_commodity])),
            "mean_own_revenue_per_trip_cut30": float(np.mean([r["own_revenue_per_trip_cut30"] for r in per_commodity])),
            "mean_own_revenue_per_trip_diff": float(np.mean(own_rev_chg)),
            "mean_own_revenue_rel_change": float(np.mean(own_rev_rel)),
            "mean_rest_revenue_per_trip_base": float(np.mean([r["rest_revenue_per_trip_base"] for r in per_commodity])),
            "mean_rest_revenue_per_trip_cut30": float(np.mean([r["rest_revenue_per_trip_cut30"] for r in per_commodity])),
            "mean_rest_revenue_per_trip_diff": float(np.mean(rest_rev_chg)),
            "mean_rest_revenue_rel_change": float(np.mean(rest_rev_rel)),
            "mean_rest_revenue_share_of_total_change": float(np.mean(rest_rev_tot_rel)),
            "profitable_single_commodity_cuts_of_12": int(sum(
                (r["own_revenue_per_trip_diff"] + r["rest_revenue_per_trip_diff"]) > 0 for r in per_commodity
            )),
            "seconds": round(time.time() - t0, 1),
            "commodities": per_commodity,
        }
        edits_out[edit] = summary
        print(
            f"[price:{edit}] base E[N]={summary['baseline_items_per_trip']:.2f} "
            f"own_items={summary['mean_own_items_rel_change']*100:+.2f}% ({summary['mean_own_items_per_trip_diff']:+.4f}/trip) "
            f"rest_items={summary['mean_rest_items_rel_change']*100:+.2f}% ({summary['mean_rest_items_per_trip_diff']:+.4f}/trip) | "
            f"own_rev={summary['mean_own_revenue_rel_change']*100:+.2f}% "
            f"rest_rev(vs_tot)={summary['mean_rest_revenue_share_of_total_change']*100:+.2f}% "
            f"rest_rev(vs_rest)={summary['mean_rest_revenue_rel_change']*100:+.2f}%",
            flush=True,
        )

    apply_edit(model, orig_state, "as_fitted")
    return {"categories": categories, "edits": edits_out}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--initialization", type=Path, required=True)
    p.add_argument("--discount-panel", type=Path)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--commodities", type=int, default=60, help="top commodities for display halo check")
    p.add_argument("--min-trips", type=int, default=30, help="min trips per store-week for display halo check")
    p.add_argument("--model-trips", type=int, default=24, help="trips per store-week for display halo check")
    p.add_argument("--n-categories", type=int, default=12, help="top commodities for price promotion check")
    p.add_argument("--trips-per-week", type=int, default=500, help="trips per validation week for price promotion check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--mode", choices=("all", "display", "price"), default="all")
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    root = data_root(args.data_root)
    data = load_bundle(root)
    model, _, blob = load_additive(args.initialization, args.checkpoint, data, root)
    freeze(model)

    panel = args.discount_panel
    if panel is None and blob.get("discount_split") and blob["discount_split"].get("discount_panel"):
        cand = Path(blob["discount_split"]["discount_panel"])
        if cand.exists():
            panel = cand
    features = Features(root, int(data["n_item"]), int(data["n_store"]), recency=False,
                        price_contract=blob.get("price_feature_contract", "chain_product_week_only"),
                        discount_panel=panel)
    batcher = Batcher(data, features)

    orig_state = {
        "rho_0_free": model.rho_0_free.detach().clone(),
        "price_kappa": model.price_kappa.detach().clone(),
        "beta": model.beta.detach().clone(),
    }
    if model.disc_kappa is not None:
        orig_state["disc_kappa"] = model.disc_kappa.detach().clone()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    slopes = slope_summary(model)
    (out_dir / "slopes.json").write_text(json.dumps(slopes, indent=1) + "\n")
    print("[slopes] " + json.dumps(slopes), flush=True)

    full_report = {
        "checkpoint": str(args.checkpoint),
        "data_root": str(root),
        "slopes": slopes,
    }

    if args.mode in ("all", "display"):
        disp_res = run_display_halo_check(data, root, model, batcher, orig_state, args, out_dir)
        full_report["display_halo"] = disp_res

    if args.mode in ("all", "price"):
        price_res = run_price_promotion_check(data, root, model, batcher, orig_state, args, out_dir)
        (out_dir / "price_promotion_surgery.json").write_text(json.dumps(price_res, indent=1) + "\n")
        full_report["price_promotion"] = {
            "categories": price_res["categories"],
            "edits": {
                k: {kk: vv for kk, vv in v.items() if kk != "commodities"}
                for k, v in price_res["edits"].items()
            },
        }

    (out_dir / "surgery_summary.json").write_text(json.dumps(full_report, indent=1) + "\n")
    print(f"[done] wrote summary to {out_dir / 'surgery_summary.json'}", flush=True)


if __name__ == "__main__":
    main()
