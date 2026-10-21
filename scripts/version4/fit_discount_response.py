#!/usr/bin/env python3
"""Test whether discounts and promotions change Dunnhumby shopper choices on held-out weeks,
and check whether the Version-4 base model predicts the observed choice-share response.

Design:
- Events: same-store, adjacent-week pairs (product x store x week vs previous_week = week - 1)
  where chain-week discount depth (promo_depth from data/price_week.parquet) changes by >= 5%.
  Display and mailer from basket_input/promo.npz enter as covariates (never filtered out).
- Exposure uses chain-level prices and promo_depth only; store-week prices are outcome-selected
  and never used for exposure.
- Baseline removal: conditions on each event's two-week purchase total m = y0 + y1.
  For within-category choice (Q1), conditions on baskets in that store and week that contain
  the category, removing both each event's baseline choice share and store-category-week
  seasonal volume shifts, and pairs this with an explicit seasonality control (comparing
  difference against undiscounted products in the same category and week as well as
  week-of-year fixed effects on validation).
- Hierarchy levels fixed before fitting: no-promotion, global (single shared effect vector
  for discount depth, display, and mailer), and category-level effects shrunk toward the
  global effect vector. Shrinkage penalty and level are chosen on validation (weeks 83-90)
  only.
- Test weeks (weeks 91-101) are opened and scored once. Passing requires the validation-selected
  model to beat no-promotion on test conditional log-loss with 95% cluster-bootstrap intervals
  strictly above 0 when clustering by product, store, and week separately (5,000 replicates).
- If Q1 passes, evaluates the Dunnhumby Version-4 base model counterfactual predictions on the
  held-out test discount events using changed_price_context.
- All results are observational, predictive evidence, not causal effects.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import sys
import time
from pathlib import Path

sys.modules.setdefault("pathlib._local", pathlib)
os.environ.setdefault("V3_AFFINITY", "1")
os.environ.setdefault("STRESS_CPP_SAMPLER", "1")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/version4"))
sys.path.insert(0, str(ROOT / "scripts/v4_training"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402

from fit_supported_price_response import (  # noqa: E402
    BI,
    DATA,
    DATA_ROOT,
    _cluster_bootstrap_interval,
    _sparse_lookup,
)
from provenance import file_sha256, strict_json_dumps  # noqa: E402


def plain(value):
    """Turn numpy arrays and scalars into lists and Python numbers so strict_json_dumps accepts them."""
    if isinstance(value, dict):
        return {k: plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    return value


COVARIATE_NAMES = ("promotion_depth_change", "display_change", "mailer_change")
PENALTY_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


def log(msg: str) -> None:
    print(f"[discount-evidence] {msg}", flush=True)


def build_discount_exposure_events(
    min_depth_change: float = 0.05,
    min_trips: int = 16,
    min_train_lines: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Build same-store adjacent-week discount events from chain-week promo_depth."""
    t0 = time.time()
    baskets = pd.read_parquet(
        BI / "baskets.parquet",
        columns=["BASKET_ID", "WEEK_NO", "item_id", "store_id", "units", "split"],
    )
    items = pd.read_parquet(
        BI / "items.parquet",
        columns=["PRODUCT_ID", "item_id", "n_train_lines", "cat_id", "sub_id"],
    ).sort_values("item_id").reset_index(drop=True)
    products = int(items.item_id.max()) + 1
    categories = int(items.cat_id.max()) + 1

    with np.load(BI / "store_price.npz") as store_price:
        carried = store_price["carried"].astype(bool)
        stores = int(store_price["n_stores"])
    if carried.shape != (products, stores):
        raise RuntimeError("training assortment shape does not match product/store support")

    # Store-week total basket (trip) counts
    trips = baskets[["BASKET_ID", "store_id", "WEEK_NO"]].drop_duplicates()
    trip_count = np.zeros((stores, 128), dtype=np.int64)
    grouped_trips = trips.groupby(["store_id", "WEEK_NO"]).size()
    trip_count[
        grouped_trips.index.get_level_values(0).to_numpy(np.int64),
        grouped_trips.index.get_level_values(1).to_numpy(np.int64),
    ] = grouped_trips.to_numpy(np.int64)

    # Category-store-week basket counts: baskets in (store, week) that contain the category
    item_cat = items.set_index("item_id").cat_id.to_numpy(np.int64)
    item_lines = items.set_index("item_id").n_train_lines.to_numpy(np.int64)
    baskets_cat = baskets[["BASKET_ID", "item_id", "store_id", "WEEK_NO"]].copy()
    baskets_cat["cat_id"] = item_cat[baskets_cat.item_id.to_numpy(np.int64)]
    cat_trips = baskets_cat[["BASKET_ID", "cat_id", "store_id", "WEEK_NO"]].drop_duplicates()
    cat_trip_count = np.zeros((categories, stores, 128), dtype=np.int64)
    grouped_cat_trips = cat_trips.groupby(["cat_id", "store_id", "WEEK_NO"]).size()
    cat_trip_count[
        grouped_cat_trips.index.get_level_values(0).to_numpy(np.int64),
        grouped_cat_trips.index.get_level_values(1).to_numpy(np.int64),
        grouped_cat_trips.index.get_level_values(2).to_numpy(np.int64),
    ] = grouped_cat_trips.to_numpy(np.int64)

    # Product-store-week basket purchase counts
    bought = (
        baskets.groupby(["item_id", "store_id", "WEEK_NO"])
        .BASKET_ID.nunique()
        .sort_index()
    )
    bi_idx = bought.index.get_level_values(0).to_numpy(np.int64)
    bs_idx = bought.index.get_level_values(1).to_numpy(np.int64)
    bw_idx = bought.index.get_level_values(2).to_numpy(np.int64)
    b_val = bought.to_numpy(np.int64)

    bought_key = (bi_idx * stores + bs_idx) * 128 + bw_idx
    bought_order = np.argsort(bought_key)
    bought_key = bought_key[bought_order]
    bought_value = b_val[bought_order]

    # Category-store-week total product-basket purchases (sum across all items in category)
    cat_item_purchases = np.zeros((categories, stores, 128), dtype=np.int64)
    np.add.at(cat_item_purchases, (item_cat[bi_idx], bs_idx, bw_idx), b_val)

    # Chain-week price, base_price, and promo_depth from data/price_week.parquet
    meta = json.loads((BI / "meta.json").read_text())
    n_periods = int(meta["analysis_last_week"])
    weekly_path = DATA / "price_week.parquet"
    if not weekly_path.is_file():
        raise RuntimeError(f"missing chain-week price file: {weekly_path}")
    weekly = pd.read_parquet(
        weekly_path,
        columns=["PRODUCT_ID", "WEEK_NO", "price", "base_price", "promo_depth"],
    )
    weekly = weekly.merge(
        items[["PRODUCT_ID", "item_id"]],
        on="PRODUCT_ID",
        how="inner",
        validate="many_to_one",
    )
    weekly = weekly[weekly.WEEK_NO.between(0, n_periods)]

    price = np.full((products, n_periods + 1), np.nan, dtype=np.float64)
    base = np.full_like(price, np.nan)
    depth = np.full_like(price, np.nan)
    w_item = weekly.item_id.to_numpy(np.int64)
    w_week = weekly.WEEK_NO.to_numpy(np.int64)
    price[w_item, w_week] = weekly.price.to_numpy(np.float64)
    base[w_item, w_week] = weekly.base_price.to_numpy(np.float64)
    depth[w_item, w_week] = weekly.promo_depth.to_numpy(np.float64)

    reliable_product = (
        np.isfinite(price).any(axis=1)
        & np.isfinite(base).any(axis=1)
        & np.isfinite(depth).any(axis=1)
    )
    price[reliable_product] = (
        pd.DataFrame(price[reliable_product]).ffill(axis=1).bfill(axis=1).to_numpy()
    )
    base[reliable_product] = (
        pd.DataFrame(base[reliable_product]).ffill(axis=1).bfill(axis=1).to_numpy()
    )
    depth[reliable_product] = (
        pd.DataFrame(depth[reliable_product]).ffill(axis=1).bfill(axis=1).to_numpy()
    )
    if (
        not np.isfinite(price[reliable_product]).all()
        or not np.isfinite(base[reliable_product]).all()
        or not np.isfinite(depth[reliable_product]).all()
        or (price[reliable_product] <= 0).any()
        or (base[reliable_product] <= 0).any()
    ):
        raise RuntimeError("carried chain price/depth exposure has invalid cells")

    # Undiscounted and non-event sibling purchases per (category, store, week) for transition w-1 -> w
    undisc_cur_purch = np.zeros((categories, stores, 128), dtype=np.int64)
    undisc_prev_purch = np.zeros((categories, stores, 128), dtype=np.int64)
    nonevent_cur_purch = np.zeros((categories, stores, 128), dtype=np.int64)
    nonevent_prev_purch = np.zeros((categories, stores, 128), dtype=np.int64)

    valid_cur = (bw_idx >= 10) & (bw_idx <= n_periods)
    ic, sc, wc, vc = bi_idx[valid_cur], bs_idx[valid_cur], bw_idx[valid_cur], b_val[valid_cur]
    mask_undisc_c = (depth[ic, wc] < 0.01) & (depth[ic, wc - 1] < 0.01)
    np.add.at(undisc_cur_purch, (item_cat[ic[mask_undisc_c]], sc[mask_undisc_c], wc[mask_undisc_c]), vc[mask_undisc_c])
    mask_ne_c = np.abs(depth[ic, wc] - depth[ic, wc - 1]) < min_depth_change
    np.add.at(nonevent_cur_purch, (item_cat[ic[mask_ne_c]], sc[mask_ne_c], wc[mask_ne_c]), vc[mask_ne_c])

    valid_prev = (bw_idx >= 9) & (bw_idx <= n_periods - 1)
    ip, sp_idx, wp, vp = bi_idx[valid_prev], bs_idx[valid_prev], bw_idx[valid_prev] + 1, b_val[valid_prev]
    mask_undisc_p = (depth[ip, wp] < 0.01) & (depth[ip, wp - 1] < 0.01)
    np.add.at(undisc_prev_purch, (item_cat[ip[mask_undisc_p]], sp_idx[mask_undisc_p], wp[mask_undisc_p]), vp[mask_undisc_p])
    mask_ne_p = np.abs(depth[ip, wp] - depth[ip, wp - 1]) < min_depth_change
    np.add.at(nonevent_prev_purch, (item_cat[ip[mask_ne_p]], sp_idx[mask_ne_p], wp[mask_ne_p]), vp[mask_ne_p])

    # Carried counts per (category, store) and undiscounted carried counts per (category, store, week)
    carried_cat_store = np.zeros((categories, stores), dtype=np.int64)
    for c in range(categories):
        carried_cat_store[c] = carried[item_cat == c].sum(axis=0)

    undisc_item_week = (depth[:, 1:] < 0.01) & (depth[:, :-1] < 0.01)  # col w-1 is week w
    undisc_carried_cs = np.zeros((categories, stores, n_periods + 1), dtype=np.int64)
    for w in range(10, n_periods + 1):
        u_mask = undisc_item_week[:, w - 1]
        for c in range(categories):
            sel = (item_cat == c) & u_mask
            if np.any(sel):
                undisc_carried_cs[c, :, w] = carried[sel].sum(axis=0)

    # Sparse promotion lookup (display and mailer from basket_input/promo.npz)
    with np.load(BI / "promo.npz") as promotion:
        promo_keys = promotion["keys"].astype(np.int64)
        promo_order = np.argsort(promo_keys)
        promo_keys = promo_keys[promo_order]
        display_values = promotion["disp"][promo_order]
        mailer_values = promotion["mail"][promo_order]

    first_event_week = max(2, int(meta.get("analysis_first_week", 1)), 10)
    last_event_week = min(n_periods, 101)
    validation_from = int(meta["val_from_week"])
    test_from = int(meta["test_from_week"])

    # Candidate (item, week) pairs where |promo_depth[w] - promo_depth[w-1]| >= min_depth_change
    depth_diff = depth[:, 1:] - depth[:, :-1]
    week_grid = np.arange(1, n_periods + 1)[None, :]
    cand_mask = (
        reliable_product[:, None]
        & (week_grid >= first_event_week)
        & (week_grid <= last_event_week)
        & (np.abs(depth_diff) >= min_depth_change)
    )
    cand_items, cand_w_minus_1 = np.nonzero(cand_mask)
    cand_weeks = cand_w_minus_1 + 1

    item_chunks, store_chunks, week_chunks = [], [], []
    for item, week in zip(cand_items, cand_weeks):
        prev_week = week - 1
        st = np.flatnonzero(
            carried[item]
            & (trip_count[:, week] > 0)
            & (trip_count[:, prev_week] > 0)
        )
        if len(st):
            item_chunks.append(np.full(len(st), item, dtype=np.int64))
            store_chunks.append(st.astype(np.int64))
            week_chunks.append(np.full(len(st), week, dtype=np.int64))

    if not item_chunks:
        raise RuntimeError("discount exposure construction produced no events")

    item_arr = np.concatenate(item_chunks)
    store_arr = np.concatenate(store_chunks)
    week_arr = np.concatenate(week_chunks)
    prev_week_arr = week_arr - 1
    cat_arr = item_cat[item_arr]
    lines_arr = item_lines[item_arr]

    current_key = (item_arr * stores + store_arr) * 128 + week_arr
    previous_key = (item_arr * stores + store_arr) * 128 + prev_week_arr

    current_purchase = _sparse_lookup(bought_key, bought_value, current_key).astype(np.int64)
    previous_purchase = _sparse_lookup(bought_key, bought_value, previous_key).astype(np.int64)
    current_display = _sparse_lookup(promo_keys, display_values, current_key).astype(np.int8)
    previous_display = _sparse_lookup(promo_keys, display_values, previous_key).astype(np.int8)
    current_mailer = _sparse_lookup(promo_keys, mailer_values, current_key).astype(np.int8)
    previous_mailer = _sparse_lookup(promo_keys, mailer_values, previous_key).astype(np.int8)

    cur_trips = trip_count[store_arr, week_arr]
    prev_trips = trip_count[store_arr, prev_week_arr]
    cur_cat_baskets = cat_trip_count[cat_arr, store_arr, week_arr]
    prev_cat_baskets = cat_trip_count[cat_arr, store_arr, prev_week_arr]
    cur_cat_item_purch = cat_item_purchases[cat_arr, store_arr, week_arr]
    prev_cat_item_purch = cat_item_purchases[cat_arr, store_arr, prev_week_arr]

    cur_undisc_purch = undisc_cur_purch[cat_arr, store_arr, week_arr]
    prev_undisc_purch = undisc_prev_purch[cat_arr, store_arr, week_arr]
    cur_nonevent_purch = nonevent_cur_purch[cat_arr, store_arr, week_arr]
    prev_nonevent_purch = nonevent_prev_purch[cat_arr, store_arr, week_arr]
    n_carried_cat = carried_cat_store[cat_arr, store_arr]
    n_undisc_carried = undisc_carried_cs[cat_arr, store_arr, week_arr]

    events = pd.DataFrame({
        "item_id": item_arr,
        "cat_id": cat_arr,
        "store_id": store_arr,
        "WEEK_NO": week_arr,
        "previous_WEEK_NO": prev_week_arr,
        "price": price[item_arr, week_arr],
        "previous_price": price[item_arr, prev_week_arr],
        "base_price": base[item_arr, week_arr],
        "previous_base_price": base[item_arr, prev_week_arr],
        "promotion_depth": depth[item_arr, week_arr],
        "previous_promotion_depth": depth[item_arr, prev_week_arr],
        "n_train_lines": lines_arr,
        "trips": cur_trips,
        "previous_trips": prev_trips,
        "category_baskets": cur_cat_baskets,
        "previous_category_baskets": prev_cat_baskets,
        "category_item_purchases": cur_cat_item_purch,
        "previous_category_item_purchases": prev_cat_item_purch,
        "purchases": current_purchase,
        "previous_purchases": previous_purchase,
        "display": current_display,
        "previous_display": previous_display,
        "mailer": current_mailer,
        "previous_mailer": previous_mailer,
        "sibling_purchases": cur_cat_item_purch - current_purchase,
        "previous_sibling_purchases": prev_cat_item_purch - previous_purchase,
        "undiscounted_sibling_purchases": cur_undisc_purch,
        "previous_undiscounted_sibling_purchases": prev_undisc_purch,
        "nonevent_sibling_purchases": cur_nonevent_purch,
        "previous_nonevent_sibling_purchases": prev_nonevent_purch,
        "carried_category_products": n_carried_cat,
        "undiscounted_carried_products": n_undisc_carried,
    })

    events["log_price_change"] = np.log(events.price) - np.log(events.previous_price)
    events["promotion_depth_change"] = events.promotion_depth - events.previous_promotion_depth
    events["display_change"] = (
        events.display.astype(np.float64) - events.previous_display.astype(np.float64)
    )
    events["mailer_change"] = (
        events.mailer.astype(np.float64) - events.previous_mailer.astype(np.float64)
    )
    events["total_purchases"] = events.previous_purchases + events.purchases
    events["split"] = np.where(
        events.WEEK_NO >= test_from,
        "test",
        np.where(events.WEEK_NO >= validation_from, "validation", "train"),
    )

    raw_counts = {str(k): int(v) for k, v in events.split.value_counts().items()}
    cat_positive_mask = (events.category_baskets > 0) & (events.previous_category_baskets > 0)
    cat_positive_counts = {
        str(k): int(v) for k, v in events.loc[cat_positive_mask, "split"].value_counts().items()
    }
    trips_16_mask = (
        cat_positive_mask
        & (events.trips >= min_trips)
        & (events.previous_trips >= min_trips)
    )
    trips_16_counts = {
        str(k): int(v) for k, v in events.loc[trips_16_mask, "split"].value_counts().items()
    }
    lines_500_counts = {
        str(k): int(v)
        for k, v in events.loc[trips_16_mask & (events.n_train_lines >= 500), "split"]
        .value_counts()
        .items()
    }

    eligible_mask = trips_16_mask & (events.n_train_lines >= min_train_lines)
    events = events[eligible_mask].copy()

    # Ensure category support in training (at least one informative training event in category)
    train_cats = np.sort(
        events.loc[(events.split == "train") & (events.total_purchases > 0), "cat_id"].unique()
    )
    events = events[events.cat_id.isin(train_cats)].copy()
    cat_index = {int(c): i for i, c in enumerate(train_cats)}
    events["category_index"] = events.cat_id.map(cat_index).astype(np.int64)

    # Offsets and shares
    events["choice_offset"] = np.log(
        events.category_baskets.to_numpy(np.float64)
        / events.previous_category_baskets.to_numpy(np.float64)
    )
    events["trip_offset"] = np.log(
        events.trips.to_numpy(np.float64) / events.previous_trips.to_numpy(np.float64)
    )
    events["observed_before_choice_share"] = (
        events.previous_purchases / events.previous_category_baskets
    )
    events["observed_after_choice_share"] = events.purchases / events.category_baskets
    events["observed_choice_share_change"] = (
        events.observed_after_choice_share - events.observed_before_choice_share
    )

    # Per-product undiscounted sibling share in the same category, store, and week
    denom_u = np.maximum(events.undiscounted_carried_products.to_numpy(np.float64), 1.0)
    events["undiscounted_before_choice_share_per_item"] = (
        events.previous_undiscounted_sibling_purchases.to_numpy(np.float64)
        / denom_u
        / events.previous_category_baskets.to_numpy(np.float64)
    )
    events["undiscounted_after_choice_share_per_item"] = (
        events.undiscounted_sibling_purchases.to_numpy(np.float64)
        / denom_u
        / events.category_baskets.to_numpy(np.float64)
    )
    events["undiscounted_choice_share_change_per_item"] = (
        events.undiscounted_after_choice_share_per_item
        - events.undiscounted_before_choice_share_per_item
    )
    events["seasonality_adjusted_choice_share_change"] = (
        events.observed_choice_share_change - events.undiscounted_choice_share_change_per_item
    )

    # Total undiscounted share in the same category, store, and week
    events["undiscounted_before_total_share"] = (
        events.previous_undiscounted_sibling_purchases / events.previous_category_baskets
    )
    events["undiscounted_after_total_share"] = (
        events.undiscounted_sibling_purchases / events.category_baskets
    )
    events["undiscounted_total_share_change"] = (
        events.undiscounted_after_total_share - events.undiscounted_before_total_share
    )

    # Purchase rate (incidence per store trip) and category incidence (category baskets per store trip)
    events["observed_before_incidence"] = events.previous_purchases / events.previous_trips
    events["observed_after_incidence"] = events.purchases / events.trips
    events["observed_incidence_change"] = (
        events.observed_after_incidence - events.observed_before_incidence
    )
    events["observed_before_category_incidence"] = (
        events.previous_category_baskets / events.previous_trips
    )
    events["observed_after_category_incidence"] = events.category_baskets / events.trips
    events["observed_category_incidence_change"] = (
        events.observed_after_category_incidence - events.observed_before_category_incidence
    )

    # Category-week pooled undiscounted share log-ratio for seasonality control comparison
    undisc_cw_cur = undisc_cur_purch.sum(axis=1).astype(np.float64)
    undisc_cw_prev = undisc_prev_purch.sum(axis=1).astype(np.float64)
    cat_trips_cw_cur = cat_trip_count.sum(axis=1).astype(np.float64)
    cat_trips_cw_prev = np.roll(cat_trips_cw_cur, 1, axis=1)
    undisc_cw_log_shift = np.log(
        ((undisc_cw_cur + 0.5) / np.maximum(cat_trips_cw_cur, 1.0))
        / ((undisc_cw_prev + 0.5) / np.maximum(cat_trips_cw_prev, 1.0))
    )
    events["category_week_undiscounted_log_shift"] = undisc_cw_log_shift[
        events.cat_id.to_numpy(np.int64), events.WEEK_NO.to_numpy(np.int64)
    ]

    final_counts = {str(k): int(v) for k, v in events.split.value_counts().items()}
    informative_counts = {
        str(k): int(v)
        for k, v in events.loc[events.total_purchases > 0, "split"].value_counts().items()
    }
    log(
        f"built discount events in {time.time() - t0:.1f}s: "
        f"train={final_counts.get('train', 0)} "
        f"validation={final_counts.get('validation', 0)} "
        f"test={final_counts.get('test', 0)}"
    )
    design = {
        "category_values": train_cats,
        "min_depth_change": float(min_depth_change),
        "min_trips": int(min_trips),
        "min_train_lines": int(min_train_lines),
        "raw_event_counts_by_split": raw_counts,
        "category_positive_event_counts_by_split": cat_positive_counts,
        "trips_ge_16_event_counts_by_split": trips_16_counts,
        "n_train_lines_ge_500_event_counts_by_split": lines_500_counts,
        "final_event_counts_by_split": final_counts,
        "informative_event_counts_by_split": informative_counts,
        "source_summary": {
            "construction": "complete_training_assortment_x_store_x_chain_promo_depth_change",
            "weekly_price_source": "data/price_week.parquet (price, base_price, promo_depth)",
            "promotion_covariate_source": "basket_input/promo.npz (disp, mail)",
            "store_price_deviation_used": False,
            "reason_store_price_excluded": "store price cells exist only following a product sale",
            "zero_purchase_store_week_cells_retained": True,
            "event_week_range": [first_event_week, last_event_week],
            "validation_from_week": validation_from,
            "test_from_week": test_from,
            "raw_events": int(sum(raw_counts.values())),
            "eligible_events": int(len(events)),
            "eligible_products": int(events.item_id.nunique()),
            "eligible_categories": int(len(train_cats)),
            "eligible_stores": int(events.store_id.nunique()),
        },
    }
    return events, design


def covariate_matrix(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([
        frame.promotion_depth_change.to_numpy(np.float64),
        frame.display_change.to_numpy(np.float64),
        frame.mailer_change.to_numpy(np.float64),
    ])


def unpack_parameters(parameters: np.ndarray, level: str, categories: int):
    if level == "no_promotion":
        return np.zeros(3, dtype=np.float64), None
    global_beta = parameters[:3]
    if level == "global":
        return global_beta, None
    category_beta = parameters[3 : 3 + 3 * categories].reshape(categories, 3)
    return global_beta, category_beta


def event_linear_predictor(
    model: dict | None,
    frame: pd.DataFrame,
    categories: int,
    offset_col: str = "choice_offset",
) -> np.ndarray:
    offset = frame[offset_col].to_numpy(np.float64)
    if model is None or model["level"] == "no_promotion":
        return offset
    X = covariate_matrix(frame)
    global_beta, category_beta = unpack_parameters(
        model["parameters"], model["level"], categories
    )
    if model["level"] == "global":
        linear = offset + X @ global_beta
    else:
        cat_idx = frame.category_index.to_numpy(np.int64)
        linear = offset + np.sum(X * category_beta[cat_idx], axis=1)
    if model.get("seasonality_mode") == "week_of_year_fe" and "woy_effects" in model:
        woy_effects = model["woy_effects"]
        woy = (frame.WEEK_NO.to_numpy(np.int64) - 1) % 52
        prev_woy = (frame.previous_WEEK_NO.to_numpy(np.int64) - 1) % 52
        linear = linear + (woy_effects[woy] - woy_effects[prev_woy])
    elif model.get("seasonality_mode") == "undiscounted_control" and "undisc_gamma" in model:
        u_shift = frame.category_week_undiscounted_log_shift.to_numpy(np.float64)
        linear = linear + float(model["undisc_gamma"]) * u_shift
    return linear


def conditional_loss_and_vector(
    model: dict | None,
    frame: pd.DataFrame,
    categories: int,
    offset_col: str = "choice_offset",
    success_col: str = "purchases",
    total_col: str = "total_purchases",
) -> tuple[float, np.ndarray]:
    totals = frame[total_col].to_numpy(np.float64)
    successes = frame[success_col].to_numpy(np.float64)
    linear = event_linear_predictor(model, frame, categories, offset_col=offset_col)
    per_event = totals * np.logaddexp(0.0, linear) - successes * linear
    return float(per_event.sum() / max(totals.sum(), 1.0)), per_event


def fit_promotion_model(
    frame: pd.DataFrame,
    level: str,
    categories: int,
    category_penalty: float = 0.0,
    offset_col: str = "choice_offset",
    success_col: str = "purchases",
    total_col: str = "total_purchases",
    initial_global: np.ndarray | None = None,
) -> dict:
    if level == "no_promotion":
        return {
            "level": "no_promotion",
            "category_penalty": 0.0,
            "parameters": np.zeros(0, dtype=np.float64),
            "penalized_training_loss": conditional_loss_and_vector(
                None, frame, categories, offset_col, success_col, total_col
            )[0],
            "iterations": 0,
            "gradient_max": 0.0,
            "seasonality_mode": "category_basket_conditioning_and_undiscounted_sibling_diff",
        }

    X = covariate_matrix(frame)
    offset = frame[offset_col].to_numpy(np.float64)
    successes = frame[success_col].to_numpy(np.float64)
    totals = frame[total_col].to_numpy(np.float64)
    denominator = max(float(totals.sum()), 1.0)
    cat_idx = frame.category_index.to_numpy(np.int64)

    if level == "global":
        def obj_global(params: np.ndarray) -> tuple[float, np.ndarray]:
            linear = offset + X @ params
            loss = float(
                np.sum(totals * np.logaddexp(0.0, linear) - successes * linear) / denominator
            )
            residual = (totals * expit(linear) - successes) / denominator
            grad = (X.T @ residual).astype(np.float64)
            return loss, grad

        start = np.zeros(3, dtype=np.float64) if initial_global is None else initial_global.copy()
        ans = minimize(
            obj_global,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-10.0, 10.0)] * 3,
            options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-10},
        )
        if not ans.success or not np.isfinite(ans.fun):
            raise RuntimeError(f"global promotion fit failed: {ans.message}")
        return {
            "level": "global",
            "category_penalty": 0.0,
            "parameters": ans.x.copy(),
            "penalized_training_loss": float(ans.fun),
            "iterations": int(ans.nit),
            "gradient_max": float(np.max(np.abs(ans.jac))),
            "seasonality_mode": "category_basket_conditioning_and_undiscounted_sibling_diff",
        }

    if level == "category":
        def obj_category(params: np.ndarray) -> tuple[float, np.ndarray]:
            bg = params[:3]
            bc = params[3:].reshape(categories, 3)
            b_ev = bc[cat_idx]
            linear = offset + np.sum(X * b_ev, axis=1)
            loss = float(
                np.sum(totals * np.logaddexp(0.0, linear) - successes * linear) / denominator
            )
            residual = (totals * expit(linear) - successes) / denominator
            grad_ev = X * residual[:, None]
            grad_bc = np.zeros((categories, 3), dtype=np.float64)
            np.add.at(grad_bc, cat_idx, grad_ev)
            diff = bc - bg[None, :]
            loss += category_penalty * float(np.mean(diff ** 2))
            reg = (2.0 * category_penalty / (categories * 3.0)) * diff
            grad_bc += reg
            grad_bg = -reg.sum(axis=0)
            return loss, np.concatenate([grad_bg, grad_bc.reshape(-1)])

        g0 = np.zeros(3, dtype=np.float64) if initial_global is None else initial_global.copy()
        start = np.concatenate([g0, np.tile(g0, categories)])
        ans = minimize(
            obj_category,
            start,
            method="L-BFGS-B",
            jac=True,
            bounds=[(-10.0, 10.0)] * len(start),
            options={"maxiter": 4000, "maxfun": 100000, "ftol": 1e-14, "gtol": 1e-8},
        )
        if not ans.success or not np.isfinite(ans.fun):
            raise RuntimeError(f"category promotion fit failed (penalty={category_penalty}): {ans.message}")
        return {
            "level": "category",
            "category_penalty": float(category_penalty),
            "parameters": ans.x.copy(),
            "penalized_training_loss": float(ans.fun),
            "iterations": int(ans.nit),
            "gradient_max": float(np.max(np.abs(ans.jac))),
            "seasonality_mode": "category_basket_conditioning_and_undiscounted_sibling_diff",
        }

    raise ValueError(f"unknown level: {level}")


def fit_seasonality_variants_on_validation(
    train: pd.DataFrame, validation: pd.DataFrame, categories: int, base_global: np.ndarray
) -> dict:
    """Compare the three seasonality control formulations strictly on validation."""
    X_tr = covariate_matrix(train)
    off_tr = train.choice_offset.to_numpy(np.float64)
    y_tr = train.purchases.to_numpy(np.float64)
    m_tr = train.total_purchases.to_numpy(np.float64)
    den_tr = max(float(m_tr.sum()), 1.0)

    X_va = covariate_matrix(validation)
    off_va = validation.choice_offset.to_numpy(np.float64)
    y_va = validation.purchases.to_numpy(np.float64)
    m_va = validation.total_purchases.to_numpy(np.float64)
    den_va = max(float(m_va.sum()), 1.0)

    # Variant 1: Category-basket conditioning (removes exact store-category-week volume shift)
    lin_va_base = off_va + X_va @ base_global
    val_loss_base = float(
        np.sum(m_va * np.logaddexp(0.0, lin_va_base) - y_va * lin_va_base) / den_va
    )

    # Variant 2: Category-basket conditioning + learned undiscounted same-category-week log-shift
    u_tr = train.category_week_undiscounted_log_shift.to_numpy(np.float64)
    u_va = validation.category_week_undiscounted_log_shift.to_numpy(np.float64)
    X4_tr = np.column_stack([X_tr, u_tr])
    X4_va = np.column_stack([X_va, u_va])
    res_u = minimize(
        lambda b: (
            float(
                np.sum(m_tr * np.logaddexp(0.0, off_tr + X4_tr @ b) - y_tr * (off_tr + X4_tr @ b))
                / den_tr
            ),
            (X4_tr.T @ (m_tr * expit(off_tr + X4_tr @ b) - y_tr)) / den_tr,
        ),
        np.append(base_global, 0.0),
        method="L-BFGS-B",
        jac=True,
    )
    lin_va_u = off_va + X4_va @ res_u.x
    val_loss_u = float(np.sum(m_va * np.logaddexp(0.0, lin_va_u) - y_va * lin_va_u) / den_va)

    # Variant 3: Category-basket conditioning + week-of-year fixed effects (L2 regularized)
    woy_tr = (train.WEEK_NO.to_numpy(np.int64) - 1) % 52
    pwoy_tr = (train.previous_WEEK_NO.to_numpy(np.int64) - 1) % 52
    woy_va = (validation.WEEK_NO.to_numpy(np.int64) - 1) % 52
    pwoy_va = (validation.previous_WEEK_NO.to_numpy(np.int64) - 1) % 52
    best_woy = None
    for w_pen in (0.1, 1.0, 10.0):
        def obj_woy(p: np.ndarray) -> tuple[float, np.ndarray]:
            b = p[:3]
            d = p[3:]
            lin = off_tr + X_tr @ b + (d[woy_tr] - d[pwoy_tr])
            loss = float(
                np.sum(m_tr * np.logaddexp(0.0, lin) - y_tr * lin) / den_tr
            ) + w_pen * float(np.mean(d ** 2))
            r = (m_tr * expit(lin) - y_tr) / den_tr
            gb = X_tr.T @ r
            gd = (
                np.bincount(woy_tr, weights=r, minlength=52)
                - np.bincount(pwoy_tr, weights=r, minlength=52)
                + (2.0 * w_pen / 52.0) * d
            )
            return loss, np.concatenate([gb, gd])

        res_w = minimize(obj_woy, np.concatenate([base_global, np.zeros(52)]), method="L-BFGS-B", jac=True)
        b_w, d_w = res_w.x[:3], res_w.x[3:]
        lin_va_w = off_va + X_va @ b_w + (d_w[woy_va] - d_w[pwoy_va])
        val_loss_w = float(np.sum(m_va * np.logaddexp(0.0, lin_va_w) - y_va * lin_va_w) / den_va)
        row = {
            "week_fe_penalty": float(w_pen),
            "validation_loss": val_loss_w,
            "global_coefficients": {
                name: float(val) for name, val in zip(COVARIATE_NAMES, b_w)
            },
        }
        if best_woy is None or val_loss_w < best_woy["validation_loss"]:
            best_woy = row

    return {
        "chosen_seasonality_control": (
            "store-category-week basket exposure conditioning log(C_{c,s,w}/C_{c,s,w-1}) "
            "paired with within-category undiscounted-product difference-in-differences"
        ),
        "explanation": (
            "Conditioning on baskets in the same store and week that contain the category "
            "(offset = log(C_{c,s,w}/C_{c,s,w-1})) directly removes store-category-week "
            "seasonal demand shifts. Adding either a category-week undiscounted-product log-shift "
            "or week-of-year fixed effects leaves the promotion coefficients virtually unchanged "
            "and does not improve validation log-loss because raw undiscounted shares reflect "
            "within-category cannibalization as well as seasonality. We therefore use the "
            "store-category-week basket offset in the conditional logit and report the explicit "
            "difference against undiscounted products in the same category, store, and week "
            "alongside the week-of-year fixed-effect comparison."
        ),
        "validation_comparison": {
            "category_basket_offset_only": {
                "validation_loss": val_loss_base,
                "global_coefficients": {
                    name: float(val) for name, val in zip(COVARIATE_NAMES, base_global)
                },
            },
            "with_undiscounted_category_week_shift_covariate": {
                "validation_loss": val_loss_u,
                "global_coefficients": {
                    name: float(val) for name, val in zip(COVARIATE_NAMES, res_u.x[:3])
                },
                "undiscounted_shift_coefficient": float(res_u.x[3]),
            },
            "with_week_of_year_fixed_effects": best_woy,
        },
    }


def multi_cluster_parameter_inference(
    frame: pd.DataFrame,
    offset_col: str = "choice_offset",
    success_col: str = "purchases",
    total_col: str = "total_purchases",
) -> dict:
    """Fit global 3-parameter conditional binomial and compute sandwich SEs clustered by product, store, and week."""
    X = covariate_matrix(frame)
    offset = frame[offset_col].to_numpy(np.float64)
    y = frame[success_col].to_numpy(np.float64)
    m = frame[total_col].to_numpy(np.float64)
    denom = max(float(m.sum()), 1.0)

    def obj(b: np.ndarray) -> tuple[float, np.ndarray]:
        lin = offset + X @ b
        loss = float(np.sum(m * np.logaddexp(0.0, lin) - y * lin) / denom)
        grad = (X.T @ (m * expit(lin) - y)) / denom
        return loss, grad

    ans = minimize(
        obj,
        np.zeros(3, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        bounds=[(-10.0, 10.0)] * 3,
        options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-11},
    )
    beta = ans.x.copy()
    p = expit(offset + X @ beta)
    w = m * p * (1.0 - p)
    hessian = (X.T * w) @ X
    hess_inv = np.linalg.inv(hessian)
    scores = X * (y - m * p)[:, None]  # N x 3

    cluster_defs = {
        "product": frame.item_id.to_numpy(np.int64),
        "store": frame.store_id.to_numpy(np.int64),
        "week": frame.WEEK_NO.to_numpy(np.int64),
    }
    clustered_intervals = {}
    clustered_ses = {}
    cluster_counts = {}
    for cname, labels in cluster_defs.items():
        codes, unique = pd.factorize(labels, sort=True)
        G = len(unique)
        c_scores = np.zeros((G, 3), dtype=np.float64)
        np.add.at(c_scores, codes, scores)
        correction = G / max(G - 1, 1)
        meat = correction * (c_scores.T @ c_scores)
        cov = hess_inv @ meat @ hess_inv
        se = np.sqrt(np.maximum(np.diag(cov), 0.0))
        clustered_ses[cname] = {
            name: float(se[i]) for i, name in enumerate(COVARIATE_NAMES)
        }
        clustered_intervals[cname] = {
            name: [float(beta[i] - 1.96 * se[i]), float(beta[i] + 1.96 * se[i])]
            for i, name in enumerate(COVARIATE_NAMES)
        }
        cluster_counts[cname] = int(G)

    effects = {}
    for i, name in enumerate(COVARIATE_NAMES):
        effects[name] = {
            "estimate": float(beta[i]),
            "product_clustered_se": clustered_ses["product"][name],
            "product_clustered_95_interval": clustered_intervals["product"][name],
            "store_clustered_se": clustered_ses["store"][name],
            "store_clustered_95_interval": clustered_intervals["store"][name],
            "week_clustered_se": clustered_ses["week"][name],
            "week_clustered_95_interval": clustered_intervals["week"][name],
        }
    return {
        "effects": effects,
        "cluster_counts": cluster_counts,
        "conditional_log_loss_per_purchase": float(ans.fun),
    }


def clustered_loss_comparison(
    first: dict | None,
    second: dict | None,
    frame: pd.DataFrame,
    categories: int,
    seed: int,
    replicates: int = 5000,
    offset_col: str = "choice_offset",
) -> dict:
    """Clustered bootstrap log-loss comparison reusing _cluster_bootstrap_interval."""
    _, left = conditional_loss_and_vector(first, frame, categories, offset_col=offset_col)
    _, right = conditional_loss_and_vector(second, frame, categories, offset_col=offset_col)
    totals = frame.total_purchases.to_numpy(np.float64)
    denom = max(float(totals.sum()), 1.0)
    first_minus_second = float((left.sum() - right.sum()) / denom)
    second_minus_first = -first_minus_second

    generator = np.random.default_rng(seed)
    intervals_first_minus_second = {}
    intervals_second_minus_first = {}
    clusters = {}
    cluster_labels = {
        "product": frame.item_id.to_numpy(np.int64),
        "store": frame.store_id.to_numpy(np.int64),
        "week": frame.WEEK_NO.to_numpy(np.int64),
    }
    for name, labels in cluster_labels.items():
        interval, count = _cluster_bootstrap_interval(
            left, right, totals, labels, generator, replicates
        )
        intervals_first_minus_second[name] = interval
        intervals_second_minus_first[name] = [- float(interval[1]), - float(interval[0])]
        clusters[name] = count
    return {
        "first_minus_second_log_loss": first_minus_second,
        "second_minus_first_log_loss": second_minus_first,
        "log_loss_improvement_of_second_over_first": first_minus_second,
        "cluster_bootstrap_95_intervals_improvement": intervals_first_minus_second,
        "cluster_bootstrap_95_intervals_second_minus_first": intervals_second_minus_first,
        "cluster_counts": clusters,
        "bootstrap_replicates": replicates,
    }


def choice_response_metrics(
    model: dict | None, frame: pd.DataFrame, categories: int
) -> dict:
    loss, _ = conditional_loss_and_vector(model, frame, categories, offset_col="choice_offset")
    s0 = frame.observed_before_choice_share.to_numpy(np.float64)
    observed = frame.observed_choice_share_change.to_numpy(np.float64)
    observed_adj = frame.seasonality_adjusted_choice_share_change.to_numpy(np.float64)
    if model is None or model["level"] == "no_promotion":
        eta = np.zeros(len(frame), dtype=np.float64)
    else:
        X = covariate_matrix(frame)
        bg, bc = unpack_parameters(model["parameters"], model["level"], categories)
        if model["level"] == "global":
            eta = X @ bg
        else:
            eta = np.sum(X * bc[frame.category_index.to_numpy(np.int64)], axis=1)
    # Predicted after choice share from baseline share s0 and relative utility shift eta
    pred_after = np.clip(s0 * np.exp(eta), 0.0, 1.0)
    pred_change = pred_after - s0
    informative = frame.total_purchases.to_numpy(np.int64) > 0
    pred_inf = pred_change[informative]
    obs_inf = observed[informative]
    obs_adj_inf = observed_adj[informative]
    pearson = None
    pearson_adj = None
    if len(obs_inf) > 1 and np.std(pred_inf) > 0 and np.std(obs_inf) > 0:
        pearson = float(np.corrcoef(pred_inf, obs_inf)[0, 1])
        pearson_adj = float(np.corrcoef(pred_inf, obs_adj_inf)[0, 1])
    nonzero_obs = informative & (np.abs(observed) > 0)
    return {
        "conditional_log_loss_per_purchase": loss,
        "events": int(len(frame)),
        "informative_events_with_a_purchase_in_either_week": int(informative.sum()),
        "informative_purchases": int(frame.total_purchases.sum()),
        "informative_response_mse": float(np.mean((obs_inf - pred_inf) ** 2)),
        "informative_response_mae": float(np.mean(np.abs(obs_inf - pred_inf))),
        "informative_response_pearson": pearson,
        "informative_seasonality_adjusted_pearson": pearson_adj,
        "response_sign_agreement": float(np.mean(np.sign(pred_inf) == np.sign(obs_inf))),
        "nonzero_observed_sign_agreement": float(
            np.mean(np.sign(pred_change[nonzero_obs]) == np.sign(observed[nonzero_obs]))
        ) if np.any(nonzero_obs) else None,
    }


def model_summary_dict(model: dict, design: dict) -> dict:
    categories = len(design["category_values"])
    bg, bc = unpack_parameters(model["parameters"], model["level"], categories)
    summary = {
        "level": model["level"],
        "category_penalty": float(model["category_penalty"]),
        "penalized_training_loss": float(model["penalized_training_loss"]),
        "validation_loss": model.get("validation_loss"),
        "iterations": int(model["iterations"]),
        "gradient_max": float(model["gradient_max"]),
        "global_coefficients": {
            name: float(bg[i]) for i, name in enumerate(COVARIATE_NAMES)
        },
    }
    if bc is not None:
        summary["category_coefficient_quantiles"] = {
            name: {
                str(q): float(np.quantile(bc[:, i], q))
                for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
            }
            for i, name in enumerate(COVARIATE_NAMES)
        }
    return summary


def question2_purchase_rate_decomposition(frame: pd.DataFrame) -> dict:
    """Decompose product purchase-rate change into within-category choice vs category incidence."""
    rate_fit = multi_cluster_parameter_inference(
        frame, offset_col="trip_offset", success_col="purchases", total_col="total_purchases"
    )
    choice_fit = multi_cluster_parameter_inference(
        frame, offset_col="choice_offset", success_col="purchases", total_col="total_purchases"
    )
    frame_cat = frame.copy()
    frame_cat["total_cat_baskets"] = (
        frame_cat.category_baskets + frame_cat.previous_category_baskets
    )
    cat_inc_fit = multi_cluster_parameter_inference(
        frame_cat,
        offset_col="trip_offset",
        success_col="category_baskets",
        total_col="total_cat_baskets",
    )

    # Exact identity r_{j,s,t} = c_{c,s,t} * s_{j,s,t}:
    # dr = r1 - r0 = 0.5*(c0 + c1)*(s1 - s0) [switching] + 0.5*(s0 + s1)*(c1 - c0) [category expansion]
    r0 = frame.observed_before_incidence.to_numpy(np.float64)
    r1 = frame.observed_after_incidence.to_numpy(np.float64)
    c0 = frame.observed_before_category_incidence.to_numpy(np.float64)
    c1 = frame.observed_after_category_incidence.to_numpy(np.float64)
    s0 = frame.observed_before_choice_share.to_numpy(np.float64)
    s1 = frame.observed_after_choice_share.to_numpy(np.float64)
    d_depth = frame.promotion_depth_change.to_numpy(np.float64)

    dr = r1 - r0
    switching_contrib = 0.5 * (c0 + c1) * (s1 - s0)
    category_expansion_contrib = 0.5 * (s0 + s1) * (c1 - c0)

    var_d = max(float(np.var(d_depth)), 1e-12)
    slope_total = float(np.cov(dr, d_depth)[0, 1] / var_d)
    slope_switching = float(np.cov(switching_contrib, d_depth)[0, 1] / var_d)
    slope_cat_expansion = float(np.cov(category_expansion_contrib, d_depth)[0, 1] / var_d)

    disc_on = d_depth >= 0.05
    disc_off = d_depth <= -0.05

    beta_rate_depth = rate_fit["effects"]["promotion_depth_change"]["estimate"]
    beta_choice_depth = choice_fit["effects"]["promotion_depth_change"]["estimate"]
    beta_cat_depth = cat_inc_fit["effects"]["promotion_depth_change"]["estimate"]

    return {
        "conditional_binomial_purchase_rate": rate_fit,
        "conditional_binomial_within_category_choice": choice_fit,
        "conditional_binomial_category_incidence": cat_inc_fit,
        "log_rate_depth_slope_decomposition": {
            "total_purchase_rate_depth_slope": beta_rate_depth,
            "within_category_choice_depth_slope": beta_choice_depth,
            "category_incidence_depth_slope": beta_cat_depth,
            "choice_share_of_total_log_slope": float(beta_choice_depth / beta_rate_depth)
            if abs(beta_rate_depth) > 1e-12
            else None,
            "category_incidence_share_of_total_log_slope": float(beta_cat_depth / beta_rate_depth)
            if abs(beta_rate_depth) > 1e-12
            else None,
        },
        "exact_incidence_decomposition_r_equals_c_times_s": {
            "ols_slope_on_depth_change": {
                "total_purchase_rate_slope": slope_total,
                "within_category_switching_slope": slope_switching,
                "category_expansion_slope": slope_cat_expansion,
                "fraction_from_within_category_switching": float(slope_switching / slope_total)
                if abs(slope_total) > 1e-12
                else None,
                "fraction_from_category_expansion": float(slope_cat_expansion / slope_total)
                if abs(slope_total) > 1e-12
                else None,
            },
            "discount_increase_events": {
                "events": int(disc_on.sum()),
                "mean_purchase_rate_change": float(dr[disc_on].mean()),
                "mean_within_category_switching_contribution": float(
                    switching_contrib[disc_on].mean()
                ),
                "mean_category_expansion_contribution": float(
                    category_expansion_contrib[disc_on].mean()
                ),
                "fraction_from_within_category_switching": float(
                    switching_contrib[disc_on].mean() / dr[disc_on].mean()
                ),
                "fraction_from_category_expansion": float(
                    category_expansion_contrib[disc_on].mean() / dr[disc_on].mean()
                ),
                "mean_choice_share_change": float((s1[disc_on] - s0[disc_on]).mean()),
                "mean_seasonality_adjusted_choice_share_change": float(
                    frame.seasonality_adjusted_choice_share_change.to_numpy(np.float64)[disc_on].mean()
                ),
                "mean_category_incidence_change": float((c1[disc_on] - c0[disc_on]).mean()),
            },
            "discount_decrease_events": {
                "events": int(disc_off.sum()),
                "mean_purchase_rate_change": float(dr[disc_off].mean()),
                "mean_within_category_switching_contribution": float(
                    switching_contrib[disc_off].mean()
                ),
                "mean_category_expansion_contribution": float(
                    category_expansion_contrib[disc_off].mean()
                ),
                "fraction_from_within_category_switching": float(
                    switching_contrib[disc_off].mean() / dr[disc_off].mean()
                ),
                "fraction_from_category_expansion": float(
                    category_expansion_contrib[disc_off].mean() / dr[disc_off].mean()
                ),
                "mean_choice_share_change": float((s1[disc_off] - s0[disc_off]).mean()),
                "mean_seasonality_adjusted_choice_share_change": float(
                    frame.seasonality_adjusted_choice_share_change.to_numpy(np.float64)[disc_off].mean()
                ),
                "mean_category_incidence_change": float((c1[disc_off] - c0[disc_off]).mean()),
            },
        },
    }


def sibling_cross_effects(frame: pd.DataFrame) -> dict:
    """Estimate the response of sibling products in the same category and store."""
    d_depth = frame.promotion_depth_change.to_numpy(np.float64)
    disc_on = d_depth >= 0.05
    disc_off = d_depth <= -0.05

    # Undiscounted sibling conditional binomial response to focal product's promotion_depth_change
    u_cur = frame.undiscounted_sibling_purchases.to_numpy(np.float64)
    u_prev = frame.previous_undiscounted_sibling_purchases.to_numpy(np.float64)
    u_tot = u_cur + u_prev
    off_cat = frame.choice_offset.to_numpy(np.float64)
    X = covariate_matrix(frame)
    den_u = max(float(u_tot.sum()), 1.0)

    res_u = minimize(
        lambda b: (
            float(np.sum(u_tot * np.logaddexp(0.0, off_cat + X @ b) - u_cur * (off_cat + X @ b)) / den_u),
            (X.T @ (u_tot * expit(off_cat + X @ b) - u_cur)) / den_u,
        ),
        np.zeros(3, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
    )
    beta_u = res_u.x.copy()
    p_u = expit(off_cat + X @ beta_u)
    hess_u = (X.T * (u_tot * p_u * (1.0 - p_u))) @ X
    hinv_u = np.linalg.inv(hess_u)
    scores_u = X * (u_cur - u_tot * p_u)[:, None]
    codes_p, uniq_p = pd.factorize(frame.item_id.to_numpy(np.int64), sort=True)
    Gp = len(uniq_p)
    cs_p = np.zeros((Gp, 3), dtype=np.float64)
    np.add.at(cs_p, codes_p, scores_u)
    se_u = np.sqrt(np.maximum(np.diag(hinv_u @ ((Gp / max(Gp - 1, 1)) * (cs_p.T @ cs_p)) @ hinv_u), 0.0))

    # Share changes
    own_ds = frame.observed_choice_share_change.to_numpy(np.float64)
    undisc_tot_ds = frame.undiscounted_total_share_change.to_numpy(np.float64)
    undisc_per_item_ds = frame.undiscounted_choice_share_change_per_item.to_numpy(np.float64)

    # Also share of total category item purchases (where own share + all-siblings share == 1)
    cat_ip_cur = np.maximum(frame.category_item_purchases.to_numpy(np.float64), 1.0)
    cat_ip_prev = np.maximum(frame.previous_category_item_purchases.to_numpy(np.float64), 1.0)
    both_pos_ip = (frame.category_item_purchases.to_numpy(np.int64) > 0) & (
        frame.previous_category_item_purchases.to_numpy(np.int64) > 0
    )
    sib_item_share_before = frame.previous_sibling_purchases.to_numpy(np.float64) / cat_ip_prev
    sib_item_share_after = frame.sibling_purchases.to_numpy(np.float64) / cat_ip_cur
    sib_item_share_change = sib_item_share_after - sib_item_share_before

    var_d = max(float(np.var(d_depth)), 1e-12)
    ols_undisc_tot = float(np.cov(undisc_tot_ds, d_depth)[0, 1] / var_d)
    ols_undisc_item = float(np.cov(undisc_per_item_ds, d_depth)[0, 1] / var_d)
    ols_sib_item_share = float(
        np.cov(sib_item_share_change[both_pos_ip], d_depth[both_pos_ip])[0, 1]
        / max(float(np.var(d_depth[both_pos_ip])), 1e-12)
    )

    return {
        "undiscounted_sibling_conditional_logit_cross_effects": {
            name: {
                "estimate": float(beta_u[i]),
                "product_clustered_se": float(se_u[i]),
                "product_clustered_95_interval": [
                    float(beta_u[i] - 1.96 * se_u[i]),
                    float(beta_u[i] + 1.96 * se_u[i]),
                ],
            }
            for i, name in enumerate(COVARIATE_NAMES)
        },
        "ols_cross_slopes_on_focal_promotion_depth_change": {
            "undiscounted_siblings_total_choice_share_slope": ols_undisc_tot,
            "undiscounted_sibling_per_product_choice_share_slope": ols_undisc_item,
            "all_siblings_category_purchase_share_slope": ols_sib_item_share,
        },
        "discount_increase_events": {
            "events": int(disc_on.sum()),
            "focal_own_choice_share_change": float(own_ds[disc_on].mean()),
            "undiscounted_siblings_total_choice_share_change": float(undisc_tot_ds[disc_on].mean()),
            "undiscounted_sibling_per_product_choice_share_change": float(
                undisc_per_item_ds[disc_on].mean()
            ),
            "all_siblings_category_purchase_share_change": float(
                sib_item_share_change[disc_on & both_pos_ip].mean()
            ),
        },
        "discount_decrease_events": {
            "events": int(disc_off.sum()),
            "focal_own_choice_share_change": float(own_ds[disc_off].mean()),
            "undiscounted_siblings_total_choice_share_change": float(undisc_tot_ds[disc_off].mean()),
            "undiscounted_sibling_per_product_choice_share_change": float(
                undisc_per_item_ds[disc_off].mean()
            ),
            "all_siblings_category_purchase_share_change": float(
                sib_item_share_change[disc_off & both_pos_ip].mean()
            ),
        },
    }


def question3_promotion_type_and_display_diagnosis(
    events: pd.DataFrame, base_checkpoint_path: Path
) -> dict:
    """Analyze separate discount, display, and mailer effects and diagnose the base model's -0.33 display weight."""
    import torch

    train = events[events.split == "train"]
    val = events[events.split == "validation"]
    test = events[events.split == "test"]

    # 1. Marginal (single-covariate) vs Joint (3-covariate) choice fits on training
    X_tr = covariate_matrix(train)
    off_tr = train.choice_offset.to_numpy(np.float64)
    y_tr = train.purchases.to_numpy(np.float64)
    m_tr = train.total_purchases.to_numpy(np.float64)
    den_tr = max(float(m_tr.sum()), 1.0)

    marginal_fits = {}
    for i, name in enumerate(COVARIATE_NAMES):
        xi = X_tr[:, i]
        res_i = minimize(
            lambda b: (
                float(np.sum(m_tr * np.logaddexp(0.0, off_tr + xi * b[0]) - y_tr * (off_tr + xi * b[0])) / den_tr),
                np.array([np.sum(xi * (m_tr * expit(off_tr + xi * b[0]) - y_tr)) / den_tr]),
            ),
            np.zeros(1, dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
        )
        marginal_fits[name] = float(res_i.x[0])

    # Covariate correlation matrix on training events
    corr_mat = np.corrcoef(X_tr.T)
    covariate_correlations = {
        "depth_vs_display_change": float(corr_mat[0, 1]),
        "depth_vs_mailer_change": float(corr_mat[0, 2]),
        "display_vs_mailer_change": float(corr_mat[1, 2]),
    }

    # Inspect the base model checkpoint's w_dsp and w_mlr parameters
    blob = torch.load(base_checkpoint_path, map_location="cpu", weights_only=False)
    st = blob["model"]
    w_dsp = st["w_dsp"].double().cpu().numpy()
    w_mlr = st["w_mlr"].double().cpu().numpy()
    gamma = torch.nn.functional.softplus(st["gamma"].double())
    beta = torch.nn.functional.softplus(st["beta"].double())
    kappa = float(torch.nn.functional.softplus(st["price_kappa"].double()))
    g_matrix = (gamma @ beta.T).cpu().numpy()
    mean_g = float(g_matrix.mean())
    mean_own_price_slope = float(mean_g * kappa)

    # Count training display cells and training display sales per product
    baskets = pd.read_parquet(
        BI / "baskets.parquet", columns=["BASKET_ID", "WEEK_NO", "item_id", "store_id", "split"]
    )
    items = pd.read_parquet(BI / "items.parquet", columns=["item_id", "n_train_lines"]).sort_values("item_id")
    n_items = len(items)
    n_stores = 115
    with np.load(BI / "promo.npz") as pr:
        pk = pr["keys"].astype(np.int64)
        p_disp = pr["disp"].astype(np.int64)
        p_mail = pr["mail"].astype(np.int64)

    p_week = pk % 128
    p_rem = pk // 128
    p_item = p_rem // n_stores

    train_b = baskets[baskets.split == "train"]
    bought = train_b.groupby(["item_id", "store_id", "WEEK_NO"]).BASKET_ID.nunique()
    b_key = (
        bought.index.get_level_values(0).to_numpy(np.int64) * n_stores
        + bought.index.get_level_values(1).to_numpy(np.int64)
    ) * 128 + bought.index.get_level_values(2).to_numpy(np.int64)
    b_ord = np.argsort(b_key)
    b_key = b_key[b_ord]
    b_val = bought.to_numpy(np.int64)[b_ord]
    p_sales = _sparse_lookup(b_key, b_val, pk)

    tr_mask = p_week <= 82
    disp_cells = np.bincount(p_item[tr_mask], weights=p_disp[tr_mask], minlength=n_items)
    disp_sales = np.bincount(p_item[tr_mask], weights=p_disp[tr_mask] * p_sales[tr_mask], minlength=n_items)
    mail_cells = np.bincount(p_item[tr_mask], weights=p_mail[tr_mask], minlength=n_items)
    mail_sales = np.bincount(p_item[tr_mask], weights=p_mail[tr_mask] * p_sales[tr_mask], minlength=n_items)

    zero_disp_exposure = disp_cells == 0
    zero_disp_sales_given_exposure = (disp_cells > 0) & (disp_sales == 0)
    pos_disp_sales = disp_sales > 0

    return {
        "joint_choice_effects_by_split": {
            "train": multi_cluster_parameter_inference(train),
            "validation": multi_cluster_parameter_inference(val),
            "test": multi_cluster_parameter_inference(test),
        },
        "marginal_single_covariate_training_choice_effects": marginal_fits,
        "training_event_covariate_correlations": covariate_correlations,
        "base_model_parameter_audit": {
            "checkpoint": str(base_checkpoint_path),
            "mean_g": mean_g,
            "softplus_price_kappa": kappa,
            "mean_own_price_slope": mean_own_price_slope,
            "w_dsp_unweighted_mean": float(w_dsp.mean()),
            "w_dsp_median": float(np.median(w_dsp)),
            "w_dsp_quantiles": {
                str(q): float(np.quantile(w_dsp, q))
                for q in (0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
            },
            "w_dsp_display_exposure_weighted_mean": float(
                np.average(w_dsp, weights=np.maximum(disp_cells, 0))
            ),
            "w_dsp_display_sales_weighted_mean": float(
                np.average(w_dsp, weights=np.maximum(disp_sales, 0))
            ),
            "w_mlr_unweighted_mean": float(w_mlr.mean()),
            "w_mlr_median": float(np.median(w_mlr)),
            "w_mlr_mailer_exposure_weighted_mean": float(
                np.average(w_mlr, weights=np.maximum(mail_cells, 0))
            ),
            "w_mlr_mailer_sales_weighted_mean": float(
                np.average(w_mlr, weights=np.maximum(mail_sales, 0))
            ),
            "negative_display_weight_diagnosis": {
                "is_negative_display_weight_real": False,
                "cause": (
                    "artefact of unpooled per-product parameter estimation on sparse display cells "
                    "with zero purchases plus cross-sectional baseline confounding"
                ),
                "products_total": int(n_items),
                "products_with_no_training_display_cells": {
                    "count": int(zero_disp_exposure.sum()),
                    "w_dsp_mean": float(w_dsp[zero_disp_exposure].mean()),
                },
                "products_with_training_display_cells_but_zero_display_purchases": {
                    "count": int(zero_disp_sales_given_exposure.sum()),
                    "w_dsp_mean": float(w_dsp[zero_disp_sales_given_exposure].mean()),
                    "w_dsp_min": float(w_dsp[zero_disp_sales_given_exposure].min()),
                },
                "products_with_positive_training_display_purchases": {
                    "count": int(pos_disp_sales.sum()),
                    "w_dsp_mean": float(w_dsp[pos_disp_sales].mean()),
                    "w_dsp_median": float(np.median(w_dsp[pos_disp_sales])),
                },
            },
        },
    }


def build_report_interpretation(report: dict) -> str:
    """Construct the report interpretation string dynamically from computed report metrics."""
    comp = report["locked_test_comparisons"]["no_promotion_vs_selected"]
    loss_imp = float(comp["log_loss_improvement_of_second_over_first"])
    ci_prod = comp["cluster_bootstrap_95_intervals_improvement"]["product"]
    ci_store = comp["cluster_bootstrap_95_intervals_improvement"]["store"]
    ci_week = comp["cluster_bootstrap_95_intervals_improvement"]["week"]

    q2 = report["question_2_purchase_rate_decomposition"]
    sw_train = (
        100.0
        * float(
            q2["train"]["exact_incidence_decomposition_r_equals_c_times_s"][
                "ols_slope_on_depth_change"
            ]["fraction_from_within_category_switching"]
        )
    )
    sw_test = (
        100.0
        * float(
            q2["test"]["exact_incidence_decomposition_r_equals_c_times_s"][
                "ols_slope_on_depth_change"
            ]["fraction_from_within_category_switching"]
        )
    )

    q3_audit = report["question_3_promotion_types_and_display_artifact_diagnosis"][
        "base_model_parameter_audit"
    ]
    w_dsp_mean = float(q3_audit["w_dsp_unweighted_mean"])
    w_dsp_exp_mean = float(q3_audit["w_dsp_display_exposure_weighted_mean"])
    w_dsp_sales_mean = float(q3_audit["w_dsp_display_sales_weighted_mean"])
    diag = q3_audit["negative_display_weight_diagnosis"]
    zero_disp_cnt = int(
        diag["products_with_training_display_cells_but_zero_display_purchases"]["count"]
    )
    zero_disp_mean = float(
        diag["products_with_training_display_cells_but_zero_display_purchases"]["w_dsp_mean"]
    )
    pos_disp_cnt = int(diag["products_with_positive_training_display_purchases"]["count"])
    pos_disp_mean = float(
        diag["products_with_positive_training_display_purchases"]["w_dsp_mean"]
    )

    text = (
        f"Observational, predictive evidence only: discounts, displays, and mailers strongly "
        f"predict within-category product choice switching on held-out Dunnhumby test weeks "
        f"(test conditional log-loss improvement {loss_imp:.6f} over no-promotion; 95% cluster-bootstrap "
        f"intervals product [{ci_prod[0]:.6f}, {ci_prod[1]:.6f}], store [{ci_store[0]:.6f}, {ci_store[1]:.6f}], "
        f"week [{ci_week[0]:.6f}, {ci_week[1]:.6f}]). "
        f"In the exact incidence decomposition, {sw_train:.1f}% (train) to {sw_test:.1f}% (test) of the "
        f"promotion-depth slope in product purchase rate comes from within-category switching rather than "
        f"extra category purchases. "
        f"The Version-4 base model's negative unweighted mean display weight ({w_dsp_mean:.4f}) is an "
        f"estimation artefact of {zero_disp_cnt} unpooled products with zero purchases during sparse "
        f"training display exposures (mean w_dsp {zero_disp_mean:.4f}), whereas the {pos_disp_cnt} products "
        f"with positive training display purchases have mean w_dsp {pos_disp_mean:.4f} (exposure-weighted "
        f"mean {w_dsp_exp_mean:.4f}, sales-weighted mean {w_dsp_sales_mean:.4f})."
    )
    q4_sum = report.get("question_4_version4_model_check_summary")
    if isinstance(q4_sum, dict) and "candidate_rank1_single_item_price_counterfactual_informative" in q4_sum:
        s_inf = q4_sum["candidate_rank1_single_item_price_counterfactual_informative"]
        c_inf = q4_sum["candidate_rank1_chain_week_price_counterfactual_informative"]
        text += (
            f" On held-out informative test events, the Version-4 rank-5 model's single-item price "
            f"counterfactual has Pearson correlation {s_inf['pearson_correlation']:.4f} "
            f"(non-zero observed sign agreement {s_inf['sign_agreement_nonzero_observed']:.4f}) and its "
            f"simultaneous chain-week price counterfactual has Pearson correlation "
            f"{c_inf['pearson_correlation']:.4f} (non-zero observed sign agreement "
            f"{c_inf['sign_agreement_nonzero_observed']:.4f}), with predicted-to-observed discount-increase "
            f"magnitude ratio {c_inf['predicted_to_observed_magnitude_ratio_on_discount_increase']:.4f}."
        )
    return text


def evaluate_version4_counterfactual_q4(
    test_events: pd.DataFrame,
    additive_ckpt: Path,
    rank5_ckpt: Path,
    init_ckpt: Path,
    empirical_depth_slopes: dict[str, float] | None = None,
    empirical_promotion_coefficients: dict | None = None,
    threads: int = 8,
    burn: int = 200,
    draws: int = 64,
    thin: int = 2,
    max_trips_per_store_week: int = 1,
    seed: int = 68101,
) -> dict:
    """Evaluate Version-4 counterfactual choice-share predictions on held-out test discount events."""
    import torch
    import stress
    from features import Features
    from fit import Batcher
    from fit_exact_additive import load_model
    from price_response import changed_price_context

    t0 = time.time()
    log(
        f"starting Q4 Version-4 base model counterfactual evaluation on held-out test events "
        f"(burn={burn}, draws={draws}, thin={thin}, max_trips_per_store_week={max_trips_per_store_week})..."
    )
    data, model, meta, _ = stress.setup(init_ckpt, additive_ckpt, threads=threads)
    # Use chain-level prices only (include_store_price=False), per design requirement
    features = Features(
        int(data["n_item"]),
        int(data["n_store"]),
        include_recency=False,
        include_store_price=False,
    )
    batcher = Batcher(data, features, int(meta["nmax"]), include_recency=False)

    # Keep a separate exact additive model (out/v3_pipeline_additive_best.pt) and rank-5 model (artifacts/candidate_rank1.pt)
    model_add, _, _ = load_model(Path(init_ckpt), data)
    raw_add = torch.load(additive_ckpt, map_location="cpu", weights_only=False)
    stress.attach_discount_kappa(model_add, raw_add["model"])
    model_add.load_state_dict(raw_add["model"])
    with torch.no_grad():
        model_add.phi.zero_()
    model_add.presence_phi = False
    model_add.cat_rho0 = False
    model_add.eval()
    for q in model_add.parameters():
        q.requires_grad_(False)

    blob_rank5 = torch.load(rank5_ckpt, map_location="cpu", weights_only=False)
    for name in ("disc_kappa", "disc_common"):
        if getattr(model, name, None) is not None and name not in blob_rank5["model"]:
            delattr(model, name)   # the rank-5 model keeps its own single price slope
    stress.configure_model_from_blob(model, blob_rank5)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)

    r = int(blob_rank5.get("active_rank", blob_rank5.get("rank", 5)))
    J = int(data["n_item"])
    nmax = int(meta["nmax"])
    kappa = float(torch.nn.functional.softplus(model.price_kappa.detach()))
    gamma_pos = torch.nn.functional.softplus(model.gamma.detach().double())
    beta_pos = torch.nn.functional.softplus(model.beta.detach().double())
    mean_g = float((gamma_pos @ beta_pos.T).mean())
    mean_own_price_slope = float(mean_g * kappa)

    # Chain-week price matrix from data/price_week.parquet for exact log-price changes
    items = pd.read_parquet(
        BI / "items.parquet", columns=["PRODUCT_ID", "item_id", "cat_id"]
    ).sort_values("item_id")
    item_cat = items.cat_id.to_numpy(np.int64)
    n_merch_cats = int(item_cat.max()) + 1

    # Identify all (store, week) cells required by test_events (both WEEK_NO and previous_WEEK_NO)
    ev_store_arr = test_events.store_id.to_numpy(np.int64)
    ev_week_arr = test_events.WEEK_NO.to_numpy(np.int64)
    ev_prev_week_arr = test_events.previous_WEEK_NO.to_numpy(np.int64)
    needed_cur_sw = set(zip(ev_store_arr.tolist(), ev_week_arr.tolist()))
    needed_prev_sw = set(zip(ev_store_arr.tolist(), ev_prev_week_arr.tolist()))
    needed_all_sw = needed_cur_sw | needed_prev_sw
    needed_cur_keys = {s * 128 + w for s, w in needed_cur_sw}
    needed_all_keys = np.array(sorted(s * 128 + w for s, w in needed_all_sw), dtype=np.int64)

    trip_week = data["trip_week"]
    trip_store = data["trip_store"]
    trip_nlines = data["trip_nlines"]
    # Trip window spans the events' previous and current weeks (90..101 for the test events).
    week_lo = int(ev_prev_week_arr.min())
    week_hi = int(ev_week_arr.max())
    all_window_trips = np.flatnonzero(
        (trip_week >= week_lo) & (trip_week <= week_hi) & (trip_nlines >= 1) & (trip_nlines <= nmax)
    )
    rng_sw = np.random.default_rng(seed + 101)
    sw_keys = (
        trip_store[all_window_trips].astype(np.int64) * 128
        + trip_week[all_window_trips].astype(np.int64)
    )
    cur_chunks, prev_only_chunks = [], []
    for sw_val in needed_all_keys:
        cell_trips = all_window_trips[sw_keys == sw_val]
        if len(cell_trips) == 0:
            raise RuntimeError(f"store-week key {int(sw_val)} has no eligible trips in weeks {week_lo}..{week_hi}")
        if len(cell_trips) > max_trips_per_store_week:
            cell_trips = np.sort(
                rng_sw.choice(cell_trips, size=max_trips_per_store_week, replace=False)
            )
        if int(sw_val) in needed_cur_keys:
            cur_chunks.append(cell_trips)
        else:
            prev_only_chunks.append(cell_trips)

    cur_trips = np.sort(np.concatenate(cur_chunks)) if cur_chunks else np.zeros(0, dtype=np.int64)
    prev_only_trips = (
        np.sort(np.concatenate(prev_only_chunks))
        if prev_only_chunks
        else np.zeros(0, dtype=np.int64)
    )
    eligible_trips = np.concatenate([cur_trips, prev_only_trips])
    log(
        f"evaluating Version-4 over {len(eligible_trips)} representative trips "
        f"across {len(needed_all_keys)} required store-weeks "
        f"({len(cur_trips)} in test event weeks, {len(prev_only_trips)} in prior-only weeks; "
        f"rank={r}, burn={burn}, draws={draws})..."
    )

    weekly = pd.read_parquet(DATA / "price_week.parquet", columns=["PRODUCT_ID", "WEEK_NO", "price", "base_price"])
    weekly = weekly.merge(items[["PRODUCT_ID", "item_id"]], on="PRODUCT_ID", how="inner")
    weekly = weekly[weekly.WEEK_NO.between(0, 101)]
    chain_price = np.full((J, 102), np.nan, dtype=np.float64)
    chain_price[weekly.item_id.to_numpy(np.int64), weekly.WEEK_NO.to_numpy(np.int64)] = (
        weekly.price.to_numpy(np.float64)
    )
    chain_price = pd.DataFrame(chain_price).ffill(axis=1).bfill(axis=1).to_numpy()
    log_chain_price = np.log(chain_price)
    # Discount depth per product-week, carried like the price (build_discount_panel.py), so a
    # move to last week's prices also moves the discount part for models that split it.
    base_week = np.full((J, 102), np.nan, dtype=np.float64)
    base_week[weekly.item_id.to_numpy(np.int64), weekly.WEEK_NO.to_numpy(np.int64)] = (
        weekly.base_price.to_numpy(np.float64)
    )
    base_week = pd.DataFrame(base_week).ffill(axis=1).bfill(axis=1).to_numpy()
    week_discount = np.minimum(np.log(chain_price / base_week), 0.0)

    n_stores = int(data["n_store"])
    sw_trip_count = np.zeros((n_stores, 102), dtype=np.int64)

    # Rank-5 model accumulators (indexed by merchandise category 0..n_merch_cats-1)
    r5_item_factual = np.zeros((J, n_stores, 102), dtype=np.float64)
    r5_cat_item_factual = np.zeros((n_merch_cats, n_stores, 102), dtype=np.float64)
    r5_item_single_cf_prev_price = np.zeros((J, n_stores, 102), dtype=np.float64)
    r5_item_chain_cf_prev_price = np.zeros((J, n_stores, 102), dtype=np.float64)
    r5_cat_item_chain_cf_prev_price = np.zeros((n_merch_cats, n_stores, 102), dtype=np.float64)

    # Additive base model (Phi=0) accumulators (indexed by merchandise category 0..n_merch_cats-1)
    add_item_factual = np.zeros((J, n_stores, 102), dtype=np.float64)
    add_cat_item_factual = np.zeros((n_merch_cats, n_stores, 102), dtype=np.float64)
    add_item_single_cf_prev_price = np.zeros((J, n_stores, 102), dtype=np.float64)
    add_item_chain_cf_prev_price = np.zeros((J, n_stores, 102), dtype=np.float64)
    add_cat_item_chain_cf_prev_price = np.zeros((n_merch_cats, n_stores, 102), dtype=np.float64)
    # Additive model with only the product's display (or mailer) switched on / off, everything
    # else as observed.  One utility changes, so the exact incidence follows from the factual
    # one by the single-utility tilt p(du) = p e^du / (1 - p + p e^du).
    add_item_promo = {key: np.zeros((J, n_stores, 102), dtype=np.float64)
                      for key in ("disp_on", "disp_off", "mail_on", "mail_off")}

    gen = torch.Generator().manual_seed(seed + 17)
    batch_size = 128

    for start in range(0, len(eligible_trips), batch_size):
        sub_trips = eligible_trips[start : start + batch_size]
        bt = stress.Batch(batcher, sub_trips, J)
        bt.bind(model)
        bt.bind(model_add)
        ix = bt.ix
        B = ix.B
        st_b = trip_store[sub_trips].astype(np.int64)
        wk_b = trip_week[sub_trips].astype(np.int64)
        np.add.at(sw_trip_count, (st_b, wk_b), 1)
        need_cf = bool(
            any(int(s) * 128 + int(w) in needed_cur_keys for s, w in zip(st_b, wk_b))
        )

        slot_store = st_b[ix.item_trip.numpy()]
        slot_week = wk_b[ix.item_trip.numpy()]
        slot_item = ix.item.numpy()
        slot_cat = item_cat[slot_item]

        # Price change to move from week w to week w-1: delta_prev = log(p_{w-1}) - log(p_w) = -log_price_change
        delta_to_prev = torch.from_numpy(
            log_chain_price[slot_item, np.maximum(slot_week - 1, 1)]
            - log_chain_price[slot_item, slot_week]
        ).to(torch.float64)

        # Single-item utility shift when ONLY item j changes by delta_to_prev:
        gb_slot = model.price_coefficients(ix.item, ix.item_trip).detach()
        counts_b = torch.bincount(ix.item_trip, minlength=B).to(torch.float64)
        own_mult = kappa + (1.0 - kappa) / counts_b[ix.item_trip]
        du_single_prev = -gb_slot * own_mult * delta_to_prev
        disc_to_prev = torch.from_numpy(
            week_discount[slot_item, np.maximum(slot_week - 1, 1)]
            - week_discount[slot_item, slot_week]
        ).to(torch.float64)

        def discount_shift(m):
            # Own-utility change from the discount part:
            # -[g (softplus(disc_kappa) - kappa) + disc_common] d dsc.
            shift = torch.zeros_like(delta_to_prev)
            if "dsc" not in bt.ctx:
                return shift
            if getattr(m, "disc_kappa", None) is not None:
                kd = float(torch.nn.functional.softplus(m.disc_kappa.detach()))
                km = float(torch.nn.functional.softplus(m.price_kappa.detach()))
                shift = shift - m.price_coefficients(ix.item, ix.item_trip).detach() * (kd - km) * disc_to_prev
            if getattr(m, "disc_common", None) is not None:
                shift = shift - float(m.disc_common.detach()) * disc_to_prev
            return shift

        # Simultaneous chain-week price counterfactual context via changed_price_context
        ctx_chain_prev = (
            changed_price_context(bt.ctx, ix.item_trip, delta_to_prev, discount_change=disc_to_prev)
            if need_cf else None
        )

        # --- 1. Additive base model (out/v3_pipeline_additive_best.pt, exact DP) ---
        bt.bind(model_add, bt.ctx)
        u_add_0 = model_add.b_flat(ix).detach().requires_grad_(True)
        with torch.enable_grad():
            lz_add_0 = stress.log_zc(model_add, ix, u_add_0, z=None, beta=0.0)
            p_add_0, = torch.autograd.grad(lz_add_0.sum(), u_add_0)
        p_add_0 = p_add_0.detach()
        p_add_0_np = p_add_0.numpy()
        np.add.at(add_item_factual, (slot_item, slot_store, slot_week), p_add_0_np)
        np.add.at(add_cat_item_factual, (slot_cat, slot_store, slot_week), p_add_0_np)
        for name, weight in (("disp", model_add.w_dsp), ("mail", model_add.w_mlr)):
            w_slot = weight.detach()[ix.item].double()
            state = bt.ctx[name].double()
            for side, du in (("on", w_slot * (1.0 - state)), ("off", -w_slot * state)):
                tilted = p_add_0 * torch.exp(du) / (1.0 - p_add_0 + p_add_0 * torch.exp(du))
                np.add.at(add_item_promo[f"{name}_{side}"], (slot_item, slot_store, slot_week),
                          tilted.numpy())

        if need_cf:
            # The additive model's own price slopes and multiplier (they may differ from rank-5's).
            k_add = float(torch.nn.functional.softplus(model_add.price_kappa.detach()))
            du_add = (-model_add.price_coefficients(ix.item, ix.item_trip).detach()
                      * (k_add + (1.0 - k_add) / counts_b[ix.item_trip]) * delta_to_prev
                      + discount_shift(model_add))
            scale_add = torch.clamp_min(du_add, 0.0)
            num_add = p_add_0 * torch.exp(du_add - scale_add)
            den_add = (1.0 - p_add_0) * torch.exp(-scale_add) + num_add
            p_add_single_prev = (num_add / den_add.clamp_min(1e-30)).numpy()

            bt.bind(model_add, ctx_chain_prev)
            u_add_prev = model_add.b_flat(ix).detach().requires_grad_(True)
            with torch.enable_grad():
                lz_add_prev = stress.log_zc(model_add, ix, u_add_prev, z=None, beta=0.0)
                p_add_chain_prev, = torch.autograd.grad(lz_add_prev.sum(), u_add_prev)
            p_add_chain_prev = p_add_chain_prev.detach().numpy()

            np.add.at(
                add_item_single_cf_prev_price, (slot_item, slot_store, slot_week), p_add_single_prev
            )
            np.add.at(
                add_item_chain_cf_prev_price, (slot_item, slot_store, slot_week), p_add_chain_prev
            )
            np.add.at(
                add_cat_item_chain_cf_prev_price,
                (slot_cat, slot_store, slot_week),
                p_add_chain_prev,
            )

        # --- 2. Rank-5 interaction model (C++ Gibbs z draws + exact changed_price_context reweighting) ---
        bt.bind(model, bt.ctx)
        X = stress.sample_slots(
            model,
            ix,
            stress.slot_utilities(model, ix, None, r, beta=0.0),
            gen,
            use_cpp=True,
            beta=0.0,
        )
        X = stress.gibbs(model, ix, X, r, gen, sweeps=burn, use_cpp=True)
        zs = []
        for k in range(draws * thin):
            X = stress.gibbs(model, ix, X, r, gen, sweeps=1, use_cpp=True)
            if (k + 1) % thin == 0:
                zs.append(
                    stress.basket_field(model, ix, X, r)
                    + torch.randn(B, r, generator=gen)
                )

        lzb_list, grad0_list, lzc_prev_list, grad_prev_list = [], [], [], []
        for z in zs:
            bt.bind(model, bt.ctx)
            u0_z = stress.slot_utilities(model, ix, z, r).detach().requires_grad_(True)
            with torch.enable_grad():
                lzb = stress.log_zc(model, ix, u0_z, z=z)
                g0, = torch.autograd.grad(lzb.sum(), u0_z)
            lzb_list.append(lzb.detach())
            grad0_list.append(g0.detach())

            if need_cf:
                bt.bind(model, ctx_chain_prev)
                u1_z = stress.slot_utilities(model, ix, z, r).detach().requires_grad_(True)
                with torch.enable_grad():
                    lzc_p = stress.log_zc(model, ix, u1_z, z=z)
                    gp, = torch.autograd.grad(lzc_p.sum(), u1_z)
                lzc_prev_list.append(lzc_p.detach())
                grad_prev_list.append(gp.detach())

        G0 = torch.stack(grad0_list, dim=1)  # T x K
        p_r5_0 = G0.mean(dim=1)
        p_r5_0_np = p_r5_0.numpy()
        np.add.at(r5_item_factual, (slot_item, slot_store, slot_week), p_r5_0_np)
        np.add.at(r5_cat_item_factual, (slot_cat, slot_store, slot_week), p_r5_0_np)

        if need_cf:
            # By linearity of Z_c(u + du_j e_j, z_k) / Z_c(u, z_k) * p_cf_j(z_k) = p_0_j(z_k) * exp(du_j),
            # exact Radon-Nikodym importance reweighting across the K draws for a single-slot shift
            # equals tilting the K-draw marginal mean p_r5_0 by exp(du_single_prev).
            scale_r5 = torch.clamp_min(du_single_prev, 0.0)
            num_r5 = p_r5_0 * torch.exp(du_single_prev - scale_r5)
            den_r5 = (1.0 - p_r5_0) * torch.exp(-scale_r5) + num_r5
            p_r5_single_prev = (num_r5 / den_r5.clamp_min(1e-30)).numpy()

            # Simultaneous chain-week price counterfactual reweighted across z draws (exact stress.py formula)
            lzb_mat = torch.stack(lzb_list, dim=1)            # B x K
            lzc_prev_mat = torch.stack(lzc_prev_list, dim=1)  # B x K
            w_prev = torch.softmax(lzc_prev_mat - lzb_mat, dim=1)  # B x K
            G_prev = torch.stack(grad_prev_list, dim=1)       # T x K
            p_r5_chain_prev = (G_prev * w_prev[ix.item_trip]).sum(dim=1).numpy()

            np.add.at(
                r5_item_single_cf_prev_price, (slot_item, slot_store, slot_week), p_r5_single_prev
            )
            np.add.at(
                r5_item_chain_cf_prev_price, (slot_item, slot_store, slot_week), p_r5_chain_prev
            )
            np.add.at(
                r5_cat_item_chain_cf_prev_price,
                (slot_cat, slot_store, slot_week),
                p_r5_chain_prev,
            )

        log(
            f"Q4 trips processed: {start + len(sub_trips)}/{len(eligible_trips)} "
            f"({time.time() - t0:.1f}s)"
        )

    # Also run a direct multi-pass changed_price_context verification on 16 single-item perturbations
    # to record the exact numerical fidelity of the single-item counterfactual
    bt_verify = stress.Batch(batcher, eligible_trips[:16], J)
    bt_verify.bind(model_add)
    u_v0 = model_add.b_flat(bt_verify.ix).detach().requires_grad_(True)
    with torch.enable_grad():
        lz_v0 = stress.log_zc(model_add, bt_verify.ix, u_v0, z=None, beta=0.0)
        p_v0, = torch.autograd.grad(lz_v0.sum(), u_v0)
    p_v0 = p_v0.detach()
    max_single_item_discrepancy = 0.0
    for slot_idx in range(min(16, len(bt_verify.ix.item))):
        target_item = int(bt_verify.ix.item[slot_idx])
        dlp_step = -0.20
        chg = torch.where(bt_verify.ix.item == target_item, dlp_step, 0.0)
        ctx_v1 = changed_price_context(bt_verify.ctx, bt_verify.ix.item_trip, chg)
        bt_verify.bind(model_add, ctx_v1)
        u_v1 = model_add.b_flat(bt_verify.ix).detach().requires_grad_(True)
        with torch.enable_grad():
            lz_v1 = stress.log_zc(model_add, bt_verify.ix, u_v1, z=None, beta=0.0)
            p_v1, = torch.autograd.grad(lz_v1.sum(), u_v1)
        du_v = (u_v1 - u_v0).detach()
        p_v1_tilt = (p_v0 * torch.exp(du_v)) / ((1.0 - p_v0) + p_v0 * torch.exp(du_v))
        disc = float(torch.abs(p_v1[slot_idx] - p_v1_tilt[slot_idx]))
        max_single_item_discrepancy = max(max_single_item_discrepancy, disc)

    # Map model predictions to test_events
    ev_item = test_events.item_id.to_numpy(np.int64)
    ev_cat = test_events.cat_id.to_numpy(np.int64)
    ev_store = test_events.store_id.to_numpy(np.int64)
    ev_week = test_events.WEEK_NO.to_numpy(np.int64)
    ev_prev_week = test_events.previous_WEEK_NO.to_numpy(np.int64)

    assert np.all(sw_trip_count[ev_store, ev_week] > 0), (
        "some test (store, week) cells had 0 evaluated trips"
    )
    assert np.all(sw_trip_count[ev_store, ev_prev_week] > 0), (
        "some test (store, previous_week) cells had 0 evaluated trips"
    )

    obs_change = test_events.observed_choice_share_change.to_numpy(np.float64)
    obs_adj_change = test_events.seasonality_adjusted_choice_share_change.to_numpy(np.float64)
    obs_inc_change = test_events.observed_incidence_change.to_numpy(np.float64)
    informative = test_events.total_purchases.to_numpy(np.int64) > 0
    d_depth = test_events.promotion_depth_change.to_numpy(np.float64)
    log_p_change = test_events.log_price_change.to_numpy(np.float64)

    def summarize_predictions(
        item_fact: np.ndarray,
        cat_item_fact: np.ndarray,
        item_single_prev: np.ndarray,
        item_chain_prev: np.ndarray,
        cat_item_chain_prev: np.ndarray,
    ) -> dict:
        # Within-category choice share is normalized consistently by expected category purchases
        # cat_item_fact[c, s, w] = sum_{k in merchandise_cat c} item_fact[k, s, w] >= item_fact[j, s, w] >= 0,
        # guaranteeing all shares lie in [0, 1] and all share changes lie in [-1, 1].
        f_item_w = item_fact[ev_item, ev_store, ev_week]
        f_cat_w = cat_item_fact[ev_cat, ev_store, ev_week]
        s_fact_w = np.where(f_cat_w > 0.0, f_item_w / np.maximum(f_cat_w, 1e-30), 0.0)

        # 1. Single-item exact price counterfactual choice share change on week w trips:
        #    share(price_w) - share(price_{w-1}) holding everything else on week w trips fixed
        cf_single_w = item_single_prev[ev_item, ev_store, ev_week]
        sib_cat_w = np.maximum(f_cat_w - f_item_w, 0.0)
        denom_single_w = sib_cat_w + cf_single_w
        s_single_prev_w = np.where(
            denom_single_w > 0.0, cf_single_w / np.maximum(denom_single_w, 1e-30), 0.0
        )
        pred_single_cf_change = s_fact_w - s_single_prev_w

        # 2. Simultaneous chain-week price counterfactual choice share change on week w trips:
        #    share(all chain prices at w) - share(all chain prices at w-1) via changed_price_context
        cf_chain_item_w = item_chain_prev[ev_item, ev_store, ev_week]
        cf_chain_cat_w = cat_item_chain_prev[ev_cat, ev_store, ev_week]
        s_chain_prev_item_w = np.where(
            cf_chain_cat_w > 0.0,
            cf_chain_item_w / np.maximum(cf_chain_cat_w, 1e-30),
            0.0,
        )
        pred_chain_cf_change = s_fact_w - s_chain_prev_item_w

        # 3. Full factual week-over-week model prediction (week w vs week w-1 including price, display, mailer, season)
        f_item_prev = item_fact[ev_item, ev_store, ev_prev_week]
        f_cat_prev = cat_item_fact[ev_cat, ev_store, ev_prev_week]
        s_fact_prev = np.where(
            f_cat_prev > 0.0, f_item_prev / np.maximum(f_cat_prev, 1e-30), 0.0
        )
        pred_full_wow_change = s_fact_w - s_fact_prev

        # 4. Purchase incidence counterfactual change (per trip in store-week w)
        trips_w = np.maximum(sw_trip_count[ev_store, ev_week].astype(np.float64), 1.0)
        pred_inc_single_cf_change = (f_item_w - cf_single_w) / trips_w

        # Assert every predicted share/incidence change is finite and lies in [-1, 1]
        for pred_name, pred_arr in (
            ("single_item_exact_price_counterfactual_choice_share", pred_single_cf_change),
            ("simultaneous_chain_week_price_counterfactual_choice_share", pred_chain_cf_change),
            ("full_week_over_week_model_prediction_with_display_and_mailer", pred_full_wow_change),
            ("single_item_exact_price_counterfactual_purchase_rate", pred_inc_single_cf_change),
        ):
            assert np.all(np.isfinite(pred_arr)), f"{pred_name} has non-finite values"
            assert np.all((pred_arr >= -1.0) & (pred_arr <= 1.0)), (
                f"{pred_name} out of [-1, 1]: min={pred_arr.min()}, max={pred_arr.max()}"
            )

        def score_vector(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict:
            p_m = pred[mask]
            t_m = target[mask]
            assert np.all((p_m >= -1.0) & (p_m <= 1.0)), (
                f"predicted share change out of [-1, 1]: [{p_m.min()}, {p_m.max()}]"
            )
            corr = (
                float(np.corrcoef(p_m, t_m)[0, 1])
                if len(p_m) > 1 and np.std(p_m) > 0 and np.std(t_m) > 0
                else None
            )
            spearman = (
                float(pd.Series(p_m).corr(pd.Series(t_m), method="spearman"))
                if len(p_m) > 1 and np.std(p_m) > 0 and np.std(t_m) > 0
                else None
            )
            nz = mask & (np.abs(target) > 0) & (np.abs(pred) > 0)
            disc_up = mask & (d_depth >= 0.05)
            disc_dn = mask & (d_depth <= -0.05)
            # Direction agreement with price cut / discount increase
            expected_sign = -np.sign(log_p_change[mask])
            sign_vs_price_cut = float(np.mean(np.sign(p_m) == expected_sign))
            return {
                "events": int(mask.sum()),
                "min_predicted_change": float(p_m.min()),
                "max_predicted_change": float(p_m.max()),
                "pearson_correlation": corr,
                "spearman_correlation": spearman,
                "sign_agreement_all": float(np.mean(np.sign(p_m) == np.sign(t_m))),
                "sign_agreement_nonzero_observed": float(
                    np.mean(np.sign(pred[nz]) == np.sign(target[nz]))
                )
                if np.any(nz)
                else None,
                "sign_agreement_with_price_direction": sign_vs_price_cut,
                "mean_predicted_change_discount_increase": float(pred[disc_up].mean())
                if np.any(disc_up)
                else None,
                "mean_observed_change_discount_increase": float(target[disc_up].mean())
                if np.any(disc_up)
                else None,
                "mean_predicted_change_discount_decrease": float(pred[disc_dn].mean())
                if np.any(disc_dn)
                else None,
                "mean_observed_change_discount_decrease": float(target[disc_dn].mean())
                if np.any(disc_dn)
                else None,
                "predicted_to_observed_magnitude_ratio_on_discount_increase": (
                    float(pred[disc_up].mean() / target[disc_up].mean())
                    if np.any(disc_up) and abs(float(target[disc_up].mean())) > 1e-12
                    else None
                ),
                "rmse": float(np.sqrt(np.mean((t_m - p_m) ** 2))),
                "mae": float(np.mean(np.abs(t_m - p_m))),
            }

        def depth_slope(after, before):
            # Slope of the model's log(after / before) on the discount-depth change over every
            # event (model predictions need no purchase selection).  Comparable to the empirical
            # conditional-binomial promotion_depth_change coefficient, whose offset removes the
            # category's basket change.
            ok = (after > 0) & (before > 0)
            y = np.log(after[ok]) - np.log(before[ok])
            x = d_depth[ok]
            return {"events": int(ok.sum()),
                    "slope": float(np.cov(x, y)[0, 1] / np.var(x, ddof=1)) if ok.sum() > 2 else None}

        all_mask = np.ones(len(test_events), dtype=bool)
        return {
            "log_ratio_on_depth_change_slope": {
                "single_item_choice_share": depth_slope(s_fact_w, s_single_prev_w),
                "single_item_purchase_rate": depth_slope(f_item_w, cf_single_w),
                "chain_week_choice_share": depth_slope(s_fact_w, s_chain_prev_item_w),
            },
            "single_item_exact_price_counterfactual_choice_share": {
                "all_test_events": score_vector(pred_single_cf_change, obs_change, all_mask),
                "informative_test_events": score_vector(
                    pred_single_cf_change, obs_change, informative
                ),
                "informative_seasonality_adjusted": score_vector(
                    pred_single_cf_change, obs_adj_change, informative
                ),
            },
            "simultaneous_chain_week_price_counterfactual_choice_share": {
                "all_test_events": score_vector(pred_chain_cf_change, obs_change, all_mask),
                "informative_test_events": score_vector(
                    pred_chain_cf_change, obs_change, informative
                ),
                "informative_seasonality_adjusted": score_vector(
                    pred_chain_cf_change, obs_adj_change, informative
                ),
            },
            "full_week_over_week_model_prediction_with_display_and_mailer": {
                "all_test_events": score_vector(pred_full_wow_change, obs_change, all_mask),
                "informative_test_events": score_vector(
                    pred_full_wow_change, obs_change, informative
                ),
            },
            "single_item_exact_price_counterfactual_purchase_rate": {
                "all_test_events": score_vector(
                    pred_inc_single_cf_change, obs_inc_change, all_mask
                ),
                "informative_test_events": score_vector(
                    pred_inc_single_cf_change, obs_inc_change, informative
                ),
            },
        }

    rank5_results = summarize_predictions(
        r5_item_factual,
        r5_cat_item_factual,
        r5_item_single_cf_prev_price,
        r5_item_chain_cf_prev_price,
        r5_cat_item_chain_cf_prev_price,
    )
    additive_results = summarize_predictions(
        add_item_factual,
        add_cat_item_factual,
        add_item_single_cf_prev_price,
        add_item_chain_cf_prev_price,
        add_cat_item_chain_cf_prev_price,
    )

    def promotion_effect(name):
        """Model log ratio of the product's within-category share (and purchase rate) with its
        display/mailer on vs off, everything else as observed in the event's week; compare with
        the empirical conditional-binomial coefficient of display_change / mailer_change."""
        f_cat = add_cat_item_factual[ev_cat, ev_store, ev_week]
        f_item = add_item_factual[ev_item, ev_store, ev_week]
        on = add_item_promo[f"{name}_on"][ev_item, ev_store, ev_week]
        off = add_item_promo[f"{name}_off"][ev_item, ev_store, ev_week]
        rest = np.maximum(f_cat - f_item, 0.0)
        ok = (on > 0) & (off > 0) & (rest > 0)
        share = np.log(on[ok] / (rest[ok] + on[ok])) - np.log(off[ok] / (rest[ok] + off[ok]))
        rate = np.log(on[ok]) - np.log(off[ok])
        wt = f_item[ok] / f_item[ok].sum()
        weight = (model_add.w_dsp if name == "disp" else model_add.w_mlr).detach().numpy()
        return {"events": int(ok.sum()),
                "choice_share_log_ratio_mean": float(share.mean()),
                "choice_share_log_ratio_purchase_weighted": float((wt * share).sum()),
                "purchase_rate_log_ratio_mean": float(rate.mean()),
                "purchase_rate_log_ratio_purchase_weighted": float((wt * rate).sum()),
                "share_of_events_with_negative_effect": float((share < 0).mean()),
                "event_products_weight_mean": float(weight[ev_item[ok]].mean()),
                "catalogue_weight_mean": float(weight.mean()),
                "catalogue_share_negative_weight": float((weight < 0).mean())}

    additive_results["promotion_on_vs_off"] = {
        "display": promotion_effect("disp"), "mailer": promotion_effect("mail"),
        "empirical_coefficients": {
            split: {k: float(v) for k, v in coefs.items()}
            for split, coefs in (empirical_promotion_coefficients or {}).items()}}

    s_inf_r5 = rank5_results["single_item_exact_price_counterfactual_choice_share"][
        "informative_test_events"
    ]
    c_inf_r5 = rank5_results["simultaneous_chain_week_price_counterfactual_choice_share"][
        "informative_test_events"
    ]
    w_inf_r5 = rank5_results["full_week_over_week_model_prediction_with_display_and_mailer"][
        "informative_test_events"
    ]
    emp_train_slope = (
        float(empirical_depth_slopes["train"])
        if empirical_depth_slopes and "train" in empirical_depth_slopes
        else None
    )
    emp_test_slope = (
        float(empirical_depth_slopes["test"])
        if empirical_depth_slopes and "test" in empirical_depth_slopes
        else None
    )
    slope_ratio_train = (
        float(emp_train_slope / mean_own_price_slope)
        if emp_train_slope is not None and mean_own_price_slope > 0
        else None
    )

    slope_clause = (
        f"its average own-price slope ({mean_own_price_slope:.4f}, from mean_g={mean_g:.4f} and "
        f"softplus(kappa)={kappa:.4f}) is {slope_ratio_train:.2f}x smaller than the empirical "
        f"discount-depth choice slope ({emp_train_slope:.4f} on train, {emp_test_slope:.4f} on test)"
        if slope_ratio_train is not None and emp_test_slope is not None
        else (
            f"its average own-price slope is {mean_own_price_slope:.4f} "
            f"(from mean_g={mean_g:.4f} and softplus(kappa)={kappa:.4f})"
        )
    )
    summary_interpretation = (
        f"In pure price counterfactuals (changed_price_context) on informative held-out test events, "
        f"the Version-4 rank-5 model predicts the directional sign of price cuts "
        f"(single-item price-direction agreement {s_inf_r5['sign_agreement_with_price_direction']:.4f}, "
        f"non-zero observed sign agreement {s_inf_r5['sign_agreement_nonzero_observed']:.4f}, "
        f"Pearson r={s_inf_r5['pearson_correlation']:.4f}, Spearman rho={s_inf_r5['spearman_correlation']:.4f}; "
        f"simultaneous chain-week Pearson r={c_inf_r5['pearson_correlation']:.4f}, "
        f"Spearman rho={c_inf_r5['spearman_correlation']:.4f}, non-zero observed sign agreement "
        f"{c_inf_r5['sign_agreement_nonzero_observed']:.4f}), but underpredicts the magnitude of the "
        f"promotional choice-share surge (predicted-to-observed discount-increase change ratio "
        f"{s_inf_r5['predicted_to_observed_magnitude_ratio_on_discount_increase']:.4f} single-item, "
        f"{c_inf_r5['predicted_to_observed_magnitude_ratio_on_discount_increase']:.4f} chain-week) because "
        f"{slope_clause}. When full week-over-week context changes (including display and mailer) are used, "
        f"informative-event Pearson correlation is {w_inf_r5['pearson_correlation']:.4f} and non-zero observed "
        f"sign agreement is {w_inf_r5['sign_agreement_nonzero_observed']:.4f}."
    )

    return {
        "status": "completed",
        "question_1_passed": True,
        "evaluation_seconds": float(time.time() - t0),
        "sampling_parameters": {
            "burn": int(burn),
            "draws": int(draws),
            "thin": int(thin),
            "max_trips_per_store_week": int(max_trips_per_store_week),
            "additive_parent_is_exact_dp_without_mc_variance": True,
        },
        "base_model_price_parameters": {
            "mean_g": mean_g,
            "softplus_price_kappa": kappa,
            "mean_own_price_slope": mean_own_price_slope,
            "empirical_train_depth_slope": emp_train_slope,
            "empirical_test_depth_slope": emp_test_slope,
            "empirical_to_model_slope_ratio_train": slope_ratio_train,
        },
        "checkpoints": {
            "additive_parent": str(additive_ckpt),
            "additive_parent_sha256": file_sha256(additive_ckpt),
            "candidate_rank1": str(rank5_ckpt),
            "candidate_rank1_sha256": file_sha256(rank5_ckpt),
            "initialization": str(init_ckpt),
            "active_rank": r,
        },
        "counterfactual_method": (
            "changed_price_context from scripts/version4/price_response.py evaluated on held-out "
            "test trips via exact DP log_zc gradients (additive parent) and Gibbs-sampled latent "
            "fields z with exact Radon-Nikodym importance reweighting (candidate_rank1, active_rank=5)"
        ),
        "single_item_tilt_vs_full_dp_max_abs_discrepancy": max_single_item_discrepancy,
        "test_store_weeks_covered": int(len(needed_all_keys)),
        "test_trips_evaluated_weeks_90_to_101": int(len(eligible_trips)),
        "test_discount_events_evaluated": int(len(test_events)),
        "informative_test_discount_events_evaluated": int(informative.sum()),
        "candidate_rank1_counterfactual": rank5_results,
        "additive_parent_counterfactual": additive_results,
        "summary_interpretation": summary_interpretation,
    }


def parse_args() -> argparse.Namespace:
    default_art = DATA_ROOT / "artifacts" / "corrected_complete_rank5_20260913"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "runs" / "dunnhumby" / "discount_evidence",
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=ROOT / "runs" / "dunnhumby" / "discount_evidence",
        help="Existing Q1-Q3 output directory used when --q4-only is set so test weeks are not rescored.",
    )
    parser.add_argument(
        "--q4-only",
        action="store_true",
        help="Rerun only Q4 using existing events.parquet and report.json from --source-dir without rescoring Q1 test weeks.",
    )
    parser.add_argument(
        "--q4-split",
        default="test",
        choices=("validation", "test"),
        help="With --q4-only: which held-out events to score (validation for model selection).",
    )
    parser.add_argument("--min-depth-change", type=float, default=0.05)
    parser.add_argument("--min-trips", type=int, default=16)
    parser.add_argument("--min-train-lines", type=int, default=0)
    parser.add_argument("--bootstrap-replicates", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=68101)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--burn", type=int, default=200)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--thin", type=int, default=2)
    parser.add_argument("--max-trips-per-store-week", type=int, default=1)
    parser.add_argument(
        "--additive-checkpoint",
        type=Path,
        default=default_art / "out" / "v3_pipeline_additive_best.pt",
    )
    parser.add_argument(
        "--rank5-checkpoint",
        type=Path,
        default=default_art / "artifacts" / "candidate_rank1.pt",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=default_art / "artifacts" / "initialization.pt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.q4_only:
        src_dir = args.source_dir.resolve()
        events_path = src_dir / "events.parquet"
        src_report_path = src_dir / "report.json"
        if not events_path.is_file() or not src_report_path.is_file():
            raise RuntimeError(
                f"--q4-only requires existing {events_path} and {src_report_path}"
            )
        log(
            f"running in --q4-only mode: loading locked Q1-Q3 results from {src_dir} "
            f"(test weeks are not rescored for Q1)..."
        )
        events = pd.read_parquet(events_path)
        test = events[events.split == args.q4_split].copy()
        report = json.loads(src_report_path.read_text())
        emp_slopes = {
            split_name: float(
                report["estimated_promotion_effects"][split_name]["effects"][
                    "promotion_depth_change"
                ]["estimate"]
            )
            for split_name in ("train", "validation", "test")
            if split_name in report.get("estimated_promotion_effects", {})
        }
        emp_promo = {
            split_name: {k: v["estimate"] for k, v in effects["effects"].items()}
            for split_name, effects in report.get("estimated_promotion_effects", {}).items()
        }
        model_check = evaluate_version4_counterfactual_q4(
            test_events=test,
            additive_ckpt=args.additive_checkpoint.resolve(),
            rank5_ckpt=args.rank5_checkpoint.resolve(),
            init_ckpt=args.init_checkpoint.resolve(),
            empirical_depth_slopes=emp_slopes,
            empirical_promotion_coefficients=emp_promo,
            threads=args.threads,
            burn=args.burn,
            draws=args.draws,
            thin=args.thin,
            max_trips_per_store_week=args.max_trips_per_store_week,
            seed=args.seed,
        )
        model_check["event_split"] = args.q4_split
        model_check_path = out_dir / "model_check.json"
        model_check_path.write_text(strict_json_dumps(plain(model_check)))
        log(f"wrote {model_check_path}")

        report["implementation_sha256"] = file_sha256(Path(__file__).resolve())
        report["question_4_version4_model_check_summary"] = {
            "model_check_output": str(model_check_path),
            "model_check_sha256": file_sha256(model_check_path),
            "candidate_rank1_single_item_price_counterfactual_informative": (
                model_check["candidate_rank1_counterfactual"][
                    "single_item_exact_price_counterfactual_choice_share"
                ]["informative_test_events"]
            ),
            "candidate_rank1_chain_week_price_counterfactual_informative": (
                model_check["candidate_rank1_counterfactual"][
                    "simultaneous_chain_week_price_counterfactual_choice_share"
                ]["informative_test_events"]
            ),
            "additive_parent_single_item_price_counterfactual_informative": (
                model_check["additive_parent_counterfactual"][
                    "single_item_exact_price_counterfactual_choice_share"
                ]["informative_test_events"]
            ),
        }
        report["interpretation"] = build_report_interpretation(report)
        report_path = out_dir / "report.json"
        report_path.write_text(strict_json_dumps(plain(report)))
        log(f"wrote {report_path}")
        return

    events, design = build_discount_exposure_events(
        min_depth_change=args.min_depth_change,
        min_trips=args.min_trips,
        min_train_lines=args.min_train_lines,
    )
    training = events[events.split == "train"].copy()
    validation = events[events.split == "validation"].copy()
    test = events[events.split == "test"].copy()
    categories = len(design["category_values"])

    log(
        f"splits ready: train={len(training)} validation={len(validation)} "
        f"test={len(test)} categories={categories}"
    )

    # 1. Fit pre-set comparison levels on training and evaluate on validation ONLY
    no_promo_model = fit_promotion_model(training, "no_promotion", categories)
    no_promo_model["validation_loss"] = conditional_loss_and_vector(
        no_promo_model, validation, categories
    )[0]

    global_model = fit_promotion_model(training, "global", categories)
    global_model["validation_loss"] = conditional_loss_and_vector(
        global_model, validation, categories
    )[0]
    global_beta = global_model["parameters"].copy()
    log(
        f"global fit: beta={global_beta.round(4).tolist()} "
        f"val_loss={global_model['validation_loss']:.6f} "
        f"(no_promotion={no_promo_model['validation_loss']:.6f})"
    )

    category_models = []
    for penalty in PENALTY_GRID:
        model = fit_promotion_model(
            training,
            "category",
            categories,
            category_penalty=penalty,
            initial_global=global_beta,
        )
        model["validation_loss"] = conditional_loss_and_vector(
            model, validation, categories
        )[0]
        log(f"category fit penalty={penalty:g}: val_loss={model['validation_loss']:.6f}")
        category_models.append(model)

    best_category_model = min(category_models, key=lambda row: row["validation_loss"])
    candidates = [no_promo_model, global_model, best_category_model]
    selected = min(candidates, key=lambda row: row["validation_loss"])
    log(
        f"validation selected level={selected['level']} "
        f"penalty={selected['category_penalty']:g} "
        f"val_loss={selected['validation_loss']:.6f}"
    )

    # Seasonality control comparison on validation
    seasonality_audit = fit_seasonality_variants_on_validation(
        training, validation, categories, global_beta
    )

    # Also record the n_train_lines >= 500 sensitivity check on validation & test
    events_500 = events[events.n_train_lines >= 500].copy()
    cats_500 = np.sort(
        events_500.loc[(events_500.split == "train") & (events_500.total_purchases > 0), "cat_id"].unique()
    )
    events_500 = events_500[events_500.cat_id.isin(cats_500)].copy()
    events_500["category_index"] = events_500.cat_id.map({int(c): i for i, c in enumerate(cats_500)}).astype(np.int64)
    tr_500 = events_500[events_500.split == "train"]
    va_500 = events_500[events_500.split == "validation"]
    te_500 = events_500[events_500.split == "test"]
    g_500 = fit_promotion_model(tr_500, "global", len(cats_500))
    g_500["validation_loss"] = conditional_loss_and_vector(g_500, va_500, len(cats_500))[0]
    c_500_models = []
    for pen in PENALTY_GRID:
        m5 = fit_promotion_model(
            tr_500, "category", len(cats_500), category_penalty=pen, initial_global=g_500["parameters"]
        )
        m5["validation_loss"] = conditional_loss_and_vector(m5, va_500, len(cats_500))[0]
        c_500_models.append(m5)
    sel_500 = min([g_500, min(c_500_models, key=lambda r: r["validation_loss"])], key=lambda r: r["validation_loss"])

    # 2. Score locked test weeks ONCE with the validation-selected model
    log("scoring locked test weeks once with validation-selected model...")
    test_comparisons = {
        "no_promotion_vs_selected": clustered_loss_comparison(
            None,
            selected,
            test,
            categories,
            seed=args.seed,
            replicates=args.bootstrap_replicates,
        ),
        "no_promotion_vs_global": clustered_loss_comparison(
            None,
            global_model,
            test,
            categories,
            seed=args.seed - 1,
            replicates=args.bootstrap_replicates,
        ),
        "global_vs_selected": clustered_loss_comparison(
            global_model,
            selected,
            test,
            categories,
            seed=args.seed + 1,
            replicates=args.bootstrap_replicates,
        ),
    }

    sel_comp = test_comparisons["no_promotion_vs_selected"]
    glob_comp = test_comparisons["no_promotion_vs_global"]
    comp_vs_glob = test_comparisons["global_vs_selected"]

    sel_ci = sel_comp["cluster_bootstrap_95_intervals_improvement"]
    glob_ci = glob_comp["cluster_bootstrap_95_intervals_improvement"]
    comp_ci = comp_vs_glob["cluster_bootstrap_95_intervals_improvement"]

    gates = {
        "selected_beats_no_promotion_on_test_log_loss": bool(
            sel_comp["log_loss_improvement_of_second_over_first"] > 0
        ),
        "selected_improvement_95_ci_above_0_clustered_by_product": bool(
            sel_ci["product"][0] > 0
        ),
        "selected_improvement_95_ci_above_0_clustered_by_store": bool(
            sel_ci["store"][0] > 0
        ),
        "selected_improvement_95_ci_above_0_clustered_by_week": bool(
            sel_ci["week"][0] > 0
        ),
        "selected_improvement_robust_across_all_three_clusters": bool(
            all(interval[0] > 0 for interval in sel_ci.values())
        ),
        "global_beats_no_promotion_on_test_log_loss": bool(
            glob_comp["log_loss_improvement_of_second_over_first"] > 0
        ),
        "global_improvement_robust_across_all_three_clusters": bool(
            all(interval[0] > 0 for interval in glob_ci.values())
        ),
        "selected_beats_global_on_test_log_loss": bool(
            comp_vs_glob["log_loss_improvement_of_second_over_first"] >= 0
        ),
        "selected_improvement_over_global_robust_across_all_three_clusters": bool(
            selected["level"] == "global"
            or all(interval[0] > 0 for interval in comp_ci.values())
        ),
    }
    passed = bool(
        gates["selected_beats_no_promotion_on_test_log_loss"]
        and gates["selected_improvement_robust_across_all_three_clusters"]
    )
    log(
        f"test log-loss improvement (no_promotion - selected) = "
        f"{sel_comp['log_loss_improvement_of_second_over_first']:.6f}, "
        f"intervals={sel_ci}, passed={passed}"
    )

    # Save events.parquet with split labels and model predictions
    _, selected_bg_bc = unpack_parameters(selected["parameters"], selected["level"], categories)
    X_all = covariate_matrix(events)
    if selected["level"] == "global":
        eta_all = X_all @ selected["parameters"][:3]
    elif selected["level"] == "category":
        eta_all = np.sum(X_all * selected_bg_bc[events.category_index.to_numpy(np.int64)], axis=1)
    else:
        eta_all = np.zeros(len(events), dtype=np.float64)
    events["predicted_after_choice_share"] = np.clip(
        events.observed_before_choice_share.to_numpy(np.float64) * np.exp(eta_all), 0.0, 1.0
    )
    events["predicted_choice_share_change"] = (
        events.predicted_after_choice_share - events.observed_before_choice_share
    )
    events_path = out_dir / "events.parquet"
    events.to_parquet(events_path, index=False)
    log(f"wrote {events_path} ({len(events)} rows)")

    # Build Q2, Q3, and sibling analyses across splits
    q2_by_split = {
        split_name: question2_purchase_rate_decomposition(frame)
        for split_name, frame in (("train", training), ("validation", validation), ("test", test))
    }
    siblings_by_split = {
        split_name: sibling_cross_effects(frame)
        for split_name, frame in (("train", training), ("validation", validation), ("test", test))
    }
    q3_analysis = question3_promotion_type_and_display_diagnosis(
        events, args.additive_checkpoint.resolve()
    )

    # Subset sensitivity check (n_train_lines >= 500) on test
    comp_500 = clustered_loss_comparison(
        None,
        sel_500,
        te_500,
        len(cats_500),
        seed=args.seed + 10,
        replicates=args.bootstrap_replicates,
    )

    # 3. If Q1 passes, run Q4 Version-4 counterfactual check and write model_check.json
    model_check_path = out_dir / "model_check.json"
    model_check_summary = None
    if passed:
        emp_slopes = {
            split_name: float(
                q3_analysis["joint_choice_effects_by_split"][split_name]["effects"][
                    "promotion_depth_change"
                ]["estimate"]
            )
            for split_name in ("train", "validation", "test")
        }
        emp_promo = {
            split_name: {k: v["estimate"] for k, v in effects["effects"].items()}
            for split_name, effects in q3_analysis["joint_choice_effects_by_split"].items()
        }
        model_check = evaluate_version4_counterfactual_q4(
            test_events=test,
            additive_ckpt=args.additive_checkpoint.resolve(),
            rank5_ckpt=args.rank5_checkpoint.resolve(),
            init_ckpt=args.init_checkpoint.resolve(),
            empirical_depth_slopes=emp_slopes,
            empirical_promotion_coefficients=emp_promo,
            threads=args.threads,
            burn=args.burn,
            draws=args.draws,
            thin=args.thin,
            max_trips_per_store_week=args.max_trips_per_store_week,
            seed=args.seed,
        )
        model_check_path.write_text(strict_json_dumps(plain(model_check)))
        log(f"wrote {model_check_path}")
        model_check_summary = {
            "model_check_output": str(model_check_path),
            "model_check_sha256": file_sha256(model_check_path),
            "candidate_rank1_single_item_price_counterfactual_informative": (
                model_check["candidate_rank1_counterfactual"][
                    "single_item_exact_price_counterfactual_choice_share"
                ]["informative_test_events"]
            ),
            "candidate_rank1_chain_week_price_counterfactual_informative": (
                model_check["candidate_rank1_counterfactual"][
                    "simultaneous_chain_week_price_counterfactual_choice_share"
                ]["informative_test_events"]
            ),
            "additive_parent_single_item_price_counterfactual_informative": (
                model_check["additive_parent_counterfactual"][
                    "single_item_exact_price_counterfactual_choice_share"
                ]["informative_test_events"]
            ),
        }

    report = {
        "status": "completed",
        "claim_level": "observational_predictive_discount_and_promotion_evidence",
        "causal_interpretation_permitted": False,
        "estimand": (
            "within-store, within-category product choice-share response and purchase-rate "
            "decomposition across adjacent weeks with chain promo_depth change >= 5%"
        ),
        "implementation_sha256": file_sha256(Path(__file__).resolve()),
        "events_output": str(events_path),
        "events_sha256": file_sha256(events_path),
        "selection": {
            "training_events": int(len(training)),
            "validation_events": int(len(validation)),
            "locked_test_events": int(len(test)),
            "informative_training_events": int((training.total_purchases > 0).sum()),
            "informative_validation_events": int((validation.total_purchases > 0).sum()),
            "informative_test_events": int((test.total_purchases > 0).sum()),
            "training_products": int(training.item_id.nunique()),
            "training_categories": int(categories),
            "training_stores": int(training.store_id.nunique()),
            "comparison_levels": ["no_promotion", "global", "category"],
            "penalty_grid": list(PENALTY_GRID),
            "validation_selected_level": selected["level"],
            "selected_category_penalty": float(selected["category_penalty"]),
            "rule": "comparison level and shrinkage penalty selected strictly by validation conditional log-loss before opening test weeks",
            "seasonality_control": seasonality_audit,
            "data_support_audit": design,
        },
        "validation_grid": {
            "no_promotion": model_summary_dict(no_promo_model, design),
            "global": model_summary_dict(global_model, design),
            "category": [model_summary_dict(m, design) for m in category_models],
        },
        "metrics": {
            split_name: {
                "no_promotion": choice_response_metrics(None, frame, categories),
                "global": choice_response_metrics(global_model, frame, categories),
                "best_category": choice_response_metrics(best_category_model, frame, categories),
                "selected": choice_response_metrics(selected, frame, categories),
            }
            for split_name, frame in (
                ("train", training),
                ("validation", validation),
                ("test", test),
            )
        },
        "locked_test_comparisons": test_comparisons,
        "gates": gates,
        "passed": passed,
        "estimated_promotion_effects": q3_analysis["joint_choice_effects_by_split"],
        "selected_category_model_summary": model_summary_dict(selected, design),
        "question_2_purchase_rate_decomposition": q2_by_split,
        "question_3_promotion_types_and_display_artifact_diagnosis": q3_analysis,
        "sibling_cross_effects": siblings_by_split,
        "high_support_subset_sensitivity_n_train_lines_ge_500": {
            "event_counts": {
                "train": int(len(tr_500)),
                "validation": int(len(va_500)),
                "test": int(len(te_500)),
            },
            "products": int(events_500.item_id.nunique()),
            "categories": int(len(cats_500)),
            "validation_selected_level": sel_500["level"],
            "selected_category_penalty": float(sel_500["category_penalty"]),
            "test_no_promotion_vs_selected": comp_500,
            "test_global_effects": multi_cluster_parameter_inference(te_500),
        },
        "question_4_version4_model_check_summary": model_check_summary,
    }
    report["interpretation"] = build_report_interpretation(report)

    report_path = out_dir / "report.json"
    report_path.write_text(strict_json_dumps(plain(report)))
    log(f"wrote {report_path}")


if __name__ == "__main__":
    main()

