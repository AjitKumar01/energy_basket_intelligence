#!/usr/bin/env python3
"""Audit real price-event provenance before choosing a response estimator.

This stage is deliberately descriptive.  It records chain price changes observed in
transaction-derived weekly price data and determines the strongest defensible claim from
the available metadata.  It never upgrades observational events to randomized treatment.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from checkpoint_io import ROOT
from provenance import file_sha256, strict_json_dumps


def event_split(week: pd.Series, validation_from: int, test_from: int) -> pd.Series:
    return pd.Series(np.where(week >= test_from, "test",
                     np.where(week >= validation_from, "validation", "train")),
                     index=week.index)


def build_events(price: pd.DataFrame, items: pd.DataFrame, meta: dict) -> pd.DataFrame:
    chosen = price.merge(items[["PRODUCT_ID", "item_id"]], on="PRODUCT_ID", how="inner")
    chosen = chosen.sort_values(["item_id", "WEEK_NO"], kind="stable").copy()
    grouped = chosen.groupby("item_id", sort=False)
    chosen["previous_week"] = grouped.WEEK_NO.shift()
    chosen["previous_price"] = grouped.price.shift()
    chosen["previous_base_price"] = grouped.base_price.shift()
    chosen["previous_promo_depth"] = grouped.promo_depth.shift()
    chosen["weeks_since_previous_observation"] = chosen.WEEK_NO - chosen.previous_week
    chosen["log_price_change"] = np.log(chosen.price) - np.log(chosen.previous_price)
    valid = (chosen.weeks_since_previous_observation == 1) & np.isfinite(chosen.log_price_change)
    changed = valid & (chosen.log_price_change.abs() >= np.log(1.01))
    event = chosen.loc[changed, [
        "item_id", "PRODUCT_ID", "previous_week", "WEEK_NO", "previous_price", "price",
        "previous_base_price", "base_price", "previous_promo_depth", "promo_depth",
        "log_price_change", "n_tx", "n_store", "weeks_since_previous_observation",
    ]].copy()
    event["event_id"] = np.arange(len(event), dtype=np.int64)
    event["event_split"] = event_split(
        event.WEEK_NO, int(meta["val_from_week"]), int(meta["test_from_week"]))
    event["price_source"] = meta["unit_price_source_column"]
    event["chain_price_observed"] = True
    event["concurrent_promotion_change"] = (
        (event.promo_depth - event.previous_promo_depth).abs() > 1e-12)
    event["assignment_unit"] = (
        "item=" + event.item_id.astype(str) + ";week=" + event.WEEK_NO.astype(str))
    event["eligible_outcome"] = "conditional incidence among observed nonempty baskets"
    return event


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--events-output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    events_output = args.events_output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    events_output.parent.mkdir(parents=True, exist_ok=True)

    price_path = ROOT / "data/price_week.parquet"
    transaction_path = ROOT / "data/tx.parquet"
    items_path = ROOT / "basket_input/items.parquet"
    meta_path = ROOT / "basket_input/meta.json"
    store_path = ROOT / "basket_input/store_price.npz"
    promo_path = ROOT / "basket_input/promo.npz"
    meta = json.loads(meta_path.read_text())
    price = pd.read_parquet(price_path)
    items = pd.read_parquet(items_path).sort_values("item_id")
    event = build_events(price, items, meta)
    event.to_parquet(events_output, index=False)
    with np.load(store_path) as store:
        carried = store["carried"]
        store_summary = {
            "observed_item_store_week_cells": int(len(store["dev"])),
            "item_store_pairs_with_any_observation": int(np.count_nonzero(carried)),
            "possible_item_store_pairs": int(carried.size),
            "pair_coverage": float(carried.mean()),
        }
    with np.load(promo_path) as promo:
        promotion_summary = {
            "cells": int(len(promo["keys"])),
            "coverage_week_min": int(promo["coverage_min_week"]),
            "coverage_week_max": int(promo["coverage_max_week"]),
        }
    split_counts = event.event_split.value_counts().to_dict()
    clean = event.loc[~event.concurrent_promotion_change]
    depth = event.log_price_change.abs()
    report = {
        "status": "completed",
        "input_sha256": {str(path): file_sha256(path) for path in (
            price_path, transaction_path, items_path, meta_path, store_path, promo_path)},
        "price_basis": meta["price_basis"],
        "unit_price_source_column": meta["unit_price_source_column"],
        "source_interpretation": (
            "weekly chain prices and store deviations are reconstructed from observed "
            "transactions; purchaser composition and sparse observation can affect them"),
        "modeled_products": int(len(items)),
        "weekly_price_rows_for_modeled_products": int(
            price.PRODUCT_ID.isin(items.PRODUCT_ID).sum()),
        "adjacent_week_price_events_at_least_1pct": int(len(event)),
        "event_counts_by_split": {str(k): int(v) for k, v in split_counts.items()},
        "distinct_event_products": int(event.item_id.nunique()),
        "distinct_event_weeks": int(event.WEEK_NO.nunique()),
        "events_without_concurrent_promo_depth_change": int(len(clean)),
        "absolute_log_change_quantiles": {
            str(q): float(depth.quantile(q)) for q in (0, .1, .5, .9, .99, 1)
        } if len(event) else {},
        "store_price": store_summary,
        "promotion": promotion_summary,
        "available_outcomes": [
            "purchased-product incidence, units and transaction value conditional on an observed basket"
        ],
        "missing_for_total_demand_or_profit": [
            "randomized price assignment or documented natural-experiment assignment",
            "eligible household-store-product opportunities including no visit/no purchase",
            "inventory and contemporaneous stockout status",
            "wholesale cost and margin",
            "competitor prices and store switching",
        ],
        "strongest_defensible_design": "observational_conditional_predictive_assessment",
        "causal_price_identification": "not_identifiable",
        "profit_identification": "not_identifiable",
        "events_output": str(events_output),
        "events_sha256": file_sha256(events_output),
        "event_definition": (
            "at least 1% chain-level loyalty-price change between adjacent observed weeks; "
            "not a verified exogenous intervention"),
    }
    output.write_text(strict_json_dumps(report))
    print(strict_json_dumps(report), end="")


if __name__ == "__main__":
    main()
