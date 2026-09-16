#!/usr/bin/env python3
"""Fit the smallest real-price response justified by same-store evidence.

For a store/product price event, y0 of n0 baskets contain the product before the
change and y1 of n1 contain it after.  Conditional on m=y0+y1, the Poisson-rate
model is

    y1 | m ~ Binomial(m, sigmoid(log(n1/n0) - sensitivity * log_price_change)).

Conditioning removes the unknown event-specific baseline purchase rate.  Nonnegative
sensitivities preserve monotone own-price response.  Training fits global, category,
and product hierarchies; validation selects shrinkage; test is opened once.
The result is observational, not causal.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit

from checkpoint_io import ROOT
from provenance import file_sha256, model_data_root, strict_json_dumps


DATA_ROOT = model_data_root(ROOT)
BI = DATA_ROOT / "basket_input"
DATA = DATA_ROOT / "data"


def _sparse_lookup(keys: np.ndarray, values: np.ndarray,
                   query: np.ndarray) -> np.ndarray:
    if len(keys) == 0:
        return np.zeros(len(query), dtype=values.dtype)
    position = np.searchsorted(keys, query)
    safe = np.minimum(position, len(keys) - 1)
    found = (position < len(keys)) & (keys[safe] == query)
    result = np.zeros(len(query), dtype=values.dtype)
    result[found] = values[position[found]]
    return result


def build_complete_exposure_events() -> tuple[pd.DataFrame, dict]:
    """Build same-store events without requiring a sale to reveal store price.

    Store-level observed-price rows are outcome selected: they exist only when a sale
    occurred.  Use the model's carried chain-week price for all stores in the static,
    training-defined assortment.  Sparse display/mailer values are contemporaneous
    covariates, while counts include explicit zeros.
    """
    baskets = pd.read_parquet(
        BI / "baskets.parquet",
        columns=["BASKET_ID", "WEEK_NO", "item_id", "store_id"])
    items = pd.read_parquet(
        BI / "items.parquet",
        columns=["PRODUCT_ID", "item_id", "n_train_lines", "cat_id"])
    products = int(items.item_id.max()) + 1
    with np.load(BI / "store_price.npz") as store_price:
        carried = store_price["carried"].astype(bool)
        stores = int(store_price["n_stores"])
    if carried.shape != (products, stores):
        raise RuntimeError("training assortment shape does not match product/store support")

    trips = baskets[["BASKET_ID", "store_id", "WEEK_NO"]].drop_duplicates()
    trip_count = np.zeros((stores, 128), dtype=np.int64)
    grouped_trips = trips.groupby(["store_id", "WEEK_NO"]).size()
    trip_count[
        grouped_trips.index.get_level_values(0).to_numpy(np.int64),
        grouped_trips.index.get_level_values(1).to_numpy(np.int64),
    ] = grouped_trips.to_numpy(np.int64)

    bought = (baskets.groupby(["item_id", "store_id", "WEEK_NO"])
              .BASKET_ID.nunique().sort_index())
    bought_key = ((bought.index.get_level_values(0).to_numpy(np.int64) * stores
                   + bought.index.get_level_values(1).to_numpy(np.int64)) * 128
                  + bought.index.get_level_values(2).to_numpy(np.int64))
    bought_order = np.argsort(bought_key)
    bought_key = bought_key[bought_order]
    bought_value = bought.to_numpy(np.int64)[bought_order]

    meta = json.loads((BI / "meta.json").read_text())
    n_periods = int(meta["analysis_last_week"])
    weekly_path = DATA / "price_week.parquet"
    if weekly_path.is_file():
        weekly = pd.read_parquet(
            weekly_path,
            columns=["PRODUCT_ID", "WEEK_NO", "price", "base_price"])
        weekly = weekly.merge(
            items[["PRODUCT_ID", "item_id"]], on="PRODUCT_ID", how="inner",
            validate="many_to_one")
        weekly_price_source = "declared price_week.parquet"
    else:
        daily_price = np.exp(np.load(BI / "log_price.npy").astype(np.float64))
        if daily_price.shape[0] != products or daily_price.shape[1] < n_periods * 7:
            raise RuntimeError("model log-price panel does not cover declared periods")
        period = np.arange(1, n_periods + 1, dtype=np.int64)
        weekly = pd.DataFrame({
            "item_id": np.repeat(np.arange(products), n_periods),
            "WEEK_NO": np.tile(period, products),
            "price": daily_price[:, ::7][:, :n_periods].reshape(-1),
        })
        weekly["base_price"] = weekly.price
        weekly_price_source = "audited canonical log_price.npy period panel"
    price = np.full((products, n_periods + 1), np.nan, dtype=np.float64)
    base = np.full_like(price, np.nan)
    price[weekly.item_id.to_numpy(np.int64), weekly.WEEK_NO.to_numpy(np.int64)] = \
        weekly.price.to_numpy(np.float64)
    base[weekly.item_id.to_numpy(np.int64), weekly.WEEK_NO.to_numpy(np.int64)] = \
        weekly.base_price.to_numpy(np.float64)
    reliable_product = np.isfinite(price).any(axis=1) & np.isfinite(base).any(axis=1)
    price[reliable_product] = (pd.DataFrame(price[reliable_product])
                               .ffill(axis=1).bfill(axis=1).to_numpy())
    base[reliable_product] = (pd.DataFrame(base[reliable_product])
                              .ffill(axis=1).bfill(axis=1).to_numpy())
    if (not np.isfinite(price[reliable_product]).all()
            or not np.isfinite(base[reliable_product]).all()
            or (price[reliable_product] <= 0).any()
            or (base[reliable_product] <= 0).any()):
        raise RuntimeError("carried chain price exposure has invalid cells")

    observed_promotion_path = BI / "promo_observed.npz"
    with np.load(observed_promotion_path if observed_promotion_path.is_file()
                 else BI / "promo.npz") as promotion:
        promo_keys = promotion["keys"].astype(np.int64)
        display_values = promotion["disp"]
        mailer_values = promotion["mail"]

    item_details = items.set_index("item_id")
    canonical = meta.get("dataset") == "canonical_external_basket"
    first_event_week = max(
        2, int(meta.get("analysis_first_week", 1)), 2 if canonical else 10)
    last_event_week = min(n_periods, n_periods if canonical else 101)
    validation_from = int(meta["val_from_week"])
    test_from = int(meta["test_from_week"])
    rows = []
    for item in np.flatnonzero(reliable_product):
        log_change = np.log(price[item, 1:]) - np.log(price[item, :-1])
        candidate_weeks = np.flatnonzero(
            (np.arange(1, n_periods + 1) >= first_event_week)
            & (np.arange(1, n_periods + 1) <= last_event_week)
            & (np.abs(log_change) >= math.log(1.01))) + 1
        for week in candidate_weeks:
            previous_week = week - 1
            store = np.flatnonzero(
                carried[item]
                & (trip_count[:, week] > 0)
                & (trip_count[:, previous_week] > 0))
            if not len(store):
                continue
            current_key = ((item * stores + store) * 128 + week).astype(np.int64)
            previous_key = ((item * stores + store) * 128
                            + previous_week).astype(np.int64)
            current_purchase = _sparse_lookup(
                bought_key, bought_value, current_key).astype(np.int64)
            previous_purchase = _sparse_lookup(
                bought_key, bought_value, previous_key).astype(np.int64)
            current_display = _sparse_lookup(
                promo_keys, display_values, current_key)
            previous_display = _sparse_lookup(
                promo_keys, display_values, previous_key)
            current_mailer = _sparse_lookup(
                promo_keys, mailer_values, current_key)
            previous_mailer = _sparse_lookup(
                promo_keys, mailer_values, previous_key)
            current_depth = 1.0 - price[item, week] / base[item, week]
            previous_depth = 1.0 - price[item, previous_week] / base[item, previous_week]
            detail = item_details.loc[item]
            rows.append(pd.DataFrame({
                "item_id": item,
                "cat_id": int(detail.cat_id),
                "store_id": store,
                "WEEK_NO": week,
                "previous_WEEK_NO": previous_week,
                "price": price[item, week],
                "previous_price": price[item, previous_week],
                "base_price": base[item, week],
                "previous_base_price": base[item, previous_week],
                "n_train_lines": int(detail.n_train_lines),
                "trips": trip_count[store, week],
                "previous_trips": trip_count[store, previous_week],
                "purchases": current_purchase,
                "previous_purchases": previous_purchase,
                "display": current_display,
                "previous_display": previous_display,
                "mailer": current_mailer,
                "previous_mailer": previous_mailer,
                "promotion_depth": current_depth,
                "previous_promotion_depth": previous_depth,
            }))
    if not rows:
        raise RuntimeError("complete exposure construction produced no price events")
    events = pd.concat(rows, ignore_index=True)
    events["log_price_change"] = np.log(events.price) - np.log(events.previous_price)
    events["promotion_depth_change"] = (
        events.promotion_depth - events.previous_promotion_depth)
    events["display_changed"] = events.display != events.previous_display
    events["mailer_changed"] = events.mailer != events.previous_mailer
    events["promotion_clean"] = (
        (events.display == 0) & (events.previous_display == 0)
        & (events.mailer == 0) & (events.previous_mailer == 0))
    events["split"] = np.where(
        events.WEEK_NO >= test_from, "test",
        np.where(events.WEEK_NO >= validation_from, "validation", "train"))
    events["observed_before_incidence"] = (
        events.previous_purchases / events.previous_trips)
    events["observed_after_incidence"] = events.purchases / events.trips
    events["observed_response"] = (
        events.observed_after_incidence - events.observed_before_incidence)
    return events, {
        "construction": "complete_training_assortment_x_store_x_chain_price_change",
        "price_exposure": str(meta.get("price_basis", "declared modeled price")),
        "weekly_price_source": weekly_price_source,
        "event_week_range": [first_event_week, last_event_week],
        "validation_from_week": validation_from,
        "test_from_week": test_from,
        "store_price_deviation_used": False,
        "reason_store_price_excluded": (
            "store price cells exist only following a product sale"),
        "zero_purchase_store_week_cells_retained": True,
        "raw_events": int(len(events)),
        "raw_products": int(events.item_id.nunique()),
        "products_without_reliable_price_exposure": int((~reliable_product).sum()),
        "promotion_event_policy": (
            "exclude any before/after event with recorded display, advertising, or "
            "special-price activity"),
        "raw_stores": int(events.store_id.nunique()),
    }


def base_eligible_events(events: pd.DataFrame,
                         minimum_product_training_lines: int = 500) -> pd.Series:
    """Eligibility rules that do not depend on the fitted training support."""
    promotion_clean = (events.promotion_clean if "promotion_clean" in events
                       else pd.Series(True, index=events.index))
    return (
        (events.promotion_depth_change.abs() < .01)
        & promotion_clean
        & (~events.display_changed)
        & (~events.mailer_changed)
        & (events.log_price_change.abs() >= math.log(1.05))
        & (events.log_price_change.abs() <= .70)
        & (events.n_train_lines >= minimum_product_training_lines)
        & (events.trips >= 16)
        & (events.previous_trips >= 16)
    )


def restrict_to_training_price_support(
        events: pd.DataFrame, minimum_product_training_lines: int = 500
        ) -> tuple[pd.DataFrame, dict]:
    """Learn each product's price-change range from eligible training rows only."""
    eligible = events[base_eligible_events(
        events, minimum_product_training_lines)].copy()
    support = (eligible[eligible.split == "train"]
               .groupby("item_id").log_price_change
               .agg(training_change_low="min", training_change_high="max",
                    training_store_events="size"))
    supported = eligible.join(support, on="item_id")
    supported = supported[
        supported.training_change_low.notna()
        & (supported.log_price_change >= supported.training_change_low)
        & (supported.log_price_change <= supported.training_change_high)
    ].copy()
    return supported, {
        "eligible_before_training_price_support": {
            str(key): int(value)
            for key, value in eligible.split.value_counts().items()
        },
        "after_training_price_support": {
            str(key): int(value)
            for key, value in supported.split.value_counts().items()
        },
    }


def prepare_events(minimum_product_training_lines: int | None = None
                   ) -> tuple[pd.DataFrame, dict]:
    events, source_summary = build_complete_exposure_events()
    if minimum_product_training_lines is None:
        meta = json.loads((BI / "meta.json").read_text())
        if meta.get("dataset") == "canonical_external_basket":
            preprocessing = json.loads((BI / "preprocessing_manifest.json").read_text())
            minimum_product_training_lines = int(
                preprocessing["cohort"]["minimum_selected_item_training_lines"])
        else:
            minimum_product_training_lines = 500
    events, support_audit = restrict_to_training_price_support(
        events, minimum_product_training_lines)
    events["total_purchases"] = events.previous_purchases + events.purchases
    supported_counts = {
        str(key): int(value) for key, value in events.split.value_counts().items()}
    training_products = set(
        events.loc[(events.split == "train") & (events.total_purchases > 0),
                   "item_id"].astype(int).tolist())
    if not training_products:
        raise RuntimeError("no products have eligible training price events")
    events = events[events.item_id.isin(training_products)].copy()
    final_counts = {
        str(key): int(value) for key, value in events.split.value_counts().items()}
    category = pd.read_parquet(
        BI / "items.parquet", columns=["item_id", "cat_id"])
    if events.cat_id.isna().any():
        raise RuntimeError("a supported product has no category")
    events["offset"] = np.log(events.trips / events.previous_trips)
    if not np.isfinite(events[["offset", "log_price_change"]].to_numpy()).all():
        raise RuntimeError("non-finite event design")
    if events.duplicated(["store_id", "item_id", "WEEK_NO"]).any():
        raise RuntimeError("duplicate store/product/week price event")
    if ((events.purchases < 0) | (events.previous_purchases < 0)
            | (events.purchases > events.trips)
            | (events.previous_purchases > events.previous_trips)).any():
        raise RuntimeError("invalid product purchase count or trip denominator")
    training = events[events.split == "train"]
    product_values = np.sort(training.item_id.unique())
    category_values = np.sort(training.cat_id.unique())
    product_index = {int(value): i for i, value in enumerate(product_values)}
    category_index = {int(value): i for i, value in enumerate(category_values)}
    product_category = (category.set_index("item_id").loc[product_values, "cat_id"]
                        .map(category_index).to_numpy(np.int64))
    events["product_index"] = events.item_id.map(product_index)
    events["category_index"] = events.cat_id.map(category_index)
    if events[["product_index", "category_index"]].isna().any().any():
        raise RuntimeError("held-out event lacks training product/category support")
    events[["product_index", "category_index"]] = events[
        ["product_index", "category_index"]].astype(np.int64)
    design = {
        "product_values": product_values,
        "category_values": category_values,
        "product_category": product_category,
        "source_summary": source_summary,
        "eligible_event_counts_before_training_price_support":
            support_audit["eligible_before_training_price_support"],
        "event_counts_after_training_price_support": supported_counts,
        "event_counts_after_informative_training_product_support": final_counts,
        "heldout_events_removed_outside_eligible_training_price_support": {
            split: (support_audit["eligible_before_training_price_support"].get(split, 0)
                    - supported_counts.get(split, 0))
            for split in ("validation", "test")
        },
        "heldout_events_removed_without_informative_training_product": {
            split: supported_counts.get(split, 0) - final_counts.get(split, 0)
            for split in ("validation", "test")
        },
        "training_product_support_rule": (
            "at least one eligible training store-event with a purchase in either week"),
        "minimum_product_training_lines": int(minimum_product_training_lines),
    }
    return events, design


def unpack(parameters: np.ndarray, level: str, categories: int, products: int):
    global_sensitivity = parameters[0]
    if level == "global":
        return global_sensitivity, None, None
    category = parameters[1:1 + categories]
    if level == "category":
        return global_sensitivity, category, None
    product = parameters[1 + categories:1 + categories + products]
    return global_sensitivity, category, product


def event_sensitivity(parameters: np.ndarray, frame: pd.DataFrame, level: str,
                      categories: int, products: int) -> np.ndarray:
    global_sensitivity, category, product = unpack(
        parameters, level, categories, products)
    if level == "global":
        return np.full(len(frame), global_sensitivity, dtype=np.float64)
    if level == "category":
        return category[frame.category_index.to_numpy(np.int64)]
    return product[frame.product_index.to_numpy(np.int64)]


def objective(parameters: np.ndarray, frame: pd.DataFrame, level: str,
              categories: int, products: int, product_category: np.ndarray,
              category_penalty: float, product_penalty: float
              ) -> tuple[float, np.ndarray]:
    x = frame.log_price_change.to_numpy(np.float64)
    offset = frame.offset.to_numpy(np.float64)
    successes = frame.purchases.to_numpy(np.float64)
    totals = frame.total_purchases.to_numpy(np.float64)
    denominator = max(float(totals.sum()), 1.0)
    sensitivity = event_sensitivity(parameters, frame, level, categories, products)
    linear = offset - x * sensitivity
    loss = float(np.sum(totals * np.logaddexp(0.0, linear) - successes * linear)
                 / denominator)
    residual = (totals * expit(linear) - successes) / denominator
    gradient = np.zeros_like(parameters)
    event_gradient = -x * residual
    global_sensitivity, category, product = unpack(
        parameters, level, categories, products)
    if level == "global":
        gradient[0] = event_gradient.sum()
        return loss, gradient
    category_event = frame.category_index.to_numpy(np.int64)
    np.add.at(gradient[1:1 + categories], category_event, event_gradient)
    category_difference = category - global_sensitivity
    loss += category_penalty * float(np.mean(category_difference ** 2))
    category_regularization = 2.0 * category_penalty * category_difference / categories
    gradient[1:1 + categories] += category_regularization
    gradient[0] = -float(category_regularization.sum())
    if level == "category":
        return loss, gradient
    # Product predictions replace category predictions; remove the category data gradient.
    gradient[1:1 + categories] -= np.bincount(
        category_event, weights=event_gradient, minlength=categories)
    product_event = frame.product_index.to_numpy(np.int64)
    np.add.at(gradient[1 + categories:], product_event, event_gradient)
    product_difference = product - category[product_category]
    loss += product_penalty * float(np.mean(product_difference ** 2))
    product_regularization = 2.0 * product_penalty * product_difference / products
    gradient[1 + categories:] += product_regularization
    np.add.at(gradient[1:1 + categories], product_category, -product_regularization)
    return loss, gradient


def initial_parameters(level: str, categories: int, products: int,
                       product_category: np.ndarray, global_value: float,
                       category_value: np.ndarray | None = None) -> np.ndarray:
    if level == "global":
        return np.asarray([global_value], dtype=np.float64)
    if category_value is None:
        category_value = np.full(categories, global_value, dtype=np.float64)
    if level == "category":
        return np.concatenate(([global_value], category_value))
    return np.concatenate((
        [global_value], category_value, category_value[product_category]))


def fit_model(frame: pd.DataFrame, level: str, categories: int, products: int,
              product_category: np.ndarray, category_penalty: float = 0.0,
              product_penalty: float = 0.0, initial: np.ndarray | None = None) -> dict:
    if initial is None:
        initial = initial_parameters(level, categories, products, product_category, .25)
    start = np.asarray(initial, dtype=np.float64)
    answer = None
    projected_gradient_max = math.inf
    total_iterations = 0
    refinement_attempts = 0
    for attempt in range(3):
        refinement_attempts = attempt + 1
        answer = minimize(
            lambda value: objective(
                value, frame, level, categories, products, product_category,
                category_penalty, product_penalty),
            start, method="L-BFGS-B", jac=True,
            bounds=[(0.0, 5.0)] * len(initial),
            options={"maxiter": 5000, "maxfun": 100000, "ftol": 1e-15,
                     "gtol": 1e-9, "maxls": 50, "maxcor": 50})
        total_iterations += int(answer.nit)
        projected_gradient = np.asarray(answer.jac).copy()
        at_lower = answer.x <= 1e-10
        at_upper = answer.x >= 5.0 - 1e-10
        projected_gradient[at_lower & (projected_gradient > 0)] = 0.0
        projected_gradient[at_upper & (projected_gradient < 0)] = 0.0
        projected_gradient_max = float(np.max(np.abs(projected_gradient)))
        if answer.success and projected_gradient_max <= 1e-5:
            break
        start = answer.x
    assert answer is not None
    if not answer.success or not np.isfinite(answer.fun) \
            or not np.isfinite(answer.x).all() or projected_gradient_max > 1e-5:
        raise RuntimeError(
            f"{level} price fit failed: {answer.message}; iterations={answer.nit}; "
            f"projected_gradient_max={projected_gradient_max:.6g}; "
            f"objective={answer.fun:.12g}")
    return {
        "level": level,
        "category_penalty": float(category_penalty),
        "product_penalty": float(product_penalty),
        "parameters": answer.x,
        "penalized_training_loss": float(answer.fun),
        "iterations": total_iterations,
        "refinement_attempts": refinement_attempts,
        "gradient_max": float(np.max(np.abs(answer.jac))),
        "projected_gradient_max": projected_gradient_max,
    }


def conditional_loss(model: dict | None, frame: pd.DataFrame,
                     categories: int, products: int) -> tuple[float, np.ndarray]:
    totals = frame.total_purchases.to_numpy(np.float64)
    successes = frame.purchases.to_numpy(np.float64)
    if model is None:
        linear = frame.offset.to_numpy(np.float64)
    else:
        sensitivity = event_sensitivity(
            model["parameters"], frame, model["level"], categories, products)
        linear = (frame.offset.to_numpy(np.float64)
                  - frame.log_price_change.to_numpy(np.float64) * sensitivity)
    per_event = totals * np.logaddexp(0.0, linear) - successes * linear
    return float(per_event.sum() / max(totals.sum(), 1.0)), per_event


def unconstrained_global_evidence(frame: pd.DataFrame) -> dict:
    """Estimate the signed global response and product-clustered uncertainty."""
    answer = minimize(
        lambda value: objective(value, frame, "global", 0, 0,
                                np.empty(0, dtype=np.int64), 0.0, 0.0),
        np.zeros(1, dtype=np.float64), method="L-BFGS-B", jac=True,
        bounds=[(-5.0, 5.0)],
        options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-11})
    if not answer.success or not np.isfinite(answer.x).all():
        raise RuntimeError(f"unconstrained global evidence fit failed: {answer.message}")
    sensitivity = float(answer.x[0])
    x = frame.log_price_change.to_numpy(np.float64)
    totals = frame.total_purchases.to_numpy(np.float64)
    successes = frame.purchases.to_numpy(np.float64)
    probability = expit(frame.offset.to_numpy(np.float64) - x * sensitivity)
    event_score = x * (successes - totals * probability)
    hessian = float(np.sum(totals * probability * (1.0 - probability) * x ** 2))
    cluster_score = pd.Series(event_score).groupby(
        frame.item_id.to_numpy(np.int64)).sum().to_numpy()
    clusters = int(len(cluster_score))
    correction = clusters / max(clusters - 1, 1)
    standard_error = math.sqrt(
        correction * float(cluster_score @ cluster_score) / hessian ** 2)
    return {
        "sensitivity": sensitivity,
        "product_clustered_standard_error": standard_error,
        "product_clustered_95_interval": [
            sensitivity - 1.96 * standard_error,
            sensitivity + 1.96 * standard_error,
        ],
        "product_clusters": clusters,
        "boundary_hit": bool(abs(sensitivity) >= 5.0 - 1e-8),
    }


def response_metrics(model: dict | None, frame: pd.DataFrame,
                     categories: int, products: int) -> dict:
    loss, _ = conditional_loss(model, frame, categories, products)
    if model is None:
        sensitivity = np.zeros(len(frame))
    else:
        sensitivity = event_sensitivity(
            model["parameters"], frame, model["level"], categories, products)
    price_change = frame.log_price_change.to_numpy(np.float64)
    p0 = frame.observed_before_incidence.to_numpy(np.float64)
    predicted_after = np.clip(p0 * np.exp(-sensitivity * price_change), 0.0, 1.0)
    predicted_response = predicted_after - p0
    observed = frame.observed_response.to_numpy(np.float64)
    informative = frame.total_purchases.to_numpy(np.float64) > 0
    predicted_informative = predicted_response[informative]
    observed_informative = observed[informative]
    pearson = None
    if (len(observed_informative) > 1
            and np.std(predicted_informative) > 0
            and np.std(observed_informative) > 0):
        pearson = float(np.corrcoef(
            predicted_informative, observed_informative)[0, 1])
    return {
        "conditional_log_loss_per_purchase": loss,
        "events": int(len(frame)),
        "informative_events_with_a_purchase_in_either_week": int(informative.sum()),
        "informative_purchases": int(frame.total_purchases.sum()),
        "informative_response_mse": float(np.mean(
            (observed_informative - predicted_informative) ** 2)),
        "informative_response_mae": float(np.mean(np.abs(
            observed_informative - predicted_informative))),
        "informative_response_pearson": pearson,
        "response_sign_agreement": float(np.mean(
            np.sign(predicted_informative) == np.sign(observed_informative))),
        "sensitivity_quantiles": {
            str(q): float(np.quantile(sensitivity, q))
            for q in (0, .1, .25, .5, .75, .9, 1)},
    }


def _cluster_bootstrap_interval(left: np.ndarray, right: np.ndarray,
                                totals: np.ndarray, labels: np.ndarray,
                                generator: np.random.Generator,
                                replicates: int) -> tuple[list[float], int]:
    codes, unique = pd.factorize(labels, sort=True)
    grouped = np.zeros((len(unique), 3), dtype=np.float64)
    np.add.at(grouped[:, 0], codes, left)
    np.add.at(grouped[:, 1], codes, right)
    np.add.at(grouped[:, 2], codes, totals)
    values = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        chosen = generator.integers(0, len(unique), len(unique))
        sample = grouped[chosen].sum(0)
        values[replicate] = (sample[0] - sample[1]) / max(sample[2], 1.0)
    return ([float(value) for value in np.quantile(values, [.025, .975])],
            int(len(unique)))


def clustered_loss_difference(first: dict | None, second: dict | None,
                              frame: pd.DataFrame, categories: int, products: int,
                              seed: int, replicates: int = 5000) -> dict:
    _, left = conditional_loss(first, frame, categories, products)
    _, right = conditional_loss(second, frame, categories, products)
    totals = frame.total_purchases.to_numpy(np.float64)
    observed = float((left.sum() - right.sum()) / totals.sum())
    generator = np.random.default_rng(seed)
    intervals = {}
    clusters = {}
    cluster_labels = {
        "product": frame.item_id.to_numpy(np.int64),
        "store": frame.store_id.to_numpy(np.int64),
        "week": frame.WEEK_NO.to_numpy(np.int64),
    }
    for name, labels in cluster_labels.items():
        interval, count = _cluster_bootstrap_interval(
            left, right, totals, labels, generator, replicates)
        intervals[name] = interval
        clusters[name] = count
    return {
        "first_minus_second_log_loss": observed,
        "cluster_bootstrap_95_intervals": intervals,
        "cluster_counts": clusters,
        "bootstrap_replicates": replicates,
    }


def serializable_model(model: dict, design: dict) -> dict:
    categories = len(design["category_values"])
    products = len(design["product_values"])
    global_sensitivity, category, product = unpack(
        model["parameters"], model["level"], categories, products)
    result = {key: value for key, value in model.items() if key != "parameters"}
    result["global_sensitivity"] = float(global_sensitivity)
    if category is not None:
        result["category_sensitivity"] = [
            {"cat_id": int(cat), "sensitivity": float(value)}
            for cat, value in zip(design["category_values"], category)]
    if product is not None:
        result["product_sensitivity"] = [
            {"item_id": int(item), "sensitivity": float(value)}
            for item, value in zip(design["product_values"], product)]
    return result


def model_summary(model: dict, design: dict) -> dict:
    categories = len(design["category_values"])
    products = len(design["product_values"])
    global_sensitivity, category, product = unpack(
        model["parameters"], model["level"], categories, products)
    active = (np.asarray([global_sensitivity]) if model["level"] == "global"
              else category if model["level"] == "category" else product)
    return {
        "level": model["level"],
        "category_penalty": model["category_penalty"],
        "product_penalty": model["product_penalty"],
        "penalized_training_loss": model["penalized_training_loss"],
        "validation_loss": model.get("validation_loss"),
        "iterations": model["iterations"],
        "refinement_attempts": model["refinement_attempts"],
        "gradient_max": model["gradient_max"],
        "projected_gradient_max": model["projected_gradient_max"],
        "global_sensitivity": float(global_sensitivity),
        "active_sensitivity_quantiles": {
            str(q): float(np.quantile(active, q)) for q in (0, .1, .5, .9, 1)},
        "coefficients_at_zero_bound": int(np.sum(active <= 1e-10)),
        "coefficients_at_upper_bound": int(np.sum(active >= 5.0 - 1e-10)),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--coefficients-output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=68101)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events, design = prepare_events()
    training = events[events.split == "train"]
    validation = events[events.split == "validation"]
    test = events[events.split == "test"]
    categories = len(design["category_values"])
    products = len(design["product_values"])

    print(f"[supported-price] train={len(training)} validation={len(validation)} "
          f"test={len(test)} products={products} categories={categories}", flush=True)

    global_model = fit_model(
        training, "global", categories, products, design["product_category"])
    global_value = float(global_model["parameters"][0])
    grid = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
    category_models = []
    for penalty in grid:
        print(f"[supported-price] fit category penalty={penalty:g}", flush=True)
        model = fit_model(
            training, "category", categories, products, design["product_category"],
            category_penalty=penalty,
            initial=initial_parameters("category", categories, products,
                                       design["product_category"], global_value))
        model["validation_loss"] = conditional_loss(
            model, validation, categories, products)[0]
        category_models.append(model)
    category_model = min(category_models, key=lambda row: row["validation_loss"])

    product_models = []
    for category_source in category_models:
        _, category_initial, _ = unpack(
            category_source["parameters"], "category", categories, products)
        for product_penalty in grid:
            category_penalty = category_source["category_penalty"]
            print(f"[supported-price] fit product category_penalty="
                  f"{category_penalty:g} product_penalty={product_penalty:g}",
                  flush=True)
            model = fit_model(
                training, "product", categories, products,
                design["product_category"],
                category_penalty=category_penalty,
                product_penalty=product_penalty,
                initial=initial_parameters(
                    "product", categories, products, design["product_category"],
                    global_value, category_initial))
            model["validation_loss"] = conditional_loss(
                model, validation, categories, products)[0]
            product_models.append(model)
    product_model = min(product_models, key=lambda row: row["validation_loss"])
    global_model["validation_loss"] = conditional_loss(
        global_model, validation, categories, products)[0]
    candidates = [global_model, category_model, product_model]
    selected = min(candidates, key=lambda row: row["validation_loss"])

    report = {
        "status": "completed",
        "claim_level": "observational_conditional_price_response",
        "estimand": "within-store product incidence rate response conditional on observed trips",
        "implementation_sha256": file_sha256(Path(__file__).resolve()),
        "selection": {
            "training_events": int(len(training)),
            "validation_events": int(len(validation)),
            "locked_test_events": int(len(test)),
            "training_products": products,
            "training_categories": categories,
            "penalty_grid": list(grid),
            "validation_selected_level": selected["level"],
            "selected_category_penalty": selected["category_penalty"],
            "selected_product_penalty": selected["product_penalty"],
            "rule": "penalties and hierarchy selected only by validation conditional log loss",
            "data_support_audit": {
                "source_summary": design["source_summary"],
                "minimum_product_training_lines":
                    design["minimum_product_training_lines"],
                "eligible_event_counts_before_training_price_support":
                    design["eligible_event_counts_before_training_price_support"],
                "event_counts_after_training_price_support":
                    design["event_counts_after_training_price_support"],
                "event_counts_after_informative_training_product_support":
                    design["event_counts_after_informative_training_product_support"],
                "heldout_events_removed_outside_eligible_training_price_support":
                    design[
                        "heldout_events_removed_outside_eligible_training_price_support"],
                "heldout_events_removed_without_informative_training_product":
                    design[
                        "heldout_events_removed_without_informative_training_product"],
            },
        },
        "validation_grid": {
            "global": model_summary(global_model, design),
            "category": [model_summary(model, design) for model in category_models],
            "product": [model_summary(model, design) for model in product_models],
        },
        "metrics": {},
        "signed_global_evidence": {},
    }
    for split, frame in (("train", training), ("validation", validation)):
        report["metrics"][split] = {
            "no_price": response_metrics(None, frame, categories, products),
            "global": response_metrics(global_model, frame, categories, products),
            "best_category": response_metrics(
                category_model, frame, categories, products),
            "best_product": response_metrics(product_model, frame, categories, products),
            "selected": response_metrics(selected, frame, categories, products),
        }
        report["signed_global_evidence"][split] = unconstrained_global_evidence(frame)
    report["metrics"]["test"] = {
        "no_price": response_metrics(None, test, categories, products),
        "global": response_metrics(global_model, test, categories, products),
        "selected": response_metrics(selected, test, categories, products),
    }
    report["signed_global_evidence"]["test"] = unconstrained_global_evidence(test)
    report["locked_test_comparisons"] = {
        "global_vs_no_price": clustered_loss_difference(
            global_model, None, test, categories, products, args.seed - 1),
        "selected_vs_no_price": clustered_loss_difference(
            selected, None, test, categories, products, args.seed),
        "selected_vs_global": clustered_loss_difference(
            selected, global_model, test, categories, products, args.seed + 1),
    }
    selected_test = report["locked_test_comparisons"]
    global_no_price = selected_test["global_vs_no_price"]
    global_no_price_intervals = global_no_price["cluster_bootstrap_95_intervals"]
    no_price_intervals = selected_test["selected_vs_no_price"][
        "cluster_bootstrap_95_intervals"]
    global_intervals = selected_test["selected_vs_global"][
        "cluster_bootstrap_95_intervals"]
    report["gates"] = {
        "global_beats_no_price_on_test":
            global_no_price["first_minus_second_log_loss"] < 0,
        "global_improvement_vs_no_price_is_robust_to_product_store_and_week":
            all(interval[1] < 0 for interval in global_no_price_intervals.values()),
        "selected_beats_no_price_on_test":
            selected_test["selected_vs_no_price"]["first_minus_second_log_loss"] < 0,
        "selected_beats_global_on_test":
            selected_test["selected_vs_global"]["first_minus_second_log_loss"] <= 0,
        "selected_improvement_vs_no_price_is_robust_to_product_store_and_week":
            all(interval[1] < 0 for interval in no_price_intervals.values()),
        "selected_complexity_improvement_vs_global_is_robust_to_product_store_and_week":
            selected["level"] == "global" or all(
                interval[1] < 0 for interval in global_intervals.values()),
    }
    global_supported = bool(
        report["gates"]["global_beats_no_price_on_test"]
        and report["gates"][
            "global_improvement_vs_no_price_is_robust_to_product_store_and_week"])
    complexity_supported = bool(
        selected["level"] == "global" or (
            report["gates"]["selected_beats_global_on_test"]
            and report["gates"][
                "selected_complexity_improvement_vs_global_is_robust_to_product_store_and_week"]))
    selected_supported = bool(
        report["gates"]["selected_beats_no_price_on_test"]
        and report["gates"][
            "selected_improvement_vs_no_price_is_robust_to_product_store_and_week"])
    certified = selected if selected_supported \
        else global_model if global_supported else None
    report["global_price_signal_supported"] = global_supported
    report["validation_selected_predictor_supported"] = selected_supported
    report["validation_selected_complexity_supported"] = complexity_supported
    report["individual_coefficient_interpretation_supported"] = False
    report["certified_level"] = certified["level"] if certified is not None else None
    report["passed"] = certified is not None
    report["interpretation"] = (
        "No price component passed the locked-test support gate. The fitted coefficients "
        "must not be loaded into the basket model or reported as price effects."
        if certified is None else
        "The certified hierarchy is an observational price predictor as a whole, not a "
        "causal effect. Individual coefficient differences are not independently supported "
        "and must not be reported as SKU-specific elasticities.")

    coefficient_path = args.coefficients_output.resolve()
    coefficient_path.parent.mkdir(parents=True, exist_ok=True)
    if certified is None:
        coefficient_path.write_text(strict_json_dumps({
            "status": "not_certified",
            "reason": "no price component passed the locked-test support gate",
        }))
    else:
        coefficient_payload = serializable_model(certified, design)
        coefficient_payload.update({
            "status": "certified_observational_predictor",
            "individual_coefficient_interpretation_supported": False,
            "validation_selected_complexity_supported": complexity_supported,
            "training_product_count": products,
            "supported_item_ids": [int(value) for value in design["product_values"]],
            "unsupported_catalogue_products_use_global_sensitivity": False,
        })
        coefficient_path.write_text(strict_json_dumps(coefficient_payload))
    report["coefficients_output"] = str(coefficient_path)
    report["coefficients_sha256"] = file_sha256(coefficient_path)
    report["coefficients_certified"] = certified is not None
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(strict_json_dumps(report))
    print(f"[supported-price] validation_selected={selected['level']} "
          f"certified={certified['level'] if certified is not None else 'none'} "
          f"validation_loss={selected['validation_loss']:.9f} "
          f"test_delta_vs_no_price="
          f"{selected_test['selected_vs_no_price']['first_minus_second_log_loss']:.9g} "
          f"passed={report['passed']}", flush=True)
    print(f"[supported-price] report={output}", flush=True)
    print(f"[supported-price] coefficients={coefficient_path}", flush=True)
    if certified is None:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
