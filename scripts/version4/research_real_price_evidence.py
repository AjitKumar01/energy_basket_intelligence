#!/usr/bin/env python3
"""Audit whether the real panel contains learnable held-out price-response evidence.

This is deliberately an observational audit.  It improves on the old chain-week
comparison by holding the product and store fixed, keeping selection independent of
held-out purchase outcomes, fitting a pooled response only on training weeks, and
evaluating the frozen fitted basket law on validation and test events.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

os.environ.setdefault("V3_AFFINITY", "1")

from checkpoint_io import ROOT, load_checkpoint
from data import build
from evaluate_observational_price_response import event_prediction
from features import Features
from fit import Batcher
from provenance import file_sha256, strict_json_dumps


torch.set_default_dtype(torch.float64)


def quantiles(values) -> dict:
    x = np.asarray(values, dtype=np.float64)
    x = x[np.isfinite(x)]
    return {str(q): float(np.quantile(x, q)) for q in (0, .1, .25, .5, .75, .9, 1)}


def internal_store_map(baskets: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_parquet(ROOT / "data/tx.parquet", columns=["BASKET_ID", "STORE_ID"])
    raw = raw.drop_duplicates("BASKET_ID")
    mapping = baskets[["BASKET_ID", "store_id"]].drop_duplicates().merge(
        raw, on="BASKET_ID", how="left", validate="one_to_one")
    if mapping.STORE_ID.isna().any():
        raise RuntimeError("a modeled basket has no raw store identifier")
    pairs = mapping[["store_id", "STORE_ID"]].drop_duplicates()
    if pairs.groupby("store_id").STORE_ID.nunique().max() != 1:
        raise RuntimeError("internal store identifiers do not map one-to-one")
    return pairs


def build_store_events() -> tuple[pd.DataFrame, dict]:
    baskets = pd.read_parquet(
        ROOT / "basket_input/baskets.parquet",
        columns=["BASKET_ID", "WEEK_NO", "item_id", "store_id"])
    items = pd.read_parquet(
        ROOT / "basket_input/items.parquet",
        columns=["PRODUCT_ID", "item_id", "n_train_lines"])
    stores = internal_store_map(baskets)
    prices = pd.read_parquet(ROOT / "data/price_store_week.parquet")
    prices = prices.merge(items, on="PRODUCT_ID", how="inner", validate="many_to_one")
    prices = prices.merge(stores, on="STORE_ID", how="inner", validate="many_to_one")

    trips = baskets[["BASKET_ID", "store_id", "WEEK_NO"]].drop_duplicates()
    denominators = trips.groupby(["store_id", "WEEK_NO"]).size().rename("trips").reset_index()
    purchases = (baskets.groupby(["store_id", "item_id", "WEEK_NO"])
                 .BASKET_ID.nunique().rename("purchases").reset_index())
    prices = prices.merge(
        denominators, on=["store_id", "WEEK_NO"], how="inner", validate="many_to_one")
    prices = prices.merge(
        purchases, on=["store_id", "item_id", "WEEK_NO"], how="left",
        validate="one_to_one")
    prices["purchases"] = prices.purchases.fillna(0).astype(np.int64)
    prices = prices.sort_values(["store_id", "item_id", "WEEK_NO"], kind="stable")
    grouped = prices.groupby(["store_id", "item_id"], sort=False)
    for column in ("WEEK_NO", "price", "base_price", "n_tx", "purchases", "trips"):
        prices[f"previous_{column}"] = grouped[column].shift()
    prices["week_gap"] = prices.WEEK_NO - prices.previous_WEEK_NO
    prices["log_price_change"] = np.log(prices.price) - np.log(prices.previous_price)
    prices["log_base_price_change"] = (
        np.log(prices.base_price) - np.log(prices.previous_base_price))
    prices["promotion_depth"] = 1.0 - prices.price / prices.base_price
    prices["previous_promotion_depth"] = (
        1.0 - prices.previous_price / prices.previous_base_price)
    prices["promotion_depth_change"] = (
        prices.promotion_depth - prices.previous_promotion_depth)
    events = prices[
        (prices.week_gap == 1)
        & (prices.log_price_change.abs() >= math.log(1.01))
        & (prices.previous_trips > 0)
    ].copy()
    events["split"] = np.where(
        events.WEEK_NO >= 91, "test",
        np.where(events.WEEK_NO >= 83, "validation", "train"))
    events["observed_before_incidence"] = events.previous_purchases / events.previous_trips
    events["observed_after_incidence"] = events.purchases / events.trips
    events["observed_response"] = (
        events.observed_after_incidence - events.observed_before_incidence)
    n_stores = int(json.loads((ROOT / "basket_input/meta.json").read_text())["n_stores"])
    with np.load(ROOT / "basket_input/promo.npz") as promotion:
        keys = promotion["keys"]

        def lookup(values, week):
            query = ((events.item_id.to_numpy(np.int64) * n_stores
                      + events.store_id.to_numpy(np.int64)) * 128
                     + week.to_numpy(np.int64))
            position = np.searchsorted(keys, query)
            safe = np.minimum(position, len(keys) - 1)
            found = (position < len(keys)) & (keys[safe] == query)
            result = np.zeros(len(events), dtype=values.dtype)
            result[found] = values[position[found]]
            return result

        for name, values in (("display", promotion["disp"]),
                             ("mailer", promotion["mail"])):
            events[name] = lookup(values, events.WEEK_NO)
            events[f"previous_{name}"] = lookup(values, events.previous_WEEK_NO)
            events[f"{name}_changed"] = events[name] != events[f"previous_{name}"]
    support = (events[events.split == "train"].groupby("item_id").log_price_change
               .agg(training_change_low="min", training_change_high="max",
                    training_store_events="size"))
    events = events.join(support, on="item_id")
    report = {
        "rows": int(len(events)),
        "products": int(events.item_id.nunique()),
        "stores": int(events.store_id.nunique()),
        "counts_by_split": {
            str(k): int(v) for k, v in events.split.value_counts().items()},
        "marketing_change_fraction": {
            split: {
                "display": float(frame.display_changed.mean()),
                "mailer": float(frame.mailer_changed.mean()),
            }
            for split, frame in events.groupby("split")
        },
        "purchases_per_event_side": {
            "previous": quantiles(events.previous_purchases),
            "current": quantiles(events.purchases),
        },
        "store_trips_per_event_side": {
            "previous": quantiles(events.previous_trips),
            "current": quantiles(events.trips),
        },
    }
    return events, report


def eligible_store_events(events: pd.DataFrame) -> pd.Series:
    return (
        (events.promotion_depth_change.abs() < .01)
        & (~events.display_changed)
        & (~events.mailer_changed)
        & (events.log_price_change.abs() >= math.log(1.05))
        & (events.log_price_change.abs() <= .70)
        & (events.n_train_lines >= 500)
        & (events.trips >= 16)
        & (events.previous_trips >= 16)
        & events.training_change_low.notna()
        & (events.log_price_change >= events.training_change_low)
        & (events.log_price_change <= events.training_change_high)
    )


def pooled_slope(frame: pd.DataFrame) -> float:
    x = frame.log_price_change.to_numpy(np.float64)
    y = frame.observed_response.to_numpy(np.float64)
    return float(x @ y / (x @ x))


def clustered_slope(frame: pd.DataFrame) -> dict:
    """Through-origin price slope with product-clustered sandwich uncertainty."""
    x = frame.log_price_change.to_numpy(np.float64)
    y = frame.observed_response.to_numpy(np.float64)
    denominator = float(x @ x)
    beta = float(x @ y / denominator)
    residual = y - beta * x
    score = pd.Series(x * residual).groupby(frame.item_id.to_numpy()).sum().to_numpy()
    clusters = len(score)
    correction = clusters / max(clusters - 1, 1)
    standard_error = math.sqrt(correction * float(score @ score) / denominator ** 2)
    return {
        "slope": beta,
        "product_cluster_standard_error": standard_error,
        "95_interval": [beta - 1.96 * standard_error, beta + 1.96 * standard_error],
        "product_clusters": clusters,
    }


def observational_metrics(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    observed = frame.observed_response.to_numpy(np.float64)
    predicted = np.asarray(prediction, dtype=np.float64)
    nonzero = (observed != 0) & (predicted != 0)
    pearson = float(np.corrcoef(predicted, observed)[0, 1]) if len(frame) > 1 else None
    rank = float(spearmanr(predicted, observed).statistic) if len(frame) > 1 else None
    return {
        "events": int(len(frame)),
        "products": int(frame.item_id.nunique()),
        "stores": int(frame.store_id.nunique()),
        "pearson": pearson,
        "spearman": rank,
        "sign_agreement": float(np.mean(
            np.sign(predicted[nonzero]) == np.sign(observed[nonzero]))) if nonzero.any() else None,
        "observed_mae_if_predict_zero": float(np.mean(np.abs(observed))),
        "prediction_mae": float(np.mean(np.abs(observed - predicted))),
        "observed_mse_if_predict_zero": float(np.mean(observed ** 2)),
        "prediction_mse": float(np.mean((observed - predicted) ** 2)),
        "predicted_response": quantiles(predicted),
        "observed_response": quantiles(observed),
    }


def split_evidence(events: pd.DataFrame, mask: pd.Series) -> dict:
    selected = events[mask]
    training = selected[selected.split == "train"]
    beta = pooled_slope(training)
    result = {"training_only_pooled_slope": beta, "splits": {}}
    for split in ("train", "validation", "test"):
        frame = selected[selected.split == split]
        metrics = observational_metrics(frame, beta * frame.log_price_change.to_numpy())
        metrics["observed_price_slope"] = clustered_slope(frame)
        metrics["mean_observed_after_price_rise"] = float(
            frame.loc[frame.log_price_change > 0, "observed_response"].mean())
        metrics["mean_observed_after_price_cut"] = float(
            frame.loc[frame.log_price_change < 0, "observed_response"].mean())
        result["splits"][split] = metrics
    return result


def product_specific_stability(events: pd.DataFrame, mask: pd.Series,
                               model_coefficient: np.ndarray,
                               seed: int) -> dict:
    slopes = {}
    for split in ("train", "validation", "test"):
        frame = events[mask & (events.split == split)]

        def summarize(group):
            x = group.log_price_change.to_numpy(np.float64)
            y = group.observed_response.to_numpy(np.float64)
            return pd.Series({"slope": float(x @ y / (x @ x)),
                              "events": int(len(group))})

        slopes[split] = frame.groupby("item_id").apply(
            summarize, include_groups=False)
    combined = (slopes["train"].add_suffix("_train")
                .join(slopes["validation"].add_suffix("_validation"), how="inner")
                .join(slopes["test"].add_suffix("_test"), how="inner"))
    combined["model_coefficient"] = model_coefficient[
        combined.index.to_numpy(np.int64)]
    result = {"products_observed_in_all_splits": int(len(combined)),
              "minimum_events_per_split": {}}
    for minimum in (1, 3, 5, 8, 10):
        eligible = combined[
            (combined.events_train >= minimum)
            & (combined.events_validation >= minimum)
            & (combined.events_test >= minimum)]
        result["minimum_events_per_split"][str(minimum)] = {
            "products": int(len(eligible)),
            "training_validation_slope_correlation": float(
                eligible.slope_train.corr(eligible.slope_validation)),
            "training_test_slope_correlation": float(
                eligible.slope_train.corr(eligible.slope_test)),
            "validation_test_slope_correlation": float(
                eligible.slope_validation.corr(eligible.slope_test)),
            "negative_slope_fraction": {
                split: float((eligible[f"slope_{split}"] < 0).mean())
                for split in ("train", "validation", "test")},
            "model_coefficient_correlation_with_negative_slope": {
                split: float(eligible.model_coefficient.corr(
                    -eligible[f"slope_{split}"]))
                for split in ("train", "validation", "test")},
        }
    # Resample products, not events, for the prespecified five-event support panel.
    eligible = combined[
        (combined.events_train >= 5)
        & (combined.events_validation >= 5)
        & (combined.events_test >= 5)]
    generator = np.random.default_rng(seed)
    intervals = {}
    for first, second, name in (
            ("slope_train", "slope_validation", "training_validation"),
            ("slope_train", "slope_test", "training_test"),
            ("slope_validation", "slope_test", "validation_test")):
        left = eligible[first].to_numpy()
        right = eligible[second].to_numpy()
        values = []
        for _ in range(5000):
            chosen = generator.integers(0, len(eligible), len(eligible))
            values.append(np.corrcoef(left[chosen], right[chosen])[0, 1])
        intervals[name] = [float(x) for x in np.nanquantile(values, [.025, .975])]
    result["five_event_product_bootstrap_95_intervals"] = intervals
    return result


def old_chain_audit() -> dict:
    events = pd.read_parquet(
        ROOT / "artifacts/remaining_verification_audited_20260914/price_events.parquet")
    clean = events[
        (events.event_split == "test")
        & (~events.concurrent_promotion_change)
        & (events.weeks_since_previous_observation == 1)
    ].copy()
    store = pd.read_parquet(
        ROOT / "data/price_store_week.parquet",
        columns=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "price"])
    current = clean[["event_id", "PRODUCT_ID", "previous_week", "WEEK_NO",
                     "log_price_change"]].merge(
        store, on=["PRODUCT_ID", "WEEK_NO"], how="left")
    previous = store.rename(columns={"WEEK_NO": "previous_week", "price": "previous_price"})
    paired = current.merge(
        previous, on=["PRODUCT_ID", "STORE_ID", "previous_week"], how="inner")
    paired["store_log_change"] = np.log(paired.price) - np.log(paired.previous_price)
    paired["same_direction"] = (
        np.sign(paired.store_log_change) == np.sign(paired.log_price_change))
    by_event = paired.groupby("event_id").agg(
        paired_stores=("STORE_ID", "nunique"),
        store_direction_agreement=("same_direction", "mean"))
    joined = clean.join(by_event, on="event_id")

    evaluated = pd.read_parquet(
        ROOT / "artifacts/remaining_verification_audited_20260914/real_price_response_events.parquet")
    evaluated = evaluated.join(by_event, on="event_id")
    p0 = evaluated.observed_before_incidence.to_numpy()
    p1 = evaluated.observed_after_incidence.to_numpy()
    n0 = evaluated.before_contexts.to_numpy()
    n1 = evaluated.after_contexts.to_numpy()
    standard_error = np.sqrt(p0 * (1 - p0) / n0 + p1 * (1 - p1) / n1)
    signal_to_noise = np.abs(evaluated.child_predicted_response.to_numpy()) / standard_error
    return {
        "clean_test_chain_events": int(len(clean)),
        "events_with_any_same_store_observed_in_both_weeks": int(
            joined.paired_stores.notna().sum()),
        "fraction_with_any_same_store": float(joined.paired_stores.notna().mean()),
        "median_store_direction_agreement_when_observed": float(
            joined.store_direction_agreement.median()),
        "old_evaluation": {
            "events": int(len(evaluated)),
            "events_with_any_same_store": int(evaluated.paired_stores.notna().sum()),
            "median_product_purchases_before": float(
                np.median(np.rint(p0 * n0))),
            "median_product_purchases_after": float(
                np.median(np.rint(p1 * n1))),
            "naive_observed_response_standard_error": quantiles(standard_error),
            "absolute_model_response_over_naive_standard_error": quantiles(signal_to_noise),
            "events_with_model_response_above_point_two_standard_errors": int(
                np.sum(signal_to_noise > .2)),
            "median_absolute_observed_over_model_response": float(
                np.median(np.abs(evaluated.observed_association))
                / np.median(np.abs(evaluated.child_predicted_response))),
            "child_parent_response_maximum_absolute_difference": float(np.max(np.abs(
                evaluated.child_predicted_response
                - evaluated.parent_predicted_response))),
        },
    }


def select_model_events(events: pd.DataFrame, mask: pd.Series, split: str,
                        count: int, seed: int) -> pd.DataFrame:
    frame = events[mask & (events.split == split)].copy()
    # The random order and one-event-per-product rule use no held-out purchase outcome.
    order = np.random.default_rng(seed).permutation(len(frame))
    frame = frame.iloc[order].drop_duplicates("item_id").head(count).copy()
    if len(frame) < count:
        raise RuntimeError(f"only {len(frame)} independent-product {split} events available")
    return frame


def evaluate_fitted_model(events: pd.DataFrame, mask: pd.Series, run_dir: Path,
                          count: int, contexts: int, particles: int,
                          seed: int) -> tuple[pd.DataFrame, dict]:
    data = build()
    parent, _, meta = load_checkpoint(
        run_dir / "out/v3_pipeline_additive_best.pt", data,
        required_capabilities=("conditional_nonempty_incidence",))
    child, child_blob, _ = load_checkpoint(
        run_dir / "artifacts/candidate_rank1.pt", data,
        required_capabilities=("conditional_nonempty_incidence", "gram_interactions"))
    for model in (parent, child):
        for parameter in model.parameters():
            parameter.requires_grad_(False)
    batcher = Batcher(
        data, Features(int(data["n_item"]), int(data["n_store"]), 712,
                       include_recency=False),
        int(meta["nmax"]), include_recency=False)
    beta = pooled_slope(events[mask & (events.split == "train")])
    rows = []
    for split_index, split in enumerate(("validation", "test")):
        selected = select_model_events(
            events, mask, split, count, seed + 1009 * split_index)
        for number, row in enumerate(selected.itertuples(index=False), start=1):
            candidate_trips = np.flatnonzero(
                (data["trip_store"] == int(row.store_id))
                & (data["trip_week"] == int(row.previous_WEEK_NO)))
            generator = np.random.default_rng(seed + 100003 * int(row.item_id)
                                              + 1009 * int(row.WEEK_NO))
            chosen = candidate_trips[generator.permutation(len(candidate_trips))[:contexts]]
            prediction = event_prediction(
                parent, child, batcher, data, chosen, int(row.item_id),
                float(row.log_price_change), particles,
                seed + 1000003 * int(row.item_id) + int(row.WEEK_NO))
            p0 = float(row.observed_before_incidence)
            p1 = float(row.observed_after_incidence)
            standard_error = math.sqrt(
                p0 * (1 - p0) / float(row.previous_trips)
                + p1 * (1 - p1) / float(row.trips))
            rows.append({
                "split": split,
                "item_id": int(row.item_id),
                "PRODUCT_ID": int(row.PRODUCT_ID),
                "store_id": int(row.store_id),
                "previous_week": int(row.previous_WEEK_NO),
                "week": int(row.WEEK_NO),
                "log_price_change": float(row.log_price_change),
                "previous_price": float(row.previous_price),
                "price": float(row.price),
                "previous_trips": int(row.previous_trips),
                "trips": int(row.trips),
                "previous_purchases": int(row.previous_purchases),
                "purchases": int(row.purchases),
                "observed_before_incidence": p0,
                "observed_after_incidence": p1,
                "observed_response": float(row.observed_response),
                "observed_response_naive_se": standard_error,
                "training_pooled_prediction": beta * float(row.log_price_change),
                "parent_prediction": prediction["parent"],
                "child_prediction": prediction["child"],
                "minimum_absolute_ess": prediction["minimum_absolute_ess"],
                "model_runtime_seconds": prediction["runtime_seconds"]["total"],
            })
            print(f"[real-price-research] {split} {number}/{count} "
                  f"item={int(row.item_id)}", flush=True)
    result = pd.DataFrame(rows)
    report = {
        "selection": {
            "rule": ("same product and store; adjacent observed weeks; at least 5% and at "
                     "most exp(0.7)-1 price change; promotion depth unchanged within 1pp; "
                     "display and mailer status unchanged; "
                     "at least 500 training purchases; at least 16 modeled trips per side; "
                     "test change inside product's training support; random one event per "
                     "product; no held-out outcome used for selection"),
            "events_per_split": count,
            "contexts_per_event": contexts,
            "particles": particles,
        },
        "training_only_pooled_slope": beta,
        "splits": {},
        "child_parent_maximum_absolute_difference": float(np.max(np.abs(
            result.child_prediction - result.parent_prediction))),
        "minimum_absolute_ess": float(result.minimum_absolute_ess.min()),
        "total_model_runtime_seconds": float(result.model_runtime_seconds.sum()),
        "checkpoint_sha256": file_sha256(run_dir / "artifacts/candidate_rank1.pt"),
        "data_fingerprint_sha256": child_blob["data_fingerprint_sha256"],
    }
    for split in ("validation", "test"):
        frame = result[result.split == split]
        report["splits"][split] = {
            "training_pooled_baseline": observational_metrics(
                frame, frame.training_pooled_prediction.to_numpy()),
            "parent_model": observational_metrics(frame, frame.parent_prediction.to_numpy()),
            "interaction_model": observational_metrics(
                frame, frame.child_prediction.to_numpy()),
            "model_signal_over_naive_observation_se": quantiles(
                np.abs(frame.child_prediction) / frame.observed_response_naive_se),
        }
    with torch.no_grad():
        gamma = torch.nn.functional.softplus(child.gamma).mean(0)
        beta_item = torch.nn.functional.softplus(child.beta)
        coefficient = (beta_item * gamma).sum(-1).cpu().numpy()
        kappa = float(torch.nn.functional.softplus(child.price_kappa))
    report["fitted_price_parameters"] = {
        "relative_price_scale_kappa": kappa,
        "mean_household_product_coefficient": quantiles(coefficient),
        "interpretation": (
            "the overall price level is penalty-calibrated; product allocation is learned "
            "from observational basket likelihood and is not externally identified"),
    }
    report["product_specific_observational_stability"] = product_specific_stability(
        events, mask, coefficient * kappa, seed)
    return result, report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path,
                        default=ROOT / "artifacts/corrected_complete_rank5_20260913")
    parser.add_argument("--events-per-split", type=int, default=80)
    parser.add_argument("--contexts-per-event", type=int, default=16)
    parser.add_argument("--particles", type=int, default=64)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=67101)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-event-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.threads)
    run_dir = args.run_dir.resolve()
    events, store_summary = build_store_events()
    mask = eligible_store_events(events)
    report = {
        "status": "completed",
        "claim_level": "observational_learnability_audit_not_causal_identification",
        "implementation_sha256": file_sha256(Path(__file__).resolve()),
        "old_chain_design": old_chain_audit(),
        "same_store_event_population": store_summary,
        "same_store_training_to_heldout_evidence": split_evidence(events, mask),
    }
    per_event, model_report = evaluate_fitted_model(
        events, mask, run_dir, args.events_per_split, args.contexts_per_event,
        args.particles, args.seed)
    per_path = args.per_event_output.resolve()
    per_path.parent.mkdir(parents=True, exist_ok=True)
    per_event.to_parquet(per_path, index=False)
    report["fitted_model_heldout_comparison"] = model_report
    report["per_event_output"] = str(per_path)
    report["per_event_sha256"] = file_sha256(per_path)
    report["limitations"] = [
        "store prices are reconstructed from transactions rather than a posted-price feed",
        "price assignment is not randomized and may share advertising, inventory, or time shocks",
        "the outcome is product incidence conditional on an observed nonempty trip",
        "same-store price availability is still selected by a transaction in the broader raw panel",
        "weekly product outcomes are sparse, so event-level comparisons have low power",
        "the audit can establish observational repeatability and learnability, not causal effects",
    ]
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(strict_json_dumps(report))
    print(strict_json_dumps(report), end="", flush=True)


if __name__ == "__main__":
    main()
