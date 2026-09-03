#!/usr/bin/env python3
"""Finite-horizon, budget-constrained promotion MDP for three customer segments.

The retailer observes a fixed promotion horizon and remaining markdown budget.  Each day
it may run no promotion or target one segment with a discount on a small product bundle.
Bundles are selected from training outcomes only.  The fitted Version-4 law evaluates
the basket response on held-out contexts using common SMC particles.

State
    (days remaining, promotion budget remaining)

Action
    no promotion, or (customer segment, five-SKU bundle, discount rate)

Budget cost
    expected markdown paid on promoted products that the model says will be purchased

Reward
    incremental basket value at undiscounted shelf prices.  Actual post-discount sales
    are also reported, but profit is not identified because wholesale costs and store-
    visit probabilities are absent from the data/model contract.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from audit_particle_counterfactual_generation import (
    ROOT, copied_context, load_checkpoint, particle_delta)
from data import build
from features import Features
from fit import Batcher
from interaction_particles import rao_blackwell_particle_statistics
from tempered_ais import annealed_smc_logz


torch.set_default_dtype(torch.float64)


def balanced_context_panel(data, labels: np.ndarray, segment: int,
                           count: int, seed: int, nmax: int) -> np.ndarray:
    """Select held-out contexts round-robin across segment households."""
    eligible = np.flatnonzero(
        (data["trip_split"] == 2)
        & (data["trip_nlines"] >= 1)
        & (data["trip_nlines"] <= nmax)
        & (labels[data["trip_user"]] == segment))
    grouped: dict[int, list[int]] = {}
    for trip in eligible:
        grouped.setdefault(int(data["trip_user"][trip]), []).append(int(trip))
    rng = np.random.default_rng(seed + 1009 * segment)
    households = np.asarray(sorted(grouped), dtype=np.int64)
    households = households[rng.permutation(len(households))]
    for household in households:
        values = np.asarray(grouped[int(household)], dtype=np.int64)
        grouped[int(household)] = values[rng.permutation(len(values))].tolist()
    selected: list[int] = []
    position = {int(household): 0 for household in households}
    target = min(count, len(eligible))
    while len(selected) < target:
        changed = False
        for household_value in households:
            household = int(household_value)
            if position[household] < len(grouped[household]):
                selected.append(grouped[household][position[household]])
                position[household] += 1
                changed = True
                if len(selected) == target:
                    break
        if not changed:
            break
    return np.asarray(selected, dtype=np.int64)


def training_product_bundles(data, labels: np.ndarray, metadata: pd.DataFrame,
                             segment: int, bundle_count: int,
                             products_per_bundle: int) -> list[dict]:
    """High-evidence category bundles, selected without validation/test outcomes."""
    item_category = metadata.cat_id.to_numpy(dtype=np.int64)
    line_trip = np.repeat(np.arange(len(data["trip_nlines"])), data["trip_nlines"])
    train_line = data["trip_split"][line_trip] == 0
    segment_line = train_line & (
        labels[data["trip_user"][line_trip]] == segment)
    global_category = np.bincount(
        item_category[data["line_item"][train_line]],
        minlength=int(data["n_cat"])).astype(np.float64)
    local_category = np.bincount(
        item_category[data["line_item"][segment_line]],
        minlength=int(data["n_cat"])).astype(np.float64)
    global_probability = (global_category + 1.0 / len(global_category)) \
        / (global_category.sum() + 1.0)
    local_probability = (local_category + 1.0 / len(local_category)) \
        / (local_category.sum() + 1.0)
    minimum = max(20, int(math.ceil(0.001 * local_category.sum())))
    log_overindex = np.log(local_probability / global_probability)
    score = np.where(
        local_category >= minimum,
        np.maximum(log_overindex, 0.0) * np.sqrt(local_category), -np.inf)
    categories = np.argsort(score)[::-1]
    categories = [int(c) for c in categories if np.isfinite(score[c])][:bundle_count]
    local_items = data["line_item"][segment_line]
    item_count = np.bincount(local_items, minlength=int(data["n_item"]))
    bundles = []
    for category in categories:
        members = np.flatnonzero(item_category == category)
        order = members[np.argsort(item_count[members], kind="stable")[::-1]]
        products = order[:products_per_bundle].astype(np.int64)
        rows = metadata.iloc[products]
        description = str(rows.COMMODITY_DESC.mode().iloc[0]).strip()
        bundles.append({
            "category": category,
            "category_description": description,
            "training_lines": int(local_category[category]),
            "log_overindex": float(log_overindex[category]),
            "products": products.tolist(),
            "product_descriptions": [
                str(value).strip() for value in rows.SUB_COMMODITY_DESC.tolist()],
            "product_training_lines": item_count[products].astype(int).tolist(),
        })
    if len(bundles) != bundle_count:
        raise RuntimeError(f"segment {segment} has only {len(bundles)} eligible bundles")
    return bundles


def solve_budget_mdp(actions: list[dict], horizon: int, budget: float,
                     bins: int, utilization_floor: float) -> dict:
    """Backward dynamic program with an expected-spend budget transition."""
    if budget <= 0.0:
        raise ValueError("budget must be positive")
    width = budget / bins
    cost_bins = []
    for action in actions:
        cost = float(action["daily_expected_markdown_spend"])
        cost_bins.append(0 if cost == 0.0 else max(1, int(math.ceil(cost / width))))
    # Positive costs are rounded *up*, so the realized spend never exceeds the declared
    # budget.  Each of the horizon actions can overstate spend by less than one bin.  Tighten
    # the terminal quantized utilization by ``horizon`` bins; then realized spend is still
    # at least utilization_floor * budget despite the accumulated rounding error.
    maximum_leftover = int(math.floor(
        (1.0 - utilization_floor) * bins - horizon + 1e-12))
    if maximum_leftover < 0:
        raise ValueError(
            "budget grid is too coarse to certify the requested utilization; "
            "increase --budget-bins")
    value = np.full((horizon + 1, bins + 1), -np.inf, dtype=np.float64)
    value[0, :maximum_leftover + 1] = 0.0
    policy = np.full((horizon + 1, bins + 1), -1, dtype=np.int32)
    for remaining_days in range(1, horizon + 1):
        previous = value[remaining_days - 1]
        for remaining_budget in range(bins + 1):
            candidates = []
            for action_index, (action, cost) in enumerate(zip(actions, cost_bins)):
                if cost <= remaining_budget and np.isfinite(
                        previous[remaining_budget - cost]):
                    candidates.append((
                        float(action["daily_incremental_list_value_lcb95"])
                        + previous[remaining_budget - cost], action_index))
            if candidates:
                best, action_index = max(candidates, key=lambda pair: pair[0])
                value[remaining_days, remaining_budget] = best
                policy[remaining_days, remaining_budget] = action_index
    if not np.isfinite(value[horizon, bins]):
        return {
            "feasible": False, "horizon_days": horizon, "budget": budget,
            "budget_bins": bins, "minimum_utilization": utilization_floor,
            "reason": "no action sequence can use the required budget without overspend",
        }

    remaining = bins
    trajectory, counts = [], Counter()
    actual_spend = robust_reward = mean_reward = net_sales = size_lift = 0.0
    for remaining_days in range(horizon, 0, -1):
        action_index = int(policy[remaining_days, remaining])
        action = actions[action_index]
        counts[action["action_id"]] += 1
        trajectory.append({
            "day": horizon - remaining_days + 1,
            "remaining_budget_before": remaining * width,
            "action_id": action["action_id"],
            "segment": action.get("segment"),
            "bundle": action.get("bundle"),
            "discount": action.get("discount", 0.0),
            "expected_markdown_spend": action["daily_expected_markdown_spend"],
            "incremental_list_value_mean": (
                action["daily_incremental_list_value_mean"]),
            "incremental_list_value_lcb95": (
                action["daily_incremental_list_value_lcb95"]),
            "incremental_post_discount_sales": (
                action["daily_incremental_post_discount_sales"]),
        })
        actual_spend += float(action["daily_expected_markdown_spend"])
        robust_reward += float(action["daily_incremental_list_value_lcb95"])
        mean_reward += float(action["daily_incremental_list_value_mean"])
        net_sales += float(action["daily_incremental_post_discount_sales"])
        size_lift += float(action["daily_incremental_distinct_products"])
        remaining -= cost_bins[action_index]
    return {
        "feasible": True,
        "horizon_days": horizon,
        "budget": budget,
        "budget_bins": bins,
        "budget_bin_width": width,
        "minimum_utilization": utilization_floor,
        "quantized_budget_utilization": 1.0 - remaining / bins,
        "expected_markdown_spend": actual_spend,
        "expected_spend_fraction_of_budget": actual_spend / budget,
        "total_robust_incremental_list_value_lcb95": robust_reward,
        "total_incremental_list_value_mean": mean_reward,
        "total_incremental_post_discount_sales": net_sales,
        "total_incremental_distinct_products": size_lift,
        "action_day_counts": dict(counts),
        "daily_policy": trajectory,
    }


@torch.no_grad()
def evaluate_segment_actions(model, batcher, features, data, trips: np.ndarray,
                             bundles: list[dict], discounts: list[float],
                             args, segment: int) -> dict:
    schedule_axis = torch.linspace(0.0, 1.0, args.levels)
    schedule = 1.0 - (1.0 - schedule_axis).pow(args.power)
    absolute_log_price = torch.from_numpy(
        np.load(ROOT / "basket_input" / "log_price.npy").astype(np.float64))
    action_specs = [
        {"bundle": bundle_index, "discount": discount,
         "products": bundle["products"]}
        for bundle_index, bundle in enumerate(bundles)
        for discount in discounts]
    accumulator = [{key: [] for key in (
        "size", "list_value", "actual_sales", "markdown", "bundle_incidence",
        "tail", "ess")} for _ in action_specs]
    baseline = {key: [] for key in ("size", "list_value", "tail")}
    smc_seconds, smc_ess = 0.0, []

    for start in range(0, len(trips), args.context_chunk):
        sub = trips[start:start + args.context_chunk]
        ix, ctx, _line_ctx, house, _li, _lt, _lc, _lu = batcher.make(sub)
        model.house, model.ctx = house, ctx
        factual_b = model.b_flat(ix).clone()
        day = torch.as_tensor(data["trip_day"][sub], dtype=torch.long)
        chain_log_price = absolute_log_price[ix.item, day[ix.item_trip]]
        chain_deviation = features.dev[ix.item, day[ix.item_trip]].double()
        slot_price = torch.exp(chain_log_price + ctx["dlp"] - chain_deviation)
        assortment_count = torch.bincount(
            ix.item_trip, minlength=ix.B).to(model.phi.dtype)

        generator = torch.Generator().manual_seed(
            args.seed + 100003 * segment + start)
        tick = time.perf_counter()
        smc = annealed_smc_logz(
            model, ix, schedule, particles=args.particles,
            mutation_steps=1, generator=generator)
        smc_seconds += time.perf_counter() - tick
        smc_ess.extend(smc.min_ess_fraction.tolist())
        size_axis = torch.arange(1, model.nmax + 1, dtype=model.phi.dtype)
        factual_stats = rao_blackwell_particle_statistics(model, ix, smc.states)
        factual_size = factual_stats.size_probability @ size_axis
        factual_incidence = factual_stats.item_incidence[ix.item_trip, ix.item]
        factual_value = torch.zeros(ix.B, dtype=model.phi.dtype).index_add_(
            0, ix.item_trip, factual_incidence * slot_price)
        baseline["size"].extend(factual_size.tolist())
        baseline["list_value"].extend(factual_value.tolist())
        baseline["tail"].extend(factual_stats.size_probability[:, 59:].sum(1).tolist())

        for action_index, action in enumerate(action_specs):
            products = torch.as_tensor(action["products"], dtype=torch.long)
            promoted = torch.isin(ix.item, products)
            log_change = math.log1p(-float(action["discount"]))
            changed = copied_context(ctx)
            changed["dlp"][promoted] += log_change
            per_trip_change = torch.zeros(ix.B, dtype=model.phi.dtype).index_add_(
                0, ix.item_trip[promoted],
                torch.full((int(promoted.sum()),), log_change,
                           dtype=model.phi.dtype))
            changed["dlp_bar"] += per_trip_change / assortment_count.clamp_min(1.0)
            model.ctx = changed
            delta = particle_delta(smc.states, model.b_flat(ix) - factual_b, ix.B)
            log_weight = torch.log_softmax(delta, dim=0)
            ess = torch.exp(-torch.logsumexp(2.0 * log_weight, dim=0)) \
                / args.particles
            stats = rao_blackwell_particle_statistics(
                model, ix, smc.states, log_weight)
            incidence = stats.item_incidence[ix.item_trip, ix.item]
            expected_size = stats.size_probability @ size_axis
            list_value = torch.zeros(ix.B, dtype=model.phi.dtype).index_add_(
                0, ix.item_trip, incidence * slot_price)
            markdown = torch.zeros(ix.B, dtype=model.phi.dtype).index_add_(
                0, ix.item_trip[promoted],
                incidence[promoted] * slot_price[promoted]
                * float(action["discount"]))
            bundle_incidence = torch.zeros(ix.B, dtype=model.phi.dtype).index_add_(
                0, ix.item_trip[promoted], incidence[promoted])
            target = accumulator[action_index]
            target["size"].extend(expected_size.tolist())
            target["list_value"].extend(list_value.tolist())
            target["actual_sales"].extend((list_value - markdown).tolist())
            target["markdown"].extend(markdown.tolist())
            target["bundle_incidence"].extend(bundle_incidence.tolist())
            target["tail"].extend(
                stats.size_probability[:, 59:].sum(1).tolist())
            target["ess"].extend(ess.tolist())
        model.ctx = ctx
        print(f"[promotion-mdp] segment={segment} contexts="
              f"{min(start + args.context_chunk, len(trips))}/{len(trips)}",
              flush=True)

    baseline_size = np.asarray(baseline["size"], dtype=np.float64)
    baseline_value = np.asarray(baseline["list_value"], dtype=np.float64)
    rows = []
    for action, values in zip(action_specs, accumulator):
        size = np.asarray(values["size"], dtype=np.float64)
        list_value = np.asarray(values["list_value"], dtype=np.float64)
        actual_sales = np.asarray(values["actual_sales"], dtype=np.float64)
        markdown = np.asarray(values["markdown"], dtype=np.float64)
        incremental_size = size - baseline_size
        incremental_list_value = list_value - baseline_value
        incremental_post_discount = actual_sales - baseline_value
        rows.append({
            "bundle": int(action["bundle"]),
            "discount": float(action["discount"]),
            "expected_size": float(size.mean()),
            "incremental_distinct_products": float(incremental_size.mean()),
            "incremental_distinct_products_se": float(
                incremental_size.std(ddof=1) / math.sqrt(len(incremental_size))),
            "list_value": float(list_value.mean()),
            "incremental_list_value": float(incremental_list_value.mean()),
            "incremental_list_value_se": float(
                incremental_list_value.std(ddof=1)
                / math.sqrt(len(incremental_list_value))),
            "incremental_list_value_lcb95": float(
                incremental_list_value.mean()
                - 1.96 * incremental_list_value.std(ddof=1)
                / math.sqrt(len(incremental_list_value))),
            "post_discount_sales": float(actual_sales.mean()),
            "incremental_post_discount_sales": float(
                incremental_post_discount.mean()),
            "incremental_post_discount_sales_se": float(
                incremental_post_discount.std(ddof=1)
                / math.sqrt(len(incremental_post_discount))),
            "markdown_spend": float(markdown.mean()),
            "promoted_bundle_incidence": float(
                np.mean(values["bundle_incidence"])),
            "tail_probability_ge_60_max": float(np.max(values["tail"])),
            "reweight_ess_min": float(np.min(values["ess"])),
        })
    return {
        "contexts": int(len(trips)),
        "distinct_households": int(np.unique(data["trip_user"][trips]).size),
        "particles_per_context": args.particles,
        "smc_seconds": smc_seconds,
        "smc_ess_min": float(np.min(smc_ess)),
        "baseline_expected_size": float(baseline_size.mean()),
        "baseline_list_value": float(baseline_value.mean()),
        "baseline_tail_probability_ge_60_max": float(np.max(baseline["tail"])),
        "actions": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--assignments", type=Path,
                        default=Path("artifacts/customer_segments.npz"))
    parser.add_argument("--segment-report", type=Path,
                        default=Path("reports/customer_segments.json"))
    parser.add_argument("--contexts-per-segment", type=int, default=64)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--levels", type=int, default=17)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--context-chunk", type=int, default=8)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42173)
    parser.add_argument("--bundles-per-segment", type=int, default=3)
    parser.add_argument("--products-per-bundle", type=int, default=5)
    parser.add_argument("--discounts", type=float, nargs="+", default=[0.10, 0.20])
    parser.add_argument("--horizon-days", type=int, default=28)
    parser.add_argument("--budget-fractions-of-maximum", type=float, nargs="+",
                        default=[0.25, 0.50, 0.75])
    parser.add_argument("--budget-bins", type=int, default=4000)
    parser.add_argument("--minimum-budget-utilization", type=float, default=0.95)
    parser.add_argument("--maximum-tail-probability", type=float, default=0.5)
    parser.add_argument("--minimum-reweight-ess", type=float, default=0.2)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/segment_promotion_mdp.json"))
    args = parser.parse_args()
    if any(not 0.0 < value < 1.0 for value in args.discounts):
        raise ValueError("discounts must lie strictly between zero and one")
    if any(not 0.0 < value <= 1.0 for value in args.budget_fractions_of_maximum):
        raise ValueError("budget fractions must lie in (0,1]")
    if not 0.0 < args.minimum_budget_utilization <= 1.0:
        raise ValueError("minimum budget utilization must lie in (0,1]")

    torch.set_num_threads(args.threads)
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() \
        else ROOT / args.checkpoint
    assignments = args.assignments if args.assignments.is_absolute() \
        else ROOT / args.assignments
    segment_report_path = args.segment_report if args.segment_report.is_absolute() \
        else ROOT / args.segment_report
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    data = build()
    model, _blob, meta = load_checkpoint(checkpoint, data)
    labels = np.load(assignments)["segment"].astype(np.int64)
    if len(labels) != int(data["n_user"]):
        raise RuntimeError("segment assignment count does not match household count")
    segment_report = json.loads(segment_report_path.read_text())
    if int(segment_report["chosen_segments"]) != 3:
        raise RuntimeError("promotion MDP requires the locked three-segment solution")
    segment_names = {int(row["segment"]): row["label"]
                     for row in segment_report["segments"]}
    metadata = pd.read_parquet(ROOT / "basket_input" / "items.parquet") \
        .sort_values("item_id")
    features = Features(int(data["n_item"]), int(data["n_store"]), 712)
    batcher = Batcher(data, features, int(meta["nmax"]))

    train = np.flatnonzero(
        (data["trip_split"] == 0) & (data["trip_nlines"] <= int(meta["nmax"])))
    training_days = max(1, np.unique(data["trip_day"][train]).size)
    total_trips_per_day = len(train) / training_days
    train_segment_count = np.bincount(
        labels[data["trip_user"][train]], minlength=3)
    traffic_share = train_segment_count / train_segment_count.sum()

    segments = []
    for segment in range(3):
        bundles = training_product_bundles(
            data, labels, metadata, segment,
            args.bundles_per_segment, args.products_per_bundle)
        trips = balanced_context_panel(
            data, labels, segment, args.contexts_per_segment,
            args.seed, int(meta["nmax"]))
        print(f"[promotion-mdp] starting segment={segment} "
              f"label={segment_names[segment]!r} contexts={len(trips)}",
              flush=True)
        evaluation = evaluate_segment_actions(
            model, batcher, features, data, trips, bundles,
            list(args.discounts), args, segment)
        evaluation.update({
            "segment": segment,
            "label": segment_names[segment],
            "training_trip_share": float(traffic_share[segment]),
            "expected_trips_per_day": float(total_trips_per_day * traffic_share[segment]),
            "bundles": bundles,
            "trips": trips.tolist(),
        })
        segments.append(evaluation)

    baseline_daily_value = sum(
        row["baseline_list_value"] * row["expected_trips_per_day"]
        for row in segments)
    actions = [{
        "action_id": "no_promotion", "segment": None, "bundle": None,
        "discount": 0.0, "daily_expected_markdown_spend": 0.0,
        "daily_incremental_list_value_mean": 0.0,
        "daily_incremental_list_value_lcb95": 0.0,
        "daily_incremental_post_discount_sales": 0.0,
        "daily_incremental_distinct_products": 0.0,
    }]
    for segment_row in segments:
        scale = segment_row["expected_trips_per_day"]
        for row in segment_row["actions"]:
            if (row["tail_probability_ge_60_max"] >= args.maximum_tail_probability
                    or row["reweight_ess_min"] < args.minimum_reweight_ess):
                continue
            bundle = int(row["bundle"])
            discount = float(row["discount"])
            actions.append({
                "action_id": f"segment_{segment_row['segment']}_bundle_{bundle}_"
                             f"discount_{int(round(100 * discount))}",
                "segment": int(segment_row["segment"]),
                "segment_label": segment_row["label"],
                "bundle": bundle,
                "bundle_description": segment_row["bundles"][bundle][
                    "category_description"],
                "products": segment_row["bundles"][bundle]["products"],
                "discount": discount,
                "daily_expected_markdown_spend": row["markdown_spend"] * scale,
                "daily_incremental_list_value_mean": (
                    row["incremental_list_value"] * scale),
                "daily_incremental_list_value_lcb95": (
                    row["incremental_list_value_lcb95"] * scale),
                "daily_incremental_post_discount_sales": (
                    row["incremental_post_discount_sales"] * scale),
                "daily_incremental_distinct_products": (
                    row["incremental_distinct_products"] * scale),
                "tail_probability_ge_60_max": row["tail_probability_ge_60_max"],
                "reweight_ess_min": row["reweight_ess_min"],
            })
    maximum_daily_spend = max(
        action["daily_expected_markdown_spend"] for action in actions)
    maximum_campaign_spend = args.horizon_days * maximum_daily_spend
    scenarios = []
    for fraction in args.budget_fractions_of_maximum:
        budget = fraction * maximum_campaign_spend
        answer = solve_budget_mdp(
            actions, args.horizon_days, budget, args.budget_bins,
            args.minimum_budget_utilization)
        answer.update({
            "budget_fraction_of_maximum_action_spend": fraction,
            "budget_fraction_of_baseline_campaign_value": (
                budget / (baseline_daily_value * args.horizon_days)),
        })
        scenarios.append(answer)

    output = {
        "checkpoint": str(checkpoint),
        "state": "(promotion days remaining, expected markdown budget remaining)",
        "actions": (
            "no promotion, or one segment-targeted five-product bundle discounted by "
            + "/".join(f"{int(100*x)}%" for x in args.discounts)),
        "horizon_days": args.horizon_days,
        "transition": (
            "one day elapses and expected markdown spend is deducted from the budget"),
        "reward": (
            "incremental basket value at undiscounted shelf prices, conditional on a "
            "shopping trip"),
        "budget_cost": (
            "discount times shelf price times model-implied promoted-product incidence"),
        "minimum_budget_utilization": args.minimum_budget_utilization,
        "training_only_action_design": True,
        "total_expected_trips_per_day": total_trips_per_day,
        "baseline_daily_list_value": baseline_daily_value,
        "segments": segments,
        "safe_daily_actions": actions,
        "maximum_campaign_spend_under_action_space": maximum_campaign_spend,
        "budget_scenarios": scenarios,
        "not_identified": [
            "wholesale cost and profit", "inventory", "probability of making a trip",
            "competitor/store switching", "quantity beyond product incidence"],
        "interpretation": (
            "This is a promotion-allocation MDP conditional on trips, not a profit or "
            "retention optimizer. Deployment requires costs, inventory, and an outside-"
            "option/visit model."),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps({
        "output": str(output_path),
        "horizon_days": args.horizon_days,
        "segments": [{
            "segment": row["segment"], "label": row["label"],
            "contexts": row["contexts"], "smc_ess_min": row["smc_ess_min"],
        } for row in segments],
        "budget_scenarios": [{
            key: scenario.get(key) for key in (
                "budget_fraction_of_maximum_action_spend", "budget",
                "feasible", "expected_spend_fraction_of_budget",
                "total_robust_incremental_list_value_lcb95",
                "total_incremental_list_value_mean",
                "total_incremental_post_discount_sales", "action_day_counts")
        } for scenario in scenarios],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
