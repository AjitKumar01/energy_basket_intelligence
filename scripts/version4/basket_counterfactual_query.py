#!/usr/bin/env python3
"""Retailer-facing, model-conditional basket counterfactual queries.

Two estimands are kept distinct:

* candidate-basket odds and exact-rest add/drop probabilities are exact under the
  fitted energy law because their common normalizer cancels;
* events that integrate over unspecified products are estimated with annealed-SMC
  particles and fail closed when bridge, reweighting, or conditional-event ESS is low.

The output is a comparison of fitted distributions under two price vectors.  It is not
an identified cross-world transition for an individual and is not a causal price effect.
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

os.environ.setdefault("V3_AFFINITY", "1")

from checkpoint_io import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from price_response import changed_price_context
from provenance import file_sha256, strict_json_dumps
from tempered_ais import annealed_smc_logz


torch.set_default_dtype(torch.float64)


def slot_for_item(ix, context: int, item: int) -> int:
    selected = torch.nonzero(
        (ix.item_trip == int(context)) & (ix.item == int(item)), as_tuple=True)[0]
    if selected.numel() != 1:
        raise ValueError(
            f"item {item} must occur exactly once in context {context}'s assortment")
    return int(selected[0])


def basket_slots(ix, context: int, items) -> torch.Tensor:
    values = [int(item) for item in items]
    if not values or len(values) != len(set(values)):
        raise ValueError("a candidate basket must contain unique items and be nonempty")
    return torch.as_tensor(
        [slot_for_item(ix, context, item) for item in values], dtype=torch.long,
        device=ix.item.device)


@torch.no_grad()
def basket_log_score(model, ix, slot_b: torch.Tensor, context: int,
                     items) -> torch.Tensor:
    """Unnormalized log probability of one declared basket under the energy law."""
    slots = basket_slots(ix, context, items)
    if slots.numel() > model.nmax:
        raise ValueError("candidate basket exceeds the checkpoint size support")
    selected_items = ix.item[slots]
    phi = model.phi[selected_items]
    total_phi = phi.sum(0)
    interaction = 0.5 * (total_phi.square().sum() - phi.square().sum())
    categories = ix.row_cat[ix.row_of[slots]]
    counts = torch.bincount(categories, minlength=model.C).to(model.phi.dtype)
    category = -(model.rho_c * counts * (counts - 1.0) * 0.5).sum()
    size = -model.rho_0()[slots.numel()]
    return slot_b[slots].sum() + interaction + category + size


@torch.no_grad()
def candidate_basket_distribution(model, ix, slot_b: torch.Tensor, context: int,
                                  candidates: list[dict]) -> dict:
    if len(candidates) < 2:
        raise ValueError("at least two candidate baskets are required")
    names = [str(row["name"]) for row in candidates]
    if len(names) != len(set(names)):
        raise ValueError("candidate basket names must be unique")
    scores = torch.stack([
        basket_log_score(model, ix, slot_b, context, row["items"])
        for row in candidates
    ])
    probability = torch.softmax(scores, 0)
    reference = scores[0]
    return {
        "conditioning": "conditional_on_one_of_the_declared_candidate_baskets",
        "reference_candidate": names[0],
        "candidates": [{
            "name": name,
            "items": [int(item) for item in row["items"]],
            "log_unnormalized_score": float(score),
            "conditional_probability": float(chance),
            "log_odds_vs_reference": float(score - reference),
        } for name, row, score, chance in zip(names, candidates, scores, probability)],
    }


@torch.no_grad()
def exact_rest_addition_probability(model, ix, slot_b: torch.Tensor, context: int,
                                    rest_items, addition_item: int) -> float:
    """P(addition present | every other product is exactly ``rest_items``)."""
    rest = [int(item) for item in rest_items]
    addition_item = int(addition_item)
    if addition_item in rest:
        raise ValueError("addition item is already part of the declared rest basket")
    without = basket_log_score(model, ix, slot_b, context, rest)
    with_item = basket_log_score(model, ix, slot_b, context,
                                 rest + [addition_item])
    return float(torch.sigmoid(with_item - without))


def _event_matches(items: set[int], specification: dict) -> bool:
    required = {int(item) for item in specification.get("required_items", [])}
    forbidden = {int(item) for item in specification.get("forbidden_items", [])}
    groups = [set(map(int, group))
              for group in specification.get("any_item_groups", [])]
    if any(not group for group in groups):
        raise ValueError("every any-item group must be nonempty")
    return required.issubset(items) and not bool(forbidden & items) \
        and all(bool(group & items) for group in groups)


def event_indicators(states, ix, context: int, specification: dict) -> torch.Tensor:
    values = []
    for particle in states:
        slots = particle[int(context)]
        items = set(map(int, ix.item[slots].detach().cpu().tolist()))
        values.append(_event_matches(items, specification))
    return torch.as_tensor(values, dtype=torch.bool, device=ix.item.device)


def _wilson_interval(probability: float, effective_n: float,
                     z: float = 1.959963984540054) -> list[float]:
    if effective_n <= 0 or not math.isfinite(effective_n):
        return [float("nan"), float("nan")]
    denominator = 1.0 + z * z / effective_n
    centre = (probability + z * z / (2.0 * effective_n)) / denominator
    radius = z * math.sqrt(
        probability * (1.0 - probability) / effective_n
        + z * z / (4.0 * effective_n * effective_n)) / denominator
    return [max(0.0, centre - radius), min(1.0, centre + radius)]


@torch.no_grad()
def weighted_event_probability(states, ix, log_weights: torch.Tensor,
                               event: dict, context: int = 0,
                               minimum_condition_ess: float = 20.0,
                               minimum_tail_ess: float = 3.0) -> dict:
    """Self-normalized event probability, optionally conditional on another event."""
    particles = len(states)
    if log_weights.shape != (particles, ix.B):
        raise ValueError("log_weights must have shape [particles, contexts]")
    weight = torch.softmax(log_weights[:, context], 0)
    target = event_indicators(states, ix, context, event).to(weight.dtype)
    condition_specification = event.get("condition", {})
    condition = event_indicators(
        states, ix, context, condition_specification).to(weight.dtype)
    restricted = weight * condition
    denominator = restricted.sum()
    if float(denominator) <= 0.0:
        return {
            "status": "inconclusive", "probability": None,
            "reason": "no particle satisfied the conditioning event",
            "condition_probability": 0.0, "condition_ess": 0.0,
        }
    probability = float((restricted * target).sum() / denominator)
    condition_ess = float(denominator.square() / restricted.square().sum())
    success_ess = probability * condition_ess
    failure_ess = (1.0 - probability) * condition_ess
    passed = (condition_ess >= minimum_condition_ess
              and min(success_ess, failure_ess) >= minimum_tail_ess)
    return {
        "status": "passed" if passed else "inconclusive",
        "probability": probability,
        "condition_probability": float(denominator),
        "condition_ess": condition_ess,
        "effective_successes": success_ess,
        "effective_failures": failure_ess,
        "approximate_95_interval": _wilson_interval(probability, condition_ess),
        "uncertainty_method": "Wilson interval using self-normalized condition ESS",
        "reason": None if passed else
            "conditional particle support is below the declared ESS thresholds",
    }


def _summarize_replicates(rows: list[dict]) -> dict:
    available = [row for row in rows if row.get("probability") is not None]
    if not available:
        return {"status": "inconclusive", "replicates": rows,
                "probability_mean": None, "replicate_standard_error": None}
    values = np.asarray([row["probability"] for row in available], dtype=np.float64)
    standard_error = (float(values.std(ddof=1) / math.sqrt(len(values)))
                      if len(values) >= 2 else None)
    return {
        "status": "passed" if len(available) == len(rows)
                    and all(row["status"] == "passed" for row in rows)
                    else "inconclusive",
        "probability_mean": float(values.mean()),
        "replicate_standard_error": standard_error,
        "replicate_range": [float(values.min()), float(values.max())],
        "replicates": rows,
    }


def item_metadata(metadata: pd.DataFrame, item: int) -> dict:
    row = metadata.loc[int(item)]
    return {
        "item_id": int(item), "product_id": int(row.PRODUCT_ID),
        "brand": str(row.BRAND), "commodity": str(row.COMMODITY_DESC),
        "sub_commodity": str(row.SUB_COMMODITY_DESC),
    }


@torch.no_grad()
def evaluate_query(model, ix, factual_context: dict, specification: dict,
                   *, particles: int, levels: int, power: float,
                   replicates: int, seed: int, minimum_smc_ess_fraction: float,
                   minimum_reweight_ess_fraction: float,
                   minimum_condition_ess: float, minimum_tail_ess: float) -> dict:
    if ix.B != 1:
        raise ValueError("a retailer query must contain exactly one customer context")
    if particles < 2 or levels < 2 or replicates < 1:
        raise ValueError("invalid particle, level, or replicate budget")
    action = specification["action"]
    action_item = int(action["item_id"])
    multiplier = float(action["price_multiplier"])
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("price multiplier must be finite and positive")
    action_slot = slot_for_item(ix, 0, action_item)
    change = torch.zeros_like(factual_context["dlp"])
    change[action_slot] = math.log(multiplier)
    counterfactual_context = changed_price_context(
        factual_context, ix.item_trip, change)

    model.ctx = factual_context
    factual_slot_b = model.b_flat(ix).clone()
    model.ctx = counterfactual_context
    counterfactual_slot_b = model.b_flat(ix).clone()
    model.ctx = factual_context

    candidates = specification.get("candidate_baskets", [])
    candidate_results = None
    if candidates:
        candidate_results = {
            "baseline": candidate_basket_distribution(
                model, ix, factual_slot_b, 0, candidates),
            "counterfactual": candidate_basket_distribution(
                model, ix, counterfactual_slot_b, 0, candidates),
        }
        base_by_name = {row["name"]: row for row in
                        candidate_results["baseline"]["candidates"]}
        changed_by_name = {row["name"]: row for row in
                           candidate_results["counterfactual"]["candidates"]}
        candidate_results["changes"] = [{
            "name": name,
            "conditional_probability_change":
                changed_by_name[name]["conditional_probability"]
                - base_by_name[name]["conditional_probability"],
            "log_score_change": changed_by_name[name]["log_unnormalized_score"]
                - base_by_name[name]["log_unnormalized_score"],
        } for name in base_by_name]

    additions = []
    for row in specification.get("exact_rest_additions", []):
        factual = exact_rest_addition_probability(
            model, ix, factual_slot_b, 0, row["rest_items"], row["addition_item"])
        counterfactual = exact_rest_addition_probability(
            model, ix, counterfactual_slot_b, 0,
            row["rest_items"], row["addition_item"])
        additions.append({
            "name": str(row["name"]),
            "conditioning": "all_other_products_are_exactly_the_declared_rest_items",
            "rest_items": list(map(int, row["rest_items"])),
            "addition_item": int(row["addition_item"]),
            "baseline_probability": factual,
            "counterfactual_probability": counterfactual,
            "probability_change": counterfactual - factual,
        })

    events = specification.get("events", [])
    event_results = {str(event["name"]): {"baseline": [], "counterfactual": []}
                     for event in events}
    numerical = []
    axis = torch.linspace(0.0, 1.0, levels)
    schedule = 1.0 - (1.0 - axis).pow(power)
    delta_slot = counterfactual_slot_b - factual_slot_b
    for replicate in range(replicates):
        print(f"[query] SMC replicate {replicate + 1}/{replicates}", flush=True)
        generator = torch.Generator().manual_seed(seed + 10007 * replicate)
        model.ctx = factual_context
        smc = annealed_smc_logz(
            model, ix, schedule, particles=particles, mutation_steps=1,
            generator=generator)
        factual_log_weights = torch.full(
            (particles, 1), -math.log(particles), dtype=model.phi.dtype)
        delta = torch.stack([
            delta_slot[particle[0]].sum() for particle in smc.states
        ]).unsqueeze(1)
        counterfactual_log_weights = torch.log_softmax(delta, 0)
        reweight_ess = float(torch.exp(
            -torch.logsumexp(2.0 * counterfactual_log_weights[:, 0], 0)) / particles)
        source = "factual_particle_reweighting"
        counterfactual_states = smc.states
        if reweight_ess < minimum_reweight_ess_fraction:
            model.ctx = counterfactual_context
            direct = annealed_smc_logz(
                model, ix, schedule, particles=particles, mutation_steps=1,
                generator=torch.Generator().manual_seed(seed + 10007 * replicate + 1))
            counterfactual_states = direct.states
            counterfactual_log_weights = factual_log_weights
            source = "direct_counterfactual_smc_fallback"
            counterfactual_smc_ess = float(direct.min_ess_fraction[0])
        else:
            counterfactual_smc_ess = None
        factual_smc_ess = float(smc.min_ess_fraction[0])
        numerical.append({
            "replicate": replicate,
            "factual_smc_minimum_bridge_ess_fraction": factual_smc_ess,
            "counterfactual_reweight_ess_fraction": reweight_ess,
            "counterfactual_source": source,
            "counterfactual_smc_minimum_bridge_ess_fraction": counterfactual_smc_ess,
            "passed": factual_smc_ess >= minimum_smc_ess_fraction and
                (counterfactual_smc_ess is None or
                 counterfactual_smc_ess >= minimum_smc_ess_fraction),
        })
        print(f"[query] replicate {replicate + 1}/{replicates} complete; "
              f"bridge_ess={factual_smc_ess:.4f} "
              f"reweight_ess={reweight_ess:.4f} source={source}", flush=True)
        for event in events:
            name = str(event["name"])
            event_results[name]["baseline"].append(weighted_event_probability(
                smc.states, ix, factual_log_weights, event, 0,
                minimum_condition_ess, minimum_tail_ess))
            event_results[name]["counterfactual"].append(weighted_event_probability(
                counterfactual_states, ix, counterfactual_log_weights, event, 0,
                minimum_condition_ess, minimum_tail_ess))
    model.ctx = factual_context

    summarized_events = {}
    for name, values in event_results.items():
        baseline = _summarize_replicates(values["baseline"])
        counterfactual = _summarize_replicates(values["counterfactual"])
        difference = None
        if baseline["probability_mean"] is not None \
                and counterfactual["probability_mean"] is not None:
            difference = (counterfactual["probability_mean"]
                          - baseline["probability_mean"])
        summarized_events[name] = {
            "baseline": baseline, "counterfactual": counterfactual,
            "probability_change": difference,
            "status": "passed" if baseline["status"] == "passed"
                and counterfactual["status"] == "passed" else "inconclusive",
        }
    numerical_passed = all(row["passed"] for row in numerical)
    event_passed = all(row["status"] == "passed"
                       for row in summarized_events.values())
    exact_passed = candidate_results is not None or bool(additions)
    return {
        "estimand": "comparison_of_fitted_basket_distributions_under_two_price_vectors",
        "causal_status": "not_identified_from_observational_fit",
        "individual_transition_status": "not_identified_without_a_cross_world_coupling",
        "action": {"item_id": action_item, "price_multiplier": multiplier,
                   "log_price_change": math.log(multiplier)},
        "candidate_basket_comparison": candidate_results,
        "exact_rest_additions": additions,
        "integrated_events": summarized_events,
        "numerical_diagnostics": numerical,
        "gates": {
            "exact_declared_scenarios": exact_passed,
            "numerical_particles": numerical_passed,
            "integrated_event_precision": event_passed,
        },
        "component_status": {
            "exact_declared_scenarios": "passed" if exact_passed else "not_requested",
            "integrated_events": "passed" if event_passed else "inconclusive",
        },
        "status": "passed" if numerical_passed and event_passed else "inconclusive",
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--query", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--levels", type=int, default=17)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--seed", type=int, default=71101)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--minimum-smc-ess-fraction", type=float, default=.20)
    parser.add_argument("--minimum-reweight-ess-fraction", type=float, default=.20)
    parser.add_argument("--minimum-condition-ess", type=float, default=20.0)
    parser.add_argument("--minimum-tail-ess", type=float, default=3.0)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (not 0 < args.minimum_smc_ess_fraction <= 1
            or not 0 < args.minimum_reweight_ess_fraction <= 1
            or args.minimum_condition_ess < 0 or args.minimum_tail_ess < 0):
        raise ValueError("ESS thresholds are outside their valid ranges")
    torch.set_num_threads(args.threads)
    specification = json.loads(args.query.read_text())
    trip = int(specification["trip"])
    data = build()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() \
        else ROOT / args.checkpoint
    model, blob, meta = load_checkpoint(
        checkpoint, data,
        required_capabilities=("conditional_nonempty_incidence", "gram_interactions"))
    if trip < 0 or trip >= len(data["trip_split"]):
        raise ValueError("query trip is outside the data universe")
    features = Features(int(data["n_item"]), int(data["n_store"]), 712,
                        include_recency=False)
    batcher = Batcher(data, features, int(meta["nmax"]), include_recency=False)
    ix, context, _line_context, house, *_ = batcher.make(
        np.asarray([trip], dtype=np.int64))
    model.house, model.ctx = house, context
    answer = evaluate_query(
        model, ix, context, specification,
        particles=args.particles, levels=args.levels, power=args.power,
        replicates=args.replicates, seed=args.seed,
        minimum_smc_ess_fraction=args.minimum_smc_ess_fraction,
        minimum_reweight_ess_fraction=args.minimum_reweight_ess_fraction,
        minimum_condition_ess=args.minimum_condition_ess,
        minimum_tail_ess=args.minimum_tail_ess)
    metadata = pd.read_parquet(ROOT / "basket_input/items.parquet") \
        .sort_values("item_id").set_index("item_id")
    declared_items = {int(answer["action"]["item_id"])}
    for row in specification.get("candidate_baskets", []):
        declared_items.update(map(int, row["items"]))
    for row in specification.get("exact_rest_additions", []):
        declared_items.update(map(int, row["rest_items"]))
        declared_items.add(int(row["addition_item"]))
    for event in specification.get("events", []):
        for key in ("required_items", "forbidden_items"):
            declared_items.update(map(int, event.get(key, [])))
        for group in event.get("any_item_groups", []):
            declared_items.update(map(int, group))
        condition = event.get("condition", {})
        for key in ("required_items", "forbidden_items"):
            declared_items.update(map(int, condition.get(key, [])))
        for group in condition.get("any_item_groups", []):
            declared_items.update(map(int, group))
    report = {
        "status": answer["status"],
        "deployment_status": "research_only_until_factual_and_causal_gates_pass",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": file_sha256(checkpoint),
        "implementation_sha256": file_sha256(Path(__file__).resolve()),
        "data_fingerprint_sha256": blob["data_fingerprint_sha256"],
        "query_path": str(args.query.resolve()),
        "query_sha256": file_sha256(args.query),
        "context": {
            "trip": trip, "split": ["train", "validation", "test"][
                int(data["trip_split"][trip])],
            "household": int(data["trip_user"][trip]),
            "store": int(data["trip_store"][trip]),
            "day": int(data["trip_day"][trip]),
            "week": int(data["trip_week"][trip]),
            "observed_size": int(data["trip_nlines"][trip]),
        },
        "items": [item_metadata(metadata, item) for item in sorted(declared_items)],
        "query_result": answer,
        "limitations": [
            "the checkpoint conditions on an existing nonempty trip",
            "the fitted assortment is the declared chain catalogue, not observed stock",
            "recency is disabled in this fitted feature contract",
            "candidate probabilities are conditional on the retailer-declared candidate set",
            "integrated-event intervals quantify Monte Carlo precision, not model uncertainty",
        ],
    }
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(strict_json_dumps(report)); os.replace(temporary, output)
    print(strict_json_dumps(report), end="", flush=True)


if __name__ == "__main__":
    main()
