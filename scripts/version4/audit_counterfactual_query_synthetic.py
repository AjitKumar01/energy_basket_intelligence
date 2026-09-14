#!/usr/bin/env python3
"""Exact-enumeration certification of the basket counterfactual query layer."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch

from audit_probability_foundations import exact_energy, make_world
from basket_counterfactual_query import (
    basket_log_score,
    candidate_basket_distribution,
    exact_rest_addition_probability,
    weighted_event_probability,
)
from conditional_basket import (
    completion_log_score,
    conditional_completion_quadrature,
    conditional_completion_smc,
)
from price_response import changed_price_context
from provenance import file_sha256, strict_json_dumps
from ragged import smolyak_grid


torch.set_default_dtype(torch.float64)


def enumerated_states(ix, baskets):
    slots = [{int(ix.item[position]): int(position)
              for position in torch.nonzero(ix.item_trip == context).flatten()}
             for context in range(ix.B)]
    return [[torch.as_tensor([slots[context][item] for item in basket],
                             dtype=torch.long)
             for context in range(ix.B)] for basket in baskets]


@torch.no_grad()
def audit(seed: int) -> dict:
    model, ix, membership, baskets = make_world(
        seed=seed, products=7, contexts=3, nmax=4, strength=.8)
    factual_context = model.ctx
    factual_energy = exact_energy(model, ix, membership)
    factual_slot_b = model.b_flat(ix).clone()
    action = (ix.item == 0).to(model.phi.dtype) * math.log(1.2)
    changed_context = changed_price_context(model.ctx, ix.item_trip, action)
    counterfactual_energy = exact_energy(model, ix, membership, changed_context)
    model.ctx = changed_context
    counterfactual_slot_b = model.b_flat(ix).clone()
    model.ctx = factual_context

    candidates = [
        {"name": "item_0_plus_anchor", "items": [0, 2]},
        {"name": "item_1_plus_anchor", "items": [1, 2]},
        {"name": "anchor_only", "items": [2]},
    ]
    result = candidate_basket_distribution(
        model, ix, counterfactual_slot_b, 0, candidates)
    lookup = {basket: index for index, basket in enumerate(baskets)}
    indices = [lookup[tuple(row["items"])] for row in candidates]
    expected_candidate = counterfactual_energy[0, indices].softmax(0)
    actual_candidate = torch.as_tensor([
        row["conditional_probability"] for row in result["candidates"]])
    candidate_error = float((actual_candidate - expected_candidate).abs().max())

    score_error = 0.0
    for context in range(ix.B):
        for index, basket in enumerate(baskets):
            score = basket_log_score(
                model, ix, factual_slot_b, context, basket)
            score_error = max(
                score_error, float((score - factual_energy[context, index]).abs()))

    rest, addition = [1, 2], 4
    actual_addition = exact_rest_addition_probability(
        model, ix, counterfactual_slot_b, 0, rest, addition)
    expected_addition = float(torch.sigmoid(
        counterfactual_energy[0, lookup[tuple(rest + [addition])]]
        - counterfactual_energy[0, lookup[tuple(rest)]]))
    addition_error = abs(actual_addition - expected_addition)

    event = {
        "required_items": [2], "forbidden_items": [0],
        "any_item_groups": [[1, 3]],
        "condition": {"required_items": [4]},
    }
    states = enumerated_states(ix, baskets)

    def oracle(energy):
        probability = energy[0].softmax(0)
        condition = membership[:, 4].bool()
        target = (membership[:, 2].bool() & ~membership[:, 0].bool()
                  & (membership[:, 1].bool() | membership[:, 3].bool()))
        return float(probability[target & condition].sum()
                     / probability[condition].sum())

    factual_event = weighted_event_probability(
        states, ix, factual_energy.T, event, minimum_condition_ess=0,
        minimum_tail_ess=0)
    counterfactual_event = weighted_event_probability(
        states, ix, counterfactual_energy.T, event, minimum_condition_ess=0,
        minimum_tail_ess=0)
    event_error = max(
        abs(factual_event["probability"] - oracle(factual_energy)),
        abs(counterfactual_event["probability"] - oracle(counterfactual_energy)))
    revealed = [[2], [1, 4], [0]]
    completion_score_error = 0.0
    for context, anchor in enumerate(revealed):
        anchor_index = lookup[tuple(sorted(anchor))]
        for index, candidate in enumerate(baskets):
            if not set(anchor).issubset(candidate):
                continue
            completion = [item for item in candidate if item not in set(anchor)]
            actual = completion_log_score(
                model, ix, factual_slot_b, context, anchor, completion)
            expected = factual_energy[context, index] - factual_energy[context, anchor_index]
            completion_score_error = max(
                completion_score_error, float((actual - expected).abs()))
    forced = conditional_completion_smc(
        model, ix, factual_slot_b, revealed, particles=4096,
        schedule=torch.linspace(0.0, 1.0, 17).tolist(),
        generator=torch.Generator().manual_seed(seed + 17))
    stop_error = 0.0
    incidence_error = 0.0
    for context, anchor in enumerate(revealed):
        allowed = membership[:, anchor].bool().all(1)
        conditional_probability = factual_energy[context, allowed].softmax(0)
        allowed_membership = membership[allowed]
        expected_stop = conditional_probability[
            allowed_membership.sum(1) == len(anchor)].sum()
        expected_incidence = conditional_probability @ allowed_membership
        expected_incidence[anchor] = 0.0
        stop_error = max(stop_error, float(
            (forced.stop_probability[context] - expected_stop).abs()))
        incidence_error = max(incidence_error, float(
            (forced.item_incidence[context] - expected_incidence).abs().max()))

    # Certify the informative random-mask protocol as well as the literal conditional.
    # A uniformly revealed a-subset gives a basket of final size m weight 1/C(m,a).
    size_log_weight = torch.empty(ix.B, model.nmax + 1)
    for context, anchor in enumerate(revealed):
        for additional in range(model.nmax + 1):
            total = len(anchor) + additional
            size_log_weight[context, additional] = (
                -math.log(math.comb(total, len(anchor)))
                if total <= model.nmax else -float("inf"))
    nodes, weights = smolyak_grid(model.Kz, model.Kz + model.nmax + 1)
    masked = conditional_completion_quadrature(
        model, ix, factual_slot_b, revealed, nodes, weights,
        completion_size_log_weight=size_log_weight)
    masked_stop_error = 0.0
    masked_incidence_error = 0.0
    masked_size_error = 0.0
    masked_normalizer_error = 0.0
    for context, anchor in enumerate(revealed):
        allowed = membership[:, anchor].bool().all(1)
        allowed_membership = membership[allowed]
        final_size = allowed_membership.sum(1).to(torch.long)
        probability = factual_energy[context, allowed].exp()
        selection = torch.as_tensor([
            1.0 / math.comb(int(size), len(anchor)) for size in final_size
        ], dtype=probability.dtype)
        probability = probability * selection
        probability = probability / probability.sum()
        expected_stop = probability[final_size == len(anchor)].sum()
        expected_incidence = probability @ allowed_membership
        expected_incidence[anchor] = 0.0
        expected_size = torch.zeros(model.nmax + 1)
        expected_size.index_add_(0, final_size-len(anchor), probability)
        masked_stop_error = max(masked_stop_error, float(
            (masked.stop_probability[context]-expected_stop).abs()))
        masked_incidence_error = max(masked_incidence_error, float(
            (masked.item_incidence[context]-expected_incidence).abs().max()))
        masked_size_error = max(masked_size_error, float(
            (masked.completion_size_probability[context]-expected_size).abs().max()))
        masked_normalizer_error = max(masked_normalizer_error, float(
            (torch.exp(-masked.log_normalizer[context])-expected_stop).abs()))
    errors = {
        "basket_log_score_max_error": score_error,
        "candidate_probability_max_error": candidate_error,
        "exact_rest_addition_error": addition_error,
        "conditional_event_probability_max_error": event_error,
        "forced_completion_score_max_error": completion_score_error,
        "forced_completion_smc_stop_max_error": stop_error,
        "forced_completion_smc_incidence_max_error": incidence_error,
        "masked_completion_quadrature_stop_max_error": masked_stop_error,
        "masked_completion_quadrature_incidence_max_error": masked_incidence_error,
        "masked_completion_quadrature_size_max_error": masked_size_error,
        "masked_completion_quadrature_normalizer_max_error": masked_normalizer_error,
    }
    tolerance = 2e-12
    gates = {name: math.isfinite(value) and value <= (
                .035 if "_smc_" in name else
                5e-11 if "masked_completion_quadrature" in name else tolerance)
             for name, value in errors.items()}
    gates["forced_completion_smc_minimum_ess"] = bool(
        float(forced.min_ess_fraction.min()) >= .2)
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "seed": seed, "products": model.J, "contexts": ix.B,
        "enumerated_nonempty_baskets": len(baskets),
        "price_action": {"item": 0, "multiplier": 1.2},
        "errors": errors, "tolerance": tolerance, "gates": gates,
        "forced_completion_smc": {
            "particles": 4096, "levels": 17,
            "stochastic_absolute_tolerance": .035,
            "minimum_ess_fraction": float(forced.min_ess_fraction.min()),
        },
        "claim": ("finite exact-enumeration verification of the query estimands; "
                  "not a universal proof or real-data causal validation"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=76101)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.seed)
    result["implementation_sha256"] = file_sha256(Path(__file__).resolve())
    result["query_engine_sha256"] = file_sha256(
        Path(__file__).with_name("basket_counterfactual_query.py"))
    result["conditional_engine_sha256"] = file_sha256(
        Path(__file__).with_name("conditional_basket.py"))
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(strict_json_dumps(result))
    print(strict_json_dumps(result), end="", flush=True)
    if result["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
