import math

import numpy as np
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
from ragged import smolyak_grid


def enumerated_states(ix, baskets):
    slots = [{int(ix.item[position]): int(position)
              for position in torch.nonzero(ix.item_trip == context).flatten()}
             for context in range(ix.B)]
    return [[torch.as_tensor([slots[context][item] for item in basket],
                             dtype=torch.long)
             for context in range(ix.B)] for basket in baskets]


def test_candidate_basket_scores_and_addition_match_exact_energy():
    model, ix, membership, baskets = make_world(
        seed=75101, products=7, contexts=2, nmax=4, strength=.8)
    energy = exact_energy(model, ix, membership)
    slot_b = model.b_flat(ix)
    chosen = [0, 5, 19]
    for index in chosen:
        actual = basket_log_score(model, ix, slot_b, 0, baskets[index])
        assert torch.allclose(actual, energy[0, index], atol=1e-12, rtol=1e-12)

    candidates = [{"name": str(index), "items": list(baskets[index])}
                  for index in chosen]
    result = candidate_basket_distribution(model, ix, slot_b, 0, candidates)
    actual_probability = torch.as_tensor([
        row["conditional_probability"] for row in result["candidates"]])
    assert torch.allclose(
        actual_probability, energy[0, chosen].softmax(0), atol=1e-12, rtol=1e-12)

    rest, addition = [0, 2], 4
    actual = exact_rest_addition_probability(model, ix, slot_b, 0, rest, addition)
    lookup = {basket: index for index, basket in enumerate(baskets)}
    expected = torch.sigmoid(
        energy[0, lookup[tuple(rest + [addition])]]
        - energy[0, lookup[tuple(rest)]])
    assert math.isclose(actual, float(expected.detach()), abs_tol=1e-12)


def test_weighted_joint_and_conditional_events_match_enumeration_under_price_action():
    model, ix, membership, baskets = make_world(
        seed=75201, products=7, contexts=2, nmax=4, strength=.8)
    states = enumerated_states(ix, baskets)
    event = {
        "required_items": [1],
        "forbidden_items": [5],
        "any_item_groups": [[2, 3]],
        "condition": {"required_items": [0]},
    }

    def oracle(energy):
        probability = energy[0].softmax(0)
        condition = membership[:, 0].bool()
        target = (membership[:, 1].bool() & ~membership[:, 5].bool()
                  & (membership[:, 2].bool() | membership[:, 3].bool()))
        return float((probability[condition & target].sum()
                      / probability[condition].sum()).detach())

    factual_energy = exact_energy(model, ix, membership)
    factual = weighted_event_probability(
        states, ix, factual_energy.T, event,
        minimum_condition_ess=0, minimum_tail_ess=0)
    assert math.isclose(factual["probability"], oracle(factual_energy), abs_tol=1e-12)

    change = (ix.item == 1).double() * math.log(1.2)
    changed = changed_price_context(model.ctx, ix.item_trip, change)
    counterfactual_energy = exact_energy(model, ix, membership, changed)
    counterfactual = weighted_event_probability(
        states, ix, counterfactual_energy.T, event,
        minimum_condition_ess=0, minimum_tail_ess=0)
    assert math.isclose(
        counterfactual["probability"], oracle(counterfactual_energy), abs_tol=1e-12)
    assert np.isfinite(counterfactual["condition_ess"])


def test_forced_cart_completion_score_and_smc_match_exact_conditional_law(native_dp):
    model, ix, membership, baskets = make_world(
        seed=75301, products=7, contexts=2, nmax=4, strength=.8)
    energy = exact_energy(model, ix, membership)
    slot_b = model.b_flat(ix).detach()
    revealed = [[0, 2], [1]]
    lookup = {basket: index for index, basket in enumerate(baskets)}

    # The transformed completion score must be exactly E(A union T)-E(A), including
    # the empty completion and completions sharing a category with the revealed cart.
    for context, anchor in enumerate(revealed):
        candidates = [(), (3,), (4,), (3, 4)]
        for completion in candidates:
            if set(anchor) & set(completion) or len(anchor) + len(completion) > model.nmax:
                continue
            expected = (energy[context, lookup[tuple(sorted(anchor + list(completion)))]]
                        - energy[context, lookup[tuple(sorted(anchor))]])
            actual = completion_log_score(
                model, ix, slot_b, context, anchor, completion)
            assert torch.allclose(actual, expected, atol=2e-12, rtol=2e-12)

    result = conditional_completion_smc(
        model, ix, slot_b, revealed, particles=4096,
        schedule=torch.linspace(0.0, 1.0, 17).tolist(),
        generator=torch.Generator().manual_seed(75302))
    for context, anchor in enumerate(revealed):
        allowed = membership[:, anchor].bool().all(1)
        probability = energy[context, allowed].softmax(0)
        conditional_membership = membership[allowed]
        expected_stop = float(probability[
            conditional_membership.sum(1) == len(anchor)].sum().detach())
        expected_incidence = probability @ conditional_membership
        expected_incidence[anchor] = 0.0
        assert abs(float(result.stop_probability[context]) - expected_stop) < .025
        assert torch.allclose(
            result.item_incidence[context], expected_incidence, atol=.035, rtol=0)
        assert float(result.min_ess_fraction[context]) > .2

    nodes, weights = smolyak_grid(model.Kz, model.Kz + model.nmax + 1)
    deterministic = conditional_completion_quadrature(
        model, ix, slot_b, revealed, nodes, weights)
    for context, anchor in enumerate(revealed):
        allowed = membership[:, anchor].bool().all(1)
        probability = energy[context, allowed].softmax(0)
        conditional_membership = membership[allowed]
        expected_stop = probability[
            conditional_membership.sum(1) == len(anchor)].sum().detach()
        expected_incidence = (probability @ conditional_membership).detach()
        expected_incidence[anchor] = 0.0
        assert torch.allclose(
            deterministic.stop_probability[context], expected_stop,
            atol=2e-11, rtol=2e-11)
        assert torch.allclose(
            torch.exp(-deterministic.log_normalizer[context]), expected_stop,
            atol=2e-11, rtol=2e-11)
        assert torch.allclose(
            deterministic.item_incidence[context], expected_incidence,
            atol=2e-11, rtol=2e-11)

    # Uniformly masking an observed subset is not ignorable: a final basket of size m
    # exposes any particular a-item anchor with probability 1/C(m,a).  Verify that the
    # size-weighted quadrature differentiates the adjusted joint law, including its item
    # marginals, rather than merely rescaling the reported size probabilities afterward.
    observation_log_weight = torch.empty(len(revealed), model.nmax + 1)
    for context, anchor in enumerate(revealed):
        for additional in range(model.nmax + 1):
            total = len(anchor) + additional
            observation_log_weight[context, additional] = (
                -math.log(math.comb(total, len(anchor)))
                if total <= model.nmax else -float("inf"))
    masked = conditional_completion_quadrature(
        model, ix, slot_b, revealed, nodes, weights,
        completion_size_log_weight=observation_log_weight)
    for context, anchor in enumerate(revealed):
        allowed = membership[:, anchor].bool().all(1)
        conditional_membership = membership[allowed]
        final_size = conditional_membership.sum(1).to(torch.long)
        probability = energy[context, allowed].exp()
        selection = torch.as_tensor([
            1.0 / math.comb(int(size), len(anchor)) for size in final_size
        ], dtype=probability.dtype)
        probability = probability * selection
        probability = probability / probability.sum()
        expected_stop = probability[
            final_size == len(anchor)].sum().detach()
        expected_incidence = (probability @ conditional_membership).detach()
        expected_incidence[anchor] = 0.0
        expected_size = torch.zeros(model.nmax + 1, dtype=probability.dtype)
        expected_size.index_add_(
            0, final_size - len(anchor), probability.detach())
        assert torch.allclose(masked.stop_probability[context], expected_stop,
                              atol=2e-11, rtol=2e-11)
        assert torch.allclose(masked.item_incidence[context], expected_incidence,
                              atol=2e-11, rtol=2e-11)
        assert torch.allclose(masked.completion_size_probability[context], expected_size,
                              atol=2e-11, rtol=2e-11)
        assert torch.allclose(torch.exp(-masked.log_normalizer[context]), expected_stop,
                              atol=2e-11, rtol=2e-11)
    assert getattr(model, "_b_override", None) is None
    assert getattr(model, "_condition_cat_count", None) is None
    assert getattr(model, "_condition_size", None) is None
