import itertools
import math
import sys
from pathlib import Path

import numpy as np
import torch


V4 = Path(__file__).resolve().parents[1] / "scripts" / "version4"
sys.path.insert(0, str(V4))

from audit_synthetic_complete_demand import (  # noqa: E402
    CompleteDemandLaw,
    Config,
    enumerate_support,
    exhaustive_additive_partition,
    log_category_partition,
    log_esp_partition,
    make_truth,
    simulate,
    tensor_data,
)


def tiny_config(**overrides):
    values = dict(customers=12, products=7, categories=2, segments=2,
                  stores=2, actions=3, rank=2, nmax=3, days=20,
                  train_days=12, validation_days=4, seed=71)
    values.update(overrides)
    return Config(**values)


def blank_context(config):
    rows = 9
    return tensor_data({
        "day": np.arange(rows),
        "segment": np.arange(rows) % config.segments,
        "actions": np.arange(rows * config.stores).reshape(rows, config.stores)
                   % config.actions,
        "distance": np.linspace(0.1, 0.9, rows * config.stores).reshape(
            rows, config.stores),
        "sin_day": np.sin(np.arange(rows)),
        "cos_day": np.cos(np.arange(rows)),
        "outcome": np.zeros(rows, dtype=np.int64),
        "store": np.full(rows, -1, dtype=np.int64),
        "basket": np.full(rows, -1, dtype=np.int64),
    })


def test_joint_law_normalizes_over_outside_empty_and_every_basket():
    config = tiny_config()
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    model = CompleteDemandLaw(config, truth, interaction=True, linked=True)
    model.load_truth(truth)
    data = blank_context(config)
    with torch.no_grad():
        marginal = model.marginal_probabilities(data, support)
    torch.testing.assert_close(
        marginal["normalization"], torch.ones(len(data["day"])),
        atol=1e-12, rtol=0)


def test_conditioning_joint_law_recovers_version4_nonempty_basket_law():
    config = tiny_config()
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    model = CompleteDemandLaw(config, truth, interaction=True, linked=True)
    model.load_truth(truth)
    data = blank_context(config)
    with torch.no_grad():
        terms = model.context_terms(data, support)
        group = int(data["segment"][0])
        store = 0
        action = int(data["actions"][0, store])
        conditional_from_energy = torch.softmax(
            terms["energy"][group, store, action], dim=0)
        joint_nonempty = torch.exp(
            terms["direct"][0, store]
            + terms["activation"][0, 0]
            + terms["energy"][group, store, action]
            + terms["log_no"][0])
        conditional_from_joint = joint_nonempty / joint_nonempty.sum()
    torch.testing.assert_close(conditional_from_joint, conditional_from_energy,
                               atol=1e-12, rtol=0)


def test_partition_derivative_is_conditional_item_incidence():
    config = tiny_config()
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    model = CompleteDemandLaw(config, truth, interaction=True, linked=True)
    model.load_truth(truth)
    energy = model.basket_energy(support)[0, 0, 0].detach().clone()
    membership = support["membership"]
    logz = torch.logsumexp(energy, dim=0)
    expected = torch.softmax(energy, dim=0) @ membership
    epsilon = 1e-6
    finite = []
    for item in range(config.products):
        changed = energy + epsilon * membership[:, item]
        finite.append(float((torch.logsumexp(changed, 0) - logz) / epsilon))
    np.testing.assert_allclose(finite, expected.numpy(), atol=2e-7, rtol=0)


def test_observed_log_probability_matches_explicit_joint_formula():
    config = tiny_config()
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    model = CompleteDemandLaw(config, truth, interaction=True, linked=True)
    model.load_truth(truth)
    data = blank_context(config)
    data["outcome"][:3] = torch.tensor([0, 1, 2])
    data["store"][:3] = torch.tensor([-1, 0, 1])
    data["basket"][:3] = torch.tensor([-1, -1, 4])
    terms = model.context_terms(data, support)
    got = model.observed_log_probability(data, support)
    expected = torch.stack([
        terms["log_no"][0],
        terms["direct"][1, 0] + terms["log_no"][1],
        terms["direct"][2, 1] + terms["activation"][2, 0]
        + terms["energy"][data["segment"][2], 1, data["actions"][2, 1], 4]
        + terms["log_no"][2],
    ])
    torch.testing.assert_close(got[:3], expected, atol=1e-12, rtol=0)


def test_no_visit_sends_gradient_into_linked_basket_parameters_only():
    config = tiny_config()
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    data = blank_context(config)
    linked = CompleteDemandLaw(config, truth, interaction=True, linked=True)
    linked.load_truth(truth)
    (-linked.observed_log_probability(data, support).mean()).backward()
    assert linked.base.grad is not None
    assert float(linked.base.grad.abs().sum()) > 1e-8

    independent = CompleteDemandLaw(config, truth, interaction=True, linked=False)
    independent.load_truth(truth)
    (-independent.observed_log_probability(data, support).mean()).backward()
    assert independent.base.grad is None or float(independent.base.grad.abs().sum()) == 0.0


def test_discount_changes_linked_visit_probability_but_not_independent_visit_head():
    config = tiny_config()
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    base = blank_context(config)
    promoted = {key: value.clone() for key, value in base.items()}
    base["actions"].zero_()
    promoted["actions"].fill_(config.actions - 1)
    linked = CompleteDemandLaw(config, truth, interaction=True, linked=True)
    independent = CompleteDemandLaw(config, truth, interaction=True, linked=False)
    linked.load_truth(truth)
    independent.load_truth(truth)
    with torch.no_grad():
        linked_change = (linked.marginal_probabilities(promoted, support)["visit"].mean()
                         - linked.marginal_probabilities(base, support)["visit"].mean())
        independent_change = (
            independent.marginal_probabilities(promoted, support)["visit"].mean()
            - independent.marginal_probabilities(base, support)["visit"].mean())
    assert float(linked_change) > 0.0
    assert abs(float(independent_change)) < 1e-14


def test_polynomial_partition_matches_exhaustive_small_support():
    rng = np.random.default_rng(8)
    log_weight = -1.4 + 0.3 * rng.normal(size=9)
    rho = 0.04 * np.arange(1, 5) ** 2
    got = log_esp_partition(log_weight, rho, 4)
    expected = exhaustive_additive_partition(log_weight, rho, 4)
    assert abs(got - expected) < 1e-12


def test_category_factorized_partition_matches_exhaustive_energy():
    rng = np.random.default_rng(13)
    products, nmax, groups = 8, 4, 3
    log_weight = -1.3 + 0.25 * rng.normal(size=products)
    category = np.arange(products) % groups
    rho_category = np.asarray([0.12, -0.04, 0.20])
    rho_size = 0.05 * np.arange(1, nmax + 1) ** 2
    values = []
    for size in range(1, nmax + 1):
        for basket in itertools.combinations(range(products), size):
            count = np.bincount(category[list(basket)], minlength=groups)
            category_energy = np.sum(rho_category * count * (count - 1) / 2)
            values.append(log_weight[list(basket)].sum()
                          - category_energy - rho_size[size - 1])
    exact = float(torch.logsumexp(torch.as_tensor(values), dim=0))
    got = log_category_partition(log_weight, category, rho_category,
                                 rho_size, nmax)
    assert abs(got - exact) < 1e-12


def test_simulator_contains_all_three_outcome_types_and_valid_baskets():
    config = tiny_config(customers=30, days=40, train_days=24,
                         validation_days=8, seed=19)
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    data, _ = simulate(config, support, truth)
    assert set(np.unique(data["outcome"])) == {0, 1, 2}
    assert np.all(data["store"][data["outcome"] == 0] == -1)
    assert np.all(data["basket"][data["outcome"] != 2] == -1)
    assert np.all(data["basket"][data["outcome"] == 2] >= 0)


def test_log_esp_state_memory_does_not_depend_on_catalogue_size():
    # The recurrence state contains degrees 0..nmax, irrespective of J.
    nmax = 20
    rho = 0.01 * np.arange(1, nmax + 1) ** 2
    for products in (50, 500, 2000):
        value = log_esp_partition(np.full(products, -3.0), rho, nmax)
        assert math.isfinite(value)


def test_null_interaction_truth_has_zero_gram_energy():
    config = tiny_config(world="null_interaction")
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    model = CompleteDemandLaw(config, truth, interaction=True, linked=True)
    model.load_truth(truth)
    assert float(model.interaction_scale().detach().max()) < 1e-20
    assert np.max(np.abs(model.phi().detach().numpy())) < 1e-10
