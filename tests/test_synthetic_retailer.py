import math

import numpy as np
import torch
import pytest

from audit_synthetic_retailer import (Config, QuantityLaw, enumerate_support,
                                      make_truth, simulate_retailer,
                                      solve_budget_policy)
from audit_synthetic_retailer import (BasketLaw, calibrate_size_marginal, dr_contrast)


def test_support_enumerates_all_nonempty_sets_through_nmax():
    support = enumerate_support(7, 3, 3)
    assert len(support["baskets"]) == sum(math.comb(7, n) for n in range(1, 4))
    assert support["membership"].shape == (63, 7)
    assert set(support["sizes"].tolist()) == {1, 2, 3}


def test_truth_is_a_normalized_complete_support_law():
    config = Config(customers=12, products=8, categories=4, segments=3,
                    stores=3, rank=2, nmax=3, days=20, train_days=12,
                    validation_days=4)
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    np.testing.assert_allclose(np.exp(truth["logp"]).sum(1), 1.0, atol=1e-12)
    assert np.isfinite(truth["logp"]).all()


def test_misspecified_truth_remains_a_normalized_probability_law():
    # Eight products is the smoke-profile catalog size and guards against fixed
    # full-profile product indices leaking into the robustness world.
    config = Config(customers=12, products=8, categories=4, segments=3,
                    stores=3, rank=2, nmax=3, days=20, train_days=12,
                    validation_days=4, world="misspecified")
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    np.testing.assert_allclose(np.exp(truth["logp"]).sum(1), 1.0, atol=1e-12)


def test_simulator_contains_explicit_zero_opportunities_and_valid_trip_lines():
    config = Config(customers=30, products=8, categories=4, segments=3,
                    stores=3, rank=2, nmax=3, days=30, train_days=18,
                    validation_days=6, seed=43)
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    result = simulate_retailer(config, support, truth)
    outcome = result["opportunity"]["purchase"]
    assert (outcome == 0).any() and (outcome == 1).any()
    assert len(result["trip_subset"]) == int(outcome.sum())
    assert result["line_trip"].max() < len(result["trip_subset"])
    assert result["line_quantity"].min() >= 1


def test_shifted_negative_binomial_probability_is_normalized():
    config = Config(customers=12, products=8, categories=4, segments=3,
                    stores=3, rank=2, nmax=3, days=20, train_days=12,
                    validation_days=4)
    model = QuantityLaw(config)
    extra = torch.arange(0, 200, dtype=torch.float64)
    item = torch.zeros(200, dtype=torch.long)
    segment = torch.zeros(200, dtype=torch.long)
    discount = torch.zeros(200)
    probability = torch.exp(model.log_probability(extra, item, segment, discount))
    torch.testing.assert_close(probability.sum(), torch.tensor(1.0), atol=1e-10, rtol=0)


def test_budget_policy_respects_budget_and_prefers_reward():
    actions = [
        {"name": "none", "cost": 0.0, "reward": 0.0},
        {"name": "small", "cost": 2.0, "reward": 3.0},
        {"name": "large", "cost": 5.0, "reward": 9.0},
    ]
    result = solve_budget_policy(actions, horizon=3, budget=10.0,
                                 bins=200, minimum_utilization=0.75)
    assert result["feasible"]
    assert result["predicted_cost"] <= 10.0
    assert result["predicted_reward"] == 18.0


def test_randomized_dr_contrast_is_unbiased_even_with_wrong_outcome_model():
    probability = np.array([.4, .3, .3])
    action = np.arange(3)
    outcomes = np.array([3., 5., 7.])
    estimate = dr_contrast(outcomes, action, probability, np.full(3, -14.), np.full(3, 99.), 1)
    assert np.isclose(probability @ estimate, 2.)
    with pytest.raises(ValueError):
        dr_contrast(outcomes, action, probability, outcomes, outcomes, 0)


def test_size_profile_matches_training_moments_without_changing_within_size_odds():
    config = Config(customers=12, products=7, categories=3, segments=3, rank=2, nmax=3)
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    model = BasketLaw(config, truth, interaction=False, seed=19)
    segment = torch.as_tensor(truth["context_segment"])
    action = torch.as_tensor(truth["context_action"])
    count = torch.tensor(np.random.default_rng(31).poisson(4, size=truth["logp"].shape), dtype=torch.float64)
    before = model.log_probability(support, segment, action).detach()
    report = calibrate_size_marginal(model, support, count, segment, action)
    after = model.log_probability(support, segment, action).detach()
    for n in range(1, config.nmax + 1):
        index = torch.nonzero(support["sizes"] == n).flatten()
        change = after[:, index] - before[:, index]
        torch.testing.assert_close(change, change[:, :1].expand_as(change), atol=1e-11, rtol=0)
        observed = count[:, index].sum() / count.sum()
        expected = (count.sum(1) / count.sum()) @ after.exp()[:, index].sum(1)
        torch.testing.assert_close(expected, observed, atol=1e-6, rtol=0)
    assert report["split"] == "train"
