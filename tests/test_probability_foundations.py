import numpy as np
import pytest
import torch

from audit_probability_foundations import (deterministic_audit, ess_adversary,
                                           exact_energy, make_world)
from price_response import (additive_uniform_price_response, changed_price_context,
                            price_jacobian, single_utility_incidence)
from uncertainty import household_cluster_se, paired_score_summary
from audit_probability_foundations import tail_ess_recovery
from fit_joint_price_utility import coefficient_vector
from interaction_particles import differentiable_log_size_beta0


def test_actual_model_dp_price_response_and_gradients_against_enumeration():
    from poly_degree_native import _extension
    if _extension is None:
        pytest.skip("native DP must be built for the integration oracle")
    reports = deterministic_audit(81271, products=7, contexts=3, nmax=4)
    for report in reports:
        assert report["passed"], report


def test_price_action_updates_mean_without_mutating_original_context():
    trip = torch.tensor([0, 0, 1, 1, 1])
    ctx = {"dlp": torch.tensor([.1, .3, -.1, .2, .5]),
           "dlp_bar": torch.tensor([.2, .2]), "disp": torch.zeros(5)}
    delta = torch.tensor([.0, -.2, .1, .0, .2])
    original = ctx["dlp"].clone()
    changed = changed_price_context(ctx, trip, delta)
    assert torch.allclose(changed["dlp_bar"], torch.tensor([.1, .3]))
    assert torch.equal(ctx["dlp"], original)
    changed["disp"].fill_(1)
    assert not ctx["disp"].any()
    with pytest.raises(ValueError, match="finite"):
        changed_price_context(ctx, trip, float("nan"))


def test_single_utility_tilt_handles_forced_absent_and_present_items():
    pi = torch.tensor([[0., .3], [1., .4]])
    pair = torch.tensor([[0., 0.], [1., .4]])
    for d in [-1000., 0., 1000.]:
        updated, ratio = single_utility_incidence(pi, pair, 0, d)
        assert torch.equal(updated, pi)
        assert torch.equal(ratio, torch.tensor([0., d]))
    with pytest.raises(ValueError, match="invalid"):
        single_utility_incidence(torch.tensor([.3, .2]), torch.tensor([.3, .4]), 0, .2)


def test_split_price_jacobian_preserves_uniform_price_direction():
    g = torch.tensor([.1, .4, 1.2])
    for kappa in [.3, 1., 8.]:
        assert torch.allclose(price_jacobian(g, kappa) @ torch.ones(3), -g)


def test_direct_current_price_joint_normalizer_matches_enumeration(native_dp):
    """The SHOPPER-style term must enter numerator and every denominator basket."""
    model, ix, membership, _ = make_world(
        seed=319, products=7, contexts=3, nmax=4, strength=0.0, kappa=1.0)
    model.set_global_price_sensitivity(0.0)
    parameter = torch.tensor([1.3], dtype=torch.float64, requires_grad=True)
    reliable = torch.tensor([True, True, False, True, True, False, True])
    category = torch.arange(model.J) % 3
    sensitivity = coefficient_vector(parameter, "global", reliable, category)
    slot_b = model.b_flat(ix) - sensitivity[ix.item] * model.ctx["dlp"]
    native = torch.logsumexp(differentiable_log_size_beta0(model, ix, slot_b), -1)

    base = exact_energy(model, ix, membership)
    price = torch.zeros(ix.B, model.J, dtype=torch.float64)
    price[ix.item_trip, ix.item] = model.ctx["dlp"]
    dense = torch.logsumexp(
        base - (price * sensitivity).matmul(membership.T), dim=-1)
    assert torch.allclose(native, dense, atol=2e-10, rtol=2e-10)
    native.sum().backward()
    assert torch.isfinite(parameter.grad).all()


def test_direct_price_coefficient_mask_and_category_pooling():
    category = torch.tensor([0, 0, 1, 1, 2])
    reliable = torch.tensor([True, False, True, True, False])
    global_value = coefficient_vector(
        torch.tensor([1.2]), "global", reliable, category)
    category_value = coefficient_vector(
        torch.tensor([.4, 1.1, 2.0]), "category", reliable, category)
    assert torch.equal(global_value, torch.tensor([1.2, 0.0, 1.2, 1.2, 0.0]))
    assert torch.equal(category_value, torch.tensor([.4, 0.0, 1.1, 1.1, 0.0]))


def test_additive_response_rejects_interacting_model():
    model, ix, *_ = make_world(products=7, contexts=3, nmax=4, strength=.2)
    with pytest.raises(ValueError, match="Phi=0"):
        additive_uniform_price_response(model, ix)


def test_tail_ess_rejects_concentrated_three_and_four_draw_bands():
    report = ess_adversary()
    assert report["passed"]
    for row in report["cases"][:3]:
        assert row["minimum_within_band_ess_fraction"] >= .2
        assert row["minimum_within_band_ess"] == 1
        assert row["rejected"]


def test_tail_ess_recovers_on_independent_enumerated_law_bank():
    report = tail_ess_recovery(83021)
    assert not report["legacy_passes_new_gate"]
    assert report["independent_minimum_ess"] >= 8
    assert report["maximum_log_normalizer_ratio_error"] < .08
    assert report["passed"]


def test_cluster_uncertainty_against_independent_household_mean_reference():
    rng = np.random.default_rng(192)
    households, repetitions = 100, 12
    effects = rng.normal(size=households)
    scores = np.repeat(effects, repetitions)
    ids = np.repeat(np.arange(households), repetitions)
    expected = effects.std(ddof=1) / np.sqrt(households)
    assert np.isclose(household_cluster_se(scores, ids), expected)
    report = paired_score_summary(scores, ids)
    assert report["standard_error_method"] == "household_cluster_robust"
    assert report["standard_error"] > 3 * report["trip_naive_standard_error"]
    assert np.isinf(household_cluster_se(scores, np.zeros_like(ids)))
    with pytest.raises(ValueError):
        paired_score_summary([0, float("nan")], [0, 1])
