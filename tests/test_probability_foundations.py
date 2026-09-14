import numpy as np
import pytest
import torch

from audit_probability_foundations import deterministic_audit, ess_adversary, make_world
from price_response import (additive_uniform_price_response, changed_price_context,
                            price_jacobian, single_utility_incidence)
from uncertainty import household_cluster_se, paired_score_summary
from audit_probability_foundations import tail_ess_recovery


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
