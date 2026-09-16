"""Tests for the cross-arm transport test's delivery simulation and calibration loss."""
import numpy as np
import torch

import run_erim_tissue_prediction_test as v1
import run_erim_tissue_prediction_test_v2 as v2
import run_erim_tissue_transport_test as transport


def _simulate(delivery, schedule=(20,), replicates=60):
    H, T, J = 150, 20, len(v1.BRANDS)
    weeks = v1.erim_weeks(198533, 198552)
    model = v2.TissueModelV2(H, 3)
    with torch.no_grad():
        model.b.fill_(-2.5)
    return transport.simulate(model, weeks, np.ones((H, T), bool), np.zeros((H, T)), np.zeros(T),
                              np.zeros((J, T)), np.full(T, 1.2), np.zeros(H), schedule, replicates, 5,
                              delivery)


def test_campaign_delivery_scales_redemptions():
    column = transport.ALL_CODES.index(20)
    low = _simulate((0.2, 0.7))["redemptions_by_code"][:, :, column].sum()
    high = _simulate((1.0, 0.95))["redemptions_by_code"][:, :, column].sum()
    assert high > 3 * low > 0


def test_redemptions_by_code_sum_to_campaign_redemptions():
    sim = _simulate((0.8, 0.9), schedule=(20, 36, 38))
    campaign = [transport.ALL_CODES.index(c) for c in (20, 36)]
    assert np.allclose(sim["redemptions_by_code"][:, :, campaign].sum(2), sim["test_coupon_redemptions"])


def test_no_campaign_notice_means_no_campaign_redemptions():
    sim = _simulate((0.0, 0.99))
    assert sim["test_coupon_redemptions"].sum() == 0.0


def test_poisson_deviance_is_zero_only_at_equality():
    observed = np.array([[3.0, 0.0, 5.0]])
    assert transport.poisson_deviance(observed, observed) < 1e-6  # zero cells use a 1e-9 floor
    assert transport.poisson_deviance(observed, observed + 1.0) > 0
