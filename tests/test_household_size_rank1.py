import numpy as np
import torch

from ragged import RaggedModel
from fit_household_size_rank1 import (
    cap_households, normalized_tilt, solve_households)


def test_rank_one_household_channel_is_common_across_products():
    model = RaggedModel(
        J=5, N=3, C=1, K=3, Kz=1, nmax=5,
        household_size_rank1=True)
    with torch.no_grad():
        model.lam.zero_()
        model.alpha[:, :-1].copy_(torch.tensor([
            [1.0, -0.5], [0.0, 0.5], [-1.0, 1.0],
            [0.5, -1.0], [-0.5, 0.0],
        ]))
        model.alpha[:, -1].fill_(27.0)  # ignored; the loading is structurally one
        model.theta.zero_()
        model.theta[:, -1].copy_(torch.tensor([0.3, -0.1, -0.2]))
        model.house = torch.tensor([0, 1])

    item = torch.arange(5).repeat(2)
    trip = torch.arange(2).repeat_interleave(5)
    value = model.b_at(item, trip, None).reshape(2, 5)
    assert torch.allclose(value[0], torch.full((5,), 0.3))
    assert torch.allclose(value[1], torch.full((5,), -0.1))


def test_projection_preserves_rank_one_forward_utility():
    generator = torch.Generator().manual_seed(41)
    model = RaggedModel(
        J=7, N=4, C=1, K=4, Kz=1, nmax=7,
        household_size_rank1=True).double()
    with torch.no_grad():
        model.alpha.copy_(torch.randn(7, 4, generator=generator))
        model.theta.copy_(torch.randn(4, 4, generator=generator))
        model.house = torch.arange(4)
    item = torch.arange(7).repeat(4)
    trip = torch.arange(4).repeat_interleave(7)
    before = model.b_at(item, trip, None)
    with torch.no_grad():
        model.project_context_gauges()
    after = model.b_at(item, trip, None)
    assert torch.allclose(before, after, atol=1e-12, rtol=1e-12)
    assert torch.allclose(
        model.alpha[:, :-1].mean(0), torch.zeros(3, dtype=torch.float64),
        atol=1e-12)
    assert torch.equal(
        model.alpha[:, -1], torch.ones(7, dtype=torch.float64))


def test_common_utility_changes_only_size_marginal_not_fixed_size_composition():
    base = np.log(np.asarray([0.20, 0.35, 0.25, 0.15, 0.05]))
    size = np.arange(1, 6, dtype=np.float64)
    kappa = 0.17
    tilted = base + kappa * size
    tilted -= np.logaddexp.reduce(tilted)

    # Every basket of the same size receives the same n*kappa energy increment.  It
    # therefore changes P(N=n) by exponential tilting but cancels from P(S | N=n).
    expected = np.exp(base + kappa * size)
    expected /= expected.sum()
    assert np.allclose(np.exp(tilted), expected)
    assert np.isclose((2 * kappa) - (2 * kappa), 0.0)


def test_concave_household_solve_matches_observed_mean_and_cap_is_monotone():
    base = np.log(np.asarray([
        [0.55, 0.30, 0.10, 0.04, 0.01],
        [0.50, 0.30, 0.13, 0.05, 0.02],
        [0.60, 0.25, 0.10, 0.04, 0.01],
        [0.45, 0.30, 0.15, 0.07, 0.03],
    ]))
    household = np.asarray([0, 0, 1, 1])
    observed = np.asarray([2, 3, 4, 5])
    got = solve_households(
        base, observed, household, np.arange(4), n_household=2, ridge=0.0)
    tilted = np.exp(normalized_tilt(base, got, household))
    size = np.arange(1, 6)
    assert np.allclose(
        [(tilted[household == h] @ size).sum() for h in range(2)],
        [observed[household == h].sum() for h in range(2)],
        atol=1e-9)

    # Use a two-item tail in this toy support by padding to the production threshold.
    padded = np.full((4, 120), -1e6)
    padded[:, :5] = base
    padded[:, 59] = np.log(1e-3)
    padded -= np.logaddexp.reduce(padded, axis=1)[:, None]
    unsafe = np.asarray([0.2, 0.2])
    safe, upper = cap_households(
        padded, household, unsafe, n_household=2, screen_tail_cap=0.35)
    probability = np.exp(normalized_tilt(padded, safe, household))
    assert probability[:, 59:].sum(1).max() <= 0.35 + 1e-10
    assert np.all(safe <= unsafe)
    assert np.isfinite(upper).any()
