import itertools

import numpy as np
import torch
from scipy.special import logsumexp

from fit_convex_natural_interactions import (
    additive_polish_decision,
    additive_statistic,
    centered_additive_basis,
    evaluate,
    paired_gain_summary,
    pair_statistic,
    project,
    projected_solve,
    split_joint_parameters,
    split_parameters,
)
from audit_particle_counterfactual_generation import selected_trip_panel


def test_pair_statistic_matches_gram_pair_energy():
    torch.manual_seed(17)
    basis, _ = torch.linalg.qr(torch.randn(13, 4, dtype=torch.float64))
    raw = torch.randn(4, 4, dtype=torch.float64)
    c_matrix = raw @ raw.T
    items = torch.tensor([1, 3, 4, 9], dtype=torch.long)
    statistic = pair_statistic(items, basis)
    phi = basis @ torch.linalg.cholesky(c_matrix)
    rows = phi[items]
    direct = 0.5 * ((rows.sum(0).square().sum()) - rows.square().sum())
    assert np.isclose(np.sum(c_matrix.numpy() * statistic), float(direct))


def test_additive_statistic_matches_centred_product_intercept_correction():
    rng = np.random.default_rng(31)
    raw, _ = np.linalg.qr(rng.normal(size=(23, 3)))
    basis = centered_additive_basis(raw)
    items = torch.tensor([1, 3, 3, 8, 17], dtype=torch.long)
    coefficient = np.asarray([0.4, -0.2, 0.1])
    statistic = additive_statistic(items, torch.as_tensor(basis))
    direct = (basis[np.unique(items.numpy())] @ coefficient).sum()
    assert np.isclose(statistic @ coefficient, direct)
    np.testing.assert_allclose(basis.mean(0), 0.0, atol=1e-14)
    np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-12)


def test_projection_enforces_psd_spectrum_and_safe_size_tail():
    rank = 3
    raw = np.array([[2.0, 4.0, -1.0], [-3.0, -2.0, 0.5],
                    [1.0, 0.5, -1.0]])
    vector = np.concatenate((raw.reshape(-1), np.array([-10.0, 0.1])))
    projected = project(vector, rank, spectral_max=0.7, zmax=12.0)
    c_matrix, theta = split_parameters(projected, rank)
    eigenvalues = np.linalg.eigvalsh(c_matrix)
    assert eigenvalues.min() >= -1e-12
    assert eigenvalues.max() <= 0.49 + 1e-12
    assert theta[1] >= 0.0
    assert theta[0] + 12.0 * theta[1] >= -1e-12


def test_joint_projection_preserves_additive_polish_coordinates():
    rank, additive_rank = 2, 2
    raw_c = np.asarray([[1.4, -0.2], [0.6, -0.1]])
    additive = np.asarray([0.35, -0.41])
    size = np.asarray([-9.0, -0.2])
    vector = np.concatenate((raw_c.reshape(-1), additive, size))
    projected = project(
        vector, rank, spectral_max=0.8, zmax=12.0,
        additive_rank=additive_rank)
    c_matrix, got_additive, got_size = split_joint_parameters(
        projected, rank, additive_rank)
    np.testing.assert_allclose(got_additive, additive)
    eigenvalues = np.linalg.eigvalsh(c_matrix)
    assert eigenvalues.min() >= -1e-12
    assert eigenvalues.max() <= 0.64 + 1e-12
    assert got_size[1] >= 0.0
    assert got_size[0] + 12.0 * got_size[1] >= -1e-12


def test_additive_polish_gate_requires_both_folds_and_positive_paired_bound():
    supported = additive_polish_decision(
        np.full(200, 0.03), np.full(180, 0.02))
    assert supported["accepted"]
    one_fold_failure = additive_polish_decision(
        np.full(200, 0.03), np.full(180, -0.001))
    assert not one_fold_failure["accepted"]
    noisy = np.r_[np.full(95, 0.1), np.full(105, -0.09)]
    uncertain = additive_polish_decision(noisy, noisy)
    assert not uncertain["accepted"]


def test_concave_solver_recovers_positive_interaction_signal_monotonically():
    rng = np.random.default_rng(12)
    contexts, draws, rank = 600, 24, 2
    generated = rng.normal(size=(contexts, draws, rank * rank + 2))
    # Symmetric matrix statistics and realistic negative size-statistic columns.
    matrices = generated[..., :rank * rank].reshape(contexts, draws, rank, rank)
    matrices = 0.5 * (matrices + matrices.swapaxes(-1, -2))
    generated[..., :rank * rank] = matrices.reshape(contexts, draws, -1)
    generated[..., -2:] = -np.abs(generated[..., -2:])
    truth = np.zeros(rank * rank + 2)
    truth[:rank * rank] = np.diag([0.35, 0.15]).reshape(-1)
    truth[-2:] = [0.05, 0.02]
    logits = np.einsum("mdp,p->md", generated, truth)
    probabilities = np.exp(logits - logits.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    selected = np.asarray([
        rng.choice(draws, p=probabilities[i]) for i in range(contexts)])
    observed = generated[np.arange(contexts), selected]
    fitted, report = projected_solve(
        observed, generated, rank, spectral_max=1.0, zmax=12.0,
        ridge=1e-3, size_ridge=1e-5, max_iterations=300,
        tolerance=2e-5, label="unit")
    c_matrix, _ = split_parameters(fitted, rank)
    result = evaluate(fitted, observed, generated)
    assert report["converged"]
    assert report["accepted_steps_monotone"]
    assert result["gain"] > 0.01
    assert np.linalg.eigvalsh(c_matrix).max() > 0.05


def test_joint_polish_recovers_additive_redistribution_and_improves_holdout():
    rng = np.random.default_rng(424)
    contexts, heldout, draws, rank, additive_rank = 1800, 1800, 32, 2, 2
    width = rank * rank + additive_rank + 2
    truth = np.zeros(width)
    truth[:rank * rank] = np.diag([0.32, 0.16]).reshape(-1)
    truth[rank * rank:rank * rank + additive_rank] = [0.75, -0.60]
    truth[-2:] = [0.08, 0.025]

    def sample(count):
        candidate = rng.normal(size=(count, draws, width))
        matrix = candidate[..., :rank * rank].reshape(
            count, draws, rank, rank)
        matrix = 0.5 * (matrix + matrix.swapaxes(-1, -2))
        candidate[..., :rank * rank] = matrix.reshape(
            count, draws, rank * rank)
        candidate[..., -2:] = -np.abs(candidate[..., -2:])
        logits = np.einsum("mdp,p->md", candidate, truth)
        probability = np.exp(logits - logits.max(1, keepdims=True))
        probability /= probability.sum(1, keepdims=True)
        selected = np.asarray([
            rng.choice(draws, p=probability[row]) for row in range(count)])
        return candidate[np.arange(count), selected], candidate

    observed, candidates = sample(contexts)
    test_observed, test_candidates = sample(heldout)
    joint, report = projected_solve(
        observed, candidates, rank, spectral_max=1.0, zmax=12.0,
        ridge=1e-3, size_ridge=1e-5, max_iterations=300,
        tolerance=5e-5, label="joint-polish-unit",
        additive_rank=additive_rank, additive_ridge=1e-3)
    staged_columns = np.r_[np.arange(rank * rank),
                           np.arange(width - 2, width)]
    staged, _ = projected_solve(
        observed[:, staged_columns], candidates[:, :, staged_columns],
        rank, spectral_max=1.0, zmax=12.0, ridge=1e-3,
        size_ridge=1e-5, max_iterations=300, tolerance=5e-5,
        label="staged-unit")
    joint_gain = evaluate(joint, test_observed, test_candidates)["gain"]
    staged_gain = evaluate(
        staged, test_observed[:, staged_columns],
        test_candidates[:, :, staged_columns])["gain"]
    _, fitted_additive, _ = split_joint_parameters(
        joint, rank, additive_rank)
    assert report["converged"] and report["accepted_steps_monotone"]
    np.testing.assert_allclose(fitted_additive, [0.75, -0.60], atol=0.08)
    assert joint_gain > staged_gain + 0.30


def test_joint_polish_improves_exact_version4_basket_likelihood():
    """End-to-end audit on every nonempty basket through size four."""
    rng = np.random.default_rng(511)
    products, rank, train, test, draws = 10, 2, 1200, 1200, 64
    baskets = [basket for size in range(1, 5)
               for basket in itertools.combinations(range(products), size)]
    membership = np.zeros((len(baskets), products))
    for row, basket in enumerate(baskets):
        membership[row, list(basket)] = 1.0
    raw = rng.normal(size=(products, rank))
    raw -= raw.mean(0)
    interaction_basis = np.linalg.qr(raw)[0]
    additive_basis = centered_additive_basis(interaction_basis)
    pair = []
    for basket in baskets:
        rows = interaction_basis[list(basket)]
        total = rows.sum(0)
        pair.append(0.5 * (np.outer(total, total) - rows.T @ rows))
    size = membership.sum(1)
    statistic = np.c_[
        np.asarray(pair).reshape(len(baskets), -1),
        membership @ additive_basis,
        -size / 10.0,
        -np.square(size / 10.0),
    ]
    width = statistic.shape[1]
    restricted_columns = np.r_[np.arange(rank * rank),
                               np.arange(width - 2, width)]
    contexts = train + test
    base_utility = -1.8 + 0.25 * rng.normal(size=products)
    contextual_utility = base_utility + 0.3 * rng.normal(
        size=(contexts, products))
    parent_energy = (membership @ contextual_utility.T).T
    parent_energy -= 0.08 * np.square(size - 1.0)[None, :]
    truth = np.zeros(width)
    truth[:rank * rank] = np.diag([0.65, 0.35]).reshape(-1)
    truth[rank * rank:rank * rank + rank] = [1.0, -0.8]
    truth[-2:] = [0.05, 0.02]
    target_energy = parent_energy + statistic @ truth
    observed_index = np.empty(contexts, dtype=np.int64)
    proposal_index = np.empty((train, draws), dtype=np.int64)
    for context in range(contexts):
        target_probability = np.exp(
            target_energy[context] - logsumexp(target_energy[context]))
        observed_index[context] = rng.choice(
            len(baskets), p=target_probability)
        if context < train:
            proposal_probability = np.exp(
                parent_energy[context] - logsumexp(parent_energy[context]))
            proposal_index[context] = rng.choice(
                len(baskets), draws, p=proposal_probability)
    observed = statistic[observed_index[:train]]
    proposals = statistic[proposal_index]
    joint, _ = projected_solve(
        observed, proposals, rank, spectral_max=1.0, zmax=1.0,
        ridge=1e-3, size_ridge=1e-5, max_iterations=300,
        tolerance=3e-3, label="exact-basket-joint",
        additive_rank=rank, additive_ridge=1e-3)
    restricted, _ = projected_solve(
        observed[:, restricted_columns], proposals[:, :, restricted_columns],
        rank, spectral_max=1.0, zmax=1.0, ridge=1e-3,
        size_ridge=1e-5, max_iterations=300, tolerance=3e-3,
        label="exact-basket-restricted")

    def exact_gain(vector, columns):
        feature = statistic[:, columns]
        correction = feature @ vector
        heldout_parent = parent_energy[train:]
        heldout_index = observed_index[train:]
        return (correction[heldout_index]
                - logsumexp(heldout_parent + correction, axis=1)
                + logsumexp(heldout_parent, axis=1))

    joint_gain = exact_gain(joint, np.arange(width))
    restricted_gain = exact_gain(restricted, restricted_columns)
    increment = paired_gain_summary(joint_gain - restricted_gain)
    assert increment["mean"] > 0.09
    assert increment["lower_95"] > 0.07


def test_generation_panel_uses_the_same_nonempty_support_as_the_model():
    data = {
        "trip_split": np.array([1, 1, 1, 0]),
        "trip_nlines": np.array([1, 2, 4, 1]),
    }
    selected = selected_trip_panel(data, 3, nmax=4, seed=8)
    assert set(selected.tolist()) == {0, 1, 2}
