import numpy as np
import torch

from fit_convex_natural_interactions import (
    evaluate,
    pair_statistic,
    project,
    projected_solve,
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


def test_generation_panel_uses_the_same_nonempty_support_as_the_model():
    data = {
        "trip_split": np.array([1, 1, 1, 0]),
        "trip_nlines": np.array([1, 2, 4, 1]),
    }
    selected = selected_trip_panel(data, 3, nmax=4, seed=8)
    assert set(selected.tolist()) == {0, 1, 2}
