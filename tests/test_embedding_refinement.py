import itertools

import numpy as np
import pytest
from scipy.special import logsumexp

from basket_incidence import incidence, off_diagonal_operator, pair_energy, pair_adjoint
from embedding_refinement import IncidenceBank, objective_gradient, fit_embeddings, evaluation, compact_factor
from pilot_embedding_refinement import projected_bank
from stratified_natural import likelihood_gain


def all_baskets(products):
    return [s for n in range(1, products + 1) for s in itertools.combinations(range(products), n)]


def oracle_bank(seed=15, products=6, contexts=100):
    rng = np.random.default_rng(seed)
    baskets = all_baskets(products)
    x = incidence(baskets, products)
    utility = rng.normal(scale=0.3, size=products) - 0.9
    energy = x @ utility
    logp = energy - logsumexp(energy)
    observed = rng.choice(len(baskets), contexts, p=np.exp(logp))
    return IncidenceBank(x[observed], incidence(baskets * contexts, products),
                         np.tile(logp, (contexts, 1)), np.zeros(len(baskets), dtype=int),
                         np.eye(products)), x, logp


def test_incidence_deduplicates_and_matrix_free_matches_explicit_pairs():
    x = incidence([[0, 1, 1], [1, 2], [0, 2, 3], [3]], 4)
    matrix = (x.T @ x).toarray() / x.shape[0]
    np.fill_diagonal(matrix, 0)
    operator = off_diagonal_operator(x)
    np.testing.assert_allclose(operator @ np.eye(4), matrix)
    np.testing.assert_allclose(operator @ np.arange(4.), matrix @ np.arange(4.))
    np.testing.assert_allclose(operator.rmatvec(np.arange(4.)), matrix.T @ np.arange(4.))


def test_energy_gradient_excludes_diagonal_and_is_rotation_invariant():
    rng = np.random.default_rng(2)
    baskets = all_baskets(6)
    x = incidence(baskets, 6)
    phi = rng.normal(size=(6, 2))
    expected = [sum(phi[i] @ phi[j] for i, j in itertools.combinations(b, 2)) for b in baskets]
    np.testing.assert_allclose(pair_energy(x, phi), expected, atol=1e-13)
    q, _ = np.linalg.qr(rng.normal(size=(2, 2)))
    np.testing.assert_allclose(pair_energy(x, phi @ q), expected, atol=1e-13)
    weight, direction = rng.normal(size=len(baskets)), rng.normal(size=phi.shape)
    eps = 1e-6
    fd = weight @ (pair_energy(x, phi + eps * direction) - pair_energy(x, phi - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(fd, np.sum(pair_adjoint(x, phi, weight) * direction), rtol=1e-8)
    np.testing.assert_allclose(pair_energy(incidence([[0], [1]], 6), phi), 0, atol=1e-14)


def test_full_objective_gradient_and_normalizer_match_exact_enumeration():
    bank, x, logp = oracle_bank()
    rng = np.random.default_rng(17)
    phi, size = rng.normal(scale=.2, size=(6, 2)), rng.normal(scale=.1, size=6)
    anchor = rng.normal(scale=.1, size=(6, 2))
    dp, ds = rng.normal(size=phi.shape), rng.normal(size=size.shape)
    value, gp, gs = objective_gradient(bank, phi, size, anchor)
    eps = 1e-6
    plus = objective_gradient(bank, phi + eps * dp, size + eps * ds, anchor)[0]
    minus = objective_gradient(bank, phi - eps * dp, size - eps * ds, anchor)[0]
    np.testing.assert_allclose((plus - minus) / (2 * eps), np.sum(gp * dp) + gs @ ds, rtol=1e-7)
    increment = pair_energy(x, phi) - size[np.asarray(x.sum(1)).ravel().astype(int) - 1]
    exact_ratio = logsumexp(logp + increment)
    observed, _ = bank.increments(phi, size)
    np.testing.assert_allclose(bank.gain(phi, size), observed - exact_ratio, atol=1e-13)
    np.testing.assert_allclose(bank.gain(np.zeros_like(phi), np.zeros_like(size)), 0, atol=1e-13)


def test_projected_statistics_equal_free_embedding_likelihood():
    bank, _, _ = oracle_bank()
    rng = np.random.default_rng(7)
    basis, _ = np.linalg.qr(rng.normal(size=(6, 2)))
    factor = rng.normal(scale=.3, size=(2, 2))
    size = rng.normal(scale=.1, size=6)
    natural = projected_bank(bank, basis)
    vector = np.r_[(factor @ factor.T).ravel(), size]
    np.testing.assert_allclose(likelihood_gain(vector, natural), bank.gain(basis @ factor, size), atol=1e-13)


def test_free_refinement_learns_outside_initial_span_and_improves_exact_law():
    rng = np.random.default_rng(713)
    baskets = all_baskets(6)
    x = incidence(baskets, 6)
    base = np.asarray(x @ np.full(6, -1.0)).ravel()
    logp = base - logsumexp(base)
    truth = np.asarray([.7, .7, .9, .9, -.6, -.6])[:, None]
    target = logp + pair_energy(x, truth)
    target -= logsumexp(target)
    observed = rng.choice(len(baskets), 1200, p=np.exp(target))
    bank = IncidenceBank(x[observed], incidence(baskets * len(observed), 6),
                         np.tile(logp, (len(observed), 1)), np.zeros(len(baskets), dtype=int), np.eye(6))
    basis = np.asarray([1., 1., 0, 0, 0, 0])[:, None] / np.sqrt(2)
    initial = basis * .6
    kwargs = dict(steps=200, fit_size=False, spectral_max=2., anchor_ridge=.001,
                  gram_ridge=.0001, size_ridge=0., size_smoothness=0.)
    fixed, _, fixed_report = fit_embeddings(bank, initial, basis=basis, **kwargs)
    free, _, free_report = fit_embeddings(bank, fixed, **kwargs)

    def exact_kl(phi):
        fitted = logp + pair_energy(x, phi)
        fitted -= logsumexp(fitted)
        return float(np.exp(target) @ (target - fitted))

    assert exact_kl(free) < exact_kl(fixed) * .5
    assert np.linalg.norm(free - basis @ (basis.T @ free)) > .1
    assert free_report["monotone"] and fixed_report["monotone"]
    assert free_report["spectral_norm"] <= 2 + 1e-12


def test_zero_phi_has_zero_gradient_and_must_not_be_used_as_free_initialization():
    bank, _, _ = oracle_bank()
    phi = np.zeros((6, 2))
    _, gradient, _ = objective_gradient(bank, phi, np.zeros(6), phi)
    np.testing.assert_array_equal(gradient, np.zeros_like(phi))


def test_evaluation_fails_ess_when_one_effective_draw_dominates():
    bank, _, _ = oracle_bank(contexts=10)
    phi = np.full((6, 1), 10.)
    result = evaluation(bank, phi, np.zeros(6), np.arange(10))
    assert not result["ess_passed"]
    assert result["minimum_within_band_ess"] < 2


def test_invalid_bank_weights_fail_closed():
    bank, _, _ = oracle_bank()
    with pytest.raises(ValueError, match="sum to one"):
        IncidenceBank(bank.observed, bank.generated, bank.log_weight + .1, bank.band, bank.size_basis)


def test_saved_factor_compacts_active_columns_without_changing_gram():
    phi = np.zeros((6, 3))
    phi[:, 2] = np.arange(6) / 10
    compact = compact_factor(phi)
    assert compact.shape == (6, 1)
    np.testing.assert_allclose(compact @ compact.T, phi @ phi.T, atol=1e-14)
    assert compact_factor(np.zeros_like(phi)).shape == (6, 0)


def test_rank_zero_quadrature_is_one_exact_base_node():
    import torch
    from pipeline_support import smolyak_rule
    class Model:
        Kz = 8
        phi = torch.zeros(6, 8, dtype=torch.float64)
    nodes, weight = smolyak_rule(Model(), 0, 2)
    assert nodes.shape == (1, 8)
    assert torch.count_nonzero(nodes) == 0
    np.testing.assert_array_equal(weight.numpy(), [1.])
