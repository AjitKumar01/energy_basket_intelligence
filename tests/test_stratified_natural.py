import itertools

import numpy as np
import torch
from scipy import sparse

from ragged import RaggedIndex, RaggedModel
from stratified_natural import (
    StratifiedNaturalBank,
    alternating_solve,
    likelihood_gain,
    linear_size_basis,
    objective_gradient,
    project_parameters,
    projected_solve,
)
from tempered_block_gibbs import (
    _exact_conditional_bernoulli,
    conditional_slots_fixed_sizes,
    conditional_slots_stratified,
    default_size_bands,
)


def _toy_bank(seed=8, contexts=80, draws=24, rank=2, categories=3, n_size=7):
    rng = np.random.default_rng(seed)
    pair_width = rank * rank
    draw_pair = rng.normal(size=(contexts, draws, pair_width))
    matrix = draw_pair.reshape(contexts, draws, rank, rank)
    matrix = 0.5 * (matrix + matrix.swapaxes(-1, -2))
    draw_pair = matrix.reshape(contexts, draws, pair_width)
    draw_size = rng.integers(0, n_size, size=(contexts, draws))
    cat_dense = -rng.binomial(3, 0.2, size=(contexts * draws, categories)).astype(float)
    log_weight = rng.normal(scale=0.3, size=(contexts, draws))
    log_weight -= np.logaddexp.reduce(log_weight, axis=1)[:, None]

    truth_pair = np.diag([0.25, 0.12]).reshape(-1)
    truth_category = np.asarray([0.04, -0.03, 0.02])
    truth_size = np.asarray([0.0, -0.06, -0.08, -0.04, 0.04, 0.12, 0.22])
    draw_logit = np.einsum("mdp,p->md", draw_pair, truth_pair)
    draw_logit += (cat_dense @ truth_category).reshape(contexts, draws)
    draw_logit -= truth_size[draw_size]
    probability = np.exp(log_weight + draw_logit)
    probability /= probability.sum(1, keepdims=True)
    selected = np.asarray([
        rng.choice(draws, p=probability[row]) for row in range(contexts)])
    observed_pair = draw_pair[np.arange(contexts), selected]
    observed_size = draw_size[np.arange(contexts), selected]
    observed_category = cat_dense[
        np.arange(contexts) * draws + selected]
    return StratifiedNaturalBank(
        observed_pair, draw_pair, observed_size, draw_size,
        sparse.csr_matrix(observed_category), sparse.csr_matrix(cat_dense),
        log_weight, n_size, linear_size_basis(n_size, [1, 3, 5, n_size]))


def test_default_size_bands_cover_support_once():
    assert default_size_bands(120) == [
        (1, 4), (5, 10), (11, 20), (21, 40), (41, 59), (60, 80), (81, 120)]
    assert default_size_bands(17) == [(1, 4), (5, 10), (11, 17)]


def test_exact_conditional_bernoulli_fallback_has_the_declared_law():
    logits = np.asarray([-1.3, -0.2, 0.4, 1.1])
    draws = _exact_conditional_bernoulli(
        logits, 2, 20000, np.random.default_rng(912))
    assert np.all(draws.sum(axis=0) == 2)
    subsets = list(itertools.combinations(range(len(logits)), 2))
    weight = np.asarray([np.exp(logits[list(subset)].sum()) for subset in subsets])
    probability = weight / weight.sum()
    empirical = np.asarray([
        np.mean(np.all(draws[list(subset)], axis=0)) for subset in subsets])
    np.testing.assert_allclose(empirical, probability, atol=0.012)


def test_linear_size_basis_interpolates_knots_and_preserves_bounds():
    basis = linear_size_basis(12, [1, 3, 7, 12])
    coefficient = np.asarray([0.0, -0.4, 0.8, 0.2])
    curve = basis @ coefficient
    np.testing.assert_allclose(curve[[0, 2, 6, 11]], coefficient)
    assert np.allclose(basis.sum(1), 1.0)
    assert curve.min() >= coefficient.min() and curve.max() <= coefficient.max()


def test_sparse_objective_gradient_matches_finite_differences_and_is_concave():
    bank = _toy_bank()
    rng = np.random.default_rng(91)
    left = project_parameters(
        rng.normal(scale=0.05, size=bank.width), bank, 2, 1.0, 0.25, 2.0)
    right = project_parameters(
        rng.normal(scale=0.05, size=bank.width), bank, 2, 1.0, 0.25, 2.0)
    kwargs = dict(interaction_ridge=2e-3, category_ridge=3e-3,
                  size_ridge=1e-4, size_smoothness=4e-3)
    value, gradient, _ = objective_gradient(left, bank, **kwargs)
    direction = rng.normal(size=bank.width)
    # Respect the fixed size-one gauge in the directional derivative.
    direction[bank.pair_width + bank.categories] = 0.0
    epsilon = 1e-6
    plus = objective_gradient(left + epsilon * direction, bank, **kwargs)[0]
    minus = objective_gradient(left - epsilon * direction, bank, **kwargs)[0]
    assert np.isclose((plus - minus) / (2 * epsilon), gradient @ direction,
                      rtol=2e-5, atol=2e-6)

    midpoint = 0.37 * left + 0.63 * right
    mid_value = objective_gradient(midpoint, bank, **kwargs)[0]
    right_value = objective_gradient(right, bank, **kwargs)[0]
    assert mid_value >= 0.37 * value + 0.63 * right_value - 1e-12


def test_sparse_projected_solver_is_monotone_and_improves_likelihood():
    bank = _toy_bank(contexts=900, draws=32)
    vector, report = projected_solve(
        bank, 2, category_bound=0.25, size_bound=2.0,
        interaction_ridge=1e-3, category_ridge=1e-3,
        size_ridge=1e-4, size_smoothness=1e-3,
        max_iterations=400, tolerance=2e-4, label="stratified-unit")
    assert report["converged"] and report["accepted_steps_monotone"]
    assert likelihood_gain(vector, bank).mean() > 0.01


def test_block_solver_reaches_same_concave_target_more_directly():
    bank = _toy_bank(contexts=900, draws=32)
    vector, report = alternating_solve(
        bank, 2, category_bound=0.25, size_bound=2.0,
        interaction_ridge=1e-3, category_ridge=1e-3,
        size_ridge=1e-4, size_smoothness=1e-3,
        max_outer_iterations=12, pair_steps=80, nuisance_iterations=200,
        tolerance=4e-4, label="stratified-block-unit")
    assert report["converged"] and report["accepted_steps_monotone"]
    assert likelihood_gain(vector, bank).mean() > 0.01


def test_block_solver_omits_frozen_category_coordinates():
    source = _toy_bank(contexts=500, draws=24)
    bank = StratifiedNaturalBank(
        source.observed_pair, source.draw_pair, source.observed_size,
        source.draw_size, sparse.csr_matrix((source.contexts, 0)),
        sparse.csr_matrix((source.contexts * source.draws, 0)),
        source.log_draw_weight, source.n_size, source.size_basis)
    assert bank.categories == 0
    assert bank.width == bank.pair_width + bank.size_width
    vector, report = alternating_solve(
        bank, 2, category_bound=0.0, size_bound=2.0,
        interaction_ridge=1e-3, category_ridge=1e-3,
        size_ridge=1e-3, size_smoothness=1e-1,
        max_outer_iterations=12, pair_steps=80, nuisance_iterations=200,
        tolerance=5e-4, label="stratified-no-category-unit")
    assert report["converged"] and report["accepted_steps_monotone"]
    assert np.isfinite(likelihood_gain(vector, bank)).all()


def test_size_stratification_is_unbiased_and_detects_rare_tail_at_equal_draw_cost():
    rng = np.random.default_rng(709)
    parent = np.asarray([0.72, 0.20, 0.079, 0.001])
    ratio_by_size = np.asarray([1.0, 1.1, 1.4, 900.0])
    exact = float(parent @ ratio_by_size)
    repetitions, draws = 3000, 40
    ordinary = np.empty(repetitions)
    stratified = np.empty(repetitions)
    # Four bands with the same total of 40 draws.  Every band is represented; allocation
    # can later be optimized from an independent pilot without changing this expectation.
    allocation = np.asarray([14, 10, 8, 8])
    for repetition in range(repetitions):
        sample = rng.choice(4, size=draws, p=parent)
        ordinary[repetition] = ratio_by_size[sample].mean()
        stratified[repetition] = sum(
            parent[n] * ratio_by_size[n]  # constant within this toy size stratum
            for n in range(4))
    assert abs(stratified.mean() - exact) < 1e-12
    assert abs(ordinary.mean() - exact) < 0.12
    assert stratified.var() < ordinary.var() * 1e-6
    assert np.quantile(ordinary, 0.5) < exact * 0.6  # ordinary banks usually miss size 4


def test_stratified_estimator_remains_accurate_along_adversarial_parameter_path():
    """The check spans a trajectory, rather than certifying one frozen checkpoint."""
    rng = np.random.default_rng(1771)
    size_mass = np.asarray([0.72, 0.20, 0.079, 0.001])
    conditional = np.asarray([
        [0.35, 0.25, 0.20, 0.12, 0.08],
        [0.28, 0.24, 0.20, 0.16, 0.12],
        [0.22, 0.21, 0.20, 0.19, 0.18],
        [0.10, 0.15, 0.20, 0.25, 0.30],
    ])
    # The rare largest-size stratum becomes important as the interaction path advances;
    # composition remains variable inside every size.
    statistic = np.asarray([
        [-0.2, -0.1, 0.0, 0.1, 0.2],
        [0.0, 0.1, 0.2, 0.3, 0.4],
        [0.2, 0.4, 0.6, 0.8, 1.0],
        [5.8, 6.1, 6.4, 6.7, 7.0],
    ])
    joint_parent = size_mass[:, None] * conditional
    flat_parent = joint_parent.reshape(-1)
    allocation = np.asarray([14, 10, 8, 8])
    repetitions = 1800
    for scale in (0.0, 0.5, 1.0, 1.5):
        ratio = np.exp(scale * statistic)
        exact = float(np.sum(joint_parent * ratio))
        ordinary, stratified = [], []
        for _ in range(repetitions):
            flat = rng.choice(flat_parent.size, size=allocation.sum(), p=flat_parent)
            ordinary.append(float(ratio.reshape(-1)[flat].mean()))
            estimate = 0.0
            for size, count in enumerate(allocation):
                composition = rng.choice(5, size=count, p=conditional[size])
                estimate += size_mass[size] * float(ratio[size, composition].mean())
            stratified.append(estimate)
        ordinary = np.asarray(ordinary)
        stratified = np.asarray(stratified)
        # Monte Carlo means agree with the exact value throughout the path.
        stratified_se = stratified.std(ddof=1) / np.sqrt(repetitions)
        assert abs(stratified.mean() - exact) <= 4.0 * stratified_se + 1e-12
        if scale >= 0.5:
            ordinary_rmse = np.sqrt(np.mean(np.square(ordinary - exact)))
            stratified_rmse = np.sqrt(np.mean(np.square(stratified - exact)))
            assert stratified_rmse < ordinary_rmse * 0.35


def _toy_model_and_index():
    model = RaggedModel(J=6, N=1, C=2, K=2, Kz=2, nmax=4, R=4).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.lam.copy_(torch.tensor([-0.8, -0.4, -0.1, 0.1, 0.3, 0.6]))
        model.rho_0_free.copy_(torch.tensor([0.0, 0.2, 0.6, 1.1]))
        model.rho_c.copy_(torch.tensor([0.15, -0.05]))
    model.house = torch.zeros(1, dtype=torch.long)
    model.ctx = None
    index = RaggedIndex(
        np.arange(6), np.asarray([0, 0, 0, 1, 1, 1]),
        np.asarray([0, 0]), np.asarray([0, 1]), 1)
    return model, index


def test_fixed_size_and_stratified_reverse_samplers_respect_declared_sizes():
    model, index = _toy_model_and_index()
    generator = torch.Generator().manual_seed(201)
    z = torch.zeros(1, model.Kz, dtype=torch.float64)
    requested = np.asarray([[1], [2], [3], [4], [2], [4]])
    fixed = conditional_slots_fixed_sizes(
        model, index, z, 0.0, requested, generator)
    assert [len(draw[0]) for draw in fixed] == requested[:, 0].tolist()
    assert all(len(torch.unique(draw[0])) == len(draw[0]) for draw in fixed)

    bands = [(1, 1), (2, 2), (3, 4)]
    states, log_weight, band = conditional_slots_stratified(
        model, index, z, 0.0, bands, [3, 3, 4], generator)
    sizes = np.asarray([len(draw[0]) for draw in states])
    for draw, band_index in enumerate(band):
        lo, hi = bands[band_index]
        assert lo <= sizes[draw] <= hi
    assert np.isclose(float(torch.logsumexp(log_weight[0], 0)), 0.0,
                      atol=1e-12)


def test_stratified_ratio_matches_exact_enumeration_in_expectation():
    """Audit the complete sampling identity, including within-size composition."""
    model, index = _toy_model_and_index()
    baskets = [basket for n in range(1, 5)
               for basket in itertools.combinations(range(6), n)]
    parent_energy = []
    child_increment = []
    c = np.asarray([[0.28, 0.04], [0.04, 0.16]])
    basis = np.asarray([
        [-.42, .10], [-.25, -.20], [-.08, .35],
        [.10, -.32], [.27, .18], [.38, -.11]])
    size_delta = np.asarray([0.0, -0.05, 0.08, 0.25])
    lam = model.lam.detach().numpy()
    rho0 = model.rho_0_free.detach().numpy()
    rhoc = model.rho_c.detach().numpy()
    category = np.asarray([0, 0, 0, 1, 1, 1])
    for basket in baskets:
        basket = np.asarray(basket)
        counts = np.bincount(category[basket], minlength=2)
        parent_energy.append(
            lam[basket].sum() - rho0[len(basket) - 1]
            - np.sum(rhoc * counts * (counts - 1) / 2))
        rows = basis[basket]
        pair = 0.5 * (np.outer(rows.sum(0), rows.sum(0)) - rows.T @ rows)
        child_increment.append(np.sum(c * pair) - size_delta[len(basket) - 1])
    parent_energy = np.asarray(parent_energy)
    child_increment = np.asarray(child_increment)
    parent_probability = np.exp(parent_energy - np.logaddexp.reduce(parent_energy))
    exact_ratio = float(parent_probability @ np.exp(child_increment))

    estimates = []
    z = torch.zeros(1, model.Kz, dtype=torch.float64)
    for seed in range(120):
        states, log_weight, _ = conditional_slots_stratified(
            model, index, z, 0.0, [(1, 1), (2, 2), (3, 3), (4, 4)],
            [8, 8, 8, 8], torch.Generator().manual_seed(800 + seed))
        increments = []
        for draw in states:
            item = index.item[draw[0]].numpy()
            rows = basis[item]
            pair = 0.5 * (np.outer(rows.sum(0), rows.sum(0)) - rows.T @ rows)
            increments.append(np.sum(c * pair) - size_delta[len(item) - 1])
        estimates.append(float(np.exp(
            log_weight[0].numpy() + np.asarray(increments)).sum()))
    assert abs(np.mean(estimates) - exact_ratio) < 0.02
