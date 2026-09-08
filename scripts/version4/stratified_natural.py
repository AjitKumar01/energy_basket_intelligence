"""Sparse concave natural-parameter objective for size-stratified MC likelihood.

This module contains no data-loading or model mutation.  It represents one frozen draw
bank for the Version-4 likelihood-ratio identity and optimizes the affine block

    K = U C U',       Delta rho_c,       Delta rho_0(1:nmax).

The draw weights may be ordinary Monte Carlo weights or the exact band weights returned
by ``conditional_slots_stratified``.  Category and size sufficient statistics remain
sparse, so enlarging rho_0 from two directions to its declared table does not create a
dense draw-by-parameter tensor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass(frozen=True)
class StratifiedNaturalBank:
    observed_pair: np.ndarray
    draw_pair: np.ndarray
    observed_size: np.ndarray
    draw_size: np.ndarray
    observed_category: sparse.csr_matrix
    draw_category: sparse.csr_matrix
    log_draw_weight: np.ndarray
    n_size: int
    size_basis: np.ndarray | None = None

    def __post_init__(self) -> None:
        op = np.asarray(self.observed_pair, dtype=np.float64)
        dp = np.asarray(self.draw_pair, dtype=np.float64)
        os = np.asarray(self.observed_size, dtype=np.int64)
        ds = np.asarray(self.draw_size, dtype=np.int64)
        lw = np.asarray(self.log_draw_weight, dtype=np.float64)
        if op.ndim != 2 or dp.ndim != 3 or dp.shape[0] != op.shape[0]:
            raise ValueError("pair statistics must be [M,P] and [M,D,P]")
        if dp.shape[2] != op.shape[1]:
            raise ValueError("observed and draw pair widths differ")
        if os.shape != (op.shape[0],) or ds.shape != dp.shape[:2]:
            raise ValueError("size statistics do not match the pair draw bank")
        if lw.shape != dp.shape[:2]:
            raise ValueError("log draw weights must have shape [M,D]")
        if np.any(os < 0) or np.any(os >= self.n_size):
            raise ValueError("observed size index is outside 0..n_size-1")
        if np.any(ds < 0) or np.any(ds >= self.n_size):
            raise ValueError("draw size index is outside 0..n_size-1")
        if np.any(np.isnan(lw)) or np.any(np.isposinf(lw)):
            raise ValueError("draw weights contain invalid values")
        if not np.all(np.isfinite(logsumexp(lw, axis=1))):
            raise ValueError("every context needs positive total draw weight")
        oc = sparse.csr_matrix(self.observed_category, dtype=np.float64)
        dc = sparse.csr_matrix(self.draw_category, dtype=np.float64)
        size_basis = (np.eye(self.n_size, dtype=np.float64)
                      if self.size_basis is None
                      else np.asarray(self.size_basis, dtype=np.float64))
        if size_basis.ndim != 2 or size_basis.shape[0] != self.n_size:
            raise ValueError("size_basis must have one row per supported size")
        if not np.isfinite(size_basis).all():
            raise ValueError("size_basis contains non-finite values")
        if oc.shape[0] != op.shape[0]:
            raise ValueError("observed category rows do not match contexts")
        if dc.shape != (op.shape[0] * dp.shape[1], oc.shape[1]):
            raise ValueError("draw category rows must be flattened context-major")
        object.__setattr__(self, "observed_pair", op)
        object.__setattr__(self, "draw_pair", dp)
        object.__setattr__(self, "observed_size", os)
        object.__setattr__(self, "draw_size", ds)
        object.__setattr__(self, "observed_category", oc)
        object.__setattr__(self, "draw_category", dc)
        object.__setattr__(self, "log_draw_weight", lw)
        object.__setattr__(self, "size_basis", size_basis)

    @property
    def contexts(self) -> int:
        return self.observed_pair.shape[0]

    @property
    def draws(self) -> int:
        return self.draw_pair.shape[1]

    @property
    def pair_width(self) -> int:
        return self.observed_pair.shape[1]

    @property
    def categories(self) -> int:
        return self.observed_category.shape[1]

    @property
    def width(self) -> int:
        return self.pair_width + self.categories + self.size_width

    @property
    def size_width(self) -> int:
        return self.size_basis.shape[1]

    def subset(self, rows: np.ndarray) -> "StratifiedNaturalBank":
        rows = np.asarray(rows, dtype=np.int64)
        flat = (rows[:, None] * self.draws
                + np.arange(self.draws, dtype=np.int64)[None, :]).reshape(-1)
        return StratifiedNaturalBank(
            self.observed_pair[rows], self.draw_pair[rows],
            self.observed_size[rows], self.draw_size[rows],
            self.observed_category[rows], self.draw_category[flat],
            self.log_draw_weight[rows], self.n_size, self.size_basis)


def linear_size_basis(n_size: int, knots: np.ndarray | list[int]) -> np.ndarray:
    """Piecewise-linear interpolation basis over sizes ``1..n_size``.

    Coefficients are the correction values at the declared knots.  Every intermediate
    row is a convex combination of adjacent knot coefficients, so box constraints on the
    coefficients also bound the complete size curve.  The first knot must be size one and
    is fixed to zero by the optimizer's gauge.
    """
    knots = np.asarray(knots, dtype=np.int64)
    if (knots.ndim != 1 or len(knots) < 2 or knots[0] != 1
            or knots[-1] != n_size or np.any(knots[1:] <= knots[:-1])):
        raise ValueError("knots must increase strictly from 1 through n_size")
    basis = np.zeros((n_size, len(knots)), dtype=np.float64)
    for size in range(1, n_size + 1):
        right = int(np.searchsorted(knots, size, side="left"))
        if right == 0:
            basis[size - 1, 0] = 1.0
        elif right == len(knots):
            basis[size - 1, -1] = 1.0
        elif knots[right] == size:
            basis[size - 1, right] = 1.0
        else:
            left = right - 1
            fraction = (size - knots[left]) / (knots[right] - knots[left])
            basis[size - 1, left] = 1.0 - fraction
            basis[size - 1, right] = fraction
    return basis


def split_parameters(vector: np.ndarray, bank: StratifiedNaturalBank
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vector = np.asarray(vector, dtype=np.float64)
    if vector.shape != (bank.width,):
        raise ValueError(f"expected {bank.width} natural parameters")
    p = bank.pair_width
    c = bank.categories
    return vector[:p], vector[p:p + c], vector[p + c:]


def project_parameters(vector: np.ndarray, bank: StratifiedNaturalBank, rank: int,
                       spectral_max: float, category_bound: float,
                       size_bound: float) -> np.ndarray:
    pair, category, size = split_parameters(vector, bank)
    if rank * rank != bank.pair_width:
        raise ValueError("rank is incompatible with the pair statistic width")
    matrix = pair.reshape(rank, rank)
    matrix = 0.5 * (matrix + matrix.T)
    value, direction = np.linalg.eigh(matrix)
    value = np.clip(value, 0.0, spectral_max * spectral_max)
    matrix = (direction * value[None, :]) @ direction.T
    category = np.clip(category, -category_bound, category_bound)
    size = np.clip(size, -size_bound, size_bound)
    # Size one fixes the common-utility/linear-size gauge for the increment.
    size[0] = 0.0
    return np.concatenate((matrix.reshape(-1), category, size))


def _second_difference_penalty(size: np.ndarray) -> tuple[float, np.ndarray]:
    if len(size) < 3:
        return 0.0, np.zeros_like(size)
    difference = size[2:] - 2.0 * size[1:-1] + size[:-2]
    gradient = np.zeros_like(size)
    gradient[:-2] += difference
    gradient[1:-1] -= 2.0 * difference
    gradient[2:] += difference
    return 0.5 * float(difference @ difference), gradient


def objective_gradient(
        vector: np.ndarray, bank: StratifiedNaturalBank, *,
        interaction_ridge: float, category_ridge: float, size_ridge: float,
        size_smoothness: float
        ) -> tuple[float, np.ndarray, np.ndarray]:
    """Penalized objective, exact gradient, and positive diagonal scale.

    The diagonal scale is only a preconditioner.  Armijo acceptance is always evaluated
    against the exact fixed-bank objective, so an approximate scale cannot change the
    optimizer's target.
    """
    pair, category, size = split_parameters(vector, bank)
    m, d = bank.contexts, bank.draws
    observed_logit = bank.observed_pair @ pair
    observed_logit += np.asarray(bank.observed_category @ category).reshape(-1)
    size_curve = bank.size_basis @ size
    observed_logit -= size_curve[bank.observed_size]
    draw_logit = np.einsum("mdp,p->md", bank.draw_pair, pair, optimize=True)
    draw_logit += np.asarray(bank.draw_category @ category).reshape(m, d)
    draw_logit -= size_curve[bank.draw_size]
    weighted_logit = bank.log_draw_weight + draw_logit
    log_ratio = logsumexp(weighted_logit, axis=1)
    objective = float(np.mean(observed_logit - log_ratio))
    weight = np.exp(weighted_logit - log_ratio[:, None])

    pair_expectation = np.einsum("md,mdp->mp", weight, bank.draw_pair,
                                  optimize=True)
    pair_gradient = np.mean(bank.observed_pair - pair_expectation, axis=0)
    flat_weight = weight.reshape(-1)
    category_observed = np.asarray(bank.observed_category.mean(axis=0)).reshape(-1)
    category_expected = np.asarray(
        bank.draw_category.T @ flat_weight).reshape(-1) / m
    category_gradient = category_observed - category_expected
    expected_size = np.zeros(bank.n_size, dtype=np.float64)
    np.add.at(expected_size, bank.draw_size.reshape(-1), flat_weight)
    expected_size /= m
    observed_size = np.bincount(
        bank.observed_size, minlength=bank.n_size).astype(np.float64) / m
    size_probability_gradient = expected_size - observed_size

    smooth_value, smooth_curve_gradient = _second_difference_penalty(size_curve)
    objective -= 0.5 * interaction_ridge * float(pair @ pair)
    objective -= 0.5 * category_ridge * float(category @ category)
    objective -= 0.5 * size_ridge * float(size_curve @ size_curve)
    objective -= size_smoothness * smooth_value
    pair_gradient -= interaction_ridge * pair
    category_gradient -= category_ridge * category
    size_gradient = bank.size_basis.T @ (
        size_probability_gradient - size_ridge * size_curve
        - size_smoothness * smooth_curve_gradient)
    gradient = np.concatenate((pair_gradient, category_gradient, size_gradient))
    # The gauge coordinate is fixed by projection; zeroing its search direction avoids a
    # useless component in the projected-gradient convergence check.
    gradient[bank.pair_width + bank.categories] = 0.0

    pair_second = np.einsum(
        "md,mdp->mp", weight, np.square(bank.draw_pair), optimize=True)
    pair_scale = np.mean(pair_second - np.square(pair_expectation), axis=0)
    category_second = np.asarray(
        bank.draw_category.power(2).T @ flat_weight).reshape(-1) / m
    category_scale = category_second - np.square(category_expected)
    size_probability_scale = expected_size * (1.0 - expected_size)
    size_scale = np.square(bank.size_basis).T @ size_probability_scale
    scale = np.concatenate((pair_scale, category_scale, size_scale))
    scale[:bank.pair_width] += interaction_ridge
    scale[bank.pair_width:bank.pair_width + bank.categories] += category_ridge
    scale[bank.pair_width + bank.categories:] += size_ridge + size_smoothness
    return objective, gradient, np.maximum(scale, 1e-8)


def likelihood_gain(vector: np.ndarray, bank: StratifiedNaturalBank) -> np.ndarray:
    pair, category, size = split_parameters(vector, bank)
    m, d = bank.contexts, bank.draws
    observed = bank.observed_pair @ pair
    observed += np.asarray(bank.observed_category @ category).reshape(-1)
    size_curve = bank.size_basis @ size
    observed -= size_curve[bank.observed_size]
    draw = np.einsum("mdp,p->md", bank.draw_pair, pair, optimize=True)
    draw += np.asarray(bank.draw_category @ category).reshape(m, d)
    draw -= size_curve[bank.draw_size]
    return observed - logsumexp(bank.log_draw_weight + draw, axis=1)


def draw_increment(vector: np.ndarray, bank: StratifiedNaturalBank) -> np.ndarray:
    """Return ``h(S_draw)`` without the fixed proposal/stratum weights."""
    pair, category, size = split_parameters(vector, bank)
    value = np.einsum("mdp,p->md", bank.draw_pair, pair, optimize=True)
    value += np.asarray(bank.draw_category @ category).reshape(
        bank.contexts, bank.draws)
    value -= (bank.size_basis @ size)[bank.draw_size]
    return value


def evaluation_summary(vector: np.ndarray, bank: StratifiedNaturalBank,
                       band_of_draw: np.ndarray) -> dict:
    """Paired gain and within-stratum weight diagnostics.

    Overall importance ESS is not meaningful for deliberately unequal stratum weights.
    The relevant diagnostic is composition coverage *inside* each size stratum.
    """
    gain = likelihood_gain(vector, bank)
    increment = draw_increment(vector, bank)
    band_of_draw = np.asarray(band_of_draw, dtype=np.int64)
    if band_of_draw.shape != (bank.draws,):
        raise ValueError("band_of_draw must have one entry per draw")
    rows = []
    all_fraction = []
    for band in np.unique(band_of_draw):
        selected = band_of_draw == band
        logits = increment[:, selected]
        probability = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        ess = 1.0 / np.square(probability).sum(axis=1)
        fraction = ess / int(selected.sum())
        # A context can have zero parent mass in this band. It has -inf fixed draw weights
        # and does not contribute to the ratio; exclude it from a composition diagnostic.
        active = np.isfinite(bank.log_draw_weight[:, selected]).any(axis=1)
        value = fraction[active]
        if len(value):
            all_fraction.append(value)
            rows.append({
                "band": int(band), "draws": int(selected.sum()),
                "active_contexts": int(active.sum()),
                "ess_fraction_min": float(value.min()),
                "ess_fraction_p01": float(np.quantile(value, 0.01)),
                "ess_fraction_median": float(np.median(value)),
            })
    pooled = np.concatenate(all_fraction) if all_fraction else np.asarray([0.0])
    return {
        "gain": float(gain.mean()),
        "gain_standard_error": float(gain.std(ddof=1) / np.sqrt(len(gain))),
        "gain_lower_95": float(gain.mean() - 1.96 * gain.std(ddof=1) / np.sqrt(len(gain))),
        "minimum_within_band_ess_fraction": float(pooled.min()),
        "p01_within_band_ess_fraction": float(np.quantile(pooled, 0.01)),
        "median_within_band_ess_fraction": float(np.median(pooled)),
        "bands": rows,
    }


def projected_solve(
        bank: StratifiedNaturalBank, rank: int, *, spectral_max: float = 1.0,
        category_bound: float = 0.25, size_bound: float = 12.0,
        interaction_ridge: float = 1e-3, category_ridge: float = 1e-3,
        size_ridge: float = 1e-6, size_smoothness: float = 1e-3,
        max_iterations: int = 300, tolerance: float = 1e-3,
        label: str = "stratified-natural"
        ) -> tuple[np.ndarray, dict]:
    vector = np.zeros(bank.width, dtype=np.float64)
    objective, gradient, scale = objective_gradient(
        vector, bank, interaction_ridge=interaction_ridge,
        category_ridge=category_ridge, size_ridge=size_ridge,
        size_smoothness=size_smoothness)
    initial = objective
    history = [objective]
    step = 1.0
    converged = False
    projected_norm = float("inf")
    for iteration in range(1, max_iterations + 1):
        unit = project_parameters(
            vector + gradient, bank, rank, spectral_max,
            category_bound, size_bound) - vector
        projected_norm = float(np.linalg.norm(unit))
        if projected_norm <= tolerance:
            converged = True
            break
        seed = gradient / scale
        accepted = False
        trial = step
        for _ in range(40):
            candidate = project_parameters(
                vector + trial * seed, bank, rank, spectral_max,
                category_bound, size_bound)
            direction = candidate - vector
            directional = float(gradient @ direction)
            if directional <= 0.0:
                candidate = project_parameters(
                    vector + trial * gradient, bank, rank, spectral_max,
                    category_bound, size_bound)
                direction = candidate - vector
                directional = float(gradient @ direction)
            candidate_objective, candidate_gradient, candidate_scale = objective_gradient(
                candidate, bank, interaction_ridge=interaction_ridge,
                category_ridge=category_ridge, size_ridge=size_ridge,
                size_smoothness=size_smoothness)
            if candidate_objective >= objective + 1e-4 * directional:
                vector = candidate
                objective = candidate_objective
                gradient = candidate_gradient
                scale = candidate_scale
                history.append(objective)
                step = min(1.5 * trial, 100.0)
                accepted = True
                break
            trial *= 0.5
        if not accepted:
            raise RuntimeError(f"{label}: Armijo line search failed")
        if iteration % 10 == 0:
            print(f"[{label}] iter={iteration} objective={objective:.8f} "
                  f"projected_grad={projected_norm:.3e}", flush=True)
    monotone = bool(np.all(np.diff(np.asarray(history)) >= -1e-12))
    if not converged:
        raise RuntimeError(f"{label}: solve did not converge")
    return vector, {
        "iterations": int(iteration), "converged": converged,
        "initial_penalized_objective": initial,
        "final_penalized_objective": objective,
        "projected_gradient_norm": projected_norm,
        "accepted_steps_monotone": monotone,
    }


def alternating_solve(
        bank: StratifiedNaturalBank, rank: int, *, spectral_max: float = 1.0,
        category_bound: float = 0.25, size_bound: float = 12.0,
        interaction_ridge: float = 1e-3, category_ridge: float = 1e-3,
        size_ridge: float = 1e-6, size_smoothness: float = 1e-3,
        max_outer_iterations: int = 20, pair_steps: int = 100,
        nuisance_iterations: int = 300, tolerance: float = 1e-3,
        label: str = "stratified-alternating"
        ) -> tuple[np.ndarray, dict]:
    """Globally optimize the same concave target with two well-scaled blocks.

    Bounded L-BFGS solves the unconstrained category/full-size block (apart from boxes and
    the fixed size-one gauge).  Projected Armijo ascent solves the small PSD interaction
    block.  Exact block maximization/monotone ascent on a differentiable concave objective
    over a compact convex product set converges to its global solution set; no Cholesky
    parameterization or nonconvex surrogate is introduced.
    """
    kwargs = dict(interaction_ridge=interaction_ridge,
                  category_ridge=category_ridge, size_ridge=size_ridge,
                  size_smoothness=size_smoothness)
    vector = np.zeros(bank.width, dtype=np.float64)
    objective, gradient, scale = objective_gradient(vector, bank, **kwargs)
    initial = objective
    history = [objective]
    pair_width = bank.pair_width
    nuisance_start = pair_width
    # rho_0(1) is omitted from the nuisance coordinates because it is the exact gauge.
    free_nuisance = np.r_[np.arange(nuisance_start,
                                    nuisance_start + bank.categories),
                          np.arange(nuisance_start + bank.categories + 1,
                                    bank.width)]
    bounds = ([(-category_bound, category_bound)] * bank.categories
              + [(-size_bound, size_bound)] * (bank.size_width - 1))
    step = 1.0
    converged = False
    nuisance_reports = []
    pair_accepted_steps = 0
    projected_norm = float("inf")
    for outer in range(1, max_outer_iterations + 1):
        fixed_pair = vector[:pair_width].copy()

        def nuisance_value_gradient(free):
            candidate = vector.copy()
            candidate[:pair_width] = fixed_pair
            candidate[free_nuisance] = free
            value, candidate_gradient, _ = objective_gradient(
                candidate, bank, **kwargs)
            return -value, -candidate_gradient[free_nuisance]

        result = minimize(
            nuisance_value_gradient, vector[free_nuisance], method="L-BFGS-B",
            jac=True, bounds=bounds,
            options={"maxiter": nuisance_iterations,
                     "ftol": 1e-12, "gtol": tolerance * 0.1,
                     "maxls": 40})
        candidate = vector.copy()
        candidate[free_nuisance] = result.x
        candidate_objective, candidate_gradient, candidate_scale = objective_gradient(
            candidate, bank, **kwargs)
        if candidate_objective < objective - 1e-10:
            raise RuntimeError(f"{label}: nuisance block decreased the objective")
        vector, objective = candidate, candidate_objective
        gradient, scale = candidate_gradient, candidate_scale
        history.append(objective)
        nuisance_reports.append({
            "outer_iteration": outer, "success": bool(result.success),
            "status": int(result.status), "message": str(result.message),
            "iterations": int(result.nit), "function_evaluations": int(result.nfev),
        })

        # Optimize only C. The nuisance coordinates remain at their conditional optimum.
        for _ in range(pair_steps):
            pair_seed = gradient[:pair_width] / scale[:pair_width]
            unit_vector = vector.copy()
            unit_vector[:pair_width] += gradient[:pair_width]
            projected_pair = project_parameters(
                unit_vector, bank, rank, spectral_max,
                category_bound, size_bound)[:pair_width]
            pair_direction = projected_pair - vector[:pair_width]
            if np.linalg.norm(pair_direction) <= tolerance * 0.25:
                break
            trial = step
            accepted = False
            for _ in range(40):
                candidate = vector.copy()
                raw = vector.copy()
                raw[:pair_width] += trial * pair_seed
                candidate[:pair_width] = project_parameters(
                    raw, bank, rank, spectral_max,
                    category_bound, size_bound)[:pair_width]
                direction = candidate[:pair_width] - vector[:pair_width]
                directional = float(gradient[:pair_width] @ direction)
                if directional <= 0.0:
                    raw = vector.copy()
                    raw[:pair_width] += trial * gradient[:pair_width]
                    candidate[:pair_width] = project_parameters(
                        raw, bank, rank, spectral_max,
                        category_bound, size_bound)[:pair_width]
                    direction = candidate[:pair_width] - vector[:pair_width]
                    directional = float(gradient[:pair_width] @ direction)
                if np.linalg.norm(direction) <= tolerance * 1e-3:
                    accepted = True
                    break
                value, candidate_gradient, candidate_scale = objective_gradient(
                    candidate, bank, **kwargs)
                if value >= objective + 1e-4 * directional:
                    vector, objective = candidate, value
                    gradient, scale = candidate_gradient, candidate_scale
                    history.append(objective)
                    step = min(1.5 * trial, 100.0)
                    accepted = True
                    pair_accepted_steps += 1
                    break
                trial *= 0.5
            if not accepted:
                raise RuntimeError(f"{label}: pair-block Armijo line search failed")
            if np.linalg.norm(direction) <= tolerance * 1e-3:
                break

        unit = project_parameters(
            vector + gradient, bank, rank, spectral_max,
            category_bound, size_bound) - vector
        projected_norm = float(np.linalg.norm(unit))
        print(f"[{label}] outer={outer} objective={objective:.8f} "
              f"projected_grad={projected_norm:.3e}", flush=True)
        if projected_norm <= tolerance:
            converged = True
            break
    monotone = bool(np.all(np.diff(np.asarray(history)) >= -1e-10))
    if not monotone:
        raise RuntimeError(f"{label}: accepted block steps were not monotone")
    if not converged:
        raise RuntimeError(f"{label}: alternating solve did not converge")
    return vector, {
        "iterations": int(outer), "converged": converged,
        "initial_penalized_objective": initial,
        "final_penalized_objective": objective,
        "projected_gradient_norm": projected_norm,
        "accepted_steps_monotone": monotone,
        "pair_accepted_steps": pair_accepted_steps,
        "nuisance_solves": nuisance_reports,
    }
