"""Dimension-adaptive sparse quadrature for a standard Gaussian measure.

This module contains no model or optimizer logic.  It implements the quadrature algebra
needed to approximate a scalar *and its derivative numerator* with the same deterministic,
downward-closed index set.  Levels are zero based:

    Q_0: 1-point probabilists' Gauss--Hermite,
    Q_l: (2*l + 1)-point probabilists' Gauss--Hermite,
    Delta_l = Q_l - Q_{l-1}.

For a finite downward-closed multi-index set Lambda, the sparse rule is

    Q_Lambda = sum_{nu in Lambda} tensor_k Delta_{nu_k}.

The non-nested Gauss--Hermite nodes are merged exactly where they coincide (notably zero).
Signed weights are intrinsic to sparse quadrature; callers must use a cancellation-safe
accumulator when the integrand has a large dynamic range.
"""

from __future__ import annotations

import functools
import itertools
from collections.abc import Callable, Iterable

import numpy as np
import torch


Index = tuple[int, ...]


def signed_log_integral(log_values: torch.Tensor, weights: torch.Tensor,
                        *, dim: int = -1) -> tuple[torch.Tensor, torch.Tensor,
                                                         torch.Tensor]:
    """Accumulate ``sum_i weights_i * exp(log_values_i)`` without hiding cancellation.

    Returns ``(log_absolute_value, sign, log_cancellation_condition)``.  The last
    quantity is

    ``log(sum_i |w_i| exp(log_values_i) / |sum_i w_i exp(log_values_i)|)``.

    A valid positive partition estimate requires ``sign == 1``.  A large cancellation
    condition is an explicit numerical failure signal; callers must not clamp a negative or
    nearly cancelled result and continue optimization.
    """
    if weights.ndim != 1:
        raise ValueError("signed_log_integral expects a one-dimensional weight vector")
    dim = dim if dim >= 0 else log_values.ndim + dim
    if dim < 0 or dim >= log_values.ndim or log_values.shape[dim] != weights.numel():
        raise ValueError("integration dimension must match the number of weights")
    if not bool(torch.isfinite(weights).all()) or bool((weights == 0).any()):
        raise ValueError("quadrature weights must be finite and nonzero")

    shape = [1] * log_values.ndim
    shape[dim] = weights.numel()
    weight = weights.to(dtype=log_values.dtype, device=log_values.device).view(shape)
    log_terms = log_values + weight.abs().log()
    negative_inf = torch.full_like(log_terms, -torch.inf)
    log_pos = torch.logsumexp(torch.where(weight > 0, log_terms, negative_inf), dim=dim)
    log_neg = torch.logsumexp(torch.where(weight < 0, log_terms, negative_inf), dim=dim)
    log_total_abs = torch.logsumexp(log_terms, dim=dim)

    positive = log_pos > log_neg
    negative = log_neg > log_pos
    high = torch.where(positive, log_pos, log_neg)
    low = torch.where(positive, log_neg, log_pos)
    # log1p is accurate when the smaller signed mass is close to the larger one.
    ratio = torch.exp(low - high).clamp(max=1.0)
    log_abs = high + torch.log1p(-ratio)
    sign = positive.to(log_values.dtype) - negative.to(log_values.dtype)
    log_abs = torch.where(sign == 0, torch.full_like(log_abs, -torch.inf), log_abs)
    log_condition = log_total_abs - log_abs
    return log_abs, sign, log_condition


def _node_key(value: float) -> float:
    """Canonical key shared by independently generated Hermite rules."""
    return round(float(value), 14)


@functools.lru_cache(maxsize=None)
def gaussian_rule(level: int) -> tuple[tuple[float, float], ...]:
    """Return ``(node, weight)`` pairs for ``N(0,1)`` at a zero-based level."""
    if level < 0:
        raise ValueError("Gaussian quadrature level must be non-negative")
    count = 2 * int(level) + 1
    nodes, weights = np.polynomial.hermite_e.hermegauss(count)
    weights = weights / np.sqrt(2.0 * np.pi)
    return tuple((_node_key(x), float(w)) for x, w in zip(nodes, weights))


@functools.lru_cache(maxsize=None)
def gaussian_difference(level: int) -> tuple[tuple[float, float], ...]:
    """Return the signed one-dimensional difference rule ``Delta_level``."""
    acc: dict[float, float] = {}
    for node, weight in gaussian_rule(level):
        acc[node] = acc.get(node, 0.0) + weight
    if level > 0:
        for node, weight in gaussian_rule(level - 1):
            acc[node] = acc.get(node, 0.0) - weight
    return tuple((node, weight) for node, weight in sorted(acc.items())
                 if abs(weight) > 1e-15)


def validate_index(index: Index, dimension: int | None = None) -> Index:
    index = tuple(int(level) for level in index)
    if dimension is not None and len(index) != int(dimension):
        raise ValueError(f"index has dimension {len(index)}, expected {dimension}")
    if not index or any(level < 0 for level in index):
        raise ValueError("a sparse-grid index must be nonempty and non-negative")
    return index


def is_downward_closed(indices: Iterable[Index], dimension: int | None = None) -> bool:
    index_set = {validate_index(index, dimension) for index in indices}
    if not index_set:
        return False
    d = len(next(iter(index_set)))
    if any(len(index) != d for index in index_set):
        return False
    for index in index_set:
        for axis, level in enumerate(index):
            if level > 0:
                parent = list(index)
                parent[axis] -= 1
                if tuple(parent) not in index_set:
                    return False
    return True


def admissible_forward_neighbors(indices: Iterable[Index]) -> tuple[Index, ...]:
    """Indices that can be added while preserving downward closure."""
    index_set = {validate_index(index) for index in indices}
    if not is_downward_closed(index_set):
        raise ValueError("forward neighbors require a downward-closed index set")
    d = len(next(iter(index_set)))
    candidates: set[Index] = set()
    for index in index_set:
        for axis in range(d):
            child = list(index)
            child[axis] += 1
            child = tuple(child)
            if child in index_set:
                continue
            admissible = True
            for parent_axis, level in enumerate(child):
                if level <= 0:
                    continue
                parent = list(child)
                parent[parent_axis] -= 1
                if tuple(parent) not in index_set:
                    admissible = False
                    break
            if admissible:
                candidates.add(child)
    return tuple(sorted(candidates))


@functools.lru_cache(maxsize=None)
def tensor_difference(index: Index) -> tuple[tuple[tuple[float, ...], float], ...]:
    """Tensor product of the one-dimensional hierarchical difference rules."""
    index = validate_index(index)
    rules = [gaussian_difference(level) for level in index]
    out = []
    for entries in itertools.product(*rules):
        node = tuple(entry[0] for entry in entries)
        weight = float(np.prod([entry[1] for entry in entries]))
        if abs(weight) > 1e-15:
            out.append((node, weight))
    return tuple(out)


def sparse_rule(indices: Iterable[Index], *, dtype=torch.float64,
                device: torch.device | str | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Assemble and merge the Gaussian sparse rule for ``indices``."""
    index_set = {validate_index(index) for index in indices}
    if not is_downward_closed(index_set):
        raise ValueError("sparse rule requires a nonempty downward-closed index set")
    acc: dict[tuple[float, ...], float] = {}
    for index in sorted(index_set):
        for node, weight in tensor_difference(index):
            acc[node] = acc.get(node, 0.0) + weight
    items = [(node, weight) for node, weight in sorted(acc.items())
             if abs(weight) > 1e-14]
    nodes = torch.tensor([node for node, _ in items], dtype=dtype, device=device)
    weights = torch.tensor([weight for _, weight in items], dtype=dtype, device=device)
    return nodes, weights


def tensor_difference_integral(index: Index, evaluator: Callable[[torch.Tensor], torch.Tensor],
                               *, dtype=torch.float64) -> torch.Tensor:
    """Evaluate a scalar- or vector-valued hierarchical surplus."""
    entries = tensor_difference(validate_index(index))
    nodes = torch.tensor([node for node, _ in entries], dtype=dtype)
    weights = torch.tensor([weight for _, weight in entries], dtype=dtype)
    values = evaluator(nodes)
    if values.shape[0] != nodes.shape[0]:
        raise ValueError("evaluator's leading dimension must equal the number of nodes")
    shape = (weights.shape[0],) + (1,) * (values.ndim - 1)
    return (values * weights.view(shape)).sum(0)


def greedy_index_sequence(dimension: int, surplus: Callable[[Index], torch.Tensor],
                          indicator: Callable[[torch.Tensor], float], *, max_indices: int,
                          tolerance: float = 0.0) -> tuple[tuple[Index, ...],
                                                          dict[Index, float]]:
    """Construct a goal-oriented downward-closed sequence by hierarchical surplus.

    ``surplus(index)`` may be Hilbert/vector valued.  ``indicator`` maps it to a scalar
    error proxy, allowing callers to combine value and parameter-block score errors.  The
    largest admissible indicator is refined first.  This routine deliberately returns the
    complete indicator history: an estimator is not certified merely because its current
    integral happens to be stable.
    """
    if dimension <= 0 or max_indices <= 0 or tolerance < 0:
        raise ValueError("dimension/max_indices must be positive and tolerance non-negative")
    zero = (0,) * int(dimension)
    accepted: set[Index] = {zero}
    sequence = [zero]
    scores: dict[Index, float] = {zero: float(indicator(surplus(zero)))}
    while len(accepted) < int(max_indices):
        frontier = admissible_forward_neighbors(accepted)
        if not frontier:
            break
        for index in frontier:
            if index not in scores:
                scores[index] = float(indicator(surplus(index)))
        best = max(frontier, key=lambda index: scores[index])
        if scores[best] <= tolerance:
            break
        accepted.add(best)
        sequence.append(best)
    return tuple(sequence), scores


def greedy_index_set(dimension: int, surplus: Callable[[Index], torch.Tensor],
                     indicator: Callable[[torch.Tensor], float], *, max_indices: int,
                     tolerance: float = 0.0) -> tuple[tuple[Index, ...], dict[Index, float]]:
    """Return the sorted set selected by :func:`greedy_index_sequence`."""
    sequence, scores = greedy_index_sequence(
        dimension, surplus, indicator, max_indices=max_indices, tolerance=tolerance)
    return tuple(sorted(sequence)), scores


def ratio_score_error_bound(value_relative_error: float, numerator_scaled_error: float,
                            reference_score_norm: float) -> float:
    """Bound the score-ratio error from value and numerator integration errors.

    If ``|Zhat-Z| <= eps_z Z`` and ``||Nhat-N|| <= eps_n Z``, then for
    ``g=N/Z`` this returns ``(eps_n + eps_z*||g||)/(1-eps_z)``.
    """
    eps_z = float(value_relative_error)
    eps_n = float(numerator_scaled_error)
    score = float(reference_score_norm)
    if not (0.0 <= eps_z < 1.0) or eps_n < 0.0 or score < 0.0:
        raise ValueError("score error bound requires eps_z in [0,1) and nonnegative norms")
    return (eps_n + eps_z * score) / (1.0 - eps_z)


def multifidelity_corrected_score(low_current: torch.Tensor, low_anchor: torch.Tensor,
                                  high_anchor: torch.Tensor) -> torch.Tensor:
    """Deterministic anchored correction ``g_L(theta)+g_H(theta0)-g_L(theta0)``."""
    if low_current.shape != low_anchor.shape or low_current.shape != high_anchor.shape:
        raise ValueError("multifidelity score tensors must have identical shapes")
    return low_current + high_anchor - low_anchor


def multifidelity_drift_bound(jacobian_discrepancy_bound: float,
                              parameter_displacement: float,
                              high_anchor_error_bound: float = 0.0) -> float:
    """High-target score error bound for an anchored low/high correction.

    If ``||J_L-J_H|| <= delta_J`` on the path from ``theta0`` to ``theta`` and the
    high rule itself differs from the exact score by at most ``eps_H``, the corrected score
    differs from the exact score by at most ``eps_H + delta_J*||theta-theta0||``.
    """
    jacobian = float(jacobian_discrepancy_bound)
    displacement = float(parameter_displacement)
    anchor_error = float(high_anchor_error_bound)
    if jacobian < 0 or displacement < 0 or anchor_error < 0:
        raise ValueError("multifidelity error bounds must be non-negative")
    return anchor_error + jacobian * displacement
