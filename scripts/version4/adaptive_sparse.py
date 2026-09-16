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

