"""Price actions and responses of the conditional Version-4 basket law.

These are distributional calculations, not assertions of causal identification. The
additive response uses fourth-order symmetric differences of exact DP expectations;
both its approximation and its numerical fidelity check are explicit.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch


def changed_price_context(context, item_trip: torch.Tensor, log_change) -> dict:
    """Apply slot-level log-price changes and update the actual assortment mean."""
    change = torch.as_tensor(log_change, dtype=context["dlp"].dtype,
                             device=context["dlp"].device)
    change = torch.broadcast_to(change, context["dlp"].shape)
    if item_trip.shape != change.shape or not bool(torch.isfinite(change).all()):
        raise ValueError("finite price changes must match the assortment slots")
    result = {key: value.clone() if torch.is_tensor(value) else value
              for key, value in context.items()}
    result["dlp"] = context["dlp"] + change
    if "dlp_bar" in context:
        batches = context["dlp_bar"].numel()
        counts = torch.bincount(item_trip, minlength=batches).to(change.dtype)
        if bool((counts == 0).any()):
            raise ValueError("every price context must have an offered product")
        total = torch.zeros_like(context["dlp_bar"]).index_add(0, item_trip, change)
        result["dlp_bar"] = context["dlp_bar"] + total / counts
    return result


def price_jacobian(coefficient: torch.Tensor, kappa=1.0) -> torch.Tensor:
    """d b_j / d log p_k for one offered catalogue; retain the common-price term."""
    if coefficient.ndim != 1 or not coefficient.numel():
        raise ValueError("one nonempty offered coefficient vector is required")
    kappa = torch.as_tensor(kappa, dtype=coefficient.dtype, device=coefficient.device)
    if (not bool(torch.isfinite(coefficient).all()) or bool((coefficient < 0).any())
            or kappa.numel() != 1 or not bool(torch.isfinite(kappa)) or float(kappa) <= 0):
        raise ValueError("price coefficients must be nonnegative and kappa positive")
    return (-kappa * torch.diag(coefficient)
            - (1.0 - kappa) / coefficient.numel() * coefficient[:, None]
            * torch.ones_like(coefficient)[None, :])


def single_utility_incidence(incidence: torch.Tensor, pair_column: torch.Tensor,
                             item: int, increment) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact finite single-utility tilt from factual incidences and one pair column.

    Returns new incidences and log(Z_new/Z_old). A real one-price action qualifies only
    if its utility map changes that one item (e.g. kappa=1). Inputs may have batch axes.
    """
    if incidence.shape != pair_column.shape or not 0 <= item < incidence.shape[-1]:
        raise ValueError("incidences and pair column must have matching item axes")
    p = incidence[..., item]
    d = torch.as_tensor(increment, dtype=p.dtype, device=p.device)
    if (not bool(torch.isfinite(incidence).all())
            or not bool(torch.isfinite(pair_column).all())
            or not bool(torch.isfinite(d).all())
            or bool(((incidence < 0) | (incidence > 1)).any())
            or not torch.allclose(pair_column[..., item], p, atol=1e-10, rtol=1e-10)
            or bool((pair_column < -1e-10).any())
            or bool((pair_column > torch.minimum(incidence, p[..., None]) + 1e-10).any())
            or bool((pair_column < incidence + p[..., None] - 1.0 - 1e-10).any())):
        raise ValueError("invalid factual Bernoulli moments or utility increment")
    # Scale both weights before forming the ratio, avoiding exp(d) overflow.
    scale = torch.clamp_min(d, 0.0)
    absent, present = torch.exp(-scale), torch.exp(d - scale)
    denominator = absent * (1.0 - p) + present * p
    boundary = (p == 0) | (p == 1)
    safe = torch.where(boundary, torch.ones_like(denominator), denominator)
    result = (absent[..., None] * (incidence - pair_column)
              + present[..., None] * pair_column) / safe[..., None]
    result = torch.where(boundary[..., None], incidence, result)
    log_ratio = torch.log(safe) + scale
    log_ratio = torch.where(p == 0, torch.zeros_like(log_ratio), log_ratio)
    log_ratio = torch.where(p == 1, d.expand_as(log_ratio), log_ratio)
    return result, log_ratio


@dataclass
class AdditivePriceResponse:
    elasticity: torch.Tensor
    size_slope: torch.Tensor
    second_fourth_discrepancy: torch.Tensor
    step: float


def additive_uniform_price_response(model, ix, size_probability=None, *,
                                    step=1e-3, atol=1e-6, rtol=2e-4
                                    ) -> AdditivePriceResponse:
    """Differentiable heterogeneous-price response through the exact additive DP.

    D4=(4 D(h)-D(2h))/3 estimates -Cov(N, sum_j g_hj Y_j), not
    -mean(g)*Var(N). Only ordinary first DP adjoints are needed, unlike differentiating
    through its once-differentiable custom adjoint. The energy perturbation is bounded
    to keep truncation small; disagreement with D(h) fails closed. This check is an
    empirical numerical diagnostic, not an exact-error bound.
    """
    from interaction_particles import differentiable_log_size_beta0

    if not math.isfinite(step) or step <= 0 or atol < 0 or rtol < 0:
        raise ValueError("positive finite step and nonnegative tolerances required")
    if float(model.phi.detach().abs().max()) != 0.0:
        raise ValueError("additive price response requires Phi=0")
    base = model.b_flat(ix)
    coefficient = model.price_coefficients(ix.item, ix.item_trip)
    if base.dtype != torch.float64:
        raise ValueError("price-response differentiation requires float64")
    maximum = float(coefficient.detach().max()) * model.nmax
    actual_step = min(step, 0.01 / max(maximum, 1.0))
    if actual_step < 1e-8:
        raise FloatingPointError("price coefficients exceed the audited difference envelope")
    if size_probability is None:
        size_probability = differentiable_log_size_beta0(model, ix, base).softmax(-1)
    grid = torch.arange(1, size_probability.shape[-1] + 1, dtype=base.dtype,
                        device=base.device)
    means = {}
    for multiplier in (-2, -1, 1, 2):
        probability = differentiable_log_size_beta0(
            model, ix, base - multiplier * actual_step * coefficient).softmax(-1)
        means[multiplier] = probability @ grid
    second = (means[1] - means[-1]) / (2 * actual_step)
    coarse = (means[2] - means[-2]) / (4 * actual_step)
    fourth = (4 * second - coarse) / 3
    discrepancy = (fourth - second).abs()
    if (not bool(torch.isfinite(fourth).all())
            or bool((discrepancy > atol + rtol * fourth.abs()).any())):
        raise FloatingPointError("uniform-price finite-difference fidelity failed")
    elasticity = fourth.mean() / (size_probability @ grid).mean()
    return AdditivePriceResponse(elasticity, fourth, discrepancy, actual_step)
