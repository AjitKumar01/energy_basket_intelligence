"""Counterfactual price contexts and the aggregate price elasticity.

A counterfactual changes the features and recomputes the law: same model, same trips,
different context.  Changing one product's price also moves the trip's mean price
deviation (``dlp_bar``) by its share of the assortment, because the price term is relative
to that mean (see model.py).
"""
from __future__ import annotations

import torch

from .exact import additive_log_size


def changed_price_context(context, item_trip, log_change, discount_change=None):
    """A copy of ``context`` with slot log prices changed by ``log_change``.

    ``discount_change`` is the matching change in discount depth for models with a
    discount panel; None treats every change as a regular-price change."""
    change = torch.broadcast_to(torch.as_tensor(log_change, dtype=context["dlp"].dtype),
                                context["dlp"].shape)
    if item_trip.shape != change.shape or not bool(torch.isfinite(change).all()):
        raise ValueError("finite price changes must match the assortment slots")
    result = {k: v.clone() if torch.is_tensor(v) else v for k, v in context.items()}
    result["dlp"] = context["dlp"] + change
    counts = torch.bincount(item_trip, minlength=context["dlp_bar"].numel()).to(change.dtype)
    if bool((counts == 0).any()):
        raise ValueError("every trip must offer a product")
    total = torch.zeros_like(context["dlp_bar"]).index_add(0, item_trip, change)
    result["dlp_bar"] = context["dlp_bar"] + total / counts
    if discount_change is not None and "dsc" in context:
        result["dsc"] = context["dsc"] + torch.broadcast_to(
            torch.as_tensor(discount_change, dtype=context["dsc"].dtype), context["dsc"].shape)
    return result


def uniform_price_elasticity(model, ix, size_probability=None, *, step=1e-3, atol=1e-6, rtol=2e-4):
    """d E[size] / d log(all prices) divided by E[size], through the exact additive DP.

    A uniform price change shifts every utility by -g_hj (kappa cancels: the relative part
    does not move).  The derivative is a fourth-order central difference,
    (4 D(h) - D(2h)) / 3; if it disagrees with the second-order one beyond the tolerance
    the step is too coarse and this raises.  Used to anchor the base model's aggregate
    elasticity during the additive fit (--elast-w)."""
    if float(model.phi.detach().abs().max()) != 0.0:
        raise ValueError("the additive price response needs phi = 0")
    base = model.slot_utilities(ix)
    g = model.price_coefficients(ix.item, ix.item_trip)
    h = min(step, 0.01 / max(float(g.detach().max()) * model.nmax, 1.0))
    if h < 1e-8:
        raise FloatingPointError("price coefficients are too large for the difference step")
    if size_probability is None:
        size_probability = additive_log_size(model, ix, base).softmax(-1)
    n = torch.arange(1, size_probability.shape[-1] + 1, dtype=base.dtype)
    mean = {m: additive_log_size(model, ix, base - m * h * g).softmax(-1) @ n for m in (-2, -1, 1, 2)}
    second = (mean[1] - mean[-1]) / (2 * h)
    fourth = (4 * second - (mean[2] - mean[-2]) / (4 * h)) / 3
    if (not bool(torch.isfinite(fourth).all())
            or bool(((fourth - second).abs() > atol + rtol * fourth.abs()).any())):
        raise FloatingPointError("uniform-price finite differences disagree")
    return fourth.mean() / (size_probability @ n).mean()


def own_price_tilt(p, du):
    """Exact incidence of one product after only its own utility changes by du:
    p e^du / (1 - p + p e^du), computed without overflow."""
    scale = torch.clamp_min(du, 0.0)
    num = p * torch.exp(du - scale)
    return num / ((1.0 - p) * torch.exp(-scale) + num).clamp_min(1e-30)
