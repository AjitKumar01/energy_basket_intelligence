"""Direct laws of basket completions conditional on a revealed cart.

For a revealed non-empty set ``A`` and an unobserved completion ``T``, this module
samples the actual fitted law ``P(T | A subset S, x)``.  It removes the revealed slots,
absorbs the Gram cross-term into the remaining item utilities, and shifts the category
and total-size potentials.  The empty completion is retained explicitly and is therefore
the model's stopping event.
"""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Sequence

import torch

from ragged import RaggedIndex
from tempered_ais import annealed_smc_logz


@dataclass
class ConditionalCompletionResult:
    log_normalizer: torch.Tensor       # [B], includes the empty completion
    stop_probability: torch.Tensor     # [B]
    expected_additional_items: torch.Tensor  # [B]
    item_incidence: torch.Tensor       # [B,J], revealed items are zero here
    completion_size_probability: torch.Tensor  # [B,nmax+1], column zero is stop
    min_ess_fraction: torch.Tensor     # [B], interaction bridge diagnostic
    states: list[list[torch.Tensor]]   # reduced-index slots, non-empty component
    reduced_index: RaggedIndex
    original_slot: torch.Tensor        # reduced slot -> original slot


@dataclass
class ConditionalQuadratureResult:
    log_normalizer: torch.Tensor
    stop_probability: torch.Tensor
    expected_additional_items: torch.Tensor
    item_incidence: torch.Tensor
    completion_size_probability: torch.Tensor
    reduced_index: RaggedIndex
    original_slot: torch.Tensor


def _anchor_slots(ix, context: int, items: Sequence[int]) -> torch.Tensor:
    values = [int(item) for item in items]
    if not values or len(values) != len(set(values)):
        raise ValueError("each revealed cart must contain unique items and be nonempty")
    slots = []
    for item in values:
        found = torch.nonzero(
            (ix.item_trip == int(context)) & (ix.item == item), as_tuple=True)[0]
        if found.numel() != 1:
            raise ValueError(
                f"revealed item {item} must occur exactly once in context {context}")
        slots.append(int(found[0]))
    return torch.as_tensor(slots, dtype=torch.long, device=ix.item.device)


def prepare_conditional_completion(model, ix, slot_b: torch.Tensor,
                                   revealed_items: Sequence[Sequence[int]]):
    """Build the exact remainder representation for ``A subset S`` conditioning."""
    if len(revealed_items) != ix.B:
        raise ValueError("one revealed cart is required per context")
    if slot_b.shape != ix.item.shape:
        raise ValueError("slot_b must contain one utility per original assortment slot")
    fixed_phi = torch.zeros(ix.B, model.Kz, dtype=model.phi.dtype,
                            device=model.phi.device)
    fixed_cat = torch.zeros(ix.B, model.C, dtype=torch.long,
                           device=model.phi.device)
    fixed_size = torch.zeros(ix.B, dtype=torch.long, device=model.phi.device)
    remove = torch.zeros(len(ix.item), dtype=torch.bool, device=ix.item.device)
    for context, items in enumerate(revealed_items):
        slots = _anchor_slots(ix, context, items)
        fixed_size[context] = slots.numel()
        if int(slots.numel()) > model.nmax:
            raise ValueError("revealed cart exceeds the model's size support")
        fixed_phi[context] = model.phi[ix.item[slots]].sum(0)
        categories = ix.row_cat[ix.row_of[slots]]
        fixed_cat[context] = torch.bincount(categories, minlength=model.C)
        remove[slots] = True
    keep = ~remove
    if not bool(keep.any()):
        raise ValueError("conditioning removed every assortment slot")
    reduced = RaggedIndex(ix.item[keep], ix.row_of[keep], ix.row_trip, ix.row_cat,
                          ix.B, device=ix.item.device)
    original_slot = torch.nonzero(keep, as_tuple=True)[0]
    adjusted_b = (slot_b[original_slot]
                  + (model.phi[reduced.item]
                     * fixed_phi[reduced.item_trip]).sum(-1))
    return reduced, adjusted_b, fixed_cat, fixed_size, original_slot


@contextmanager
def conditional_model_state(model, adjusted_b, fixed_cat, fixed_size):
    """Install and reliably restore the evaluation-only conditional state."""
    old_override = getattr(model, "_b_override", None)
    old_cat = getattr(model, "_condition_cat_count", None)
    old_size = getattr(model, "_condition_size", None)
    model._b_override = adjusted_b
    model._condition_cat_count = fixed_cat
    model._condition_size = fixed_size
    try:
        yield
    finally:
        model._b_override = old_override
        model._condition_cat_count = old_cat
        model._condition_size = old_size


@torch.no_grad()
def completion_log_score(model, ix, slot_b: torch.Tensor, context: int,
                         revealed_items: Sequence[int],
                         completion_items: Sequence[int]) -> torch.Tensor:
    """Exact unnormalised conditional score; the empty completion has score zero."""
    revealed = [int(item) for item in revealed_items]
    completion = [int(item) for item in completion_items]
    if set(revealed) & set(completion):
        raise ValueError("revealed and completion items must be disjoint")
    if len(completion) != len(set(completion)):
        raise ValueError("completion items must be unique")
    from basket_counterfactual_query import basket_log_score
    base = basket_log_score(model, ix, slot_b, context, revealed)
    if not completion:
        return torch.zeros((), dtype=model.phi.dtype, device=model.phi.device)
    return basket_log_score(
        model, ix, slot_b, context, revealed + completion) - base


@torch.no_grad()
def conditional_completion_smc(model, ix, slot_b: torch.Tensor,
                               revealed_items: Sequence[Sequence[int]], *,
                               particles: int, schedule: Sequence[float],
                               mutation_steps: int = 1,
                               generator: torch.Generator | None = None
                               ) -> ConditionalCompletionResult:
    """Estimate completion incidence and stopping under the forced-cart law."""
    (reduced, adjusted_b, fixed_cat, fixed_size,
     original_slot) = prepare_conditional_completion(
        model, ix, slot_b, revealed_items)
    with conditional_model_state(model, adjusted_b, fixed_cat, fixed_size):
        result = annealed_smc_logz(
            model, reduced, schedule, particles=particles,
            mutation_steps=mutation_steps, generator=generator)

    # Z_total = 1 + Z_nonempty; the unit term is the empty completion.
    log_total = torch.logaddexp(torch.zeros_like(result.log_z), result.log_z)
    nonempty_probability = torch.sigmoid(result.log_z)
    stop_probability = torch.sigmoid(-result.log_z)
    incidence = torch.zeros(ix.B, model.J, dtype=model.phi.dtype,
                            device=model.phi.device)
    size = torch.zeros(ix.B, model.nmax + 1, dtype=model.phi.dtype,
                       device=model.phi.device)
    size[:, 0] = stop_probability
    for particle in result.states:
        for context, slots in enumerate(particle):
            weight = nonempty_probability[context] / float(particles)
            items = reduced.item[slots]
            if items.numel():
                incidence[context].index_add_(
                    0, items, torch.full_like(items, weight, dtype=model.phi.dtype))
                size[context, int(items.numel())] += weight
    axis = torch.arange(model.nmax + 1, dtype=model.phi.dtype,
                        device=model.phi.device)
    expected = (size * axis).sum(-1)
    if not bool(torch.isfinite(log_total).all()):
        raise FloatingPointError("non-finite conditional completion normalizer")
    if float((size.sum(-1) - 1.0).abs().max()) > 1e-8:
        raise RuntimeError("conditional completion size probabilities do not sum to one")
    return ConditionalCompletionResult(
        log_normalizer=log_total, stop_probability=stop_probability,
        expected_additional_items=expected, item_incidence=incidence,
        completion_size_probability=size,
        min_ess_fraction=result.min_ess_fraction, states=result.states,
        reduced_index=reduced, original_slot=original_slot)


def conditional_completion_quadrature(model, ix, slot_b: torch.Tensor,
                                      revealed_items: Sequence[Sequence[int]],
                                      nodes: torch.Tensor,
                                      weights: torch.Tensor, *,
                                      completion_size_log_weight: torch.Tensor | None = None,
                                      ) -> ConditionalQuadratureResult:
    """Deterministic conditional marginals from an explicitly supplied quadrature rule.

    ``completion_size_log_weight`` optionally defines a known observation mechanism on
    completion size.  For example, revealing a uniformly selected pair from a final
    basket multiplies every completion of size ``n`` by ``1 / C(n+2, 2)``.  Applying the
    weight before differentiation is essential: post-hoc adjustment of only the size law
    does not produce the corresponding item incidences.
    """
    (reduced, adjusted_b, fixed_cat, fixed_size,
     original_slot) = prepare_conditional_completion(
        model, ix, slot_b, revealed_items)
    adjusted_b = adjusted_b.detach().requires_grad_(True)
    old_quad = getattr(model, "quad", None)
    old_quad_a = getattr(model, "quad_a", None)
    model.quad = (nodes.to(dtype=model.phi.dtype, device=model.phi.device),
                  weights.to(dtype=model.phi.dtype, device=model.phi.device))
    model.quad_a = None
    try:
        with conditional_model_state(model, adjusted_b, fixed_cat, fixed_size):
            with torch.enable_grad():
                log_normalizer, size = model.log_Z(
                    reduced, drop_empty=False, return_size=True)
                if completion_size_log_weight is not None:
                    log_weight = torch.as_tensor(
                        completion_size_log_weight, dtype=size.dtype,
                        device=size.device)
                    if log_weight.ndim == 1:
                        log_weight = log_weight.unsqueeze(0)
                    if log_weight.shape[-1] != size.shape[-1] or \
                            log_weight.shape[0] not in (1, size.shape[0]):
                        raise ValueError(
                            "completion-size log weight must have shape [n] or [B,n]")
                    if float(log_weight[:, 0].abs().max()) > 1e-14:
                        raise ValueError(
                            "empty-completion log weight must be zero for normalization")
                    weighted_log_size = torch.log(size.clamp_min(1e-300)) + log_weight
                    log_normalizer = log_normalizer + torch.logsumexp(
                        weighted_log_size, dim=-1)
                    size = torch.softmax(weighted_log_size, dim=-1)
                slot_incidence = torch.autograd.grad(
                    log_normalizer.sum(), adjusted_b)[0]
    finally:
        model.quad = old_quad
        model.quad_a = old_quad_a
    incidence = torch.zeros(ix.B, model.J, dtype=model.phi.dtype,
                            device=model.phi.device)
    flat = reduced.item_trip * model.J + reduced.item
    incidence.view(-1).index_add_(0, flat, slot_incidence.detach())
    size = size.detach()
    axis = torch.arange(size.shape[1], dtype=model.phi.dtype,
                        device=model.phi.device)
    expected = (size * axis).sum(-1)
    if float((size.sum(-1) - 1.0).abs().max()) > 1e-9:
        raise RuntimeError("quadrature completion size probabilities do not sum to one")
    stop_from_normalizer = torch.exp(-log_normalizer.detach())
    if float((size[:, 0] - stop_from_normalizer).abs().max()) > 1e-8:
        raise RuntimeError(
            "conditional empty-completion mass disagrees with its normalizer")
    return ConditionalQuadratureResult(
        log_normalizer=log_normalizer.detach(),
        stop_probability=size[:, 0], expected_additional_items=expected,
        item_incidence=incidence, completion_size_probability=size,
        reduced_index=reduced, original_slot=original_slot)
