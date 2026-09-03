"""Positive basket particles for the unchanged version-4 interaction energy.

The exactly tractable base law retains utilities, customer/context effects, prices,
category-count potentials, total-size potential and complete assortment support.  Only the
Gram pair statistic is importance weighted:

    p_1(S|x) / p_0(S|x) proportional exp(V_phi(S)).

This module is deliberately independent of the historical QMC controllers and of the
Bernoulli-mixture basket proposal.  It supplies the fast one-bridge estimator, reusable
score statistics and positive basket particles.  ``tempered_ais`` remains the fallback when
the one-bridge effective sample size is inadequate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from poly_degree_native import log_poly_tree_degree_native
from ragged import esp_log_bucketed, seg_max
from tempered_ais import exact_logz_beta0
from tempered_block_gibbs import (basket_interaction, conditional_slots_levels,
                                  conditional_slots_repeated)


BasketParticles = List[List[torch.Tensor]]  # [particle][trip] assortment-slot indices


@dataclass
class DirectInteractionResult:
    log_z: torch.Tensor                 # [B]
    log_z0: torch.Tensor                # [B], exact no-Gram normalizer
    log_ratio: torch.Tensor             # [B]
    ess_fraction: torch.Tensor          # [B]
    interaction: torch.Tensor           # [P,B]
    log_weights: torch.Tensor           # [P,B], normalized
    states: BasketParticles             # exact p0 draws before weighting


@dataclass
class AdditiveCounterfactualResult:
    log_z: torch.Tensor                 # [B], same-particle counterfactual normalizer
    log_ratio_to_factual: torch.Tensor  # [B]
    ess_fraction: torch.Tensor          # [B]
    log_weights: torch.Tensor           # [P,B], normalized counterfactual weights


@dataclass
class WeightedParticleStatistics:
    item_incidence: torch.Tensor        # [B,J]
    size_probability: torch.Tensor      # [B,nmax], sizes 1..nmax
    category_pairs: torch.Tensor        # [B,C], E choose(n_c,2)
    interaction: torch.Tensor           # [B], E V(S)
    phi_score: torch.Tensor             # [J,Kz], sum over trip expectations


def differentiable_log_size_beta0(model, ix, slot_b: Optional[torch.Tensor] = None
                                  ) -> torch.Tensor:
    """Exact differentiable no-Gram log mass for sizes 1 through ``nmax``.

    ``tempered_ais.exact_logz_beta0`` intentionally runs under ``no_grad`` for sampling.
    Training controls need the same algebra with a live graph.  An optional per-assortment
    ``slot_b`` leaf exposes exact item incidences and Hessian-vector products without
    conflating customer contexts that share a global product id.
    """
    slot_b = model.b_flat(ix) if slot_b is None else slot_b
    if slot_b.shape != ix.item.shape:
        raise ValueError("slot_b must provide one utility for every assortment slot")
    scale = seg_max(slot_b.unsqueeze(0), ix.item_trip, ix.B)[0]
    centred = slot_b - scale[ix.item_trip]
    log_e = esp_log_bucketed(
        centred.unsqueeze(0), ix.row_of, ix.n_rows, model.R,
        ix.row_size, ix.item_pos)[0]
    degree_axis = torch.arange(model.R + 1, dtype=slot_b.dtype,
                               device=slot_b.device)
    log_g = (log_e - model.rho_c[ix.row_cat].unsqueeze(-1)
             * model.pair_feature(degree_axis))

    category = torch.full((1, ix.B * ix.Cpad, model.R + 1), -float("inf"),
                          dtype=slot_b.dtype, device=slot_b.device)
    category[..., 0] = 0.0
    category = category.index_copy(1, ix.flat_slot, log_g.unsqueeze(0)).view(
        1, ix.B, ix.Cpad, model.R + 1)
    degrees = torch.zeros(ix.B * ix.Cpad, dtype=torch.long, device=slot_b.device)
    degrees[ix.flat_slot] = ix.row_size.clamp(max=model.R)
    degrees = degrees.view(ix.B, ix.Cpad)
    slope = (category[..., 1:] / degree_axis[1:]).amax(
        dim=(2, 3)).clamp_min(0.0)
    tilted = category - slope[:, :, None, None] * degree_axis
    log_a = log_poly_tree_degree_native(
        tilted.contiguous(), degrees.contiguous(), model.nmax)
    log_a = log_a + slope.unsqueeze(-1) * degree_axis
    size_axis = torch.arange(log_a.shape[-1], dtype=slot_b.dtype,
                             device=slot_b.device)
    return (log_a[0] + size_axis * scale.unsqueeze(-1)
            - model.rho_0()[:log_a.shape[-1]])[:, 1:]


def differentiable_logz_beta0(model, ix, slot_b: Optional[torch.Tensor] = None
                              ) -> torch.Tensor:
    """Exact differentiable no-Gram log normalizer for every trip."""
    return torch.logsumexp(differentiable_log_size_beta0(model, ix, slot_b), dim=-1)


@torch.no_grad()
def direct_interaction_particles(model, ix, particles: int,
                                 generator: Optional[torch.Generator] = None
                                 ) -> DirectInteractionResult:
    """Estimate ``Z`` by exact-base importance sampling with positive weights.

    The base forward dynamic program is evaluated once and reused for every independent
    reverse draw.  ``log_z`` is the logarithm of an unbiased estimator on the Z scale; as
    usual, the logarithm itself has finite-particle downward bias.
    """
    particles = int(particles)
    if particles < 2:
        raise ValueError("particles must be at least two")
    log_z0 = exact_logz_beta0(model, ix)
    z0 = torch.zeros(ix.B, model.Kz, dtype=model.phi.dtype,
                     device=model.phi.device)
    states = conditional_slots_repeated(model, ix, z0, 0.0, particles, generator)
    interaction = torch.stack([
        basket_interaction(model, ix, state) for state in states])
    log_ratio = torch.logsumexp(interaction, dim=0) - math.log(particles)
    log_weights = torch.log_softmax(interaction, dim=0)
    ess = torch.exp(-torch.logsumexp(2.0 * log_weights, dim=0))
    return DirectInteractionResult(
        log_z=log_z0 + log_ratio,
        log_z0=log_z0,
        log_ratio=log_ratio,
        ess_fraction=ess / particles,
        interaction=interaction,
        log_weights=log_weights,
        states=states)


@torch.no_grad()
def reweight_additive_counterfactual(
        model, ix, factual: DirectInteractionResult,
        factual_slot_b: torch.Tensor, counterfactual_slot_b: torch.Tensor
        ) -> AdditiveCounterfactualResult:
    """Reuse factual particles for a price/customer additive-utility intervention.

    The version-4 interaction, category and size terms are untouched.  Hence the exact
    Radon--Nikodym increment for a basket is simply the sum of
    ``counterfactual_slot_b - factual_slot_b`` over its selected assortment slots.  The
    returned normalizer estimate is positive on the Z scale.  A low ESS is a diagnostic to
    rerun :func:`direct_interaction_particles` under the counterfactual base context, not a
    reason to clip weights or change the model.
    """
    if factual_slot_b.shape != ix.item.shape or counterfactual_slot_b.shape != ix.item.shape:
        raise ValueError("factual and counterfactual utilities must match assortment slots")
    particles = len(factual.states)
    delta_slot = (counterfactual_slot_b - factual_slot_b).to(
        dtype=model.phi.dtype, device=model.phi.device)
    delta = torch.zeros(particles, ix.B, dtype=model.phi.dtype,
                        device=model.phi.device)
    for p, particle in enumerate(factual.states):
        for b, slots in enumerate(particle):
            delta[p, b] = delta_slot[slots].sum()
    unnormalized = factual.interaction + delta
    log_mean_weight = torch.logsumexp(unnormalized, dim=0) - math.log(particles)
    log_weights = torch.log_softmax(unnormalized, dim=0)
    ess = torch.exp(-torch.logsumexp(2.0 * log_weights, dim=0))
    log_z = factual.log_z0 + log_mean_weight
    return AdditiveCounterfactualResult(
        log_z=log_z,
        log_ratio_to_factual=log_z - factual.log_z,
        ess_fraction=ess / particles,
        log_weights=log_weights)


@torch.no_grad()
def weighted_particle_statistics(model, ix, states: BasketParticles,
                                 log_weights: Optional[torch.Tensor] = None
                                 ) -> WeightedParticleStatistics:
    """Return Fisher sufficient statistics from positive basket particles.

    ``log_weights`` is ``[particles,B]``.  Passing ``None`` treats final resampled SMC
    particles as equally weighted.  The signs of the normalizer derivatives are
    ``+item_incidence``, ``-size_probability`` and ``-category_pairs``.  ``phi_score`` is
    already the positive derivative of the Gram interaction and is summed over trips.
    """
    particles = len(states)
    if particles < 1 or any(len(level) != ix.B for level in states):
        raise ValueError("states must have shape [particles][trips]")
    dt, dev = model.phi.dtype, model.phi.device
    if log_weights is None:
        weight = torch.full((particles, ix.B), 1.0 / particles,
                            dtype=dt, device=dev)
    else:
        if log_weights.shape != (particles, ix.B):
            raise ValueError("log_weights must have shape [particles,B]")
        weight = torch.softmax(log_weights.to(dtype=dt, device=dev), dim=0)

    incidence = torch.zeros(ix.B, model.J, dtype=dt, device=dev)
    size = torch.zeros(ix.B, model.nmax, dtype=dt, device=dev)
    category_pairs = torch.zeros(ix.B, model.C, dtype=dt, device=dev)
    expected_interaction = torch.zeros(ix.B, dtype=dt, device=dev)
    phi_score = torch.zeros_like(model.phi)

    for p, particle in enumerate(states):
        for b, slots in enumerate(particle):
            slots = slots.to(device=dev, dtype=torch.long)
            n = int(slots.numel())
            if n < 1 or n > model.nmax:
                raise ValueError("particle lies outside the declared non-empty size support")
            w = weight[p, b]
            items = ix.item[slots]
            incidence[b].index_add_(
                0, items, w.expand(n))
            size[b, n - 1] += w
            cats = ix.row_cat[ix.row_of[slots]]
            counts = torch.bincount(cats, minlength=model.C).to(dt)
            category_pairs[b] += w * counts * (counts - 1.0) * 0.5

            selected_phi = model.phi[items]
            mu = selected_phi.sum(0)
            contribution = mu.unsqueeze(0) - selected_phi
            phi_score.index_add_(0, items, w * contribution)
            expected_interaction[b] += w * 0.5 * (
                mu.square().sum() - selected_phi.square().sum())

    return WeightedParticleStatistics(
        item_incidence=incidence,
        size_probability=size,
        category_pairs=category_pairs,
        interaction=expected_interaction,
        phi_score=phi_score)


@torch.no_grad()
def rao_blackwell_particle_statistics(model, ix, states: BasketParticles,
                                      log_weights: Optional[torch.Tensor] = None
                                      ) -> WeightedParticleStatistics:
    """One-site Rao--Blackwell Fisher statistics under the exact version-4 law.

    For each item ``j`` and sampled basket, condition on the basket with ``j`` removed. The
    exact add-one
    logit includes utility, the Gram interaction with the remaining basket, the original
    category-count increment and the original total-size increment.  Every coordinate is
    therefore replaced by its conditional expectation.  This is an estimator variance
    reduction, not pseudolikelihood training: the outer particles still target the full
    joint basket law.
    """
    particles = len(states)
    if particles < 1 or any(len(particle) != ix.B for particle in states):
        raise ValueError("states must have shape [particles][trips]")
    dt, dev = model.phi.dtype, model.phi.device
    if log_weights is None:
        weight = torch.full((particles, ix.B), 1.0 / particles,
                            dtype=dt, device=dev)
    else:
        if log_weights.shape != (particles, ix.B):
            raise ValueError("log_weights must have shape [particles,B]")
        weight = torch.softmax(log_weights.to(dtype=dt, device=dev), dim=0)

    incidence = torch.zeros(ix.B, model.J, dtype=dt, device=dev)
    size = torch.zeros(ix.B, model.nmax, dtype=dt, device=dev)
    category_pairs = torch.zeros(ix.B, model.C, dtype=dt, device=dev)
    expected_interaction = torch.zeros(ix.B, dtype=dt, device=dev)
    phi_score = torch.zeros_like(model.phi)
    slot_b = model.b_flat(ix).detach()
    rho0 = model.rho_0().detach()

    slot_item = ix.item
    slot_trip = ix.item_trip
    slot_cat = ix.row_cat[ix.row_of]
    slot_phi = model.phi[slot_item]
    slot_phi_norm = slot_phi.square().sum(1)
    trip_assortment_size = torch.bincount(slot_trip, minlength=ix.B).to(dt)
    flat_incidence_index = slot_trip * model.J + slot_item
    flat_category_index = slot_trip * model.C + slot_cat
    for p, particle in enumerate(states):
        selected = torch.zeros_like(slot_trip, dtype=torch.bool)
        for b, selected_slots in enumerate(particle):
            selected_slots = selected_slots.to(device=dev, dtype=torch.long)
            if selected_slots.numel() and not bool(
                    (slot_trip[selected_slots] == b).all()):
                raise ValueError("particle contains a slot outside its trip assortment")
            selected[selected_slots] = True
        old = selected.to(dt)

        basket_size = torch.zeros(ix.B, dtype=dt, device=dev)
        basket_size.index_add_(0, slot_trip, old)
        if bool(((basket_size < 1) | (basket_size > model.nmax)).any()):
            raise ValueError("particle lies outside the declared non-empty size support")

        mu = torch.zeros(ix.B, model.Kz, dtype=dt, device=dev)
        mu.index_add_(0, slot_trip, slot_phi * old.unsqueeze(1))
        counts_flat = torch.zeros(ix.B * model.C, dtype=dt, device=dev)
        counts_flat.index_add_(0, flat_category_index, old)
        counts = counts_flat.view(ix.B, model.C)

        rest_n = basket_size[slot_trip] - old
        rest_cat = counts[slot_trip, slot_cat] - old
        projection = (slot_phi * mu[slot_trip]).sum(1) - old * slot_phi_norm
        rest_index = rest_n.to(torch.long)
        add_index = (rest_index + 1).clamp(max=model.nmax)
        size_increment = rho0[add_index] - rho0[rest_index]
        logit = (slot_b + projection - model.rho_c[slot_cat] * rest_cat
                 - size_increment)
        conditional = torch.sigmoid(logit)
        conditional = torch.where(rest_n == 0, torch.ones_like(conditional),
                                  conditional)
        conditional = torch.where(rest_n >= model.nmax,
                                  torch.zeros_like(conditional), conditional)
        slot_weight = weight[p, slot_trip]

        incidence.view(-1).index_add_(
            0, flat_incidence_index, slot_weight * conditional)
        rest_mu = mu[slot_trip] - old.unsqueeze(1) * slot_phi
        phi_score.index_add_(
            0, slot_item, slot_weight.unsqueeze(1) * conditional.unsqueeze(1) * rest_mu)

        # Each item-specific conditional expectation is unbiased for the complete
        # statistic.  Average the J_b one-site representations within each trip.
        inv_j = trip_assortment_size[slot_trip].reciprocal()
        averaged_weight = slot_weight * inv_j
        lower = rest_index
        upper = lower + 1
        valid_lower = lower > 0
        if bool(valid_lower.any()):
            lower_index = (slot_trip[valid_lower] * model.nmax
                           + lower[valid_lower] - 1)
            size.view(-1).index_add_(
                0, lower_index,
                averaged_weight[valid_lower] * (1.0 - conditional[valid_lower]))
        # When rest_n == nmax, adding the item is outside support and has probability
        # exactly zero.  Excluding that zero term also avoids an nmax+1 array index.
        valid_upper = upper <= model.nmax
        if bool(valid_upper.any()):
            upper_index = (slot_trip[valid_upper] * model.nmax
                           + upper[valid_upper] - 1)
            size.view(-1).index_add_(
                0, upper_index,
                averaged_weight[valid_upper] * conditional[valid_upper])

        current_pairs = counts * (counts - 1.0) * 0.5
        category_pairs += weight[p].unsqueeze(1) * current_pairs
        correction = ((conditional - old) * rest_cat * averaged_weight)
        category_pairs.view(-1).index_add_(
            0, flat_category_index, correction)

        selected_norm = torch.zeros(ix.B, dtype=dt, device=dev)
        selected_norm.index_add_(0, slot_trip, old * slot_phi_norm)
        current_v = 0.5 * (mu.square().sum(1) - selected_norm)
        conditional_v = current_v[slot_trip] + (conditional - old) * projection
        conditional_v_sum = torch.zeros(ix.B, dtype=dt, device=dev)
        conditional_v_sum.index_add_(0, slot_trip, conditional_v)
        expected_interaction += (weight[p] * conditional_v_sum
                                 / trip_assortment_size)

    return WeightedParticleStatistics(
        item_incidence=incidence,
        size_probability=size,
        category_pairs=category_pairs,
        interaction=expected_interaction,
        phi_score=phi_score)


def controlled_particle_statistics(model, ix, states: BasketParticles,
                                   log_weights: torch.Tensor
                                   ) -> WeightedParticleStatistics:
    """Exact-base controls plus one-site Rao--Blackwell interaction statistics.

    For item incidence and basket size, the no-interaction expectation is available from
    one differentiable base DP.  The controlled estimate is

    ``base_exact + target_weighted_sample - base_unweighted_sample``.

    At zero interaction the two sample terms cancel exactly, leaving the exact base score.
    At nonzero interaction only the interaction-induced correction is stochastic.  The
    Phi and interaction blocks use exact one-site conditional expectations; an
    exact base Phi control would require a native second derivative, which is deliberately
    not assumed here.
    """
    weighted = weighted_particle_statistics(model, ix, states, log_weights)
    unweighted = weighted_particle_statistics(model, ix, states)
    conditional = rao_blackwell_particle_statistics(model, ix, states, log_weights)

    with torch.enable_grad():
        slot_b = model.b_flat(ix).detach().requires_grad_(True)
        log_size = differentiable_log_size_beta0(model, ix, slot_b)
        log_z = torch.logsumexp(log_size, dim=-1)
        slot_incidence, rho_c_gradient = torch.autograd.grad(
            log_z.sum(), (slot_b, model.rho_c))
        slot_incidence = slot_incidence.detach()
        base_category_total = -rho_c_gradient.detach()
    base_incidence = torch.zeros(ix.B, model.J, dtype=model.phi.dtype,
                                 device=model.phi.device)
    base_incidence.index_put_((ix.item_trip, ix.item), slot_incidence,
                              accumulate=True)
    base_size = torch.softmax(log_size.detach(), dim=-1)
    controlled_category_total = (base_category_total
                                 + weighted.category_pairs.sum(0)
                                 - unweighted.category_pairs.sum(0))

    return WeightedParticleStatistics(
        item_incidence=(base_incidence + weighted.item_incidence
                        - unweighted.item_incidence),
        size_probability=(base_size + weighted.size_probability
                          - unweighted.size_probability),
        category_pairs=(controlled_category_total.unsqueeze(0)
                        .expand(ix.B, -1) / ix.B),
        interaction=conditional.interaction,
        phi_score=conditional.phi_score)


def fisher_negative_surrogate(model, ix, statistics: WeightedParticleStatistics
                              ) -> torch.Tensor:
    """Scalar whose gradient is the supplied Fisher negative phase.

    Particle statistics are treated as fixed Monte Carlo estimates.  Differentiating this
    scalar propagates item incidence through the original ``b_flat`` customer/price/context
    architecture, size probability through the original ``rho_0`` parameterization,
    category pairs through ``rho_c``, and the pair score through ``Phi``.
    """
    if statistics.item_incidence.shape != (ix.B, model.J):
        raise ValueError("item incidence has the wrong shape")
    if statistics.size_probability.shape != (ix.B, model.nmax):
        raise ValueError("size probability has the wrong shape")
    if statistics.category_pairs.shape != (ix.B, model.C):
        raise ValueError("category-pair statistic has the wrong shape")
    if statistics.phi_score.shape != model.phi.shape:
        raise ValueError("Phi score has the wrong shape")

    slot_incidence = statistics.item_incidence[ix.item_trip, ix.item].detach()
    utility = (slot_incidence * model.b_flat(ix)).sum()
    category = -(statistics.category_pairs.detach().sum(0) * model.rho_c).sum()
    size = -(statistics.size_probability.detach()
             * model.rho_0()[1:model.nmax + 1].unsqueeze(0)).sum()
    interaction = (statistics.phi_score.detach() * model.phi).sum()
    return (utility + category + size + interaction) / ix.B


@torch.no_grad()
def resample_particles(states: BasketParticles, log_weights: torch.Tensor, draws: int,
                       generator: Optional[torch.Generator] = None
                       ) -> BasketParticles:
    """Draw positive basket particles independently for each trip."""
    particles = len(states)
    draws = int(draws)
    if particles < 1 or draws < 1:
        raise ValueError("states and draws must be nonempty")
    B = len(states[0])
    if log_weights.shape != (particles, B):
        raise ValueError("log_weights must have shape [particles,B]")
    probability = torch.softmax(log_weights, dim=0)
    ancestors = [torch.multinomial(probability[:, b], draws, replacement=True,
                                   generator=generator) for b in range(B)]
    return [[states[int(ancestors[b][p])][b].clone() for b in range(B)]
            for p in range(draws)]


@torch.no_grad()
def blocked_rejuvenation(model, ix, states: BasketParticles, beta: float = 1.0,
                         steps: int = 1,
                         generator: Optional[torch.Generator] = None
                         ) -> BasketParticles:
    """Apply exact-invariant Hubbard--Stratonovich blocked updates in parallel."""
    particles = len(states)
    if particles < 1 or int(steps) < 0:
        raise ValueError("invalid states or step count")
    beta = float(beta)
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must lie in [0,1]")
    current = states
    root = math.sqrt(beta)
    for _ in range(int(steps)):
        latent = torch.zeros(particles, ix.B, model.Kz, dtype=model.phi.dtype,
                             device=model.phi.device)
        for p, particle in enumerate(current):
            for b, slots in enumerate(particle):
                latent[p, b] = root * model.phi[ix.item[slots]].sum(0)
        latent += torch.randn(latent.shape, dtype=latent.dtype, device=latent.device,
                              generator=generator)
        current = conditional_slots_levels(
            model, ix, latent, [beta] * particles, generator)
    return current


@torch.no_grad()
def independence_rejuvenation(model, ix, states: BasketParticles, steps: int = 1,
                              generator: Optional[torch.Generator] = None
                              ) -> tuple[BasketParticles, torch.Tensor]:
    """Exact-invariant independence MH using the tractable base law as proposal.

    The acceptance ratio is ``exp(V(proposed)-V(current))`` because all base-energy and
    proposal-normalizer terms cancel.  This is cheap and useful precisely in the regime
    where direct base importance sampling has healthy overlap.
    """
    particles = len(states)
    if particles < 1 or int(steps) < 0:
        raise ValueError("invalid states or step count")
    current = [[slots.clone() for slots in particle] for particle in states]
    accepted = torch.zeros(ix.B, dtype=model.phi.dtype, device=model.phi.device)
    if int(steps) == 0:
        return current, accepted
    z0 = torch.zeros(ix.B, model.Kz, dtype=model.phi.dtype,
                     device=model.phi.device)
    for _ in range(int(steps)):
        proposal = conditional_slots_repeated(model, ix, z0, 0.0, particles, generator)
        old_v = torch.stack([basket_interaction(model, ix, s) for s in current])
        new_v = torch.stack([basket_interaction(model, ix, s) for s in proposal])
        take = (torch.rand(old_v.shape, dtype=old_v.dtype, device=old_v.device,
                           generator=generator).log()
                < (new_v - old_v).clamp_max(0.0))
        for p in range(particles):
            for b in range(ix.B):
                if bool(take[p, b]):
                    current[p][b] = proposal[p][b]
        accepted += take.to(model.phi.dtype).mean(0)
    return current, accepted / max(int(steps), 1)
