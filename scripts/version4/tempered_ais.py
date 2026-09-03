"""Annealed-SMC scalar normalizer for difficult version-4 evaluation contexts.

Training should use the Fisher negative phase in :mod:`fit_tempered`.  This module is the
slower independent fallback when an actual scalar log Z is required and QMC diagnostics
indicate separated mass.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch

from tempered_block_gibbs import (basket_interaction, conditional_slots_levels,
                                  conditional_slots_repeated)
from tempered_block_gibbs import conditional_log_tables_levels


@dataclass
class AnnealedSMCResult:
    log_z: torch.Tensor          # [B]
    log_z0: torch.Tensor         # exact no-Gram normalizer
    log_ratio: torch.Tensor      # SMC log estimate of Z_1/Z_0
    min_ess_fraction: torch.Tensor
    states: List[List[torch.Tensor]]  # final equally weighted basket particles


@torch.no_grad()
def exact_logz_beta0(model, ix) -> torch.Tensor:
    """Exact non-empty log normalizer with the Gram interaction switched off.

    At beta zero the latent integral is constant.  This deliberately uses the same
    degree-tilted category/cardinality DP as the exact block sampler; the generic
    polynomial likelihood path can lose numerical range in nearly full baskets.
    """
    z = torch.zeros(1, ix.B, model.Kz, dtype=model.phi.dtype,
                    device=model.phi.device)
    _log_g, _centred, log_size = conditional_log_tables_levels(model, ix, z, [0.0])
    return torch.logsumexp(log_size[0], -1)


def _resample(states, probability, generator):
    particles, batches = probability.shape
    out = [[None for _ in range(batches)] for _ in range(particles)]
    for b in range(batches):
        ancestor = torch.multinomial(
            probability[:, b], particles, replacement=True, generator=generator)
        for p in range(particles):
            out[p][b] = states[int(ancestor[p])][b].clone()
    return out


@torch.no_grad()
def annealed_smc_logz(model, ix, schedule: Sequence[float], particles: int = 32,
                      mutation_steps: int = 1,
                      generator: Optional[torch.Generator] = None
                      ) -> AnnealedSMCResult:
    """Estimate the original log Z with an unbiased SMC estimate on the Z scale.

    ``schedule`` is deterministic and must run from zero to one.  Multinomial resampling
    is performed at every bridge point.  The product of average incremental weights is an
    unbiased estimate of ``Z_1/Z_0``; its logarithm has the usual finite-particle downward
    bias, measured with independent calls/replicates.
    """
    beta = torch.as_tensor(schedule, dtype=model.phi.dtype, device=model.phi.device)
    if beta.ndim != 1 or beta.numel() < 2 or float(beta[0]) != 0.0 or float(beta[-1]) != 1.0:
        raise ValueError("schedule must begin at 0 and end at 1")
    if not bool((beta[1:] > beta[:-1]).all()):
        raise ValueError("schedule must be strictly increasing")
    particles = int(particles)
    if particles < 2 or int(mutation_steps) < 1:
        raise ValueError("particles >= 2 and mutation_steps >= 1 are required")

    log_z0 = exact_logz_beta0(model, ix)
    # At beta zero all particles share exactly the same conditional basket law.  Reuse one
    # forward category/size DP and perform only the independent reverse draws.
    z0 = torch.zeros(ix.B, model.Kz, dtype=model.phi.dtype,
                     device=model.phi.device)
    states = conditional_slots_repeated(model, ix, z0, 0.0, particles, generator)
    log_ratio = torch.zeros(ix.B, dtype=model.phi.dtype, device=model.phi.device)
    min_ess = torch.ones(ix.B, dtype=model.phi.dtype, device=model.phi.device)

    previous = 0.0
    for stage, current_tensor in enumerate(beta[1:]):
        current = float(current_tensor)
        interaction = torch.stack([
            basket_interaction(model, ix, particle) for particle in states])
        log_weight = (current - previous) * interaction
        log_mean = torch.logsumexp(log_weight, 0) - math.log(particles)
        log_ratio += log_mean
        probability = torch.softmax(log_weight, 0)
        ess = 1.0 / probability.square().sum(0)
        min_ess = torch.minimum(min_ess, ess / particles)
        states = _resample(states, probability, generator)

        # No terminal mutation is needed for the normalizer estimate.  Earlier kernels are
        # exactly invariant for p_beta and make the next incremental weight valid.
        if stage + 1 < beta.numel() - 1:
            for _ in range(int(mutation_steps)):
                latent = torch.zeros(particles, ix.B, model.Kz,
                                     dtype=model.phi.dtype, device=model.phi.device)
                root = math.sqrt(current)
                for p in range(particles):
                    for b, slots in enumerate(states[p]):
                        latent[p, b] = root * model.phi[ix.item[slots]].sum(0)
                latent += torch.randn(latent.shape, dtype=latent.dtype,
                                      device=latent.device, generator=generator)
                states = conditional_slots_levels(
                    model, ix, latent, [current] * particles, generator)
        previous = current
    return AnnealedSMCResult(log_z0 + log_ratio, log_z0, log_ratio, min_ess, states)
