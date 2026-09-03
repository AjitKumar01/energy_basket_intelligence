#!/usr/bin/env python3
"""Fresh training gate for the unchanged version-4 energy basket model.

The negative likelihood score comes from exact-base positive interaction particles.  This
file intentionally imports none of the historical QMC training controllers and accepts no
resume checkpoint.  It is a bounded optimizer gate: establish likelihood direction,
latency and particle overlap first; a long-run launcher should only be enabled after this
gate succeeds.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch

from data import build
from features import Features
from fit import (Batcher, build_observed_basic_scores,
                 build_observed_phi_operator, full_observed_phi_score,
                 observed_basic_scores, observed_phi_score,
                 optimizer_parameter_groups)
from interaction_particles import (controlled_particle_statistics,
                                   direct_interaction_particles,
                                   fisher_negative_surrogate,
                                   rao_blackwell_particle_statistics)
from ragged import RaggedModel
from sparse_artifact import load_sparse_initialization_artifact
from tempered_ais import annealed_smc_logz


torch.set_default_dtype(torch.float64)


class _Tee:
    def __init__(self, stream, path: Path, append: bool = False):
        self.stream = stream
        self.file = path.open("a" if append else "w", buffering=1)

    def write(self, value):
        self.stream.write(value)
        self.file.write(value)
        return len(value)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path,
                        default=Path("out/v3_version4_sparse_init.pt"))
    parser.add_argument("--label", default="run252_interaction_particle_gate")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--particles", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--lam-lr-scale", type=float, default=0.1)
    parser.add_argument("--phi-lr", type=float, default=0.05)
    parser.add_argument("--rho0-lr", type=float, default=0.05)
    parser.add_argument("--rho-c-lr", type=float, default=0.02)
    parser.add_argument("--rho-c-every", type=int, default=10)
    parser.add_argument("--rho-c-energy-trust", type=float, default=0.5)
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--clip", type=float, default=20.0)
    parser.add_argument("--eval-trips", type=int, default=24)
    parser.add_argument("--eval-particles", type=int, default=32)
    parser.add_argument("--eval-replicates", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=20)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--ess-fallback", type=float, default=0.30)
    parser.add_argument("--smc-particles", type=int, default=32)
    parser.add_argument("--smc-levels", type=int, default=17)
    parser.add_argument("--smc-power", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=25201)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--resume", type=Path,
                        help="resume this same run from an estimator checkpoint")
    return parser.parse_args()


def supported_trips(data, split: int, nmax: int) -> np.ndarray:
    trips = np.flatnonzero(data["trip_split"] == split)
    return trips[data["trip_nlines"][trips] <= nmax]


def observed_category_pairs(data, trips: np.ndarray, categories: int) -> torch.Tensor:
    """Exact training-average choose(n_c,2), independent of customer context."""
    total = np.zeros(categories, dtype=np.float64)
    pointer, line_category = data["line_ptr"], data["line_cat"]
    for trip in np.asarray(trips, dtype=np.int64):
        count = np.bincount(
            line_category[int(pointer[trip]):int(pointer[trip + 1])],
            minlength=categories)
        total += count * (count - 1.0) * 0.5
    return torch.as_tensor(total / len(trips), dtype=torch.float64)


def category_capacities(data, categories: int, nmax: int) -> np.ndarray:
    """Largest available count in each category over every store assortment."""
    pointer = data["store_cat_ptr"]
    stores = int(data["n_store"])
    answer = np.zeros(categories, dtype=np.int64)
    for store in range(stores):
        start = store * categories
        widths = pointer[start + 1:start + categories + 1] - pointer[start:start + categories]
        answer = np.maximum(answer, np.minimum(widths, nmax))
    return answer


def maximum_category_pair_energy(delta: torch.Tensor, capacities: np.ndarray,
                                 nmax: int) -> float:
    """Exact support maximum of sum_c |delta_c| choose(n_c,2).

    This dynamic program is evaluated only at the delayed category update.  If its value
    is at most ``epsilon``, every supported basket obeys
    ``|Delta E_category(S)| <= epsilon``.  Consequently the old/new normalized density
    ratio lies in ``[exp(-2 epsilon), exp(2 epsilon)]``: an update cannot create an unseen
    remote phase in one step.
    """
    value = delta.detach().abs().cpu().numpy()
    if value.shape != (len(capacities),):
        raise ValueError("one category displacement is required per capacity")
    dp = np.full(nmax + 1, -np.inf, dtype=np.float64)
    dp[0] = 0.0
    for coefficient, capacity in zip(value, capacities):
        limit = min(int(capacity), int(nmax))
        if coefficient == 0.0 or limit < 2:
            continue
        previous = dp.copy()
        updated = previous.copy()
        for count in range(2, limit + 1):
            candidate = (previous[:nmax + 1 - count]
                         + coefficient * count * (count - 1.0) * 0.5)
            updated[count:] = np.maximum(updated[count:], candidate)
        dp = updated
    return float(np.max(dp))


def trust_category_step(delta: torch.Tensor, capacities: np.ndarray, nmax: int,
                        energy_trust: float) -> tuple[torch.Tensor, float, float]:
    """Scale a category displacement to a rigorous complete-support energy ball."""
    if energy_trust <= 0:
        raise ValueError("category energy trust must be positive")
    radius = maximum_category_pair_energy(delta, capacities, nmax)
    scale = min(1.0, float(energy_trust) / max(radius, 1e-300))
    return delta * scale, radius, scale


@torch.no_grad()
def replace_positive_phase_gradient_(parameter: torch.nn.Parameter,
                                     batch_positive: torch.Tensor,
                                     full_positive: torch.Tensor) -> None:
    """Turn ``negative - batch_positive`` into ``negative - full_positive``."""
    if parameter.grad is None:
        raise ValueError("parameter has no gradient to control")
    if parameter.grad.shape != batch_positive.shape or \
            batch_positive.shape != full_positive.shape:
        raise ValueError("positive-phase controls must match the parameter shape")
    parameter.grad.add_(batch_positive - full_positive)


@torch.no_grad()
def particle_likelihood(model, batcher, trips, particles, replicates, seed,
                        ess_fallback=0.0, smc_particles=32, smc_levels=17,
                        smc_power=2.0):
    """Same-trip likelihood, averaging independent estimates on the Z scale."""
    ix, ctx, line_ctx, house, li, lt, lc, _lq = batcher.make(trips)
    model.house, model.ctx = house, ctx
    estimates, ess = [], []
    for replicate in range(replicates):
        result = direct_interaction_particles(
            model, ix, particles,
            torch.Generator().manual_seed(seed + 104729 * replicate))
        estimates.append(result.log_z)
        ess.append(result.ess_fraction)
    log_z = torch.logsumexp(torch.stack(estimates), dim=0) - math.log(replicates)
    minimum_ess = torch.stack(ess).amin(0)
    fallback = torch.nonzero(minimum_ess < float(ess_fallback), as_tuple=True)[0]
    smc_minimum = torch.ones_like(minimum_ess)
    if fallback.numel():
        sub_trips = np.asarray(trips)[fallback.cpu().numpy()]
        sub_ix, sub_ctx, _sub_line_ctx, sub_house, *_ = batcher.make(sub_trips)
        model.house, model.ctx = sub_house, sub_ctx
        time_axis = torch.linspace(0.0, 1.0, int(smc_levels))
        schedule = 1.0 - (1.0 - time_axis).pow(float(smc_power))
        smc_estimates, diagnostics = [], []
        for replicate in range(replicates):
            smc = annealed_smc_logz(
                model, sub_ix, schedule, particles=int(smc_particles),
                mutation_steps=1,
                generator=torch.Generator().manual_seed(
                    seed + 7_000_001 + 104729 * replicate))
            smc_estimates.append(smc.log_z)
            diagnostics.append(smc.min_ess_fraction)
        log_z[fallback] = (torch.logsumexp(torch.stack(smc_estimates), dim=0)
                           - math.log(replicates))
        smc_minimum[fallback] = torch.stack(diagnostics).amin(0)
        model.house, model.ctx = house, ctx
    energy = model.energy(li, lt, lc, ix.B, line_ctx)
    return {
        "basket_loglik": float((energy - log_z).mean()),
        "energy": float(energy.mean()),
        "log_z": float(log_z.mean()),
        "ess_min": float(torch.stack(ess).min()),
        "ess_median": float(torch.stack(ess).median()),
        "fallback_trips": int(fallback.numel()),
        "smc_ess_min": float(smc_minimum[fallback].min()
                             if fallback.numel() else 1.0),
    }


@torch.no_grad()
def _subset_particle_states(ix, states, old_trip_indices, sub_ix):
    """Map selected original-trip basket slots into a freshly built subset index."""
    maps = []
    for b in range(sub_ix.B):
        slots = torch.nonzero(sub_ix.item_trip == b, as_tuple=True)[0]
        maps.append({int(sub_ix.item[slot]): int(slot) for slot in slots})
    answer = []
    for particle in states:
        row = []
        for new_b, old_b in enumerate(old_trip_indices.tolist()):
            products = ix.item[particle[int(old_b)]]
            row.append(torch.as_tensor(
                sorted(maps[new_b][int(product)] for product in products),
                dtype=torch.long, device=sub_ix.item.device))
        answer.append(row)
    return answer


@torch.no_grad()
def replace_low_ess_statistics(model, batcher, trips, ix, direct, statistics,
                               ess_floor, smc_particles, smc_levels, smc_power,
                               generator):
    """Replace only low-overlap trip scores by fixed-ladder interaction SMC scores."""
    low = torch.nonzero(direct.ess_fraction < float(ess_floor), as_tuple=True)[0]
    if not low.numel():
        return statistics, 0, 1.0
    outer_house, outer_ctx = model.house, model.ctx
    sub_trips = np.asarray(trips)[low.cpu().numpy()]
    sub_ix, sub_ctx, _sub_line_ctx, sub_house, *_ = batcher.make(sub_trips)
    model.house, model.ctx = sub_house, sub_ctx
    sub_states = _subset_particle_states(ix, direct.states, low, sub_ix)
    direct_low = controlled_particle_statistics(
        model, sub_ix, sub_states, direct.log_weights[:, low])
    time_axis = torch.linspace(0.0, 1.0, int(smc_levels))
    schedule = 1.0 - (1.0 - time_axis).pow(float(smc_power))
    smc = annealed_smc_logz(
        model, sub_ix, schedule,
        particles=int(smc_particles), mutation_steps=1, generator=generator)
    smc_low = rao_blackwell_particle_statistics(model, sub_ix, smc.states)

    statistics.item_incidence[low] = smc_low.item_incidence
    statistics.size_probability[low] = smc_low.size_probability
    statistics.interaction[low] = smc_low.interaction
    # Category and Phi statistics enter the surrogate only through their trip sums.
    statistics.category_pairs[0] += (smc_low.category_pairs.sum(0)
                                     - direct_low.category_pairs.sum(0))
    statistics.phi_score += smc_low.phi_score - direct_low.phi_score
    model.house, model.ctx = outer_house, outer_ctx
    return statistics, int(low.numel()), float(smc.min_ess_fraction.min())


def atomic_save(path: Path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    args = parse_args()
    resume_blob = None
    if args.resume is not None:
        resume_blob = torch.load(args.resume, map_location="cpu", weights_only=False)
        if resume_blob.get("estimator") != "exact_base_positive_interaction_particles":
            raise SystemExit("resume checkpoint belongs to a different estimator")
        saved = resume_blob.get("config", {})
        invariant = (
            "artifact", "label", "batch", "particles", "lr", "lam_lr_scale",
            "phi_lr", "rho0_lr", "rho_c_lr", "rho_c_every",
            "rho_c_energy_trust", "momentum", "weight_decay", "clip",
            "eval_trips", "eval_particles", "eval_replicates", "eval_every",
            "ess_fallback", "smc_particles", "smc_levels", "smc_power", "seed")
        for name in invariant:
            old, new = saved.get(name), getattr(args, name)
            if name == "artifact":
                old, new = str(old), str(new)
            if old != new:
                raise SystemExit(
                    f"resume configuration mismatch for {name}: {old!r} != {new!r}")
        if int(resume_blob["iter"]) >= args.iters:
            raise SystemExit("resume checkpoint has already reached the requested iterations")
    if min(args.iters, args.batch, args.particles, args.eval_trips,
           args.eval_particles, args.eval_replicates, args.threads) < 1:
        raise SystemExit("iteration, batch, particle, evaluation and thread counts must be positive")
    if min(args.rho_c_every, args.eval_every, args.log_every,
           args.smc_particles, args.smc_levels) < 1 or \
            args.rho_c_energy_trust <= 0 or args.smc_power <= 0:
        raise SystemExit("rho-c-every, eval-every, log-every and trust must be positive")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    out = Path("out")
    out.mkdir(exist_ok=True)
    log_path = out / f"v3_{args.label}.log"
    history_path = out / f"v3_{args.label}_history.json"
    checkpoint_path = out / f"v3_{args.label}.pt"
    best_checkpoint_path = out / f"v3_{args.label}_best.pt"
    sys.stdout = _Tee(sys.stdout, log_path, append=resume_blob is not None)
    sys.stderr = _Tee(sys.stderr, log_path, append=resume_blob is not None)

    data = build()
    raw = torch.load(args.artifact, map_location="cpu", weights_only=False)
    metadata = raw["metadata"]
    J, N, C, S = (int(data[name]) for name in
                  ("n_item", "n_user", "n_cat", "n_store"))
    model = RaggedModel(
        J, N, C, K=metadata["K"], Kz=metadata["Kz"],
        nmax=metadata["nmax"], R=metadata["R"], seed=metadata["seed"],
        S=S, Kp=metadata["Kp"], phi_init=0.0)
    restored = load_sparse_initialization_artifact(args.artifact, model)
    model._poly_degree_native = True
    model._esp_native = True
    model._esp_log_blocked = True
    if J != 5455 or os.environ.get("V3_AFFINITY", "0") != "1" or model.R < model.nmax:
        raise SystemExit("artifact is not the complete-support version-4 model")

    features = Features(J, S, 712)
    batcher = Batcher(data, features, model.nmax)
    train = supported_trips(data, 0, model.nmax)
    valid = supported_trips(data, 1, model.nmax)
    rng = np.random.default_rng(args.seed)
    valid = valid[np.random.default_rng(args.seed + 1).permutation(len(valid))]
    fixed_valid = valid[:args.eval_trips]
    generator = torch.Generator().manual_seed(args.seed + 2)

    # These are exact finite-training averages of context-independent sufficient
    # statistics.  Replacing their minibatch versions changes variance, not the objective.
    print("[interaction-fit] constructing exact full-training sufficient statistics",
          flush=True)
    full_basic = build_observed_basic_scores(data, train, J, model.nmax)
    full_basic = {name: value.to(dtype=model.phi.dtype, device=model.phi.device)
                  for name, value in full_basic.items()}
    full_category_pairs = observed_category_pairs(data, train, C).to(model.phi.device)
    full_phi_operator = build_observed_phi_operator(data, train, J)
    rho_c_capacities = category_capacities(data, C, model.nmax)

    groups = optimizer_parameter_groups(
        model, args.lr, lam_lr_scale=args.lam_lr_scale,
        taste_lr_scale=1.0, taste_weight_decay=0.0)
    separate_ids = {id(model.phi), id(model.rho_0_free), id(model.rho_c)}
    for group in groups:
        group["params"] = [p for p in group["params"] if id(p) not in separate_ids]
    groups = [group for group in groups if group["params"]]
    # Coordinatewise Adam turns a tiny, noisy score for every catalogue row into a
    # full-sized step.  That is especially destructive for alpha and sparse context item
    # tables.  One scalar SGD scale per declared block preserves the Fisher-score
    # magnitude and direction; no parameter or interaction row is frozen.
    main_optimizer = torch.optim.SGD(
        groups, lr=args.lr, momentum=args.momentum,
        weight_decay=args.weight_decay)
    scale_optimizer = torch.optim.SGD([
        {"params": [model.phi], "lr": args.phi_lr,
         "momentum": args.momentum, "weight_decay": args.weight_decay,
         "name": "phi"},
        {"params": [model.rho_0_free], "lr": args.rho0_lr,
         "momentum": args.momentum, "weight_decay": args.weight_decay,
         "name": "rho0"},
    ], lr=1.0)

    if resume_blob is not None:
        if resume_blob.get("fresh_artifact_digest") != restored["model_state_sha256"]:
            raise SystemExit("resume checkpoint and immutable artifact digests differ")
        model.load_state_dict(resume_blob["model"], strict=True)
        main_optimizer.load_state_dict(resume_blob["main_optimizer"])
        scale_optimizer.load_state_dict(resume_blob["scale_optimizer"])

    print(f"[interaction-fit] immutable fresh artifact {restored['model_state_sha256']}",
          flush=True)
    print(f"[interaction-fit] unchanged version-4 law J={J}, C={C}, "
          f"Kz={model.Kz}, nmax=R={model.nmax}", flush=True)
    print(f"[interaction-fit] positive exact-base particles P={args.particles}; "
          "no QMC, skipped trip, or retry", flush=True)
    print(f"[interaction-fit] optimizer scale-preserving SGD lr={args.lr:g}, lam scale="
          f"{args.lam_lr_scale:g}; shared-scale SGD phi={args.phi_lr:g}, "
          f"rho0={args.rho0_lr:g}", flush=True)
    print("[interaction-fit] exact positive-phase controls: item, size, category, Phi; "
          f"rho_c update every {args.rho_c_every} steps with complete-support "
          f"energy trust {args.rho_c_energy_trust:g}", flush=True)
    print(f"[interaction-fit] log: {log_path.resolve()}", flush=True)
    print(f"[interaction-fit] checkpoint: {checkpoint_path.resolve()}", flush=True)

    started = time.perf_counter()
    reservoir_generator = torch.Generator().manual_seed(args.seed + 3)
    if resume_blob is None:
        initial = particle_likelihood(
            model, batcher, fixed_valid, args.eval_particles, args.eval_replicates,
            args.seed + 1_000_003, args.ess_fallback, args.smc_particles,
            args.smc_levels, args.smc_power)
        print("[interaction-fit] initial fixed-validation "
              f"LL={initial['basket_loglik']:.6f}, E={initial['energy']:.4f}, "
              f"logZ={initial['log_z']:.4f}, ESS>={initial['ess_min']:.4f}", flush=True)
        records = []
        evaluations = [{"iter": 0, **initial}]
        best_validation = float(initial["basket_loglik"])
        best_iteration = 0
        rank_reservoir = {"trip": [], "observed": [], "model_a": [], "model_b": []}
        rho_c_gradient_sum = torch.zeros_like(model.rho_c)
        rho_c_gradient_count = 0
        resume_iteration = 0
    else:
        resume_iteration = int(resume_blob["iter"])
        initial = resume_blob["initial_validation"]
        records = resume_blob["records"]
        evaluations = resume_blob["evaluations"]
        best_validation = float(resume_blob["best_validation"])
        best_iteration = int(resume_blob["best_iteration"])
        rank_reservoir = resume_blob["rank_reservoir"]
        rho_c_gradient_sum = resume_blob.get(
            "rho_c_gradient_sum", torch.zeros_like(model.rho_c)).clone()
        rho_c_gradient_count = int(resume_blob.get("rho_c_gradient_count", 0))
        state = resume_blob.get("random_state")
        if state is not None:
            rng.bit_generator.state = state["numpy_batch"]
            generator.set_state(state["particle"])
            reservoir_generator.set_state(state["reservoir"])
            torch.random.set_rng_state(state["torch_global"])
            random_note = "restored exact RNG states"
        else:
            # Format-2 checkpoints written before RNG persistence can reproduce the
            # minibatch stream exactly. Particle streams receive deterministic new seeds;
            # this is a valid stochastic continuation but not bit-identical replay.
            rng = np.random.default_rng(args.seed)
            for _ in range(resume_iteration):
                rng.choice(len(train), size=args.batch, replace=False)
            generator.manual_seed(args.seed + 2 + 1_000_003 * resume_iteration)
            reservoir_generator.manual_seed(args.seed + 3 + 1_000_033 * resume_iteration)
            random_note = "reconstructed batches; deterministically renewed particle RNGs"
        print(f"[interaction-fit] RESUME from iteration {resume_iteration}; "
              f"best LL={best_validation:.6f} at {best_iteration}; {random_note}",
              flush=True)

    def checkpoint_payload(iteration, latest_validation):
        return {
            "format": 2,
            "estimator": "exact_base_positive_interaction_particles",
            "fresh_artifact_digest": restored["model_state_sha256"],
            "iter": iteration,
            "model": model.state_dict(),
            "main_optimizer": main_optimizer.state_dict(),
            "scale_optimizer": scale_optimizer.state_dict(),
            "config": vars(args),
            "initial_validation": initial,
            "latest_validation": latest_validation,
            "best_validation": best_validation,
            "best_iteration": best_iteration,
            "records": records,
            "evaluations": evaluations,
            "rank_reservoir": rank_reservoir,
            "rho_c_gradient_sum": rho_c_gradient_sum,
            "rho_c_gradient_count": rho_c_gradient_count,
            "random_state": {
                "numpy_batch": rng.bit_generator.state,
                "particle": generator.get_state(),
                "reservoir": reservoir_generator.get_state(),
                "torch_global": torch.random.get_rng_state(),
            },
        }

    if resume_blob is None:
        atomic_save(best_checkpoint_path, checkpoint_payload(0, initial))
    for iteration in range(resume_iteration + 1, args.iters + 1):
        tick = time.perf_counter()
        trips = train[rng.choice(len(train), size=args.batch, replace=False)]
        ix, ctx, line_ctx, house, li, lt, lc, _lq = batcher.make(trips)
        model.house, model.ctx = house, ctx
        result = direct_interaction_particles(model, ix, args.particles, generator)
        # The rank audit reuses particles already paid for by training.  Two independent
        # weighted resamples per context form model-vs-model null and cross-fit operators;
        # observed baskets are stored beside the identical contexts.
        probability = torch.softmax(result.log_weights, dim=0)
        for b, trip in enumerate(trips):
            ancestor = torch.multinomial(
                probability[:, b], 2, replacement=True,
                generator=reservoir_generator)
            rank_reservoir["trip"].append(int(trip))
            rank_reservoir["observed"].append(
                li[lt == b].detach().cpu().numpy().astype(np.uint16))
            rank_reservoir["model_a"].append(
                ix.item[result.states[int(ancestor[0])][b]].detach().cpu()
                .numpy().astype(np.uint16))
            rank_reservoir["model_b"].append(
                ix.item[result.states[int(ancestor[1])][b]].detach().cpu()
                .numpy().astype(np.uint16))
        statistics = controlled_particle_statistics(
            model, ix, result.states, result.log_weights)
        statistics, fallback_trips, smc_ess_min = replace_low_ess_statistics(
            model, batcher, trips, ix, result, statistics,
            args.ess_fallback, args.smc_particles, args.smc_levels,
            args.smc_power, generator)
        data_energy = model.energy(li, lt, lc, ix.B, line_ctx).mean()
        negative = fisher_negative_surrogate(model, ix, statistics)
        loss = -(data_energy - negative)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite score surrogate at iteration {iteration}")

        main_optimizer.zero_grad(set_to_none=True)
        scale_optimizer.zero_grad(set_to_none=True)
        loss.backward()

        # Exact positive-phase controls.  If h_batch and h_full are energy scores,
        # grad(loss) = negative - h_batch becomes negative - h_full by adding their
        # difference.  The expected likelihood gradient is unchanged.
        batch_basic = observed_basic_scores(model, li, lt, ix.B)
        replace_positive_phase_gradient_(
            model.lam, batch_basic["lam"], full_basic["lam"])
        replace_positive_phase_gradient_(
            model.rho_0_free, batch_basic["rho_0_free"],
            full_basic["rho_0_free"])
        batch_phi = observed_phi_score(model, li, lt, ix.B)
        full_phi = full_observed_phi_score(full_phi_operator, model.phi)
        replace_positive_phase_gradient_(model.phi, batch_phi, full_phi)
        count = torch.bincount(
            lt * model.C + lc, minlength=ix.B * model.C).view(ix.B, model.C)
        count = count.to(model.rho_c.dtype)
        batch_category_pairs = (count * (count - 1.0) * 0.5).mean(0)
        # The energy score for rho_c is minus the pair statistic.
        replace_positive_phase_gradient_(
            model.rho_c, -batch_category_pairs, -full_category_pairs)

        phi_grad_rms = float(model.phi.grad.square().mean().sqrt())
        rho0_grad_rms = float(model.rho_0_free.grad.square().mean().sqrt())
        rho_c_grad_rms = float(model.rho_c.grad.square().mean().sqrt())
        rho_c_gradient_sum.add_(model.rho_c.grad.detach())
        rho_c_gradient_count += 1
        # rho_c has its own delayed, support-certified update below.
        model.rho_c.grad = None
        grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip))
        if not math.isfinite(grad_norm):
            raise FloatingPointError(f"non-finite gradient at iteration {iteration}")
        main_optimizer.step()
        scale_optimizer.step()
        rho_c_radius = 0.0
        rho_c_scale = 0.0
        if iteration % args.rho_c_every == 0 or iteration == args.iters:
            proposal = (-args.rho_c_lr * rho_c_gradient_sum
                        / max(rho_c_gradient_count, 1))
            trusted, rho_c_radius, rho_c_scale = trust_category_step(
                proposal, rho_c_capacities, model.nmax,
                args.rho_c_energy_trust)
            with torch.no_grad():
                model.rho_c.add_(trusted)
            rho_c_gradient_sum.zero_()
            rho_c_gradient_count = 0
        model.project_context_gauges()  # exact parameter gauge; probabilities unchanged
        model.project_rho_c(-1.5)       # finite complete-support numerical domain

        size_axis = torch.arange(1, model.nmax + 1, dtype=model.phi.dtype)
        model_size = float((statistics.size_probability * size_axis).sum(1).mean())
        observed_size = float(torch.bincount(lt, minlength=ix.B).double().mean())
        row = {
            "iter": iteration,
            "surrogate": float(-loss.detach()),
            "ess_min": float(result.ess_fraction.min()),
            "ess_median": float(result.ess_fraction.median()),
            "fallback_trips": fallback_trips,
            "smc_ess_min": smc_ess_min,
            "log_ratio_mean": float(result.log_ratio.mean()),
            "observed_size": observed_size,
            "model_size": model_size,
            "grad_norm": grad_norm,
            "phi_grad_rms": phi_grad_rms,
            "rho0_grad_rms": rho0_grad_rms,
            "rho_c_grad_rms": rho_c_grad_rms,
            "rho_c_proposal_energy_radius": rho_c_radius,
            "rho_c_trust_scale": rho_c_scale,
            "phi_row_rms": float(model.phi.norm() / math.sqrt(model.J)),
            "seconds": time.perf_counter() - tick,
        }
        records.append(row)
        if iteration % args.log_every == 0 or iteration == args.iters:
            window = records[-min(args.log_every, len(records)):]
            print(f"[interaction-fit] step {iteration:4d} score="
                  f"{np.mean([x['surrogate'] for x in window]):9.4f} "
                  f"ESS>={min(x['ess_min'] for x in window):.4f} size(obs/model)="
                  f"{np.mean([x['observed_size'] for x in window]):.2f}/"
                  f"{np.mean([x['model_size'] for x in window]):.2f} "
                  f"grad={np.mean([x['grad_norm'] for x in window]):.3f} "
                  f"phi_g={np.mean([x['phi_grad_rms'] for x in window]):.2e} "
                  f"rho0_g={np.mean([x['rho0_grad_rms'] for x in window]):.2e} "
                  f"rho_c_g={np.mean([x['rho_c_grad_rms'] for x in window]):.2e} "
                  f"fallback={sum(x['fallback_trips'] for x in window)} "
                  f"{np.mean([x['seconds'] for x in window]):.3f}s/it", flush=True)

        if iteration % args.eval_every == 0 or iteration == args.iters:
            validation = particle_likelihood(
                model, batcher, fixed_valid, args.eval_particles,
                args.eval_replicates, args.seed + 1_000_003,
                args.ess_fallback, args.smc_particles, args.smc_levels,
                args.smc_power)
            evaluations.append({"iter": iteration, **validation})
            change = validation["basket_loglik"] - initial["basket_loglik"]
            print(f"[interaction-fit] validation {iteration}: "
                  f"LL={validation['basket_loglik']:.6f}, change={change:+.6f}, "
                  f"E={validation['energy']:.4f}, logZ={validation['log_z']:.4f}, "
                  f"ESS>={validation['ess_min']:.4f}, "
                  f"fallback={validation['fallback_trips']}, "
                  f"SMC-ESS>={validation['smc_ess_min']:.3f}", flush=True)
            if validation["basket_loglik"] > best_validation:
                best_validation = float(validation["basket_loglik"])
                best_iteration = iteration
                atomic_save(best_checkpoint_path,
                            checkpoint_payload(iteration, validation))
                print(f"[interaction-fit] new best checkpoint at {iteration}", flush=True)
            atomic_save(checkpoint_path, checkpoint_payload(iteration, validation))
            history_path.write_text(json.dumps({
                "fresh_artifact_digest": restored["model_state_sha256"],
                "initial_validation": initial,
                "latest_validation": validation,
                "best_validation": best_validation,
                "best_iteration": best_iteration,
                "evaluations": evaluations,
                "records": records,
                "rank_reservoir_contexts": len(rank_reservoir["trip"]),
                "wall_seconds": time.perf_counter() - started,
            }, indent=2) + "\n")

    final = evaluations[-1]
    change = final["basket_loglik"] - initial["basket_loglik"]
    print(f"[interaction-fit] completed in {time.perf_counter() - started:.1f}s; "
          f"best LL={best_validation:.6f} at {best_iteration}; "
          f"rank reservoir={len(rank_reservoir['trip']):,} contexts; "
          f"saved {checkpoint_path} and {history_path}", flush=True)


if __name__ == "__main__":
    main()
