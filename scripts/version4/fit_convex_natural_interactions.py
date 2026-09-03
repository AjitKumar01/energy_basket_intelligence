#!/usr/bin/env python3
"""Deterministic constrained MC-MLE for the Version-4 interaction/size block.

The additive model is an exact proposal.  In a fixed orthonormal product basis U,

    K = U C U',                 0 <= C <= spectral_max^2 I,
    Delta rho_0(n) = a (n/10) + c (n/10)^2,

so the log-density ratio is linear in the natural parameters (C, a, c).  Fixed
common-random-number draws from the additive parent turn the likelihood-ratio
objective into a deterministic concave function.  Projected gradient ascent with
Armijo backtracking therefore has one global target and every accepted optimization
step increases the sampled objective.

The quadratic size coefficient is nonnegative and Delta rho_0(nmax) >= 0.  These
two linear constraints prevent the correction from creating an attractive large-size
tail while still permitting the empirically necessary negative linear coefficient.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch
from scipy.special import logsumexp

from audit_particle_counterfactual_generation import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from fit_interaction_particles import supported_trips
from tempered_block_gibbs import conditional_slots_repeated


torch.set_default_dtype(torch.float64)


def spectral_basis(path: Path, rank: int, score_mass: float) -> tuple[np.ndarray, int]:
    spectral = np.load(path)
    raw = np.asarray(spectral["eigenvectors"][:, :rank], dtype=np.float64)
    values = np.asarray(spectral["eigenvalues"][:rank], dtype=np.float64)
    row_mass = (np.square(raw) * np.clip(values, 0.0, None)[None, :]).sum(1)
    if not np.isfinite(row_mass).all() or row_mass.sum() <= 0:
        raise RuntimeError("spectral basis has no finite positive score mass")
    order = np.argsort(row_mass)[::-1]
    cumulative = np.cumsum(row_mass[order]) / row_mass.sum()
    keep_count = min(len(row_mass), int(np.searchsorted(cumulative, score_mass) + 1))
    keep = np.zeros(len(row_mass), dtype=bool)
    keep[order[:keep_count]] = True
    raw[~keep] = 0.0
    basis, _ = np.linalg.qr(raw)
    return basis, keep_count


def pair_statistic(items: torch.Tensor, basis: torch.Tensor) -> np.ndarray:
    """F(S) such that tr(C F(S)) is the Version-4 pair energy."""
    rows = basis[torch.unique(items)]
    total = rows.sum(0)
    return (0.5 * (torch.outer(total, total) - rows.T @ rows)).cpu().numpy()


def size_statistic(size: np.ndarray | float) -> np.ndarray:
    z = np.asarray(size, dtype=np.float64) / 10.0
    return np.stack((-z, -z * z), axis=-1)


def flatten_parameters(c_matrix: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.concatenate((np.asarray(c_matrix).reshape(-1), np.asarray(theta)))


def split_parameters(vector: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    return vector[:rank * rank].reshape(rank, rank), vector[rank * rank:]


def project_size(theta: np.ndarray, zmax: float) -> np.ndarray:
    """Euclidean projection onto c>=0 and a+zmax*c>=0 (a convex wedge)."""
    value = np.asarray(theta, dtype=np.float64)
    if value[1] >= 0.0 and value[0] + zmax * value[1] >= 0.0:
        return value.copy()
    candidates = [np.zeros(2, dtype=np.float64)]
    # Boundary ray c=0, a>=0.
    candidates.append(np.asarray([max(value[0], 0.0), 0.0]))
    # Boundary ray a+zmax*c=0, c>=0.
    direction = np.asarray([-zmax, 1.0])
    coefficient = max(float(value @ direction) / float(direction @ direction), 0.0)
    candidates.append(coefficient * direction)
    return min(candidates, key=lambda x: float(np.square(x - value).sum()))


def project(vector: np.ndarray, rank: int, spectral_max: float,
            zmax: float) -> np.ndarray:
    c_matrix, theta = split_parameters(vector, rank)
    c_matrix = 0.5 * (c_matrix + c_matrix.T)
    eigenvalue, eigenvector = np.linalg.eigh(c_matrix)
    eigenvalue = np.clip(eigenvalue, 0.0, spectral_max * spectral_max)
    c_matrix = (eigenvector * eigenvalue[None, :]) @ eigenvector.T
    return flatten_parameters(c_matrix, project_size(theta, zmax))


def objective_gradient(vector: np.ndarray, observed: np.ndarray, draws: np.ndarray,
                       rank: int, ridge: float, size_ridge: float
                       ) -> tuple[float, np.ndarray, np.ndarray]:
    logits = np.einsum("mdp,p->md", draws, vector, optimize=True)
    log_normalizer = logsumexp(logits, axis=1) - np.log(draws.shape[1])
    objective = float(np.mean(observed @ vector - log_normalizer))
    centred = logits - logsumexp(logits, axis=1, keepdims=True)
    weight = np.exp(centred)
    expectation = np.einsum("md,mdp->mp", weight, draws, optimize=True)
    gradient = np.mean(observed - expectation, axis=0)
    second = np.einsum("md,mdp->mp", weight, np.square(draws), optimize=True)
    fisher_diagonal = np.mean(second - np.square(expectation), axis=0)
    interaction_width = rank * rank
    objective -= 0.5 * ridge * float(vector[:interaction_width] @
                                     vector[:interaction_width])
    objective -= 0.5 * size_ridge * float(vector[interaction_width:] @
                                          vector[interaction_width:])
    gradient[:interaction_width] -= ridge * vector[:interaction_width]
    gradient[interaction_width:] -= size_ridge * vector[interaction_width:]
    fisher_diagonal[:interaction_width] += ridge
    fisher_diagonal[interaction_width:] += size_ridge
    return objective, gradient, fisher_diagonal


def projected_solve(observed: np.ndarray, draws: np.ndarray, rank: int,
                    spectral_max: float, zmax: float, ridge: float,
                    size_ridge: float, max_iterations: int, tolerance: float,
                    label: str) -> tuple[np.ndarray, dict]:
    vector = np.zeros(rank * rank + 2, dtype=np.float64)
    objective, gradient, fisher_diagonal = objective_gradient(
        vector, observed, draws, rank, ridge, size_ridge)
    initial_objective = objective
    step = 1.0
    history = [objective]
    converged = False
    projected_norm = float("inf")
    for iteration in range(1, max_iterations + 1):
        unit_projection = project(
            vector + gradient, rank, spectral_max, zmax) - vector
        projected_norm = float(np.linalg.norm(unit_projection))
        if projected_norm <= tolerance:
            converged = True
            break
        # A diagonal conditional-Fisher metric removes the otherwise severe scale
        # mismatch between size and pair statistics. Projection plus Armijo still
        # determines acceptance, so preconditioning cannot decrease the objective.
        direction_seed = gradient / np.maximum(fisher_diagonal, 1e-6)
        accepted = False
        trial_step = step
        for _ in range(40):
            candidate = project(
                vector + trial_step * direction_seed, rank, spectral_max, zmax)
            direction = candidate - vector
            directional = float(gradient @ direction)
            if directional <= 0.0:
                candidate = project(
                    vector + trial_step * gradient, rank, spectral_max, zmax)
                direction = candidate - vector
                directional = float(gradient @ direction)
            if np.linalg.norm(direction) <= tolerance * 0.1:
                converged = projected_norm <= 10.0 * tolerance
                accepted = True
                break
            candidate_objective, candidate_gradient, candidate_fisher_diagonal = objective_gradient(
                candidate, observed, draws, rank, ridge, size_ridge)
            if candidate_objective >= objective + 1e-4 * directional:
                vector = candidate
                objective = candidate_objective
                gradient = candidate_gradient
                fisher_diagonal = candidate_fisher_diagonal
                history.append(objective)
                step = min(trial_step * 1.5, 100.0)
                accepted = True
                break
            trial_step *= 0.5
        if not accepted:
            raise RuntimeError(f"{label}: Armijo line search failed")
        if iteration % 10 == 0 or converged:
            print(f"[natural-mcle] {label} iter={iteration} objective={objective:.8f} "
                  f"projected_grad={projected_norm:.3e} step={trial_step:.3e}",
                  flush=True)
        if converged:
            break
    monotone = bool(np.all(np.diff(np.asarray(history)) >= -1e-12))
    report = {
        "iterations": iteration,
        "converged": converged,
        "initial_penalized_objective": initial_objective,
        "final_penalized_objective": objective,
        "projected_gradient_norm": projected_norm,
        "accepted_steps_monotone": monotone,
        "final_step": step,
    }
    if not monotone:
        raise RuntimeError(f"{label}: accepted objective was not monotone")
    if not converged:
        raise RuntimeError(
            f"{label}: projected solve did not converge in {max_iterations} iterations")
    return vector, report


def evaluate(vector: np.ndarray, observed: np.ndarray, draws: np.ndarray) -> dict:
    logits = np.einsum("mdp,p->md", draws, vector, optimize=True)
    gain = observed @ vector - (
        logsumexp(logits, axis=1) - np.log(draws.shape[1]))
    normalized = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    ess = 1.0 / np.square(normalized).sum(axis=1)
    observed_size = -10.0 * observed[:, -2]
    draw_size = -10.0 * draws[:, :, -2]
    model_size_mean = float(np.mean(np.sum(normalized * draw_size, axis=1)))
    model_size_second = float(np.mean(
        np.sum(normalized * np.square(draw_size), axis=1)))
    return {
        "gain": float(gain.mean()),
        "gain_se": float(gain.std(ddof=1) / np.sqrt(len(gain))),
        "gain_lcb95": float(gain.mean() - 1.96 * gain.std(ddof=1) / np.sqrt(len(gain))),
        "ess_min": float(ess.min()),
        "ess_p01": float(np.quantile(ess, 0.01)),
        "ess_median": float(np.median(ess)),
        "ess_mean": float(ess.mean()),
        "ess_fraction_median": float(np.median(ess) / draws.shape[1]),
        "observed_size_mean": float(observed_size.mean()),
        "observed_size_variance": float(observed_size.var()),
        "model_size_mean": model_size_mean,
        "model_size_variance": model_size_second - model_size_mean ** 2,
    }


def atomic_save(path: Path, payload: dict) -> None:
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--spectral", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=12000)
    parser.add_argument("--draws", type=int, default=64)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--rank", type=int, default=6)
    parser.add_argument("--score-mass", type=float, default=1.0)
    parser.add_argument("--spectral-max", type=float, default=1.0)
    parser.add_argument("--ridges", type=float, nargs="+",
                        default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    parser.add_argument("--size-ridge", type=float, default=1e-6)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--tolerance", type=float, default=1e-3,
                        help=("Euclidean projected-gradient tolerance; 1e-3 leaves "
                              "less than roughly 1e-4 nats in observed probes"))
    parser.add_argument("--minimum-crossfit-gain", type=float, default=0.005)
    parser.add_argument("--minimum-half-gain", type=float, default=0.0)
    parser.add_argument("--minimum-ess-fraction", type=float, default=0.20)
    parser.add_argument("--minimum-ess-p01", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=29201)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/candidate.pt"))
    args = parser.parse_args()
    if args.contexts < 4 or args.draws < 2:
        raise ValueError("at least four contexts and two draws are required")
    torch.set_num_threads(args.threads)
    data = build()
    parent_path = args.parent if args.parent.is_absolute() else ROOT / args.parent
    spectral_path = args.spectral if args.spectral.is_absolute() else ROOT / args.spectral
    model, parent_blob, meta = load_checkpoint(parent_path, data)
    if float(model.phi.abs().max()) != 0.0:
        raise RuntimeError("natural-parameter proposal must be a Phi=0 additive parent")
    basis_np, keep_count = spectral_basis(
        spectral_path, args.rank, args.score_mass)
    basis = torch.as_tensor(basis_np, dtype=model.phi.dtype)

    population = supported_trips(data, 0, int(meta["nmax"]))
    context_count = len(population) if args.contexts == 0 else args.contexts
    if context_count > len(population):
        raise ValueError("requested more contexts than the supported training population")
    rng = np.random.default_rng(args.seed)
    trips = population[rng.permutation(len(population))[:context_count]]
    half = rng.random(context_count) < 0.5
    if half.all() or (~half).all():
        raise RuntimeError("degenerate cross-fit split")
    width = args.rank * args.rank + 2
    observed = np.empty((context_count, width), dtype=np.float64)
    draws = np.empty((context_count, args.draws, width), dtype=np.float64)
    batcher = Batcher(
        data, Features(int(data["n_item"]), int(data["n_store"]), 712),
        int(meta["nmax"]))
    generator = torch.Generator().manual_seed(args.seed + 1)
    for start in range(0, context_count, args.batch):
        sub = trips[start:start + args.batch]
        ix, ctx, _line_ctx, house, li, lt, _lc, _lq = batcher.make(sub)
        model.house, model.ctx = house, ctx
        z = torch.zeros(ix.B, model.Kz, dtype=model.phi.dtype)
        states = conditional_slots_repeated(model, ix, z, 0.0, args.draws, generator)
        for local in range(ix.B):
            observed_items = li[lt == local]
            observed[start + local, :args.rank * args.rank] = pair_statistic(
                observed_items, basis).reshape(-1)
            observed[start + local, args.rank * args.rank:] = size_statistic(
                float(torch.unique(observed_items).numel()))
            for draw_index, state in enumerate(states):
                items = ix.item[state[local]]
                unique_size = float(torch.unique(items).numel())
                draws[start + local, draw_index, :args.rank * args.rank] = \
                    pair_statistic(items, basis).reshape(-1)
                draws[start + local, draw_index, args.rank * args.rank:] = \
                    size_statistic(unique_size)
        if (start // args.batch + 1) % 10 == 0 or start + args.batch >= context_count:
            print(f"[natural-mcle] sampled {min(start + args.batch, context_count)}/"
                  f"{context_count} contexts", flush=True)

    zmax = float(model.rho_0_free.numel()) / 10.0
    ridge_rows = []
    candidates = {}
    for ridge in args.ridges:
        vector_a, solve_a = projected_solve(
            observed[half], draws[half], args.rank, args.spectral_max, zmax,
            ridge, args.size_ridge, args.max_iterations, args.tolerance,
            f"ridge={ridge:g}/half-a")
        vector_b, solve_b = projected_solve(
            observed[~half], draws[~half], args.rank, args.spectral_max, zmax,
            ridge, args.size_ridge, args.max_iterations, args.tolerance,
            f"ridge={ridge:g}/half-b")
        a_on_b = evaluate(vector_a, observed[~half], draws[~half])
        b_on_a = evaluate(vector_b, observed[half], draws[half])
        row = {
            "ridge": ridge,
            "a_fit_b": a_on_b,
            "b_fit_a": b_on_a,
            "mean_crossfit_gain": 0.5 * (a_on_b["gain"] + b_on_a["gain"]),
            "minimum_crossfit_gain": min(a_on_b["gain"], b_on_a["gain"]),
            "solve_a": solve_a,
            "solve_b": solve_b,
        }
        ridge_rows.append(row)
        candidates[ridge] = (vector_a, vector_b)
    eligible = [row for row in ridge_rows
                if row["minimum_crossfit_gain"] > args.minimum_half_gain]
    selected = max(eligible or ridge_rows, key=lambda row: row["mean_crossfit_gain"])
    selected_ridge = float(selected["ridge"])
    vector, full_solve = projected_solve(
        observed, draws, args.rank, args.spectral_max, zmax, selected_ridge,
        args.size_ridge, args.max_iterations, args.tolerance, "full")
    full = evaluate(vector, observed, draws)
    c_matrix, theta = split_parameters(vector, args.rank)
    c_eigenvalues = np.linalg.eigvalsh(c_matrix)[::-1]
    crossfit_ess_fraction = min(
        selected["a_fit_b"]["ess_fraction_median"],
        selected["b_fit_a"]["ess_fraction_median"])
    crossfit_ess_p01 = min(
        selected["a_fit_b"]["ess_p01"], selected["b_fit_a"]["ess_p01"])
    accepted = bool(
        selected["minimum_crossfit_gain"] > args.minimum_half_gain
        and selected["mean_crossfit_gain"] >= args.minimum_crossfit_gain
        and crossfit_ess_fraction >= args.minimum_ess_fraction
        and crossfit_ess_p01 >= args.minimum_ess_p01
        and full_solve["converged"] and full_solve["accepted_steps_monotone"])
    result = {
        "method": "constrained_common_random_number_monte_carlo_mle",
        "parent": str(parent_path),
        "parent_iteration": int(parent_blob["iter"]),
        "spectral": str(spectral_path),
        "contexts": context_count,
        "draws_per_context": args.draws,
        "rank": args.rank,
        "natural_parameters": width,
        "interaction_products": keep_count,
        "score_mass": args.score_mass,
        "half_contexts": [int(half.sum()), int((~half).sum())],
        "spectral_max": args.spectral_max,
        "size_tail_constraints": {"quadratic_nonnegative": True,
                                  "delta_rho_at_nmax_nonnegative": True,
                                  "nmax": int(model.rho_0_free.numel())},
        "ridge_audit": ridge_rows,
        "selected_ridge": selected_ridge,
        "selected_crossfit_gain": selected["mean_crossfit_gain"],
        "selected_minimum_half_gain": selected["minimum_crossfit_gain"],
        "crossfit_ess_fraction": crossfit_ess_fraction,
        "crossfit_ess_p01": crossfit_ess_p01,
        "minimum_required_crossfit_gain": args.minimum_crossfit_gain,
        "minimum_required_half_gain": args.minimum_half_gain,
        "minimum_required_ess_fraction": args.minimum_ess_fraction,
        "minimum_required_ess_p01": args.minimum_ess_p01,
        "full_solve": full_solve,
        "full_fit": full,
        "candidate_c": c_matrix.tolist(),
        "candidate_c_eigenvalues": c_eigenvalues.tolist(),
        "theta_scaled": theta.tolist(),
        "rho0_linear_a": float(theta[0] / 10.0),
        "rho0_quadratic_c": float(theta[1] / 100.0),
        "accepted_for_smolyak_audit": accepted,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not accepted:
        raise RuntimeError("natural-parameter candidate failed cross-fit or ESS gate")

    eigenvalue, eigenvector = np.linalg.eigh(c_matrix)
    positive = eigenvalue > max(float(eigenvalue.max()) * 1e-10, 1e-12)
    factor = eigenvector[:, positive] * np.sqrt(eigenvalue[positive])[None, :]
    phi = basis_np @ factor
    with torch.no_grad():
        model.phi.zero_()
        model.phi[:, :phi.shape[1]].copy_(torch.as_tensor(phi, dtype=model.phi.dtype))
        n = torch.arange(1, model.rho_0_free.numel() + 1,
                         dtype=model.rho_0_free.dtype)
        model.rho_0_free.add_(theta[0] * n / 10.0 + theta[1] * n.square() / 100.0)
    payload = {
        "format": 3,
        "estimator": "constrained_crn_monte_carlo_mle_version4_natural_block",
        "iter": 0,
        "model": model.state_dict(),
        "config": {**parent_blob["config"],
                   "artifact": parent_blob["config"]["artifact"]},
        "parent": str(parent_path),
        "parent_iteration": int(parent_blob["iter"]),
        "active_rank": int(phi.shape[1]),
        "interaction_products": keep_count,
        "best_validation": None,
        "best_iteration": 0,
        "evaluations": [],
        "records": [],
        "natural_mcle_report": str(report_path),
    }
    atomic_save(output, payload)
    print(f"[natural-mcle] accepted checkpoint: {output}", flush=True)


if __name__ == "__main__":
    main()
