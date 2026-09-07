#!/usr/bin/env python3
"""Deterministic constrained MC-MLE for the Version-4 interaction/size block.

The additive model is an exact proposal.  In a fixed orthonormal product basis U,

    K = U C U',                 0 <= C <= spectral_max^2 I,
    Delta b_j = a' v_j,
    Delta rho_0(n) = s_1 (n/10) + s_2 (n/10)^2,

where v_j is a centred orthonormal basis spanning the selected interaction directions.
The additive correction is optional; when enabled, the log-density ratio is linear in
the natural parameters (C, a, s_1, s_2).  Fixed
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

from checkpoint_io import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from pipeline_support import supported_trips
from provenance import file_sha256, strict_json_dumps
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


def additive_statistic(items: torch.Tensor, basis: torch.Tensor) -> np.ndarray:
    """Sufficient statistic for a centred low-rank product-utility correction."""
    return basis[torch.unique(items)].sum(0).cpu().numpy()


def centered_additive_basis(basis: np.ndarray) -> np.ndarray:
    """Orthonormal centred span used to polish item intercepts without a size gauge."""
    centered = np.asarray(basis, dtype=np.float64) - np.asarray(
        basis, dtype=np.float64).mean(axis=0, keepdims=True)
    value, singular, _ = np.linalg.svd(centered, full_matrices=False)
    if singular[-1] <= max(float(singular[0]), 1.0) * 1e-10:
        raise RuntimeError("centred interaction span loses an additive direction")
    return value


def size_statistic(size: np.ndarray | float) -> np.ndarray:
    z = np.asarray(size, dtype=np.float64) / 10.0
    return np.stack((-z, -z * z), axis=-1)


def flatten_parameters(c_matrix: np.ndarray, theta: np.ndarray) -> np.ndarray:
    return np.concatenate((np.asarray(c_matrix).reshape(-1), np.asarray(theta)))


def split_parameters(vector: np.ndarray, rank: int) -> tuple[np.ndarray, np.ndarray]:
    return vector[:rank * rank].reshape(rank, rank), vector[rank * rank:]


def split_joint_parameters(vector: np.ndarray, rank: int,
                           additive_rank: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    interaction_width = rank * rank
    expected = interaction_width + additive_rank + 2
    if len(vector) != expected:
        raise ValueError(f"expected {expected} natural parameters, received {len(vector)}")
    c_matrix = vector[:interaction_width].reshape(rank, rank)
    additive = vector[interaction_width:interaction_width + additive_rank]
    size = vector[-2:]
    return c_matrix, additive, size


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
            zmax: float, additive_rank: int = 0) -> np.ndarray:
    c_matrix, additive, theta = split_joint_parameters(
        vector, rank, additive_rank)
    c_matrix = 0.5 * (c_matrix + c_matrix.T)
    eigenvalue, eigenvector = np.linalg.eigh(c_matrix)
    eigenvalue = np.clip(eigenvalue, 0.0, spectral_max * spectral_max)
    c_matrix = (eigenvector * eigenvalue[None, :]) @ eigenvector.T
    return np.concatenate((c_matrix.reshape(-1), additive,
                           project_size(theta, zmax)))


def objective_gradient(vector: np.ndarray, observed: np.ndarray, draws: np.ndarray,
                       rank: int, ridge: float, size_ridge: float,
                       additive_rank: int = 0, additive_ridge: float = 0.0
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
    additive_slice = slice(interaction_width, interaction_width + additive_rank)
    size_slice = slice(interaction_width + additive_rank, None)
    objective -= 0.5 * ridge * float(vector[:interaction_width] @
                                     vector[:interaction_width])
    objective -= 0.5 * additive_ridge * float(
        vector[additive_slice] @ vector[additive_slice])
    objective -= 0.5 * size_ridge * float(vector[size_slice] @ vector[size_slice])
    gradient[:interaction_width] -= ridge * vector[:interaction_width]
    gradient[additive_slice] -= additive_ridge * vector[additive_slice]
    gradient[size_slice] -= size_ridge * vector[size_slice]
    fisher_diagonal[:interaction_width] += ridge
    fisher_diagonal[additive_slice] += additive_ridge
    fisher_diagonal[size_slice] += size_ridge
    return objective, gradient, fisher_diagonal


def projected_solve(observed: np.ndarray, draws: np.ndarray, rank: int,
                    spectral_max: float, zmax: float, ridge: float,
                    size_ridge: float, max_iterations: int, tolerance: float,
                    label: str, *, additive_rank: int = 0,
                    additive_ridge: float = 0.0) -> tuple[np.ndarray, dict]:
    vector = np.zeros(rank * rank + additive_rank + 2, dtype=np.float64)
    objective, gradient, fisher_diagonal = objective_gradient(
        vector, observed, draws, rank, ridge, size_ridge,
        additive_rank, additive_ridge)
    initial_objective = objective
    step = 1.0
    history = [objective]
    converged = False
    projected_norm = float("inf")
    for iteration in range(1, max_iterations + 1):
        unit_projection = project(
            vector + gradient, rank, spectral_max, zmax, additive_rank) - vector
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
                vector + trial_step * direction_seed,
                rank, spectral_max, zmax, additive_rank)
            direction = candidate - vector
            directional = float(gradient @ direction)
            if directional <= 0.0:
                candidate = project(
                    vector + trial_step * gradient,
                    rank, spectral_max, zmax, additive_rank)
                direction = candidate - vector
                directional = float(gradient @ direction)
            if np.linalg.norm(direction) <= tolerance * 0.1:
                if projected_norm <= 10.0 * tolerance:
                    converged = True
                    accepted = True
                    break
                raise RuntimeError(
                    f"{label}: line search stalled above convergence tolerance")
            candidate_objective, candidate_gradient, candidate_fisher_diagonal = objective_gradient(
                candidate, observed, draws, rank, ridge, size_ridge,
                additive_rank, additive_ridge)
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
    gain = likelihood_gain(vector, observed, draws)
    logits = np.einsum("mdp,p->md", draws, vector, optimize=True)
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


def likelihood_gain(vector: np.ndarray, observed: np.ndarray,
                    draws: np.ndarray) -> np.ndarray:
    """Per-context CRN likelihood-ratio gain over the additive proposal."""
    logits = np.einsum("mdp,p->md", draws, vector, optimize=True)
    return observed @ vector - (
        logsumexp(logits, axis=1) - np.log(draws.shape[1]))


def paired_gain_summary(delta: np.ndarray) -> dict:
    value = np.asarray(delta, dtype=np.float64)
    if value.ndim != 1 or len(value) < 2 or not np.isfinite(value).all():
        raise ValueError("paired gain requires at least two finite contexts")
    mean = float(value.mean())
    standard_error = float(value.std(ddof=1) / np.sqrt(len(value)))
    return {
        "contexts": int(len(value)),
        "mean": mean,
        "standard_error": standard_error,
        "lower_95": mean - 1.96 * standard_error,
        "upper_95": mean + 1.96 * standard_error,
    }


def additive_polish_decision(delta_a_on_b: np.ndarray,
                             delta_b_on_a: np.ndarray,
                             minimum_lower_95: float = 0.0) -> dict:
    """Predeclared nested-model gate for the optional additive correction."""
    summary_a = paired_gain_summary(delta_a_on_b)
    summary_b = paired_gain_summary(delta_b_on_a)
    combined = paired_gain_summary(np.concatenate((delta_a_on_b, delta_b_on_a)))
    accepted = bool(
        summary_a["mean"] > 0.0 and summary_b["mean"] > 0.0
        and combined["lower_95"] > minimum_lower_95)
    return {
        "accepted": accepted,
        "minimum_required_combined_lower_95": minimum_lower_95,
        "a_fit_b_increment": summary_a,
        "b_fit_a_increment": summary_b,
        "combined_increment": combined,
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
    parser.add_argument("--joint-additive-polish", action="store_true",
                        help=("jointly fit a centred rank-r correction to the existing "
                              "product intercepts in the accepted interaction span"))
    parser.add_argument("--additive-polish-ridge", type=float, default=1e-3)
    parser.add_argument("--minimum-additive-polish-lcb", type=float, default=0.0,
                        help=("minimum paired 95%% lower bound for accepting the joint "
                              "additive correction over the restricted interaction fit"))
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
    model, parent_blob, meta = load_checkpoint(
        parent_path, data,
        required_capabilities=("conditional_nonempty_incidence",))
    if float(model.phi.abs().max()) != 0.0:
        raise RuntimeError("natural-parameter proposal must be a Phi=0 additive parent")
    basis_np, keep_count = spectral_basis(
        spectral_path, args.rank, args.score_mass)
    spectral_report = json.loads(spectral_path.with_suffix(".json").read_text())
    if spectral_report.get("basis_sha256") != file_sha256(spectral_path):
        raise RuntimeError("spectral basis failed its report content digest")
    if spectral_report.get("parent_sha256") != file_sha256(parent_path):
        raise RuntimeError("spectral basis was not built from this additive checkpoint")
    if spectral_report.get("data_fingerprint_sha256") != parent_blob.get(
            "data_fingerprint_sha256"):
        raise RuntimeError("spectral basis and additive checkpoint data differ")
    basis = torch.as_tensor(basis_np, dtype=model.phi.dtype)
    additive_rank = args.rank if args.joint_additive_polish else 0
    additive_basis_np = (centered_additive_basis(basis_np)
                         if additive_rank else np.empty((len(basis_np), 0)))
    additive_basis = torch.as_tensor(additive_basis_np, dtype=model.phi.dtype)

    population = supported_trips(data, 0, int(meta["nmax"]))
    context_count = len(population) if args.contexts == 0 else args.contexts
    if context_count > len(population):
        raise ValueError("requested more contexts than the supported training population")
    rng = np.random.default_rng(args.seed)
    trips = population[rng.permutation(len(population))[:context_count]]
    half = rng.random(context_count) < 0.5
    if half.all() or (~half).all():
        raise RuntimeError("degenerate cross-fit split")
    width = args.rank * args.rank + additive_rank + 2
    observed = np.empty((context_count, width), dtype=np.float64)
    draws = np.empty((context_count, args.draws, width), dtype=np.float64)
    features = Features(int(data["n_item"]), int(data["n_store"]), 712,
                        include_recency=False)
    batcher = Batcher(data, features, int(meta["nmax"]), include_recency=False)
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
            if additive_rank:
                begin = args.rank * args.rank
                observed[start + local, begin:begin + additive_rank] = \
                    additive_statistic(observed_items, additive_basis)
            observed[start + local, -2:] = size_statistic(
                float(torch.unique(observed_items).numel()))
            for draw_index, state in enumerate(states):
                items = ix.item[state[local]]
                unique_size = float(torch.unique(items).numel())
                draws[start + local, draw_index, :args.rank * args.rank] = \
                    pair_statistic(items, basis).reshape(-1)
                if additive_rank:
                    begin = args.rank * args.rank
                    draws[start + local, draw_index,
                          begin:begin + additive_rank] = \
                        additive_statistic(items, additive_basis)
                draws[start + local, draw_index, -2:] = \
                    size_statistic(unique_size)
        if (start // args.batch + 1) % 10 == 0 or start + args.batch >= context_count:
            print(f"[natural-mcle] sampled {min(start + args.batch, context_count)}/"
                  f"{context_count} contexts", flush=True)

    zmax = float(model.rho_0_free.numel()) / 10.0
    # Ridge selection remains anchored to the established interaction-plus-size model.
    # The additive polish is tested only after that choice, so it cannot hijack interaction
    # acceptance or choose its own favorable regularization from the same cross-fit panel.
    restricted_columns = np.r_[np.arange(args.rank * args.rank),
                               np.arange(width - 2, width)]
    restricted_observed = observed[:, restricted_columns]
    restricted_draws = draws[:, :, restricted_columns]
    ridge_rows = []
    restricted_candidates = {}
    for ridge in args.ridges:
        vector_a, solve_a = projected_solve(
            restricted_observed[half], restricted_draws[half],
            args.rank, args.spectral_max, zmax,
            ridge, args.size_ridge, args.max_iterations, args.tolerance,
            f"ridge={ridge:g}/half-a")
        vector_b, solve_b = projected_solve(
            restricted_observed[~half], restricted_draws[~half],
            args.rank, args.spectral_max, zmax,
            ridge, args.size_ridge, args.max_iterations, args.tolerance,
            f"ridge={ridge:g}/half-b")
        a_on_b = evaluate(
            vector_a, restricted_observed[~half], restricted_draws[~half])
        b_on_a = evaluate(
            vector_b, restricted_observed[half], restricted_draws[half])
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
        restricted_candidates[float(ridge)] = (vector_a, vector_b)
    eligible = [row for row in ridge_rows
                if row["minimum_crossfit_gain"] > args.minimum_half_gain]
    selected = max(eligible or ridge_rows, key=lambda row: row["mean_crossfit_gain"])
    selected_ridge = float(selected["ridge"])
    restricted_a, restricted_b = restricted_candidates[selected_ridge]
    additive_gate = {
        "requested": bool(additive_rank),
        "accepted": False,
        "reason": "not requested",
    }
    if additive_rank:
        try:
            joint_a, joint_solve_a = projected_solve(
                observed[half], draws[half], args.rank, args.spectral_max, zmax,
                selected_ridge, args.size_ridge, args.max_iterations, args.tolerance,
                "joint-polish/half-a", additive_rank=additive_rank,
                additive_ridge=args.additive_polish_ridge)
            joint_b, joint_solve_b = projected_solve(
                observed[~half], draws[~half], args.rank, args.spectral_max, zmax,
                selected_ridge, args.size_ridge, args.max_iterations, args.tolerance,
                "joint-polish/half-b", additive_rank=additive_rank,
                additive_ridge=args.additive_polish_ridge)
        except RuntimeError as exc:
            polish_accepted = False
            additive_gate = {
                "requested": True,
                "accepted": False,
                "minimum_required_combined_lower_95": (
                    args.minimum_additive_polish_lcb),
                "reason": f"joint correction solver rejected safely: {exc}",
            }
        else:
            delta_a_on_b = (
                likelihood_gain(joint_a, observed[~half], draws[~half])
                - likelihood_gain(restricted_a, restricted_observed[~half],
                                  restricted_draws[~half]))
            delta_b_on_a = (
                likelihood_gain(joint_b, observed[half], draws[half])
                - likelihood_gain(restricted_b, restricted_observed[half],
                                  restricted_draws[half]))
            decision = additive_polish_decision(
                delta_a_on_b, delta_b_on_a,
                args.minimum_additive_polish_lcb)
            polish_accepted = decision["accepted"]
            additive_gate = {
                "requested": True,
                **decision,
                "solve_a": joint_solve_a,
                "solve_b": joint_solve_b,
                "reason": ("cross-fitted paired likelihood improvement"
                           if polish_accepted else
                           "joint correction did not beat the restricted fit out of fold"),
            }
    else:
        polish_accepted = False

    if polish_accepted:
        try:
            vector, full_solve = projected_solve(
                observed, draws, args.rank, args.spectral_max, zmax, selected_ridge,
                args.size_ridge, args.max_iterations, args.tolerance, "full-joint",
                additive_rank=additive_rank,
                additive_ridge=args.additive_polish_ridge)
        except RuntimeError as exc:
            polish_accepted = False
            additive_gate["accepted"] = False
            additive_gate["reason"] = (
                f"full joint correction solver rejected safely: {exc}")
    if polish_accepted:
        a_on_b = evaluate(joint_a, observed[~half], draws[~half])
        b_on_a = evaluate(joint_b, observed[half], draws[half])
    else:
        restricted_full, full_solve = projected_solve(
            restricted_observed, restricted_draws, args.rank,
            args.spectral_max, zmax, selected_ridge, args.size_ridge,
            args.max_iterations, args.tolerance, "full-restricted")
        vector = np.zeros(width, dtype=np.float64)
        vector[:args.rank * args.rank] = restricted_full[:args.rank * args.rank]
        vector[-2:] = restricted_full[-2:]
        a_on_b = evaluate(
            restricted_a, restricted_observed[~half], restricted_draws[~half])
        b_on_a = evaluate(
            restricted_b, restricted_observed[half], restricted_draws[half])
    full = evaluate(vector, observed, draws)
    c_matrix, additive_delta, theta = split_joint_parameters(
        vector, args.rank, additive_rank)
    c_eigenvalues = np.linalg.eigvalsh(c_matrix)[::-1]
    crossfit_ess_fraction = min(
        a_on_b["ess_fraction_median"], b_on_a["ess_fraction_median"])
    crossfit_ess_p01 = min(a_on_b["ess_p01"], b_on_a["ess_p01"])
    accepted = bool(
        min(a_on_b["gain"], b_on_a["gain"]) > args.minimum_half_gain
        and 0.5 * (a_on_b["gain"] + b_on_a["gain"])
        >= args.minimum_crossfit_gain
        and crossfit_ess_fraction >= args.minimum_ess_fraction
        and crossfit_ess_p01 >= args.minimum_ess_p01
        and full_solve["converged"] and full_solve["accepted_steps_monotone"])
    result = {
        "method": "constrained_common_random_number_monte_carlo_mle",
        "parent": str(parent_path),
        "parent_iteration": int(parent_blob["iter"]),
        "parent_sha256": file_sha256(parent_path),
        "data_fingerprint_sha256": parent_blob["data_fingerprint_sha256"],
        "spectral": str(spectral_path),
        "spectral_sha256": file_sha256(spectral_path),
        "contexts": context_count,
        "draws_per_context": args.draws,
        "rank": args.rank,
        "natural_parameters": width,
        "joint_additive_polish": bool(args.joint_additive_polish),
        "additive_polish_rank": additive_rank,
        "additive_polish_ridge": args.additive_polish_ridge,
        "additive_polish_coefficients": additive_delta.tolist(),
        "additive_polish_gate": additive_gate,
        "additive_polish_catalogue_mean": float(
            (additive_basis_np @ additive_delta).mean()) if additive_rank else 0.0,
        "additive_polish_catalogue_rms": float(np.sqrt(np.mean(np.square(
            additive_basis_np @ additive_delta)))) if additive_rank else 0.0,
        "interaction_products": keep_count,
        "score_mass": args.score_mass,
        "half_contexts": [int(half.sum()), int((~half).sum())],
        "spectral_max": args.spectral_max,
        "size_tail_constraints": {"quadratic_nonnegative": True,
                                  "delta_rho_at_nmax_nonnegative": True,
                                  "nmax": int(model.rho_0_free.numel())},
        "ridge_audit": ridge_rows,
        "selected_ridge": selected_ridge,
        "selected_crossfit_gain": 0.5 * (a_on_b["gain"] + b_on_a["gain"]),
        "selected_minimum_half_gain": min(a_on_b["gain"], b_on_a["gain"]),
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
    report_path.write_text(strict_json_dumps(result))
    print(strict_json_dumps(result), end="")
    if not accepted:
        raise RuntimeError("natural-parameter candidate failed cross-fit or ESS gate")

    eigenvalue, eigenvector = np.linalg.eigh(c_matrix)
    positive = eigenvalue > max(float(eigenvalue.max()) * 1e-10, 1e-12)
    factor = eigenvector[:, positive] * np.sqrt(eigenvalue[positive])[None, :]
    phi = basis_np @ factor
    with torch.no_grad():
        model.phi.zero_()
        model.phi[:, :phi.shape[1]].copy_(torch.as_tensor(phi, dtype=model.phi.dtype))
        if additive_rank:
            model.lam.add_(torch.as_tensor(
                additive_basis_np @ additive_delta, dtype=model.lam.dtype))
        n = torch.arange(1, model.rho_0_free.numel() + 1,
                         dtype=model.rho_0_free.dtype)
        model.rho_0_free.add_(theta[0] * n / 10.0 + theta[1] * n.square() / 100.0)
    payload = {
        "format": 3,
        "estimator": "constrained_crn_monte_carlo_mle_version4_natural_block",
        "iter": 0,
        "model": model.state_dict(),
        "fresh_artifact_digest": parent_blob["fresh_artifact_digest"],
        "data_fingerprint_sha256": parent_blob["data_fingerprint_sha256"],
        "config": {**parent_blob["config"],
                   "artifact": parent_blob["config"]["artifact"]},
        "parent": str(parent_path),
        "parent_iteration": int(parent_blob["iter"]),
        "parent_sha256": file_sha256(parent_path),
        "spectral_sha256": file_sha256(spectral_path),
        "active_rank": int(phi.shape[1]),
        "interaction_products": keep_count,
        "best_validation": None,
        "best_iteration": 0,
        "evaluations": [],
        "records": [],
        "trained_capabilities": {
            "conditional_nonempty_incidence": True,
            "gram_interactions": True,
            "recency": False,
            "quantities": False,
            "arrival_or_null_basket": False,
        },
        "natural_mcle_report": str(report_path),
    }
    atomic_save(output, payload)
    print(f"[natural-mcle] accepted checkpoint: {output}", flush=True)


if __name__ == "__main__":
    main()
