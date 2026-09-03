#!/usr/bin/env python3
"""Fit and safely cap the rank-one household size direction.

For a fixed Version-4 checkpoint, adding kappa_h to every product utility tilts only the
conditional size law:

    p_k(n | x) proportional to p_0(n | x) exp(n kappa_h).

Each household problem is one-dimensional and strictly concave. Ridge is selected by
swapped within-household trip halves. The final scalar is capped by a deterministic
screen-tail constraint. No basket-composition probability conditional on size changes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import logsumexp

from audit_particle_counterfactual_generation import ROOT, load_checkpoint
from audit_population_size import resumable_screen, screen_signature
from data import build
from features import Features
from fit import Batcher
from fit_interaction_particles import supported_trips


torch.set_default_dtype(torch.float64)


def normalized_tilt(log_probability: np.ndarray, kappa: np.ndarray,
                    household: np.ndarray) -> np.ndarray:
    size = np.arange(1, log_probability.shape[1] + 1, dtype=np.float64)
    value = log_probability + kappa[household, None] * size
    value -= logsumexp(value, axis=1, keepdims=True)
    return value


def solve_households(log_probability: np.ndarray, observed: np.ndarray,
                     household: np.ndarray, selected: np.ndarray, n_household: int,
                     ridge: float) -> np.ndarray:
    """Unique penalized maximizer for each household, using monotone bisection."""
    size = np.arange(1, log_probability.shape[1] + 1, dtype=np.float64)
    result = np.zeros(n_household, dtype=np.float64)
    for h in range(n_household):
        index = selected[household[selected] == h]
        if not len(index):
            continue
        target = float(observed[index].sum())
        def score(value: float) -> float:
            tilted = log_probability[index] + value * size
            tilted -= logsumexp(tilted, axis=1, keepdims=True)
            return (target - float((np.exp(tilted) @ size).sum())
                    - ridge * value)

        lower, upper = -0.5, 0.5
        while score(lower) < 0.0:
            lower *= 2.0
        while score(upper) > 0.0:
            upper *= 2.0
        for _ in range(44):
            value = 0.5 * (lower + upper)
            if score(value) > 0.0:
                lower = value
            else:
                upper = value
        result[h] = 0.5 * (lower + upper)
    return result


def gain(log_probability: np.ndarray, observed: np.ndarray,
         household: np.ndarray, selected: np.ndarray,
         kappa: np.ndarray) -> np.ndarray:
    size = np.arange(1, log_probability.shape[1] + 1, dtype=np.float64)
    value = kappa[household[selected]]
    return (value * observed[selected]
            - logsumexp(
                log_probability[selected] + value[:, None] * size, axis=1))


def chronological_folds(household: np.ndarray, day: np.ndarray,
                        n_household: int) -> np.ndarray:
    fold = np.zeros(len(household), dtype=np.int8)
    for h in range(n_household):
        index = np.flatnonzero(household == h)
        index = index[np.argsort(day[index], kind="stable")]
        fold[index[1::2]] = 1
    return fold


def cap_households(log_probability: np.ndarray, household: np.ndarray,
                   kappa: np.ndarray, n_household: int,
                   screen_tail_cap: float) -> tuple[np.ndarray, np.ndarray]:
    """Project onto kappa_h <= u_h, where the worst screen tail equals the cap."""
    size = np.arange(1, log_probability.shape[1] + 1, dtype=np.float64)
    result = kappa.copy()
    upper_bound = np.full(n_household, np.inf, dtype=np.float64)
    for h in range(n_household):
        index = np.flatnonzero(household == h)
        if not len(index):
            continue

        def maximum_tail(value: float) -> float:
            tilted = log_probability[index] + value * size
            tilted -= logsumexp(tilted, axis=1, keepdims=True)
            return float(np.exp(tilted)[:, 59:].sum(1).max())

        if maximum_tail(result[h]) <= screen_tail_cap:
            continue
        lower, upper = -0.5, result[h]
        while maximum_tail(lower) > screen_tail_cap:
            lower *= 2.0
        for _ in range(44):
            value = 0.5 * (lower + upper)
            if maximum_tail(value) <= screen_tail_cap:
                lower = value
            else:
                upper = value
        result[h] = upper_bound[h] = 0.5 * (lower + upper)
    return result, upper_bound


def seed_population_cache(checkpoint: Path, population_output: Path,
                          population: np.ndarray, rank: int, levels: list[int],
                          log_probability: np.ndarray, observed: np.ndarray) -> str:
    """Write the exactly tilted low-rule law for the final checkpoint.

    The common household shift changes the cached size law analytically, so rerunning the
    full-catalogue DP would be redundant. The ordinary population audit verifies the
    signature before accepting this cache and still performs its q(confirm) panels.
    """
    signature = screen_signature(checkpoint, population, rank, levels)
    prefix = population_output.with_name(
        f"{population_output.stem}.screen-{signature[:12]}")
    np.save(str(prefix) + ".log_probability.npy",
            np.asarray(log_probability, dtype=np.float64))
    np.save(str(prefix) + ".observed.npy",
            np.asarray(observed, dtype=np.int16))
    np.save(str(prefix) + ".level.npy",
            np.full(len(population), levels[0], dtype=np.int16))
    progress = {
        "signature": signature, "contexts": len(population),
        "rank": rank, "levels": levels,
        "completed_contexts": len(population), "complete": True,
    }
    Path(str(prefix) + ".progress.json").write_text(
        json.dumps(progress, indent=2) + "\n")
    return str(prefix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--screen-level", type=int, required=True)
    parser.add_argument("--screen-tail-cap", type=float, default=0.35)
    parser.add_argument("--ridge-grid", type=float, nargs="+",
                        default=[800, 1600, 2400, 3200, 4800, 6400, 9600])
    parser.add_argument("--minimum-crossfit-gain", type=float, default=0.0)
    parser.add_argument("--chunk", type=int, default=48)
    parser.add_argument("--contexts", type=int, default=0,
                        help="training contexts; 0 uses the complete population")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/candidate_rank1.pt"))
    parser.add_argument("--report", type=Path,
                        default=Path("artifacts/candidate_rank1.json"))
    parser.add_argument("--population-output", type=Path,
                        default=Path("reports/population_size.json"))
    args = parser.parse_args()
    if not 0 < args.screen_tail_cap < 0.5:
        raise ValueError("screen-tail-cap must lie strictly between zero and 0.5")
    if any(value < 0 for value in args.ridge_grid):
        raise ValueError("ridge values must be nonnegative")

    torch.set_num_threads(args.threads)
    data = build()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() \
        else ROOT / args.checkpoint
    model, blob, meta = load_checkpoint(checkpoint, data)
    if not model.household_size_rank1:
        raise RuntimeError(
            "checkpoint does not use the identified rank-one household-size coordinate")
    population = supported_trips(data, 0, int(meta["nmax"]))
    if args.contexts < 0:
        raise ValueError("contexts must be nonnegative")
    if args.contexts:
        population = population[:min(args.contexts, len(population))]
    batcher = Batcher(
        data, Features(int(data["n_item"]), int(data["n_store"]), 712),
        int(meta["nmax"]))
    report_path = args.report if args.report.is_absolute() else ROOT / args.report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    levels = [args.screen_level, args.screen_level + 1, args.screen_level + 2]
    observed, base_log_probability, used_level, provenance = resumable_screen(
        model, batcher, population, checkpoint, args.rank, levels, args.chunk,
        report_path, f"rank1-base-q{args.screen_level}")
    household = data["trip_user"][population].astype(np.int64, copy=False)
    day = data["trip_day"][population].astype(np.int64, copy=False)
    n_household = int(data["n_user"])
    fold = chronological_folds(household, day, n_household)
    audit = []
    for ridge in args.ridge_grid:
        kappa_0 = solve_households(
            base_log_probability, observed, household,
            np.flatnonzero(fold == 0), n_household, ridge)
        kappa_1 = solve_households(
            base_log_probability, observed, household,
            np.flatnonzero(fold == 1), n_household, ridge)
        gain_0 = gain(
            base_log_probability, observed, household,
            np.flatnonzero(fold == 0), kappa_1)
        gain_1 = gain(
            base_log_probability, observed, household,
            np.flatnonzero(fold == 1), kappa_0)
        heldout = np.concatenate((gain_0, gain_1))
        audit.append({
            "ridge": float(ridge),
            "gain": float(heldout.mean()),
            "gain_se": float(heldout.std(ddof=1) / np.sqrt(len(heldout))),
            "gain_lcb95": float(
                heldout.mean() - 1.96 * heldout.std(ddof=1) / np.sqrt(len(heldout))),
            "fold_0_gain": float(gain_0.mean()),
            "fold_1_gain": float(gain_1.mean()),
        })
    selected = max(audit, key=lambda row: row["gain"])
    if selected["gain_lcb95"] <= args.minimum_crossfit_gain:
        raise RuntimeError(
            "rank-one household size gain did not pass its cross-fit gate")
    kappa = solve_households(
        base_log_probability, observed, household,
        np.arange(len(population)), n_household, selected["ridge"])
    kappa, upper_bound = cap_households(
        base_log_probability, household, kappa, n_household,
        args.screen_tail_cap)
    final_gain = gain(
        base_log_probability, observed, household,
        np.arange(len(population)), kappa)
    tilted = normalized_tilt(base_log_probability, kappa, household)
    probability = np.exp(tilted)
    size = np.arange(1, int(meta["nmax"]) + 1, dtype=np.float64)
    tail = probability[:, 59:].sum(1)
    expected = probability @ size

    # theta_c subtracts the unweighted household mean. Transfer the removed global
    # utility shift into rho_0 so the implemented law receives exactly kappa_h.
    mean_kappa = float(kappa.mean())
    with torch.no_grad():
        model.theta[:, -1].add_(torch.as_tensor(
            kappa, dtype=model.theta.dtype, device=model.theta.device))
        model.project_context_gauges()
        model.rho_0_free.sub_(mean_kappa * torch.arange(
            1, model.nmax + 1, dtype=model.rho_0_free.dtype,
            device=model.rho_0_free.device))
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    result_blob = dict(blob)
    result_blob["model"] = model.state_dict()
    result_blob["household_size_rank1"] = {
        "selected_ridge": selected["ridge"],
        "screen_tail_cap": args.screen_tail_cap,
        "mean_gauge_transfer": mean_kappa,
    }
    temporary = Path(str(output) + ".tmp")
    torch.save(result_blob, temporary)
    temporary.replace(output)

    population_output = (
        args.population_output if args.population_output.is_absolute()
        else ROOT / args.population_output)
    cache_prefix = seed_population_cache(
        output, population_output, population, args.rank, levels,
        tilted, observed)
    result = {
        "method": "identified_rank_one_household_common_utility",
        "parent": str(checkpoint),
        "output": str(output),
        "contexts": int(len(population)),
        "households": n_household,
        "crossfit_ridge_audit": audit,
        "selected_ridge": selected["ridge"],
        "selected_crossfit_gain": selected["gain"],
        "selected_crossfit_gain_se": selected["gain_se"],
        "selected_crossfit_gain_lcb95": selected["gain_lcb95"],
        "full_fit_gain": float(final_gain.mean()),
        "full_fit_gain_se": float(
            final_gain.std(ddof=1) / np.sqrt(len(final_gain))),
        "screen_tail_cap": args.screen_tail_cap,
        "capped_households": int(np.isfinite(upper_bound).sum()),
        "kappa_quantiles": np.quantile(
            kappa, [0, .01, .1, .5, .9, .99, 1]).tolist(),
        "expected_size_mean": float(expected.mean()),
        "tail_rate_ge_60": float(tail.mean()),
        "maximum_screen_tail_ge_60": float(tail.max()),
        "contexts_tail_probability_ge_half": int((tail >= .5).sum()),
        "contexts_expected_size_ge_40": int((expected >= 40).sum()),
        "base_screen": provenance,
        "final_population_cache_prefix": cache_prefix,
        "interpretation": (
            "The common household shift is an existing b_jv direction. It changes only "
            "the size marginal; fixed-size composition and the sampling recursion are "
            "unchanged. Confirmation quadrature remains mandatory."),
    }
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
