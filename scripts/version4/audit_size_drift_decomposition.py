#!/usr/bin/env python3
"""Exploratory decomposition of out-of-sample basket-size calibration drift."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch

os.environ.setdefault("V3_AFFINITY", "1")

from checkpoint_io import load_checkpoint
from data import build
from features import Features
from fit import Batcher
from pipeline_support import collect_size_law, smolyak_rule, supported_trips
from provenance import file_sha256, strict_json_dumps
from uncertainty import paired_score_summary


torch.set_default_dtype(torch.float64)


CACHE_SCHEMA = 1


def atomic_write_text(path: Path, value: str) -> None:
    """Replace a text artifact only after its complete payload is on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def atomic_savez(path: Path, **arrays) -> None:
    """Atomically persist a compressed NumPy archive."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def save_stage_cache(path: Path, trips: np.ndarray, expected: np.ndarray,
                     diagnostic: dict, signature: dict) -> None:
    atomic_savez(
        path,
        schema=np.asarray(CACHE_SCHEMA, dtype=np.int64),
        trips=np.asarray(trips, dtype=np.int64),
        expected=np.asarray(expected, dtype=np.float64),
        diagnostic_json=np.asarray(strict_json_dumps(diagnostic)),
        signature_json=np.asarray(strict_json_dumps(signature)),
    )


def load_stage_cache(path: Path, trips: np.ndarray,
                     signature: dict) -> tuple[np.ndarray, dict] | None:
    """Return a stage only when its inputs and protocol match exactly."""
    if not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as payload:
            if int(payload["schema"]) != CACHE_SCHEMA:
                return None
            cached_trips = np.asarray(payload["trips"], dtype=np.int64)
            if not np.array_equal(cached_trips, np.asarray(trips, dtype=np.int64)):
                return None
            if json.loads(str(payload["signature_json"].item())) != signature:
                return None
            expected = np.asarray(payload["expected"], dtype=np.float64)
            diagnostic = json.loads(str(payload["diagnostic_json"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None
    if expected.shape != (len(trips),) or not np.isfinite(expected).all():
        return None
    return expected, diagnostic


def quadrature_chunk(quadrature, requested: int, node_context_cap: int) -> int:
    """Bound peak quadrature work while preserving the requested rule."""
    nodes = max(1, len(quadrature[1]))
    return max(1, min(int(requested), int(node_context_cap) // nodes))


def household_balanced_panel(data, population: np.ndarray, count: int,
                             seed: int) -> np.ndarray:
    """Round-robin households after deterministic within-household shuffling."""
    grouped: dict[int, list[int]] = {}
    for trip in np.asarray(population, dtype=np.int64):
        grouped.setdefault(int(data["trip_user"][trip]), []).append(int(trip))
    rng = np.random.default_rng(seed)
    households = np.asarray(sorted(grouped), dtype=np.int64)
    households = households[rng.permutation(len(households))]
    for household in households:
        values = np.asarray(grouped[int(household)], dtype=np.int64)
        grouped[int(household)] = values[rng.permutation(len(values))].tolist()
    target = min(int(count), len(population))
    selected: list[int] = []
    offset = 0
    while len(selected) < target:
        added = False
        for household in households:
            values = grouped[int(household)]
            if offset < len(values):
                selected.append(values[offset]); added = True
                if len(selected) == target:
                    break
        if not added:
            break
        offset += 1
    if len(selected) != target:
        raise RuntimeError("household-balanced panel construction was incomplete")
    return np.asarray(selected, dtype=np.int64)


def trip_panel(population: np.ndarray, count: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    count = min(int(count), len(population))
    return np.asarray(population, dtype=np.int64)[
        rng.choice(len(population), size=count, replace=False)]


def active_rank(model) -> int:
    singular = torch.linalg.svdvals(model.phi.detach())
    if not len(singular) or float(singular[0]) == 0:
        return 0
    return int((singular > singular[0] * 1e-10).sum())


@torch.no_grad()
def expected_size(model, batcher, trips: np.ndarray, rank: int, chunk: int,
                  tolerance: float, label: str,
                  node_context_cap: int = 16_000) -> tuple[np.ndarray, dict]:
    axis = np.arange(1, model.nmax + 1, dtype=np.float64)
    if rank == 0:
        rule = smolyak_rule(model, 0, 0)
        _observed, log_probability = collect_size_law(
            model, batcher, trips, rule,
            quadrature_chunk(rule, chunk, node_context_cap), label + "-exact")
        return np.exp(log_probability) @ axis, {
            "status": "passed", "rank": 0, "primary_rule": "exact",
            "maximum_adjacent_rule_gap": 0.0, "followup_contexts": 0,
            "maximum_followup_gap": 0.0, "rule_nodes": {"exact": 1},
            "effective_chunks": {"exact": quadrature_chunk(
                rule, chunk, node_context_cap)}}
    low_rule = smolyak_rule(model, rank, rank + 1)
    high_rule = smolyak_rule(model, rank, rank + 2)
    effective_chunks = {
        f"q{rank + 1}": quadrature_chunk(low_rule, chunk, node_context_cap),
        f"q{rank + 2}": quadrature_chunk(high_rule, chunk, node_context_cap),
    }
    rule_nodes = {f"q{rank + 1}": len(low_rule[1]),
                  f"q{rank + 2}": len(high_rule[1])}
    _observed, low = collect_size_law(
        model, batcher, trips, low_rule, effective_chunks[f"q{rank + 1}"],
        label + f"-q{rank + 1}")
    _observed, high = collect_size_law(
        model, batcher, trips, high_rule, effective_chunks[f"q{rank + 2}"],
        label + f"-q{rank + 2}")
    low_mean = np.exp(low) @ axis
    high_mean = np.exp(high) @ axis
    gap = np.abs(high_mean - low_mean)
    failures = np.flatnonzero(gap > tolerance)
    reference = high_mean.copy()
    followup_gap = np.empty(0, dtype=np.float64)
    final_gap = np.empty(0, dtype=np.float64)
    unresolved = np.empty(0, dtype=np.int64)
    if len(failures):
        followup_rule = smolyak_rule(model, rank, rank + 3)
        followup_chunk = quadrature_chunk(
            followup_rule, chunk, node_context_cap)
        rule_nodes[f"q{rank + 3}"] = len(followup_rule[1])
        effective_chunks[f"q{rank + 3}"] = followup_chunk
        _observed, followup = collect_size_law(
            model, batcher, trips[failures], followup_rule, followup_chunk,
            label + f"-q{rank + 3}-followup")
        followup_mean = np.exp(followup) @ axis
        followup_gap = np.abs(followup_mean - high_mean[failures])
        reference[failures] = followup_mean
        unresolved_local = np.flatnonzero(followup_gap > tolerance)
        unresolved = failures[unresolved_local]
        if len(unresolved):
            final_rule = smolyak_rule(model, rank, rank + 4)
            final_chunk = quadrature_chunk(final_rule, chunk, node_context_cap)
            rule_nodes[f"q{rank + 4}"] = len(final_rule[1])
            effective_chunks[f"q{rank + 4}"] = final_chunk
            _observed, final = collect_size_law(
                model, batcher, trips[unresolved], final_rule, final_chunk,
                label + f"-q{rank + 4}-final")
            final_mean = np.exp(final) @ axis
            final_gap = np.abs(final_mean - followup_mean[unresolved_local])
            reference[unresolved] = final_mean
    passed = bool(not len(unresolved) or final_gap.max() <= tolerance)
    return reference, {
        "status": "passed" if passed else "inconclusive", "rank": rank,
        "primary_rule": f"q{rank + 2}", "tolerance_items": tolerance,
        "maximum_adjacent_rule_gap": float(gap.max()),
        "followup_contexts": int(len(failures)),
        "maximum_followup_gap": float(followup_gap.max()) if len(followup_gap) else 0.0,
        "final_followup_contexts": int(len(unresolved)),
        "maximum_final_followup_gap": float(final_gap.max()) if len(final_gap) else 0.0,
        "rule_nodes": rule_nodes, "effective_chunks": effective_chunks}


def score_summary(values: np.ndarray, households: np.ndarray) -> dict:
    return paired_score_summary(np.asarray(values, dtype=np.float64), households)


def panel_report(data, trips: np.ndarray, expected: dict[str, np.ndarray]) -> dict:
    observed = np.asarray(data["trip_nlines"][trips], dtype=np.float64)
    household = np.asarray(data["trip_user"][trips], dtype=np.int64)
    means = {"observed": float(observed.mean())}
    means.update({name: float(value.mean()) for name, value in expected.items()})
    return {
        "contexts": int(len(trips)),
        "households": int(np.unique(household).size),
        "means": means,
        "residuals": {name + "_minus_observed": score_summary(value - observed, household)
                      for name, value in expected.items()},
        "stage_increments": {
            "interaction_minus_additive": score_summary(
                expected["interaction"] - expected["additive"], household),
            "household_correction_minus_interaction": score_summary(
                expected["final"] - expected["interaction"], household),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=68101)
    parser.add_argument("--quadrature-tolerance", type=float, default=.02)
    parser.add_argument("--chunk", type=int, default=48)
    parser.add_argument("--node-context-cap", type=int, default=16_000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-context-output", type=Path, required=True)
    parser.add_argument("--resume-dir", type=Path)
    args = parser.parse_args(); started = time.monotonic()
    if args.chunk < 1 or args.node_context_cap < 1:
        parser.error("--chunk and --node-context-cap must be positive")
    torch.set_num_threads(args.threads)
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    state_output = output.parent / "run_state.json"
    resume_dir = ((args.resume_dir.resolve() if args.resume_dir else
                   output.parent / "drift_checkpoints"))
    resume_dir.mkdir(parents=True, exist_ok=True)
    completed_stages: list[str] = []
    active_stage: str | None = "initialization"

    def write_state(status: str, error: dict | None = None) -> None:
        payload = {
            "status": status, "active_stage": active_stage,
            "completed_stages": completed_stages,
            "updated_unix_seconds": time.time(),
            "runtime_seconds": time.monotonic() - started,
            "resume_dir": str(resume_dir),
        }
        if error is not None:
            payload["error"] = error
        atomic_write_text(state_output, strict_json_dumps(payload))

    write_state("running")
    default_hook = sys.excepthook

    def failure_hook(kind, value, traceback) -> None:
        write_state("failed", {"type": kind.__name__, "message": str(value)})
        default_hook(kind, value, traceback)

    sys.excepthook = failure_hook
    run = args.run_dir.resolve(); data = build()
    paths = {
        "additive": run / "out/v3_pipeline_additive_best.pt",
        "interaction": run / "artifacts/candidate.pt",
        "final": run / "artifacts/candidate_rank1.pt",
    }
    models = {}; blobs = {}; metadata = {}; checkpoint_hashes = {}
    for name, path in paths.items():
        model, blob, meta = load_checkpoint(path, data)
        for parameter in model.parameters(): parameter.requires_grad_(False)
        models[name], blobs[name], metadata[name] = model, blob, meta
        checkpoint_hashes[name] = file_sha256(path)
    supports = {int(meta["nmax"]) for meta in metadata.values()}
    if len(supports) != 1:
        raise RuntimeError("checkpoint size supports differ")
    nmax = supports.pop()
    batcher = Batcher(data, Features(int(data["n_item"]), int(data["n_store"]),
                                     include_recency=False), nmax, include_recency=False)
    split_codes = {"train": 0, "validation": 1, "test": 2}
    panels = {}; numerical = {}; arrays = {}
    full_observed = {}
    for split_index, (split_name, split_code) in enumerate(split_codes.items()):
        population = supported_trips(data, split_code, nmax)
        full_observed[split_name] = {
            "contexts": int(len(population)),
            "trip_weighted_mean": float(data["trip_nlines"][population].mean()),
            "households": int(np.unique(data["trip_user"][population]).size),
        }
        chosen = {
            "trip_weighted": trip_panel(
                population, args.contexts, args.seed + 1009 * split_index),
            "household_balanced": household_balanced_panel(
                data, population, args.contexts, args.seed + 1009 * split_index + 1),
        }
        union = np.unique(np.concatenate(list(chosen.values())))
        arrays[split_name + "_union_trips"] = union
        arrays[split_name + "_observed"] = data["trip_nlines"][union]
        expected = {}
        numerical[split_name] = {}
        for model_name, model in models.items():
            active_stage = split_name + "-" + model_name
            write_state("running")
            signature = {
                "cache_schema": CACHE_SCHEMA, "split": split_name,
                "model": model_name,
                "checkpoint_sha256": checkpoint_hashes[model_name],
                "data_fingerprint_sha256": blobs["final"]["data_fingerprint_sha256"],
                "active_rank": active_rank(model), "nmax": nmax,
                "quadrature_tolerance": float(args.quadrature_tolerance),
                "node_context_cap": int(args.node_context_cap),
            }
            cache_path = resume_dir / (active_stage + ".npz")
            cached = load_stage_cache(cache_path, union, signature)
            if cached is None:
                value, diagnostic = expected_size(
                    model, batcher, union, active_rank(model), args.chunk,
                    args.quadrature_tolerance, active_stage,
                    args.node_context_cap)
                save_stage_cache(cache_path, union, value, diagnostic, signature)
                print(f"[checkpoint] saved {active_stage} -> {cache_path}", flush=True)
            else:
                value, diagnostic = cached
                print(f"[checkpoint] resumed {active_stage} <- {cache_path}", flush=True)
            expected[model_name] = value
            numerical[split_name][model_name] = diagnostic
            arrays[split_name + "_" + model_name + "_expected"] = value
            completed_stages.append(active_stage)
            write_state("running")
        panels[split_name] = {}
        for panel_name, trips in chosen.items():
            index = np.searchsorted(union, trips)
            panels[split_name][panel_name] = panel_report(
                data, trips, {name: value[index] for name, value in expected.items()})
            arrays[split_name + "_" + panel_name + "_trips"] = trips

    validation_trips = arrays["validation_household_balanced_trips"]
    validation_union = arrays["validation_union_trips"]
    validation_index = np.searchsorted(validation_union, validation_trips)
    validation_residual = (arrays["validation_final_expected"][validation_index]
                           - data["trip_nlines"][validation_trips])
    training = supported_trips(data, 0, nmax)
    training_count = np.bincount(data["trip_user"][training],
                                 minlength=int(data["n_user"]))
    household = data["trip_user"][validation_trips]
    boundaries = np.quantile(training_count[household], [0, .25, .5, .75, 1])
    activity = []
    for index, (low, high) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        selected = ((training_count[household] >= low)
                    & ((training_count[household] <= high) if index == 3
                       else (training_count[household] < high)))
        activity.append({
            "training_trip_count_range": [float(low), float(high)],
            "contexts": int(selected.sum()),
            "mean_training_trip_count": float(training_count[household][selected].mean()),
            "final_minus_observed": score_summary(
                validation_residual[selected], household[selected]),
        })

    per_context = args.per_context_output.resolve()
    per_context.parent.mkdir(parents=True, exist_ok=True)
    atomic_savez(per_context, **arrays)
    report = {
        "status": "completed", "claim_level": "post_hoc_exploratory_diagnostic",
        "implementation_sha256": file_sha256(Path(__file__).resolve()),
        "run_dir": str(run), "data_fingerprint_sha256":
            blobs["final"]["data_fingerprint_sha256"],
        "checkpoints": {name: {"path": str(path), "sha256": checkpoint_hashes[name],
                               "active_rank": active_rank(models[name])}
                        for name, path in paths.items()},
        "panel_contexts_requested": int(args.contexts), "seed": int(args.seed),
        "node_context_cap": int(args.node_context_cap),
        "full_observed_populations": full_observed,
        "panels": panels, "numerical_fidelity": numerical,
        "validation_household_activity": activity,
        "per_context_output": str(per_context),
        "per_context_sha256": file_sha256(per_context),
        "limitations": [
            "post-hoc diagnosis after validation and test inspection",
            "panel comparisons diagnose transport and weighting; they do not identify a causal drift mechanism",
            "basket incidence conditions on an observed nonempty shopping trip",
        ],
        "runtime_seconds": time.monotonic() - started,
    }
    atomic_write_text(output, strict_json_dumps(report))
    active_stage = None
    write_state("completed")
    print(strict_json_dumps(report), end="")


if __name__ == "__main__":
    main()
