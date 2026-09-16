#!/usr/bin/env python3
"""Fail-closed full-population basket-size and Smolyak-tail audit."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("V3_AFFINITY", "1")

from checkpoint_io import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from pipeline_support import (TAIL_THRESHOLD_SELECTION, collect_size_law,
                              install_quadrature, size_tail_threshold, smolyak_rule,
                              supported_trips)
from provenance import file_sha256, strict_json_dumps
from uncertainty import household_cluster_se


torch.set_default_dtype(torch.float64)


def metrics(log_probability: np.ndarray, observed: np.ndarray,
            tail_threshold: int, low_observed_threshold: int) -> dict:
    probability = np.exp(log_probability)
    size = np.arange(1, probability.shape[1] + 1, dtype=np.float64)
    mean = probability @ size
    tail = probability[:, tail_threshold - 1:].sum(1)
    observed_tail = np.asarray(observed) >= tail_threshold
    low_observed = np.asarray(observed) < low_observed_threshold
    return {
        "contexts": int(len(observed)),
        "observed_mean": float(np.mean(observed)),
        "model_mean": float(np.mean(mean)),
        "tail_threshold": int(tail_threshold),
        "low_observed_threshold": int(low_observed_threshold),
        "observed_tail_rate": float(np.mean(observed_tail)),
        "model_tail_rate": float(np.mean(tail)),
        "contexts_expected_size_ge_low_threshold": int(
            np.sum(mean >= low_observed_threshold)),
        "contexts_expected_size_ge_tail_threshold": int(
            np.sum(mean >= tail_threshold)),
        "contexts_tail_probability_ge_half": int(np.sum(tail >= 0.5)),
        "maximum_conditional_mean": float(np.max(mean)),
        "maximum_tail_probability": float(np.max(tail)),
        "maximum_tail_probability_when_observed_is_low": float(
            np.max(tail[low_observed]) if np.any(low_observed) else 0.0),
    }


@torch.no_grad()
def one_size_panel(model, batcher, trips, quadrature):
    """Evaluate one panel, raising when the requested signed rule is invalid."""
    install_quadrature(model, quadrature)
    ix, ctx, _line_ctx, house, _li, lt, _lc, _lq = batcher.make(trips)
    model.house, model.ctx = house, ctx
    _logz, probability = model.log_Z(ix, drop_empty=True, return_size=True)
    value = np.log(np.clip(probability.cpu().numpy(), 1e-300, None))
    value -= np.logaddexp.reduce(value, axis=1)[:, None]
    observed = torch.bincount(lt, minlength=ix.B).cpu().numpy()
    return observed, value


def resilient_size_panel(model, batcher, trips, rules, levels):
    """Use the cheap rule when valid; bisect failures and escalate only those trips."""
    try:
        observed, value = one_size_panel(model, batcher, trips, rules[0])
        return observed, value, np.full(len(trips), levels[0], dtype=np.int16)
    except FloatingPointError:
        if len(trips) > 1:
            middle = len(trips) // 2
            left = resilient_size_panel(
                model, batcher, trips[:middle], rules, levels)
            right = resilient_size_panel(
                model, batcher, trips[middle:], rules, levels)
            return tuple(np.concatenate((left[i], right[i])) for i in range(3))
        last_error = None
        for quadrature, level in zip(rules[1:], levels[1:]):
            try:
                observed, value = one_size_panel(
                    model, batcher, trips, quadrature)
                return observed, value, np.full(1, level, dtype=np.int16)
            except FloatingPointError as error:
                last_error = error
        raise FloatingPointError(
            f"all screen escalation levels failed for trip {int(trips[0])}: "
            f"{last_error}") from last_error


def collect_resilient_size_law(model, batcher, trips, rules, levels,
                               chunk: int, label: str):
    """Evaluate a panel while escalating only contexts with invalid signed masses."""
    if len(rules) != len(levels) or not rules:
        raise ValueError("one numerical level is required for every quadrature rule")
    observed, log_probability, used_level = [], [], []
    for start in range(0, len(trips), chunk):
        sub = trips[start:start + chunk]
        got_observed, got_probability, got_level = resilient_size_panel(
            model, batcher, sub, rules, levels)
        observed.append(got_observed)
        log_probability.append(got_probability)
        used_level.append(got_level)
        if ((start // chunk + 1) % 20 == 0 or start + chunk >= len(trips)):
            print(f"[size-law] {label} {min(start + chunk, len(trips))}/"
                  f"{len(trips)}", flush=True)
    return (np.concatenate(observed), np.concatenate(log_probability),
            np.concatenate(used_level))


def level_usage(levels: np.ndarray) -> dict[str, int]:
    unique, counts = np.unique(levels, return_counts=True)
    return {str(int(level)): int(count)
            for level, count in zip(unique, counts)}


def implementation_signature():
    """Bind resumable numerical caches to the complete Version-4 implementation."""
    digest = hashlib.sha256()
    source_root = Path(__file__).resolve().parent
    for path in sorted(source_root.glob("*.py")) + sorted(source_root.glob("*.cpp")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def screen_signature(checkpoint, population, rank, levels, data_fingerprint):
    digest = hashlib.sha256()
    with checkpoint.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    digest.update(np.ascontiguousarray(population, dtype=np.int64).tobytes())
    digest.update(f"rank={rank};levels={levels}".encode())
    digest.update(str(data_fingerprint).encode())
    digest.update(implementation_signature().encode())
    return digest.hexdigest()


def resumable_screen(model, batcher, population, checkpoint, rank, levels,
                     chunk, output, label, data_fingerprint):
    """Checkpoint the population panel and resume after interruption or cancellation."""
    signature = screen_signature(
        checkpoint, population, rank, levels, data_fingerprint)
    prefix = output.with_name(f"{output.stem}.screen-{signature[:12]}")
    probability_path = Path(str(prefix) + ".log_probability.npy")
    observed_path = Path(str(prefix) + ".observed.npy")
    level_path = Path(str(prefix) + ".level.npy")
    progress_path = Path(str(prefix) + ".progress.json")
    shape = (len(population), int(model.nmax))
    expected = {
        "signature": signature, "contexts": len(population),
        "rank": rank, "levels": levels,
    }
    start = 0
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        cache_exists = all(path.exists() for path in (
            probability_path, observed_path, level_path))
        if (cache_exists
                and all(progress.get(key) == value
                        for key, value in expected.items())):
            candidate = int(progress.get("completed_contexts", 0))
            if 0 <= candidate <= len(population):
                start = candidate
    mode = "r+" if start > 0 else "w+"
    probability = np.lib.format.open_memmap(
        probability_path, mode=mode, dtype=np.float64, shape=shape)
    observed = np.lib.format.open_memmap(
        observed_path, mode=mode, dtype=np.int16, shape=(len(population),))
    used_level = np.lib.format.open_memmap(
        level_path, mode=mode, dtype=np.int16, shape=(len(population),))
    rules = [smolyak_rule(model, rank, level) for level in levels]
    for position in range(start, len(population), chunk):
        stop = min(position + chunk, len(population))
        got_observed, got_probability, got_level = resilient_size_panel(
            model, batcher, population[position:stop], rules, levels)
        observed[position:stop] = got_observed
        probability[position:stop] = got_probability
        used_level[position:stop] = got_level
        if ((position // chunk + 1) % 20 == 0 or stop == len(population)):
            probability.flush(); observed.flush(); used_level.flush()
            progress = {**expected, "completed_contexts": stop,
                        "complete": stop == len(population)}
            temporary = Path(str(progress_path) + ".tmp")
            temporary.write_text(strict_json_dumps(progress))
            temporary.replace(progress_path)
            print(f"[population-screen] {label} {stop}/{len(population)}",
                  flush=True)
    levels_used, counts = np.unique(np.asarray(used_level), return_counts=True)
    escalated = np.flatnonzero(np.asarray(used_level) > levels[0])
    provenance = {
        "signature": signature,
        "cache_prefix": str(prefix),
        "resumed_from_context": start,
        "level_counts": {str(int(level)): int(count)
                         for level, count in zip(levels_used, counts)},
        "escalated_contexts": int(len(escalated)),
        "escalated_trip_ids_first_100": population[escalated[:100]].tolist(),
        "escalated_trip_ids_truncated": bool(len(escalated) > 100),
    }
    return (np.asarray(observed), np.asarray(probability),
            np.asarray(used_level), provenance)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"),
                        default="train")
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--screen-level", type=int, default=8)
    parser.add_argument("--confirm-level", type=int, default=9)
    parser.add_argument("--followup-level", type=int,
                        help="higher rule for contexts failing screen/confirm mean fidelity")
    parser.add_argument("--confirm-contexts", type=int, default=96)
    parser.add_argument("--calibration-contexts", type=int, default=2048,
                        help="random contexts used to estimate q(screen)->q(confirm) tail bias")
    parser.add_argument("--calibration-seed", type=int, default=32191)
    parser.add_argument("--contexts", type=int, default=0,
                        help="screen contexts; 0 means the complete supported population")
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--maximum-low-observed-tail", type=float, default=0.5)
    parser.add_argument("--maximum-tail-rate-ratio", type=float, default=2.0)
    parser.add_argument("--tail-rate-slack", type=float, default=5e-4)
    parser.add_argument("--calibration-margin", type=float, default=5e-4,
                        help="separate absolute observed/model tail-calibration margin")
    parser.add_argument("--tail-threshold", type=int,
                        help="first basket size in the scientifically relevant upper tail")
    parser.add_argument("--low-observed-threshold", type=int,
                        help="observed size below which a large predicted tail is unsafe")
    parser.add_argument("--maximum-screen-confirm-mean-gap",
                        "--maximum-q9-q8-mean-gap",
                        dest="maximum_screen_confirm_mean_gap",
                        type=float, default=1.0,
                        help=("diagnostic maximum expected-size difference between the "
                              "screen and confirmation rules; the old q9-q8 spelling is "
                              "retained as a command-line alias"))
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/population_size.json"))
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    data = build()
    split_code = {"train": 0, "validation": 1, "test": 2}[args.split]
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() \
        else ROOT / args.checkpoint
    model, blob, meta = load_checkpoint(
        checkpoint, data,
        required_capabilities=("conditional_nonempty_incidence", "gram_interactions"))
    training_population = supported_trips(data, 0, int(meta["nmax"]))
    training_sizes = np.asarray(data["trip_nlines"])[training_population]
    observed_training_maximum = int(training_sizes.max())
    support_extension_threshold = (
        observed_training_maximum + 1
        if observed_training_maximum < int(model.nmax) else None)
    tail_threshold = size_tail_threshold(
        data, int(meta["nmax"]), args.tail_threshold)
    low_observed_threshold = args.low_observed_threshold or min(
        40, max(1, tail_threshold - 1))
    if not 1 <= low_observed_threshold < tail_threshold:
        raise ValueError("low-observed-threshold must be within 1..tail-threshold-1")
    population = supported_trips(data, split_code, int(meta["nmax"]))
    full_population_size = len(population)
    if args.contexts < 0:
        raise ValueError("contexts must be nonnegative")
    if args.contexts:
        population = population[:min(args.contexts, len(population))]
    features = Features(int(data["n_item"]), int(data["n_store"]), include_recency=False)
    batcher = Batcher(data, features, int(meta["nmax"]), include_recency=False)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    screen_levels = [args.screen_level, args.confirm_level,
                     args.confirm_level + 1]
    observed, q8, used_level, screen_provenance = resumable_screen(
        model, batcher, population, checkpoint, args.rank, screen_levels,
        args.chunk, output, f"{args.split}-q{args.screen_level}",
        blob["data_fingerprint_sha256"])
    screen = metrics(q8, observed, tail_threshold, low_observed_threshold)
    probability = np.exp(q8)
    size = np.arange(1, probability.shape[1] + 1, dtype=np.float64)
    risk = (probability[:, tail_threshold - 1:].sum(1)
            + (probability @ size) / float(model.nmax))
    count = min(args.confirm_contexts, len(population))
    chosen_index = np.argsort(risk, kind="stable")[-count:]
    confirm_trips = population[chosen_index]
    confirm_levels = [args.confirm_level, args.confirm_level + 1,
                      args.confirm_level + 2]
    confirm_rules = [smolyak_rule(model, args.rank, level)
                     for level in confirm_levels]
    confirm_observed, q9, confirm_used_level = collect_resilient_size_law(
        model, batcher, confirm_trips, confirm_rules, confirm_levels, args.chunk,
        f"tail-q{args.confirm_level}")
    confirm_output = output.with_name(output.stem + "_confirm_per_trip.npz")
    np.savez_compressed(
        confirm_output, trips=confirm_trips, observed=confirm_observed,
        screen_log_probability=q8[chosen_index],
        confirm_log_probability=q9, confirm_level=confirm_used_level)
    confirm = metrics(
        q9, confirm_observed, tail_threshold, low_observed_threshold)
    extension_screen = None
    extension_confirm = None
    if support_extension_threshold is not None:
        extension_screen = metrics(
            q8, observed, support_extension_threshold,
            observed_training_maximum)
        extension_confirm = metrics(
            q9, confirm_observed, support_extension_threshold,
            observed_training_maximum)
    q8_mean = np.exp(q8[chosen_index]) @ size
    q9_mean = np.exp(q9) @ size
    fidelity = {
        "mean_absolute_expected_size_gap": float(np.mean(np.abs(q9_mean - q8_mean))),
        "maximum_absolute_expected_size_gap": float(np.max(np.abs(q9_mean - q8_mean))),
    }
    failed_fidelity = np.flatnonzero(
        np.abs(q9_mean - q8_mean) > args.maximum_screen_confirm_mean_gap)
    followup_level = args.followup_level or args.confirm_level + 1
    followup = {
        "level": followup_level, "contexts": int(len(failed_fidelity)),
        "status": "not_needed" if not len(failed_fidelity) else "running",
    }
    if len(failed_fidelity):
        followup_trips = confirm_trips[failed_fidelity]
        followup_observed, followup_log_probability = collect_size_law(
            model, batcher, followup_trips,
            smolyak_rule(model, args.rank, followup_level), args.chunk,
            f"fidelity-q{followup_level}")
        followup_probability = np.exp(followup_log_probability)
        followup_mean = followup_probability @ size
        followup_gap = np.abs(followup_mean - q9_mean[failed_fidelity])
        followup_output = output.with_name(
            output.stem + "_fidelity_followup_per_trip.npz")
        np.savez_compressed(
            followup_output, trips=followup_trips, observed=followup_observed,
            screen_log_probability=q8[chosen_index][failed_fidelity],
            confirm_log_probability=q9[failed_fidelity],
            followup_log_probability=followup_log_probability)
        followup.update({
            "status": ("passed" if float(followup_gap.max())
                       <= args.maximum_screen_confirm_mean_gap else "failed"),
            "mean_absolute_followup_confirm_gap": float(followup_gap.mean()),
            "maximum_absolute_followup_confirm_gap": float(followup_gap.max()),
            "per_trip_output": str(followup_output),
        })
    numerical_fidelity_passed = bool(
        not len(failed_fidelity) or followup["status"] == "passed")
    q8_tail = np.exp(q8[:, tail_threshold - 1:]).sum(1)
    q9_probability = np.exp(q9)
    q9_tail = q9_probability[:, tail_threshold - 1:].sum(1)
    positive_mean_error = float(np.max(np.maximum(q9_mean - q8_mean, 0.0)))
    positive_tail_error = float(np.max(np.maximum(
        q9_tail - q8_tail[chosen_index], 0.0)))
    chosen_mask = np.zeros(len(population), dtype=bool)
    chosen_mask[chosen_index] = True
    omitted = ~chosen_mask
    omitted_low = omitted & (observed < low_observed_threshold)
    omitted_mean_upper = float(
        np.max((probability @ size)[omitted]) + positive_mean_error
        if np.any(omitted) else 0.0)
    omitted_tail_upper = float(
        np.max(q8_tail[omitted_low]) + positive_tail_error
        if np.any(omitted_low) else 0.0)
    adaptive_confirmation = {
        "policy": (
            f"q{args.screen_level} ranks every context; q{args.confirm_level} confirms "
            "the highest-risk panel; the largest positive confirm-minus-screen error "
            "on that panel is added to the largest unconfirmed screen-rule value as a "
            "conservative empirical envelope"),
        "positive_expected_size_error_envelope": positive_mean_error,
        "positive_tail_probability_error_envelope": positive_tail_error,
        "unconfirmed_expected_size_upper_envelope": omitted_mean_upper,
        "unconfirmed_low_observed_tail_upper_envelope": omitted_tail_upper,
    }

    calibration_count = min(args.calibration_contexts, len(population))
    if calibration_count < 2:
        raise ValueError("calibration-contexts must select at least two contexts")
    calibration_rng = np.random.default_rng(args.calibration_seed)
    calibration_index = calibration_rng.choice(
        len(population), size=calibration_count, replace=False)
    calibration_trips = population[calibration_index]
    calibration_observed, calibration_q9, calibration_used_level = \
        collect_resilient_size_law(
        model, batcher, calibration_trips, confirm_rules, confirm_levels, args.chunk,
        f"calibration-q{args.confirm_level}")
    calibration_q9_tail = np.exp(
        calibration_q9[:, tail_threshold - 1:]).sum(1)
    calibration_q8_tail = q8_tail[calibration_index]
    tail_difference = calibration_q9_tail - calibration_q8_tail
    tail_bias = float(tail_difference.mean())
    tail_bias_se = household_cluster_se(
        tail_difference, data["trip_user"][calibration_trips])
    corrected_tail = float(screen["model_tail_rate"] + tail_bias)
    corrected_tail_upper = float(corrected_tail + 1.96 * tail_bias_se)
    calibration_output = output.with_name(
        output.stem + "_calibration_per_trip.npz")
    np.savez_compressed(
        calibration_output, trips=calibration_trips,
        observed=calibration_observed,
        screen_tail_probability=calibration_q8_tail,
        confirm_tail_probability=calibration_q9_tail,
        confirm_level=calibration_used_level)
    tail_calibration = {
        "contexts": calibration_count,
        "seed": args.calibration_seed,
        "requested_confirm_level": args.confirm_level,
        "level_counts": level_usage(calibration_used_level),
        "escalated_contexts": int(np.sum(
            calibration_used_level > args.confirm_level)),
        "screen_tail_rate": float(calibration_q8_tail.mean()),
        "confirm_tail_rate": float(calibration_q9_tail.mean()),
        "confirm_minus_screen_tail_bias": tail_bias,
        "bias_standard_error": tail_bias_se,
        "full_screen_bias_corrected_tail_rate": corrected_tail,
        "full_screen_bias_corrected_tail_rate_95_upper": corrected_tail_upper,
        "per_trip_output": str(calibration_output),
    }
    calibration_error = corrected_tail - screen["observed_tail_rate"]
    screen_calibration_score = q8_tail - (
        observed >= tail_threshold).astype(np.float64)
    screen_calibration_se = household_cluster_se(
        screen_calibration_score, data["trip_user"][population])
    # Do not assume independence between the full-panel screen error and the random
    # quadrature-bias panel.  Adding their 95% radii is conservative under dependence.
    calibration_radius = 1.96 * (screen_calibration_se + tail_bias_se)
    calibration_interval = [calibration_error-calibration_radius,
                            calibration_error+calibration_radius]
    if abs(calibration_error) > args.calibration_margin:
        calibration_status = "failed"
    elif (calibration_interval[0] >= -args.calibration_margin
          and calibration_interval[1] <= args.calibration_margin):
        calibration_status = "passed"
    else:
        calibration_status = "inconclusive"
    separate_calibration = {
        "status": calibration_status,
        "model_minus_observed_tail_rate": calibration_error,
        "screen_household_cluster_standard_error": screen_calibration_se,
        "quadrature_bias_household_cluster_standard_error": tail_bias_se,
        "conservative_95_interval": calibration_interval,
        "absolute_margin": args.calibration_margin,
        "interpretation": ("descriptive split-population calibration with estimated "
                           "household-cluster and quadrature-bias uncertainty; not "
                           "future-population coverage"),
    }
    support_extension = {
        "status": "not_applicable",
        "observed_training_maximum": observed_training_maximum,
        "model_support_maximum": int(model.nmax),
        "interpretation": "model support does not extend beyond the observed training maximum",
    }
    if support_extension_threshold is not None:
        extension_q8_tail = np.exp(
            q8[:, support_extension_threshold - 1:]).sum(1)
        extension_calibration_q8 = extension_q8_tail[calibration_index]
        extension_calibration_q9 = np.exp(
            calibration_q9[:, support_extension_threshold - 1:]).sum(1)
        extension_difference = (
            extension_calibration_q9 - extension_calibration_q8)
        extension_bias = float(extension_difference.mean())
        extension_bias_se = household_cluster_se(
            extension_difference, data["trip_user"][calibration_trips])
        extension_corrected_rate = float(
            extension_screen["model_tail_rate"] + extension_bias)
        support_extension = {
            "status": "not_identifiable",
            "observed_training_maximum": observed_training_maximum,
            "model_support_maximum": int(model.nmax),
            "first_unobserved_size": support_extension_threshold,
            "screen": extension_screen,
            "high_risk_confirmation_for_primary_tail": extension_confirm,
            "quadrature_bias_corrected_population_rate": extension_corrected_rate,
            "quadrature_bias_standard_error": extension_bias_se,
            "quadrature_bias_corrected_rate_95_upper": float(
                extension_corrected_rate + 1.96 * extension_bias_se),
            "observed_events_in_evaluated_split": int(
                np.sum(observed >= support_extension_threshold)),
            "interpretation": (
                "No training basket identifies probabilities above the observed "
                "training maximum. Nonzero mass there is model extrapolation and is "
                "reported separately from calibration on the observed tail."),
        }
    allowed_rate = (args.maximum_tail_rate_ratio
                    * screen["observed_tail_rate"]
                    + args.tail_rate_slack)
    gates = {
        "population_tail_rate_calibrated":
            corrected_tail_upper <= allowed_rate,
        "no_low_observed_context_has_majority_extreme_tail":
            confirm["maximum_tail_probability_when_observed_is_low"]
            <= args.maximum_low_observed_tail,
        "adaptive_screen_covers_unconfirmed_extreme_tail":
            omitted_tail_upper <= args.maximum_low_observed_tail,
        "adaptive_screen_covers_unconfirmed_expected_size":
            omitted_mean_upper < float(tail_threshold),
    }
    result = {
        "checkpoint": str(checkpoint), "split": args.split,
        "checkpoint_sha256": file_sha256(checkpoint),
        "data_fingerprint_sha256": blob["data_fingerprint_sha256"],
        "trained_capabilities": blob["trained_capabilities"],
        "full_population_contexts": int(full_population_size),
        "screened_complete_population": bool(len(population) == full_population_size),
        "support": f"1..{model.nmax}", "rank": args.rank,
        "tail_threshold": tail_threshold,
        "tail_threshold_selection": (
            "explicit_command_line" if args.tail_threshold is not None else
            TAIL_THRESHOLD_SELECTION),
        "low_observed_threshold": low_observed_threshold,
        "screen_level": args.screen_level,
        "confirm_level": args.confirm_level,
        "screen_estimator": {
            "policy": ("use the requested screen level when every signed size mass is "
                       "positive; bisect invalid batches and escalate only invalid "
                       "contexts through confirm_level and confirm_level+1"),
            **screen_provenance,
        },
        "screen": screen, "high_risk_confirmation": confirm,
        "high_risk_per_trip_output": str(confirm_output),
        "high_risk_confirmation_numerics": {
            "requested_confirm_level": args.confirm_level,
            "level_counts": level_usage(confirm_used_level),
            "escalated_contexts": int(np.sum(
                confirm_used_level > args.confirm_level)),
            "policy": ("use the requested confirmation rule when all signed size "
                       "masses are nonnegative; bisect invalid batches and escalate "
                       "only invalid contexts through two higher rules"),
        },
        "quadrature_fidelity": {
            **fidelity,
            "screen_confirm_one_item_fidelity_gate": (
                fidelity["maximum_absolute_expected_size_gap"]
                <= args.maximum_screen_confirm_mean_gap),
            "interpretation": ("reported as estimator fidelity; model safety is decided "
                               f"by q{args.confirm_level} confirmation and conservative "
                               "coverage envelopes"),
        },
        "higher_rule_fidelity_followup": followup,
        "numerical_fidelity_status": (
            "passed" if numerical_fidelity_passed else "failed"),
        "tail_calibration": separate_calibration,
        "unobserved_support_extension": support_extension,
        "adaptive_high_risk_confirmation": adaptive_confirmation,
        "random_confirm_tail_calibration": tail_calibration,
        "allowed_model_tail_rate": allowed_rate,
        "gates": gates, "passed": bool(all(gates.values())),
        "safety_status": "passed" if all(gates.values()) else "failed",
        "interpretation": (
            "The low rule screens the requested population panel. A context whose "
            "signed size masses are invalid is evaluated by the next positive rule "
            "rather than treated as a model probability. The confirm rule re-evaluates "
            f"the highest-risk contexts. A random q{args.confirm_level} panel corrects "
            f"aggregate q{args.screen_level} tail bias. The screen/confirm mean gap "
            f"remains a diagnostic; safety uses confirmed q{args.confirm_level} tails "
            "and an explicit error envelope. Failure blocks production certification "
            "but preserves resumable state."),
    }
    output.write_text(strict_json_dumps(result))
    print(strict_json_dumps(result), end="")
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
