#!/usr/bin/env python3
"""Separate fixed-law sampler correctness from real-basket factual calibration."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

os.environ.setdefault("V3_AFFINITY", "1")

from checkpoint_io import load_checkpoint
from data import build
from features import Features
from fit import Batcher
from interaction_particles import (direct_interaction_particles,
                                   weighted_particle_expected_size)
from pipeline_support import collect_size_law, smolyak_rule, supported_trips
from provenance import file_sha256, strict_json_dumps
from uncertainty import paired_score_summary


torch.set_default_dtype(torch.float64)


def household_balanced_panel(data, population, count, seed):
    grouped = {}
    for trip in population:
        grouped.setdefault(int(data["trip_user"][trip]), []).append(int(trip))
    rng = np.random.default_rng(seed)
    households = np.asarray(sorted(grouped), dtype=np.int64)
    households = households[rng.permutation(len(households))]
    for household in households:
        values = np.asarray(grouped[int(household)], dtype=np.int64)
        grouped[int(household)] = values[rng.permutation(len(values))].tolist()
    selected, offset = [], 0
    while len(selected) < min(count, len(population)):
        changed = False
        for household in households:
            values = grouped[int(household)]
            if offset < len(values):
                selected.append(values[offset]); changed = True
                if len(selected) == min(count, len(population)):
                    break
        if not changed:
            break
        offset += 1
    return np.asarray(selected, dtype=np.int64)


def tv(left, right):
    left, right = np.asarray(left, float), np.asarray(right, float)
    left = left / max(left.sum(), 1); right = right / max(right.sum(), 1)
    return float(.5 * np.abs(left - right).sum())


@torch.no_grad()
def sample_replicate(model, batcher, trips, particles, seed, chunk, item_category):
    sizes = np.empty(len(trips), dtype=np.int16)
    particle_expected_size = np.empty(len(trips), dtype=np.float64)
    item_count = np.zeros(model.J); category_count = np.zeros(model.C)
    ess = np.empty(len(trips)); flat_items, ptr = [], [0]
    for start in range(0, len(trips), chunk):
        sub = trips[start:start + chunk]
        ix, ctx, _lc, house, *_ = batcher.make(sub)
        model.house, model.ctx = house, ctx
        result = direct_interaction_particles(
            model, ix, particles,
            torch.Generator().manual_seed(seed + start))
        particle_expected_size[start:start + len(sub)] = (
            weighted_particle_expected_size(result.states, result.log_weights)
            .cpu().numpy())
        generator = torch.Generator().manual_seed(seed + 900001 + start)
        for b in range(ix.B):
            chosen = int(torch.multinomial(
                result.log_weights[:, b].exp(), 1, generator=generator))
            items = ix.item[result.states[chosen][b]].cpu().numpy().astype(np.int64)
            where = start + b
            sizes[where] = len(items); flat_items.extend(items.tolist()); ptr.append(len(flat_items))
            np.add.at(item_count, items, 1)
            np.add.at(category_count, item_category[items], 1)
        ess[start:start + len(sub)] = result.ess_fraction.cpu().numpy()
        print(f"[generation-calibration] seed={seed} contexts="
              f"{min(start + chunk, len(trips))}/{len(trips)}", flush=True)
    return (sizes, particle_expected_size, item_count, category_count, ess,
            np.asarray(flat_items), np.asarray(ptr))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint", type=Path,
        help=("explicit final checkpoint; when omitted, retain the historical "
              "RUN_DIR/artifacts/candidate_rank1.pt layout"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-context-output", type=Path, required=True)
    parser.add_argument("--samples-output", type=Path, required=True)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=32)
    args = parser.parse_args()
    started = time.monotonic(); torch.set_num_threads(args.threads)
    protocol_path = args.protocol.resolve(); protocol = json.loads(protocol_path.read_text())
    spec = protocol["generation"]; run_dir = args.run_dir.resolve()
    checkpoint = (args.checkpoint.resolve() if args.checkpoint is not None
                  else run_dir / "artifacts/candidate_rank1.pt")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"final checkpoint not found: {checkpoint}")
    data = build(); model, blob, meta = load_checkpoint(
        checkpoint, data,
        required_capabilities=("conditional_nonempty_incidence", "gram_interactions"))
    for parameter in model.parameters(): parameter.requires_grad_(False)
    population = supported_trips(data, 1, int(meta["nmax"]))
    trips = household_balanced_panel(
        data, population, int(spec["contexts"]), int(spec["seed"]))
    if len(trips) != min(int(spec["contexts"]), len(population)):
        raise RuntimeError("failed to construct the frozen generation panel")
    batcher = Batcher(data, Features(int(data["n_item"]), int(data["n_store"]),
                                     include_recency=False), int(meta["nmax"]),
                      include_recency=False)
    singular = torch.linalg.svdvals(model.phi); rank = int((singular > singular[0]*1e-10).sum())
    lower_level, reference_level, followup_level = rank + 2, rank + 3, rank + 4
    observed, lower_log_probability = collect_size_law(
        model, batcher, trips, smolyak_rule(model, rank, lower_level),
        args.chunk, f"generation-q{lower_level}")
    _, reference_log_probability = collect_size_law(
        model, batcher, trips, smolyak_rule(model, rank, reference_level),
        args.chunk, f"generation-q{reference_level}")
    axis = np.arange(1, model.nmax + 1)
    lower_mean = np.exp(lower_log_probability) @ axis
    reference_rule_mean = np.exp(reference_log_probability) @ axis
    reference_tolerance = float(spec["model_sampling_absolute_items"])
    initial_reference_gap = np.abs(reference_rule_mean - lower_mean)
    reference_mean = reference_rule_mean.copy()
    reference_rule = np.full(len(trips), reference_level, dtype=np.int16)
    reference_failures = np.flatnonzero(initial_reference_gap > reference_tolerance)
    q9_followup_gap = np.empty(0, dtype=np.float64)
    if len(reference_failures):
        _, followup_log_probability = collect_size_law(
            model, batcher, trips[reference_failures],
            smolyak_rule(model, rank, followup_level), args.chunk,
            f"generation-q{followup_level}-followup")
        followup_mean = np.exp(followup_log_probability) @ axis
        q9_followup_gap = np.abs(
            followup_mean - reference_rule_mean[reference_failures])
        reference_mean[reference_failures] = followup_mean
        reference_rule[reference_failures] = followup_level
    reference_fidelity_pass = bool(
        not len(reference_failures) or np.max(q9_followup_gap) <= reference_tolerance)
    item_category = model.cat_of.detach().cpu().numpy().astype(np.int64)
    observed_item = np.zeros(model.J); observed_category = np.zeros(model.C)
    observed_item_by_context = np.zeros((len(trips), model.J), dtype=np.int8)
    observed_category_by_context = np.zeros((len(trips), model.C), dtype=np.int8)
    for context_index, trip in enumerate(trips):
        lo, hi = int(data["line_ptr"][trip]), int(data["line_ptr"][trip + 1])
        items = np.unique(data["line_item"][lo:hi])
        np.add.at(observed_item, items, 1); np.add.at(observed_category, item_category[items], 1)
        observed_item_by_context[context_index, items] = 1
        np.add.at(observed_category_by_context[context_index], item_category[items], 1)
    seeds = [int(spec["seed"]) + 1009 * (i + 1) for i in range(args.replicates)]
    generated, particle_expected, ess, pooled_item, pooled_category = [], [], [], [], []
    sample_items, sample_ptr = [], []
    for seed in seeds:
        size, expected_size, item, category, overlap, flat, ptr = sample_replicate(
            model, batcher, trips, int(spec["particles"]), seed, args.chunk, item_category)
        generated.append(size); particle_expected.append(expected_size)
        ess.append(overlap); pooled_item.append(item); pooled_category.append(category)
        sample_items.append(flat); sample_ptr.append(ptr)
    generated = np.asarray(generated); particle_expected = np.asarray(particle_expected)
    ess = np.asarray(ess)
    replicate_mean = particle_expected.mean(1)
    difference = replicate_mean - reference_mean.mean()
    mc_se = float(replicate_mean.std(ddof=1) / math.sqrt(args.replicates))
    margin = float(spec["observed_mean_equivalence_margin_items"])
    mc_half_width = float(spec["model_sampling_mc_se_multiplier"]) * mc_se
    sampler_tolerance = float(spec["model_sampling_absolute_items"])
    error = float(abs(difference.mean()))
    if not reference_fidelity_pass:
        sampler_status = "inconclusive"
    elif ess.min() < .2:
        sampler_status = "failed"
    elif error + mc_half_width <= sampler_tolerance:
        sampler_status = "passed"
    elif max(0.0, error - mc_half_width) > sampler_tolerance:
        sampler_status = "failed"
    else:
        sampler_status = "inconclusive"
    calibration = paired_score_summary(
        reference_mean - observed, data["trip_user"][trips])
    factual_pass = bool(calibration["95_interval"][0] >= -margin
                        and calibration["95_interval"][1] <= margin)
    factual_status = ("not_assessed" if not reference_fidelity_pass else
                      "passed" if factual_pass else "failed")
    per_path = args.per_context_output.resolve(); per_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(per_path, trips=trips, household=data["trip_user"][trips],
                        observed_size=observed,
                        lower_rule_expected_size=lower_mean,
                        reference_rule_expected_size=reference_rule_mean,
                        lower_quadrature_level=np.asarray(lower_level),
                        reference_quadrature_level_base=np.asarray(reference_level),
                        generated_size=generated,
                        particle_expected_size=particle_expected,
                        reference_expected_size=reference_mean,
                        reference_quadrature_level=reference_rule,
                        ess_fraction=ess, seeds=np.asarray(seeds))
    sample_path = args.samples_output.resolve(); sample_path.parent.mkdir(parents=True, exist_ok=True)
    sample_record = {}
    for index, (items, ptr) in enumerate(zip(sample_items, sample_ptr)):
        sample_record[f"items_{index}"] = items
        sample_record[f"ptr_{index}"] = ptr
    np.savez_compressed(sample_path, **sample_record)
    generated_size_count = np.bincount(generated.ravel(), minlength=model.nmax + 1)[1:]
    observed_size_count = np.bincount(observed, minlength=model.nmax + 1)[1:]
    training_maximum = int(np.max(
        data["trip_nlines"][data["trip_split"] == 0]))
    split_rng = np.random.default_rng(int(spec["seed"]) + 700001)
    split_order = split_rng.permutation(len(trips))
    split_left, split_right = np.array_split(split_order, 2)
    observed_split_half = {
        "contexts_per_half": [int(len(split_left)), int(len(split_right))],
        "size_total_variation": tv(
            np.bincount(observed[split_left], minlength=model.nmax + 1)[1:],
            np.bincount(observed[split_right], minlength=model.nmax + 1)[1:]),
        "category_total_variation": tv(
            observed_category_by_context[split_left].sum(0),
            observed_category_by_context[split_right].sum(0)),
        "item_total_variation": tv(
            observed_item_by_context[split_left].sum(0),
            observed_item_by_context[split_right].sum(0)),
    }
    generated_pairwise = []
    for left in range(args.replicates):
        for right in range(left + 1, args.replicates):
            generated_pairwise.append({
                "size_total_variation": tv(
                    np.bincount(generated[left], minlength=model.nmax + 1)[1:],
                    np.bincount(generated[right], minlength=model.nmax + 1)[1:]),
                "category_total_variation": tv(
                    pooled_category[left], pooled_category[right]),
                "item_total_variation": tv(
                    pooled_item[left], pooled_item[right]),
            })
    output = {
        "status": "completed", "checkpoint": str(checkpoint),
        "checkpoint_sha256": file_sha256(checkpoint),
        "data_fingerprint_sha256": blob["data_fingerprint_sha256"],
        "protocol_sha256": file_sha256(protocol_path), "split": "validation",
        "panel_selection": "deterministic household-balanced panel",
        "contexts": len(trips), "distinct_households": int(np.unique(data["trip_user"][trips]).size),
        "particles": int(spec["particles"]), "independent_replicates": args.replicates,
        "sampler_correctness": {
            "status": sampler_status,
            "reference_fitted_mean": float(reference_mean.mean()),
            "generated_mean": float(generated.mean()),
            "particle_expected_mean": float(particle_expected.mean()),
            "particle_expected_replicate_means": replicate_mean.tolist(),
            "particle_expected_minus_fitted_mean": float(difference.mean()),
            "replicate_mc_se": mc_se, "equivalence_margin": sampler_tolerance,
            "mc_half_width": mc_half_width,
            "minimum_ess_fraction": float(ess.min()),
            "reference_fidelity": {
                "status": "passed" if reference_fidelity_pass else "failed",
                "tolerance_items": reference_tolerance,
                "lower_quadrature_level": lower_level,
                "reference_quadrature_level": reference_level,
                "followup_quadrature_level": followup_level,
                "maximum_lower_reference_expected_size_gap": float(initial_reference_gap.max()),
                "followup_contexts": int(len(reference_failures)),
                "maximum_reference_followup_gap": (
                    float(q9_followup_gap.max()) if len(q9_followup_gap) else 0.0),
            },
        },
        "factual_calibration": {
            "status": factual_status,
            "observed_mean": float(observed.mean()), "fitted_mean": float(reference_mean.mean()),
            "fitted_minus_observed": calibration,
            "equivalence_margin_items": margin,
            "size_total_variation": tv(observed_size_count, generated_size_count),
            "category_total_variation": tv(observed_category, np.sum(pooled_category, axis=0)),
            "item_total_variation": tv(observed_item, np.sum(pooled_item, axis=0)),
            "observed_split_half_noise_reference": observed_split_half,
            "generated_replicate_pairwise_total_variation": {
                key: {
                    "mean": float(np.mean([row[key] for row in generated_pairwise])),
                    "maximum": float(np.max([row[key] for row in generated_pairwise])),
                }
                for key in (
                    "size_total_variation", "category_total_variation",
                    "item_total_variation")
            },
            "unobserved_support_extension": {
                "status": ("not_identifiable" if training_maximum < model.nmax
                           else "not_applicable"),
                "observed_training_maximum": training_maximum,
                "model_support_maximum": int(model.nmax),
                "observed_validation_baskets_above_training_maximum": int(
                    np.sum(observed > training_maximum)),
                "generated_baskets_above_training_maximum": int(
                    np.sum(generated > training_maximum)),
                "generated_rate_above_training_maximum": float(
                    np.mean(generated > training_maximum)),
                "interpretation": (
                    "generated sizes above the training maximum are extrapolations, "
                    "not empirically calibrated outcomes"),
            },
            "interpretation": "fixed observed contexts; household-cluster uncertainty conditions on the fit",
        },
        "per_context_output": str(per_path), "per_context_sha256": file_sha256(per_path),
        "samples_output": str(sample_path), "samples_sha256": file_sha256(sample_path),
        "runtime_seconds": time.monotonic()-started,
    }
    out = args.output.resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(strict_json_dumps(output)); print(strict_json_dumps(output), end="")


if __name__ == "__main__": main()
