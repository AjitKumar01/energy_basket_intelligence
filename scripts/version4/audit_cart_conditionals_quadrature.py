#!/usr/bin/env python3
"""Adjacent-rule deterministic audit of cart-conditional probabilities on real data."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("V3_AFFINITY", "1")

from checkpoint_io import load_checkpoint
from conditional_basket import conditional_completion_quadrature
from data import build
from evaluate_retail_applications import (
    balanced_size_panel, basket, binary_metrics, midrank, retrieval_metrics,
)
from features import Features
from fit import Batcher
from provenance import file_sha256, strict_json_dumps
from ragged import smolyak_grid


torch.set_default_dtype(torch.float64)


def padded_rule(model, active_rank, level):
    active, weights = smolyak_grid(active_rank, level)
    nodes = torch.zeros(len(weights), model.Kz, dtype=model.phi.dtype)
    nodes[:, :active_rank] = active
    return nodes, weights


@torch.no_grad()
def panel(args):
    started = time.monotonic(); torch.set_num_threads(args.threads)
    data = build(); checkpoint = args.checkpoint.resolve()
    model, blob, meta = load_checkpoint(
        checkpoint, data,
        required_capabilities=("conditional_nonempty_incidence", "gram_interactions"))
    for parameter in model.parameters(): parameter.requires_grad_(False)
    singular = torch.linalg.svdvals(model.phi)
    active_rank = int((singular > singular[0] * 1e-10).sum())
    low_level, high_level = active_rank + 2, active_rank + 3
    low_rule = padded_rule(model, active_rank, low_level)
    high_rule = padded_rule(model, active_rank, high_level)
    batcher = Batcher(
        data, Features(int(data["n_item"]), int(data["n_store"]),
                       include_recency=False), int(meta["nmax"]), include_recency=False)
    population = np.flatnonzero(
        (data["trip_split"] == 1) & (data["trip_nlines"] <= int(meta["nmax"])))
    trips = balanced_size_panel(data, population, args.per_class, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    anchors, hidden, labels = [], [], []
    for trip in trips:
        observed = basket(data, int(trip))
        anchor = observed if len(observed) == 2 else rng.choice(observed, 2, replace=False)
        anchors.append(list(map(int, anchor)))
        hidden.append([int(item) for item in observed if item not in set(anchor)])
        labels.append(int(len(observed) == 2))

    primary_stop, primary_expected, primary_incidence = [], [], []
    low_stop, low_expected, low_incidence = [], [], []
    for start in range(0, len(trips), args.chunk):
        sub = trips[start:start + args.chunk]
        ix, ctx, _lc, house, *_ = batcher.make(sub)
        model.house, model.ctx = house, ctx
        slot_b = model.b_flat(ix).detach()
        lo = conditional_completion_quadrature(
            model, ix, slot_b, anchors[start:start + len(sub)], *low_rule)
        hi = conditional_completion_quadrature(
            model, ix, slot_b, anchors[start:start + len(sub)], *high_rule)
        primary_stop.append(hi.stop_probability.numpy())
        primary_expected.append(hi.expected_additional_items.numpy())
        primary_incidence.append(hi.item_incidence.numpy())
        low_stop.append(lo.stop_probability.numpy())
        low_expected.append(lo.expected_additional_items.numpy())
        low_incidence.append(lo.item_incidence.numpy())
        print(f"[cart-quadrature] contexts={start + len(sub)}/{len(trips)}", flush=True)

    stop = np.concatenate(primary_stop); expected = np.concatenate(primary_expected)
    incidence = np.concatenate(primary_incidence)
    stop_low = np.concatenate(low_stop); expected_low = np.concatenate(low_expected)
    incidence_low = np.concatenate(low_incidence)
    high_stop, high_expected = stop.copy(), expected.copy()
    initial_stop_gap = np.abs(stop - stop_low)
    initial_size_gap = np.abs(expected - expected_low)
    initial_incidence_gap = np.max(np.abs(incidence - incidence_low), axis=1)
    followup_indices = np.flatnonzero(
        (initial_stop_gap > args.maximum_stop_gap)
        | (initial_size_gap > args.maximum_size_gap)
        | (initial_incidence_gap > args.maximum_incidence_gap))
    reference_stop, reference_expected = stop.copy(), expected.copy()
    reference_incidence = incidence.copy()
    reference_level = np.full(len(trips), high_level, dtype=np.int16)
    follow_level = active_rank + 4
    follow_rule = padded_rule(model, active_rank, follow_level)
    follow_stop_gap, follow_size_gap, follow_incidence_gap = [], [], []
    for offset in range(0, len(followup_indices), args.followup_chunk):
        selected = followup_indices[offset:offset + args.followup_chunk]
        sub = trips[selected]
        ix, ctx, _lc, house, *_ = batcher.make(sub)
        model.house, model.ctx = house, ctx
        slot_b = model.b_flat(ix).detach()
        follow = conditional_completion_quadrature(
            model, ix, slot_b, [anchors[int(index)] for index in selected],
            *follow_rule)
        follow_stop = follow.stop_probability.numpy()
        follow_expected = follow.expected_additional_items.numpy()
        follow_incidence = follow.item_incidence.numpy()
        follow_stop_gap.extend(np.abs(follow_stop - stop[selected]).tolist())
        follow_size_gap.extend(np.abs(follow_expected - expected[selected]).tolist())
        follow_incidence_gap.extend(
            np.max(np.abs(follow_incidence - incidence[selected]), axis=1).tolist())
        reference_stop[selected] = follow_stop
        reference_expected[selected] = follow_expected
        reference_incidence[selected] = follow_incidence
        reference_level[selected] = follow_level
        print(f"[cart-quadrature-followup] contexts="
              f"{min(offset + args.followup_chunk, len(followup_indices))}/"
              f"{len(followup_indices)}", flush=True)

    stop, expected, incidence = reference_stop, reference_expected, reference_incidence
    per_context = []
    ranks, recalls = [], {5: [], 10: [], 20: [], 100: []}
    for start in range(0, len(trips), args.chunk):
        sub = trips[start:start + args.chunk]
        ix, *_ = batcher.make(sub)
        for local, trip in enumerate(sub):
            absolute = start + local
            candidates = ix.item[ix.item_trip == local].numpy()
            candidates = np.asarray([
                item for item in candidates if int(item) not in set(anchors[absolute])])
            scores = incidence[absolute, candidates]
            positions = np.flatnonzero(np.isin(candidates, hidden[absolute]))
            if len(positions):
                hidden_ranks = [midrank(scores, int(position)) for position in positions]
                ranks.append(min(hidden_ranks))
                for cutoff in recalls:
                    recalls[cutoff].append(np.mean(np.asarray(hidden_ranks) <= cutoff))
            per_context.append({
                "trip": int(trip), "household": int(data["trip_user"][trip]),
                "revealed_items": anchors[absolute], "hidden_items": hidden[absolute],
                "stop_label": labels[absolute], "reference_level": int(reference_level[absolute]),
                "stop_probability": float(stop[absolute]),
                "expected_additional_items": float(expected[absolute]),
                "level_7_stop_probability": float(stop_low[absolute]),
                "level_7_expected_additional_items": float(expected_low[absolute]),
                "level_8_stop_probability": float(high_stop[absolute]),
                "level_8_expected_additional_items": float(high_expected[absolute]),
            })
    partial = np.asarray(labels) == 0
    actual = np.asarray([len(value) for value in hidden], dtype=float)
    error = expected[partial] - actual[partial]
    cross = retrieval_metrics(ranks)
    cross.update({f"mean_hidden_set_recall_at_{cutoff}": float(np.mean(value))
                  for cutoff, value in recalls.items()})
    adjacent = {
        "maximum_initial_adjacent_stop_probability_gap": float(initial_stop_gap.max()),
        "maximum_initial_adjacent_expected_size_gap": float(initial_size_gap.max()),
        "maximum_initial_adjacent_item_incidence_gap": float(initial_incidence_gap.max()),
        "followup_contexts": int(len(followup_indices)),
        "maximum_high_followup_stop_probability_gap": float(max(follow_stop_gap, default=0.0)),
        "maximum_high_followup_expected_size_gap": float(max(follow_size_gap, default=0.0)),
        "maximum_high_followup_item_incidence_gap": float(max(follow_incidence_gap, default=0.0)),
    }
    gates = {
        "stop_probability": adjacent["maximum_high_followup_stop_probability_gap"]
            <= args.maximum_stop_gap,
        "expected_additional_items": adjacent["maximum_high_followup_expected_size_gap"]
            <= args.maximum_size_gap,
        "item_incidence": adjacent["maximum_high_followup_item_incidence_gap"]
            <= args.maximum_incidence_gap,
    }
    return {
        "status": "passed" if all(gates.values()) else "failed",
        "split": "validation", "panel_seed": args.seed,
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "data_fingerprint_sha256": blob["data_fingerprint_sha256"],
        "panel": {"contexts": len(trips), "per_stop_class": args.per_class,
                  "distinct_households": int(np.unique(data["trip_user"][trips]).size)},
        "quadrature": {
            "active_rank": active_rank, "low_level": low_level,
            "low_nodes": len(low_rule[1]), "high_level": high_level,
            "high_nodes": len(high_rule[1]),
            "followup_level": follow_level, "followup_nodes": len(follow_rule[1]),
            **adjacent,
            "declared_tolerances": {
                "maximum_stop_probability_gap": args.maximum_stop_gap,
                "maximum_expected_size_gap": args.maximum_size_gap,
                "maximum_item_incidence_gap": args.maximum_incidence_gap,
            },
            "gates": gates,
        },
        "real_time_cross_sell": cross,
        "basket_completion": {
            "partial_cases": int(partial.sum()),
            "observed_additional_items_mean": float(actual[partial].mean()),
            "predicted_additional_items_mean": float(expected[partial].mean()),
            "mean_error_items": float(error.mean()), "mae_items": float(np.abs(error).mean()),
            "rmse_items": float(np.sqrt(np.mean(error**2))),
        },
        "stopping_probability": binary_metrics(labels, stop),
        "per_context": per_context, "runtime_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=32)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--followup-chunk", type=int, default=2)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=91501)
    parser.add_argument("--maximum-stop-gap", type=float, default=1e-4)
    parser.add_argument("--maximum-size-gap", type=float, default=.02)
    parser.add_argument("--maximum-incidence-gap", type=float, default=1e-4)
    args = parser.parse_args(); result = panel(args)
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(strict_json_dumps(result))
    print(strict_json_dumps({key: value for key, value in result.items()
                             if key != "per_context"}), end="", flush=True)


if __name__ == "__main__": main()
