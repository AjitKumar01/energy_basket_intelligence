#!/usr/bin/env python3
"""Held-out and model-conditional evaluation of retailer application primitives.

This script deliberately does not turn observational scenarios into causal claims.  It
evaluates what the transaction data can identify: sequential completion, retrospective
stop classification, candidate-bundle retrieval, and a category-choice proxy for
substitution.  Price, promotion, assortment, forecasting, and segmentation evidence is
consolidated separately from the existing locked reports.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("V3_AFFINITY", "1")

from checkpoint_io import ROOT, load_checkpoint
from conditional_basket import (
    completion_log_score,
    conditional_completion_smc,
)
from data import build
from features import Features
from fit import Batcher, popularity_logits
from provenance import file_sha256, strict_json_dumps


torch.set_default_dtype(torch.float64)


def basket(data, trip: int) -> np.ndarray:
    lo, hi = int(data["line_ptr"][trip]), int(data["line_ptr"][trip + 1])
    return np.unique(data["line_item"][lo:hi]).astype(np.int64, copy=False)


def balanced_size_panel(data, population, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    groups = {"complete_two": [], "partial_larger": []}
    used_households = {name: set() for name in groups}
    shuffled = np.asarray(population)[rng.permutation(len(population))]
    deferred = {name: [] for name in groups}
    for trip in shuffled:
        items = basket(data, int(trip))
        name = "complete_two" if len(items) == 2 else (
            "partial_larger" if len(items) >= 3 else None)
        if name is None or len(groups[name]) >= per_class:
            continue
        household = int(data["trip_user"][trip])
        if household in used_households[name]:
            deferred[name].append(int(trip)); continue
        groups[name].append(int(trip)); used_households[name].add(household)
        if all(len(value) >= per_class for value in groups.values()):
            break
    for name in groups:
        for trip in deferred[name]:
            if len(groups[name]) >= per_class:
                break
            groups[name].append(trip)
    if any(len(value) < per_class for value in groups.values()):
        raise RuntimeError("not enough supported size-two and size-three-plus trips")
    return np.asarray(groups["complete_two"] + groups["partial_larger"], dtype=np.int64)


def midrank(score: np.ndarray, target: int) -> float:
    value = score[target]
    return 1.0 + float(np.count_nonzero(score > value)) \
        + 0.5 * float(np.count_nonzero(score == value) - 1)


def retrieval_metrics(ranks, cutoffs=(5, 10, 20, 100)):
    values = np.asarray(ranks, dtype=np.float64)
    answer = {
        "cases": int(len(values)),
        "mrr": float(np.mean(1.0 / values)) if len(values) else None,
        "median_rank": float(np.median(values)) if len(values) else None,
        "mean_rank": float(np.mean(values)) if len(values) else None,
    }
    for cutoff in cutoffs:
        answer[f"recall_at_{cutoff}"] = (
            float(np.mean(values <= cutoff)) if len(values) else None)
    return answer


def binary_metrics(label, probability):
    label = np.asarray(label, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    clipped = np.clip(probability, 1e-12, 1 - 1e-12)
    positive = probability[label == 1]
    negative = probability[label == 0]
    auc = float(np.mean(
        (positive[:, None] > negative[None, :])
        + .5 * (positive[:, None] == negative[None, :])))
    return {
        "cases": int(len(label)), "positive_cases": int(label.sum()),
        "brier_score": float(np.mean((probability - label) ** 2)),
        "log_loss": float(np.mean(-(label * np.log(clipped)
                                    + (1 - label) * np.log(1 - clipped)))),
        "roc_auc": auc,
        "mean_stop_probability_complete_size_two": float(positive.mean()),
        "mean_stop_probability_partial_size_three_plus": float(negative.mean()),
    }


def same_subcommodity_candidates(metadata, available, target, excluded):
    label = metadata.loc[int(target), "SUB_COMMODITY_DESC"]
    candidates = [int(item) for item in available
                  if int(item) not in excluded
                  and metadata.loc[int(item), "SUB_COMMODITY_DESC"] == label]
    return candidates


@torch.no_grad()
def evaluate(args):
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    data = build()
    checkpoint = args.checkpoint.resolve()
    model, blob, meta = load_checkpoint(
        checkpoint, data,
        required_capabilities=("conditional_nonempty_incidence", "gram_interactions"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    features = Features(int(data["n_item"]), int(data["n_store"]), 712,
                        include_recency=False)
    batcher = Batcher(data, features, int(meta["nmax"]), include_recency=False)
    population = np.flatnonzero(
        (data["trip_split"] == 1) & (data["trip_nlines"] <= int(meta["nmax"])))
    trips = balanced_size_panel(data, population, args.per_class, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    revealed, hidden, stop_label = [], [], []
    for trip in trips:
        observed = basket(data, int(trip))
        if len(observed) == 2:
            anchor = observed.copy()
        else:
            anchor = rng.choice(observed, size=2, replace=False)
        revealed.append([int(item) for item in anchor])
        hidden.append([int(item) for item in observed if item not in set(anchor)])
        stop_label.append(int(len(observed) == 2))

    metadata = pd.read_parquet(ROOT / "basket_input/items.parquet").sort_values("item_id")
    metadata = metadata.set_index("item_id", drop=False)
    popularity = popularity_logits(
        data, np.flatnonzero(data["trip_split"] == 0)).numpy()
    schedule_axis = torch.linspace(0.0, 1.0, args.levels)
    schedule = (1.0 - (1.0 - schedule_axis).pow(args.power)).tolist()

    per_context = []
    all_incidence = []
    all_stop = []
    all_expected = []
    all_logz = []
    all_ess = []
    cross_ranks = []
    cross_recall = {5: [], 10: [], 20: [], 100: []}
    bundle_rank, bundle_pop_rank, bundle_candidates = [], [], []
    substitution_rank, substitution_pop_rank, substitution_candidates = [], [], []

    for start in range(0, len(trips), args.chunk):
        sub = trips[start:start + args.chunk]
        anchors = revealed[start:start + len(sub)]
        ix, ctx, _lc, house, *_ = batcher.make(sub)
        model.house, model.ctx = house, ctx
        slot_b = model.b_flat(ix).detach().clone()
        replicate = []
        for repeat in range(args.replicates):
            result = conditional_completion_smc(
                model, ix, slot_b, anchors, particles=args.particles,
                schedule=schedule, mutation_steps=args.mutation_steps,
                generator=torch.Generator().manual_seed(
                    args.seed + 100003 * (repeat + 1) + start))
            replicate.append(result)
            print(f"[cart-conditional] contexts={start + len(sub)}/{len(trips)} "
                  f"replicate={repeat + 1}/{args.replicates} "
                  f"min_ess={float(result.min_ess_fraction.min()):.6f}", flush=True)
        incidence = torch.stack([row.item_incidence for row in replicate]).mean(0).cpu().numpy()
        stop = torch.stack([row.stop_probability for row in replicate]).mean(0).cpu().numpy()
        expected = torch.stack([
            row.expected_additional_items for row in replicate]).mean(0).cpu().numpy()
        log_total = torch.logsumexp(torch.stack([
            row.log_normalizer for row in replicate]), 0) - math.log(args.replicates)
        log_total = log_total.cpu().numpy()
        ess = torch.stack([row.min_ess_fraction for row in replicate]).cpu().numpy()
        all_incidence.append(incidence); all_stop.append(stop)
        all_expected.append(expected); all_logz.append(log_total); all_ess.append(ess)

        for local, trip in enumerate(sub):
            absolute = start + local
            anchor = anchors[local]
            withheld = hidden[absolute]
            available_slots = torch.nonzero(ix.item_trip == local, as_tuple=True)[0]
            available = ix.item[available_slots].cpu().numpy()
            candidates = np.asarray([
                item for item in available if int(item) not in set(anchor)], dtype=np.int64)
            candidate_score = incidence[local, candidates]
            hidden_position = np.flatnonzero(np.isin(candidates, withheld))
            if len(hidden_position):
                ranks = [midrank(candidate_score, int(position)) for position in hidden_position]
                cross_ranks.append(min(ranks))
                order = candidates[np.argsort(-candidate_score, kind="stable")]
                for cutoff in cross_recall:
                    cross_recall[cutoff].append(
                        len(set(withheld) & set(order[:cutoff])) / len(withheld))

            observed = anchor + withheld
            if len(withheld) >= 2:
                actual_pair = withheld[:2]
                cats = [int(model.cat_of[item]) for item in actual_pair]
                pools = []
                for category in cats:
                    pool = [int(item) for item in available
                            if int(model.cat_of[int(item)]) == category
                            and int(item) not in set(observed)]
                    pools.append(pool)
                negatives = []
                attempts = 0
                while all(pools) and len(negatives) < args.bundle_negatives \
                        and attempts < args.bundle_negatives * 20:
                    pair = (int(rng.choice(pools[0])), int(rng.choice(pools[1])))
                    attempts += 1
                    if pair[0] != pair[1] and pair not in negatives:
                        negatives.append(pair)
                if len(negatives) >= 5:
                    pairs = [tuple(actual_pair)] + negatives
                    scores = np.asarray([float(completion_log_score(
                        model, ix, slot_b, local, anchor, pair)) for pair in pairs])
                    pop_scores = np.asarray([
                        popularity[first] + popularity[second] for first, second in pairs])
                    bundle_rank.append(midrank(scores, 0))
                    bundle_pop_rank.append(midrank(pop_scores, 0))
                    bundle_candidates.append(len(pairs))

            # Proxy only: can the model recover an observed withheld SKU among products
            # in the same subcommodity?  It does not reveal what happens if that SKU is
            # actually unavailable.
            if withheld:
                target = withheld[0]
                peers = same_subcommodity_candidates(
                    metadata, available, target, set(anchor))
                if target in peers and len(peers) >= 2:
                    scores = np.asarray([float(completion_log_score(
                        model, ix, slot_b, local, anchor, [item])) for item in peers])
                    target_position = peers.index(target)
                    substitution_rank.append(midrank(scores, target_position))
                    substitution_pop_rank.append(midrank(
                        popularity[np.asarray(peers)], target_position))
                    substitution_candidates.append(len(peers))

            exact_hidden_score = float(completion_log_score(
                model, ix, slot_b, local, anchor, withheld))
            per_context.append({
                "trip": int(trip), "household": int(data["trip_user"][trip]),
                "observed_size": int(len(observed)), "revealed_items": anchor,
                "hidden_items": withheld, "stop_label": stop_label[absolute],
                "stop_probability": float(stop[local]),
                "expected_additional_items": float(expected[local]),
                "actual_additional_items": int(len(withheld)),
                "actual_completion_log_probability": exact_hidden_score - float(log_total[local]),
                "minimum_smc_ess_fraction": float(ess[:, local].min()),
            })

    stop_probability = np.concatenate(all_stop)
    expected_additional = np.concatenate(all_expected)
    ess = np.concatenate(all_ess, axis=1)
    partial = np.asarray(stop_label) == 0
    actual_additional = np.asarray([len(value) for value in hidden], dtype=float)
    completion_error = expected_additional[partial] - actual_additional[partial]
    result = {
        "status": "completed",
        "claim_level": "heldout_predictive_and_model_conditional_not_causal",
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "data_fingerprint_sha256": blob["data_fingerprint_sha256"],
        "split": "validation", "panel_seed": args.seed,
        "panel": {
            "contexts": int(len(trips)), "per_stop_class": args.per_class,
            "distinct_households": int(np.unique(data["trip_user"][trips]).size),
            "construction": ("size-two completed baskets are positive stop cases; two "
                             "revealed items from size-three-plus baskets are negative cases"),
        },
        "sampler": {
            "particles": args.particles, "levels": args.levels,
            "schedule_power": args.power, "mutation_steps": args.mutation_steps,
            "replicates": args.replicates,
            "minimum_bridge_ess_fraction": float(ess.min()),
            "all_context_replicate_mean_ess_fraction": float(ess.mean()),
        },
        "real_time_cross_sell": {
            **retrieval_metrics(cross_ranks),
            **{f"mean_hidden_set_recall_at_{k}": float(np.mean(value))
               for k, value in cross_recall.items()},
            "estimand": "P(item in eventual completion | revealed cart, context)",
        },
        "basket_completion": {
            "partial_cases": int(partial.sum()),
            "observed_additional_items_mean": float(actual_additional[partial].mean()),
            "predicted_additional_items_mean": float(expected_additional[partial].mean()),
            "mean_error_items": float(completion_error.mean()),
            "mae_items": float(np.abs(completion_error).mean()),
            "rmse_items": float(np.sqrt(np.mean(completion_error ** 2))),
            "mean_actual_completion_log_probability": float(np.mean([
                row["actual_completion_log_probability"] for row in per_context if not row["stop_label"]])),
        },
        "stopping_probability": binary_metrics(stop_label, stop_probability),
        "personalized_bundle_retrieval": {
            "model": retrieval_metrics(bundle_rank, cutoffs=(1, 5, 10)),
            "popularity": retrieval_metrics(bundle_pop_rank, cutoffs=(1, 5, 10)),
            "mean_candidates": float(np.mean(bundle_candidates)) if bundle_candidates else None,
            "protocol": ("observed hidden pair versus same-category matched negative pairs; "
                         "exact candidate-completion scores conditional on two revealed items"),
        },
        "stockout_substitution_proxy": {
            "model": retrieval_metrics(substitution_rank, cutoffs=(1, 5, 10)),
            "popularity": retrieval_metrics(substitution_pop_rank, cutoffs=(1, 5, 10)),
            "mean_candidates": (float(np.mean(substitution_candidates))
                                if substitution_candidates else None),
            "identified_claim": "same-subcommodity held-out choice relevance",
            "not_identified_claim": "choice after the desired item is unavailable",
        },
        "limitations": [
            "transactions record final sets, not within-trip item order",
            "the two-item reveal is a retrospective masking protocol, not a logged cart stream",
            "stockout substitution has no availability intervention or lost-demand label",
            "bundle negatives are matched candidate alternatives, not randomized offers",
        ],
        "per_context": per_context,
        "runtime_seconds": time.monotonic() - started,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-class", type=int, default=32)
    parser.add_argument("--particles", type=int, default=128)
    parser.add_argument("--levels", type=int, default=13)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--mutation-steps", type=int, default=1)
    parser.add_argument("--replicates", type=int, default=2)
    parser.add_argument("--bundle-negatives", type=int, default=31)
    parser.add_argument("--chunk", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=91501)
    args = parser.parse_args()
    output = evaluate(args)
    path = args.output.resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(strict_json_dumps(output))
    print(strict_json_dumps({key: value for key, value in output.items()
                             if key != "per_context"}), end="", flush=True)


if __name__ == "__main__":
    main()
