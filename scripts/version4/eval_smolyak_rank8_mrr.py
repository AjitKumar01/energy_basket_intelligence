#!/usr/bin/env python3
"""Locked recommendation evaluation for a fixed-rank Version-4 checkpoint.

The default reproduces the published/run310 protocol: hide one bought item and rank every
available candidate by its exact add-one conditional energy.  This score is independent
of Z, so attaching a Smolyak rule to it is both unnecessary and a source of avoidable
protocol drift.  Marginal-incidence variants remain available explicitly for diagnostics.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch

from audit_particle_counterfactual_generation import ROOT, load_checkpoint
from data import build
from eval_mrr_cutoffs import popularity_ranks
from features import Features
from fit import Batcher, popularity_logits, rec_eval
from ragged import smolyak_grid


torch.set_default_dtype(torch.float64)


def metrics(ranks):
    ranks = np.asarray(ranks, dtype=float)
    reciprocal = 1.0 / ranks
    result = {
        "cases": int(len(ranks)),
        "mrr": float(reciprocal.mean()),
        "mrr_se": float(reciprocal.std(ddof=1) / math.sqrt(len(ranks))),
        "median_rank": float(np.median(ranks)),
        "mean_rank": float(np.mean(ranks)),
    }
    for cutoff in (5, 10, 20, 100):
        hit = ranks <= cutoff
        result[f"mrr_at_{cutoff}"] = float(np.where(hit, reciprocal, 0.0).mean())
        result[f"recall_at_{cutoff}"] = float(hit.mean())
    return result


def midrank(score, position):
    target = score[position]
    greater = np.count_nonzero(score > target)
    tied = np.count_nonzero(score == target)
    return 1.0 + greater + 0.5 * (tied - 1)


@torch.no_grad()
def locked_add_one(model, batcher, data, trips, seed):
    """Exact conditional add-one ranks on one common hidden-item manifest."""
    popularity = popularity_logits(
        data, np.flatnonzero(data["trip_split"] == 0)).numpy()
    rng = np.random.default_rng(seed + 17)
    ranks = {name: [] for name in (
        "popularity", "additive_utility", "structured_no_gram",
        "full_interaction")}
    candidate_counts = []
    hidden_terms = []
    for start in range(0, len(trips), 24):
        sub = np.asarray(trips[start:start + 24], dtype=np.int64)
        ix, ctx, _line_ctx, house, line_item, line_trip, _line_cat, _line_q = \
            batcher.make(sub)
        model.house, model.ctx = house, ctx
        utility = model.b_flat(ix)
        slot_category = model.cat_of[ix.item]
        for basket_index in range(ix.B):
            observed = torch.unique(line_item[line_trip == basket_index])
            if observed.numel() < 2:
                continue
            hidden = int(observed[int(rng.integers(observed.numel()))])
            remainder = observed[observed != hidden]
            slots = torch.nonzero(
                ix.item_trip == basket_index, as_tuple=True)[0]
            available = ix.item[slots]
            keep = ~torch.isin(available, remainder)
            slots, available = slots[keep], available[keep]
            hidden_position = torch.nonzero(
                available == hidden, as_tuple=True)[0]
            if hidden_position.numel() != 1:
                continue
            position = int(hidden_position[0])
            candidate_category = slot_category[slots]
            category_count = torch.bincount(
                model.cat_of[remainder], minlength=model.C)
            gram = model.phi[available] @ model.phi[remainder].sum(0)
            category = (-model.rho_c[candidate_category]
                        * category_count[candidate_category])
            additive = utility[slots]
            structured = additive + category
            scores = {
                "popularity": popularity[available.numpy()],
                "additive_utility": additive.numpy(),
                "structured_no_gram": structured.numpy(),
                "full_interaction": (structured + gram).numpy(),
            }
            for name, score in scores.items():
                ranks[name].append(midrank(score, position))
            candidate_counts.append(available.numel())
            hidden_terms.append((float(additive[position]),
                                 float(category[position]),
                                 float(gram[position])))
    summaries = {name: metrics(value) for name, value in ranks.items()}
    full = np.asarray(ranks["full_interaction"], dtype=np.float64)
    additive = np.asarray(ranks["additive_utility"], dtype=np.float64)
    gain = 1.0 / full - 1.0 / additive
    gain_se = float(gain.std(ddof=1) / math.sqrt(len(gain)))
    summaries["comparison"] = {
        "full_beats_additive_fraction": float(np.mean(full < additive)),
        "full_ties_additive_fraction": float(np.mean(full == additive)),
        "mean_rank_change_full_minus_additive": float(np.mean(full - additive)),
        "mrr_gain_full_minus_additive": float(gain.mean()),
        "mrr_gain_standard_error": gain_se,
        "mrr_gain_95_interval": [float(gain.mean() - 1.96 * gain_se),
                                 float(gain.mean() + 1.96 * gain_se)],
        "mean_candidates": float(np.mean(candidate_counts)),
        "hidden_energy_terms_mean": dict(zip(
            ("additive", "category", "gram"),
            np.mean(hidden_terms, axis=0).tolist())),
    }
    return summaries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--trips", type=int, default=2000)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--rank", type=int, default=0,
                        help="expected active rank; <=0 infers it from the checkpoint")
    parser.add_argument("--level", type=int, default=0,
                        help="Smolyak level; <=0 uses active_rank+2")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--protocol", choices=(
        "locked-add-one", "conditioned-incidence", "unconditioned-incidence"),
        default="locked-add-one")
    parser.add_argument("--seed", type=int, default=2560202,
                        help="trip-manifest seed; hidden-item seed is seed+17")
    parser.add_argument("--conditioned", action="store_true",
                        help="deprecated alias for --protocol conditioned-incidence")
    parser.add_argument("--output", type=Path,
                        default=Path("out/v3_smolyak_rank8_mrr.json"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = build()
    ckpt = args.ckpt if args.ckpt.is_absolute() else ROOT / args.ckpt
    model, blob, meta = load_checkpoint(ckpt, data)
    singular = torch.linalg.svdvals(model.phi)
    active_rank = int((singular > singular[0] * 1e-10).sum())
    if args.rank > 0 and active_rank != args.rank:
        raise RuntimeError(
            f"checkpoint active interaction rank is {active_rank}, not {args.rank}")
    model.eval()
    batcher = Batcher(data, Features(int(data["n_item"]), int(data["n_store"]), 712),
                      int(meta["nmax"]))
    split = {"validation": 1, "test": 2}[args.split]
    population = np.flatnonzero((data["trip_split"] == split) &
                                (data["trip_nlines"] <= int(meta["nmax"])))
    trips = population[
        np.random.default_rng(args.seed).permutation(len(population))[:args.trips]]
    protocol = "conditioned-incidence" if args.conditioned else args.protocol
    base = {
        "checkpoint": str(ckpt),
        "checkpoint_iteration": int(blob["iter"]),
        "active_rank": active_rank,
        "requested_test_trips": int(len(trips)),
        "trip_manifest_seed": args.seed,
    }
    if protocol == "locked-add-one":
        recommendation = locked_add_one(model, batcher, data, trips, args.seed)
        output = {
            **base,
            "protocol": (
                "hide one test-basket item; exact conditional add-one energy over "
                "the complete contemporaneous store assortment, with midranks for ties"),
            "normalizer_required": False,
            "smolyak_level": None,
            "smolyak_nodes": 0,
            "model": recommendation["full_interaction"],
            "popularity": recommendation["popularity"],
            "recommendation": recommendation,
        }
    else:
        level = args.level if args.level > 0 else active_rank + 2
        active_nodes, weights = smolyak_grid(active_rank, level)
        nodes = torch.zeros(len(weights), model.Kz, dtype=model.phi.dtype)
        nodes[:, :active_rank] = active_nodes
        model.quad = (nodes, weights)
        model.quad_a = None
        conditioned = protocol == "conditioned-incidence"
        ranks = rec_eval(model, batcher, trips, seed=args.seed + 17,
                         chunk=args.chunk, return_ranks=True,
                         conditioned=conditioned)
        pop = popularity_ranks(data, trips, seed=args.seed + 17)
        if len(pop) != len(ranks):
            raise RuntimeError(
                "model and popularity retained different holdout cases")
        reciprocal_gain = 1.0 / ranks - 1.0 / pop
        gain_se = float(
            reciprocal_gain.std(ddof=1) / math.sqrt(len(ranks)))
        output = {
            **base,
            "smolyak_level": level,
            "smolyak_nodes": len(weights),
            "normalizer_required": True,
            "protocol": (
                "deterministic conditional basket-completion incidence; complete support"
                if conditioned else
                "deterministic unconditional marginal incidence pi=dlogZ/db; complete support"),
            "model": metrics(ranks),
            "popularity": metrics(pop),
            "paired_mrr_gain": float(reciprocal_gain.mean()),
            "paired_mrr_gain_se": gain_se,
            "paired_mrr_gain_95_interval": [
                float(reciprocal_gain.mean() - 1.96 * gain_se),
                float(reciprocal_gain.mean() + 1.96 * gain_se)],
        }
    path = args.output if args.output.is_absolute() else ROOT / args.output
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
