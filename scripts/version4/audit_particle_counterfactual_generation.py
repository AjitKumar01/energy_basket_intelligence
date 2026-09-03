#!/usr/bin/env python3
"""Read-only counterfactual and generation audit for interaction-particle checkpoints.

The audit never calls the historical Smolyak/QMC incidence or sampler.  It constructs
equally weighted samples from the unchanged version-4 basket law with interaction-
tempered SMC.  Price interventions are then evaluated by the exact additive-energy
Radon--Nikodym derivative on those same factual particles.  Basket generation applies an
additional beta=1 invariant blocked update to the final SMC population.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import pandas as pd
import torch

from data import build
from features import Features
from fit import Batcher
from interaction_particles import (blocked_rejuvenation,
                                   rao_blackwell_particle_statistics)
from ragged import RaggedModel
from sparse_artifact import load_sparse_initialization_artifact
from tempered_ais import annealed_smc_logz


torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--trips", type=int, default=8)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--levels", type=int, default=17)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--rejuvenation", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2561900)
    parser.add_argument("--actions", type=float, nargs="+",
                        default=[math.log(.8), math.log(.9), 0.0,
                                 math.log(1.1), math.log(1.2)])
    parser.add_argument("--output", type=Path,
                        default=Path("out/v3_particle_counterfactual_generation.json"))
    return parser.parse_args()


def load_checkpoint(path: Path, data):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    artifact = Path(blob["config"]["artifact"])
    if not artifact.is_absolute():
        # A historical checkpoint can live in another clone.  Resolve its relative
        # initialization artifact beside that clone before falling back to this one.
        checkpoint_root = path.resolve().parents[1]
        beside_checkpoint = checkpoint_root / artifact
        artifact = (beside_checkpoint if beside_checkpoint.exists()
                    else ROOT / artifact)
    raw = torch.load(artifact, map_location="cpu", weights_only=False)
    meta = raw["metadata"]
    model = RaggedModel(
        int(data["n_item"]), int(data["n_user"]), int(data["n_cat"]),
        K=int(meta["K"]), Kz=int(meta["Kz"]), nmax=int(meta["nmax"]),
        R=int(meta["R"]), seed=int(meta["seed"]), S=int(data["n_store"]),
        Kp=int(meta["Kp"]), phi_init=0.0,
        household_size_rank1=bool(meta.get("household_size_rank1", False)))
    load_sparse_initialization_artifact(artifact, model)
    model.load_state_dict(blob["model"], strict=True)
    model._poly_degree_native = True
    model._esp_native = True
    model._esp_log_blocked = True
    model.double().eval()
    return model, blob, meta


def copied_context(ctx):
    return {key: value.clone() if torch.is_tensor(value) else value
            for key, value in ctx.items()}


def particle_delta(states, delta_slot, batches):
    answer = torch.zeros(len(states), batches, dtype=delta_slot.dtype)
    for p, particle in enumerate(states):
        for b, slots in enumerate(particle):
            answer[p, b] = delta_slot[slots].sum()
    return answer


def selected_trip_panel(data, count, nmax, seed):
    candidates = np.flatnonzero(
        (data["trip_split"] == 1) & (data["trip_nlines"] <= nmax)
        & (data["trip_nlines"] >= 1))
    rng = np.random.default_rng(seed)
    return candidates[rng.permutation(len(candidates))[:count]]


def named_basket(items, metadata, limit=8):
    names = []
    for item in items[:limit]:
        text = str(metadata.SUB_COMMODITY_DESC.iloc[int(item)]).strip()
        names.append({"item": int(item), "description": text})
    return names


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    ckpt = args.ckpt if args.ckpt.is_absolute() else ROOT / args.ckpt
    data = build()
    model, blob, meta = load_checkpoint(ckpt, data)
    batcher = Batcher(data, Features(int(data["n_item"]), int(data["n_store"]), 712),
                      int(meta["nmax"]))
    trips = selected_trip_panel(data, args.trips, int(meta["nmax"]), args.seed)
    ix, ctx, _line_ctx, house, line_item, line_trip, _line_cat, _line_q = batcher.make(trips)
    model.house, model.ctx = house, ctx

    axis = torch.linspace(0.0, 1.0, args.levels)
    schedule = 1.0 - (1.0 - axis).pow(args.power)
    generator = torch.Generator().manual_seed(args.seed + 1)
    print(f"[generation] starting SMC: trips={ix.B} particles={args.particles} "
          f"levels={args.levels}", flush=True)
    started = time.perf_counter()
    smc = annealed_smc_logz(model, ix, schedule, particles=args.particles,
                            mutation_steps=1, generator=generator)
    smc_seconds = time.perf_counter() - started
    print(f"[generation] SMC complete in {smc_seconds:.1f}s; computing audits",
          flush=True)
    # Rao--Blackwellization gives nonzero, low-variance incidence estimates even for a
    # purchased SKU absent from a finite outer particle population.
    factual_stats = rao_blackwell_particle_statistics(model, ix, smc.states)
    size_axis = torch.arange(1, model.nmax + 1, dtype=model.phi.dtype)
    factual_size = (factual_stats.size_probability * size_axis).sum(1)
    factual_b = model.b_flat(ix).clone()

    rng = np.random.default_rng(args.seed + 2)
    chosen_slots, chosen_items = [], []
    for b in range(ix.B):
        bought = torch.unique(line_item[line_trip == b]).cpu().numpy()
        chosen = int(bought[rng.integers(len(bought))])
        slot = torch.nonzero((ix.item_trip == b) & (ix.item == chosen), as_tuple=True)[0]
        if not slot.numel():
            raise RuntimeError("purchased item is absent from its store assortment")
        chosen_slots.append(int(slot[0]))
        chosen_items.append(chosen)
    chosen_slots = torch.as_tensor(chosen_slots, dtype=torch.long)
    assortment_size = torch.bincount(ix.item_trip, minlength=ix.B).to(model.phi.dtype)

    rows = []
    factual_own = factual_stats.item_incidence[
        torch.arange(ix.B), torch.as_tensor(chosen_items)]
    for action in args.actions:
        uniform = copied_context(ctx)
        uniform["dlp"] += action
        uniform["dlp_bar"] += action
        model.ctx = uniform
        uniform_b = model.b_flat(ix).clone()
        uniform_delta = particle_delta(smc.states, uniform_b - factual_b, ix.B)
        uniform_log_weight = torch.log_softmax(uniform_delta, dim=0)
        uniform_ess = torch.exp(-torch.logsumexp(2.0 * uniform_log_weight, dim=0)) \
            / args.particles
        uniform_stats = rao_blackwell_particle_statistics(
            model, ix, smc.states, uniform_log_weight)
        uniform_size = (uniform_stats.size_probability * size_axis).sum(1)

        own = copied_context(ctx)
        own["dlp"][chosen_slots] += action
        own["dlp_bar"] += action / assortment_size
        model.ctx = own
        own_b = model.b_flat(ix).clone()
        own_delta = particle_delta(smc.states, own_b - factual_b, ix.B)
        own_log_weight = torch.log_softmax(own_delta, dim=0)
        own_ess = torch.exp(-torch.logsumexp(2.0 * own_log_weight, dim=0)) \
            / args.particles
        own_stats = rao_blackwell_particle_statistics(
            model, ix, smc.states, own_log_weight)
        own_incidence = own_stats.item_incidence[
            torch.arange(ix.B), torch.as_tensor(chosen_items)]
        rows.append({
            "price_multiplier": math.exp(action),
            "log_price_change": action,
            "own_incidence_mean": float(own_incidence.mean()),
            "own_incidence_retained": float(
                (own_incidence / factual_own.clamp_min(1e-12)).mean()),
            "uniform_expected_size": float(uniform_size.mean()),
            "uniform_size_change": float((uniform_size - factual_size).mean()),
            "uniform_reweight_ess_min": float(uniform_ess.min()),
            "own_reweight_ess_min": float(own_ess.min()),
        })

    model.ctx = ctx
    generated_states = blocked_rejuvenation(
        model, ix, smc.states, beta=1.0, steps=args.rejuvenation,
        generator=torch.Generator().manual_seed(args.seed + 3))
    metadata = pd.read_parquet(ROOT / "basket_input" / "items.parquet").sort_values("item_id")
    item_category = metadata.cat_id.to_numpy(dtype=np.int64)
    generated_sizes, invalid, duplicates = [], 0, 0
    generated_categories = np.zeros(int(data["n_cat"]), dtype=np.float64)
    observed_categories = np.zeros_like(generated_categories)
    examples = []
    for b in range(ix.B):
        allowed = set(ix.item[ix.item_trip == b].cpu().numpy().tolist())
        observed = torch.unique(line_item[line_trip == b]).cpu().numpy().tolist()
        np.add.at(observed_categories, item_category[np.asarray(observed, dtype=int)], 1)
        for p in range(len(generated_states)):
            items = ix.item[generated_states[p][b]].cpu().numpy().tolist()
            generated_sizes.append(len(items))
            invalid += int(any(item not in allowed for item in items))
            duplicates += int(len(items) != len(set(items)))
            np.add.at(generated_categories, item_category[np.asarray(items, dtype=int)], 1)
        examples.append({
            "trip": int(trips[b]),
            "observed": named_basket(observed, metadata),
            "generated": named_basket(
                ix.item[generated_states[0][b]].cpu().numpy().tolist(), metadata),
        })
    generated_categories /= max(generated_categories.sum(), 1.0)
    observed_categories /= max(observed_categories.sum(), 1.0)
    generated_sizes = np.asarray(generated_sizes, dtype=np.float64)
    observed_sizes = data["trip_nlines"][trips].astype(np.float64)
    output = {
        "checkpoint": str(ckpt),
        "checkpoint_iteration": int(blob["iter"]),
        "best_iteration": int(blob["best_iteration"]),
        "trips": trips.tolist(),
        "particles_per_trip": args.particles,
        "smc_levels": args.levels,
        "smc_seconds": smc_seconds,
        "smc_ess_min": float(smc.min_ess_fraction.min()),
        "smc_ess_median": float(smc.min_ess_fraction.median()),
        "factual_expected_size": float(factual_size.mean()),
        "observed_size_mean": float(observed_sizes.mean()),
        "counterfactuals": rows,
        "generation": {
            "baskets": int(generated_sizes.size),
            "generated_size_mean": float(generated_sizes.mean()),
            "generated_size_variance": float(generated_sizes.var()),
            "observed_size_mean": float(observed_sizes.mean()),
            "observed_size_variance": float(observed_sizes.var()),
            "category_total_variation": float(
                0.5 * np.abs(generated_categories - observed_categories).sum()),
            "invalid_assortment_baskets": invalid,
            "duplicate_item_baskets": duplicates,
            "examples": examples,
        },
    }
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    summary = {
        key: output[key] for key in (
            "checkpoint", "particles_per_trip", "smc_levels", "smc_seconds",
            "smc_ess_min", "smc_ess_median", "factual_expected_size",
            "observed_size_mean")
    }
    summary["counterfactuals"] = output["counterfactuals"]
    summary["generation"] = {
        key: value for key, value in output["generation"].items()
        if key != "examples"
    }
    summary["full_report"] = str(args.output)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
