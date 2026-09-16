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
import math
import os
import time
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import pandas as pd
import torch

from checkpoint_io import ROOT, load_checkpoint
from data import BI, build
from features import Features
from fit import Batcher
from interaction_particles import (blocked_rejuvenation,
                                   rao_blackwell_particle_statistics)
from pipeline_support import named_basket, particle_delta
from tempered_ais import annealed_smc_logz
from provenance import file_sha256, strict_json_dumps
from price_response import changed_price_context


torch.set_default_dtype(torch.float64)


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
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--panel-input", type=Path,
                        help="NPZ containing a frozen `trips` array")
    parser.add_argument("--panel-output", type=Path,
                        help="write the exact selected trip/household panel")
    parser.add_argument("--per-context-output", type=Path,
                        help="write factual, action, ESS and generated-size arrays")
    parser.add_argument("--actions", type=float, nargs="+",
                        default=[math.log(.8), math.log(.9), 0.0,
                                 math.log(1.1), math.log(1.2)])
    parser.add_argument("--output", type=Path,
                        default=Path("out/v3_particle_counterfactual_generation.json"))
    return parser.parse_args()


def selected_trip_panel(data, count, nmax, seed, split=1):
    candidates = np.flatnonzero(
        (data["trip_split"] == int(split)) & (data["trip_nlines"] <= nmax)
        & (data["trip_nlines"] >= 1))
    rng = np.random.default_rng(seed)
    return candidates[rng.permutation(len(candidates))[:count]]


@torch.no_grad()
def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    ckpt = args.ckpt if args.ckpt.is_absolute() else ROOT / args.ckpt
    data = build()
    model, blob, meta = load_checkpoint(
        ckpt, data,
        required_capabilities=("conditional_nonempty_incidence", "gram_interactions"))
    features = Features(int(data["n_item"]), int(data["n_store"]), include_recency=False)
    batcher = Batcher(data, features, int(meta["nmax"]), include_recency=False)
    split_code = {"validation": 1, "test": 2}[args.split]
    if args.panel_input:
        with np.load(args.panel_input.resolve()) as panel:
            trips = panel["trips"].astype(np.int64, copy=True)
        if len(trips) != args.trips:
            raise ValueError("frozen panel size differs from --trips")
    else:
        trips = selected_trip_panel(
            data, args.trips, int(meta["nmax"]), args.seed, split_code)
    if not np.all(data["trip_split"][trips] == split_code):
        raise ValueError("generation panel contains a trip from the wrong split")
    if args.panel_output:
        panel_output = args.panel_output.resolve(); panel_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(panel_output, trips=trips,
                            household=data["trip_user"][trips], split=split_code)
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

    rows, per_action = [], []
    factual_own = factual_stats.item_incidence[
        torch.arange(ix.B), torch.as_tensor(chosen_items)]
    for action in args.actions:
        uniform = changed_price_context(ctx, ix.item_trip, action)
        model.ctx = uniform
        uniform_b = model.b_flat(ix).clone()
        uniform_delta = particle_delta(smc.states, uniform_b - factual_b, ix.B)
        uniform_log_weight = torch.log_softmax(uniform_delta, dim=0)
        uniform_ess = torch.exp(-torch.logsumexp(2.0 * uniform_log_weight, dim=0)) \
            / args.particles
        uniform_stats = rao_blackwell_particle_statistics(
            model, ix, smc.states, uniform_log_weight)
        uniform_size = (uniform_stats.size_probability * size_axis).sum(1)

        own_change = torch.zeros_like(ctx["dlp"])
        own_change[chosen_slots] = action
        own = changed_price_context(ctx, ix.item_trip, own_change)
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
        per_action.append({"uniform_size": uniform_size.cpu().numpy(),
                           "uniform_ess": uniform_ess.cpu().numpy(),
                           "own_incidence": own_incidence.cpu().numpy(),
                           "own_ess": own_ess.cpu().numpy()})

    model.ctx = ctx
    generated_states = blocked_rejuvenation(
        model, ix, smc.states, beta=1.0, steps=args.rejuvenation,
        generator=torch.Generator().manual_seed(args.seed + 3))
    metadata = pd.read_parquet(Path(BI) / "items.parquet").sort_values("item_id")
    item_category = model.cat_of.detach().cpu().numpy().astype(np.int64)
    if (len(item_category) != model.J or (item_category < 0).any()
            or (item_category >= int(data["n_cat"])).any()):
        raise RuntimeError("fitted item-category mapping is incompatible with model dimensions")
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
    if args.per_context_output:
        per_context = args.per_context_output.resolve()
        per_context.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            per_context, trips=trips, household=data["trip_user"][trips],
            observed_size=observed_sizes, factual_expected_size=factual_size.cpu().numpy(),
            factual_own_incidence=factual_own.cpu().numpy(), chosen_items=np.asarray(chosen_items),
            actions=np.asarray(args.actions),
            uniform_expected_size=np.stack([x["uniform_size"] for x in per_action]),
            uniform_ess_fraction=np.stack([x["uniform_ess"] for x in per_action]),
            own_incidence=np.stack([x["own_incidence"] for x in per_action]),
            own_ess_fraction=np.stack([x["own_ess"] for x in per_action]),
            generated_size=generated_sizes.reshape(ix.B, len(generated_states)))
    price_component = blob.get("supported_price_component")
    price_disabled = (
        blob.get("price_response_estimator")
        == "fixed_zero_after_failed_heldout_support"
        or (isinstance(price_component, dict)
            and price_component.get("level") == "disabled"))
    output = {
        "checkpoint": str(ckpt),
        "checkpoint_sha256": file_sha256(ckpt),
        "data_fingerprint_sha256": blob["data_fingerprint_sha256"],
        "trained_capabilities": blob["trained_capabilities"],
        "checkpoint_iteration": int(blob["iter"]),
        "best_iteration": int(blob["best_iteration"]),
        "trips": trips.tolist(),
        "split": args.split,
        "particles_per_trip": args.particles,
        "smc_levels": args.levels,
        "smc_seconds": smc_seconds,
        "smc_ess_min": float(smc.min_ess_fraction.min()),
        "smc_ess_median": float(smc.min_ess_fraction.median()),
        "factual_expected_size": float(factual_size.mean()),
        "observed_size_mean": float(observed_sizes.mean()),
        "price_counterfactual_capability": {
            "status": "not_available" if price_disabled else "model_scenario_only",
            "reason": (
                "held-out price evidence failed; price coefficients are fixed to zero"
                if price_disabled else
                "the fitted response supports model scenarios, not a causal price claim"),
            "price_response_estimator": blob.get("price_response_estimator"),
            "supported_price_component": price_component,
        },
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
    if args.panel_output:
        output["panel_output"] = str(panel_output)
        output["panel_sha256"] = file_sha256(panel_output)
    if args.per_context_output:
        output["per_context_output"] = str(per_context)
        output["per_context_sha256"] = file_sha256(per_context)
    args.output = args.output if args.output.is_absolute() else ROOT / args.output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(strict_json_dumps(output))
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
    print(strict_json_dumps(summary), end="")


if __name__ == "__main__":
    main()
