#!/usr/bin/env python3
"""Verify final-model price responses by quadrature and independent particle banks."""
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

from checkpoint_io import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from interaction_particles import (direct_interaction_particles,
                                   rao_blackwell_expected_size,
                                   reweight_additive_counterfactual)
from pipeline_support import install_quadrature, smolyak_rule
from price_response import changed_price_context
from provenance import file_sha256, strict_json_dumps


torch.set_default_dtype(torch.float64)


def action_change(name, ix, line_item, line_trip):
    if name == "uniform_0.8":
        return torch.full(ix.item.shape, math.log(.8), dtype=torch.float64,
                          device=ix.item.device)
    if name == "uniform_1.2":
        return torch.full(ix.item.shape, math.log(1.2), dtype=torch.float64,
                          device=ix.item.device)
    change = torch.zeros_like(ix.item, dtype=torch.float64)
    if name == "observed_sku_0.8":
        for b in range(ix.B):
            item = int(torch.unique(line_item[line_trip == b]).min())
            slot = torch.nonzero((ix.item_trip == b) & (ix.item == item), as_tuple=True)[0]
            if not len(slot):
                raise RuntimeError("observed item absent from declared assortment")
            change[int(slot[0])] = math.log(.8)
        return change
    if name == "catalogue_prefix_bundle_0.8":
        for b in range(ix.B):
            slots = torch.nonzero(ix.item_trip == b, as_tuple=True)[0][:5]
            change[slots] = math.log(.8)
        return change
    raise ValueError(f"unknown action {name}")


def replicate_fidelity(estimates, reference, spec):
    """Equivalence interval for independent estimates versus a deterministic target."""
    differences = np.asarray(estimates, dtype=np.float64) - float(reference)
    if differences.ndim != 1 or len(differences) < 2 or not np.isfinite(differences).all():
        raise ValueError("at least two finite independent replicate estimates are required")
    mc_se = float(differences.std(ddof=1) / math.sqrt(len(differences)))
    tolerance = float(spec["agreement_absolute_items"])
    radius = float(spec["agreement_combined_mc_se_multiplier"]) * mc_se
    error = float(abs(differences.mean()))
    if error + radius <= tolerance:
        status = "passed"
    elif max(0.0, error - radius) > tolerance:
        status = "failed"
    else:
        status = "inconclusive"
    return {"mean_minus_reference": float(differences.mean()),
            "replicate_mc_se": mc_se, "equivalence_radius": radius,
            "tolerance": tolerance, "status": status,
            "passed": status == "passed"}


@torch.no_grad()
def quadrature_means(model, batcher, trips, rank, level, actions, chunk):
    install_quadrature(model, smolyak_rule(model, rank, level))
    factual, changed = [], {name: [] for name in actions}
    axis = torch.arange(1, model.nmax + 1, dtype=model.phi.dtype)
    for start in range(0, len(trips), chunk):
        sub = trips[start:start + chunk]
        ix, ctx, _lc, house, line_item, line_trip, *_ = batcher.make(sub)
        model.house, model.ctx = house, ctx
        _, probability = model.log_Z(ix, drop_empty=True, return_size=True)
        factual.append((probability @ axis).cpu().numpy())
        for name in actions:
            change = action_change(name, ix, line_item, line_trip)
            model.ctx = changed_price_context(ctx, ix.item_trip, change)
            _, probability = model.log_Z(ix, drop_empty=True, return_size=True)
            changed[name].append((probability @ axis).cpu().numpy())
        model.ctx = ctx
        print(f"[price-numerics] quadrature q{level} {min(start + chunk, len(trips))}/"
              f"{len(trips)}", flush=True)
    return np.concatenate(factual), {name: np.concatenate(value)
                                     for name, value in changed.items()}


@torch.no_grad()
def particle_comparison(model, batcher, trips, actions, budgets, seeds, chunk):
    shape = (len(budgets), len(seeds), len(actions), len(trips))
    direct = np.full(shape, np.nan); reweighted = np.empty(shape)
    factual_mean = np.empty((len(budgets), len(seeds), len(trips)))
    factual_ess = np.empty_like(factual_mean)
    direct_ess = np.full(shape, np.nan); reweight_ess = np.empty(shape)
    for pi, particles in enumerate(budgets):
        for si, seed in enumerate(seeds):
            for start in range(0, len(trips), chunk):
                sub = trips[start:start + chunk]
                ix, ctx, _lc, house, line_item, line_trip, *_ = batcher.make(sub)
                model.house, model.ctx = house, ctx
                factual_b = model.b_flat(ix).clone()
                factual = direct_interaction_particles(
                    model, ix, particles,
                    torch.Generator().manual_seed(seed + 1000003 * pi + start))
                sl = slice(start, start + len(sub))
                factual_mean[pi, si, sl] = rao_blackwell_expected_size(
                    model, ix, factual.states, factual.log_weights).cpu().numpy()
                factual_ess[pi, si, sl] = factual.ess_fraction.cpu().numpy()
                for ai, name in enumerate(actions):
                    change = action_change(name, ix, line_item, line_trip)
                    target_ctx = changed_price_context(ctx, ix.item_trip, change)
                    model.ctx = target_ctx
                    target_b = model.b_flat(ix).clone()
                    rw = reweight_additive_counterfactual(
                        model, ix, factual, factual_b, target_b)
                    reweighted[pi, si, ai, sl] = rao_blackwell_expected_size(
                        model, ix, factual.states, rw.log_weights).cpu().numpy()
                    if pi == len(budgets)-1 and ai == 0:
                        target = direct_interaction_particles(
                            model, ix, particles,
                            torch.Generator().manual_seed(
                                seed + 1000003 * pi + 170003 * (ai + 1) + start))
                        direct[pi, si, ai, sl] = rao_blackwell_expected_size(
                            model, ix, target.states, target.log_weights).cpu().numpy()
                        direct_ess[pi, si, ai, sl] = target.ess_fraction.cpu().numpy()
                    reweight_ess[pi, si, ai, sl] = rw.ess_fraction.cpu().numpy()
                    model.ctx = ctx
                print(f"[price-numerics] particles={particles} seed={seed} "
                      f"contexts={min(start + chunk, len(trips))}/{len(trips)}", flush=True)
    return factual_mean, factual_ess, direct, reweighted, direct_ess, reweight_ess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-context-output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=16)
    args = parser.parse_args()
    started = time.monotonic()
    torch.set_num_threads(args.threads)
    run_dir, protocol_path = args.run_dir.resolve(), args.protocol.resolve()
    protocol = json.loads(protocol_path.read_text())
    spec = protocol["numerical_price"]
    checkpoint = run_dir / "artifacts/candidate_rank1.pt"
    validation_scores = run_dir / "reports/likelihood_validation_per_trip.npz"
    data = build()
    model, blob, meta = load_checkpoint(
        checkpoint, data,
        required_capabilities=("conditional_nonempty_incidence", "gram_interactions"))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    with np.load(validation_scores) as saved:
        trips = saved["trips"][:int(spec["contexts"])].copy()
    if len(trips) != int(spec["contexts"]) or not np.all(data["trip_split"][trips] == 1):
        raise RuntimeError("locked numerical panel is incomplete or not validation")
    actions = ["uniform_0.8", "uniform_1.2", "observed_sku_0.8",
               "catalogue_prefix_bundle_0.8"]
    budgets = [int(x) for x in spec["particle_budgets"]]
    seeds = [int(x) for x in spec["replicate_seeds"]]
    batcher = Batcher(data, Features(int(data["n_item"]), int(data["n_store"]), 712,
                                    include_recency=False), int(meta["nmax"]),
                      include_recency=False)
    rank = int((torch.linalg.svdvals(model.phi) >
                torch.linalg.svdvals(model.phi)[0] * 1e-10).sum())
    q7_factual, q7_action = quadrature_means(
        model, batcher, trips, rank, rank + 2, actions, args.chunk)
    q8_factual, q8_action = quadrature_means(
        model, batcher, trips, rank, rank + 3, actions, args.chunk)
    factual, factual_ess, direct, reweighted, direct_ess, reweight_ess = particle_comparison(
        model, batcher, trips, actions, budgets, seeds, args.chunk)
    per_path = args.per_context_output.resolve()
    per_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(per_path, trips=trips, household=data["trip_user"][trips],
                        actions=np.asarray(actions), budgets=np.asarray(budgets),
                        seeds=np.asarray(seeds), q7_factual=q7_factual,
                        q8_factual=q8_factual,
                        q7_action=np.stack([q7_action[x] for x in actions]),
                        q8_action=np.stack([q8_action[x] for x in actions]),
                        particle_factual=factual, factual_ess_fraction=factual_ess,
                        particle_direct=direct, particle_reweighted=reweighted,
                        direct_ess_fraction=direct_ess,
                        reweight_ess_fraction=reweight_ess)
    rows, gates = [], {}
    for ai, name in enumerate(actions):
        q7_response = q7_action[name] - q7_factual
        q8_response = q8_action[name] - q8_factual
        quadrature_response_gap = float(np.max(np.abs(q8_response - q7_response)))
        quadrature_factual_gap = float(np.max(np.abs(q8_factual - q7_factual)))
        by_budget = []
        for pi, particles in enumerate(budgets):
            reweighted_target = reweighted[pi, :, ai].mean(-1)
            factual_target = factual[pi].mean(-1)
            reweighted_response = reweighted_target - factual_target
            reweight_reference = replicate_fidelity(
                reweighted_response, q8_response.mean(), spec)
            direct_available = bool(np.isfinite(direct[pi, :, ai]).all())
            if direct_available:
                direct_target = direct[pi, :, ai].mean(-1)
                direct_response = direct_target - factual_target
                agreement = replicate_fidelity(
                    direct_target - reweighted_target, 0.0, spec)
                direct_reference = replicate_fidelity(
                    direct_response, q8_response.mean(), spec)
            else:
                agreement = {"status": "not_assessed", "passed": False}
                direct_reference = {"status": "not_assessed", "passed": False}
            ess_values = [factual_ess[pi].min(), reweight_ess[pi, :, ai].min()]
            if direct_available:
                ess_values.append(direct_ess[pi, :, ai].min())
            minimum_fraction = float(min(ess_values))
            minimum_absolute = minimum_fraction * particles
            by_budget.append({"particles": particles,
                              "direct_stress_check": direct_available,
                              "direct_minus_reweight_mean": agreement.get("mean_minus_reference"),
                              "replicate_mc_se": agreement.get("replicate_mc_se"),
                              "agreement_tolerance": agreement.get("tolerance"),
                              "agreement_status": agreement["status"],
                              "agreement_passed": agreement["passed"],
                              "q8_response_reference": float(q8_response.mean()),
                              "direct_response_fidelity": direct_reference,
                              "reweight_response_fidelity": reweight_reference,
                              "reference_fidelity_passed": (
                                  reweight_reference["passed"] and
                                  (not direct_available or direct_reference["passed"])),
                              "minimum_ess_fraction": minimum_fraction,
                              "minimum_absolute_ess": minimum_absolute,
                              "ess_passed": (minimum_fraction >= spec["minimum_fractional_ess"]
                                  and minimum_absolute >= spec["minimum_absolute_ess_at_64"]
                                  * particles / 64)})
        primary_status = by_budget[-1]["reweight_response_fidelity"]["status"]
        hard_failure = (quadrature_response_gap > float(spec["agreement_absolute_items"])
                        or not all(row["ess_passed"] for row in by_budget)
                        or primary_status == "failed")
        action_status = ("failed" if hard_failure else "inconclusive"
                         if primary_status == "inconclusive" else "passed")
        rows.append({"action": name,
                     "q7_response_mean": float(q7_response.mean()),
                     "q8_response_mean": float(q8_response.mean()),
                     "q7_q8_response_difference": float(q7_response.mean()-q8_response.mean()),
                     "maximum_per_context_q7_q8_response_gap": quadrature_response_gap,
                     "maximum_per_context_q7_q8_factual_gap": quadrature_factual_gap,
                     "particle_budgets": by_budget, "status": action_status,
                     "passed": action_status == "passed"})
        gates[name] = action_status == "passed"
    action_statuses = {row["action"]: row["status"] for row in rows}
    overall_status = ("failed" if "failed" in action_statuses.values() else
                      "inconclusive" if "inconclusive" in action_statuses.values()
                      else "passed")
    output = {
        "status": overall_status,
        "checkpoint": str(checkpoint), "checkpoint_sha256": file_sha256(checkpoint),
        "data_fingerprint_sha256": blob["data_fingerprint_sha256"],
        "protocol": str(protocol_path), "protocol_sha256": file_sha256(protocol_path),
        "panel": "first 128 saved validation-likelihood contexts",
        "contexts": len(trips), "active_rank": rank, "actions": rows,
        "action_statuses": action_statuses,
        "gates": gates, "per_context_output": str(per_path),
        "per_context_sha256": file_sha256(per_path),
        "runtime_seconds": time.monotonic() - started,
        "interpretation": ("Quadrature convergence, highest-budget reweighted-reference "
            "equivalence and ESS gate the fixed-support model-conditional response. The "
            "highest-budget independent direct estimate is a deliberately non-gating stress "
            "diagnostic. None of these checks identifies a causal price effect."),
    }
    out_path = args.output.resolve(); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(strict_json_dumps(output)); print(strict_json_dumps(output), end="")
    if output["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
