#!/usr/bin/env python3
"""Paired complete-support likelihood: fixed-rank model versus its Phi=0 parent."""
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
from evalall import load_any
from features import Features
from fit import Batcher
from interaction_particles import differentiable_logz_beta0
from ragged import RaggedModel, smolyak_grid


torch.set_default_dtype(torch.float64)


def load_version4_checkpoint(path, data):
    """Load both artifact-backed modern checkpoints and audited format-2 runs."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    if "artifact" in blob.get("config", {}):
        return load_checkpoint(path, data)
    if blob.get("format") != 2 or "model" not in blob or "data" not in blob:
        raise RuntimeError(f"unsupported checkpoint format: {path}")
    state, meta = blob["model"], blob["data"]
    required = ("nmax", "R", "n_item", "n_cat")
    if any(key not in meta for key in required):
        raise RuntimeError(f"format-2 checkpoint lacks support metadata: {path}")
    model = RaggedModel(
        int(data["n_item"]), int(data["n_user"]), int(data["n_cat"]),
        K=int(state["alpha"].shape[1]), Kz=int(state["phi"].shape[1]),
        nmax=int(meta["nmax"]), R=int(meta["R"]), S=int(data["n_store"]),
        Kp=int(state["gamma"].shape[1]), phi_init=0.0)
    load_any(path, model, int(data["n_item"]), data)
    model.double().eval()
    return model, blob, meta


def rule(model, rank, level):
    active, weights = smolyak_grid(rank, level)
    nodes = torch.zeros(len(weights), model.Kz, dtype=model.phi.dtype)
    nodes[:, :rank] = active
    return nodes, weights


@torch.no_grad()
def exact_parent(model, batcher, trips, chunk):
    result = []
    for start in range(0, len(trips), chunk):
        ix, ctx, line_ctx, house, li, lt, lc, _ = batcher.make(
            trips[start:start + chunk])
        model.house, model.ctx = house, ctx
        result.append((model.energy(li, lt, lc, ix.B, line_ctx)
                       - differentiable_logz_beta0(model, ix)).cpu())
    return torch.cat(result).numpy()


@torch.no_grad()
def interaction_values(model, batcher, trips, quadrature, chunk):
    model.quad = quadrature; model.quad_a = None
    result, cancellation = [], 0.0
    for start in range(0, len(trips), chunk):
        ix, ctx, line_ctx, house, li, lt, lc, _ = batcher.make(
            trips[start:start + chunk])
        model.house, model.ctx = house, ctx
        result.append((model.energy(li, lt, lc, ix.B, line_ctx)
                       - model.log_Z(ix, drop_empty=True)).cpu())
        cancellation = max(cancellation, float(model._last_quad_log_condition.max()))
    return torch.cat(result).numpy(), cancellation


def summary(delta):
    se = float(delta.std(ddof=1) / math.sqrt(len(delta)))
    mean = float(delta.mean())
    return {"trips": int(len(delta)), "mean": mean, "standard_error": se,
            "95_interval": [mean - 1.96 * se, mean + 1.96 * se],
            "median": float(np.median(delta))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--trips", type=int, default=384)
    parser.add_argument("--rank", type=int, default=0,
                        help="expected rank; <=0 infers from checkpoint")
    parser.add_argument("--target-level", type=int, default=0,
                        help="<=0 uses active_rank+2")
    parser.add_argument("--audit-trips", type=int, default=8)
    parser.add_argument("--maximum-audit-error-bound", type=float, default=0.0,
                        help=("maximum |q-target minus q-audit| 95%% upper bound; "
                              "0 reports the diagnostic without enforcing it"))
    parser.add_argument("--require-certified-gain", action="store_true",
                        help=("require the child-minus-parent lower 95%% bound, after "
                              "the quadrature error allowance, to be positive"))
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("out/v3_rank8_parent_likelihood.json"))
    parser.add_argument("--per-trip-output", type=Path, default=None,
                        help=("optional NPZ containing the ordered manifest and raw "
                              "paired scores; defaults beside --output"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = build()
    parent_path = args.parent if args.parent.is_absolute() else ROOT / args.parent
    child_path = args.child if args.child.is_absolute() else ROOT / args.child
    parent, parent_blob, meta = load_version4_checkpoint(parent_path, data)
    child, child_blob, child_meta = load_version4_checkpoint(child_path, data)
    if float(parent.phi.detach().abs().max()) != 0.0:
        raise RuntimeError("parent must have Phi=0")
    singular = torch.linalg.svdvals(child.phi)
    active_rank = int((singular > singular[0] * 1e-10).sum())
    if args.rank > 0 and active_rank != args.rank:
        raise RuntimeError(
            f"child active rank is {active_rank}, not {args.rank}")
    if int(meta["nmax"]) != int(child_meta["nmax"]):
        raise RuntimeError("parent and child supports differ")
    batcher = Batcher(data, Features(int(data["n_item"]), int(data["n_store"]), 712),
                      int(meta["nmax"]))
    split = {"validation": 1, "test": 2}[args.split]
    population = np.flatnonzero((data["trip_split"] == split)
                                & (data["trip_nlines"] <= int(meta["nmax"])))
    trips = population[np.random.default_rng(args.seed).permutation(len(population))[
        :args.trips]]
    parent_value = exact_parent(parent, batcher, trips, args.chunk)
    target_level = (args.target_level if args.target_level > 0
                    else active_rank + 2)
    low_level, audit_level = target_level - 1, target_level + 1
    low_value, cancel_low = interaction_values(
        child, batcher, trips, rule(child, active_rank, low_level), args.chunk)
    target_value, cancel_target = interaction_values(
        child, batcher, trips, rule(child, active_rank, target_level), args.chunk)
    n_audit = min(args.audit_trips, len(trips))
    audit_value, cancel_audit = interaction_values(
        child, batcher, trips[:n_audit],
        rule(child, active_rank, audit_level), min(args.chunk, n_audit))
    lines = (data["line_ptr"][trips + 1] - data["line_ptr"][trips]).astype(
        np.int64, copy=False)
    parent_summary = summary(parent_value)
    child_summary = summary(target_value)
    gain_summary = summary(target_value - parent_value)
    audit_summary = summary(target_value[:n_audit] - audit_value)
    audit_bound = (abs(audit_summary["mean"])
                   + 1.96 * audit_summary["standard_error"])
    certified_gain_lower = gain_summary["95_interval"][0] - audit_bound
    numerical_pass = (args.maximum_audit_error_bound <= 0.0
                      or audit_bound <= args.maximum_audit_error_bound)
    gain_pass = (not args.require_certified_gain
                 or certified_gain_lower > 0.0)
    result = {
        "parent": str(parent_path), "parent_iteration": int(parent_blob["iter"]),
        "child": str(child_path), "child_iteration": int(child_blob["iter"]),
        "split": args.split, "complete_support": f"1..{int(meta['nmax'])}",
        "active_rank": active_rank,
        "levels": {"low": low_level, "target": target_level,
                   "audit": audit_level},
        "exact_parent_log_likelihood": parent_summary,
        "target_child_log_likelihood": child_summary,
        "target_child_minus_exact_parent": gain_summary,
        "low_minus_target": summary(low_value - target_value),
        "target_minus_audit": audit_summary,
        "numerical_certification": {
            "audit_trips": n_audit,
            "absolute_audit_error_95_upper_bound": audit_bound,
            "maximum_allowed_audit_error_bound": args.maximum_audit_error_bound,
            "target_level_numerically_accepted": numerical_pass,
            "child_gain_lower_95_after_audit_allowance": certified_gain_lower,
            "certified_positive_gain_required": args.require_certified_gain,
            "certified_positive_gain": certified_gain_lower > 0.0,
            "passed": bool(numerical_pass and gain_pass),
        },
        "max_log_cancellation": {
            "low": cancel_low, "target": cancel_target,
            "audit": cancel_audit},
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    per_trip_output = args.per_trip_output
    if per_trip_output is None:
        per_trip_output = output.with_name(output.stem + "_per_trip.npz")
    elif not per_trip_output.is_absolute():
        per_trip_output = ROOT / per_trip_output
    np.savez_compressed(
        per_trip_output, trips=trips, lines=lines,
        exact_parent=parent_value, target_child=target_value,
        low_child=low_value, audit_trips=trips[:n_audit],
        audit_child=audit_value)
    result["per_trip_output"] = str(per_trip_output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["numerical_certification"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
