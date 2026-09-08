#!/usr/bin/env python3
"""Decompose Version-4 size log odds into catalogue pressure and rho_0 increments."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch

from checkpoint_io import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from interaction_particles import differentiable_log_size_beta0
from provenance import file_sha256, strict_json_dumps


torch.set_default_dtype(torch.float64)


@torch.no_grad()
def beta0_components(model, batcher, trips: np.ndarray, chunk: int):
    h, log_probability = [], []
    rho = model.rho_0()[1:].detach().cpu().numpy()
    for start in range(0, len(trips), chunk):
        ix, ctx, _line_ctx, house, *_ = batcher.make(trips[start:start + chunk])
        model.house, model.ctx = house, ctx
        log_mass = differentiable_log_size_beta0(model, ix).cpu().numpy()
        h.append(log_mass + rho[None, :])
        log_probability.append(
            log_mass - np.logaddexp.reduce(log_mass, axis=1)[:, None])
    return np.concatenate(h), np.concatenate(log_probability)


def interval_summary(values: np.ndarray) -> dict:
    # values are increments from n to n+1 and therefore indexed by starting size n.
    bands = ((1, 4), (5, 10), (11, 20), (21, 40), (41, 80), (81, 119))
    answer = {}
    for lo, hi in bands:
        section = values[:, lo - 1:hi]
        answer[f"{lo}:{hi + 1}"] = {
            "mean": float(section.mean()),
            "p95": float(np.quantile(section, 0.95)),
            "maximum": float(section.max()),
        }
    return answer


def panel_summary(name: str, selected: np.ndarray, observed: np.ndarray,
                  child_log_probability: np.ndarray,
                  child_h: np.ndarray, child_beta0_h: np.ndarray,
                  parent_h: np.ndarray, child_rho: np.ndarray,
                  parent_rho: np.ndarray) -> dict:
    log_odds = np.diff(child_log_probability[selected], axis=1)
    h_increment = np.diff(child_h[selected], axis=1)
    rho_increment = np.diff(child_rho)[None, :]
    gram_increment = np.diff(
        child_h[selected] - child_beta0_h[selected], axis=1)
    nuisance_increment = np.diff(
        child_beta0_h[selected] - parent_h[selected], axis=1)
    rho_change_increment = np.diff(child_rho - parent_rho)[None, :]
    tail = log_odds[:, 19:]
    positive_tail = tail > 0
    return {
        "name": name, "contexts": int(len(selected)),
        "observed_size_mean": float(observed[selected].mean()),
        "model_expected_size_mean": float(np.mean(
            np.exp(child_log_probability[selected]) @
            np.arange(1, child_log_probability.shape[1] + 1))),
        "contexts_with_any_positive_size_log_odds_after_20": int(
            positive_tail.any(axis=1).sum()),
        "maximum_size_log_odds_after_20": float(tail.max()),
        "argmax_starting_size_after_20": int(np.unravel_index(
            np.argmax(tail), tail.shape)[1] + 20),
        "size_log_odds": interval_summary(log_odds),
        "catalogue_pressure_increment": interval_summary(h_increment),
        "gram_contribution_to_pressure_increment": interval_summary(gram_increment),
        "non_gram_child_minus_parent_pressure_increment": interval_summary(
            nuisance_increment),
        "rho0_increment": interval_summary(np.broadcast_to(
            rho_increment, log_odds.shape)),
        "child_minus_parent_rho0_increment": interval_summary(np.broadcast_to(
            rho_change_increment, log_odds.shape)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--confirmed-panel", type=Path,
                        default=Path("reports/population_size_confirm_per_trip.npz"))
    parser.add_argument("--contexts-per-panel", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/size_phase_diagnostic.json"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = build()
    parent_path = args.parent if args.parent.is_absolute() else ROOT / args.parent
    child_path = args.child if args.child.is_absolute() else ROOT / args.child
    panel_path = (args.confirmed_panel if args.confirmed_panel.is_absolute()
                  else ROOT / args.confirmed_panel)
    parent, parent_blob, meta = load_checkpoint(parent_path, data)
    child, child_blob, child_meta = load_checkpoint(child_path, data)
    if int(meta["nmax"]) != int(child_meta["nmax"]):
        raise ValueError("parent and child supports differ")
    panel = np.load(panel_path)
    trips = np.asarray(panel["trips"], dtype=np.int64)
    observed = np.asarray(panel["observed"], dtype=np.int64)
    child_log_probability = np.asarray(panel["confirm_log_probability"], dtype=np.float64)
    child_log_probability -= np.logaddexp.reduce(
        child_log_probability, axis=1)[:, None]
    if args.contexts_per_panel * 2 > len(trips):
        raise ValueError("confirmed panel is too small for two diagnostic groups")
    size_axis = np.arange(1, child.nmax + 1, dtype=np.float64)
    expected = np.exp(child_log_probability) @ size_axis
    order = np.argsort(expected)
    lower_risk = order[:args.contexts_per_panel]
    high = order[-args.contexts_per_panel:]
    batcher = Batcher(
        data, Features(int(data["n_item"]), int(data["n_store"]), 712,
                       include_recency=False), child.nmax, include_recency=False)
    parent_h, _ = beta0_components(parent, batcher, trips, args.chunk)
    child_beta0_h, _ = beta0_components(child, batcher, trips, args.chunk)
    child_rho = child.rho_0()[1:].detach().cpu().numpy()
    parent_rho = parent.rho_0()[1:].detach().cpu().numpy()
    # The confirmed probability determines H up to one context-specific additive constant;
    # size increments, which are the identified diagnostic, remove that constant.
    child_h = child_log_probability + child_rho[None, :]
    report = {
        "method": "identified_size_log_odds_decomposition",
        "identity": "log P(N=n+1)/P(N=n) = Delta H_x(n) - Delta rho_0(n)",
        "parent": str(parent_path), "parent_sha256": file_sha256(parent_path),
        "child": str(child_path), "child_sha256": file_sha256(child_path),
        "parent_iteration": int(parent_blob["iter"]),
        "child_iteration": int(child_blob["iter"]),
        "confirmed_panel": str(panel_path), "panel_contexts": int(len(trips)),
        "selection": (
            "highest and lowest child expected-size contexts inside the existing "
            "independently confirmed high-risk panel"),
        "high_expected_size": panel_summary(
            "high_expected_size", high, observed, child_log_probability,
            child_h, child_beta0_h, parent_h, child_rho, parent_rho),
        "lower_expected_size_within_confirmed_panel": panel_summary(
            "lower_expected_size_within_confirmed_panel", lower_risk, observed,
            child_log_probability,
            child_h, child_beta0_h, parent_h, child_rho, parent_rho),
        "per_context_output": str(args.output.with_suffix(".npz")),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output.with_suffix(".npz"), trips=trips, observed=observed,
        expected_size=expected, high_index=high, lower_risk_index=lower_risk,
        child_log_probability=child_log_probability,
        child_h_increment=np.diff(child_h, axis=1),
        child_beta0_h_increment=np.diff(child_beta0_h, axis=1),
        parent_h_increment=np.diff(parent_h, axis=1),
        child_rho_increment=np.diff(child_rho),
        parent_rho_increment=np.diff(parent_rho))
    report["per_context_output"] = str(output.with_suffix(".npz"))
    output.write_text(strict_json_dumps(report))
    print(strict_json_dumps(report), end="")


if __name__ == "__main__":
    main()
