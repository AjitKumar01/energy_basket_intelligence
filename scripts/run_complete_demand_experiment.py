#!/usr/bin/env python3
"""Run the predeclared complete-demand synthetic experiment matrix."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "scripts" / "version4" / "audit_synthetic_complete_demand.py"
REPORTS = ROOT / "reports"


def execute(command: list[str]) -> float:
    print("[complete-demand-driver] " + " ".join(command), flush=True)
    tick = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    return time.perf_counter() - tick


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=104017)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--output", type=Path,
                        default=REPORTS / "synthetic_complete_demand_manifest.json")
    args = parser.parse_args()
    if args.replicates < 1:
        parser.error("--replicates must be positive")
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Strong-signal replicates test reproducible recovery.  Weak and null worlds test
    # that the validation gate declines unsupported interaction capacity.
    design = [("linked_interaction", args.seed + replicate)
              for replicate in range(args.replicates)]
    design.extend([
        ("weak_interaction", args.seed + 1000),
        ("null_interaction", args.seed + 2000),
    ])
    records = []
    started = time.perf_counter()
    for world, seed in design:
        output = REPORTS / f"synthetic_complete_demand_{world}_seed{seed}.json"
        command = [
            sys.executable, "-u", str(AUDIT), "--profile", args.profile,
            "--world", world, "--threads", str(args.threads),
            "--seed", str(seed), "--output", str(output),
        ]
        runtime = execute(command)
        report = json.loads(output.read_text())
        acceptance = report["training"]["linked_interaction"][
            "interaction_acceptance"]
        records.append({
            "world": world,
            "seed": seed,
            "report": str(output.relative_to(ROOT)),
            "runtime_seconds": runtime,
            "interaction_accepted": bool(acceptance["accepted"]),
            "validation_interaction_lower_95": float(
                acceptance["paired_validation_lower_95"]),
            "test_interaction_gain": report["evaluation"]["paired_gains"][
                "interaction_minus_additive"],
            "test_linked_minus_independent": report["evaluation"]["paired_gains"][
                "linked_minus_independent"],
            "kernel_correlation": float(report["evaluation"]["parameter_recovery"][
                "interaction_kernel_correlation"]),
            "linked_counterfactual_mae": report["counterfactual"]["linked_mae"],
            "independent_counterfactual_mae": report["counterfactual"][
                "independent_mae"],
            "normalization_error": max(report["evaluation"][
                "maximum_probability_normalization_error"].values()),
        })

    strong = [row for row in records if row["world"] == "linked_interaction"]
    weak = next(row for row in records if row["world"] == "weak_interaction")
    null = next(row for row in records if row["world"] == "null_interaction")
    evidence_checks = {
        "all_strong_interactions_accepted": all(
            row["interaction_accepted"] for row in strong),
        "all_strong_test_gains_positive_at_95": all(
            row["test_interaction_gain"]["interval_95"][0] > 0
            for row in strong),
        "all_strong_kernels_recovered": all(
            row["kernel_correlation"] > 0.95 for row in strong),
        "weak_interaction_rejected_when_underpowered": not weak["interaction_accepted"],
        "null_interaction_rejected": not null["interaction_accepted"],
        "linked_beats_independent_in_every_world": all(
            row["test_linked_minus_independent"]["interval_95"][0] > 0
            for row in records),
        "all_joint_probabilities_normalized": all(
            row["normalization_error"] < 1e-12 for row in records),
        "linked_counterfactual_demand_better_than_independent": all(
            row["linked_counterfactual_mae"]["expected_distinct_items"]
            < row["independent_counterfactual_mae"]["expected_distinct_items"]
            for row in records),
    }
    if args.profile == "smoke":
        # Smoke runs validate plumbing on deliberately underpowered samples.  Statistical
        # recovery gates belong to the full profile and must not be weakened merely to
        # make a small run green.
        checks = {
            "all_joint_probabilities_normalized": evidence_checks[
                "all_joint_probabilities_normalized"],
            "all_reports_completed": len(records) == args.replicates + 2,
        }
    else:
        checks = evidence_checks
    failing = [name for name, passed in checks.items() if not passed]
    manifest = {
        "schema": 1,
        "experiment": "complete-demand joint synthetic matrix",
        "profile": args.profile,
        "base_seed": args.seed,
        "strong_replicates": args.replicates,
        "records": records,
        "predeclared_checks": checks,
        "failing_checks": failing,
        "runtime_seconds": time.perf_counter() - started,
        "scope_warning": (
            "Passing exact tests establishes correctness and recovery in the declared "
            "synthetic worlds, not causal validity on retailer data."),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[complete-demand-driver] manifest={args.output}", flush=True)
    if failing:
        raise SystemExit("predeclared synthetic checks failed: " + ", ".join(failing))


if __name__ == "__main__":
    main()
