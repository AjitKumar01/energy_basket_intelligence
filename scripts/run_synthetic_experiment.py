#!/usr/bin/env python3
"""Run both synthetic audits used by the stakeholder demonstration."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "scripts" / "version4"
REPORTS = ROOT / "reports"


def run(command: list[str]) -> float:
    print("[synthetic-driver] " + " ".join(command), flush=True)
    tick = time.perf_counter()
    subprocess.run(command, cwd=ROOT, check=True)
    return time.perf_counter() - tick


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=73021)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    REPORTS.mkdir(exist_ok=True)
    exact = REPORTS / "synthetic_exact_certification.json"
    retailer = REPORTS / "synthetic_retailer_experiment.json"
    misspecified = REPORTS / "synthetic_retailer_misspecified.json"
    exact_command = [
        sys.executable, "-u", str(V4 / "audit_synthetic_interactions.py"),
        "--threads", str(args.threads), "--seed", str(args.seed + 11),
        "--output", str(exact),
    ]
    if args.profile == "smoke":
        exact_command += [
            "--items", "8", "--contexts", "3", "--categories", "2",
            "--nmax", "3", "--rank", "2", "--train", "500",
            "--validation", "150", "--test", "250", "--strengths", "0", "0.7",
            "--replicates", "1", "--steps", "30", "--eval-every", "5",
            "--patience", "4",
        ]
    seconds_exact = run(exact_command)
    retailer_command = [
        sys.executable, "-u", str(V4 / "audit_synthetic_retailer.py"),
        "--profile", args.profile, "--threads", str(args.threads),
        "--seed", str(args.seed), "--output", str(retailer),
    ]
    seconds_retailer = run(retailer_command)
    misspecified_command = [
        sys.executable, "-u", str(V4 / "audit_synthetic_retailer.py"),
        "--profile", args.profile, "--world", "misspecified",
        "--threads", str(args.threads), "--seed", str(args.seed + 101),
        "--output", str(misspecified),
    ]
    seconds_misspecified = run(misspecified_command)
    exact_result = json.loads(exact.read_text())
    retailer_result = json.loads(retailer.read_text())
    misspecified_result = json.loads(misspecified.read_text())
    manifest = {
        "schema": 1,
        "profile": args.profile,
        "seed": args.seed,
        "threads": args.threads,
        "exact_certification": str(exact.relative_to(ROOT)),
        "complete_retailer": str(retailer.relative_to(ROOT)),
        "misspecified_retailer": str(misspecified.relative_to(ROOT)),
        "runtime_seconds": {
            "exact_certification": seconds_exact,
            "complete_retailer": seconds_retailer,
            "misspecified_retailer": seconds_misspecified,
            "total": seconds_exact + seconds_retailer + seconds_misspecified,
        },
        "exact_support_baskets": exact_result["support_baskets"],
        "retailer_support_baskets": retailer_result["support_baskets"],
        "retailer_opportunities": retailer_result["opportunities"],
        "retailer_trips": retailer_result["trips"],
        "misspecified_retailer_trips": misspecified_result["trips"],
        "scope_warning": (
            "Synthetic results demonstrate recovery under known simulated truth; "
            "they do not replace held-out real-data or randomized-retailer evidence."),
    }
    destination = REPORTS / "synthetic_experiment_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[synthetic-driver] manifest={destination}", flush=True)


if __name__ == "__main__":
    main()
