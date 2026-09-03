#!/usr/bin/env python3
"""Train and evaluate all verified external baselines on the locked test manifest."""
from __future__ import annotations

import argparse
import fcntl
import os
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "scripts" / "version4"
PY = sys.executable
ALL_MODELS = ("multinomial", "bernoulli", "dpp", "ndpp", "shopper")


def parse_models(raw: str):
    models = tuple(part.strip().lower() for part in raw.split(",") if part.strip())
    if not models:
        raise argparse.ArgumentTypeError("--models must select at least one baseline")
    unknown = sorted(set(models) - set(ALL_MODELS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown baseline(s): {', '.join(unknown)}")
    if len(models) != len(set(models)):
        raise argparse.ArgumentTypeError("--models contains a duplicate baseline")
    return models


def acquire_single_runner_lock(profile: str):
    """Prevent concurrent baseline drivers from writing the same checkpoints."""
    lock_path = ROOT / "artifacts" / f"baselines_{profile}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    stream = lock_path.open("a+")
    try:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise SystemExit(
            f"another {profile} baseline driver holds {lock_path}") from error
    stream.seek(0)
    stream.truncate()
    stream.write(f"pid={os.getpid()}\n")
    stream.flush()
    return stream


def run(command, environment, dry_run=False):
    print("[baselines] " + shlex.join(map(str, command)), flush=True)
    if dry_run:
        return
    completed = subprocess.run(
        list(map(str, command)), cwd=ROOT, env=environment, check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("converged", "published-1000", "smoke"),
                        default="converged")
    parser.add_argument("--skip-training", action="store_true",
                        help="score already-created checkpoints")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-training", action="store_true",
                        help="resume a fresh lineage after an interruption")
    parser.add_argument("--maximum-updates", type=int, default=60000,
                        help="fail-closed safety ceiling for the converged profile")
    parser.add_argument("--shopper-orders", type=int, default=8192,
                        help="ordering samples used for final SHOPPER set likelihood")
    parser.add_argument(
        "--models", type=parse_models, default=ALL_MODELS,
        help=("comma-separated baseline subset; for the external comparison use "
              "bernoulli,dpp,ndpp"))
    args = parser.parse_args()
    runner_lock = None if args.dry_run else acquire_single_runner_lock(args.profile)
    environment = os.environ.copy()
    environment["V3_AFFINITY"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT / "artifacts" / "native" / "lib"), str(V4),
         environment.get("PYTHONPATH", "")])
    smoke = args.profile == "smoke"
    converged = args.profile == "converged"
    iterations = 2 if smoke else (args.maximum_updates if converged else 1000)
    tag = ("pipeline_smoke" if smoke else
           ("pipeline_converged" if converged else "pipeline1000"))
    eval_every = 1 if smoke else (500 if converged else 200)
    if not args.skip_training:
        for model in args.models:
            command = [
                PY, "-u", V4 / "train_baseline_verified.py",
                "--model", model, "--tag", tag,
                "--iters", iterations, "--batch", 2 if smoke else 24,
                "--lr", 0.002, "--R", 120, "--nmax", 120,
                "--eval-every", eval_every,
                "--n-val", 16 if smoke else 512,
                "--eval-chunk", 2 if smoke else 8,
                "--eval-orders", 8 if smoke else 512,
            ]
            # These rank/cardinality kernels are small enough that the runtime default of
            # 15 threads spends more time at OpenMP barriers than doing arithmetic.
            # A frozen benchmark on the reference 15-core CPU selects four for all three
            # exact-normalizer families without changing their arithmetic.
            command.extend(["--threads", 4])
            if converged:
                command.extend([
                    "--scheduler", "plateau", "--require-convergence",
                    "--plateau-patience", 3,
                    "--convergence-delta", 0.002,
                    "--convergence-patience", 8, "--floor-patience", 4,
                    "--minimum-epochs", 2.0, "--lr-floor", 0.02,
                    "--train-orders", 8,
                ])
            elif smoke:
                # Exercise the complete certificate/checkpoint path in two updates.
                # These deliberately relaxed settings have no statistical meaning.
                command.extend([
                    "--scheduler", "plateau", "--require-convergence",
                    "--plateau-patience", 0, "--convergence-delta", 100,
                    "--convergence-patience", 1, "--floor-patience", 1,
                    "--minimum-epochs", 0, "--lr-floor", 1,
                ])
            if args.resume_training:
                command.append("--resume")
            run(command, environment, args.dry_run)
    audit = [
        PY, "-u", V4 / "audit_other_baselines_fair.py",
        "--full-per-trip", "reports/likelihood_test_per_trip.npz",
        "--full-key", "target_child", "--split", "test",
        "--iteration", iterations, "--baseline-tag", tag,
        "--shopper-orders", 8 if smoke else args.shopper_orders,
        "--maximum-trips", 16 if smoke else 0,
        "--models", ",".join(args.models),
    ]
    output = ("reports/baselines_converged" if converged else
              ("reports/baselines_smoke" if smoke else "reports/baselines"))
    if args.models != ALL_MODELS:
        output += "_" + "_".join(args.models)
    audit.extend(["--output", output])
    if converged or smoke:
        # Converged models stop at different terminal updates. Score only the
        # validation-selected checkpoints carrying a passed certificate.
        where = audit.index("--iteration")
        audit[where + 1] = 0
        audit.extend(["--checkpoint-kind", "best", "--require-converged"])
    run(audit, environment, args.dry_run)


if __name__ == "__main__":
    main()
