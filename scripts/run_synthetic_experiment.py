#!/usr/bin/env python3
"""Run the probability-foundation oracle and complete synthetic retailer experiments.

Each invocation owns a fresh directory, a persistent aggregate log, per-stage logs, and
an atomic status manifest. Failures stop the pipeline and record the failing stage.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "scripts" / "version4"


class Tee:
    def __init__(self, original, path):
        self.original = original
        self.stream = path.open("x", buffering=1)

    def write(self, text):
        self.original.write(text)
        self.stream.write(text)
        return len(text)

    def flush(self):
        self.original.flush()
        self.stream.flush()


def write_manifest(path, value):
    temporary = path.with_suffix(".pending.json")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def source_identity():
    paths = sorted((ROOT / "scripts").rglob("*.py"))
    paths += sorted((ROOT / "scripts").rglob("*.cpp"))
    paths += sorted((ROOT / "tests").glob("*.py"))
    return {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths}


def run(command: list[str], log_path: Path) -> float:
    print("[synthetic-driver] " + " ".join(command), flush=True)
    tick = time.perf_counter()
    with log_path.open("x", buffering=1) as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
            assert process.stdout is not None
            for line in process.stdout:
                log.write(line)
                print(line, end="", flush=True)
            code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)
    return time.perf_counter() - tick


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=73021)
    parser.add_argument("--output-dir", type=Path,
                        help="new directory for logs and reports; refuses to overwrite a prior run")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = (args.output_dir if args.output_dir is not None else
                   ROOT / "artifacts" / f"synthetic_probability_{args.profile}_{stamp}")
    if not destination.is_absolute():
        destination = ROOT / destination
    destination.mkdir(parents=True, exist_ok=False)
    sys.stdout = Tee(sys.stdout, destination / "pipeline.log")
    sys.stderr = sys.stdout
    exact = destination / "synthetic_exact_certification.json"
    retailer = destination / "synthetic_retailer_experiment.json"
    misspecified = destination / "synthetic_retailer_misspecified.json"
    foundations = destination / "probability_foundations.json"
    manifest_path = destination / "manifest.json"
    sources = source_identity()
    manifest = {
        "schema": 2, "status": "running", "pid": os.getpid(),
        "profile": args.profile, "seed": args.seed, "threads": args.threads,
        "started_utc": stamp, "python": sys.version, "platform": platform.platform(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "git_branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "source_sha256": sources, "stages": [],
        "scope": "Synthetic implementation verification and recovery, not a universal proof or real-retailer causal evidence",
    }
    write_manifest(manifest_path, manifest)
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
    retailer_command = [
        sys.executable, "-u", str(V4 / "audit_synthetic_retailer.py"),
        "--profile", args.profile, "--threads", str(args.threads),
        "--seed", str(args.seed), "--output", str(retailer),
    ]
    misspecified_command = [
        sys.executable, "-u", str(V4 / "audit_synthetic_retailer.py"),
        "--profile", args.profile, "--world", "misspecified",
        "--threads", str(args.threads), "--seed", str(args.seed + 101),
        "--output", str(misspecified),
    ]
    stages = [
        ("native_build", [sys.executable, str(V4 / "setup_poly_degree_native.py"),
                          "build_ext", "--build-lib", "artifacts/native/lib",
                          "--build-temp", "artifacts/native/temp"], None),
        ("probability_foundations", [sys.executable, "-u", str(V4 / "audit_probability_foundations.py"),
                                    "--profile", args.profile, "--threads", str(args.threads),
                                    "--seed", str(args.seed + 10000), "--output", str(foundations)], foundations),
        ("exact_interaction_recovery", exact_command, exact),
        ("complete_retailer", retailer_command, retailer),
        ("misspecified_retailer", misspecified_command, misspecified),
    ]
    if args.profile == "full":
        # Two predeclared additional seeds, not selected after observing their results.
        for offset in (20000, 40000):
            for world in ("well_specified", "misspecified"):
                name = f"retailer_{world}_seed{args.seed + offset}"
                output = destination / f"{name}.json"
                command = [sys.executable, "-u", str(V4 / "audit_synthetic_retailer.py"),
                           "--profile", "full", "--world", world, "--threads", str(args.threads),
                           "--seed", str(args.seed + offset), "--output", str(output)]
                stages.append((name, command, output))
    started = time.perf_counter()
    try:
        for name, command, output in stages:
            row = {"name": name, "status": "running", "command": command,
                   "log": str(destination / f"{name}.log"),
                   "report": str(output) if output else None}
            manifest["stages"].append(row)
            manifest["current_stage"] = name
            write_manifest(manifest_path, manifest)
            if source_identity() != sources:
                raise RuntimeError("source files changed during the experiment; restart in a new run directory")
            print(f"[synthetic-driver] starting {name}; log={row['log']}", flush=True)
            row["runtime_seconds"] = run(command, Path(row["log"]))
            if source_identity() != sources:
                raise RuntimeError("source files changed during the experiment stage")
            if output is not None:
                report = json.loads(output.read_text())
                if name == "probability_foundations" and report.get("passed") is not True:
                    raise RuntimeError("probability foundations failed its predeclared gates")
                if args.profile == "full" and "retailer" in name and report.get("acceptance", {}).get("passed") is not True:
                    raise RuntimeError(f"{name} failed its generation/policy acceptance gates: {report.get('acceptance')}")
                row["report_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
            row["status"] = "completed"
            row["exit_code"] = 0
            write_manifest(manifest_path, manifest)
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        if manifest["stages"]:
            manifest["stages"][-1]["status"] = "failed"
            manifest["stages"][-1]["exit_code"] = getattr(error, "returncode", 1)
        manifest["runtime_seconds"] = time.perf_counter() - started
        write_manifest(manifest_path, manifest)
        raise
    manifest["status"] = "completed"
    manifest["foundations_passed"] = True
    manifest["recovery_reports_require_interpretation"] = True
    manifest["current_stage"] = None
    manifest["runtime_seconds"] = time.perf_counter() - started
    write_manifest(manifest_path, manifest)
    print(f"[synthetic-driver] completed; manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
