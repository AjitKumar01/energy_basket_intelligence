#!/usr/bin/env python3
"""Run the two frozen real-data retailer counterfactual examples sequentially."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--particles", type=int, default=256)
    parser.add_argument("--levels", type=int, default=17)
    parser.add_argument("--replicates", type=int, default=4)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    query_dir = ROOT / "artifacts/retail_counterfactual_queries_20260914"
    examples = [
        ("soy_to_dairy_with_bread", query_dir / "soy_to_dairy_with_bread.json"),
        ("drop_butter_keep_milk_bread", query_dir / "drop_butter_keep_milk_bread.json"),
    ]
    manifest = {
        "status": "running", "started_unix_seconds": time.time(),
        "checkpoint": str(args.checkpoint.resolve()),
        "particles": args.particles, "levels": args.levels,
        "replicates": args.replicates, "threads": args.threads,
        "stages": [],
    }
    manifest_path = output / "manifest.json"; atomic_json(manifest_path, manifest)
    for index, (name, query) in enumerate(examples):
        report = output / (name + ".json")
        log = output / (name + ".log")
        command = [
            sys.executable, "-u",
            str(ROOT / "scripts/version4/basket_counterfactual_query.py"),
            "--checkpoint", str(args.checkpoint.resolve()),
            "--query", str(query), "--particles", str(args.particles),
            "--levels", str(args.levels), "--replicates", str(args.replicates),
            "--threads", str(args.threads), "--seed", str(77101 + 1000 * index),
            "--output", str(report),
        ]
        stage = {"name": name, "status": "running", "query": str(query),
                 "report": str(report), "log": str(log),
                 "started_unix_seconds": time.time()}
        manifest["stages"].append(stage); atomic_json(manifest_path, manifest)
        with log.open("w") as stream:
            completed = subprocess.run(
                command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                check=False)
        stage["returncode"] = completed.returncode
        stage["status"] = "completed" if completed.returncode == 0 else "failed"
        stage["finished_unix_seconds"] = time.time()
        atomic_json(manifest_path, manifest)
        if completed.returncode != 0:
            manifest["status"] = "failed"; atomic_json(manifest_path, manifest)
            raise SystemExit(completed.returncode)
    manifest["status"] = "completed"
    manifest["finished_unix_seconds"] = time.time()
    atomic_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
