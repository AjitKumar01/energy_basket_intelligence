#!/usr/bin/env python3
"""Isolated rank confirmation and embedding pilot; never promotes a checkpoint.

Can adopt an in-flight precision run. One rank is selected BEFORE a single
predeclared confirmation stream. Failure is retained and never triggers seed
search or relaxed thresholds. Original run directories remain read-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from run_synthetic_experiment import ROOT, V4, Tee, source_identity, write_manifest

sys.path.insert(0, str(V4))
import numpy as np
from provenance import file_sha256, strict_json_dumps
from uncertainty import paired_score_summary


def confirmed_rank(precision, confirmation, selected):
    """No reselection on the confirmation stream, and no threshold changes."""
    for report in (precision, confirmation):
        if report.get("predeclared_stability_threshold") != 0.5:
            return False
        if not report.get("rank_stability", {}).get(str(selected), {}).get("accepted"):
            return False
    return bool(precision["parent_sha256"] == confirmation["parent_sha256"]
                and precision["data_fingerprint_sha256"] == confirmation["data_fingerprint_sha256"]
                and precision["context_seed"] == confirmation["context_seed"]
                and precision["contexts"] == confirmation["contexts"]
                and precision["draw_seed"] != confirmation["draw_seed"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--precision-basis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wait-for-precision", action="store_true")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--contexts", type=int, default=2048)
    parser.add_argument("--validation-contexts", type=int, default=512)
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    destination = args.output_dir.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    sys.stdout = Tee(sys.stdout, destination / "pipeline.log")
    sys.stderr = sys.stdout
    parent, precision_path = args.parent.resolve(), args.precision_basis.resolve()
    manifest_path = destination / "manifest.json"
    identity = source_identity()
    manifest = {
        "status": "running", "started_utc": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(), "parent": str(parent), "parent_sha256": file_sha256(parent),
        "precision_basis": str(precision_path), "source_sha256": identity,
        "source_hash_scope": "follow-up stages; adopted precision worker did not record a launch-time source snapshot",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "predeclared_protocol": {"rank_rule": "largest primary-stable rank; same rank must confirm",
            "stability_threshold": 0.5, "confirmation_draw_seed": 36602,
            "confirmation_draws": 8, "pilot_seed": 91301, "ll_rec_seed": 91401,
            "refinement_anchor_ridge": 0.01, "initialization_if_fixed_zero": "0.05 times spectral basis",
            "recommendation_screen": "paired validation mean MRR change must be nonnegative; report clustered CI",
            "no_test_data": True, "automatic_promotion": False},
        "stages": [],
    }
    write_manifest(manifest_path, manifest)

    def stage(name, filename, *arguments, allow_rejection=False):
        if source_identity() != identity or file_sha256(parent) != manifest["parent_sha256"]:
            raise RuntimeError("source or additive parent changed during recovery")
        command = [sys.executable, "-u", str(V4 / filename), *map(str, arguments)]
        log_path = destination / f"{len(manifest['stages']) + 1:02d}_{name}.log"
        row = {"name": name, "status": "running", "command": command, "log": str(log_path)}
        manifest["stages"].append(row)
        write_manifest(manifest_path, manifest)
        tick = time.monotonic()
        print("[recovery] " + " ".join(command), flush=True)
        with log_path.open("x", buffering=1) as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, bufsize=1) as process:
                for line in process.stdout:
                    log.write(line); print(line, end="", flush=True)
                code = process.wait()
        row.update({"returncode": code, "seconds": time.monotonic() - tick,
                    "status": "completed" if code == 0 else "rejected" if code == 2 else "failed",
                    "log_sha256": file_sha256(log_path)})
        write_manifest(manifest_path, manifest)
        if code and not (allow_rejection and code == 2):
            raise RuntimeError(f"stage {name} returned {code}; see {log_path}")
        return code

    try:
        deadline = time.monotonic() + (7200 if args.wait_for_precision else 0)
        while not precision_path.with_suffix(".json").exists():
            if time.monotonic() >= deadline:
                raise RuntimeError("precision report missing; no stale basis will be substituted")
            print(f"[recovery] waiting for precision report: {precision_path.with_suffix('.json')}", flush=True)
            time.sleep(30)
        precision = json.loads(precision_path.with_suffix(".json").read_text())
        if (precision["parent_sha256"] != file_sha256(parent)
                or precision["basis_sha256"] != file_sha256(precision_path)
                or precision.get("model_draws_per_context") != 8
                or precision.get("predeclared_stability_threshold") != 0.5):
            raise RuntimeError("precision report does not match the predeclared parent/precision/gate")
        selected = precision["largest_stable_rank"]
        manifest.update({"precision_report": precision, "selected_rank_before_confirmation": selected})
        write_manifest(manifest_path, manifest)
        if selected is None:
            manifest.update({"status": "rank_rejected", "decision": "retain additive; no supported rank"})
            return
        confirmation_path = destination / "spectral_confirmation.npz"
        stage("rank_confirmation", "build_spectral_phi_initialization.py",
              "--parent", parent, "--trips", precision["contexts"], "--draws", 8,
              "--rank", 8, "--seed", precision["context_seed"], "--draw-seed", 36602,
              "--threads", args.threads, "--output", confirmation_path, allow_rejection=True)
        confirmation = json.loads(confirmation_path.with_suffix(".json").read_text())
        if confirmation["basis_sha256"] != file_sha256(confirmation_path):
            raise RuntimeError("confirmation basis content digest mismatch")
        with np.load(precision_path) as first, np.load(confirmation_path) as second:
            if any(not np.array_equal(first[key], second[key]) for key in ("trips", "half_a")):
                raise RuntimeError("precision and confirmation context/half manifests differ")
        manifest["confirmation_report"] = confirmation
        if not confirmed_rank(precision, confirmation, selected):
            manifest.update({"status": "rank_rejected", "decision": "preselected rank failed confirmation; no seed search"})
            return
        pilot = destination / "pilot"
        stage("embedding_pilot", "pilot_embedding_refinement.py", "--parent", parent,
              "--spectral", precision_path, "--rank", selected, "--threads", args.threads,
              "--contexts", args.contexts, "--validation-contexts", args.validation_contexts,
              "--steps", args.steps, "--output-dir", pilot)
        for name in ("fixed", "free"):
            stage(f"{name}_likelihood", "compare_rank8_parent_likelihood.py",
                  "--parent", parent, "--child", pilot / f"{name}.pt",
                  "--split", "validation", "--trips", args.validation_contexts,
                  "--audit-trips", min(128, args.validation_contexts), "--seed", 91401,
                  "--maximum-audit-error-bound", 0.002, "--require-certified-gain",
                  "--threads", args.threads, "--output", destination / f"{name}_likelihood.json",
                  allow_rejection=True)
            stage(f"{name}_recommendations", "eval_smolyak_rank8_mrr.py",
                  "--ckpt", pilot / f"{name}.pt", "--parent", parent,
                  "--split", "validation", "--trips", args.validation_contexts, "--seed", 91401,
                  "--threads", args.threads, "--output", destination / f"{name}_recommendations.json")
        first = np.load(destination / "fixed_likelihood_per_trip.npz")
        second = np.load(destination / "free_likelihood_per_trip.npz")
        if not np.array_equal(first["trips"], second["trips"]):
            raise RuntimeError("likelihood comparison trip manifests differ")
        ll_delta = paired_score_summary(second["target_child"] - first["target_child"], first["household"])
        ll_reports = [json.loads((destination / f"{name}_likelihood.json").read_text()) for name in ("fixed", "free")]
        allowance = sum(r["numerical_certification"]["empirical_adjacent_rule_allowance"] for r in ll_reports)
        rec = [json.loads((destination / f"{name}_recommendations.json").read_text())["recommendation"]["per_case"]
               for name in ("fixed", "free")]
        if rec[0]["trips"] != rec[1]["trips"] or rec[0]["hidden_items"] != rec[1]["hidden_items"]:
            raise RuntimeError("recommendation manifests differ")
        ranks = [np.asarray(r["ranks"]["full_interaction"], dtype=float) for r in rec]
        recommendation_delta = {"mrr": paired_score_summary(1 / ranks[1] - 1 / ranks[0], rec[0]["households"]),
            "recall": {str(k): paired_score_summary((ranks[1] <= k).astype(float) - (ranks[0] <= k), rec[0]["households"])
                       for k in (5, 10, 20, 100)}}
        pilot_report = json.loads((pilot / "report.json").read_text())
        if source_identity() != identity:
            raise RuntimeError("source changed during final recovery stage")
        eligible = bool(pilot_report["passed_screening_for_further_audit"]
                        and all(r["numerical_certification"]["passed"] for r in ll_reports)
                        and ll_delta["95_interval"][0] > allowance
                        and recommendation_delta["mrr"]["mean"] >= 0)
        decision = {"free_minus_fixed_likelihood": ll_delta, "adjacent_rule_allowance_sum": allowance,
                    "free_minus_fixed_recommendations": recommendation_delta,
                    "free_eligible_for_larger_training_study": eligible,
                    "certified": False, "promoted": False,
                    "next_step": "larger predeclared fit and untouched test/population audits" if eligible
                                 else "retain fixed/additive baseline; inspect recorded failure causes"}
        (destination / "decision.json").write_text(strict_json_dumps(decision))
        manifest.update({"status": "completed", "decision": decision})
    except BaseException as error:
        manifest.update({"status": "failed", "error": repr(error)})
        raise
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        manifest["artifact_sha256"] = {str(p.relative_to(destination)): file_sha256(p)
            for p in destination.rglob("*") if p.is_file() and p != manifest_path
            and p.name != "pipeline.log"}
        write_manifest(manifest_path, manifest)
        print(f"[recovery] status={manifest['status']} manifest={manifest_path}", flush=True)


if __name__ == "__main__":
    main()
