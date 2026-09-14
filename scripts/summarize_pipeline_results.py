#!/usr/bin/env python3
"""Collect one isolated pipeline's results, without substituting historical reports."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


REPORTS = {
    "likelihood_validation": ["exact_parent_log_likelihood", "target_child_log_likelihood",
                              "target_child_minus_exact_parent", "numerical_certification"],
    "likelihood_test": ["exact_parent_log_likelihood", "target_child_log_likelihood",
                        "target_child_minus_exact_parent", "numerical_certification"],
    "recommendation": ["model", "popularity", "protocol", "normalizer_required"],
    "generation_counterfactual": ["factual_expected_size", "observed_size_mean",
                                   "particles_per_trip", "smc_ess_min", "counterfactuals"],
    "customer_segments": ["chosen_segments", "method", "distribution_metrics"],
    "interaction_embedding_audit": ["structure", "selection", "heldout_cross_affinity_audit", "limitations"],
    "population_size": ["passed", "full_population_contexts", "screened_complete_population",
                         "gates", "random_confirm_tail_calibration", "interpretation"],
    "size_phase_diagnostic": ["interpretation"],
    "segment_promotion_mdp": ["reward", "horizon_days", "not_identified", "interpretation"],
}


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def collect(run_dir):
    manifests = sorted(run_dir.glob("invocation_*/manifest.json"))
    if not manifests:
        raise FileNotFoundError("no invocation manifest in the requested run directory")
    manifest_path = manifests[-1]
    manifest = json.loads(manifest_path.read_text())
    checkpoint = run_dir / "artifacts/candidate_rank1.pt"
    checkpoint_sha = digest(checkpoint) if checkpoint.is_file() else None
    result = {
        "run_dir": str(run_dir), "manifest": str(manifest_path),
        "pipeline_status": manifest.get("status"), "pipeline_error": manifest.get("error"),
        "checkpoint_sha256": checkpoint_sha,
        "stages": [{k: v for k, v in stage.items() if k != "command"}
                   for stage in manifest.get("stages", [])],
        "reports": {}, "missing_reports": [], "mismatched_reports": [],
        "scope": "Original-data conditional incidence model. Price scenarios are model-conditional, not identified causal profit or retention.",
        "warning": "Completion is not proof of generator calibration, causal effects, or policy profit; inspect each diagnostic and its uncertainty.",
    }
    for name, keys in REPORTS.items():
        path = run_dir / "reports" / f"{name}.json"
        if not path.is_file():
            result["missing_reports"].append(name)
            continue
        report = json.loads(path.read_text())
        identity = report.get("checkpoint_sha256", report.get("child_sha256"))
        if checkpoint_sha is None or identity != checkpoint_sha:
            result["mismatched_reports"].append(name)
            continue
        values = {key: report[key] for key in keys if key in report}
        if name == "generation_counterfactual":
            values["generation"] = {k: v for k, v in report.get("generation", {}).items()
                                    if k != "examples"}
        if name == "recommendation":
            values["paired_comparisons"] = report.get("recommendation", {}).get("comparison")
        if name == "customer_segments":
            values["segments"] = [{k: s[k] for k in ("segment", "label", "households", "simulation")
                                   if k in s} for s in report.get("segments", [])]
        if name == "segment_promotion_mdp":
            values["budget_scenarios"] = [{k: v for k, v in s.items() if k != "daily_policy"}
                                           for s in report.get("budget_scenarios", [])]
        if name == "size_phase_diagnostic":
            values = report
        result["reports"][name] = {"path": str(path), "sha256": digest(path), "results": values}
    result["all_expected_reports_present_and_matched"] = not (
        result["missing_reports"] or result["mismatched_reports"])
    likelihood = [result["reports"].get(f"likelihood_{split}", {}).get("results", {})
                  for split in ("validation", "test")]
    likelihood_passed = bool(all(
        row.get("numerical_certification", {}).get("passed") is True
        for row in likelihood))
    population = result["reports"].get("population_size", {}).get("results", {})
    result["capability_status"] = {
        "conditional_incidence_likelihood": (
            "passed" if likelihood_passed else "failed"),
        "population_tail_safety": (
            "passed" if population.get("passed") is True else "failed"),
        "numerical_price_verified": "not_assessed",
        "factual_calibration": "not_assessed",
        "observed_price_response_evaluated": "not_assessed",
        "causal_price_identification": "not_identifiable",
        "policy_value_evaluated": "not_identifiable",
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    result = collect(root)
    output = root / "results_summary.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    lines = ["# Complete-pipeline results", "", f"Pipeline status: {result['pipeline_status']}", "",
             result["warning"], "", result["scope"], ""]
    if result["pipeline_error"]:
        lines += [f"Pipeline error: {result['pipeline_error']}", ""]
    lines += ["## Capability verdicts", "", "```json",
              json.dumps(result["capability_status"], indent=2), "```", ""]
    for name, report in result["reports"].items():
        lines += [f"## {name}", "", f"[Full report](reports/{name}.json)", "",
                  "```json", json.dumps(report["results"], indent=2, allow_nan=False), "```", ""]
    lines += ["## Missing or mismatched reports", "",
              json.dumps({k: result[k] for k in ("missing_reports", "mismatched_reports")}, indent=2), ""]
    (root / "RESULTS.md").write_text("\n".join(lines))
    print(f"[results] status={result['pipeline_status']} reports={len(result['reports'])}/{len(REPORTS)} summary={output}")


if __name__ == "__main__":
    main()
