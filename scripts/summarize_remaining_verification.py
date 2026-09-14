#!/usr/bin/env python3
"""Build fail-closed capability verdicts for the remaining-verification run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from version4.provenance import file_sha256, strict_json_dumps


ALLOWED = {"passed", "failed", "inconclusive", "not_assessed", "not_identifiable"}


def read(path):
    return json.loads(path.read_text()) if path.is_file() else None


def report_link(root, path):
    return str(path.relative_to(root)) if path and path.is_file() else None


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir",type=Path,required=True)
    args=parser.parse_args(); root=args.run_dir.resolve(); protocol=root/"verification_protocol.json"
    protocol_value=json.loads(protocol.read_text())
    benchmark=Path(protocol_value["benchmark_run"]).resolve()
    protocol_hash=file_sha256(protocol)
    expected_checkpoint=protocol_value["checkpoint_sha256"]
    expected_parent=protocol_value["parent_sha256"]
    expected_data=protocol_value["data_fingerprint_sha256"]
    synthetic=read(root/"synthetic_verification/manifest.json")
    original=read(root/"original_probability_audit/report.json")
    numerical=read(root/"real_numerical_price_audit.json")
    generation=read(root/"real_generation_calibration.json")
    tail=read(root/"real_tail_fidelity.json")
    provenance=read(root/"price_data_provenance.json")
    observed=read(root/"real_price_response_evaluation.json")
    policy=read(root/"independent_policy_evaluation.json")
    recommendation=read(benchmark/"reports/recommendation.json")
    likelihood_validation=read(benchmark/"reports/likelihood_validation.json")
    likelihood_test=read(benchmark/"reports/likelihood_test.json")
    lineage_errors=[]
    def lineage(report,name,*,checkpoint=True,parent=False,protocol_bound=False,
                data=True):
        if report is None: return False
        expected={}
        if checkpoint: expected["checkpoint_sha256"]=expected_checkpoint
        if parent: expected["parent_sha256"]=expected_parent
        if protocol_bound: expected["protocol_sha256"]=protocol_hash
        if data: expected["data_fingerprint_sha256"]=expected_data
        errors=[key for key,value in expected.items() if report.get(key)!=value]
        lineage_errors.extend(f"{name}:{key}" for key in errors)
        return not errors
    synthetic_ok=bool(synthetic and synthetic.get("status")=="completed"
                      and synthetic.get("foundations_passed") is True
                      and synthetic.get("stages")
                      and all(row.get("status")=="completed" for row in synthetic.get("stages",[])))
    original_inputs=set(original.get("input_sha256",{}).values()) if original else set()
    original_ok=bool(original and original.get("implementation_checks_passed") is True
                     and original.get("data_fingerprint_sha256")==expected_data
                     and {expected_checkpoint,expected_parent} <= original_inputs)
    numerical_lineage=lineage(numerical,"numerical",protocol_bound=True)
    generation_lineage=lineage(generation,"generation",protocol_bound=True)
    tail_lineage=lineage(tail,"tail")
    provenance_lineage=bool(
        provenance and provenance.get("status")=="completed"
        and Path(provenance.get("events_output","")).resolve()==(root/"price_events.parquet")
        and (root/"price_events.parquet").is_file()
        and provenance.get("events_sha256")==file_sha256(root/"price_events.parquet"))
    if provenance and not provenance_lineage: lineage_errors.append("provenance:events")
    observed_lineage=bool(
        lineage(observed,"observed",parent=True,protocol_bound=True)
        and provenance_lineage
        and observed.get("events_sha256")==provenance.get("events_sha256")) if observed else False
    if observed and provenance_lineage and observed.get("events_sha256")!=provenance.get("events_sha256"):
        lineage_errors.append("observed:events_sha256")
    policy_lineage=lineage(policy,"policy")
    likelihood_ok=all(x and lineage(x,"likelihood",parent=True)
                      and x.get("numerical_certification",{}).get("passed") is True
                      for x in (likelihood_validation,likelihood_test))
    capabilities={
        "synthetic_theory_implementation": "passed" if synthetic_ok else "failed",
        "conditional_incidence_likelihood": "passed" if likelihood_ok else "failed",
        "numerical_price_verified": (
            numerical.get("status", "failed")
            if synthetic_ok and original_ok and numerical and numerical_lineage
            else "failed" if numerical else "not_assessed"),
        "factual_sampler_correctness": generation.get("sampler_correctness",{}).get("status","not_assessed") if generation and generation_lineage else "failed" if generation else "not_assessed",
        "factual_calibration": generation.get("factual_calibration",{}).get("status","not_assessed") if generation and generation_lineage else "failed" if generation else "not_assessed",
        "tail_numerical_fidelity": tail.get("numerical_fidelity_status","not_assessed") if tail and tail_lineage else "failed" if tail else "not_assessed",
        "population_tail_safety": tail.get("safety_status","not_assessed") if tail and tail_lineage else "failed" if tail else "not_assessed",
        "tail_calibration": tail.get("tail_calibration",{}).get("status","not_assessed") if tail and tail_lineage else "failed" if tail else "not_assessed",
        "observed_price_response_evaluated": (
            ("passed" if observed.get("observed_price_response_evaluated")=="completed"
             else observed.get("observed_price_response_evaluated","not_assessed"))
            if observed and observed_lineage else "failed" if observed else "not_assessed"),
        "causal_price_identification": provenance.get("causal_price_identification","not_assessed") if provenance and provenance_lineage else "failed" if provenance else "not_assessed",
        "policy_value_evaluated": policy.get("policy_value_evaluated","not_assessed") if policy and policy_lineage else "failed" if policy else "not_assessed",
    }
    if recommendation and lineage(recommendation,"recommendation",parent=True):
        interval=recommendation.get("recommendation",{}).get("comparison",{}).get("mrr_full_minus_fitted_parent",{}).get("95_interval")
        capabilities["incremental_recommendation_value"] = ("passed" if interval and interval[0]>0 else "inconclusive")
    elif recommendation:
        capabilities["incremental_recommendation_value"] = "failed"
    else:
        capabilities["incremental_recommendation_value"] = "not_assessed"
    if any(value not in ALLOWED for value in capabilities.values()):
        raise ValueError(f"invalid capability status: {capabilities}")
    expected=[root/"synthetic_verification/manifest.json",root/"original_probability_audit/report.json",
              root/"real_numerical_price_audit.json",root/"real_generation_calibration.json",
              root/"real_tail_fidelity.json",root/"price_data_provenance.json",
              root/"real_price_response_evaluation.json",root/"independent_policy_evaluation.json"]
    result={"status":"completed" if all(path.is_file() for path in expected) else "incomplete",
        "protocol":str(protocol),"protocol_sha256":protocol_hash,"benchmark_run":str(benchmark),
        "capabilities":capabilities,"reports":{path.stem:{"path":str(path),"sha256":file_sha256(path)} for path in expected if path.is_file()},
        "missing_reports":[str(path) for path in expected if not path.is_file()],
        "lineage_errors":lineage_errors,
        "interpretation":("Execution completion is distinct from capability acceptance. Causal price and profit/policy value remain unavailable without an intervention design and required outcomes.")}
    (root/"verification_status.json").write_text(strict_json_dumps(result))
    lines=["# Remaining-verification report","",f"Execution status: **{result['status']}**","",
           "## Capability verdicts","","| Capability | Verdict |","| --- | --- |"]
    lines += [f"| {name.replace('_',' ')} | **{status}** |" for name,status in capabilities.items()]
    lines += ["","## Evidence","",
              f"- Frozen protocol: `{report_link(root,protocol)}`.",
              f"- Synthetic verification: `{report_link(root,root/'synthetic_verification/manifest.json')}`.",
              f"- Fixed-law numerical price audit: `{report_link(root,root/'real_numerical_price_audit.json')}`.",
              f"- Factual generation calibration: `{report_link(root,root/'real_generation_calibration.json')}`.",
              f"- Tail safety/calibration/fidelity: `{report_link(root,root/'real_tail_fidelity.json')}`.",
              f"- Price provenance: `{report_link(root,root/'price_data_provenance.json')}`.",
              f"- Observational response assessment: `{report_link(root,root/'real_price_response_evaluation.json')}`.",
              f"- Policy assessment: `{report_link(root,root/'independent_policy_evaluation.json')}`.","",
              "A `not_identifiable` verdict is a scientific result, not a runtime failure. The fitted law may be used only for capabilities that passed under the frozen protocol.",""]
    (root/"VERIFICATION_REPORT.md").write_text("\n".join(lines))
    print(strict_json_dumps(result),end="")


if __name__=="__main__": main()
