#!/usr/bin/env python3
"""Execute the frozen synthetic-then-real remaining-verification protocol."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]; V4=ROOT/"scripts/version4"


def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""): h.update(block)
    return h.hexdigest()


def sources():
    paths=sorted((ROOT/"scripts").rglob("*.py"))+sorted((ROOT/"scripts").rglob("*.cpp"))+sorted((ROOT/"tests").glob("*.py"))
    paths += [path for path in (ROOT/"pytest.ini",ROOT/"requirements.txt") if path.is_file()]
    return {str(path.relative_to(ROOT)):digest(path) for path in paths}


def atomic(path,value):
    pending=path.with_suffix(".pending.json"); pending.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n"); pending.replace(path)


def run_stage(manifest,manifest_path,name,command,log_path,expected_source,accepted=(0,),
              report=None,timeout_seconds=None,immutable=None):
    row={"name":name,"status":"running","command":command,"log":str(log_path)}
    manifest["stages"].append(row); manifest["current_stage"]=name; atomic(manifest_path,manifest)
    if sources()!=expected_source: raise RuntimeError("source changed after verification freeze")
    if immutable and any(not path.is_file() or digest(path)!=expected
                         for path,expected in immutable.items()):
        raise RuntimeError("frozen protocol/checkpoint input changed before stage")
    started=time.monotonic()
    with log_path.open("x",buffering=1) as log:
        process=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                               text=True,timeout=timeout_seconds)
    row["runtime_seconds"]=time.monotonic()-started; row["exit_code"]=process.returncode
    if sources()!=expected_source: raise RuntimeError(f"source changed during {name}")
    if immutable and any(not path.is_file() or digest(path)!=expected
                         for path,expected in immutable.items()):
        raise RuntimeError(f"frozen protocol/checkpoint input changed during {name}")
    if process.returncode not in accepted:
        row["status"]="failed"; atomic(manifest_path,manifest); raise subprocess.CalledProcessError(process.returncode,command)
    if report and not report.is_file():
        row["status"]="failed"; atomic(manifest_path,manifest)
        raise FileNotFoundError(f"stage {name} did not create required report {report}")
    row["status"]="completed" if process.returncode==0 else "scientific_gate_failed"
    if report and report.is_file(): row["report"],row["report_sha256"]=str(report),digest(report)
    atomic(manifest_path,manifest)


def main():
    stage_names=("regression_tests","synthetic_verification","isolated_original_probability",
                 "price_data_provenance","real_numerical_price","real_generation_calibration",
                 "real_tail_fidelity","observational_price_response")
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir",type=Path,default=ROOT/"artifacts/remaining_verification_20260914"); parser.add_argument("--threads",type=int,default=4); parser.add_argument("--start-at",choices=stage_names,default="regression_tests"); parser.add_argument("--resume-manifest",type=Path,help="required provenance manifest when --start-at skips stages")
    args=parser.parse_args(); run=args.run_dir.resolve(); protocol=run/"verification_protocol.json"
    os.environ.setdefault("V3_AFFINITY", "1")
    if not protocol.is_file(): raise SystemExit("frozen verification_protocol.json is required")
    benchmark=(ROOT/json.loads(protocol.read_text())["benchmark_run"]).resolve()
    checkpoint=benchmark/"artifacts/candidate_rank1.pt"; parent=benchmark/"out/v3_pipeline_additive_best.pt"
    expected=json.loads(protocol.read_text())
    if digest(checkpoint)!=expected["checkpoint_sha256"] or digest(parent)!=expected["parent_sha256"]: raise SystemExit("frozen checkpoint lineage mismatch")
    stamp=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); invocation=run/f"invocation_{stamp}"; invocation.mkdir(parents=True,exist_ok=False)
    frozen=sources(); diff=subprocess.run(["git","diff","--binary"],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    (invocation/"source_diff.patch").write_text(diff)
    status=subprocess.run(["git","status","--short"],cwd=ROOT,capture_output=True,text=True,check=True).stdout
    (invocation/"git_status.txt").write_text(status)
    snapshot=invocation/"source_snapshot"
    for directory in ("scripts","tests"):
        shutil.copytree(ROOT/directory,snapshot/directory,
                        ignore=shutil.ignore_patterns("__pycache__","*.pyc"))
    for filename in ("pytest.ini","requirements.txt"):
        if (ROOT/filename).is_file(): shutil.copy2(ROOT/filename,snapshot/filename)
    manifest_path=invocation/"manifest.json"
    manifest={"schema":1,"status":"running","pid":os.getpid(),"started_utc":stamp,"run_dir":str(run),"start_at":args.start_at,
        "protocol":str(protocol),"protocol_sha256":digest(protocol),"checkpoint_sha256":digest(checkpoint),
        "parent_sha256":digest(parent),"source_sha256":frozen,"python":sys.version,"platform":platform.platform(),
        "packages":{name:importlib.metadata.version(name) for name in ("numpy","pandas","scipy","scikit-learn","torch","pyarrow")},"stages":[]}
    atomic(manifest_path,manifest); py=sys.executable
    def interrupted(signum,_frame):
        raise InterruptedError(f"verification interrupted by signal {signum}")
    for name in ("SIGHUP","SIGINT","SIGTERM"):
        if hasattr(signal,name): signal.signal(getattr(signal,name),interrupted)
    tail_spec=expected["tail"]
    policy_spec=expected["policy"]
    stages=[
      ("regression_tests",[py,"-m","pytest","-q"],(0,),None),
      ("synthetic_verification",[py,"-u","scripts/run_synthetic_experiment.py","--profile","full","--threads",str(args.threads),"--output-dir",str(run/"synthetic_verification")],(0,),run/"synthetic_verification/manifest.json"),
      ("isolated_original_probability",[py,"-u",str(V4/"audit_original_probability.py"),"--run-dir",str(benchmark),"--contexts","128","--threads",str(args.threads),"--output-dir",str(run/"original_probability_audit")],(0,),run/"original_probability_audit/report.json"),
      ("price_data_provenance",[py,"-u",str(V4/"audit_price_data_provenance.py"),"--output",str(run/"price_data_provenance.json"),"--events-output",str(run/"price_events.parquet")],(0,),run/"price_data_provenance.json"),
      ("real_numerical_price",[py,"-u",str(V4/"audit_real_price_numerics.py"),"--run-dir",str(benchmark),"--protocol",str(protocol),"--output",str(run/"real_numerical_price_audit.json"),"--per-context-output",str(run/"real_numerical_price_per_context.npz"),"--threads",str(args.threads)],(0,2),run/"real_numerical_price_audit.json"),
      ("real_generation_calibration",[py,"-u",str(V4/"audit_real_generation_calibration.py"),"--run-dir",str(benchmark),"--protocol",str(protocol),"--output",str(run/"real_generation_calibration.json"),"--per-context-output",str(run/"real_generation_per_context.npz"),"--samples-output",str(run/"real_generation_samples.npz"),"--threads",str(args.threads)],(0,),run/"real_generation_calibration.json"),
      ("real_tail_fidelity",[py,"-u",str(V4/"audit_population_size.py"),"--checkpoint",str(checkpoint),"--split",str(tail_spec["split"]),"--rank","5","--screen-level",str(tail_spec["screen_level"]),"--confirm-level",str(tail_spec["confirm_level"]),"--followup-level",str(tail_spec["followup_level"]),"--confirm-contexts",str(tail_spec["confirm_contexts"]),"--calibration-contexts",str(tail_spec["calibration_contexts"]),"--calibration-seed",str(tail_spec["calibration_seed"]),"--contexts",str(tail_spec["contexts"]),"--chunk","48","--threads",str(args.threads),"--calibration-margin",str(tail_spec["calibration_absolute_margin"]),"--output",str(run/"real_tail_fidelity.json")],(0,2),run/"real_tail_fidelity.json"),
      ("observational_price_response",[py,"-u",str(V4/"evaluate_observational_price_response.py"),"--run-dir",str(benchmark),"--protocol",str(protocol),"--events",str(run/"price_events.parquet"),"--output",str(run/"real_price_response_evaluation.json"),"--per-event-output",str(run/"real_price_response_events.parquet"),"--threads",str(args.threads)],(0,),run/"real_price_response_evaluation.json")]
    started=time.monotonic()
    try:
        start_index=stage_names.index(args.start_at)
        prior=None
        if start_index:
            if args.resume_manifest is None:
                raise ValueError("--resume-manifest is required when --start-at skips stages")
            prior_path=args.resume_manifest.resolve(); prior=json.loads(prior_path.read_text())
            for key,value in (("protocol_sha256",digest(protocol)),
                              ("checkpoint_sha256",digest(checkpoint)),
                              ("parent_sha256",digest(parent)),
                              ("source_sha256",frozen)):
                if prior.get(key)!=value:
                    raise RuntimeError(f"resume manifest {key} does not match this invocation")
            manifest["resume_manifest"],manifest["resume_manifest_sha256"]=(
                str(prior_path),digest(prior_path))
            atomic(manifest_path,manifest)
        maximum_seconds=float(expected["maximum_wall_clock_hours"])*3600.0
        immutable={protocol:digest(protocol),checkpoint:digest(checkpoint),
                   parent:digest(parent)}

        def remaining_time():
            remaining=maximum_seconds-(time.monotonic()-started)
            if remaining<=0:
                raise TimeoutError("frozen verification wall-clock budget exhausted")
            return remaining

        for index,(name,command,accepted,report) in enumerate(stages):
            if index<start_index:
                matches=[row for row in prior.get("stages",[]) if row.get("name")==name]
                if len(matches)!=1 or matches[0].get("status") not in {
                        "completed","scientific_gate_failed"}:
                    raise RuntimeError(
                        f"resume manifest lacks a directly completed stage {name}")
                old=matches[0]
                if report and (not report.is_file() or old.get("report")!=str(report)
                               or old.get("report_sha256")!=digest(report)):
                    raise RuntimeError(f"reused report identity mismatch for {name}")
                manifest["stages"].append({"name":name,"status":"reused",
                    "source_manifest":str(args.resume_manifest.resolve()),
                    "report":str(report) if report else None,
                    "report_sha256":digest(report) if report else None})
                atomic(manifest_path,manifest)
                continue
            run_stage(manifest,manifest_path,name,command,invocation/f"{name}.log",frozen,
                      accepted,report,remaining_time(),immutable)
        numerical=json.loads((run/"real_numerical_price_audit.json").read_text())
        generation=json.loads((run/"real_generation_calibration.json").read_text())
        tail=json.loads((run/"real_tail_fidelity.json").read_text())
        prerequisites={
            "synthetic_theory":bool(
                json.loads((run/"synthetic_verification/manifest.json").read_text())
                .get("foundations_passed") is True),
            "original_implementation":bool(
                json.loads((run/"original_probability_audit/report.json").read_text())
                .get("implementation_checks_passed") is True),
            "numerical_price":numerical.get("status")=="passed",
            "sampler_correctness":generation.get("sampler_correctness",{}).get("status")=="passed",
            "factual_calibration":generation.get("factual_calibration",{}).get("status")=="passed",
            "tail_numerical_fidelity":tail.get("numerical_fidelity_status")=="passed",
            "tail_safety":tail.get("safety_status")=="passed",
            "tail_calibration":tail.get("tail_calibration",{}).get("status")=="passed",
        }
        if all(prerequisites.values()):
            command=[py,"-u",str(V4/"run_segment_pricing_mdp.py"),"--checkpoint",str(checkpoint),"--assignments",str(benchmark/"artifacts/customer_segments.npz"),"--segment-report",str(benchmark/"reports/customer_segments.json"),"--contexts-per-segment",str(policy_spec["contexts_per_segment"]),"--particles",str(policy_spec["particles"]),"--threads",str(args.threads),"--minimum-budget-utilization",str(policy_spec["minimum_budget_utilization"]),"--output",str(run/"policy_model_conditional.json"),"--frozen-policy-output",str(run/"frozen_policy.json"),"--independent-evaluation-output",str(run/"independent_policy_evaluation.json")]
            run_stage(manifest,manifest_path,"corrected_policy",command,
                      invocation/"corrected_policy.log",frozen,(0,),
                      run/"independent_policy_evaluation.json",remaining_time(),immutable)
        else:
            failed=[name for name,passed in prerequisites.items() if not passed]
            reason=("policy computation not run because frozen prerequisites did not pass: "
                    +", ".join(failed))
            identity={"checkpoint_sha256":digest(checkpoint),
                      "data_fingerprint_sha256":expected["data_fingerprint_sha256"]}
            (run/"frozen_policy.json").write_text(json.dumps({"status":"not_assessed",
                "reason":reason,**identity},indent=2)+"\n")
            (run/"independent_policy_evaluation.json").write_text(json.dumps({
                "model_conditional_status":"not_assessed",
                "policy_value_evaluated":"not_identifiable","reason":reason,
                **identity},indent=2)+"\n")
            manifest["stages"].append({"name":"corrected_policy","status":"not_assessed",
                                       "reason":reason,"prerequisites":prerequisites})
        run_stage(manifest,manifest_path,"summary",[py,"-u","scripts/summarize_remaining_verification.py","--run-dir",str(run)],invocation/"summary.log",frozen,(0,),run/"verification_status.json",remaining_time(),immutable)
    except BaseException as error:
        if manifest["stages"] and manifest["stages"][-1].get("status")=="running":
            manifest["stages"][-1]["status"]="failed"
        manifest["status"]="failed"; manifest["error"]=repr(error); manifest["runtime_seconds"]=time.monotonic()-started; atomic(manifest_path,manifest); raise
    manifest["status"]="completed"; manifest["current_stage"]=None; manifest["runtime_seconds"]=time.monotonic()-started; atomic(manifest_path,manifest)
    print(f"[remaining-verification] completed: {run/'VERIFICATION_REPORT.md'}",flush=True)


if __name__=="__main__": main()
