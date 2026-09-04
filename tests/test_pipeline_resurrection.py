import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_pipeline_for_test", ROOT / "scripts" / "run_pipeline.py")
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pipeline)

from checkpoint_io import resolve_initialization_artifact


def install_data_fingerprint(monkeypatch, tmp_path):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    target = tmp_path / "basket_input" / "model_data_fingerprint.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    body = {"schema_version": 1, "fixture": "pipeline-resurrection"}
    digest = hashlib.sha256(json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
        allow_nan=False).encode("utf-8")).hexdigest()
    target.write_text(json.dumps({**body, "fingerprint_sha256": digest}))
    return digest


def test_stage_suffix_contract():
    assert pipeline.runs_stage("data", "initialize")
    assert pipeline.runs_stage("rank", "rank")
    assert pipeline.runs_stage("rank", "certification")
    assert not pipeline.runs_stage("rank", "additive")
    assert not pipeline.runs_stage("certification", "evaluation")


def test_raw_directory_auto_detects_repository_local_bundle(tmp_path, monkeypatch):
    sibling = tmp_path / "missing-sibling"
    local = tmp_path / "repository-local"
    local.mkdir()
    monkeypatch.delenv("NF_RAW_DIR", raising=False)
    monkeypatch.setattr(pipeline, "RAW_DEFAULT", sibling)
    monkeypatch.setattr(pipeline, "RAW_LOCAL", local)
    assert pipeline.resolve_raw_directory() == local.resolve()


def test_explicit_raw_directory_overrides_auto_detection(tmp_path, monkeypatch):
    explicit = tmp_path / "explicit"
    sibling = tmp_path / "sibling"
    local = tmp_path / "repository-local"
    for path in (explicit, sibling, local):
        path.mkdir()
    monkeypatch.setenv("NF_RAW_DIR", str(explicit))
    monkeypatch.setattr(pipeline, "RAW_DEFAULT", sibling)
    monkeypatch.setattr(pipeline, "RAW_LOCAL", local)
    assert pipeline.resolve_raw_directory() == explicit.resolve()


def test_additive_resurrection_validates_initialization_lineage(tmp_path, monkeypatch):
    fingerprint = install_data_fingerprint(monkeypatch, tmp_path)
    initialization = tmp_path / "initialization.pt"
    additive = tmp_path / "additive.pt"
    torch.save({
        "metadata": {"household_size_rank1": True,
                     "data_fingerprint_sha256": fingerprint},
        "model_state": {},
        "model_state_sha256": "matching-digest",
    }, initialization)
    torch.save({
        "model": {"weight": torch.ones(1)},
        "estimator": "exact_version4_no_gram_dynamic_program",
        "fresh_artifact_digest": "matching-digest",
        "data_fingerprint_sha256": fingerprint,
    }, additive)
    got = pipeline.validate_additive(additive, initialization, dry_run=False)
    assert got["fresh_artifact_digest"] == "matching-digest"

    bad = torch.load(additive, weights_only=False)
    bad["fresh_artifact_digest"] = "different-digest"
    torch.save(bad, additive)
    with pytest.raises(SystemExit, match="was not trained from"):
        pipeline.validate_additive(additive, initialization, dry_run=False)


def test_initialization_from_another_dataset_is_rejected(tmp_path, monkeypatch):
    install_data_fingerprint(monkeypatch, tmp_path)
    initialization = tmp_path / "initialization.pt"
    torch.save({
        "metadata": {"household_size_rank1": True,
                     "data_fingerprint_sha256": "another-dataset"},
        "model_state": {}, "model_state_sha256": "digest",
    }, initialization)
    with pytest.raises(SystemExit, match="does not match"):
        pipeline.validate_initialization(initialization, dry_run=False)


def test_full_additive_resurrection_rejects_unfinished_latest_checkpoint(
        tmp_path, monkeypatch):
    fingerprint = install_data_fingerprint(monkeypatch, tmp_path)
    initialization = tmp_path / "initialization.pt"
    best = tmp_path / "best.pt"
    latest = tmp_path / "latest.pt"
    torch.save({
        "metadata": {"household_size_rank1": True,
                     "data_fingerprint_sha256": fingerprint},
        "model_state": {},
        "model_state_sha256": "digest",
    }, initialization)
    base = {
        "model": {"weight": torch.ones(1)},
        "estimator": "exact_version4_no_gram_dynamic_program",
        "fresh_artifact_digest": "digest",
        "data_fingerprint_sha256": fingerprint,
        "best_iteration": 4100,
        "config": {
            "batch": 128, "require_convergence": True,
            "convergence_min_updates": 4000, "convergence_patience": 8,
            "min_lr": 6.25e-5,
        },
    }
    torch.save({**base, "iter": 4100}, best)
    torch.save({
        **base, "iter": 5000,
        "scheduler": {"learning_rate": 6.25e-5, "evaluations_since_best": 7},
    }, latest)
    with pytest.raises(SystemExit, match="partial or reached its ceiling"):
        pipeline.validate_completed_additive(
            best, latest, initialization, profile="full", dry_run=False)

    complete = torch.load(latest, weights_only=False)
    complete["scheduler"]["evaluations_since_best"] = 8
    torch.save(complete, latest)
    assert pipeline.validate_completed_additive(
        best, latest, initialization, profile="full", dry_run=False) is not None


def test_rank_and_candidate_resurrection_validation(tmp_path, monkeypatch):
    fingerprint = install_data_fingerprint(monkeypatch, tmp_path)
    initialization = tmp_path / "initialization.pt"
    candidate = tmp_path / "candidate_rank1.pt"
    basis = tmp_path / "interaction_basis_rank8.npz"
    torch.save({
        "metadata": {"household_size_rank1": True,
                     "data_fingerprint_sha256": fingerprint},
        "model_state": {},
        "model_state_sha256": "digest",
    }, initialization)
    torch.save({
        "model": {"weight": torch.ones(1)},
        "estimator": "constrained_crn_monte_carlo_mle_version4_natural_block",
        "active_rank": 5,
        "config": {"batch": 128, "artifact": str(initialization)},
        "fresh_artifact_digest": "digest",
        "data_fingerprint_sha256": fingerprint,
        "household_size_rank1": {"incremental_recalibration_supported": False},
    }, candidate)
    basis.write_bytes(b"basis")
    basis.with_suffix(".json").write_text(json.dumps({
        "data_fingerprint_sha256": fingerprint,
        "basis_sha256": pipeline.file_sha256(basis),
        "rank_stability": {
            "8": {"accepted": False},
            "7": {"accepted": False},
            "6": {"accepted": False},
            "5": {"accepted": True},
            "4": {"accepted": True},
        }
    }))

    got = pipeline.validate_candidate(
        candidate, initialization, final=True, profile="full", dry_run=False)
    assert got["active_rank"] == 5
    assert pipeline.selected_rank_from_report(
        basis, maximum_rank=8, dry_run=False) == 5


def test_checkpoint_absolute_initialization_path_relocates_to_new_clone(tmp_path):
    clone = tmp_path / "new-clone"
    checkpoint = clone / "artifacts" / "candidate_rank1.pt"
    local_initialization = clone / "artifacts" / "initialization.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint-placeholder")
    local_initialization.write_bytes(b"initialization-placeholder")
    old_absolute = Path("/old/machine/project/artifacts/initialization.pt")
    assert resolve_initialization_artifact(
        checkpoint, old_absolute) == local_initialization


def test_certification_rejects_failed_or_corrupt_evaluation_bundle(tmp_path, monkeypatch):
    fingerprint = install_data_fingerprint(monkeypatch, tmp_path)
    report = tmp_path / "reports"
    artifact = tmp_path / "artifacts"
    report.mkdir()
    artifact.mkdir()
    monkeypatch.setattr(pipeline, "REPORT", report)
    monkeypatch.setattr(pipeline, "ART", artifact)
    candidate = artifact / "candidate_rank1.pt"
    candidate.write_bytes(b"test-candidate")
    candidate_digest = pipeline.file_sha256(candidate)
    names = (
        "likelihood_validation.json", "likelihood_test.json",
        "recommendation.json", "generation_counterfactual.json",
        "customer_segments.json", "interaction_embedding_audit.json",
    )
    for name in names:
        payload = ({"numerical_certification": {"passed": True}}
                   if name.startswith("likelihood_") else {"complete": True})
        payload.update({
            "checkpoint_sha256": candidate_digest,
            "data_fingerprint_sha256": fingerprint,
        })
        (report / name).write_text(json.dumps(payload))
    np.savez(artifact / "customer_segments.npz", assignment=np.asarray([0, 1]))
    segment_report = json.loads((report / "customer_segments.json").read_text())
    segment_report["assignments_sha256"] = pipeline.file_sha256(
        artifact / "customer_segments.npz")
    (report / "customer_segments.json").write_text(json.dumps(segment_report))
    pipeline.validate_evaluation_outputs(
        profile="full", candidate=candidate, dry_run=False)

    (report / "likelihood_test.json").write_text(json.dumps({
        "numerical_certification": {"passed": False},
        "checkpoint_sha256": candidate_digest,
        "data_fingerprint_sha256": fingerprint,
    }))
    with pytest.raises(SystemExit, match="did not pass"):
        pipeline.validate_evaluation_outputs(
            profile="full", candidate=candidate, dry_run=False)


@pytest.mark.parametrize(
    ("start_at", "present", "absent"),
    [
        ("interaction", "fit_convex_natural_interactions.py", "fit_exact_additive.py"),
        ("evaluation", "compare_rank8_parent_likelihood.py",
         "fit_convex_natural_interactions.py"),
        ("certification", "audit_population_size.py",
         "compare_rank8_parent_likelihood.py"),
    ],
)
def test_dry_run_executes_only_requested_stage_suffix(
        tmp_path, monkeypatch, capsys, start_at, present, absent):
    scripts = ROOT / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "ART", tmp_path / "artifacts")
    monkeypatch.setattr(pipeline, "REPORT", tmp_path / "reports")
    monkeypatch.setattr(pipeline, "V4", tmp_path / "scripts" / "version4")
    monkeypatch.setattr(pipeline, "preflight", lambda **_kwargs: None)
    monkeypatch.setattr(
        sys, "argv", ["run_pipeline.py", "--dry-run", "--profile", "smoke",
                      "--start-at", start_at])
    pipeline.main()
    output = capsys.readouterr().out
    assert f"execution window: {start_at} -> certification" in output
    assert present in output
    assert absent not in output


def test_full_test_score_is_reporting_only_not_a_gain_gate(
        tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(pipeline, "ART", tmp_path / "artifacts")
    monkeypatch.setattr(pipeline, "REPORT", tmp_path / "reports")
    monkeypatch.setattr(pipeline, "V4", tmp_path / "scripts" / "version4")
    monkeypatch.setattr(pipeline, "preflight", lambda **_kwargs: None)
    monkeypatch.setattr(
        sys, "argv", ["run_pipeline.py", "--dry-run", "--profile", "full",
                      "--start-at", "evaluation", "--stop-after", "evaluation"])
    pipeline.main()
    likelihood_commands = [
        line for line in capsys.readouterr().out.splitlines()
        if "compare_rank8_parent_likelihood.py" in line]
    assert len(likelihood_commands) == 2
    validation = next(line for line in likelihood_commands
                      if "--split validation" in line)
    test = next(line for line in likelihood_commands if "--split test" in line)
    assert "--require-certified-gain" in validation
    assert "--require-certified-gain" not in test
