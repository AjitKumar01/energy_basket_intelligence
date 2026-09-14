import importlib.util
import json
from pathlib import Path


spec = importlib.util.spec_from_file_location("pipeline_summary", Path(__file__).resolve().parents[1] / "scripts/summarize_pipeline_results.py")
summary = importlib.util.module_from_spec(spec)
spec.loader.exec_module(summary)


def test_summary_preserves_failure_and_never_uses_historical_reports(tmp_path):
    invocation = tmp_path / "invocation_20260913"
    invocation.mkdir()
    (invocation / "manifest.json").write_text(json.dumps({"status": "failed", "error": "ESS failed"}))
    value = summary.collect(tmp_path)
    assert value["pipeline_status"] == "failed"
    assert value["pipeline_error"] == "ESS failed"
    assert len(value["missing_reports"]) == len(summary.REPORTS)
    assert not value["all_expected_reports_present_and_matched"]


def test_summary_checks_final_checkpoint_identity(tmp_path):
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "reports").mkdir()
    invocation = tmp_path / "invocation_20260913"
    invocation.mkdir()
    (invocation / "manifest.json").write_text('{"status":"completed"}')
    checkpoint = tmp_path / "artifacts/candidate_rank1.pt"
    checkpoint.write_bytes(b"new checkpoint")
    (tmp_path / "reports/recommendation.json").write_text(json.dumps({
        "checkpoint_sha256": "old checkpoint", "model": {"mrr": 1.0}}))
    value = summary.collect(tmp_path)
    assert value["mismatched_reports"] == ["recommendation"]
    assert "recommendation" not in value["reports"]
    (tmp_path / "reports/recommendation.json").write_text(json.dumps({
        "checkpoint_sha256": summary.digest(checkpoint), "model": {"mrr": .1}}))
    value = summary.collect(tmp_path)
    assert value["reports"]["recommendation"]["results"]["model"]["mrr"] == .1
