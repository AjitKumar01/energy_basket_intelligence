import importlib.util
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "run_pipeline_for_test", ROOT / "scripts" / "run_pipeline.py")
pipeline = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(pipeline)


def dry_run(tmp_path, monkeypatch, capsys, *arguments: str) -> str:
    """Print the command graph for ``arguments`` without executing anything."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    for key in ("ART", "REPORT", "OUT"):
        monkeypatch.setattr(pipeline, key, getattr(pipeline, key))
    monkeypatch.setattr(pipeline, "preflight", lambda **_kwargs: None)
    monkeypatch.setenv("ENERGY_MODEL_DATA_ROOT", str(tmp_path))   # the bundle is required
    monkeypatch.setattr(sys, "argv", ["run_pipeline.py", "--dry-run", *arguments])
    pipeline.main()
    return capsys.readouterr().out


def test_dry_run_writes_nothing_and_points_stages_at_the_run_directory(
        tmp_path, monkeypatch, capsys):
    destination = tmp_path / "new_run"
    output = dry_run(tmp_path, monkeypatch, capsys, "--profile", "smoke",
                     "--run-dir", str(destination))
    assert f"--output-dir {destination / 'out'}" in output
    assert f"--parent {destination / 'out/v3_pipeline_additive_best.pt'}" in output
    assert not destination.exists()
    # Top-level CLI imports from scripts/, without requiring a custom PYTHONPATH.
    from version4.io_utils import strict_json_dumps
    assert callable(strict_json_dumps)


def test_dry_run_lists_every_stage_in_order(tmp_path, monkeypatch, capsys):
    output = dry_run(tmp_path, monkeypatch, capsys, "--profile", "smoke",
                     "--joint-refinement")
    scripts = ["initialize_version4.py", "fit_supported_price_response.py",
               "fit_exact_additive.py", "build_spectral_phi_initialization.py",
               "fit_stratified_natural_interactions.py", "fit_household_size_rank1.py",
               "fit_joint_refinement.py", "compare_rank8_parent_likelihood.py",
               "eval_smolyak_rank8_mrr.py", "audit_population_size.py",
               "run_segment_pricing_mdp.py"]
    positions = [output.index(name) for name in scripts]
    assert positions == sorted(positions)


def test_stop_after_ends_the_run_early(tmp_path, monkeypatch, capsys):
    output = dry_run(tmp_path, monkeypatch, capsys, "--profile", "smoke",
                     "--stop-after", "rank")
    assert "build_spectral_phi_initialization.py" in output
    assert "fit_stratified_natural_interactions.py" not in output


def test_dry_run_uses_stratified_estimator_by_default(tmp_path, monkeypatch, capsys):
    output = dry_run(tmp_path, monkeypatch, capsys, "--profile", "smoke",
                     "--stop-after", "interaction")
    assert "fit_stratified_natural_interactions.py" in output
    assert "--band-draws 1 1 1 1 1 1 1" in output
    assert "--category-bound 0.0" in output
    assert "--size-ridge 0.001 --size-smoothness 0.1" in output


def test_stratified_band_draws_follow_data_dependent_support():
    assert pipeline.stratified_band_draws(10, full=True) == [16, 16]
    assert pipeline.stratified_band_draws(10, full=False) == [1, 1]
    assert pipeline.stratified_band_draws(120, full=True) == [16, 16, 12, 8, 16, 16, 16]


def test_full_test_score_is_reporting_only_not_a_gain_gate(tmp_path, monkeypatch, capsys):
    output = dry_run(tmp_path, monkeypatch, capsys, "--profile", "full",
                     "--stop-after", "evaluation")
    likelihood_commands = [line for line in output.splitlines()
                           if "compare_rank8_parent_likelihood.py" in line]
    assert len(likelihood_commands) == 2
    validation = next(line for line in likelihood_commands if "--split validation" in line)
    test = next(line for line in likelihood_commands if "--split test" in line)
    assert "--require-certified-gain" in validation
    assert "--require-certified-gain" not in test


def test_failed_price_evidence_verdict_fits_zero_price_instead_of_aborting(tmp_path):
    report = tmp_path / "report.json"
    coefficients = tmp_path / "coefficients.json"
    report.write_text(json.dumps({"passed": False}))
    assert pipeline.price_configuration_from_evidence(2, report, coefficients) == (
        "--disable-price-response",)
    report.write_text(json.dumps({"passed": True}))
    assert pipeline.price_configuration_from_evidence(0, report, coefficients) == (
        "--supported-price-coefficients", coefficients)
    for status, passed in ((1, False), (2, True), (0, False)):
        report.write_text(json.dumps({"passed": passed}))
        with pytest.raises(SystemExit):
            pipeline.price_configuration_from_evidence(status, report, coefficients)


def test_rank_and_interaction_reports_are_read_for_the_rank(tmp_path):
    report = tmp_path / "basis.json"
    report.write_text(json.dumps({"largest_stable_rank": 3}))
    assert pipeline.selected_rank(report, maximum_rank=8) == 3
    report.write_text(json.dumps({"largest_stable_rank": None}))
    with pytest.raises(SystemExit, match="split-half gate"):
        pipeline.selected_rank(report, maximum_rank=8)
    report.write_text(json.dumps({"candidate_c_eigenvalues": [2.0, 1.0, 1e-14]}))
    assert pipeline.fitted_interaction_rank(report) == 2


def test_refinement_decision_selects_the_evaluated_model(tmp_path, capsys):
    report = tmp_path / "joint_refinement.json"
    report.write_text(json.dumps({"decision": {"accepted": True}}))
    assert pipeline.refinement_accepted(report)
    report.write_text(json.dumps({"decision": {"accepted": False, "reason": "no gain"}}))
    assert not pipeline.refinement_accepted(report)
    assert "no gain" in capsys.readouterr().out


def test_pipeline_reports_declared_support_without_availability_contract(
        tmp_path, monkeypatch, capsys):
    (tmp_path / "basket_input").mkdir()
    (tmp_path / "basket_input" / "meta.json").write_text('{"price_basis": "x"}')
    monkeypatch.setenv("ENERGY_MODEL_DATA_ROOT", str(tmp_path))
    assert pipeline.report_availability_contract() == {"enabled": False}
    assert "declared catalogue" in capsys.readouterr().out
    (tmp_path / "basket_input" / "meta.json").write_text(
        '{"availability_contract": {"enabled": true, "rule": "retail_first_sale", "floor": 0.003,'
        ' "unconfirmed_lines_by_split": {"train": 5}}}')
    assert pipeline.report_availability_contract()["rule"] == "retail_first_sale"
    assert "retail_first_sale" in capsys.readouterr().out
