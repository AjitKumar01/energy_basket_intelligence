import json
from pathlib import Path

import numpy as np
import pytest
import torch

from checkpoint_io import floating_state_dtype, require_capabilities
from fit import Batcher
from provenance import (build_data_fingerprint, file_sha256,
                        load_data_fingerprint, strict_json_dumps)


def test_strict_json_maps_numpy_nonfinite_values_to_null():
    encoded = strict_json_dumps({
        "finite": np.float64(1.25),
        "missing": np.float64(np.nan),
        "infinite": float("inf"),
        "count": np.int64(4),
        "path": Path("report.json"),
    })
    decoded = json.loads(encoded)
    assert decoded == {
        "finite": 1.25, "missing": None, "infinite": None,
        "count": 4, "path": "report.json",
    }
    assert "NaN" not in encoded and "Infinity" not in encoded


def test_checkpoint_capabilities_fail_closed():
    blob = {"trained_capabilities": {
        "conditional_nonempty_incidence": True,
        "gram_interactions": False,
    }}
    require_capabilities(blob, "conditional_nonempty_incidence")
    with pytest.raises(ValueError, match="gram_interactions"):
        require_capabilities(blob, "gram_interactions")
    with pytest.raises(ValueError, match="predates"):
        require_capabilities({}, "conditional_nonempty_incidence")


def test_checkpoint_dtype_is_taken_from_certified_state_not_process_default():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        state = {
            "weight": torch.ones(2, dtype=torch.float64),
            "enabled": torch.tensor(True),
            "index": torch.tensor([0, 1], dtype=torch.int64),
        }
        assert torch.get_default_dtype() == torch.float32
        assert floating_state_dtype(state) == torch.float64
    finally:
        torch.set_default_dtype(previous)


def test_checkpoint_dtype_rejects_mixed_floating_state():
    with pytest.raises(ValueError, match="exactly one floating dtype"):
        floating_state_dtype({
            "left": torch.ones(1, dtype=torch.float32),
            "right": torch.ones(1, dtype=torch.float64),
        })


def test_selected_batch_contract_does_not_compute_recency():
    data = {
        "n_cat": 1,
        "store_cat_ptr": np.asarray([0, 2], dtype=np.int64),
        "store_items": np.asarray([0, 1], dtype=np.int64),
        "line_ptr": np.asarray([0, 1], dtype=np.int64),
        "trip_store": np.asarray([0], dtype=np.int64),
        "trip_day": np.asarray([10], dtype=np.int64),
        "trip_week": np.asarray([2], dtype=np.int64),
        "trip_user": np.asarray([0], dtype=np.int64),
        "line_item": np.asarray([1], dtype=np.int64),
        "line_cat": np.asarray([0], dtype=np.int64),
        "line_units": np.asarray([1], dtype=np.int64),
    }

    class FeatureStub:
        include_recency = False

        @staticmethod
        def gather(item, _store, _day, _week):
            zero = torch.zeros(len(item), dtype=torch.float64)
            return zero, zero, zero

        @staticmethod
        def recency(*_args):
            raise AssertionError("the selected no-recency path performed a lookup")

    batcher = Batcher(data, FeatureStub(), nmax=4, include_recency=False)
    _ix, context, line_context, *_ = batcher.make(np.asarray([0]))
    assert "rec" not in context
    assert "rec" not in line_context


def test_model_data_fingerprint_detects_changed_model_input(tmp_path):
    basket = tmp_path / "basket_input"
    data = tmp_path / "data"
    basket.mkdir()
    data.mkdir()
    (basket / "preprocessing_manifest.json").write_text(json.dumps({
        "cohort": {"products": 2}, "raw_sha256": {"raw": "a"},
        "derived_sha256": {"derived": "b"},
    }))
    (basket / "items_affinity.parquet").write_bytes(b"partition")
    (basket / "affinity_manifest.json").write_text(json.dumps({
        "partition_sha256": file_sha256(basket / "items_affinity.parquet"),
    }))
    (basket / "meta.json").write_text(json.dumps({"price_basis": "loyalty"}))
    (data / "build_meta.json").write_text(json.dumps({"price_basis": "loyalty"}))
    (basket / "v3_index_affinity.npz").write_bytes(b"ragged-index")

    first = build_data_fingerprint(tmp_path)
    assert load_data_fingerprint(tmp_path, verify_files=True) == first
    (basket / "v3_index_affinity.npz").write_bytes(b"changed-index")
    with pytest.raises(ValueError, match="model-facing files differ"):
        load_data_fingerprint(tmp_path, verify_files=True)


def test_fingerprint_verifies_recorded_optional_panels_only(tmp_path, monkeypatch):
    monkeypatch.delenv("ENERGY_MODEL_DATA_ROOT", raising=False)
    basket = tmp_path / "basket_input"
    data = tmp_path / "data"
    basket.mkdir()
    data.mkdir()
    (basket / "preprocessing_manifest.json").write_text(json.dumps({"cohort": {}}))
    (basket / "items_affinity.parquet").write_bytes(b"partition")
    (basket / "affinity_manifest.json").write_text(json.dumps({
        "partition_sha256": file_sha256(basket / "items_affinity.parquet"),
    }))
    (basket / "meta.json").write_text(json.dumps({"price_basis": "shelf"}))
    (data / "build_meta.json").write_text(json.dumps({"price_basis": "shelf"}))
    (basket / "v3_index_affinity.npz").write_bytes(b"ragged-index")
    first = build_data_fingerprint(tmp_path)
    assert "price_evidence_panel" not in first["files"]

    # An optional panel that appears after the identity was written does not invalidate it.
    (data / "price_week.parquet").write_bytes(b"later-panel")
    assert load_data_fingerprint(tmp_path, verify_files=True) == first

    # A recorded optional panel is still re-hashed.
    second = build_data_fingerprint(tmp_path)
    assert "price_evidence_panel" in second["files"]
    (data / "price_week.parquet").write_bytes(b"tampered-panel")
    with pytest.raises(ValueError, match="model-facing files differ"):
        load_data_fingerprint(tmp_path, verify_files=True)


def test_failed_rank_attempt_never_reuses_previous_basis(tmp_path, monkeypatch):
    import test_pipeline_resurrection as resurrection

    pipeline = resurrection.pipeline
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    monkeypatch.setattr(pipeline, "ART", artifact)
    old_basis = artifact / "interaction_basis_rank8.npz"
    old_report = old_basis.with_suffix(".json")
    old_basis.write_bytes(b"old-basis")
    old_report.write_text('{"largest_stable_rank": 8}')
    parent = tmp_path / "parent.pt"
    parent.write_bytes(b"parent")

    class FailedDriver:
        dry_run = False

        @staticmethod
        def run(_command, allow_failure=False):
            assert allow_failure
            return 2

    with pytest.raises(SystemExit, match="no prior basis was reused"):
        pipeline.rank_selection(FailedDriver(), parent, 100)
    assert old_basis.read_bytes() == b"old-basis"
    assert old_report.read_text() == '{"largest_stable_rank": 8}'


def test_failed_price_evidence_verdict_fits_zero_price_instead_of_aborting(tmp_path):
    import test_pipeline_resurrection as resurrection

    pipeline = resurrection.pipeline
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


def test_pipeline_reports_declared_support_without_availability_contract(tmp_path, monkeypatch, capsys):
    import importlib.util
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("pipeline_for_availability", root / "scripts/run_pipeline.py")
    pipeline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pipeline)
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
