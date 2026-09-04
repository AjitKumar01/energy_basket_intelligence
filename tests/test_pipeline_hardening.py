import json
from pathlib import Path

import numpy as np
import pytest
import torch

from checkpoint_io import require_capabilities
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
