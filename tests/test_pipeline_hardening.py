import json
from pathlib import Path

import numpy as np
import pytest
import torch

from fit import Batcher
from io_utils import strict_json_dumps
from model_io import floating_state_dtype, require_capabilities


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


def test_model_capabilities_fail_closed():
    blob = {"trained_capabilities": {
        "conditional_nonempty_incidence": True,
        "gram_interactions": False,
    }}
    require_capabilities(blob, "conditional_nonempty_incidence")
    with pytest.raises(ValueError, match="gram_interactions"):
        require_capabilities(blob, "gram_interactions")
    with pytest.raises(ValueError, match="declares no trained capabilities"):
        require_capabilities({}, "conditional_nonempty_incidence")


def test_model_dtype_is_taken_from_saved_state_not_process_default():
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


def test_model_dtype_rejects_mixed_floating_state():
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
