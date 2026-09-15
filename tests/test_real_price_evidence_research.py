import math

import numpy as np
import pandas as pd
import pytest

from research_real_price_evidence import (
    build_store_events,
    clustered_slope,
    eligible_store_events,
    pooled_slope,
)


def test_outcome_selected_store_price_audit_is_retired():
    with pytest.raises(RuntimeError, match="conditions on the outcome"):
        build_store_events()


def test_price_evidence_slope_and_cluster_interval_recover_exact_relation():
    frame = pd.DataFrame({
        "item_id": [0, 0, 1, 1],
        "log_price_change": [-.2, .1, -.1, .3],
        "observed_response": [.4, -.2, .2, -.6],
    })
    assert math.isclose(pooled_slope(frame), -2.0)
    result = clustered_slope(frame)
    assert math.isclose(result["slope"], -2.0)
    assert result["product_clusters"] == 2
    assert result["product_cluster_standard_error"] < 1e-12


def test_price_event_eligibility_does_not_read_heldout_purchase_response():
    frame = pd.DataFrame({
        "promotion_depth_change": [0.0, 0.0],
        "display_changed": [False, False],
        "mailer_changed": [False, False],
        "log_price_change": [math.log(1.1), math.log(.9)],
        "n_train_lines": [600, 600],
        "trips": [20, 20],
        "previous_trips": [20, 20],
        "training_change_low": [-.2, -.2],
        "training_change_high": [.2, .2],
        "observed_response": [-1.0, 1.0],
    })
    original = eligible_store_events(frame)
    frame["observed_response"] = np.array([1000.0, -1000.0])
    assert np.array_equal(original, eligible_store_events(frame))
