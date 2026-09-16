"""Tests for the randomized ERIM coupon-cell estimators."""
import numpy as np
import pandas as pd

from run_erim_coupon_experiment import (
    ARM_OF_CELL, STRATUM_OF_CELL, cell_level_exact, permuted_arms, stratified_difference,
)


def test_cell_design_repeats_three_arms_inside_size_strata():
    assert [ARM_OF_CELL[c] for c in (1, 2, 3, 13, 14, 15)] == ["control", "A", "B"] * 2
    assert {STRATUM_OF_CELL[c] for c in range(1, 7)} == {0}
    assert {STRATUM_OF_CELL[c] for c in range(13, 16)} == {2}


def test_stratified_difference_weights_strata_by_size():
    arm = np.array(["A", "control", "A", "control", "control", "A", "control", "B"])
    stratum = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y = np.array([3.0, 1.0, 3.0, 1.0, 0.0, 10.0, 0.0, 99.0])
    # stratum 0: 4 units, diff 2; stratum 1: 3 comparable units, diff 10
    assert np.isclose(stratified_difference(y, arm, stratum, "A", "control"), (4 * 2 + 3 * 10) / 7)


def test_permutation_keeps_arm_counts_within_strata():
    rng = np.random.default_rng(0)
    arm = np.array(["A", "B", "control"] * 4)
    stratum = np.repeat([0, 1], 6)
    permuted = permuted_arms(arm, stratum, rng)
    for s in (0, 1):
        assert sorted(permuted[stratum == s]) == sorted(arm[stratum == s])


def test_cell_level_exact_p_value_is_minimal_when_every_cell_moves_together():
    cells = np.repeat(np.arange(1, 16), 4)
    effect = np.where(np.vectorize(ARM_OF_CELL.get)(cells) == "A", 1.0, 0.0)
    frame = pd.DataFrame({"cell": cells, "y": effect + 0.01 * cells})
    result = cell_level_exact(frame, "y", "A", "control")
    assert result["cells_positive"] == 5
    assert np.isclose(result["exact_p_two_sided"], result["minimum_attainable_p"])
