import math

import numpy as np
import pytest

from eval_smolyak_rank8_mrr import (
    paired_rank_metrics, recommendation_comparisons)


def test_paired_rank_metrics_has_explicit_direction_and_correct_standard_error():
    candidate = np.asarray([1.0, 2.0, 4.0, 8.0])
    reference = np.asarray([2.0, 2.0, 8.0, 4.0])
    result = paired_rank_metrics(
        candidate, reference, candidate="candidate", reference="reference")

    reciprocal_gain = 1.0 / candidate - 1.0 / reference
    expected_se = reciprocal_gain.std(ddof=1) / math.sqrt(len(candidate))
    assert result["candidate"] == "candidate"
    assert result["reference"] == "reference"
    assert result["cases"] == 4
    assert result["candidate_beats_reference_fraction"] == 0.5
    assert result["candidate_ties_reference_fraction"] == 0.25
    assert result["mean_rank_change_candidate_minus_reference"] == -0.25
    assert result["mrr_gain_candidate_minus_reference"] == pytest.approx(0.125)
    assert result["mrr_gain_standard_error"] == pytest.approx(expected_se)
    assert result["mrr_gain_95_interval"] == pytest.approx([
        0.125 - 1.96 * expected_se, 0.125 + 1.96 * expected_se])


def test_recommendation_comparisons_separates_gram_and_broader_structure():
    ranks = {
        "additive_utility": [10.0, 10.0, 10.0, 10.0],
        "structured_no_gram": [5.0, 10.0, 20.0, 10.0],
        "full_interaction": [4.0, 20.0, 10.0, 10.0],
    }
    comparisons = recommendation_comparisons(ranks)

    gram = comparisons["gram_interaction_vs_structured_no_gram"]
    broad = comparisons["full_structure_vs_additive_utility"]
    category = comparisons["category_structure_vs_additive_utility"]
    assert gram["reference"] == "structured_no_gram"
    assert broad["reference"] == "additive_utility"
    assert category["candidate"] == "structured_no_gram"
    assert gram["mrr_gain_candidate_minus_reference"] != pytest.approx(
        broad["mrr_gain_candidate_minus_reference"])


@pytest.mark.parametrize("bad", [
    np.asarray([1.0]),
    np.asarray([1.0, 0.0]),
    np.asarray([1.0, np.nan]),
])
def test_paired_rank_metrics_rejects_invalid_inputs(bad):
    with pytest.raises(ValueError):
        paired_rank_metrics(
            bad, np.ones_like(bad), candidate="candidate", reference="reference")
