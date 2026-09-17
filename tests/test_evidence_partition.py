"""Tests for the model-free substitution-evidence partition."""
import numpy as np
import pytest

from evidence_partition import cluster_block, estimate_kappa, settings_from


def block(score):
    score = np.array(score, dtype=float)
    return score, np.ones_like(score, dtype=bool) & ~np.eye(len(score), dtype=bool)


def test_strong_substitutes_merge_and_unrelated_products_stay_alone():
    s = np.zeros((5, 5))
    s[np.ix_([0, 1, 2], [0, 1, 2])] = -1.0
    score, evidence = block(s)
    groups = cluster_block(score, evidence, settings_from({}))
    assert [0, 1, 2] in groups and [3] in groups and [4] in groups


def test_complement_pairs_are_never_grouped():
    s = np.full((3, 3), -1.0)
    s[0, 2] = s[2, 0] = 0.5                        # cannot-link
    score, evidence = block(s)
    groups = cluster_block(score, evidence, settings_from({}))
    assert not any(0 in g and 2 in g for g in groups)


def test_group_size_cap_and_homogeneity_threshold_hold():
    score, evidence = block(np.full((6, 6), -1.0))
    groups = cluster_block(score, evidence, settings_from({"maximum_group_size": 4}))
    assert max(len(g) for g in groups) <= 4 and sorted(sum(groups, [])) == list(range(6))
    weak, evidence = block(np.full((3, 3), -0.1))  # substitutes, but weaker than the threshold
    assert all(len(g) == 1 for g in cluster_block(weak, evidence, settings_from({})))


def test_pairs_without_evidence_do_not_link():
    score = np.full((2, 2), -2.0)
    evidence = np.zeros((2, 2), dtype=bool)
    assert cluster_block(score, evidence, settings_from({})) == [[0], [1]]


def test_kappa_is_large_for_poisson_data_and_small_for_overdispersed_data():
    rng = np.random.default_rng(0)
    expected = rng.uniform(1, 20, 4000)
    poisson = rng.poisson(expected)
    mixed = rng.poisson(expected * rng.gamma(0.5, 2.0, 4000))
    assert estimate_kappa(poisson, expected) > 50
    assert 0.3 < estimate_kappa(mixed, expected) < 0.8


def test_settings_are_validated():
    with pytest.raises(ValueError):
        settings_from({"unassigned": "drop"})
    with pytest.raises(ValueError):
        settings_from({"evidence_threshold": 0.1})
