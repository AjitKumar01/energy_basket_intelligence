"""Tests for the model-free substitution-evidence partition."""
import itertools

import numpy as np
import pandas as pd
import pytest

import evidence_partition as ep
from evidence_partition import cluster_block, estimate_kappa, household_evidence, objective, settings_from


def block(score):
    score = np.array(score, dtype=float)
    return score, ~np.eye(len(score), dtype=bool)


def greedy_reference(score, evidence, settings):
    """Dense brute-force greedy merging: the most negative feasible average, ties by min ids."""
    s = np.where(evidence, score, 0.0)
    cannot = evidence & (score >= settings["complement_threshold"])
    groups = [[i] for i in range(len(s))]
    while True:
        best = None
        for a, b in itertools.combinations(range(len(groups)), 2):
            ga, gb = groups[a], groups[b]
            average = s[np.ix_(ga, gb)].sum() / (len(ga) * len(gb))
            merged = ga + gb
            if average >= 0 or len(merged) > settings["maximum_group_size"]:
                continue
            if cannot[np.ix_(merged, merged)].any():
                continue
            pairs = len(merged) * (len(merged) - 1) / 2
            if s[np.ix_(merged, merged)].sum() / 2 / pairs > settings["evidence_threshold"]:
                continue
            key = (average, min(ga[0], gb[0]), max(ga[0], gb[0]))
            if best is None or key < best[0]:
                best = (key, a, b)
        if best is None:
            return sorted(sorted(g) for g in groups)
        _, a, b = best
        groups = [g for i, g in enumerate(groups) if i not in (a, b)] + [sorted(groups[a] + groups[b])]
        groups.sort(key=lambda g: g[0])


def random_case(rng, n):
    score = rng.normal(-0.2, 0.8, (n, n))
    score = (score + score.T) / 2
    evidence = rng.random((n, n)) < 0.6
    evidence = evidence | evidence.T
    np.fill_diagonal(evidence, False)
    return score, evidence


def feasible(groups, score, evidence, settings):
    s = np.where(evidence, score, 0.0)
    for g in groups:
        if len(g) > settings["maximum_group_size"]:
            return False
        if len(g) >= 2:
            if (evidence & (score >= settings["complement_threshold"]))[np.ix_(g, g)].any():
                return False
            if s[np.ix_(g, g)].sum() / 2 / (len(g) * (len(g) - 1) / 2) > settings["evidence_threshold"]:
                return False
    return True


def test_sparse_merging_matches_the_dense_greedy_reference(monkeypatch):
    rng = np.random.default_rng(1)
    monkeypatch.setattr(ep, "MOVE_PASSES", 0)
    for _ in range(30):
        score, evidence = random_case(rng, int(rng.integers(4, 25)))
        settings = settings_from({"maximum_group_size": int(rng.integers(2, 8))})
        assert cluster_block(score, evidence, settings) == greedy_reference(score, evidence, settings)


def test_moves_end_feasible_and_locally_optimal():
    rng = np.random.default_rng(2)
    for _ in range(30):
        n = int(rng.integers(4, 20))
        score, evidence = random_case(rng, n)
        settings = settings_from({"maximum_group_size": int(rng.integers(2, 8))})
        groups = cluster_block(score, evidence, settings)
        assert sorted(sum(groups, [])) == list(range(n)) and feasible(groups, score, evidence, settings)
        base = objective(score, evidence, groups)
        for j in range(n):
            source = next(i for i, g in enumerate(groups) if j in g)
            rest = [k for k in groups[source] if k != j]
            for t in list(range(len(groups))) + ["alone"]:
                if t == source:
                    continue
                trial = [list(g) for g in groups]
                trial[source] = rest
                if t == "alone":
                    if not rest:
                        continue
                    trial.append([j])
                else:
                    trial[t] = sorted(trial[t] + [j])
                trial = [g for g in trial if g]
                if feasible(trial, score, evidence, settings):
                    assert objective(score, evidence, trial) <= base + 1e-9


def test_strong_substitutes_merge_and_complements_never_share_a_group():
    s = np.zeros((5, 5))
    s[np.ix_([0, 1, 2], [0, 1, 2])] = -1.0
    score, evidence = block(s)
    assert cluster_block(score, evidence, settings_from({})) == [[0, 1, 2], [3], [4]]
    s = np.full((3, 3), -1.0)
    s[0, 2] = s[2, 0] = 0.5
    score, evidence = block(s)
    assert not any(0 in g and 2 in g for g in cluster_block(score, evidence, settings_from({})))


def test_pairs_without_evidence_do_not_link():
    assert cluster_block(np.full((2, 2), -2.0), np.zeros((2, 2), dtype=bool), settings_from({})) == [[0], [1]]


def tiny_bundle(tmp_path, rng):
    rows = []
    for trip in range(120):
        user, store, week = int(rng.integers(0, 6)), int(rng.integers(0, 2)), int(rng.integers(1, 9))
        for item in np.flatnonzero(rng.random(6) < 0.35):
            rows.append(dict(BASKET_ID=trip, user_id=user, WEEK_NO=week, item_id=int(item), store_id=store,
                             split="train"))
    baskets = pd.DataFrame(rows)
    baskets.to_parquet(tmp_path / "baskets.parquet")
    pd.DataFrame({"item_id": range(6)}).to_parquet(tmp_path / "items.parquet")
    first = rng.integers(0, 5, (6, 2)).astype(np.int16)
    np.savez(tmp_path / "availability.npz", first_period=first)
    return baskets, first


def test_household_evidence_equals_the_trip_level_formula(tmp_path):
    rng = np.random.default_rng(3)
    baskets, first = tiny_bundle(tmp_path, rng)
    trips = baskets.groupby("BASKET_ID").agg(user=("user_id", "first"), store=("store_id", "first"),
                                            week=("WEEK_NO", "first"), size=("item_id", "size"))
    Y = np.zeros((len(trips), 6))
    index = {t: i for i, t in enumerate(trips.index)}
    for row in baskets.itertuples():
        Y[index[row.BASKET_ID], row.item_id] = 1
    stocked = (first[:, trips.store.to_numpy()].T <= trips.week.to_numpy()[:, None]).astype(float)
    size = trips["size"].to_numpy()[:, None].astype(float)
    P = np.zeros_like(Y)
    for user in trips.user.unique():
        r = trips.user.to_numpy() == user
        a = (Y[r] * stocked[r]).sum(0)
        S = (stocked[r] * size[r]).sum(0)
        rate = np.divide(a, S, out=np.zeros_like(a), where=S > 0)
        P[r] = np.minimum(1, rate * size[r]) * stocked[r]
    O = (Y * stocked).T @ (Y * stocked)
    E = P.T @ P
    np.fill_diagonal(O, 0)
    np.fill_diagonal(E, 0)
    blocks = [np.array([0, 2, 4]), np.array([1, 3, 5])]
    evidence, n_trips = household_evidence(tmp_path, blocks)
    assert n_trips == len(trips)
    for block, ev in zip(blocks, evidence):
        assert np.allclose(ev.observed, O[np.ix_(block, block)])
        assert np.allclose(ev.expected, E[np.ix_(block, block)], rtol=1e-12, atol=1e-12)


def test_dense_pair_guard_fails_clearly(tmp_path):
    tiny_bundle(tmp_path, np.random.default_rng(4))
    with pytest.raises(ValueError, match="maximum_dense_pairs"):
        household_evidence(tmp_path, [np.arange(6)], maximum_dense_pairs=35)


def test_kappa_is_large_for_poisson_data_and_small_for_overdispersed_data():
    rng = np.random.default_rng(0)
    expected = rng.uniform(1, 20, 4000)
    assert estimate_kappa(rng.poisson(expected), expected) > 50
    assert 0.3 < estimate_kappa(rng.poisson(expected * rng.gamma(0.5, 2.0, 4000)), expected) < 0.8


def test_settings_are_validated():
    for bad in ({"unassigned": "drop"}, {"evidence_threshold": 0.1}, {"maximum_group_size": 1},
                {"category_boundary": "yes"}, {"colour": 1}):
        with pytest.raises(ValueError):
            settings_from(bad)
