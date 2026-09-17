"""Model-free substitute groups from household-level co-purchase evidence.

Builds the product partition once, from training baskets only, before any model is fitted:

1. Pair evidence. For products j, k and each household h, over training trips t with basket
   size s_t:  p_tj = stocked_tj * min(1, a_hj * s_t / S_hj), where a_hj counts h's trips
   containing j and S_hj sums s_t over h's trips on which j was stocked. Then
       O_jk = sum_t y_tj y_tk stocked_tj stocked_tk,    E_jk = sum_t p_tj p_tk stocked_tk stocked_tj,
       score_jk = log((O_jk + kappa) / (E_jk + kappa)),
   with kappa the Gamma-Poisson (negative binomial) maximum-likelihood shrinkage over all
   evaluated pairs. Stocking is per product (store availability panel), a product-level
   stand-in for "trips where both were stocked". Pairs with E_jk below
   ``minimum_expected_cooccurrence`` carry no evidence (score treated as 0).
   Reading: score <= evidence_threshold is a can-link (substitutes); score >=
   complement_threshold with evidence is a cannot-link (complements).
2. Groups. Within each boundary block (a catalogue category, or the whole catalogue), start
   from singletons and repeatedly merge the two groups with the most negative average
   between-group score, subject to: merged size <= maximum_group_size; merged average
   within-group score <= evidence_threshold; no cannot-link pair inside. Then move single
   products between groups while the objective sum_{groups} sum_{j<k} -score_jk increases and
   the constraints hold. Deterministic.
3. Products left alone stay singletons (no pairs, inert penalty) or are pooled into one
   residual group per category (``unassigned``).

Every setting is declared in the dataset config before the partition is built.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

EVIDENCE_DEFAULTS = {
    "category_boundary": True,
    "minimum_expected_cooccurrence": 5.0,
    "evidence_threshold": -0.3,
    "complement_threshold": 0.3,
    "maximum_group_size": 24,
    "unassigned": "singleton",
}
UNASSIGNED_RULES = ("singleton", "residual_per_category")
TRIP_CHUNK = 20000


@dataclass
class Evidence:
    items: np.ndarray        # item ids of the block, ascending
    observed: np.ndarray     # [n, n]
    expected: np.ndarray     # [n, n]


def settings_from(config: dict) -> dict:
    settings = {**EVIDENCE_DEFAULTS, **{k: v for k, v in config.items() if k in EVIDENCE_DEFAULTS}}
    if settings["unassigned"] not in UNASSIGNED_RULES:
        raise ValueError(f"unassigned must be one of {UNASSIGNED_RULES}")
    if not settings["evidence_threshold"] < 0 < settings["complement_threshold"]:
        raise ValueError("evidence_threshold must be negative and complement_threshold positive")
    if int(settings["maximum_group_size"]) < 2:
        raise ValueError("maximum_group_size must be at least 2")
    return settings


def load_trips(basket_input: Path, split: str = "train", households: np.ndarray | None = None):
    """Trip-by-product purchase matrix, trip household, and trip stocking lookup."""
    baskets = pd.read_parquet(basket_input / "baskets.parquet",
                              columns=["BASKET_ID", "user_id", "WEEK_NO", "item_id", "store_id", "split"])
    baskets = baskets[baskets.split == split]
    if households is not None:
        baskets = baskets[baskets.user_id.isin(households)]
    items = pd.read_parquet(basket_input / "items.parquet").sort_values("item_id").reset_index(drop=True)
    trip_codes, trip = np.unique(baskets.BASKET_ID.to_numpy(), return_inverse=True)
    J = len(items)
    Y = sparse.csr_matrix((np.ones(len(baskets)), (trip, baskets.item_id.to_numpy())), shape=(len(trip_codes), J))
    Y.data[:] = 1.0
    first = baskets.groupby(trip).first()
    household = first.user_id.to_numpy()
    store = first.store_id.to_numpy()
    week = first.WEEK_NO.to_numpy()
    availability = basket_input / "availability.npz"
    first_period = None
    if availability.is_file():
        with np.load(availability) as panel:
            first_period = panel["first_period"].astype(np.int64)
    return items, Y, household, store, week, first_period


def _stocked(first_period, store, week, cols):
    if first_period is None:
        return np.ones((len(store), len(cols)), dtype=bool)
    return first_period[np.ix_(cols, np.arange(first_period.shape[1]))][:, store].T <= week[:, None]


def pair_evidence(Y, household, store, week, first_period, cols: np.ndarray) -> Evidence:
    """Observed and household-expected co-purchase counts among products ``cols``."""
    Ys = Y[:, cols]
    size = np.asarray(Y.sum(1)).ravel()                                # full basket size
    codes, hh = np.unique(household, return_inverse=True)
    n_h, n = len(codes), len(cols)
    A = np.zeros((n_h, n))
    S = np.zeros((n_h, n))
    for start in range(0, Ys.shape[0], TRIP_CHUNK):
        sl = slice(start, start + TRIP_CHUNK)
        stocked = _stocked(first_period, store[sl], week[sl], cols)
        yb = Ys[sl].toarray() * stocked
        np.add.at(A, hh[sl], yb)
        np.add.at(S, hh[sl], stocked * size[sl, None])
    O = np.zeros((n, n))
    E = np.zeros((n, n))
    for start in range(0, Ys.shape[0], TRIP_CHUNK):
        sl = slice(start, start + TRIP_CHUNK)
        stocked = _stocked(first_period, store[sl], week[sl], cols)
        yb = Ys[sl].toarray() * stocked
        rate = np.divide(A[hh[sl]], S[hh[sl]], out=np.zeros((len(hh[sl]), n)), where=S[hh[sl]] > 0)
        P = np.minimum(1.0, rate * size[sl, None]) * stocked
        O += yb.T @ yb
        E += P.T @ P
    np.fill_diagonal(O, 0.0)
    np.fill_diagonal(E, 0.0)
    return Evidence(items=np.asarray(cols), observed=O, expected=E)


def estimate_kappa(observed: np.ndarray, expected: np.ndarray) -> float:
    """Gamma(kappa, kappa) mixing of Poisson(E * theta): maximum marginal likelihood."""
    keep = expected > 0
    o, e = observed[keep], expected[keep]
    if len(o) == 0:
        return 1.0

    def negative(log_kappa):
        k = math.exp(log_kappa)
        return -np.sum(gammaln(o + k) - gammaln(k) + k * np.log(k / (k + e)) + o * np.log(e / (k + e)))
    result = minimize_scalar(negative, bounds=(-6.0, 12.0), method="bounded")
    return float(math.exp(result.x))


def cluster_block(score: np.ndarray, evidence: np.ndarray, settings: dict) -> list[list[int]]:
    """Constrained average-linkage merging, then single-product moves (indices into the block)."""
    n = score.shape[0]
    s = np.where(evidence, score, 0.0)
    np.fill_diagonal(s, 0.0)
    cannot = evidence & (score >= settings["complement_threshold"])
    threshold = float(settings["evidence_threshold"])
    cap = int(settings["maximum_group_size"])
    groups = [[i] for i in range(n)]

    def within_sum(members):
        idx = np.asarray(members)
        return s[np.ix_(idx, idx)].sum() / 2.0

    def allowed(members):
        if len(members) > cap:
            return False
        idx = np.asarray(members)
        if len(idx) >= 2:
            if cannot[np.ix_(idx, idx)].any():
                return False
            pairs = len(idx) * (len(idx) - 1) / 2.0
            if within_sum(members) / pairs > threshold:
                return False
        return True

    while True:
        m = len(groups)
        if m < 2:
            break
        membership = np.zeros((m, n))
        for g, members in enumerate(groups):
            membership[g, members] = 1.0
        between = membership @ s @ membership.T
        sizes = membership.sum(1)
        average = between / np.outer(sizes, sizes)
        iu = np.triu_indices(m, 1)
        order = np.argsort(average[iu], kind="stable")
        merged = False
        for idx in order:
            a, b = int(iu[0][idx]), int(iu[1][idx])
            if average[a, b] >= 0:
                break
            candidate = sorted(groups[a] + groups[b])
            if allowed(candidate):
                groups = [g for i, g in enumerate(groups) if i not in (a, b)] + [candidate]
                groups.sort(key=lambda g: g[0])
                merged = True
                break
        if not merged:
            break

    for _ in range(50):                                   # local refinement
        moved = False
        for j in range(n):
            source = next(g for g in groups if j in g)
            rest = [k for k in source if k != j]
            if rest and not allowed(rest):
                continue
            gain_leave = s[j, rest].sum() if rest else 0.0
            best, best_gain = None, 1e-9
            for target in groups:
                if target is source:
                    continue
                gain = -s[j, target].sum() + gain_leave
                if gain > best_gain and allowed(sorted(target + [j])):
                    best, best_gain = target, gain
            if best is not None:
                source.remove(j)
                best.append(j)
                best.sort()
                groups = sorted([g for g in groups if g], key=lambda g: g[0])
                moved = True
        if not moved:
            break
    return [sorted(g) for g in groups]


def build_evidence_partition(basket_input: Path, settings: dict, households: np.ndarray | None = None):
    """Group id per product, plus a manifest-ready summary."""
    items, Y, household, store, week, first_period = load_trips(basket_input, "train", households)
    J = len(items)
    category = items.COMMODITY_DESC.astype(str).to_numpy()
    if settings["category_boundary"]:
        blocks = [np.flatnonzero(category == c) for c in sorted(set(category))]
    else:
        blocks = [np.arange(J)]
    evidence = [pair_evidence(Y, household, store, week, first_period, cols) for cols in blocks]
    all_o = np.concatenate([e.observed[np.triu_indices(len(e.items), 1)] for e in evidence])
    all_e = np.concatenate([e.expected[np.triu_indices(len(e.items), 1)] for e in evidence])
    kappa = estimate_kappa(all_o, all_e)
    minimum = float(settings["minimum_expected_cooccurrence"])
    group_members, summary_pairs = [], {"evaluated": 0, "with_evidence": 0, "can_link": 0, "cannot_link": 0}
    group_scores = []
    for block in evidence:
        score = np.log((block.observed + kappa) / (block.expected + kappa))
        has = block.expected >= minimum
        np.fill_diagonal(has, False)
        iu = np.triu_indices(len(block.items), 1)
        summary_pairs["evaluated"] += len(iu[0])
        summary_pairs["with_evidence"] += int(has[iu].sum())
        summary_pairs["can_link"] += int((has & (score <= settings["evidence_threshold"]))[iu].sum())
        summary_pairs["cannot_link"] += int((has & (score >= settings["complement_threshold"]))[iu].sum())
        for members in cluster_block(score, has, settings):
            group_members.append(block.items[members])
            idx = np.asarray(members)
            if len(idx) >= 2:
                sub = np.where(has, score, 0.0)[np.ix_(idx, idx)]
                group_scores.append(float(sub[np.triu_indices(len(idx), 1)].mean()))
            else:
                group_scores.append(None)
    assigned = [g for g in group_members if len(g) >= 2]
    singles = [g for g in group_members if len(g) < 2]
    if settings["unassigned"] == "residual_per_category" and singles:
        pooled = {}
        for g in singles:
            pooled.setdefault(category[g[0]], []).extend(g.tolist())
        final = assigned + [np.array(sorted(v)) for _, v in sorted(pooled.items())]
    else:
        final = group_members
    cat_rank = {c: i for i, c in enumerate(sorted(set(category)))}
    final = sorted(final, key=lambda g: (cat_rank[category[g[0]]], int(g.min())))
    group_id = np.full(J, -1, dtype=np.int32)
    for gid, members in enumerate(final):
        group_id[members] = gid
    sizes = np.bincount(group_id)
    summary = {
        "settings": settings, "kappa": kappa, "pairs": summary_pairs,
        "groups_with_pairs": int(len(assigned)), "unassigned_products": int(sum(len(g) for g in singles)),
        "group_size": {"singletons": int((sizes == 1).sum()), "median": float(np.median(sizes)),
                       "maximum": int(sizes.max())},
        "multi_product_group_mean_scores": [round(x, 4) for x in group_scores if x is not None],
        "training_trips": int(Y.shape[0]),
    }
    return group_id, summary
