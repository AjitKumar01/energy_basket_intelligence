"""Model-free substitute groups from household-level co-purchase evidence.

Builds the product partition once, from training baskets only, before any model is fitted.
Method, proofs and complexity: paper/SUBSTITUTION_EVIDENCE_PARTITION.md.

1. Pair evidence. For household h and training trip t with basket size s_t:
       p_tj = st_tj * min(1, a_hj * s_t / S_hj),   a_hj = sum_t y_tj st_tj,   S_hj = sum_t s_t st_tj,
       O_jk = sum_t y_tj y_tk st_tj st_tk,          E_jk = sum_t p_tj p_tk,
       score_jk = log((O_jk + kappa) / (E_jk + kappa)),
   where st_tj marks product j as stocked at the trip's store-week and kappa is the
   Gamma-Poisson maximum-likelihood shrinkage. A household contributes only to pairs of
   products it bought (p_tj = 0 when a_hj = 0), so E and O are accumulated household by
   household over those products: exact, and linear in trips.
2. Groups. Pairs with E_jk >= minimum_expected_cooccurrence carry evidence (others score 0).
   Within each block (a category, or the whole catalogue), greedy merging of the feasible
   group pair with the most negative average between-group score (sparse sums and a lazy
   heap), then single-product moves, including out to a singleton, while the objective
   sum_{groups} sum_{j<k} -score_jk increases. Constraints: size <= maximum_group_size,
   average within-group score <= evidence_threshold, no pair with
   score >= complement_threshold.
3. Leftovers stay singletons or are pooled into one residual group per category; the
   manifest records each group's role.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import gammaln

EVIDENCE_DEFAULTS = {
    "category_boundary": True,
    "minimum_expected_cooccurrence": 5.0,
    "evidence_threshold": -0.3,
    "complement_threshold": 0.3,
    "maximum_group_size": 24,
    "unassigned": "singleton",
    "maximum_dense_pairs": 50_000_000,
}
UNASSIGNED_RULES = ("singleton", "residual_per_category")
MOVE_PASSES = 50
GAIN_TOLERANCE = 1e-9


@dataclass
class Evidence:
    items: np.ndarray        # item ids of the block, ascending
    observed: np.ndarray     # [n, n]
    expected: np.ndarray     # [n, n]


def settings_from(config: dict) -> dict:
    """Declared settings with defaults, validated (raises ValueError)."""
    unknown = set(config).difference(EVIDENCE_DEFAULTS).difference({"partition"})
    if unknown:
        raise ValueError(f"unknown substitution-evidence settings {sorted(unknown)}")
    settings = {**EVIDENCE_DEFAULTS, **{k: v for k, v in config.items() if k in EVIDENCE_DEFAULTS}}
    if not isinstance(settings["category_boundary"], bool):
        raise ValueError("category_boundary must be true or false")
    if settings["unassigned"] not in UNASSIGNED_RULES:
        raise ValueError(f"unassigned must be one of {UNASSIGNED_RULES}")
    if not float(settings["evidence_threshold"]) < 0 < float(settings["complement_threshold"]):
        raise ValueError("evidence_threshold must be negative and complement_threshold positive")
    if float(settings["minimum_expected_cooccurrence"]) < 0:
        raise ValueError("minimum_expected_cooccurrence must be nonnegative")
    if int(settings["maximum_group_size"]) < 2:
        raise ValueError("maximum_group_size must be at least 2")
    if int(settings["maximum_dense_pairs"]) < 1:
        raise ValueError("maximum_dense_pairs must be positive")
    return settings


# ----------------------------------------------------------------------------- evidence
def _training_lines(basket_input: Path, split: str, households):
    lines = pd.read_parquet(basket_input / "baskets.parquet",
                            columns=["BASKET_ID", "user_id", "WEEK_NO", "item_id", "store_id", "split"])
    lines = lines[lines.split == split]
    if households is not None:
        lines = lines[lines.user_id.isin(households)]
    return lines.drop_duplicates(["BASKET_ID", "item_id"])


def household_evidence(basket_input: Path, blocks: list[np.ndarray], split: str = "train",
                       households=None, maximum_dense_pairs: int = EVIDENCE_DEFAULTS["maximum_dense_pairs"]
                       ) -> tuple[list[Evidence], int]:
    """Exact O and E for every block, accumulated household by household.

    blocks: disjoint arrays of item ids. Returns the per-block evidence and the trip count.
    """
    for block in blocks:
        if len(block) ** 2 > int(maximum_dense_pairs):
            raise ValueError(
                f"a block of {len(block)} products needs {len(block) ** 2:,} dense pair entries, above "
                f"maximum_dense_pairs={int(maximum_dense_pairs):,}; use category_boundary or raise the limit")
    items = pd.read_parquet(basket_input / "items.parquet", columns=["item_id"])
    J = int(items.item_id.max()) + 1
    block_of = np.full(J, -1)
    local = np.zeros(J, dtype=np.int64)
    for b, block in enumerate(blocks):
        block_of[block] = b
        local[block] = np.arange(len(block))
    first_period = None
    availability = basket_input / "availability.npz"
    if availability.is_file():
        with np.load(availability) as panel:
            first_period = panel["first_period"].astype(np.int64)
    lines = _training_lines(basket_input, split, households)
    trip_codes, trip = np.unique(lines.BASKET_ID.to_numpy(), return_inverse=True)
    size = np.bincount(trip).astype(float)                       # full basket size s_t
    trips = pd.DataFrame({"trip": trip, "user": lines.user_id.to_numpy(), "store": lines.store_id.to_numpy(),
                          "week": lines.WEEK_NO.to_numpy()}).drop_duplicates("trip").sort_values("trip")
    trip_store, trip_week, trip_user = trips.store.to_numpy(), trips.week.to_numpy(), trips.user.to_numpy()
    observed = [np.zeros((len(b), len(b))) for b in blocks]
    expected = [np.zeros((len(b), len(b))) for b in blocks]
    item = lines.item_id.to_numpy()
    order = np.lexsort((item, trip, trip_user[trip]))
    user_sorted = trip_user[trip][order]
    starts = np.flatnonzero(np.r_[True, user_sorted[1:] != user_sorted[:-1]])
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        rows = order[start:end]
        h_trips, trip_pos = np.unique(trip[rows], return_inverse=True)
        h_items_all = item[rows]
        for b in np.unique(block_of[h_items_all]):
            if b < 0:
                continue
            in_block = block_of[h_items_all] == b
            products, prod_pos = np.unique(h_items_all[in_block], return_inverse=True)
            Y = np.zeros((len(h_trips), len(products)))
            Y[trip_pos[in_block], prod_pos] = 1.0
            if first_period is None:
                stocked = np.ones_like(Y)
            else:
                stocked = (first_period[products][:, trip_store[h_trips]].T <= trip_week[h_trips][:, None]).astype(float)
            s = size[h_trips][:, None]
            yb = Y * stocked
            a = yb.sum(0)
            S = (stocked * s).sum(0)
            rate = np.divide(a, S, out=np.zeros_like(a), where=S > 0)
            P = np.minimum(1.0, rate[None, :] * s) * stocked
            idx = local[products]
            observed[b][np.ix_(idx, idx)] += yb.T @ yb
            expected[b][np.ix_(idx, idx)] += P.T @ P
    evidence = []
    for b, block in enumerate(blocks):
        np.fill_diagonal(observed[b], 0.0)
        np.fill_diagonal(expected[b], 0.0)
        evidence.append(Evidence(items=np.asarray(block), observed=observed[b], expected=expected[b]))
    return evidence, int(len(trip_codes))


def estimate_kappa(observed: np.ndarray, expected: np.ndarray) -> float:
    """Gamma(kappa, kappa) mixing of Poisson(E * theta): maximum marginal likelihood."""
    keep = expected > 0
    o, e = observed[keep], expected[keep]
    if len(o) == 0:
        return 1.0

    def negative(log_kappa):
        k = math.exp(log_kappa)
        return -np.sum(gammaln(o + k) - gammaln(k) + k * np.log(k / (k + e)) + o * np.log(e / (k + e)))
    return float(math.exp(minimize_scalar(negative, bounds=(-6.0, 12.0), method="bounded").x))


# ----------------------------------------------------------------------------- grouping
def cluster_block(score: np.ndarray, evidence: np.ndarray, settings: dict) -> list[list[int]]:
    """Constrained sparse average-linkage merging, then single-product moves.

    Indices are positions in the block. Only evidence edges are stored; pairs without
    evidence score 0 and never make an average negative.
    """
    n = score.shape[0]
    threshold = float(settings["evidence_threshold"])
    cap = int(settings["maximum_group_size"])
    upper = np.triu(evidence, 1)
    ei, ej = np.nonzero(upper)
    values = score[ei, ej].astype(float)
    cannot = (values >= float(settings["complement_threshold"])).astype(int)

    # ---- merging: sums between groups are shared [G, K] lists in both directions
    between = [dict() for _ in range(n)]
    for i, j, v, c in zip(ei.tolist(), ej.tolist(), values.tolist(), cannot.tolist()):
        cell = [v, c]
        between[i][j] = cell
        between[j][i] = cell
    size = [1] * n
    within = [0.0] * n
    first = list(range(n))
    members = [[i] for i in range(n)]
    version = [0] * n
    alive = [True] * n
    heap = []

    def push(a, b):
        g = between[a][b][0]
        average = g / (size[a] * size[b])
        if average < 0:
            lo, hi = (a, b) if first[a] < first[b] else (b, a)
            heapq.heappush(heap, (average, first[lo], first[hi], lo, hi, version[lo], version[hi]))

    for i, j, v in zip(ei.tolist(), ej.tolist(), values.tolist()):
        if v < 0:
            push(i, j)
    while heap:
        _, _, _, a, b, va, vb = heapq.heappop(heap)
        if not (alive[a] and alive[b]) or version[a] != va or version[b] != vb:
            continue
        g, k = between[a][b]
        merged = size[a] + size[b]
        if merged > cap or k > 0:
            continue
        total = within[a] + within[b] + g
        if total / (merged * (merged - 1) / 2.0) > threshold:
            continue
        keep, drop = (a, b) if first[a] < first[b] else (b, a)
        del between[keep][drop]
        for c, cell in between[drop].items():
            if c == keep:
                continue
            del between[c][drop]
            target = between[keep].get(c)
            if target is None:
                target = [0.0, 0]
                between[keep][c] = target
                between[c][keep] = target
            target[0] += cell[0]
            target[1] += cell[1]
        between[drop] = {}
        alive[drop] = False
        size[keep] = merged
        within[keep] = total
        members[keep] = members[keep] + members[drop]
        version[keep] += 1
        for c in between[keep]:
            push(keep, c)

    # ---- single-product moves with sparse product-to-group sums
    group = np.empty(n, dtype=np.int64)
    for gid in range(n):
        if alive[gid]:
            group[members[gid]] = gid
    neighbours = [[] for _ in range(n)]
    for i, j, v, c in zip(ei.tolist(), ej.tolist(), values.tolist(), cannot.tolist()):
        neighbours[i].append((j, v, c))
        neighbours[j].append((i, v, c))
    link = [dict() for _ in range(n)]          # L[j][g] = [sum of scores, cannot-link count]
    for j in range(n):
        for k, v, c in neighbours[j]:
            cell = link[j].setdefault(int(group[k]), [0.0, 0])
            cell[0] += v
            cell[1] += c
    group_size = {gid: size[gid] for gid in range(n) if alive[gid]}
    group_within = {gid: within[gid] for gid in range(n) if alive[gid]}
    next_id = n

    def pairs(m):
        return m * (m - 1) / 2.0

    for _ in range(MOVE_PASSES):
        moved = False
        for j in range(n):
            source = int(group[j])
            l_source = link[j].get(source, [0.0, 0])[0]
            remaining = group_size[source] - 1
            if remaining >= 2 and (group_within[source] - l_source) / pairs(remaining) > threshold:
                continue
            best, best_gain = None, GAIN_TOLERANCE
            if group_size[source] > 1 and l_source > best_gain:          # move out on its own
                best, best_gain = "alone", l_source
            for target in sorted(link[j]):
                if target == source or target not in group_size:
                    continue
                l_target, c_target = link[j][target]
                gain = l_source - l_target
                if gain <= best_gain + 1e-12 or c_target > 0 or group_size[target] + 1 > cap:
                    continue
                if (group_within[target] + l_target) / pairs(group_size[target] + 1) > threshold:
                    continue
                best, best_gain = target, gain
            if best is None:
                continue
            if best == "alone":
                best, l_target = next_id, 0.0
                next_id += 1
                group_size[best], group_within[best] = 0, 0.0
            else:
                l_target = link[j][best][0]
            group_within[source] -= l_source
            group_size[source] -= 1
            group_within[best] += l_target
            group_size[best] += 1
            group[j] = best
            for k, v, c in neighbours[j]:
                old = link[k][source]
                old[0] -= v
                old[1] -= c
                cell = link[k].setdefault(best, [0.0, 0])
                cell[0] += v
                cell[1] += c
            if group_size[source] == 0:
                del group_size[source], group_within[source]
            moved = True
        if not moved:
            break
    groups = {}
    for j in range(n):
        groups.setdefault(int(group[j]), []).append(j)
    return sorted((sorted(g) for g in groups.values()), key=lambda g: g[0])


def objective(score: np.ndarray, evidence: np.ndarray, groups: list[list[int]]) -> float:
    """sum over groups of sum_{j<k} -score_jk (no-evidence pairs contribute 0)."""
    s = np.where(evidence, score, 0.0)
    return float(sum(-s[np.ix_(g, g)][np.triu_indices(len(g), 1)].sum() for g in groups if len(g) > 1))


def build_evidence_partition(basket_input: Path, settings: dict, households=None):
    """Group id per product, plus a manifest-ready summary."""
    items = pd.read_parquet(basket_input / "items.parquet").sort_values("item_id").reset_index(drop=True)
    J = len(items)
    category = items.COMMODITY_DESC.astype(str).to_numpy()
    if settings["category_boundary"]:
        blocks = [np.flatnonzero(category == c) for c in sorted(set(category))]
    else:
        blocks = [np.arange(J)]
    evidence, trips = household_evidence(basket_input, blocks, "train", households,
                                         int(settings["maximum_dense_pairs"]))
    all_o = np.concatenate([e.observed[np.triu_indices(len(e.items), 1)] for e in evidence])
    all_e = np.concatenate([e.expected[np.triu_indices(len(e.items), 1)] for e in evidence])
    kappa = estimate_kappa(all_o, all_e)
    minimum = float(settings["minimum_expected_cooccurrence"])
    group_members, pair_summary, group_scores = [], {"evaluated": 0, "with_evidence": 0, "can_link": 0,
                                                     "cannot_link": 0}, []
    for block in evidence:
        score = np.log((block.observed + kappa) / (block.expected + kappa))
        has = block.expected >= minimum
        np.fill_diagonal(has, False)
        iu = np.triu_indices(len(block.items), 1)
        pair_summary["evaluated"] += len(iu[0])
        pair_summary["with_evidence"] += int(has[iu].sum())
        pair_summary["can_link"] += int((has & (score <= settings["evidence_threshold"]))[iu].sum())
        pair_summary["cannot_link"] += int((has & (score >= settings["complement_threshold"]))[iu].sum())
        for members in cluster_block(score, has, settings):
            group_members.append(block.items[members])
            if len(members) >= 2:
                sub = np.where(has, score, 0.0)[np.ix_(members, members)]
                group_scores.append(float(sub[np.triu_indices(len(members), 1)].mean()))
    assigned = [g for g in group_members if len(g) >= 2]
    singles = [g for g in group_members if len(g) < 2]
    roles = {}
    final = [(g, "evidence") for g in assigned]
    if settings["unassigned"] == "residual_per_category" and singles:
        pooled = {}
        for g in singles:
            pooled.setdefault(category[g[0]], []).extend(g.tolist())
        final += [(np.array(sorted(v)), "residual") for _, v in sorted(pooled.items())]
    else:
        final += [(g, "singleton") for g in singles]
    cat_rank = {c: i for i, c in enumerate(sorted(set(category)))}
    final.sort(key=lambda gr: (cat_rank[category[gr[0][0]]], int(gr[0].min())))
    group_id = np.full(J, -1, dtype=np.int32)
    for gid, (members, role) in enumerate(final):
        group_id[members] = gid
        roles.setdefault(role, []).append(gid)
    sizes = np.bincount(group_id)
    summary = {
        "settings": settings, "kappa": kappa, "pairs": pair_summary, "training_trips": trips,
        "groups_by_role": {role: len(ids) for role, ids in roles.items()},
        "residual_group_ids": roles.get("residual", []),
        "residual_groups_note": ("residual groups pool unassigned products per category and are exempt "
                                 "from the size and homogeneity constraints") if "residual" in roles else None,
        "unassigned_products": int(sum(len(g) for g in singles)),
        "group_size": {"singletons": int((sizes == 1).sum()), "median": float(np.median(sizes)),
                       "maximum": int(sizes.max())},
        "evidence_group_mean_scores": [round(x, 4) for x in group_scores],
    }
    return group_id, summary
