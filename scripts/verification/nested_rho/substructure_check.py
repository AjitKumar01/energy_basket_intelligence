"""Model-free screen: do merchandise categories contain substitute sub-groups?

For products j, k in one category, training trips give observed co-purchases O_jk and a
household-adjusted expectation E_jk = sum_h q_h a_hj a_hk, with a_hj the household's trips
containing j and q_h = sum_t s_t^2 / (sum_t s_t)^2 (independent placement of each household's
purchases across its own trips). A set of pairs has shortfall log((sum O + 1) / (sum E + 1));
under a pair penalty rho this is roughly -rho, so differences between pair sets read as
differences in substitution strength.

Two tests per category, both held out so that no grouping is judged on the data that built it:

1. Split-half clustering. Cluster products on half of the households (average linkage on the
   pair shortfall), then on the other half compare within-cluster with between-cluster
   shortfall. Null: random groupings with the same cluster sizes. Repeated over splits.
2. Declared metadata (real data only): same-manufacturer pairs (UPC manufacturer prefix)
   against different-manufacturer pairs, with a household bootstrap interval.

Calibration: the screen is also run on the known-truth decision-experiment worlds A (flat
truth) and B (nested truth), where the right answer per category is known.

Run from the repository root:
  python scripts/verification/nested_rho/substructure_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "nested_rho_decision"
MIN_LINES = 30
SPLITS = 10
NULL_DRAWS = 300
CLUSTERS = (2, 3)


def pair_stats(Y, hh, n_households):
    """O [J, J] and E [J, J] from a trip-by-product 0/1 sparse matrix."""
    Y = sparse.csr_matrix(Y, dtype=np.float64)
    O = (Y.T @ Y).toarray()
    np.fill_diagonal(O, 0)
    size = np.asarray(Y.sum(1)).ravel()
    H = sparse.csr_matrix((np.ones(len(hh)), (hh, np.arange(len(hh)))), shape=(n_households, len(hh)))
    A = (H @ Y).toarray()
    S = H @ size
    S2 = H @ size ** 2
    q = np.where(S > 0, S2 / np.maximum(S, 1) ** 2, 0.0)
    E = (A * q[:, None]).T @ A
    E = E - np.diag(np.diag(E))
    # same-trip pairs of a product with itself are excluded; E also counts j, k placed on one trip
    return O, E


def shortfall(O, E, mask):
    return float(np.log((O[mask].sum() + 1.0) / (E[mask].sum() + 1.0)))


def split_half(Y, hh, n_households, items, rng):
    households = np.unique(hh)
    results = {k: [] for k in CLUSTERS}
    for _ in range(SPLITS):
        half = rng.permutation(households)[: len(households) // 2]
        in_a = np.isin(hh, half)
        Oa, Ea = pair_stats(Y[in_a][:, items], hh[in_a], n_households)
        Ob, Eb = pair_stats(Y[~in_a][:, items], hh[~in_a], n_households)
        score = np.log((Oa + 1.0) / (Ea + 1.0))
        D = score - score.min()
        np.fill_diagonal(D, 0)
        Z = linkage(squareform((D + D.T) / 2, checks=False), method="average")
        off = ~np.eye(len(items), dtype=bool)
        for k in CLUSTERS:
            labels = fcluster(Z, k, criterion="maxclust")
            if len(np.unique(labels)) < 2:
                continue

            def contrast(lab):
                same = (lab[:, None] == lab[None]) & off
                return shortfall(Ob, Eb, same) - shortfall(Ob, Eb, ~same & off)
            observed = contrast(labels)
            null = np.array([contrast(rng.permutation(labels)) for _ in range(NULL_DRAWS)])
            results[k].append({"contrast": observed, "null_mean": float(null.mean()),
                               "null_sd": float(null.std()),
                               "p_value": float((1 + (null <= observed).sum()) / (1 + len(null))),
                               "cluster_sizes": np.bincount(labels)[1:].tolist()})
    summary = {}
    for k, rows in results.items():
        if rows:
            summary[f"k{k}"] = {
                "median_contrast": float(np.median([r["contrast"] for r in rows])),
                "median_null_mean": float(np.median([r["null_mean"] for r in rows])),
                "median_excess_over_null": float(np.median([r["contrast"] - r["null_mean"] for r in rows])),
                "median_p_value": float(np.median([r["p_value"] for r in rows])),
                "splits_p_below_0.05": int(sum(r["p_value"] < 0.05 for r in rows)),
                "splits": len(rows),
            }
    return summary


def metadata_contrast(Y, hh, n_households, items, groups, rng, reps=500):
    """Same-group minus different-group shortfall with a household bootstrap."""
    Ysub = sparse.csr_matrix(Y[:, items])
    households = np.unique(hh)
    same = (groups[:, None] == groups[None]) & ~np.eye(len(items), dtype=bool)
    diff = (groups[:, None] != groups[None])
    if same.sum() == 0 or diff.sum() == 0:
        return None
    # per-household sums of O and E over the two pair sets
    per = np.zeros((n_households, 4))
    order = np.argsort(hh, kind="stable")
    bounds = np.searchsorted(hh[order], households)
    ends = np.append(bounds[1:], len(order))
    for h, start, end in zip(households, bounds, ends):
        rows = order[start:end]
        O, E = pair_stats(Ysub[rows], np.zeros(len(rows), dtype=int), 1)
        per[h] = [O[same].sum(), E[same].sum(), O[diff].sum(), E[diff].sum()]

    def contrast(weights):
        t = weights @ per
        return float(np.log((t[0] + 1) / (t[1] + 1)) - np.log((t[2] + 1) / (t[3] + 1)))
    base = np.zeros(n_households); base[households] = 1
    draws = []
    for _ in range(reps):
        w = np.bincount(rng.choice(households, len(households)), minlength=n_households).astype(float)
        draws.append(contrast(w))
    t = base @ per
    return {"same_group_shortfall": float(np.log((t[0] + 1) / (t[1] + 1))),
            "different_group_shortfall": float(np.log((t[2] + 1) / (t[3] + 1))),
            "contrast": contrast(base),
            "interval_95": [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))],
            "same_group_pairs": int(same.sum() // 2), "groups": int(len(np.unique(groups)))}


def run_synthetic(world):
    d = np.load(OUT / world / "world.npz", allow_pickle=True)
    Y, hh, week = d["Y"], d["hh"], d["week"]
    train = week < 16
    Y, hh = Y[train], hh[train]
    lines = Y.sum(0)
    rng = np.random.default_rng(17)
    out = {}
    for c in range(6):
        items = np.array([j for j in range(c * 10, c * 10 + 10) if lines[j] >= MIN_LINES])
        out[f"cat{c}"] = {"products": len(items),
                          "category_shortfall": shortfall(*pair_stats(Y[:, items], hh, hh.max() + 1),
                                                          ~np.eye(len(items), dtype=bool)),
                          "split_half": split_half(Y, hh, hh.max() + 1, items, rng)}
    return out


def run_erim():
    base = ROOT / "data/erim_basket/model_input_availability_category/basket_input"
    baskets = pd.read_parquet(base / "baskets.parquet", columns=["BASKET_ID", "user_id", "item_id", "split"])
    baskets = baskets[baskets["split"] == "train"]
    items = pd.read_parquet(base / "items.parquet")
    trip_codes, trip = np.unique(baskets["BASKET_ID"].to_numpy(), return_inverse=True)
    J = int(items["item_id"].max()) + 1
    Y = sparse.csr_matrix((np.ones(len(baskets)), (trip, baskets["item_id"].to_numpy())),
                          shape=(len(trip_codes), J))
    Y.data[:] = 1.0
    hh_of_trip = np.zeros(len(trip_codes), dtype=int)
    hh_of_trip[trip] = baskets["user_id"].to_numpy()
    n_h = int(hh_of_trip.max()) + 1
    lines = np.asarray(Y.sum(0)).ravel()
    manufacturer = items.sort_values("item_id")["product_id"].str.split(":").str[1].str[2:7].to_numpy()
    rng = np.random.default_rng(23)
    out = {}
    for category, frame in items.groupby("category"):
        ids = np.sort(frame["item_id"].to_numpy())
        ids = ids[lines[ids] >= MIN_LINES]
        O, E = pair_stats(Y[:, ids], hh_of_trip, n_h)
        out[category] = {
            "products": int(len(ids)), "train_lines": int(lines[ids].sum()),
            "category_shortfall": shortfall(O, E, ~np.eye(len(ids), dtype=bool)),
            "split_half": split_half(Y.tocsr(), hh_of_trip, n_h, ids, rng),
            "manufacturer": metadata_contrast(Y.tocsr(), hh_of_trip, n_h, ids, manufacturer[ids], rng),
        }
        print(category, json.dumps(out[category]["split_half"]), flush=True)
    return out


def main():
    report = {"min_train_lines": MIN_LINES, "splits": SPLITS, "null_draws": NULL_DRAWS}
    for world in ("A", "B"):
        if (OUT / world / "world.npz").exists():
            report[f"synthetic_{world}"] = run_synthetic(world)
            print(world, json.dumps({c: v["split_half"] for c, v in report[f"synthetic_{world}"].items()}), flush=True)
    report["erim"] = run_erim()
    (OUT / "substructure_check.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
