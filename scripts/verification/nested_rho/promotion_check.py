"""Is ERIM's same-manufacturer co-purchase excess explained by promotions or shifting loyalty?

Follow-up to substructure_check.py (paper/ERIM_SUBSTRUCTURE_SCREEN.md). For each category it
compares same-manufacturer with different-manufacturer pairs, using a leave-one-trip-out
expectation so that finer household adjustments are not biased towards the observed counts:

    p_tj = min(1, (a_hj - y_tj) * s_t / (S_h - s_t)),   E_jk = sum_t p_tj p_tk,

where a_hj and S_h count household h's other trips in the adjustment cell. Contrast =
log((O_same + 1) / (E_same + 1)) - log((O_diff + 1) / (E_diff + 1)).

Variants:
  all_trips            household adjustment, every trip and pair
  unpromoted_flags     a pair on a trip counts only if neither product has display, feature
                       advertising or a special-price flag in that store-week
  unpromoted_strict    as above, and neither product sells >= 5% below its store's median price
  promoted_either      complement of unpromoted_strict (at least one of the two promoted)
  household_quarter    adjustment cell = household x 13-week block (shifting loyalty)
  household_quarter_unpromoted_strict   both

Calibration: the adjustment variants are also run on the known-truth synthetic worlds, which
have no promotions; the contrast must stay near 0 (world A) and near -1 for true nested lines
(world B cat0, cat2).

Run from the repository root:
  python scripts/verification/nested_rho/promotion_check.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "artifacts" / "nested_rho_decision"
MIN_LINES = 30
BOOT = 300
PRICE_CUT = 0.95


def pair_sums(Y, cell, keep, same, diff):
    """O and leave-one-trip-out E summed over same and different pair sets.

    Y [T, J] 0/1 trips x category products; cell [T] adjustment cell; keep [T, J] 0/1 whether a
    product counts on that trip (pairs count only if both are kept).
    """
    T, J = Y.shape
    size = Y.sum(1)
    order = np.unique(cell, return_inverse=True)[1]
    n = order.max() + 1
    A = np.zeros((n, J)); np.add.at(A, order, Y)
    S = np.zeros(n); np.add.at(S, order, size)
    other_s = S[order] - size
    rate = np.where(other_s[:, None] > 0, (A[order] - Y) / np.maximum(other_s, 1)[:, None], 0.0)
    P = np.minimum(1.0, rate * size[:, None]) * keep
    Yk = Y * keep
    O = Yk.T @ Yk
    E = P.T @ P
    return np.array([O[same].sum(), E[same].sum(), O[diff].sum(), E[diff].sum()])


def contrast(t):
    return float(np.log((t[0] + 1) / (t[1] + 1)) - np.log((t[2] + 1) / (t[3] + 1)))


def household_bootstrap(Y, cell, household, keep, same, diff, rng):
    """Contrast with a household bootstrap (households resampled with their trips)."""
    households, inverse = np.unique(household, return_inverse=True)
    per = np.zeros((len(households), 4))
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(len(households)))
    ends = np.append(bounds[1:], len(order))
    for i, (start, end) in enumerate(zip(bounds, ends)):
        rows = order[start:end]
        per[i] = pair_sums(Y[rows], cell[rows], keep[rows], same, diff)
    total = per.sum(0)
    draws = [contrast(np.bincount(rng.choice(len(households), len(households)),
                                  minlength=len(households)) @ per) for _ in range(BOOT)]
    return {"contrast": contrast(total), "interval_95": [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
            "same_observed": float(total[0]), "same_expected": float(total[1]),
            "different_observed": float(total[2]), "different_expected": float(total[3])}


def run_erim(rng):
    canonical = ROOT / "data/erim_basket/canonical"
    tx = pd.read_parquet(canonical / "transactions.parquet",
                         columns=["basket_id", "customer_id", "store_id", "period", "product_id", "category"])
    promo = pd.read_parquet(canonical / "promotions.parquet",
                            columns=["product_id", "store_id", "period", "display", "advertised", "special_price"])
    prices = pd.read_parquet(canonical / "store_week_prices.parquet",
                             columns=["product_id", "store_id", "period", "average_price", "price_source"])
    prices = prices[prices.price_source == "retail_aggregate"].copy()
    prices["regular"] = prices.groupby(["product_id", "store_id"]).average_price.transform("median")
    prices["price_cut"] = prices.average_price < PRICE_CUT * prices.regular
    flags = promo.assign(flag=promo.display | promo.advertised | promo.special_price)
    cells = flags[["product_id", "store_id", "period", "flag"]].merge(
        prices[["product_id", "store_id", "period", "price_cut"]], how="outer",
        on=["product_id", "store_id", "period"])
    cells["flag"] = cells.flag.astype("boolean").fillna(False).astype(bool)
    cells["price_cut"] = cells.price_cut.astype("boolean").fillna(False).astype(bool)
    covered = set(promo.product_id.str.split(":").str[0].unique())
    report = {}
    for category, frame in tx.groupby("category"):
        lines = frame.product_id.value_counts()
        products = np.array(sorted(lines[lines >= MIN_LINES].index))
        frame = frame[frame.product_id.isin(products)]
        trips = frame.groupby("basket_id").agg(customer_id=("customer_id", "first"),
                                              store_id=("store_id", "first"), period=("period", "first"))
        trip_index = pd.Series(np.arange(len(trips)), index=trips.index)
        pidx = pd.Series(np.arange(len(products)), index=products)
        Y = np.zeros((len(trips), len(products)))
        Y[trip_index[frame.basket_id].to_numpy(), pidx[frame.product_id].to_numpy()] = 1
        # promotion state of every category product at each trip's store-week
        cat_cells = cells[cells.product_id.isin(products)]
        sw = trips[["store_id", "period"]].drop_duplicates()
        grid = sw.merge(pd.DataFrame({"product_id": products}), how="cross").merge(
            cat_cells, how="left", on=["product_id", "store_id", "period"])
        grid["flag"] = grid.flag.astype("boolean").fillna(False).astype(bool)
        grid["price_cut"] = grid.price_cut.astype("boolean").fillna(False).astype(bool)
        key = trips.store_id.to_numpy() * 1000 + trips.period.to_numpy()
        sw_key = grid.store_id.to_numpy() * 1000 + grid.period.to_numpy()
        sw_codes, sw_inverse = np.unique(sw_key, return_inverse=True)
        flag_sw = np.zeros((len(sw_codes), len(products)), bool)
        cut_sw = np.zeros((len(sw_codes), len(products)), bool)
        col = pidx[grid.product_id].to_numpy()
        flag_sw[sw_inverse, col] = grid.flag.to_numpy()
        cut_sw[sw_inverse, col] = grid.price_cut.to_numpy()
        row = np.searchsorted(sw_codes, key)
        flag, cut = flag_sw[row], cut_sw[row]
        manufacturer = np.array([p.split(":")[1][2:7] for p in products])
        same = (manufacturer[:, None] == manufacturer[None]) & ~np.eye(len(products), dtype=bool)
        diff = manufacturer[:, None] != manufacturer[None]
        household = trips.customer_id.to_numpy()
        quarter = household * 10 + (trips.period.to_numpy() - 1) // 13
        ones = np.ones_like(Y)
        clean_flags = (~flag).astype(float)
        clean_strict = (~(flag | cut)).astype(float)
        entry = {"products": int(len(products)), "trips": int(len(trips)), "manufacturers": int(len(set(manufacturer))),
                 "promotion_feed": category in covered,
                 "share_purchase_lines_promoted_flags": float((Y * flag).sum() / Y.sum()),
                 "share_purchase_lines_promoted_strict": float((Y * (flag | cut)).sum() / Y.sum())}
        variants = {"all_trips": (household, ones), "household_quarter": (quarter, ones)}
        if category in covered:
            variants.update({"unpromoted_flags": (household, clean_flags),
                             "unpromoted_strict": (household, clean_strict),
                             "household_quarter_unpromoted_strict": (quarter, clean_strict)})
            # promoted_either: pairs where at least one product is promoted = all minus both clean
        for name, (cell, keep) in variants.items():
            entry[name] = household_bootstrap(Y, cell, household, keep, same, diff, rng)
        if category in covered:
            a, c = entry["all_trips"], entry["unpromoted_strict"]
            t = np.array([a["same_observed"] - c["same_observed"], a["same_expected"] - c["same_expected"],
                          a["different_observed"] - c["different_observed"], a["different_expected"] - c["different_expected"]])
            entry["promoted_either"] = {"contrast": contrast(t), "same_observed": float(t[0]),
                                        "different_observed": float(t[2])}
        report[category] = entry
        print(category, json.dumps({k: (round(v["contrast"], 3), [round(x, 2) for x in v.get("interval_95", [])])
                                    for k, v in entry.items() if isinstance(v, dict)}), flush=True)
    return report


def run_synthetic(world, rng):
    d = np.load(OUT / world / "world.npz", allow_pickle=True)
    Yall, hh, week = d["Y"], d["hh"], d["week"]
    out = {}
    for c in (0, 1, 2, 4):
        items = np.arange(c * 10, c * 10 + 10)
        present = Yall[:, items].sum(1) > 0
        Y = Yall[present][:, items]
        household = hh[present]
        groups = np.array([0] * 5 + [1] * 5)
        same = (groups[:, None] == groups[None]) & ~np.eye(10, dtype=bool)
        diff = groups[:, None] != groups[None]
        quarter = household * 10 + week[present] // 8
        out[f"cat{c}"] = {name: household_bootstrap(Y, cell, household, np.ones_like(Y), same, diff, rng)["contrast"]
                          for name, cell in (("all_trips", household), ("household_quarter", quarter))}
    return out


def main():
    rng = np.random.default_rng(31)
    report = {"price_cut_threshold": PRICE_CUT, "min_lines": MIN_LINES, "bootstrap": BOOT, "splits": "all"}
    for world in ("A", "B"):
        if (OUT / world / "world.npz").exists():
            report[f"synthetic_{world}"] = run_synthetic(world, rng)
            print(world, json.dumps(report[f"synthetic_{world}"]), flush=True)
    report["erim"] = run_erim(rng)
    (OUT / "promotion_check.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
