#!/usr/bin/env python3
"""Analyse the randomized Cottonelle coupon cells in the ERIM tissue archive.

ERIM assigned 2,975 Sioux Falls panel households to 15 test cells (tissue file 4). The
purchase file records a cell identifier and a special code for test coupons. Cells
repeat a three-arm pattern of test-coupon redemptions:

    control  cells 1, 4, 7, 10, 13   market-wide drops only
    arm A    cells 2, 5, 8, 11, 14   control drops plus extra $1.00 (1985 wk 33) and
                                     $0.70 (1986 wk 5) coupons
    arm B    cells 3, 6, 9, 12, 15   $1.00 (wk 33) and $0.75 (wk 49) coupons in place
                                     of the control $0.70 drops (wk 37, wk 49)

Cell sizes form three strata (cells 1-6, 7-12, 13-15), inside which households are
exchangeable. The analysis is intention-to-treat on assigned households, with
stratum-weighted differences in means and randomization inference that permutes
household labels within strata.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/version4"))

from external_choice_data import read_erim_purchase  # noqa: E402

RAW = ROOT / "data/erim_basket/raw/tissue"
COTTONELLE = ("0054000494000", "0054000494100", "0054000494200", "0054000496000")
ARM_OF_CELL = {cell: ("control", "A", "B")[(cell - 1) % 3] for cell in range(1, 16)}
STRATUM_OF_CELL = {cell: (0 if cell <= 6 else 1 if cell <= 12 else 2) for cell in range(1, 16)}
WINDOWS = {
    "pre": (198505, 198532),      # before the first cell-specific drop
    "test": (198533, 198625),     # cell-specific drops and their redemption tail
    "post": (198626, 198723),     # after the last test redemption (basket-model window)
}
CONTRASTS = (("A", "control"), ("B", "control"), ("A", "B"))
OUTCOMES = (
    "cottonelle_units", "cottonelle_buyer", "cottonelle_spend_pre_coupon",
    "cottonelle_manufacturer_coupon_value", "cottonelle_net_spend",
    "other_tissue_units", "tissue_units", "tissue_spend_pre_coupon",
    "test_coupon_redemptions", "shopping_weeks",
)


def read_assignment() -> pd.DataFrame:
    frame = pd.read_fwf(RAW / "tissue_f4.dat", widths=[8, 2], names=["hh", "cell"], dtype=str)
    frame["household"] = frame.hh.astype(int)
    frame["market"] = frame.hh.str[:2].astype(int)
    frame["cell"] = frame.cell.astype(int)
    if frame.household.duplicated().any() or set(frame.market) != {1}:
        raise SystemExit("tissue cell file must assign each Sioux Falls household once")
    frame["arm"] = frame.cell.map(ARM_OF_CELL)
    frame["stratum"] = frame.cell.map(STRATUM_OF_CELL)
    return frame[["household", "cell", "arm", "stratum"]]


def read_demographics(households: pd.Index) -> pd.DataFrame:
    widths = [8, 1, 1, 2, 2, 2, 2, 1, 1, 1, 2, 2]
    names = ["hh", "cable", "metered", "status", "cats", "dogs", "tvs", "residence_type",
             "residence_status", "residence_duration", "income", "members"]
    frame = pd.read_fwf(RAW / "tissue_f8.dat", widths=widths, names=names, dtype=str)
    frame = frame[frame.hh.str[:2] == "01"].copy()
    frame["household"] = frame.hh.astype(int)
    for column in ("cable", "metered", "income", "members"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.drop_duplicates("household", keep="last").set_index("household")
    return frame.reindex(households)[["cable", "metered", "income", "members"]]


def read_shopping_weeks() -> pd.DataFrame:
    frame = pd.read_fwf(RAW / "tissue_f7.dat", widths=[8, 6, 2, 12], dtype=str,
                        names=["hh", "week", "visits", "dollars"])
    frame = frame[frame.hh.str[:2] == "01"]
    return pd.DataFrame({"household": frame.hh.astype(int),
                         "week": frame.week.astype(int)})


def household_outcomes(purchases: pd.DataFrame, shopping: pd.DataFrame,
                       households: pd.Index, window: tuple[int, int]) -> pd.DataFrame:
    low, high = window
    rows = purchases[purchases.week.between(low, high)]
    cottonelle = rows[rows.upc_text.isin(COTTONELLE)]
    group = lambda frame: frame.groupby("household")  # noqa: E731
    out = pd.DataFrame(index=households)
    out["cottonelle_units"] = group(cottonelle).units.sum()
    out["cottonelle_spend_pre_coupon"] = group(cottonelle).expenditure.sum()
    out["cottonelle_manufacturer_coupon_value"] = group(cottonelle).coupon_value.sum()
    out["tissue_units"] = group(rows).units.sum()
    out["tissue_spend_pre_coupon"] = group(rows).expenditure.sum()
    out["test_coupon_redemptions"] = group(rows[rows.test_coupon]).units.size()
    weeks = shopping[shopping.week.between(low, high)]
    out["shopping_weeks"] = weeks.groupby("household").week.nunique()
    out = out.fillna(0.0)
    out["cottonelle_buyer"] = (out.cottonelle_units > 0).astype(float)
    out["cottonelle_net_spend"] = out.cottonelle_spend_pre_coupon - out.cottonelle_manufacturer_coupon_value
    out["other_tissue_units"] = out.tissue_units - out.cottonelle_units
    return out[list(OUTCOMES)]


def stratified_difference(values: np.ndarray, arm: np.ndarray, stratum: np.ndarray,
                          treated: str, control: str) -> float:
    total, weight = 0.0, 0
    for s in np.unique(stratum):
        t = (stratum == s) & (arm == treated)
        c = (stratum == s) & (arm == control)
        n = int(t.sum() + c.sum())
        total += n * (values[t].mean() - values[c].mean())
        weight += n
    return total / weight


def permuted_arms(arm: np.ndarray, stratum: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = arm.copy()
    for s in np.unique(stratum):
        index = np.flatnonzero(stratum == s)
        out[index] = arm[rng.permutation(index)]
    return out


def analyse_window(frame: pd.DataFrame, columns: list[str], n_permutations: int,
                   seed: int) -> dict:
    arm = frame.arm.to_numpy()
    stratum = frame.stratum.to_numpy()
    values = {column: frame[column].to_numpy(float) for column in columns}
    rng = np.random.default_rng(seed)
    draws = [permuted_arms(arm, stratum, rng) for _ in range(n_permutations)]
    result = {}
    for column in columns:
        y = values[column]
        entry = {"mean_by_arm": {a: float(y[arm == a].mean()) for a in ("control", "A", "B")}}
        for treated, control in CONTRASTS:
            estimate = stratified_difference(y, arm, stratum, treated, control)
            null = np.array([stratified_difference(y, d, stratum, treated, control) for d in draws])
            base = entry["mean_by_arm"][control]
            entry[f"{treated}_minus_{control}"] = {
                "difference": float(estimate),
                "relative_to_comparison_mean": float(estimate / base) if base else None,
                "randomization_p_two_sided": float((1 + np.sum(np.abs(null) >= abs(estimate) - 1e-12))
                                                   / (1 + n_permutations)),
                "randomization_95_null_band": [float(np.quantile(null, 0.025)),
                                               float(np.quantile(null, 0.975))],
            }
        result[column] = entry
    return result


def cell_level_exact(frame: pd.DataFrame, column: str, treated: str, control: str) -> dict:
    """Conservative check treating whole cells as the randomized units within triples."""
    cells = frame.groupby("cell")[column].mean()
    triples = [(1, 2, 3), (4, 5, 6), (7, 8, 9), (10, 11, 12), (13, 14, 15)]
    position = {"control": 0, "A": 1, "B": 2}
    pairs = [(cells[t[position[treated]]], cells[t[position[control]]]) for t in triples]
    sizes = [frame.cell.isin(t).sum() for t in triples]
    observed = sum(n * (a - b) for n, (a, b) in zip(sizes, pairs)) / sum(sizes)
    null = []
    for mask in range(2 ** len(pairs)):
        signs = [(-1 if mask >> i & 1 else 1) for i in range(len(pairs))]
        null.append(sum(n * s * (a - b) for n, s, (a, b) in zip(sizes, signs, pairs)) / sum(sizes))
    null = np.array(null)
    return {"difference": float(observed),
            "exact_p_two_sided": float(np.mean(np.abs(null) >= abs(observed) - 1e-12)),
            "minimum_attainable_p": float(2 / len(null)),
            "cells_positive": int(sum(a > b for a, b in pairs))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/erim_coupon_experiment")
    parser.add_argument("--permutations", type=int, default=4999)
    parser.add_argument("--seed", type=int, default=20260916)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    assignment = read_assignment()
    households = pd.Index(assignment.household, name="household")
    purchases = read_erim_purchase(RAW / "tissue_f1.dat")
    purchases = purchases[purchases.market == 1].copy()
    purchases["upc_text"] = purchases.upc.astype(str).str.zfill(13)
    purchases["coupon_value"] = purchases.manufacturer_coupon_value_raw / 100.0
    purchases["test_coupon"] = ~purchases.special_coupon_code.isin([0])

    # Assignment audit: purchase-record cells must agree with the cell file.
    recorded = purchases[purchases.cell > 0].groupby("household").cell.agg(lambda s: set(s))
    joined = assignment.set_index("household").cell.reindex(recorded.index)
    mismatched = int(sum(c not in s for c, s in zip(joined, recorded) if pd.notna(c)))
    unassigned_cell_records = int((purchases.cell > 0).sum()
                                  - purchases[purchases.cell > 0].household.isin(households).sum())

    # Which arm redeems which test-coupon code (defines the treatment contrast).
    coded = purchases[purchases.test_coupon & purchases.household.isin(households)]
    coded = coded.merge(assignment, on="household")
    code_by_arm = pd.crosstab(coded.special_coupon_code, coded.arm)
    code_by_arm = code_by_arm[code_by_arm.sum(axis=1) >= 20]
    code_meta = coded.groupby("special_coupon_code").agg(
        first_week=("week", "min"), last_week=("week", "max"),
        median_value=("coupon_value", "median"))

    shopping = read_shopping_weeks()
    demographics = read_demographics(households)
    report = {
        "design": {
            "households": int(len(assignment)),
            "households_by_arm": assignment.arm.value_counts().to_dict(),
            "households_by_cell": assignment.cell.value_counts().sort_index().to_dict(),
            "strata": {"cells 1-6": 0, "cells 7-12": 1, "cells 13-15": 2},
            "windows": WINDOWS,
            "cottonelle_upcs": COTTONELLE,
            "purchase_cell_mismatches": mismatched,
            "cell_coded_records_outside_assignment": unassigned_cell_records,
            "test_coupon_redemptions_by_code_and_arm": {
                str(code): {**{k: int(v) for k, v in row.items()},
                            "first_week": int(code_meta.loc[code, "first_week"]),
                            "last_week": int(code_meta.loc[code, "last_week"]),
                            "median_face_value": float(code_meta.loc[code, "median_value"])}
                for code, row in code_by_arm.iterrows()},
        },
        "windows": {},
    }

    base = assignment.set_index("household")
    for offset, (name, window) in enumerate(WINDOWS.items()):
        frame = base.join(household_outcomes(purchases, shopping, households, window))
        columns = list(OUTCOMES)
        if name == "pre":
            frame = frame.join(demographics)
            columns += ["cable", "metered", "income", "members"]
            frame[["income", "members"]] = frame[["income", "members"]].fillna(
                frame[["income", "members"]].median())
            frame[["cable", "metered"]] = frame[["cable", "metered"]].fillna(0)
        report["windows"][name] = analyse_window(frame, columns, args.permutations,
                                                 args.seed + offset)
        if name == "test":
            report["windows"][name]["cell_level_exact"] = {
                f"{column}:{t}_minus_{c}": cell_level_exact(frame, column, t, c)
                for column in ("cottonelle_units", "cottonelle_buyer", "cottonelle_spend_pre_coupon",
                               "tissue_units", "tissue_spend_pre_coupon")
                for t, c in CONTRASTS[:2]}
            paid = frame.groupby("arm")[["cottonelle_units", "cottonelle_spend_pre_coupon",
                                         "cottonelle_manufacturer_coupon_value"]].sum()
            report["windows"][name]["price_paid_per_cottonelle_unit"] = {
                arm: {"shelf": float(r.cottonelle_spend_pre_coupon / r.cottonelle_units),
                      "net_of_manufacturer_coupons": float(
                          (r.cottonelle_spend_pre_coupon - r.cottonelle_manufacturer_coupon_value)
                          / r.cottonelle_units)}
                for arm, r in paid.iterrows()}
        frame.to_parquet(args.output / f"households_{name}.parquet")

    path = args.output / "report.json"
    path.write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report["design"], indent=2, default=str))
    for name, window in report["windows"].items():
        print(f"\n== {name}")
        for column, entry in window.items():
            if column in ("cell_level_exact", "price_paid_per_cottonelle_unit"):
                print(column, json.dumps(entry, indent=1))
                continue
            means = " ".join(f"{a}={v:.3f}" for a, v in entry["mean_by_arm"].items())
            tests = " | ".join(
                f"{t}-{c}: {entry[f'{t}_minus_{c}']['difference']:+.3f} "
                f"p={entry[f'{t}_minus_{c}']['randomization_p_two_sided']:.3f}"
                for t, c in CONTRASTS)
            print(f"{column:40s} {means} || {tests}")
    print("\nwrote", path)


if __name__ == "__main__":
    main()
