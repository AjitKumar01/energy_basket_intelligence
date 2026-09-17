"""Do ERIM categories split into product types that substitute across brands?

Follow-up to promotion_check.py. Product types are declared from catalogue labels before
looking at co-purchase (rules in TYPE_RULES). Pairs within a category fall into four classes
by (same type?, same manufacturer?). Each class's shortfall is compared with the reference
class "different type, different manufacturer", using the leave-one-trip-out household
expectation and household bootstrap of promotion_check.py. A negative value means the class
is co-purchased less than the reference, i.e. substitutes more strongly.

Nested substitution by type predicts: same type (either manufacturer) < 0.

Run from the repository root:
  python scripts/verification/nested_rho/type_check.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from promotion_check import MIN_LINES, OUT, ROOT, pair_sums

BOOT = 300


def sugar_type(label):
    if re.search(r"SGR SUB|SWTNR|EQUAL|SWT LW", label):
        return "substitute"
    if " PW " in f" {label} ":
        return "powdered"
    if "BRN" in label:
        return "brown"
    return "granulated" if "SGR" in label else None


def marg_type(label):
    if re.search(r"SQZ|LIQ", label):
        return "squeeze"
    if re.search(r"\b\d?TB\b", label):
        return "tub"
    if re.search(r"\b\d?ST\b", label):
        return "stick"
    return None


TYPE_RULES = {
    "tuna": lambda s: "oil" if " OIL " in f" {s} " else ("water" if "WTR" in s else None),
    "pbutter": lambda s: "creamy" if " CRM " in f" {s} " else ("chunky" if " CHK " in f" {s} " else None),
    "sugar": sugar_type,
    "marg": marg_type,
    "tissue": lambda s: "1ply" if " 1P " in f" {s} " else ("2ply" if " 2P " in f" {s} " else None),
}
CLASSES = ("same_type_same_manufacturer", "same_type_different_manufacturer",
           "different_type_same_manufacturer")


def main():
    canonical = ROOT / "data/erim_basket/canonical"
    tx = pd.read_parquet(canonical / "transactions.parquet",
                         columns=["basket_id", "customer_id", "product_id", "category"])
    labels = pd.read_parquet(canonical / "products.parquet").set_index("product_id").label
    rng = np.random.default_rng(41)
    report = {}
    for category, rule in TYPE_RULES.items():
        frame = tx[tx.category == category]
        lines = frame.product_id.value_counts()
        products = np.array(sorted(p for p in lines[lines >= MIN_LINES].index if rule(labels[p])))
        types = np.array([rule(labels[p]) for p in products])
        manufacturer = np.array([p.split(":")[1][2:7] for p in products])
        frame = frame[frame.product_id.isin(products)]
        trips = frame.groupby("basket_id").customer_id.first()
        ti = pd.Series(np.arange(len(trips)), index=trips.index)
        pi = pd.Series(np.arange(len(products)), index=products)
        Y = np.zeros((len(trips), len(products)))
        Y[ti[frame.basket_id].to_numpy(), pi[frame.product_id].to_numpy()] = 1
        household = trips.to_numpy()
        off = ~np.eye(len(products), dtype=bool)
        st = types[:, None] == types[None]
        sm = manufacturer[:, None] == manufacturer[None]
        masks = {"same_type_same_manufacturer": st & sm & off, "same_type_different_manufacturer": st & ~sm,
                 "different_type_same_manufacturer": ~st & sm, "reference": ~st & ~sm}
        # per-household sums for every class: [O, E] x class
        codes, inverse = np.unique(household, return_inverse=True)
        order = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[order], np.arange(len(codes)))
        ends = np.append(bounds[1:], len(order))
        per = np.zeros((len(codes), 8))
        ones = np.ones_like(Y)
        for i, (a, b) in enumerate(zip(bounds, ends)):
            rows = order[a:b]
            s1 = pair_sums(Y[rows], household[rows], ones[rows], masks["same_type_same_manufacturer"], masks["reference"])
            s2 = pair_sums(Y[rows], household[rows], ones[rows], masks["same_type_different_manufacturer"],
                           masks["different_type_same_manufacturer"])
            per[i] = [s1[0], s1[1], s2[0], s2[1], s2[2], s2[3], s1[2], s1[3]]

        def classes(t):
            ref = np.log((t[6] + 1) / (t[7] + 1))
            return [float(np.log((t[2 * k] + 1) / (t[2 * k + 1] + 1)) - ref) for k in range(3)]
        total = classes(per.sum(0))
        draws = np.array([classes(np.bincount(rng.choice(len(codes), len(codes)), minlength=len(codes)) @ per)
                          for _ in range(BOOT)])
        pairs = {k: int(m.sum() // 2) for k, m in masks.items()}
        report[category] = {
            "products": int(len(products)), "types": dict(zip(*np.unique(types, return_counts=True))),
            "pairs_by_class": pairs,
            **{name: {"vs_reference": total[k], "interval_95": [float(np.quantile(draws[:, k], .025)),
                                                                 float(np.quantile(draws[:, k], .975))]}
               for k, name in enumerate(CLASSES)}}
        report[category]["types"] = {k: int(v) for k, v in report[category]["types"].items()}
        print(category, report[category]["types"], {n: (round(report[category][n]["vs_reference"], 2),
                                                          [round(x, 2) for x in report[category][n]["interval_95"]])
                                                      for n in CLASSES}, flush=True)
    (OUT / "type_check.json").write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
