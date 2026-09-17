#!/usr/bin/env python3
"""Pre-training checks of the substitution-evidence partition (step 5 of the design).

For a model-data bundle:
  * out-of-sample: the same household statistic recomputed on validation trips, aggregated
    over within-group pairs, same-category pairs in different groups, and cross-category
    pairs (the anchor); within-group pairs should keep a clear shortfall;
  * stability: the partition rebuilt on 80% household subsamples (5 draws), adjusted Rand
    index against the full-data partition (products in multi-product groups only, and all);
  * synthetic truth (with --truth): same-group precision and recall against the true
    substitute pairs (same category with net pair effect phi_j.phi_k - rho_c < 0), true
    complement pairs (net effect > 0.1) grouped, and adjusted Rand index against the true
    categories.

Criteria (written after a smoke build had shown the synthetic partition, see
paper/SUBSTITUTION_EVIDENCE_PARTITION.md): synthetic same-group precision >= 0.9, no true
complement pair grouped, stability ARI >= 0.8; real data: validation within-group shortfall
below the same-category-other-group shortfall.

  python scripts/verification/partition/check_evidence_partition.py --basket-input <bundle>/basket_input \
      [--settings '{"unassigned": "singleton"}'] [--truth data/synthetic_capability_world/truth.npz] --output out.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts/version4"))
from evidence_partition import (build_evidence_partition, load_trips, pair_evidence,  # noqa: E402
                                settings_from)


def aggregate(O, E, mask):
    return {"pairs": int(mask.sum() // 2), "observed": float(O[mask].sum() / 2), "expected": float(E[mask].sum() / 2),
            "shortfall": float(np.log((O[mask].sum() + 1) / (E[mask].sum() + 1)))}


def out_of_sample(basket_input: Path, group_id: np.ndarray, split: str = "validation"):
    items, Y, household, store, week, first_period = load_trips(basket_input, split)
    cols = np.arange(len(items))
    ev = pair_evidence(Y, household, store, week, first_period, cols)
    category = items.COMMODITY_DESC.astype(str).to_numpy()
    sizes = np.bincount(group_id)
    off = ~np.eye(len(items), dtype=bool)
    same_group = (group_id[:, None] == group_id[None]) & off
    same_cat = (category[:, None] == category[None]) & off
    return {"split": split,
            "within_group": aggregate(ev.observed, ev.expected, same_group),
            "same_category_other_group": aggregate(ev.observed, ev.expected, same_cat & ~same_group),
            "cross_category": aggregate(ev.observed, ev.expected, ~same_cat & off),
            "multi_product_groups": int((sizes >= 2).sum())}


def stability(basket_input: Path, settings: dict, full: np.ndarray, draws: int = 5, seed: int = 7):
    items, _, household, *_ = load_trips(basket_input, "train")
    households = np.unique(household)
    rng = np.random.default_rng(seed)
    grouped = np.bincount(full)[full] >= 2
    out = []
    for _ in range(draws):
        subset = rng.choice(households, int(0.8 * len(households)), replace=False)
        g, _ = build_evidence_partition(basket_input, settings, households=subset)
        out.append({"ari_all": float(adjusted_rand_score(full, g)),
                    "ari_grouped_products": float(adjusted_rand_score(full[grouped], g[grouped])) if grouped.sum() > 1 else None})
    return {"draws": out, "mean_ari_all": float(np.mean([d["ari_all"] for d in out])),
            "mean_ari_grouped_products": (float(np.mean([d["ari_grouped_products"] for d in out]))
                                          if out[0]["ari_grouped_products"] is not None else None)}


def truth_agreement(truth_path: Path, group_id: np.ndarray):
    t = np.load(truth_path)
    category = t["category"]
    net = t["phi"] @ t["phi"].T - np.where(category[:, None] == category[None], t["rho_c"][category][:, None], 0.0)
    off = ~np.eye(len(category), dtype=bool)
    same_group = (group_id[:, None] == group_id[None]) & off
    substitute = (category[:, None] == category[None]) & off & (net < 0)
    complement = off & (net > 0.1)
    grouped = same_group.sum()
    return {"true_substitute_pairs": int(substitute.sum() // 2), "grouped_pairs": int(grouped // 2),
            "same_group_precision": float((same_group & substitute).sum() / grouped) if grouped else None,
            "same_group_recall": float((same_group & substitute).sum() / substitute.sum()),
            "true_complement_pairs_grouped": int((same_group & complement).sum() // 2),
            "ari_vs_true_categories": float(adjusted_rand_score(category, group_id))}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--basket-input", type=Path, required=True)
    parser.add_argument("--settings", default="{}")
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    settings = settings_from(json.loads(args.settings))
    group_id, summary = build_evidence_partition(args.basket_input, settings)
    report = {"settings": settings, "summary": {k: v for k, v in summary.items() if k != "multi_product_group_mean_scores"},
              "out_of_sample": out_of_sample(args.basket_input, group_id),
              "stability": stability(args.basket_input, settings, group_id)}
    if args.truth:
        report["truth"] = truth_agreement(args.truth, group_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
