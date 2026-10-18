#!/usr/bin/env python3
"""Compare per-trip test-likelihood differences between a Version-4 run and the Category Chain
(and Certified Version-4), grouped by basket features (size, distinct categories, multi-item
categories, same-subcategory pairs, and generator recipe triplets)."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import types
from pathlib import Path

sys.modules.setdefault("pathlib._local", pathlib)
os.environ.setdefault("V3_AFFINITY", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/version4"))
sys.path.insert(0, str(ROOT / "scripts/synthetic"))
sys.path.insert(0, str(ROOT.parent / "basket_chain"))

import numpy as np  # noqa: E402
from data import build  # noqa: E402
import generate_stress_world as gsw  # noqa: E402
from basket_chain.data import load_bundle  # noqa: E402
from scripts.evaluate_stress import load_model as load_chain_model, loglik as chain_loglik  # noqa: E402


def main():
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--per-trip", type=Path, default=ROOT / "runs/stress/hs_conv.per_trip.npz")
    p.add_argument("--ref-test", type=Path, default=A / "reports/likelihood_test_paired_per_trip.npz")
    p.add_argument("--chain-run", type=Path, default=ROOT.parent / "basket_chain/runs/stress/fit_h64_p10")
    p.add_argument("--bundle", type=Path, default=ROOT / "data/stress_world/model_input_category")
    p.add_argument("--world", type=Path, default=ROOT / "data/stress_world")
    p.add_argument("--output", type=Path, default=None)
    args = p.parse_args()

    data = build()
    ref_test = np.load(args.ref_test)
    model_npz = np.load(args.per_trip)
    trips = ref_test["trips"].astype(np.int64)
    if not np.array_equal(trips, model_npz["trips"]):
        order = {int(t): i for i, t in enumerate(model_npz["trips"])}
        ll_model = np.array([model_npz["ll"][order[int(t)]] for t in trips])
    else:
        ll_model = model_npz["ll"]

    chain_cache = ROOT / "runs/stress/chain_eval_splits.npz"
    if chain_cache.exists():
        c_npz = np.load(chain_cache)
        if np.array_equal(trips, c_npz["test_trips_4096"]):
            ll_chain = c_npz["chain_test_4096"]
        else:
            bundle = load_bundle(args.bundle)
            chain_model, _ = load_chain_model(args.chain_run, bundle)
            ll_chain = chain_loglik(chain_model, bundle, trips)
    else:
        bundle = load_bundle(args.bundle)
        chain_model, _ = load_chain_model(args.chain_run, bundle)
        ll_chain = chain_loglik(chain_model, bundle, trips)

    ll_cert = ref_test["target_child"]
    ll_parent = ref_test["exact_parent"]
    gap_model = ll_chain - ll_model
    gap_cert = ll_chain - ll_cert

    # Reconstruct exact generator recipes from seed=20260924
    g_args = types.SimpleNamespace(
        seed=20260924, households=3000, stores=15, departments=6,
        categories_per_department=8, mission_variants=1, weeks=52,
        train_weeks=32, validation_weeks=10, scenario_products=16
    )
    world = gsw.World(g_args, np.random.default_rng(g_args.seed))
    recipes = []
    for m_idx, t in enumerate(world.templates):
        for r in t["recipes"]:
            recipes.append((gsw.MISSIONS[m_idx], set(int(c) for c in r)))

    o = np.load(args.world / "observed.npz")
    orig_cat = o["category"]
    orig_sub = o["subcategory"]
    lptr = data["line_ptr"]
    line_item = data["line_item"]
    line_cat = data["line_cat"]

    n_trips = len(trips)
    b_size = np.zeros(n_trips, dtype=np.int64)
    n_cats = np.zeros(n_trips, dtype=np.int64)
    n_multi_cats = np.zeros(n_trips, dtype=np.int64)
    max_cat_cnt = np.zeros(n_trips, dtype=np.int64)
    has_same_sub_pair = np.zeros(n_trips, dtype=bool)
    n_recipes_hit = np.zeros(n_trips, dtype=np.int64)

    for idx, t in enumerate(trips):
        items = line_item[lptr[t]:lptr[t + 1]]
        cats = line_cat[lptr[t]:lptr[t + 1]]
        b_size[idx] = len(items)
        u_cats, counts = np.unique(cats, return_counts=True)
        n_cats[idx] = len(u_cats)
        n_multi_cats[idx] = int((counts >= 2).sum())
        max_cat_cnt[idx] = int(counts.max())
        subs = orig_sub[items]
        for c_val in u_cats[counts >= 2]:
            s_in_c = subs[cats == c_val]
            if len(np.unique(s_in_c)) < len(s_in_c):
                has_same_sub_pair[idx] = True
        cat_set = set(orig_cat[items].tolist())
        n_recipes_hit[idx] = sum(1 for _, rset in recipes if rset.issubset(cat_set))

    total_gap = float(gap_model.sum())
    groups_out = {}

    def report_group(title, key, mask_list):
        print(f"\n=== {title} ===")
        print("Group                          |  Trips ( Share) | Mean |S| |  Chain LL |   Model LL | Gap/Trip | Gap/Item | Share of Total Gap | CertV4 Gap")
        rows = []
        for label, m in mask_list:
            cnt = int(m.sum())
            if cnt == 0:
                continue
            sh = cnt / n_trips
            g_m = float(gap_model[m].mean())
            g_contrib = float(gap_model[m].sum() / n_trips)
            g_share = float(gap_model[m].sum() / total_gap) if abs(total_gap) > 1e-12 else 0.0
            s_m = float(b_size[m].mean())
            c_m = float(ll_chain[m].mean())
            m_m = float(ll_model[m].mean())
            cv_m = float(gap_cert[m].mean())
            print(f"{label:30s} | {cnt:6d} ({100*sh:5.1f}%) | {s_m:8.2f} | {c_m:9.4f} | {m_m:10.4f} | {g_m:+8.4f} | {g_m/s_m:+8.4f} | {g_contrib:+7.4f} ({100*g_share:5.1f}%) | {cv_m:+10.4f}")
            rows.append({
                "label": label, "trips": cnt, "share_trips": round(sh, 4),
                "mean_size": round(s_m, 3), "chain_mean_ll": round(c_m, 4),
                "model_mean_ll": round(m_m, 4), "gap_per_trip": round(g_m, 4),
                "gap_per_item": round(g_m / s_m, 4), "gap_contribution": round(g_contrib, 4),
                "share_of_total_gap": round(g_share, 4), "cert_v4_gap_per_trip": round(cv_m, 4),
            })
        groups_out[key] = rows

    report_group("1. By Presence / Count of Multi-Item Categories (n_c >= 2)", "multi_item", [
        ("All test trips", np.ones(n_trips, dtype=bool)),
        ("0 multi-item cats (all n_c<=1)", n_multi_cats == 0),
        ("1+ multi-item cats (any n_c>=2)", n_multi_cats >= 1),
        ("  1 multi-item cat", n_multi_cats == 1),
        ("  2 multi-item cats", n_multi_cats == 2),
        ("  3+ multi-item cats", n_multi_cats >= 3),
        ("  max(n_c) == 2", max_cat_cnt == 2),
        ("  max(n_c) >= 3", max_cat_cnt >= 3),
        ("  n_c>=2 (diff subcats only)", (n_multi_cats >= 1) & (~has_same_sub_pair)),
        ("  n_c>=2 (has same-subcat pair)", (n_multi_cats >= 1) & has_same_sub_pair),
    ])

    report_group("2. By Basket Size |S|", "basket_size", [
        ("1 - 4 items", (b_size >= 1) & (b_size <= 4)),
        ("5 - 7 items", (b_size >= 5) & (b_size <= 7)),
        ("8 - 10 items", (b_size >= 8) & (b_size <= 10)),
        ("11 - 14 items", (b_size >= 11) & (b_size <= 14)),
        ("15+ items", b_size >= 15),
    ])

    report_group("3. By Number of Distinct Categories K_cat", "distinct_categories", [
        ("1 - 3 categories", (n_cats >= 1) & (n_cats <= 3)),
        ("4 - 6 categories", (n_cats >= 4) & (n_cats <= 6)),
        ("7 - 9 categories", (n_cats >= 7) & (n_cats <= 9)),
        ("10 - 12 categories", (n_cats >= 10) & (n_cats <= 12)),
        ("13+ categories", n_cats >= 13),
    ])

    report_group("4. By Recipe Trips (3-Category Generator Recipe Triplet Present)", "recipes", [
        ("No recipe (0 recipes)", n_recipes_hit == 0),
        ("Recipe trip (1+ recipes)", n_recipes_hit >= 1),
        ("  1 recipe triplet", n_recipes_hit == 1),
        ("  2+ recipe triplets", n_recipes_hit >= 2),
        ("No recipe & 0 multi-item cats", (n_recipes_hit == 0) & (n_multi_cats == 0)),
        ("Recipe & 0 multi-item cats", (n_recipes_hit >= 1) & (n_multi_cats == 0)),
        ("No recipe & 1+ multi-item cats", (n_recipes_hit == 0) & (n_multi_cats >= 1)),
        ("Recipe & 1+ multi-item cats", (n_recipes_hit >= 1) & (n_multi_cats >= 1)),
    ])

    report_group("5. Controlling for Basket Size: 0 vs 1+ Multi-Item Cats", "size_x_multi", [
        ("|S| 5-7, 0 multi-item cats", (b_size >= 5) & (b_size <= 7) & (n_multi_cats == 0)),
        ("|S| 5-7, 1+ multi-item cats", (b_size >= 5) & (b_size <= 7) & (n_multi_cats >= 1)),
        ("|S| 8-10, 0 multi-item cats", (b_size >= 8) & (b_size <= 10) & (n_multi_cats == 0)),
        ("|S| 8-10, 1+ multi-item cats", (b_size >= 8) & (b_size <= 10) & (n_multi_cats >= 1)),
        ("|S| 11-14, 0 multi-item cats", (b_size >= 11) & (b_size <= 14) & (n_multi_cats == 0)),
        ("|S| 11-14, 1+ multi-item cats", (b_size >= 11) & (b_size <= 14) & (n_multi_cats >= 1)),
    ])

    extra_items = b_size - n_cats
    X = np.column_stack([np.ones(n_trips), n_cats, extra_items, n_recipes_hit])
    beta, *_ = np.linalg.lstsq(X, gap_model, rcond=None)
    ols = []
    print("\nOLS decomposition of per-trip gap (Chain - Model):")
    for name, b_val, mean_x in zip(["Intercept", "n_cats (distinct categories)", "extra_items (|S| - n_cats)", "n_recipes_hit"], beta, X.mean(0)):
        print(f"  {name:30s}: coef={b_val:+.4f} nats/unit, mean_x={mean_x:.3f}, mean_contrib={b_val*mean_x:+.4f} nats")
        ols.append({"term": name, "coef": round(float(b_val), 4), "mean_x": round(float(mean_x), 4),
                    "mean_contrib": round(float(b_val * mean_x), 4)})

    out_path = args.output or args.per_trip.with_name(args.per_trip.stem.replace(".per_trip", "") + "_grouped_gap.json")
    payload = {
        "per_trip": str(args.per_trip),
        "trips": int(n_trips),
        "chain_mean_ll": round(float(ll_chain.mean()), 4),
        "model_mean_ll": round(float(ll_model.mean()), 4),
        "certified_v4_mean_ll": round(float(ll_cert.mean()), 4),
        "phi0_parent_mean_ll": round(float(ll_parent.mean()), 4),
        "mean_gap_vs_chain": round(float(gap_model.mean()), 4),
        "groups": groups_out,
        "ols_decomposition": ols,
    }
    out_path.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
