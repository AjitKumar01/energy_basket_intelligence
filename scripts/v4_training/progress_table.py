#!/usr/bin/env python3
"""Stress-world progression table, read only from saved run files (no numbers typed by hand).

For each run: test log-likelihood (4,096 reference trips), paired gain over the previous run,
gain over certified Version-4 with its household-clustered interval, gap to the category chain,
classifier AUCs (single seed from the eval, 3-seed mean +- sd), generated basket size, price
correlations, the multi-item breadth jump, importance-sampling diagnostics and the pass rules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs/stress"
DIAG = {"hs_conv": "multicount_pull_diagnostic.json"}


def row(name, prev, chain):
    e = json.loads((RUNS / f"{name}.eval.json").read_text())
    L, C, P = e["likelihood"], e["classifier"], e["price"]
    pt = np.load(RUNS / f"{name}.per_trip.npz")
    assert (pt["trips"] == chain["test_trips_4096"]).all()
    out = {"run": name, "test_ll": L["mean_ll"],
           "vs_certified": L["vs_certified_v4"], "vs_certified_95": L["vs_certified_v4_household_95"],
           "gap_to_chain": float((chain["chain_test_4096"] - pt["ll"]).mean()),
           "auc_category_1seed": C["category_presence_auc"], "auc_full_1seed": C["full_basket_auc"],
           "size_generated_1seed": C["size_mean_generated"], "size_real": C["size_mean_real"],
           "own_price": P["own"]["correlation"], "sibling": P["same_subcategory"]["correlation"],
           "is_ess_mean": L["is_ess_fraction_mean"], "is_half_split_max": L["is_half_split_abs_max"],
           "pass_rules": e.get("pass_rules")}
    if prev is not None:
        d = pt["ll"] - np.load(RUNS / f"{prev}.per_trip.npz")["ll"]
        out["paired_gain_vs_prev"] = float(d.mean())
        out["paired_gain_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
    a = RUNS / f"{name}_3seed_auc.json"
    if a.exists():
        a = json.loads(a.read_text())
        out.update(auc_category_3seed=a["category_presence_auc_mean"], auc_category_3seed_sd=a["category_presence_auc_sd"],
                   auc_full_3seed=a["full_basket_auc_mean"], size_generated_3seed=a["size_mean_generated_mean"])
    d = RUNS / DIAG.get(name, f"{name}_multicount_diagnostic.json")
    if d.exists():
        s = json.loads(d.read_text())["summary"]
        out["breadth_jump_model"] = s["model_cross_category_pull"]["delta_other_cats_n2_minus_n1"]
        out["breadth_jump_real"] = s["real_cross_category_pull"]["delta_other_cats_n2_minus_n1"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--runs", nargs="+", default=["hs_conv", "hs_m1", "hs_m2", "hs_m3"])
    p.add_argument("--output", type=Path, default=RUNS / "progress_table.json")
    args = p.parse_args()
    chain = np.load(RUNS / "chain_eval_splits.npz")
    rows = [row(n, args.runs[i - 1] if i else None, chain) for i, n in enumerate(args.runs)]
    args.output.write_text(json.dumps({"chain_test_ll": float(chain["chain_test_4096"].mean()), "rows": rows},
                                      indent=1) + "\n")

    def f(v, k=3):
        return "—" if v is None else f"{v:.{k}f}"
    print(f"category chain test LL {chain['chain_test_4096'].mean():.4f}")
    print("| run | test LL | paired gain (se) | vs certified [95%] | gap to chain | cat AUC 1-seed / 3-seed | "
          "full AUC 3-seed | size gen 3-seed (real) | own | sibling | breadth jump (real) | rules |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        rules = "/".join("P" if v else "F" for v in r["pass_rules"].values()) if r["pass_rules"] else "—"
        gain = f"{r['paired_gain_vs_prev']:+.3f} ({r['paired_gain_se']:.3f})" if "paired_gain_vs_prev" in r else "—"
        ci = r["vs_certified_95"]
        print(f"| {r['run']} | {r['test_ll']:.4f} | {gain} | {r['vs_certified']:+.3f} [{ci[0]:.2f}, {ci[1]:.2f}] | "
              f"{r['gap_to_chain']:.3f} | {r['auc_category_1seed']:.4f} / {f(r.get('auc_category_3seed'), 4)} "
              f"± {f(r.get('auc_category_3seed_sd'), 4)} | {f(r.get('auc_full_3seed'), 4)} | "
              f"{f(r.get('size_generated_3seed'), 2)} ({r['size_real']:.2f}) | {r['own_price']:.3f} | {r['sibling']:.3f} | "
              f"{f(r.get('breadth_jump_model'))} ({f(r.get('breadth_jump_real'))}) | {rules} |")


if __name__ == "__main__":
    main()
