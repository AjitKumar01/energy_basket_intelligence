#!/usr/bin/env python3
"""Diagnose hs_conv model samples vs real training baskets:
1. For each category c, distribution of n_c given n_c >= 1 (k = 1, 2, 3, >=4).
2. Presence rate of other categories c' != c given n_c = 1 vs n_c >= 2 (overall and on coupled pairs),
   and how P(n_c >= 2 | n_c >= 1) varies with the number of other active categories K_{-c}."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from pathlib import Path

sys.modules.setdefault("pathlib._local", pathlib)
os.environ.setdefault("V3_AFFINITY", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/version4"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import stress  # noqa: E402
from stress import Batch, gibbs, sample_slots, setup, slot_utilities  # noqa: E402
from pipeline_support import supported_trips  # noqa: E402

torch.set_default_dtype(torch.float64)


def compute_stats(nc_mat: np.ndarray, psi_cos: np.ndarray):
    """nc_mat: [T, C] integer category counts per trip."""
    T, C = nc_mat.shape
    pres = nc_mat > 0
    k_tot = pres.sum(axis=1)                                    # [T] number of active categories
    size_tot = nc_mat.sum(axis=1)                               # [T] basket size

    # 1. Conditional distribution P(n_c = k | n_c >= 1) per category and pooled
    cat_rows = []
    pooled_pos = nc_mat[pres]
    pooled_dist = {
        "p_present_mean": float(pres.mean()),
        "mean_nc_given_pos": float(pooled_pos.mean()),
        "p_1_given_pos": float((pooled_pos == 1).mean()),
        "p_2_given_pos": float((pooled_pos == 2).mean()),
        "p_3_given_pos": float((pooled_pos == 3).mean()),
        "p_4plus_given_pos": float((pooled_pos >= 4).mean()),
    }

    # 2. Cross-category pull: given n_c = 1 vs n_c >= 2
    other_cats_n0, other_cats_n1, other_cats_n2 = [], [], []
    pair_diff_all = []
    pair_diff_pos_coupled = []
    pair_diff_top_coupled = []

    for c in range(C):
        m0 = nc_mat[:, c] == 0
        m1 = nc_mat[:, c] == 1
        m2 = nc_mat[:, c] >= 2
        m_pos = nc_mat[:, c] >= 1
        n_pos = int(m_pos.sum())
        if n_pos > 0:
            vals = nc_mat[m_pos, c]
            p1 = float((vals == 1).mean())
            p2 = float((vals == 2).mean())
            p3 = float((vals == 3).mean())
            p4 = float((vals >= 4).mean())
            m_nc = float(vals.mean())
        else:
            p1 = p2 = p3 = p4 = m_nc = 0.0

        k_other = k_tot - pres[:, c].astype(np.int64)
        k_o_0 = float(k_other[m0].mean()) if m0.sum() > 0 else np.nan
        k_o_1 = float(k_other[m1].mean()) if m1.sum() > 0 else np.nan
        k_o_2 = float(k_other[m2].mean()) if m2.sum() > 0 else np.nan
        if m0.sum() >= 20:
            other_cats_n0.append(k_o_0)
        if m1.sum() >= 20 and m2.sum() >= 10:
            other_cats_n1.append(k_o_1)
            other_cats_n2.append(k_o_2)
            # Pairwise P(n_{c'} > 0 | n_c >= 2) - P(n_{c'} > 0 | n_c == 1)
            rate_1 = pres[m1].mean(axis=0)
            rate_2 = pres[m2].mean(axis=0)
            diff_c = rate_2 - rate_1
            for cp in range(C):
                if cp == c:
                    continue
                pair_diff_all.append(float(diff_c[cp]))
                if psi_cos[c, cp] > 0.15:
                    pair_diff_pos_coupled.append(float(diff_c[cp]))
                if psi_cos[c, cp] > 0.35:
                    pair_diff_top_coupled.append(float(diff_c[cp]))

        cat_rows.append({
            "cat": c,
            "n_pos": n_pos,
            "n_n1": int(m1.sum()),
            "n_n2plus": int(m2.sum()),
            "p_present": float(m_pos.mean()),
            "mean_nc_given_pos": m_nc,
            "p_1_given_pos": p1,
            "p_2_given_pos": p2,
            "p_3_given_pos": p3,
            "p_4plus_given_pos": p4,
            "other_cats_given_n0": k_o_0,
            "other_cats_given_n1": k_o_1,
            "other_cats_given_n2plus": k_o_2,
            "delta_other_cats_n2_minus_n1": float(k_o_2 - k_o_1) if not (np.isnan(k_o_1) or np.isnan(k_o_2)) else np.nan,
        })

    # 3. Also check P(n_c >= 2 | n_c >= 1, K_{-c} in bin) across bins of K_{-c}
    k_other_mat = k_tot[:, None] - pres.astype(np.int64)
    bins = [(0, 3, "0-3 other cats"), (4, 6, "4-6 other cats"), (7, 9, "7-9 other cats"),
            (10, 12, "10-12 other cats"), (13, 48, "13+ other cats")]
    by_k_other = []
    for lo, hi, label in bins:
        mask = pres & (k_other_mat >= lo) & (k_other_mat <= hi)
        if mask.sum() > 0:
            vals = nc_mat[mask]
            by_k_other.append({
                "bin": label,
                "active_cat_instances": int(mask.sum()),
                "p_nc_ge_2_given_pos": float((vals >= 2).mean()),
                "p_nc_eq_2_given_pos": float((vals == 2).mean()),
                "p_nc_ge_3_given_pos": float((vals >= 3).mean()),
                "mean_nc_given_pos": float(vals.mean()),
            })

    return {
        "mean_basket_size": float(size_tot.mean()),
        "mean_active_categories": float(k_tot.mean()),
        "pooled_nc_given_pos": pooled_dist,
        "cross_category_pull": {
            "eligible_categories": len(other_cats_n1),
            "mean_other_cats_given_n0": float(np.mean(other_cats_n0)),
            "mean_other_cats_given_n1": float(np.mean(other_cats_n1)),
            "mean_other_cats_given_n2plus": float(np.mean(other_cats_n2)),
            "delta_other_cats_n2_minus_n1": float(np.mean(other_cats_n2) - np.mean(other_cats_n1)),
            "pairwise_pres_diff_all_mean": float(np.mean(pair_diff_all)),
            "pairwise_pres_diff_pos_coupled_mean": float(np.mean(pair_diff_pos_coupled)),
            "pairwise_pres_diff_top_coupled_mean": float(np.mean(pair_diff_top_coupled)),
        },
        "p_multicount_by_other_active_cats": by_k_other,
        "per_category": cat_rows,
    }


def main():
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initialization", type=Path, default=A / "artifacts/initialization.pt")
    p.add_argument("--additive", type=Path, default=A / "out/v3_pipeline_additive_best.pt")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs/stress/hs_conv.pt")
    p.add_argument("--burn", type=int, default=200)
    p.add_argument("--seed", type=int, default=41001)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, default=ROOT / "runs/stress/multicount_pull_diagnostic.json")
    args = p.parse_args()

    stress.USE_CPP_SAMPLER = True
    t0 = time.time()
    data, model, meta, batcher = setup(args.initialization, args.additive, args.threads)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    stress.configure_model_from_blob(model, blob)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    r = int(blob["rank"])
    J, C, nmax = int(data["n_item"]), int(data["n_cat"]), int(meta["nmax"])

    # Compute category-level cosine similarity from learned Phi
    phi = model.phi[:, :r].detach().numpy()
    cat_of = model.cat_of.long().numpy()
    psi_bar = np.zeros((C, r))
    for c in range(C):
        psi_bar[c] = phi[cat_of == c].mean(axis=0)
    norms = np.linalg.norm(psi_bar, axis=1, keepdims=True).clip(min=1e-8)
    psi_cos = (psi_bar / norms) @ (psi_bar / norms).T

    all_train_trips = supported_trips(data, 0, nmax)
    v_perm = np.random.default_rng(args.seed + 99).permutation(all_train_trips)
    train_trips = np.sort(v_perm[2048:])
    batches = [Batch(batcher, train_trips[s:s + 256], J) for s in range(0, len(train_trips), 256)]

    gen = torch.Generator().manual_seed(args.seed + 333)
    nc_real_list, nc_mod_list = [], []
    with torch.no_grad():
        for bt in batches:
            bt.bind(model)
            X_mod = sample_slots(model, bt.ix, slot_utilities(model, bt.ix, None, r, beta=0.0), gen, beta=0.0)
            X_mod = gibbs(model, bt.ix, X_mod, r, gen, sweeps=args.burn)
            for mask, dest in ((bt.obs, nc_real_list), (X_mod, nc_mod_list)):
                nc_row = torch.zeros(bt.ix.n_rows).index_add_(0, bt.ix.row_of, mask.double())
                nc_bc = torch.zeros((bt.ix.B, C), dtype=torch.int64)
                nc_bc[bt.ix.row_trip, bt.ix.row_cat] = nc_row.long()
                dest.append(nc_bc.numpy())

    nc_real = np.concatenate(nc_real_list, axis=0)
    nc_mod = np.concatenate(nc_mod_list, axis=0)

    real_stats = compute_stats(nc_real, psi_cos)
    mod_stats = compute_stats(nc_mod, psi_cos)

    # Per-category comparison summary
    r_cats = real_stats["per_category"]
    m_cats = mod_stats["per_category"]
    p1_diff = np.array([m["p_1_given_pos"] - r["p_1_given_pos"] for r, m in zip(r_cats, m_cats)])
    p2_diff = np.array([m["p_2_given_pos"] - r["p_2_given_pos"] for r, m in zip(r_cats, m_cats)])
    p3_diff = np.array([m["p_3_given_pos"] - r["p_3_given_pos"] for r, m in zip(r_cats, m_cats)])
    p4_diff = np.array([m["p_4plus_given_pos"] - r["p_4plus_given_pos"] for r, m in zip(r_cats, m_cats)])

    delta_real = np.array([r["delta_other_cats_n2_minus_n1"] for r in r_cats if not np.isnan(r["delta_other_cats_n2_minus_n1"])])
    delta_mod = np.array([m["delta_other_cats_n2_minus_n1"] for r, m in zip(r_cats, m_cats) if not np.isnan(r["delta_other_cats_n2_minus_n1"]) and not np.isnan(m["delta_other_cats_n2_minus_n1"])])

    summary = {
        "train_trips": int(len(train_trips)),
        "seconds": round(time.time() - t0, 1),
        "real_pooled": real_stats["pooled_nc_given_pos"],
        "model_pooled": mod_stats["pooled_nc_given_pos"],
        "real_mean_size": real_stats["mean_basket_size"],
        "model_mean_size": mod_stats["mean_basket_size"],
        "real_mean_active_cats": real_stats["mean_active_categories"],
        "model_mean_active_cats": mod_stats["mean_active_categories"],
        "per_cat_count_dist_diff": {
            "p1_mean_diff": round(float(p1_diff.mean()), 4),
            "p1_mean_abs_diff": round(float(np.abs(p1_diff).mean()), 4),
            "p1_max_abs_diff": round(float(np.abs(p1_diff).max()), 4),
            "p2_mean_diff": round(float(p2_diff.mean()), 4),
            "p2_mean_abs_diff": round(float(np.abs(p2_diff).mean()), 4),
            "p2_max_abs_diff": round(float(np.abs(p2_diff).max()), 4),
            "p3_mean_diff": round(float(p3_diff.mean()), 4),
            "p3_mean_abs_diff": round(float(np.abs(p3_diff).mean()), 4),
            "p3_max_abs_diff": round(float(np.abs(p3_diff).max()), 4),
            "p4plus_mean_diff": round(float(p4_diff.mean()), 4),
            "p4plus_mean_abs_diff": round(float(np.abs(p4_diff).mean()), 4),
            "p4plus_max_abs_diff": round(float(np.abs(p4_diff).max()), 4),
        },
        "real_cross_category_pull": real_stats["cross_category_pull"],
        "model_cross_category_pull": mod_stats["cross_category_pull"],
        "real_by_other_active_cats": real_stats["p_multicount_by_other_active_cats"],
        "model_by_other_active_cats": mod_stats["p_multicount_by_other_active_cats"],
    }

    payload = {
        "summary": summary,
        "real": real_stats,
        "model": mod_stats,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
