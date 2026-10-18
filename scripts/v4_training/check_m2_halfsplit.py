#!/usr/bin/env python3
"""Check hs_m2 per-trip IS half-split diagnostics and re-evaluate |half_diff| > 0.5 trips with 4x draws."""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("ENERGY_MODEL_DATA_ROOT", str(ROOT / "data/stress_world/model_input_category"))
sys.path.insert(0, str(ROOT / "scripts/version4"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import stress  # noqa: E402


def main():
    t0 = time.time()
    stress.USE_CPP_SAMPLER = True
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    initialization = A / "artifacts/initialization.pt"
    additive = A / "out/v3_pipeline_additive_best.pt"
    reference = A / "reports/likelihood_test_paired_per_trip.npz"
    ckpt = ROOT / "runs/stress/hs_m2.pt"
    npz_path = ROOT / "runs/stress/hs_m2.per_trip.npz"
    eval_path = ROOT / "runs/stress/hs_m2.eval.json"

    data, model, meta, batcher = stress.setup(initialization, additive, threads=4)
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    stress.configure_model_from_blob(model, blob)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    r = int(blob["rank"])
    J = int(data["n_item"])

    ref = np.load(reference)
    trips = ref["trips"].astype(np.int64)
    saved_npz = np.load(npz_path)
    saved_eval = json.loads(eval_path.read_text())

    # 1. Exact reproduction of the 1x (1024-draw) evaluation stream (seed = 41001 + 5 = 41006)
    gen = torch.Generator().manual_seed(41001 + 5)
    eval_batch = 256
    burn, components, thin, draws = 200, 128, 4, 1024

    ll_1x, half_diff_1x, ess_1x = [], [], []
    for s in range(0, len(trips), eval_batch):
        bt = stress.Batch(batcher, trips[s:s + eval_batch], J)
        bt.bind(model)
        est, dhalf, ess, _ = stress.is_log_z(model, bt, r, gen, burn, components, thin, draws)
        e = stress.obs_energy(model, bt, r)
        ll_1x.append((e - est).numpy())
        half_diff_1x.append(dhalf.numpy())
        ess_1x.append(ess.numpy())

    ll_1x = np.concatenate(ll_1x)
    half_diff_1x = np.concatenate(half_diff_1x)
    ess_1x = np.concatenate(ess_1x)

    max_ll_diff = float(np.max(np.abs(ll_1x - saved_npz["ll"])))
    abs_hd = np.abs(half_diff_1x)
    print(f"[repro] max |ll_1x - saved_ll| = {max_ll_diff:.3e}")
    print(f"[repro] abs_mean half_diff = {abs_hd.mean():.6f} (eval.json: {saved_eval['likelihood']['is_half_split_abs_mean']:.6f})")
    print(f"[repro] abs_max  half_diff = {abs_hd.max():.6f} (eval.json: {saved_eval['likelihood']['is_half_split_abs_max']:.6f})")

    # Save enriched per_trip.npz with half_diff and ess included
    np.savez(npz_path, trips=trips, ll=ll_1x, half_diff=half_diff_1x, ess=ess_1x)

    # 2. Identify trips with |half_diff| > 0.5 nats
    hi_mask = abs_hd > 0.5
    hi_idx = np.flatnonzero(hi_mask)
    hi_trips = trips[hi_idx]
    print(f"[halfsplit > 0.5] count = {len(hi_idx)} / {len(trips)} ({100.0 * len(hi_idx) / len(trips):.2f}%)")
    print(f"  signed > +0.5: {int((half_diff_1x > 0.5).sum())}, signed < -0.5: {int((half_diff_1x < -0.5).sum())}")
    print(f"  thresholds: >0.25: {int((abs_hd > 0.25).sum())}, >0.5: {int((abs_hd > 0.5).sum())}, >1.0: {int((abs_hd > 1.0).sum())}, >2.0: {int((abs_hd > 2.0).sum())}")

    # 3. Re-evaluate those trips with 4x draws (draws = 4096)
    gen_4x = torch.Generator().manual_seed(41001 + 505)
    bt_hi = stress.Batch(batcher, hi_trips, J)
    bt_hi.bind(model)
    est_4x, dhalf_4x, ess_4x, _ = stress.is_log_z(model, bt_hi, r, gen_4x, burn, components, thin, draws * 4)
    e_hi = stress.obs_energy(model, bt_hi, r)
    ll_hi_4x = (e_hi - est_4x).numpy()
    dhalf_hi_4x = dhalf_4x.numpy()
    ess_hi_4x = ess_4x.numpy()

    # Also re-evaluate with 4x components AND 4x draws (components=512, draws=4096) as a secondary check
    gen_4x_comp = torch.Generator().manual_seed(41001 + 707)
    est_4xc, dhalf_4xc, ess_4xc, _ = stress.is_log_z(model, bt_hi, r, gen_4x_comp, burn, components * 4, thin, draws * 4)
    ll_hi_4xc = (e_hi - est_4xc).numpy()
    dhalf_hi_4xc = dhalf_4xc.numpy()
    ess_hi_4xc = ess_4xc.numpy()

    ll_patched_4x = ll_1x.copy()
    ll_patched_4x[hi_idx] = ll_hi_4x
    hd_patched_4x = half_diff_1x.copy()
    hd_patched_4x[hi_idx] = dhalf_hi_4x
    ess_patched_4x = ess_1x.copy()
    ess_patched_4x[hi_idx] = ess_hi_4x

    ll_patched_4xc = ll_1x.copy()
    ll_patched_4xc[hi_idx] = ll_hi_4xc

    mean_1x = float(ll_1x.mean())
    mean_4x = float(ll_patched_4x.mean())
    delta_mean_4x = mean_4x - mean_1x
    delta_subset_mean = float((ll_hi_4x - ll_1x[hi_idx]).mean())

    lptr = data["line_ptr"]
    per_trip_rows = []
    for k, idx_val in enumerate(hi_idx):
        t_id = int(trips[idx_val])
        items_t = data["line_item"][lptr[t_id]:lptr[t_id + 1]]
        cats_t = data["line_cat"][lptr[t_id]:lptr[t_id + 1]]
        row = {
            "test_pos": int(idx_val),
            "trip_id": t_id,
            "basket_size": int(len(items_t)),
            "n_cats": int(len(np.unique(cats_t))),
            "ll_1x": float(ll_1x[idx_val]),
            "half_diff_1x": float(half_diff_1x[idx_val]),
            "abs_half_diff_1x": float(abs_hd[idx_val]),
            "ess_1x": float(ess_1x[idx_val]),
            "ll_4x": float(ll_hi_4x[k]),
            "half_diff_4x": float(dhalf_hi_4x[k]),
            "abs_half_diff_4x": float(abs(dhalf_hi_4x[k])),
            "ess_4x": float(ess_hi_4x[k]),
            "delta_ll_4x_minus_1x": float(ll_hi_4x[k] - ll_1x[idx_val]),
            "ll_4x_comp512": float(ll_hi_4xc[k]),
            "abs_half_diff_4x_comp512": float(abs(dhalf_hi_4xc[k])),
            "ess_4x_comp512": float(ess_hi_4xc[k]),
        }
        per_trip_rows.append(row)

    out = {
        "n_test_trips": int(len(trips)),
        "draws_1x": int(draws),
        "draws_4x": int(draws * 4),
        "repro_max_abs_diff": max_ll_diff,
        "count_abs_half_split_gt_0.5": int(len(hi_idx)),
        "fraction_abs_half_split_gt_0.5": float(len(hi_idx) / len(trips)),
        "count_signed_gt_pos_0.5": int((half_diff_1x > 0.5).sum()),
        "count_signed_lt_neg_0.5": int((half_diff_1x < -0.5).sum()),
        "threshold_counts_1x": {
            "gt_0.25": int((abs_hd > 0.25).sum()),
            "gt_0.50": int((abs_hd > 0.50).sum()),
            "gt_1.00": int((abs_hd > 1.00).sum()),
            "gt_2.00": int((abs_hd > 2.00).sum()),
        },
        "mean_test_ll_1x": mean_1x,
        "mean_test_ll_after_4x_on_gt_0.5": mean_4x,
        "mean_test_ll_change_4x_minus_1x": delta_mean_4x,
        "subset_mean_ll_1x": float(ll_1x[hi_idx].mean()),
        "subset_mean_ll_4x": float(ll_hi_4x.mean()),
        "subset_mean_ll_change_4x_minus_1x": delta_subset_mean,
        "subset_abs_half_split_mean_1x": float(abs_hd[hi_idx].mean()),
        "subset_abs_half_split_mean_4x": float(np.abs(dhalf_hi_4x).mean()),
        "subset_abs_half_split_max_1x": float(abs_hd[hi_idx].max()),
        "subset_abs_half_split_max_4x": float(np.abs(dhalf_hi_4x).max()),
        "overall_abs_half_split_mean_after_4x": float(np.abs(hd_patched_4x).mean()),
        "overall_abs_half_split_max_after_4x": float(np.abs(hd_patched_4x).max()),
        "overall_ess_mean_after_4x": float(ess_patched_4x.mean()),
        "count_still_gt_0.5_after_4x": int((np.abs(dhalf_hi_4x) > 0.5).sum()),
        "mean_test_ll_after_4x_comp512_on_gt_0.5": float(ll_patched_4xc.mean()),
        "mean_test_ll_change_4x_comp512_minus_1x": float(ll_patched_4xc.mean() - mean_1x),
        "elapsed_seconds": round(time.time() - t0, 2),
        "trips_gt_0.5": per_trip_rows,
    }
    out_path = ROOT / "runs/stress/hs_m2_halfsplit_4x.json"
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "trips_gt_0.5"}, indent=2))


if __name__ == "__main__":
    main()
