#!/usr/bin/env python3
"""Diagnose why Version 4 loses the display effect on Dunnhumby (Phase 2, item 5).

Tests four candidate causes with exact numbers on existing checkpoints and the
Dunnhumby and stress-world bundles (no model fits; <= 2 threads; <= 2 GB RAM):
  (a) Selection: does display go to products, stores or weeks with otherwise low
      demand, and do free per-product weights soak that up?
  (b) Collinearity: is display collinear with discount, log-price deviation or
      mailer in the same store-week, so the weights split one effect?
  (c) Version-4 absorption & structural attenuation: do the Version-4 latent term
      and purchase history explain the display-week lift, or why does w_dsp drop
      in Stage 4 (PL -> SML -> history refit) and how much of the +36% -> +14%
      drop in halo_real.py is parameter drift vs sparse-commodity sampling variance?
  (d) Overfitting of free per-product weights: how do display events and displayed
      purchases per product relate to fitted w_dsp when there is no pooling/prior?
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from pathlib import Path

sys.modules.setdefault("pathlib._local", pathlib)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.nn.functional import softplus  # noqa: E402

torch.set_default_dtype(torch.float64)

from checks.halo_real import regress, within  # noqa: E402
from v4.bundle import data_root, load_bundle, supported_trips  # noqa: E402
from v4.checkpoints import _load, batcher_for, freeze, load_model  # noqa: E402
from v4.exact import incidence as exact_incidence  # noqa: E402
from v4.features import WEEK_STRIDE  # noqa: E402
from v4.sampling import draw_from_model, gibbs  # noqa: E402


def summarize_vec(x: np.ndarray, weights: dict[str, np.ndarray] | None = None) -> dict:
    q = np.quantile(x, [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95])
    out = {
        "mean": float(x.mean()),
        "sd": float(x.std(ddof=1)),
        "min": float(x.min()),
        "p05": float(q[0]),
        "p10": float(q[1]),
        "p25": float(q[2]),
        "median": float(q[3]),
        "p75": float(q[4]),
        "p90": float(q[5]),
        "p95": float(q[6]),
        "max": float(x.max()),
        "share_negative": float((x < 0).mean()),
        "share_strictly_positive": float((x > 0).mean()),
        "share_below_minus_1": float((x < -1.0).mean()),
    }
    if weights:
        for k, w in weights.items():
            w_sum = float(w.sum())
            if w_sum > 0:
                out[f"weighted_mean_{k}"] = float((x * w).sum() / w_sum)
                out[f"weighted_share_neg_{k}"] = float(((x < 0) * w).sum() / w_sum)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data-root", type=Path,
                   default=Path("/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence"))
    p.add_argument("--base-ckpt", type=Path,
                   default=Path("/Users/ajit/Projects/nf_dunnhumby/ebi_v4t/runs/dunnhumby_refit/out/v3_pipeline_additive_best.pt"))
    p.add_argument("--v4-dir", type=Path,
                   default=Path("/Users/ajit/Projects/nf_dunnhumby/ebi_v4_pipeline/runs/dunnhumby"))
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    t0 = time.time()
    root = data_root(args.data_root)
    data = load_bundle(root)
    J = int(data["n_item"])
    S = int(data["n_store"])
    C = int(data["n_cat"])

    items = pd.read_parquet(root / "basket_input/items.parquet").sort_values("item_id").reset_index(drop=True)
    n_train_lines = items["n_train_lines"].to_numpy(np.float64)
    commodity = items["COMMODITY_DESC"].to_numpy()

    ckpts = {
        "base": args.base_ckpt,
        "v4_rec_pl": args.v4_dir / "v4_rec_pl.pt",
        "v4_rec": args.v4_dir / "v4_rec.pt",
        "v4_m4c_pl": args.v4_dir / "v4_m4c_pl.pt",
        "v4_m4c": args.v4_dir / "v4_m4c.pt",
        "v4_hist_pl": args.v4_dir / "v4_hist_pl.pt",
        "v4_hist": args.v4_dir / "v4_hist.pt",
        "v4_hist_inclusive_pl": args.v4_dir / "v4_hist_inclusive_pl.pt",
        "v4_hist_inclusive": args.v4_dir / "v4_hist_inclusive.pt",
    }
    states = {name: _load(path)["model"] for name, path in ckpts.items() if path.exists()}
    print(f"[diagnose] loaded {len(states)} checkpoint state dicts in {time.time() - t0:.1f}s", flush=True)

    # ---- 1. Exact training exposure and purchase counts per product on Dunnhumby ----
    split = data["trip_split"]
    trip_store = data["trip_store"].astype(np.int64)
    trip_week = data["trip_week"].astype(np.int64)
    trip_day = data["trip_day"].astype(np.int64)
    lp, li = data["line_ptr"], data["line_item"].astype(np.int64)
    line_trip = np.repeat(np.arange(len(lp) - 1, dtype=np.int64), np.diff(lp))

    ptr, store_items = data["store_cat_ptr"], data["store_items"]
    offered = np.zeros((S, J), dtype=bool)
    for s in range(S):
        lo, hi = int(ptr[s * C]), int(ptr[(s + 1) * C])
        offered[s, store_items[lo:hi]] = True

    promo = np.load(root / "basket_input/promo.npz")
    p_order = np.argsort(promo["keys"].astype(np.int64))
    p_keys = promo["keys"].astype(np.int64)[p_order]
    p_disp = promo["disp"][p_order].astype(np.float64)
    p_mail = promo["mail"][p_order].astype(np.float64)

    dev = np.load(root / "basket_input/log_price_dev.npy").astype(np.float64)
    base_blob = _load(args.base_ckpt)
    dsc_path = (base_blob.get("discount_split") or {}).get("discount_panel")
    dsc = np.load(dsc_path).astype(np.float64) if dsc_path else np.zeros_like(dev)

    train_trips = np.flatnonzero(split == 0)
    max_week = int(trip_week.max()) + 1
    trips_sw = np.zeros((S, max_week), dtype=np.int64)
    np.add.at(trips_sw, (trip_store[train_trips], trip_week[train_trips]), 1)

    trips_per_store = trips_sw.sum(axis=1)
    offered_train_trips = offered.T @ trips_per_store

    pj = (p_keys // WEEK_STRIDE) // S
    ps = (p_keys // WEEK_STRIDE) % S
    pw = p_keys % WEEK_STRIDE
    in_bounds = (pw < max_week) & (ps < S) & (pj < J)
    valid_p = in_bounds & offered[ps.clip(0, S - 1), pj.clip(0, J - 1)]
    sw_trips_at_promo = np.where(valid_p, trips_sw[ps.clip(0, S - 1), pw.clip(0, max_week - 1)], 0)

    disp_train_trips = np.bincount(pj, weights=sw_trips_at_promo * p_disp, minlength=J)
    mail_train_trips = np.bincount(pj, weights=sw_trips_at_promo * p_mail, minlength=J)

    train_line_mask = split[line_trip] == 0
    tl_trip = line_trip[train_line_mask]
    tl_item = li[train_line_mask]
    tl_store = trip_store[tl_trip]
    tl_week = trip_week[tl_trip]
    tl_day = trip_day[tl_trip]
    tl_key = (tl_item * S + tl_store) * WEEK_STRIDE + tl_week
    pos = np.searchsorted(p_keys, tl_key).clip(0, len(p_keys) - 1)
    hit = p_keys[pos] == tl_key
    tl_disp = np.where(hit, p_disp[pos], 0.0)
    tl_mail = np.where(hit, p_mail[pos], 0.0)
    tl_dsc = dsc[tl_item, tl_day]

    total_train_purch = np.bincount(tl_item, minlength=J).astype(np.float64)
    disp_train_purch = np.bincount(tl_item, weights=tl_disp, minlength=J)
    mail_train_purch = np.bincount(tl_item, weights=tl_mail, minlength=J)
    nondisp_train_purch = total_train_purch - disp_train_purch
    nondisp_train_trips = np.maximum(offered_train_trips - disp_train_trips, 0.0)

    all_train_val_mask = split[line_trip] <= 1
    top60_comms = pd.Series(commodity[li[all_train_val_mask]]).value_counts().index[:60].to_list()
    in_top60_comm = np.isin(commodity, top60_comms)

    weights_dict = {
        "train_purchases": total_train_purch,
        "disp_train_trips": disp_train_trips,
        "disp_train_purchases": disp_train_purch,
        "top60_comm_disp_trips": disp_train_trips * in_top60_comm,
    }

    param_summary = {}
    for name, st in states.items():
        w_dsp = st["w_dsp"].numpy()
        w_mlr = st["w_mlr"].numpy()
        row = {
            "w_dsp": summarize_vec(w_dsp, weights_dict),
            "w_mlr": summarize_vec(w_mlr, {
                "train_purchases": total_train_purch,
                "mail_train_trips": mail_train_trips,
                "mail_train_purchases": mail_train_purch,
            }),
        }
        if "disc_common" in st and st["disc_common"] is not None:
            row["disc_common"] = float(st["disc_common"].item())
        if "disc_kappa" in st and st["disc_kappa"] is not None:
            row["disc_kappa_softplus"] = float(softplus(st["disc_kappa"]).item())
        if "price_kappa" in st and st["price_kappa"] is not None:
            row["price_kappa_softplus"] = float(softplus(st["price_kappa"]).item())
        if "psi_item" in st and st["psi_item"] is not None:
            row["psi_item_mean"] = st["psi_item"].mean(0).tolist()
        if "hist_size" in st and st["hist_size"] is not None:
            row["hist_size"] = float(st["hist_size"].item())
        param_summary[name] = row

    # ---- 2. Candidate (d): Overfitting of free per-product weights by display purchase count ----
    buckets = [
        ("never_displayed (0 disp trips)", disp_train_trips == 0),
        ("displayed, 0 purchases", (disp_train_trips > 0) & (disp_train_purch == 0)),
        ("1 disp purchase", disp_train_purch == 1),
        ("2-4 disp purchases", (disp_train_purch >= 2) & (disp_train_purch <= 4)),
        ("5-9 disp purchases", (disp_train_purch >= 5) & (disp_train_purch <= 9)),
        ("10-24 disp purchases", (disp_train_purch >= 10) & (disp_train_purch <= 24)),
        ("25-99 disp purchases", (disp_train_purch >= 25) & (disp_train_purch <= 99)),
        ("100+ disp purchases", disp_train_purch >= 100),
    ]
    overfitting_table = []
    for label, mask in buckets:
        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            continue
        p_rate_disp = float(disp_train_purch[idx].sum() / max(disp_train_trips[idx].sum(), 1.0))
        p_rate_nondisp = float(nondisp_train_purch[idx].sum() / max(nondisp_train_trips[idx].sum(), 1.0))
        emp_log_lift = (float(np.log(max(p_rate_disp, 1e-12) / max(p_rate_nondisp, 1e-12)))
                        if disp_train_purch[idx].sum() > 0 else float("-inf"))
        entry = {
            "bucket": label,
            "n_products": int(len(idx)),
            "share_of_catalogue": float(len(idx) / J),
            "share_of_disp_trips": float(disp_train_trips[idx].sum() / max(disp_train_trips.sum(), 1.0)),
            "share_of_disp_purchases": float(disp_train_purch[idx].sum() / max(disp_train_purch.sum(), 1.0)),
            "share_of_all_purchases": float(total_train_purch[idx].sum() / max(total_train_purch.sum(), 1.0)),
            "pooled_rate_disp": p_rate_disp,
            "pooled_rate_nondisp": p_rate_nondisp,
            "pooled_empirical_log_lift": emp_log_lift,
            "models": {
                mname: {
                    "mean": float(states[mname]["w_dsp"].numpy()[idx].mean()),
                    "median": float(np.median(states[mname]["w_dsp"].numpy()[idx])),
                    "share_neg": float((states[mname]["w_dsp"].numpy()[idx] < 0).mean()),
                }
                for mname in ("base", "v4_rec_pl", "v4_rec", "v4_hist") if mname in states
            },
        }
        overfitting_table.append(entry)

    mailer_zero_purch_mask = (mail_train_trips > 0) & (mail_train_purch == 0)
    mailer_summary = {
        "never_mailed_products": int((mail_train_trips == 0).sum()),
        "mailed_zero_purch_products": int(mailer_zero_purch_mask.sum()),
        "mailed_pos_purch_products": int((mail_train_purch > 0).sum()),
        "base_w_mlr_mean_on_zero_purch": float(states["base"]["w_mlr"].numpy()[mailer_zero_purch_mask].mean()),
        "base_w_mlr_mean_on_pos_purch": float(states["base"]["w_mlr"].numpy()[mail_train_purch > 0].mean()),
        "v4_hist_w_mlr_mean_on_zero_purch": float(states["v4_hist"]["w_mlr"].numpy()[mailer_zero_purch_mask].mean()),
        "v4_hist_w_mlr_mean_on_pos_purch": float(states["v4_hist"]["w_mlr"].numpy()[mail_train_purch > 0].mean()),
    }

    # ---- 3. Candidate (a): Selection into display ----
    disp_rate_prod = disp_train_trips / np.maximum(offered_train_trips, 1.0)
    base_rate_prod = nondisp_train_purch / np.maximum(nondisp_train_trips, 1.0)
    active_disp = disp_train_trips > 0
    both_pos = (disp_train_purch > 0) & (nondisp_train_purch > 0)
    prod_emp_log_lift = np.zeros(J)
    prod_emp_log_lift[both_pos] = np.log(
        (disp_train_purch[both_pos] / disp_train_trips[both_pos])
        / (nondisp_train_purch[both_pos] / nondisp_train_trips[both_pos])
    )

    train_p_mask = valid_p & (sw_trips_at_promo > 0) & (p_disp > 0)
    pj_tr, ps_tr, pw_tr = pj[train_p_mask], ps[train_p_mask], pw[train_p_mask]
    w_tr = sw_trips_at_promo[train_p_mask]
    wk_idx = (pw_tr - 1) % 52

    selection_models = {}
    for mname in ("base", "v4_rec", "v4_hist"):
        if mname not in states:
            continue
        st = states[mname]
        mu, delta = st["mu"].numpy(), st["delta"].numpy()
        delta = delta - delta.mean(0, keepdims=True)
        zeta, xi = st["zeta"].numpy(), st["xi"].numpy()
        xi = xi - xi.mean(0, keepdims=True)
        season_at_disp = (mu[pj_tr] * delta[wk_idx]).sum(-1)
        store_at_disp = (zeta[pj_tr] * xi[ps_tr]).sum(-1)

        week_counts = np.bincount((trip_week[train_trips] - 1) % 52, minlength=delta.shape[0]).astype(np.float64)
        week_counts /= week_counts.sum()
        season_baseline = (mu[pj_tr] * (week_counts @ delta)[None, :]).sum(-1)

        store_counts = trips_per_store.astype(np.float64) / trips_per_store.sum()
        store_baseline = (zeta[pj_tr] * (store_counts @ xi)[None, :]).sum(-1)

        selection_models[mname] = {
            "season_utility_at_disp_mean": float(np.average(season_at_disp, weights=w_tr)),
            "season_utility_baseline_mean": float(np.average(season_baseline, weights=w_tr)),
            "season_utility_diff_disp_minus_base": float(np.average(season_at_disp - season_baseline, weights=w_tr)),
            "store_utility_at_disp_mean": float(np.average(store_at_disp, weights=w_tr)),
            "store_utility_baseline_mean": float(np.average(store_baseline, weights=w_tr)),
            "store_utility_diff_disp_minus_base": float(np.average(store_at_disp - store_baseline, weights=w_tr)),
        }

    selection_summary = {
        "corr_disp_rate_with_log_train_lines": float(np.corrcoef(disp_rate_prod, np.log1p(n_train_lines))[0, 1]),
        "corr_disp_rate_with_base_lam": float(np.corrcoef(disp_rate_prod, states["base"]["lam"].numpy())[0, 1]),
        "mean_nondisp_rate_never_displayed_products": float(base_rate_prod[~active_disp].mean()),
        "mean_nondisp_rate_displayed_products": float(base_rate_prod[active_disp].mean()),
        "mean_nondisp_rate_zero_disp_purch_products": float(
            base_rate_prod[(disp_train_trips > 0) & (disp_train_purch == 0)].mean()
        ),
        "both_pos_products": int(both_pos.sum()),
        "empirical_log_lift_unweighted_mean_both_pos": float(prod_emp_log_lift[both_pos].mean()),
        "empirical_log_lift_unweighted_share_neg_both_pos": float((prod_emp_log_lift[both_pos] < 0).mean()),
        "empirical_log_lift_disp_trip_weighted_mean": float(
            np.average(prod_emp_log_lift[both_pos], weights=disp_train_trips[both_pos])
        ),
        "corr_emp_log_lift_with_base_w_dsp_both_pos": float(
            np.corrcoef(prod_emp_log_lift[both_pos], states["base"]["w_dsp"].numpy()[both_pos])[0, 1]
        ),
        "corr_emp_log_lift_with_v4_hist_w_dsp_both_pos": float(
            np.corrcoef(prod_emp_log_lift[both_pos], states["v4_hist"]["w_dsp"].numpy()[both_pos])[0, 1]
        ),
        "model_season_and_store_selection": selection_models,
    }

    # ---- 4. Candidate (b): Collinearity of display with discount, price deviation, and mailer ----
    rng = np.random.default_rng(42)
    sample_trips = np.sort(rng.choice(train_trips, size=min(4000, len(train_trips)), replace=False))
    batcher_base = batcher_for(data, root, base_blob, recency=False, purchase_history=False)
    disp_all, mail_all, dlp_all, dsc_all, item_all = [], [], [], [], []
    for s in range(0, len(sample_trips), 256):
        bt = batcher_base.make(sample_trips[s:s + 256])
        disp_all.append(bt.ctx["disp"].numpy())
        mail_all.append(bt.ctx["mail"].numpy())
        dlp_all.append(bt.ctx["dlp"].numpy())
        dsc_all.append(bt.ctx["dsc"].numpy())
        item_all.append(bt.ix.item.numpy())
    disp_v = np.concatenate(disp_all)
    mail_v = np.concatenate(mail_all)
    dlp_v = np.concatenate(dlp_all)
    dsc_v = np.concatenate(dsc_all)
    item_v = np.concatenate(item_all)

    dsc_raw_all = dsc - dsc.max(axis=1, keepdims=True)
    sample_days = np.repeat(trip_day[sample_trips], [int(offered[trip_store[t]].sum()) for t in sample_trips])
    raw_dsc_v = dsc_raw_all[item_v, sample_days]

    on_disp = disp_v > 0.5
    off_disp = ~on_disp
    on_mail = mail_v > 0.5
    on_disc_5pct = raw_dsc_v <= -0.05
    on_disc_any = raw_dsc_v < -0.01

    collinearity_summary = {
        "sampled_slots": int(len(disp_v)),
        "slot_display_rate": float(on_disp.mean()),
        "slot_mailer_rate": float(on_mail.mean()),
        "slot_discount_5pct_rate": float(on_disc_5pct.mean()),
        "slot_discount_1pct_rate": float(on_disc_any.mean()),
        "given_display_share_with_discount_1pct": float(on_disc_any[on_disp].mean()),
        "given_display_share_with_discount_5pct": float(on_disc_5pct[on_disp].mean()),
        "given_display_share_with_mailer": float(on_mail[on_disp].mean()),
        "given_nodisplay_share_with_discount_1pct": float(on_disc_any[off_disp].mean()),
        "given_nodisplay_share_with_discount_5pct": float(on_disc_5pct[off_disp].mean()),
        "given_nodisplay_share_with_mailer": float(on_mail[off_disp].mean()),
        "given_mailer_share_with_discount_5pct": float(on_disc_5pct[on_mail].mean()),
        "given_mailer_share_with_display": float(on_disp[on_mail].mean()),
        "mean_raw_discount_depth_when_displayed": float(-raw_dsc_v[on_disp].mean()),
        "mean_raw_discount_depth_when_not_displayed": float(-raw_dsc_v[off_disp].mean()),
        "mean_centred_dsc_when_displayed": float(dsc_v[on_disp].mean()),
        "mean_centred_dsc_when_not_displayed": float(dsc_v[off_disp].mean()),
        "mean_dlp_when_displayed": float(dlp_v[on_disp].mean()),
        "mean_dlp_when_not_displayed": float(dlp_v[off_disp].mean()),
        "corr_disp_with_mail": float(np.corrcoef(disp_v, mail_v)[0, 1]),
        "corr_disp_with_neg_raw_dsc": float(np.corrcoef(disp_v, -raw_dsc_v)[0, 1]),
        "corr_disp_with_neg_centred_dsc": float(np.corrcoef(disp_v, -dsc_v)[0, 1]),
        "corr_disp_with_neg_dlp": float(np.corrcoef(disp_v, -dlp_v)[0, 1]),
        "corr_mail_with_neg_raw_dsc": float(np.corrcoef(mail_v, -raw_dsc_v)[0, 1]),
        "purchased_lines_share_on_display": float(tl_disp.mean()),
        "purchased_lines_share_on_mailer": float(tl_mail.mean()),
        "purchased_lines_given_display_mean_centred_dsc": float(tl_dsc[tl_disp > 0.5].mean()),
    }

    price_utility_at_display = {}
    for mname in ("base", "v4_rec", "v4_hist"):
        if mname not in states:
            continue
        st = states[mname]
        g_bar = softplus(st["gamma"]).mean(0)
        beta_j = softplus(st["beta"])
        g_j = (beta_j * g_bar[None, :]).sum(-1).numpy()
        kappa = float(softplus(st["price_kappa"]).item())
        disc_kappa = (float(softplus(st["disc_kappa"]).item())
                      if "disc_kappa" in st and st["disc_kappa"] is not None else kappa)
        disc_common = (float(st["disc_common"].item())
                       if "disc_common" in st and st["disc_common"] is not None else 0.0)
        eff_disc_slope = g_j * disc_kappa + disc_common
        eff_reg_slope = g_j * kappa
        g_slot = g_j[item_v]
        u_price_slot = -g_slot * kappa * dlp_v - (g_slot * (disc_kappa - kappa) + disc_common) * dsc_v
        u_disp_slot = st["w_dsp"].numpy()[item_v] * disp_v
        u_mail_slot = st["w_mlr"].numpy()[item_v] * mail_v
        price_utility_at_display[mname] = {
            "mean_g_j": float(g_j.mean()),
            "kappa": kappa,
            "disc_kappa": disc_kappa,
            "disc_common": disc_common,
            "mean_effective_discount_slope": float(eff_disc_slope.mean()),
            "mean_effective_regular_price_slope": float(eff_reg_slope.mean()),
            "price_utility_diff_disp_minus_nondisp": float(u_price_slot[on_disp].mean() - u_price_slot[off_disp].mean()),
            "disp_utility_on_displayed_slots": float(u_disp_slot[on_disp].mean()),
            "mail_utility_diff_disp_minus_nondisp": float(u_mail_slot[on_disp].mean() - u_mail_slot[off_disp].mean()),
            "total_promo_price_utility_diff_disp_minus_nondisp": float(
                (u_price_slot + u_disp_slot + u_mail_slot)[on_disp].mean()
                - (u_price_slot + u_disp_slot + u_mail_slot)[off_disp].mean()
            ),
        }
    collinearity_summary["price_vs_display_utility_decomposition"] = price_utility_at_display

    # ---- 5. Candidate (c): Version-4 history & latent absorption + halo_real decomposition ----
    model_v4_hist, meta_v4, parent_v4, _, _ = load_model(args.v4_dir / "v4_hist.pt", data, root)
    freeze(model_v4_hist)
    batcher_v4 = batcher_for(data, root, parent_v4, recency=True, purchase_history=True)
    init_path = _load(args.v4_dir / "v4_hist.pt")["initialization"]
    model_base, _, _, _, _ = load_model(args.base_ckpt, data, root, initialization=init_path)
    freeze(model_base)

    sample_v4_trips = np.sort(rng.choice(supported_trips(data, 0, int(meta_v4["nmax"])), size=512, replace=False))
    h_disp, h_nondisp, zphi_disp, zphi_nondisp = [], [], [], []
    disp_slots_v4, obs_slots_v4, p_base_slots, p_v4_slots, p_v4_with_base_wdsp_slots = [], [], [], [], []

    gen = torch.Generator().manual_seed(12345)
    with torch.no_grad():
        for s in range(0, len(sample_v4_trips), 128):
            bt = batcher_v4.make(sample_v4_trips[s:s + 128])
            ix = bt.ix
            bt.bind(model_v4_hist)
            h = model_v4_hist.history_utilities(ix.item, bt.ctx)
            o = model_v4_hist.history_offset(ix)
            h_centred = h - o[ix.item_trip]
            z_obs = model_v4_hist.basket_field(ix, bt.observed)
            ph = model_v4_hist.phi_r()[ix.item]
            zphi = (z_obs[ix.item_trip] * ph).sum(-1) - 0.5 * ph.square().sum(-1)

            d_mask = bt.ctx["disp"].numpy() > 0.5
            h_disp.append(h_centred.numpy()[d_mask])
            h_nondisp.append(h_centred.numpy()[~d_mask])
            zphi_disp.append(zphi.numpy()[d_mask])
            zphi_nondisp.append(zphi.numpy()[~d_mask])

            bt.bind(model_base)
            p_base = exact_incidence(model_base, ix, model_base.slot_utilities(ix))[1]

            bt.bind(model_v4_hist)
            X = draw_from_model(model_v4_hist, ix, gen, 60)
            p_v4 = torch.zeros_like(p_base)
            p_v4_bwdsp = torch.zeros_like(p_base)
            orig_wdsp = model_v4_hist.w_dsp.detach().clone()
            base_wdsp = model_base.w_dsp.detach().clone()
            for _ in range(8):
                X = gibbs(model_v4_hist, ix, X, gen)
                z = model_v4_hist.basket_field(ix, X) + torch.randn(ix.B, model_v4_hist.rank, generator=gen)
                p_v4 += exact_incidence(model_v4_hist, ix, model_v4_hist.slot_utilities_given_z(ix, z), z=z)[1] / 8.0
                model_v4_hist.w_dsp.copy_(base_wdsp)
                p_v4_bwdsp += exact_incidence(model_v4_hist, ix, model_v4_hist.slot_utilities_given_z(ix, z), z=z)[1] / 8.0
                model_v4_hist.w_dsp.copy_(orig_wdsp)

            disp_slots_v4.append(bt.ctx["disp"].numpy())
            obs_slots_v4.append(bt.observed.numpy().astype(np.float64))
            p_base_slots.append(p_base.numpy())
            p_v4_slots.append(p_v4.numpy())
            p_v4_with_base_wdsp_slots.append(p_v4_bwdsp.numpy())

    disp_s = np.concatenate(disp_slots_v4) > 0.5
    obs_s = np.concatenate(obs_slots_v4)
    p_base_s = np.concatenate(p_base_slots)
    p_v4_s = np.concatenate(p_v4_slots)
    p_v4_bwdsp_s = np.concatenate(p_v4_with_base_wdsp_slots)

    freq_disp = disp_train_purch >= 10
    top_disp = disp_train_purch >= 50
    stage_drift = {}
    for mname in ("base", "v4_rec_pl", "v4_rec", "v4_m4c_pl", "v4_m4c", "v4_hist_pl", "v4_hist", "v4_hist_inclusive"):
        if mname not in states:
            continue
        w = states[mname]["w_dsp"].numpy()
        stage_drift[mname] = {
            "all_mean": float(w.mean()),
            "all_share_neg": float((w < 0).mean()),
            "disp_trip_weighted_mean": float(np.average(w, weights=disp_train_trips)),
            "disp_purch_weighted_mean": float(np.average(w, weights=disp_train_purch)),
            "ge10_disp_purch_mean": float(w[freq_disp].mean()),
            "ge10_disp_purch_share_neg": float((w[freq_disp] < 0).mean()),
            "ge50_disp_purch_mean": float(w[top_disp].mean()),
            "ge50_disp_purch_share_neg": float((w[top_disp] < 0).mean()),
        }

    v4_absorption_summary = {
        "mean_centred_history_utility_disp": float(np.concatenate(h_disp).mean()),
        "mean_centred_history_utility_nondisp": float(np.concatenate(h_nondisp).mean()),
        "diff_centred_history_utility_disp_minus_nondisp": float(
            np.concatenate(h_disp).mean() - np.concatenate(h_nondisp).mean()
        ),
        "mean_latent_pair_utility_at_obs_basket_disp": float(np.concatenate(zphi_disp).mean()),
        "mean_latent_pair_utility_at_obs_basket_nondisp": float(np.concatenate(zphi_nondisp).mean()),
        "diff_latent_pair_utility_disp_minus_nondisp": float(
            np.concatenate(zphi_disp).mean() - np.concatenate(zphi_nondisp).mean()
        ),
        "observed_rate_on_disp_slots": float(obs_s[disp_s].mean()),
        "observed_rate_on_nondisp_slots": float(obs_s[~disp_s].mean()),
        "base_pred_rate_on_disp_slots": float(p_base_s[disp_s].mean()),
        "base_pred_rate_on_nondisp_slots": float(p_base_s[~disp_s].mean()),
        "v4_hist_pred_rate_on_disp_slots": float(p_v4_s[disp_s].mean()),
        "v4_hist_pred_rate_on_nondisp_slots": float(p_v4_s[~disp_s].mean()),
        "v4_hist_with_base_wdsp_pred_rate_on_disp_slots": float(p_v4_bwdsp_s[disp_s].mean()),
        "v4_hist_with_base_wdsp_pred_rate_on_nondisp_slots": float(p_v4_bwdsp_s[~disp_s].mean()),
        "sml_score_grad_per_disp_slot_at_base_wdsp": float((obs_s[disp_s] - p_v4_bwdsp_s[disp_s]).mean()),
        "sml_score_grad_per_disp_slot_at_v4_hist_wdsp": float((obs_s[disp_s] - p_v4_s[disp_s]).mean()),
        "stage_by_stage_w_dsp_drift": stage_drift,
    }

    # ---- 6. Stress world check (Question 3) ----
    stress_ckpts = {
        "stress_base": Path("/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/stress_clean_confirm/out/v3_pipeline_additive_best.pt"),
        "stress_hs_rec_pl": Path("/Users/ajit/Projects/nf_dunnhumby/ebi_v4t/runs/stress/hs_rec_pl.pt"),
        "stress_hs_rec": Path("/Users/ajit/Projects/nf_dunnhumby/ebi_v4t/runs/stress/hs_rec.pt"),
        "stress_v4_complik": Path("/Users/ajit/Projects/nf_dunnhumby/ebi_v4_competition_lik/runs/stress/v4_complik.pt"),
    }
    stress_models = {}
    for k, p_ck in stress_ckpts.items():
        if p_ck.exists():
            w = _load(p_ck)["model"]["w_dsp"].numpy()
            stress_models[k] = {
                "w_dsp_mean": float(w.mean()),
                "w_dsp_sd": float(w.std(ddof=1)),
                "w_dsp_median": float(np.median(w)),
                "w_dsp_share_neg": float((w < 0).mean()),
            }
    stress_summary = {
        "J": 476,
        "S": 15,
        "true_generator_promo_utility_bonus": 0.5,
        "true_generator_promo_price_multiplier": 0.75,
        "varies_across_stores_within_week": False,
        "co_timed_with_25pct_price_cut": True,
        "models": stress_models,
    }

    report = {
        "runtime_seconds": round(time.time() - t0, 2),
        "parameter_summaries": param_summary,
        "candidate_d_overfitting_by_display_purchases": overfitting_table,
        "candidate_d_mailer_comparison": mailer_summary,
        "candidate_a_selection": selection_summary,
        "candidate_b_collinearity": collinearity_summary,
        "candidate_c_v4_absorption_and_stage_drift": v4_absorption_summary,
        "stress_world_display": stress_summary,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[diagnose] wrote {args.out} in {time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
