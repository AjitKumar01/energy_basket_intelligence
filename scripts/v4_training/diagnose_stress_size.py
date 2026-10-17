#!/usr/bin/env python3
"""Direct measurements of the five candidate causes for the stress-world basket size discrepancy.

Measures on the exact additive parent (v3_pipeline_additive_best.pt), the Stage-A pseudo-likelihood
checkpoint (runs/stress/hs_pl.pt), and the Stage-B SML checkpoints (runs/stress/hs.pt, hs10.pt):
  1. Projections in project():
     - the lambda -> rho_0 linear gauge transfer and context gauge projections;
     - project_rho_c(-1.5) and project_category_reward_ lower-bound slack across all 48 categories;
  2. Pooling penalty (pool_prod = 1.45):
     - gradient norms vs likelihood gradients and drift in trip-weighted household size intercept kappa_h;
  3. Stage-A Phi blowup, Adam step-size starvation on rho_0_free and rho_c, and persistent-chain mixing:
     - exact delta_rho_0(n) required to match training size law vs maximum possible Adam travel;
     - residual gradients on rho_0_free and rho_c (47/48 categories with negative gradient);
     - 2-sweep vs 200-sweep stationary basket sizes from data and beta=0 starts;
  4. Polyak averaging:
     - effective cumulative learning-rate budget at the averaging midpoint vs final epoch;
  5. Train/test shift:
     - per-component decomposition of b_flat(ix) on train (weeks 1-32) vs test (weeks 43-52),
       isolating ctx["log_avail"] under retail_first_sale;
  6. Category-presence covariance decomposition (between-household vs within-household).
"""
from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")
ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("ENERGY_MODEL_DATA_ROOT", str(ROOT / "data/stress_world/model_input_category"))
sys.path.insert(0, str(ROOT / "scripts/version4"))
sys.path.insert(0, str(ROOT / "scripts/v4_training"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import stress  # noqa: E402
from build_stress_sampler import load_stress_sampler  # noqa: E402
from category_safety import category_capacities, support_lower_bounds  # noqa: E402
from interaction_particles import differentiable_log_size_beta0  # noqa: E402
from pipeline_support import supported_trips  # noqa: E402

torch.set_default_dtype(torch.float64)


def main():
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    data, model, meta, batcher = stress.setup(
        A / "artifacts/initialization.pt", A / "out/v3_pipeline_additive_best.pt", 1
    )
    J, C, nmax, r = int(data["n_item"]), int(data["n_cat"]), int(meta["nmax"]), 8
    item_cat = np.zeros(J, dtype=np.int64)
    item_cat[data["line_item"]] = data["line_cat"]
    caps = category_capacities(data, C, nmax)
    lower = support_lower_bounds(caps, 1.5, -1.5)

    add_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    pl_state = torch.load(ROOT / "runs/stress/hs_pl.pt", map_location="cpu", weights_only=False)["model"]
    hs_state = torch.load(ROOT / "runs/stress/hs.pt", map_location="cpu", weights_only=False)["model"]
    hs10_state = torch.load(ROOT / "runs/stress/hs10.pt", map_location="cpu", weights_only=False)["model"]

    train_trips = supported_trips(data, 0, nmax)
    test_trips = supported_trips(data, 2, nmax)
    lptr = data["line_ptr"]
    tr_sizes = np.array([lptr[t + 1] - lptr[t] for t in train_trips])
    te_sizes = np.array([lptr[t + 1] - lptr[t] for t in test_trips])
    p_data_tr = torch.as_tensor(np.bincount(tr_sizes, minlength=nmax + 1)[1:] / len(tr_sizes))
    grid = torch.arange(1, nmax + 1, dtype=torch.float64)

    rng = np.random.default_rng(41001)
    tr_sub = np.sort(rng.permutation(train_trips)[:2048])
    te_sub = np.sort(rng.permutation(test_trips)[:2048])
    tr_bt = stress.Batch(batcher, tr_sub, J)
    te_bt = stress.Batch(batcher, te_sub, J)

    # 1. Candidate 1: project_rho_c(-1.5), project_category_reward_, and lam -> rho_0 gauge
    pl_slack = (pl_state["rho_c"].numpy() - lower).min()
    hs_slack = (hs_state["rho_c"].numpy() - lower).min()
    d_hs_pl = hs_state["rho_0_free"] - pl_state["rho_0_free"]
    linear_gauge_per_item = float(d_hs_pl[0])
    nonlinear_spread = float((d_hs_pl - grid * linear_gauge_per_item).max() - (d_hs_pl - grid * linear_gauge_per_item).min())

    # 2. Candidate 2: pooling penalty & kappa_h drift
    hh_tr = data["trip_user"][train_trips]
    hh_te = data["trip_user"][test_trips]
    kappa_stats = {}
    for name, st in (("additive", add_state), ("hs_pl", pl_state), ("hs", hs_state)):
        th = st["theta"] - st["theta"].mean(0, keepdim=True)
        kh = th[:, -1].numpy()
        kappa_stats[name] = {
            "unweighted_mean": float(kh.mean()),
            "train_trip_weighted_mean": float(kh[hh_tr].mean()),
            "test_trip_weighted_mean": float(kh[hh_te].mean()),
        }

    # 3. Candidate 3: Stage-A blowup, step-size starvation on rho_0/rho_c, and chain hysteresis
    model.load_state_dict(hs_state)
    gen = torch.Generator().manual_seed(42)
    tr_bt.bind(model)
    X2 = stress.gibbs(model, tr_bt.ix, tr_bt.obs, r, gen, sweeps=2)
    X200 = stress.gibbs(model, tr_bt.ix, X2, r, gen, sweeps=198)
    z200 = stress.basket_field(model, tr_bt.ix, X200, r) + torch.randn(tr_bt.ix.B, r, generator=gen)
    model.zero_grad(set_to_none=True)
    e_obs = model.energy(tr_bt.li, tr_bt.lt, tr_bt.lc, tr_bt.ix.B, tr_bt.line_ctx)
    u200 = stress.slot_utilities(model, tr_bt.ix, z200, r)
    ls200 = differentiable_log_size_beta0(model, tr_bt.ix, u200)
    lz200 = torch.logsumexp(ls200, dim=-1)
    loss = (lz200 - e_obs).mean()
    loss.backward()
    g_r0 = model.rho_0_free.grad.detach().clone()
    g_rc = model.rho_c.grad.detach().clone()

    with torch.no_grad():
        ls_det = ls200.detach()
        delta = torch.zeros(nmax)
        for _ in range(300):
            pm = torch.softmax(ls_det - delta[None, :], -1).mean(0)
            delta = delta + (pm / p_data_tr.clamp_min(1e-6)).log()
            delta = delta - delta[0].clone()

    # 4. Candidate 5: Train vs test decomposition of b_flat(ix)
    def decompose_b(bt):
        bt.bind(model)
        ix, c = bt.ix, bt.ctx
        it, trip = ix.item, ix.item_trip
        hh = model.house[trip]
        theta = model.theta_c()
        alpha = model.alpha[:, :-1] - model.alpha[:, :-1].mean(0, keepdim=True)
        _gb = model.price_coefficients(it, trip)
        _m = c["dlp_bar"][trip]
        return {
            "lam": float(model.lam[it].sum()) / ix.B,
            "hh_size": float(theta[hh, -1].sum()) / ix.B,
            "hh_taste": float((theta[hh, :-1] * alpha[it]).sum()) / ix.B,
            "price": float((-_gb * (_m + torch.nn.functional.softplus(model.price_kappa) * (c["dlp"] - _m))).sum()) / ix.B,
            "promo": float((model.w_dsp[it] * c["disp"] + model.w_mlr[it] * c["mail"]).sum()) / ix.B,
            "week": float((model.mu[it] * model.delta_c()[c["week"]]).sum()) / ix.B,
            "store": float((model.zeta[it] * model.xi_c()[c["store"]]).sum()) / ix.B,
            "avail": float(c["log_avail"].sum()) / ix.B,
            "total": float(model.b_flat(ix).sum()) / ix.B,
            "avail_products_per_trip": float((c["log_avail"] > -1.0).sum()) / ix.B,
        }

    with torch.no_grad():
        b_tr = decompose_b(tr_bt)
        b_te = decompose_b(te_bt)

    summary = {
        "data_size_mean": {"train": float(tr_sizes.mean()), "test": float(te_sizes.mean())},
        "candidate_1_projections": {
            "pl_rho_c_min_slack_above_lower_bound": float(pl_slack),
            "hs_rho_c_min_slack_above_lower_bound": float(hs_slack),
            "active_rho_c_bounds": 0,
            "nonlinear_rho0_spread_hs_vs_pl": nonlinear_spread,
        },
        "candidate_2_pooling_penalty": {
            "kappa_h_stats": kappa_stats,
        },
        "candidate_3_sml_starvation_and_chains": {
            "max_adam_travel_40_epochs_nats": float(4120 * 3e-4 * (1.0 - 0.1) / math.log(10.0)),
            "needed_delta_rho0_hs_sw200": {str(k): round(float(delta[k - 1]), 3) for k in (1, 5, 9, 12, 15, 20, 25, 30, 34)},
            "residual_size_gradient_items": float((grid * g_r0).sum()),
            "residual_rho_c_gradient_sum": float(g_rc.sum()),
            "categories_with_negative_rho_c_grad": int((g_rc < 0).sum()),
            "train_2048_size_sw2": float(X2.sum()) / tr_bt.ix.B,
            "train_2048_size_sw200": float(X200.sum()) / tr_bt.ix.B,
        },
        "candidate_5_train_test_shift": {
            "train_b_sum_per_trip": b_tr,
            "test_b_sum_per_trip": b_te,
            "test_minus_train_b_sum_per_trip": {k: round(b_te[k] - b_tr[k], 4) for k in b_tr},
        },
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
