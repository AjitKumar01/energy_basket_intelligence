#!/usr/bin/env python3
"""Validate Step-1 Gate E importance-sampling log Z estimator (is_log_z vs is_log_z_orig)
against exact enumeration across interaction strengths on 200 small-world contexts,
for baseline Version-4, M1 (free category-count potential f_c(n_c)), M2 (M1 + presence-based s(S)),
M3 (2D category-count/size DP), M4a (category-nested quadratic), and M4b (per-category IV slope
(lambda_c - 1) * IV_c(x), including exact enumeration vs DP, score and gradient checks with lambda != 1,
and lambda = 1 reproduction of hs_rec.pt on 256 trips)."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("V3_AFFINITY", "1")
os.environ.setdefault(
    "ENERGY_MODEL_DATA_ROOT",
    "/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/data/stress_world/model_input_category",
)
sys.modules.setdefault("pathlib._local", pathlib)

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/version4"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from hs import is_log_z, is_log_z_orig, log_zc  # noqa: E402
from world import Support, contexts, exact_log_z, make_truth  # noqa: E402

torch.set_default_dtype(torch.float64)


def run_mode(
    args,
    sup,
    hp,
    cw,
    use_f_extra=False,
    presence_phi=False,
    cat_rho0=False,
    u_cat=False,
    iv_slope=False,
):
    J, H, W, K, r, nmax = 18, 100, 8, 3, 3, 5
    results = []
    for strength in args.strengths:
        truth, week_price = make_truth(
            args.seed,
            strength,
            J=J,
            H=H,
            W=W,
            K=K,
            r=r,
            nmax=nmax,
            use_f_extra=use_f_extra,
            presence_phi=presence_phi,
            cat_rho0=cat_rho0,
            u_cat=u_cat,
            iv_slope=iv_slope,
        )
        pp = week_price[torch.as_tensor(cw)]

        # Verify exact beta=0 DP normalizer log_zc against brute-force enumeration at Phi=0
        if strength == args.strengths[0]:
            t_zero, _ = make_truth(
                args.seed,
                strength,
                J=J,
                H=H,
                W=W,
                K=K,
                r=r,
                nmax=nmax,
                use_f_extra=use_f_extra,
                presence_phi=presence_phi,
                cat_rho0=cat_rho0,
                u_cat=u_cat,
                iv_slope=iv_slope,
            )
            with torch.no_grad():
                t_zero.p["phi"].zero_()
                if presence_phi:
                    t_zero.p["psi"].zero_()
                    t_zero.p["delta_phi"].zero_()
                if u_cat:
                    t_zero.p["u_cat"].zero_()
                ex0 = exact_log_z(t_zero, sup, hp, pp)
                b0 = t_zero.b(hp, pp)
                iv_sh0 = t_zero.iv_shift_from_b(b0)
                dp0 = log_zc(
                    b0,
                    t_zero.part,
                    t_zero.p["rho_c"],
                    t_zero.rho0(),
                    f_extra=t_zero.f_extra(),
                    cat_shift=None,
                    cat_rho0=cat_rho0,
                    iv_shift=iv_sh0,
                )
                dp_err = float((ex0 - dp0).abs().max())
                assert dp_err < 1e-12, f"DP log_zc mismatch vs enumeration at Phi=0: {dp_err}"

        exact = exact_log_z(truth, sup, hp, pp)

        est_o, ess_o = is_log_z_orig(
            truth,
            hp,
            pp,
            torch.Generator().manual_seed(args.seed + 12),
            burn=args.is_burn,
            components=args.is_components,
            thin=args.is_thin,
            draws=args.is_draws,
        )
        err_o = (est_o - exact).abs()

        est_f, ess_f = is_log_z(
            truth,
            hp,
            pp,
            torch.Generator().manual_seed(args.seed + 12),
            burn=args.is_burn,
            components=args.is_components,
            thin=args.is_thin,
            draws=args.is_draws,
        )
        err_f = (est_f - exact).abs()

        row = {
            "strength": float(strength),
            "orig": {
                "mean_abs_err": round(float(err_o.mean()), 5),
                "max_abs_err": round(float(err_o.max()), 5),
                "mean_ess": round(float(ess_o.mean()), 4),
                "min_ess": round(float(ess_o.min()), 4),
                "pass_gate_E": bool(err_o.mean() <= 0.01 and err_o.max() <= 0.05),
            },
            "fixed": {
                "mean_abs_err": round(float(err_f.mean()), 5),
                "max_abs_err": round(float(err_f.max()), 5),
                "mean_ess": round(float(ess_f.mean()), 4),
                "min_ess": round(float(ess_f.min()), 4),
                "pass_gate_E": bool(err_f.mean() <= 0.01 and err_f.max() <= 0.05),
            },
        }
        mode_tag = (
            "m4b"
            if iv_slope
            else (
                "m4a"
                if u_cat
                else ("m3" if cat_rho0 else ("m2" if presence_phi else ("m1" if use_f_extra else "baseline")))
            )
        )
        print(json.dumps({"mode": mode_tag, **row}), flush=True)
        results.append(row)
    return results


def check_m4b_dp_and_gradients_small_world(sup: Support, hp: torch.Tensor, cw: np.ndarray, seed: int = 101) -> dict:
    """On small worlds with lambda != 1, verify:
      1. Exact enumeration vs DP (`log_zc`) across both `cat_rho0=False` and `cat_rho0=True`
         and multiple lambda_c values in (0, 1).
      2. Exact score E(y) - log Z(x) and autograd gradients d/dtheta (E(y) - log Z(x))
         at Phi=0 (DP vs exact enumeration and vs central finite differences) and at Phi!=0
         (exact enumeration autograd vs central finite differences) for all parameters including
         `iv_slope_raw`, `lam`, `theta`, `alpha`, `graw`, `delta`, `rho_c`, `rho0_free`, `f_extra`, `u_cat`.
      3. `stress.py` M4b path (`obs_energy`, `log_zc`, `pseudo_loglik`, `row_iv_shift`) score and
         autograd gradients vs central finite differences on a small stress-world batch with lambda != 1.
    """
    J, H, W, K, r, nmax = 18, 100, 8, 3, 3, 5
    hp_sub = hp[:32]
    cw_sub = cw[:32]

    # 1. Exact enumeration vs DP across cat_rho0 in (False, True) and several lambda vectors != 1
    dp_checks = []
    for cat_rho0_flag in (False, True):
        for s_offset in range(3):
            t_zero, week_price = make_truth(
                seed + s_offset,
                1.0,
                J=J,
                H=H,
                W=W,
                K=K,
                r=r,
                nmax=nmax,
                use_f_extra=True,
                presence_phi=True,
                cat_rho0=cat_rho0_flag,
                u_cat=True,
                iv_slope=True,
            )
            pp_sub = week_price[torch.as_tensor(cw_sub)]
            with torch.no_grad():
                t_zero.p["phi"].zero_()
                t_zero.p["psi"].zero_()
                t_zero.p["delta_phi"].zero_()
                ex = exact_log_z(t_zero, sup, hp_sub, pp_sub)
                b = t_zero.b(hp_sub, pp_sub)
                iv_sh = t_zero.iv_shift_from_b(b)
                dp = log_zc(
                    b,
                    t_zero.part,
                    t_zero.p["rho_c"],
                    t_zero.rho0(),
                    f_extra=t_zero.f_extra(),
                    cat_shift=None,
                    cat_rho0=cat_rho0_flag,
                    iv_shift=iv_sh,
                )
                max_err = float((ex - dp).abs().max())
                mean_err = float((ex - dp).abs().mean())
                assert max_err < 1e-12, f"M4b DP vs exact mismatch: {max_err}"
                lam_c = torch.sigmoid(t_zero.p["iv_slope_raw"])
                dp_checks.append(
                    {
                        "cat_rho0": cat_rho0_flag,
                        "seed_offset": s_offset,
                        "lambda_c_min": round(float(lam_c.min()), 6),
                        "lambda_c_max": round(float(lam_c.max()), 6),
                        "lambda_c_mean": round(float(lam_c.mean()), 6),
                        "max_abs_err": max_err,
                        "mean_abs_err": mean_err,
                    }
                )

    # 2. Score and gradient checks on small world at Phi=0 (DP vs exact enumeration & finite diff)
    #    and at Phi != 0 (exact enumeration vs finite diff)
    t_phi0, week_price = make_truth(
        seed,
        3.0,
        J=J,
        H=H,
        W=W,
        K=K,
        r=r,
        nmax=nmax,
        use_f_extra=True,
        presence_phi=True,
        cat_rho0=True,
        u_cat=True,
        iv_slope=True,
    )
    pp_sub = week_price[torch.as_tensor(cw_sub)]
    with torch.no_grad():
        t_phi0.p["phi"].zero_()
        t_phi0.p["psi"].zero_()
        t_phi0.p["delta_phi"].zero_()

    # Pick 32 synthetic non-empty baskets from sup.M
    rng_y = np.random.default_rng(seed + 99)
    y_idx = rng_y.choice(len(sup.M), size=len(hp_sub), replace=True)
    y_obs = sup.M[torch.as_tensor(y_idx)]  # B x J

    def dp_mean_ll(m):
        b = m.b(hp_sub, pp_sub)
        iv_sh = m.iv_shift_from_b(b)
        lz = log_zc(
            b,
            m.part,
            m.p["rho_c"],
            m.rho0(),
            f_extra=m.f_extra(),
            cat_shift=None,
            cat_rho0=True,
            iv_shift=iv_sh,
        )
        en = m.energy(y_obs, hp_sub, pp_sub)
        return (en - lz).mean()

    def exact_mean_ll(m):
        from world import energy_table

        E = energy_table(m, sup, hp_sub, pp_sub)
        lz_ex = torch.logsumexp(E, dim=1)
        en = m.energy(y_obs, hp_sub, pp_sub)
        return (en - lz_ex).mean()

    for q in t_phi0.p.values():
        q.requires_grad_(True)
    param_names_phi0 = [
        "iv_slope_raw",
        "lam",
        "theta",
        "alpha",
        "graw",
        "delta",
        "rho_c",
        "rho0_free",
        "f_extra",
        "u_cat",
    ]
    ll_dp = dp_mean_ll(t_phi0)
    grads_dp = torch.autograd.grad(ll_dp, [t_phi0.p[n] for n in param_names_phi0])
    ll_ex0 = exact_mean_ll(t_phi0)
    grads_ex0 = torch.autograd.grad(ll_ex0, [t_phi0.p[n] for n in param_names_phi0])
    score_dp_vs_exact_err = float((ll_dp - ll_ex0).abs().item())
    assert score_dp_vs_exact_err < 1e-12, f"Score DP vs exact mismatch: {score_dp_vs_exact_err}"

    grad_dp_vs_exact_errs = {}
    grad_dp_vs_fd_errs = {}
    eps = 1e-6
    for name, g_dp, g_ex in zip(param_names_phi0, grads_dp, grads_ex0):
        err_ex = float((g_dp - g_ex).abs().max().item())
        assert err_ex < 1e-12, f"Grad DP vs exact mismatch on {name}: {err_ex}"
        grad_dp_vs_exact_errs[name] = err_ex

        par = t_phi0.p[name]
        flat = par.detach().view(-1)
        g_flat = g_dp.detach().view(-1)
        n_check = min(len(flat), 6)
        max_fd_err = 0.0
        with torch.no_grad():
            for idx in range(n_check):
                orig_v = float(flat[idx].item())
                flat[idx] = orig_v + eps
                lp = float(dp_mean_ll(t_phi0).item())
                flat[idx] = orig_v - eps
                lm = float(dp_mean_ll(t_phi0).item())
                flat[idx] = orig_v
                fd = (lp - lm) / (2.0 * eps)
                max_fd_err = max(max_fd_err, abs(fd - float(g_flat[idx].item())))
        assert max_fd_err < 1e-7, f"Grad DP vs FD mismatch on {name}: {max_fd_err}"
        grad_dp_vs_fd_errs[name] = max_fd_err

    # Also check full interacting model (Phi != 0) exact score & gradient vs finite differences
    t_full, _ = make_truth(
        seed,
        3.0,
        J=J,
        H=H,
        W=W,
        K=K,
        r=r,
        nmax=nmax,
        use_f_extra=True,
        presence_phi=True,
        cat_rho0=True,
        u_cat=True,
        iv_slope=True,
    )
    for q in t_full.p.values():
        q.requires_grad_(True)
    ll_full = exact_mean_ll(t_full)
    full_names = ["iv_slope_raw", "lam", "theta", "alpha", "graw", "delta", "psi", "delta_phi", "u_cat", "rho_c"]
    grads_full = torch.autograd.grad(ll_full, [t_full.p[n] for n in full_names])
    grad_full_vs_fd_errs = {}
    for name, g_f in zip(full_names, grads_full):
        par = t_full.p[name]
        flat = par.detach().view(-1)
        g_flat = g_f.detach().view(-1)
        n_check = min(len(flat), 6)
        max_fd_err = 0.0
        with torch.no_grad():
            for idx in range(n_check):
                orig_v = float(flat[idx].item())
                flat[idx] = orig_v + eps
                lp = float(exact_mean_ll(t_full).item())
                flat[idx] = orig_v - eps
                lm = float(exact_mean_ll(t_full).item())
                flat[idx] = orig_v
                fd = (lp - lm) / (2.0 * eps)
                max_fd_err = max(max_fd_err, abs(fd - float(g_flat[idx].item())))
        assert max_fd_err < 1e-7, f"Full model grad vs FD mismatch on {name}: {max_fd_err}"
        grad_full_vs_fd_errs[name] = max_fd_err

    # 3. Check stress.py's M4b implementation (obs_energy, log_zc, pseudo_loglik) score & gradients
    import stress

    init_path = Path(
        "/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/stress_clean_confirm/artifacts/initialization.pt"
    )
    add_path = Path(
        "/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/stress_clean_confirm/out/v3_pipeline_additive_best.pt"
    )
    blob = torch.load(ROOT / "runs/stress/hs_rec.pt", map_location="cpu", weights_only=False)
    data, model, meta, batcher = stress.setup(
        init_path,
        add_path,
        4,
        recency=True,
        iv_slope=True,
    )
    stress.configure_model_from_blob(model, blob)
    model.has_iv_slope = True
    r_rank = int(blob["rank"])
    J_stress = int(data["n_item"])
    C_stress = int(data["n_cat"])
    if getattr(model, "iv_slope_raw", None) is None:
        model.iv_slope_raw = torch.nn.Parameter(torch.zeros(C_stress, dtype=torch.float64))
    with torch.no_grad():
        gen_init = torch.Generator().manual_seed(777)
        model.iv_slope_raw.copy_(torch.linspace(-1.2, 0.8, C_stress) + 0.1 * torch.randn(C_stress, generator=gen_init))

    bt = stress.Batch(batcher, np.arange(16, dtype=np.int64), J_stress)
    bt.bind(model)
    z_fixed = torch.randn(bt.ix.B, r_rank, generator=torch.Generator().manual_seed(888))

    def stress_surrogate_obj():
        bt.bind(model)
        u = stress.slot_utilities(model, bt.ix, z_fixed, r_rank)
        en = stress.obs_energy(model, bt, r_rank)
        lz = stress.log_zc(model, bt.ix, u, z=z_fixed)
        return (en - lz).mean()

    for q in model.parameters():
        q.requires_grad_(True)
    obj_val = stress_surrogate_obj()
    stress_params = [
        ("iv_slope_raw", model.iv_slope_raw),
        ("lam", model.lam),
        ("alpha", model.alpha),
        ("theta", model.theta),
        ("mu", model.mu),
        ("delta", model.delta),
        ("rho_c", model.rho_c),
        ("rho_0_free", model.rho_0_free),
        ("f_extra", model.f_extra),
        ("u_cat", model.u_cat),
        ("psi_item", model.psi_item),
    ]
    grads_stress = torch.autograd.grad(obj_val, [p for _, p in stress_params])
    stress_grad_fd_errs = {}
    gauge_centered = {"alpha", "theta", "delta"}
    for (name, par), g_s in zip(stress_params, grads_stress):
        max_fd_err = 0.0
        if name in gauge_centered:
            # RaggedModel detaches .mean(0, keepdim=True) on alpha/theta/delta so gradients
            # live in ambient coordinates and are projected to zero mean after Adam; a zero-mean
            # perturbation (+eps on row r0, -eps on row r1 in the same column) keeps .mean(0) fixed.
            g_mat = g_s.detach()
            nz_rows = torch.nonzero(g_mat[:, 0].abs() > 1e-10, as_tuple=False).view(-1)
            r0 = int(nz_rows[0].item()) if len(nz_rows) > 0 else 0
            r1 = int(nz_rows[1].item()) if len(nz_rows) > 1 else (1 if par.shape[0] > 1 else 0)
            with torch.no_grad():
                for col in range(min(par.shape[1], 4)):
                    v0 = float(par[r0, col].item())
                    v1 = float(par[r1, col].item())
                    par[r0, col] = v0 + eps
                    par[r1, col] = v1 - eps
                    lp = float(stress_surrogate_obj().item())
                    par[r0, col] = v0 - eps
                    par[r1, col] = v1 + eps
                    lm = float(stress_surrogate_obj().item())
                    par[r0, col] = v0
                    par[r1, col] = v1
                    fd = (lp - lm) / (2.0 * eps)
                    ag = float((g_mat[r0, col] - g_mat[r1, col]).item())
                    max_fd_err = max(max_fd_err, abs(fd - ag))
        else:
            flat = par.detach().view(-1)
            g_flat = g_s.detach().view(-1)
            nz = torch.nonzero(g_flat.abs() > 1e-10, as_tuple=False).view(-1)
            indices = nz[:5].tolist() if len(nz) > 0 else list(range(min(len(flat), 5)))
            with torch.no_grad():
                for idx in indices:
                    orig_v = float(flat[idx].item())
                    flat[idx] = orig_v + eps
                    lp = float(stress_surrogate_obj().item())
                    flat[idx] = orig_v - eps
                    lm = float(stress_surrogate_obj().item())
                    flat[idx] = orig_v
                    fd = (lp - lm) / (2.0 * eps)
                    max_fd_err = max(max_fd_err, abs(fd - float(g_flat[idx].item())))
        assert max_fd_err < 1e-7, f"stress.py M4b grad vs FD mismatch on {name}: {max_fd_err}"
        stress_grad_fd_errs[name] = max_fd_err

    # Also verify pseudo_loglik gradients with respect to iv_slope_raw and lam against finite differences
    for q in model.parameters():
        q.requires_grad_(True)
    bt.bind(model)
    pll_val = stress.pseudo_loglik(model, bt.ix, bt.obs, r_rank)
    g_pll_iv, g_pll_lam = torch.autograd.grad(pll_val, [model.iv_slope_raw, model.lam])
    pll_grad_fd_errs = {}
    for name, par, g_p in (("iv_slope_raw", model.iv_slope_raw, g_pll_iv), ("lam", model.lam, g_pll_lam)):
        flat = par.detach().view(-1)
        g_flat = g_p.detach().view(-1)
        nz = torch.nonzero(g_flat.abs() > 1e-10, as_tuple=False).view(-1)
        indices = nz[:5].tolist() if len(nz) > 0 else list(range(min(len(flat), 5)))
        max_fd_err = 0.0
        with torch.no_grad():
            for idx in indices:
                orig_v = float(flat[idx].item())
                flat[idx] = orig_v + eps
                bt.bind(model)
                lp = float(stress.pseudo_loglik(model, bt.ix, bt.obs, r_rank).item())
                flat[idx] = orig_v - eps
                bt.bind(model)
                lm = float(stress.pseudo_loglik(model, bt.ix, bt.obs, r_rank).item())
                flat[idx] = orig_v
                fd = (lp - lm) / (2.0 * eps)
                max_fd_err = max(max_fd_err, abs(fd - float(g_flat[idx].item())))
        assert max_fd_err < 1e-7, f"stress.py pseudo_loglik grad vs FD mismatch on {name}: {max_fd_err}"
        pll_grad_fd_errs[name] = max_fd_err

    return {
        "dp_vs_exact_enumeration": dp_checks,
        "phi0_score_dp_vs_exact_abs_err": score_dp_vs_exact_err,
        "phi0_grad_dp_vs_exact_max_abs_err": grad_dp_vs_exact_errs,
        "phi0_grad_dp_vs_finite_diff_max_abs_err": grad_dp_vs_fd_errs,
        "full_interacting_grad_vs_finite_diff_max_abs_err": grad_full_vs_fd_errs,
        "stress_py_m4b_grad_vs_finite_diff_max_abs_err": stress_grad_fd_errs,
        "stress_py_pseudo_loglik_grad_vs_finite_diff_max_abs_err": pll_grad_fd_errs,
    }


def check_lambda_1_reproduces_hs_rec_256_trips() -> dict:
    """Verify that M4b with lambda = 1 reproduces runs/stress/hs_rec.pt scores to within 1e-10
    on 256 test trips (checking obs_energy, log_zc, pseudo_loglik, and full IS log-likelihood)."""
    import stress

    stress.USE_CPP_SAMPLER = True
    ckpt_path = ROOT / "runs/stress/hs_rec.pt"
    init_path = Path(
        "/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/stress_clean_confirm/artifacts/initialization.pt"
    )
    add_path = Path(
        "/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/stress_clean_confirm/out/v3_pipeline_additive_best.pt"
    )
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    r_rank = int(blob["rank"])

    data_a, model_a, _, batcher_a = stress.setup(init_path, add_path, 4, recency=True, iv_slope=False)
    stress.configure_model_from_blob(model_a, blob)
    model_a.eval()

    data_b, model_b, _, batcher_b = stress.setup(init_path, add_path, 4, recency=True, iv_slope=True)
    stress.configure_model_from_blob(model_b, blob)
    C = int(data_b["n_cat"])
    if getattr(model_b, "iv_slope_raw", None) is None:
        model_b.iv_slope_raw = torch.nn.Parameter(torch.full((C,), 40.0, dtype=torch.float64))
    else:
        with torch.no_grad():
            model_b.iv_slope_raw.fill_(40.0)
    model_b.has_iv_slope = True
    model_b.iv_slope_override = torch.ones(C, dtype=torch.float64)
    model_b.eval()

    for q in model_a.parameters():
        q.requires_grad_(False)
    for q in model_b.parameters():
        q.requires_grad_(False)

    J = int(data_a["n_item"])
    ref_npz = np.load(
        "/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/stress_clean_confirm/reports/likelihood_test_paired_per_trip.npz"
    )
    rows_256 = ref_npz["trips"][:256].astype(np.int64)

    bt_a = stress.Batch(batcher_a, rows_256, J)
    bt_b = stress.Batch(batcher_b, rows_256, J)
    bt_a.bind(model_a)
    bt_b.bind(model_b)

    z_test = torch.randn(256, r_rank, generator=torch.Generator().manual_seed(12345))
    with torch.no_grad():
        en_a = stress.obs_energy(model_a, bt_a, r_rank)
        en_b = stress.obs_energy(model_b, bt_b, r_rank)
        u_a = stress.slot_utilities(model_a, bt_a.ix, z_test, r_rank)
        u_b = stress.slot_utilities(model_b, bt_b.ix, z_test, r_rank)
        lz_a = stress.log_zc(model_a, bt_a.ix, u_a, z=z_test)
        lz_b = stress.log_zc(model_b, bt_b.ix, u_b, z=z_test)
        pll_a = stress.pseudo_loglik(model_a, bt_a.ix, bt_a.obs, r_rank)
        pll_b = stress.pseudo_loglik(model_b, bt_b.ix, bt_b.obs, r_rank)

    gen_a = torch.Generator().manual_seed(41001 + 5)
    gen_b = torch.Generator().manual_seed(41001 + 5)
    isz_a, _, ess_a, _ = stress.is_log_z(model_a, bt_a, r_rank, gen_a, burn=200, components=128, thin=4, draws=1024)
    isz_b, _, ess_b, _ = stress.is_log_z(model_b, bt_b, r_rank, gen_b, burn=200, components=128, thin=4, draws=1024)
    with torch.no_grad():
        ll_a = stress.obs_energy(model_a, bt_a, r_rank) - isz_a
        ll_b = stress.obs_energy(model_b, bt_b, r_rank) - isz_b

    # Also test without iv_slope_override using sigmoid(40.0) = 1.0 in float64
    model_b.iv_slope_override = None
    gen_b_raw = torch.Generator().manual_seed(41001 + 5)
    isz_b_raw, *_ = stress.is_log_z(model_b, bt_b, r_rank, gen_b_raw, burn=200, components=128, thin=4, draws=1024)
    with torch.no_grad():
        ll_b_raw = stress.obs_energy(model_b, bt_b, r_rank) - isz_b_raw

    saved_npz = np.load(ROOT / "runs/stress/hs_rec.per_trip.npz")
    saved_ll_256 = torch.as_tensor(saved_npz["ll"][:256], dtype=torch.float64)

    max_en_diff = float((en_a - en_b).abs().max().item())
    max_lz_diff = float((lz_a - lz_b).abs().max().item())
    max_pll_diff = abs(float(pll_a - pll_b))
    max_ll_diff = float((ll_a - ll_b).abs().max().item())
    max_ll_raw_diff = float((ll_a - ll_b_raw).abs().max().item())
    max_saved_diff = float((ll_b - saved_ll_256).abs().max().item())

    assert max_en_diff < 1e-10, f"obs_energy diff {max_en_diff} >= 1e-10"
    assert max_lz_diff < 1e-10, f"log_zc diff {max_lz_diff} >= 1e-10"
    assert max_pll_diff < 1e-10, f"pseudo_loglik diff {max_pll_diff} >= 1e-10"
    assert max_ll_diff < 1e-10, f"per-trip IS LL diff {max_ll_diff} >= 1e-10"
    assert max_ll_raw_diff < 1e-10, f"per-trip IS LL (sigmoid(40)) diff {max_ll_raw_diff} >= 1e-10"
    assert max_saved_diff < 1e-10, f"per-trip IS LL vs saved hs_rec.per_trip.npz diff {max_saved_diff} >= 1e-10"

    return {
        "n_trips": 256,
        "obs_energy_max_abs_diff": max_en_diff,
        "log_zc_max_abs_diff": max_lz_diff,
        "pseudo_loglik_abs_diff": max_pll_diff,
        "per_trip_ll_max_abs_diff": max_ll_diff,
        "per_trip_ll_sigmoid40_max_abs_diff": max_ll_raw_diff,
        "per_trip_ll_vs_saved_npz_max_abs_diff": max_saved_diff,
        "mean_ll_256": float(ll_b.mean().item()),
        "mean_ess_256": float(ess_b.mean().item()),
        "pass_1e10": True,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("baseline", "m1", "m2", "m3", "m4a", "m4b", "all"), default="m4b")
    p.add_argument("--seed", type=int, default=101)
    p.add_argument("--strengths", type=float, nargs="+", default=[1.0, 3.0, 7.0, 25.0, 50.0, 75.0])
    p.add_argument("--panel", type=int, default=200)
    p.add_argument("--is-burn", type=int, default=200)
    p.add_argument("--is-components", type=int, default=256)
    p.add_argument("--is-thin", type=int, default=4)
    p.add_argument("--is-draws", type=int, default=2048)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, default=Path("runs/stress/gate_e_m4b.json"))
    args = p.parse_args()

    torch.set_num_threads(args.threads)
    J, H, W, nmax = 18, 100, 8, 5
    sup = Support(J, nmax)
    ch, cw = contexts(H, W)
    panel = np.random.default_rng(args.seed + 4).choice(len(ch), size=args.panel, replace=False)
    hp = torch.as_tensor(ch[panel])
    cw_p = cw[panel]

    if args.mode == "baseline":
        results = run_mode(args, sup, hp, cw_p, use_f_extra=False, presence_phi=False, cat_rho0=False)
        payload = {
            "panel_contexts": args.panel,
            "seed": args.seed,
            "is_burn": args.is_burn,
            "is_components": args.is_components,
            "is_draws": args.is_draws,
            "gate_E_thresholds": {"mean_abs_err_le": 0.01, "max_abs_err_le": 0.05},
            "results": results,
        }
    elif args.mode == "m1":
        results = run_mode(args, sup, hp, cw_p, use_f_extra=True, presence_phi=False, cat_rho0=False)
        payload = {"mode": "m1", "results": results}
    elif args.mode == "m2":
        results = run_mode(args, sup, hp, cw_p, use_f_extra=True, presence_phi=True, cat_rho0=False)
        payload = {"mode": "m2", "results": results}
    elif args.mode == "m3":
        results = run_mode(args, sup, hp, cw_p, use_f_extra=True, presence_phi=True, cat_rho0=True)
        payload = {"mode": "m3", "results": results}
    elif args.mode == "m4a":
        results = run_mode(args, sup, hp, cw_p, use_f_extra=True, presence_phi=True, cat_rho0=True, u_cat=True)
        payload = {"mode": "m4a", "results": results}
    elif args.mode == "m4b":
        dp_and_grad = check_m4b_dp_and_gradients_small_world(sup, hp, cw_p, seed=args.seed)
        print(json.dumps({"check": "dp_and_gradients_small_world", "status": "pass"}), flush=True)
        lam1_check = check_lambda_1_reproduces_hs_rec_256_trips()
        print(json.dumps({"check": "lambda_1_reproduces_hs_rec_256_trips", **lam1_check}), flush=True)
        results = run_mode(
            args,
            sup,
            hp,
            cw_p,
            use_f_extra=True,
            presence_phi=True,
            cat_rho0=True,
            u_cat=True,
            iv_slope=True,
        )
        payload = {
            "mode": "m4b",
            "panel_contexts": args.panel,
            "seed": args.seed,
            "is_burn": args.is_burn,
            "is_components": args.is_components,
            "is_draws": args.is_draws,
            "gate_E_thresholds": {"mean_abs_err_le": 0.01, "max_abs_err_le": 0.05},
            "small_world_dp_and_gradient_checks": dp_and_grad,
            "lambda_1_reproduces_hs_rec_256_trips": lam1_check,
            "results": results,
        }
    else:
        payload = {
            "panel_contexts": args.panel,
            "seed": args.seed,
            "is_burn": args.is_burn,
            "is_components": args.is_components,
            "is_draws": args.is_draws,
            "gate_E_thresholds": {"mean_abs_err_le": 0.01, "max_abs_err_le": 0.05},
            "baseline": run_mode(args, sup, hp, cw_p, use_f_extra=False, presence_phi=False, cat_rho0=False),
            "m1": run_mode(args, sup, hp, cw_p, use_f_extra=True, presence_phi=False, cat_rho0=False),
            "m2": run_mode(args, sup, hp, cw_p, use_f_extra=True, presence_phi=True, cat_rho0=False),
            "m3": run_mode(args, sup, hp, cw_p, use_f_extra=True, presence_phi=True, cat_rho0=True),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
