#!/usr/bin/env python3
"""Diagnose where Version-4 (M3 / M4a / M4b) loses to the Category Chain and to the
true stress-world generator across all 4,096 test trips:
  - the 1,887 test trips with at most one item per category (n_c in {0, 1}),
  - the 2,209 test trips with two or more items in at least one category (max_c n_c >= 2),
  - all 4,096 test trips combined.

Splits each model's (and the true generator's) per-trip log-likelihood into three exact parts:
  log P(S) = log P^+(x) + log P(n | x) + log P(S | n)
where:
  1. which categories are present: x_c = 1(n_c > 0)
  2. how many items in each present category: n_c given x_c = 1
  3. which products, given the counts: S_c given n_c

The three parts telescope to the saved per-trip log-likelihood totals to within < 1e-10
(machine precision, ~1e-14) on every single trip.

Outputs (written under --out-dir, e.g. runs/stress/hs_m4a_gap_full/):
  - pattern_estimator_check.json
  - per_trip.npz
  - truth_ceiling.json
  - summary.json
  - run.log
"""
from __future__ import annotations

import argparse
import json
import math
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
sys.path.insert(0, str(ROOT / "scripts/synthetic"))
sys.path.insert(0, str(ROOT.parent / "basket_chain"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.nn.functional import logsigmoid, softplus  # noqa: E402

import stress  # noqa: E402
from stress import (  # noqa: E402
    Batch,
    basket_field,
    configure_model_from_blob,
    gibbs,
    obs_energy,
    phi_r,
    row_cat_shift,
    sample_slots,
    setup,
    slot_utilities,
)
from ragged import esp_log_bucketed, seg_max  # noqa: E402
from hs import Partition, V4, _category_polys, is_log_z as hs_is_log_z, log_zc as hs_log_zc  # noqa: E402
from world import Support, draw_trips, energy_table, make_truth  # noqa: E402
from basket_chain.data import batch_context, batch_members, load_bundle  # noqa: E402
from scripts.evaluate_stress import load_model as load_chain_model  # noqa: E402
import generate_stress_world as gsw  # noqa: E402

torch.set_default_dtype(torch.float64)


# ==============================================================================
# 1. Small-world exact brute-force check of the log P(category pattern) estimator
# ==============================================================================
def log_zc_pattern_small(
    u: torch.Tensor,
    part: Partition,
    rho_c: torch.Tensor,
    rho0: torch.Tensor,
    nc_obs: torch.Tensor,
    f_extra: torch.Tensor | None = None,
    cat_shift: torch.Tensor | None = None,
    cat_rho0: bool = False,
) -> torch.Tensor:
    """Exact conditional pattern log-sum log Z_c(n_obs; z) given utilities u(z) on small worlds."""
    polys, _ = _category_polys(u, part, rho_c, f_extra=f_extra, cat_shift=cat_shift)
    B = u.shape[0]
    total = torch.zeros(B, dtype=u.dtype)
    for c, p in enumerate(polys):
        kc = nc_obs[:, c].long()
        total = total + p.gather(1, kc[:, None]).squeeze(1)
    size_idx = (nc_obs > 0).sum(1).long() if cat_rho0 else nc_obs.sum(1).long()
    return total - rho0[size_idx - 1]


@torch.no_grad()
def sample_given_u_and_pattern_small(
    u: torch.Tensor, part: Partition, nc_obs: torch.Tensor, gen: torch.Generator
) -> torch.Tensor:
    """Exact draw S ~ P(S | u(z), n(S) = nc_obs) on small worlds by within-category ESP backtracking."""
    B, J = u.shape
    u_max = u.amax(dim=1, keepdim=True).clamp(min=-20.0, max=20.0)
    w = torch.exp(u - u_max)
    X = torch.zeros(B, J, dtype=u.dtype)
    for c, items in enumerate(part.members):
        m = len(items)
        top_c = min(m, part.nmax)
        if m == 1:
            X[:, items[0]] = nc_obs[:, c].double()
            continue
        e = torch.zeros((B, top_c + 1), dtype=u.dtype)
        e[:, 0] = 1.0
        steps = [e.clone()]
        cur_d = 0
        for i in items:
            wi = w[:, i:i + 1]
            nd = min(cur_d + 1, top_c)
            e[:, 1:nd + 1] = e[:, 1:nd + 1] + e[:, :nd].clone() * wi
            cur_d = nd
            steps.append(e.clone())
        need = nc_obs[:, c].long().clone()
        for pos in range(len(items) - 1, -1, -1):
            i = int(items[pos])
            prev, cur = steps[pos], steps[pos + 1]
            have = need > 0
            nk = need.clamp(min=1)
            num = w[:, i] * prev.gather(1, (nk - 1).clamp(max=prev.shape[1] - 1)[:, None]).squeeze(1)
            den = cur.gather(1, nk.clamp(max=cur.shape[1] - 1)[:, None]).squeeze(1).clamp_min(1e-300)
            p_take = torch.where(have, (num / den).clamp(0.0, 1.0), torch.zeros(B, dtype=u.dtype))
            take = (torch.rand(B, generator=gen) < p_take) & have
            X[:, i] = take.double()
            need = need - take.long()
        if int((need != 0).sum()) != 0:
            raise RuntimeError("small-world pattern backtrack left slots unfilled")
    return X


@torch.no_grad()
def is_log_z_pattern_small(
    model: V4,
    h: torch.Tensor,
    price: torch.Tensor,
    nc_obs: torch.Tensor,
    gen: torch.Generator,
    burn: int = 200,
    components: int = 256,
    thin: int = 4,
    draws: int = 2048,
    defensive: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pattern-conditioned importance-sampling estimator of log Z(n_obs) on small worlds."""
    phi = model.eff_phi()
    B, r = h.shape[0], phi.shape[1]
    rho_c, rho0, f_ext = model.p["rho_c"], model.rho0(), model.f_extra()
    b0 = model.b(h, price)
    phi_sq = phi.square().sum(1)[None, :]
    u0 = b0 - 0.5 * phi_sq

    _, sv, Vt = torch.linalg.svd(phi, full_matrices=False)
    seed_u = [b0, u0]
    for d in range(r):
        if float(sv[d]) > 0.5:
            for mult in (1.5, 3.0):
                z_dir = (mult * sv[d]) * Vt[d]
                seed_u.append(u0 + (z_dir @ phi.T)[None, :])
                seed_u.append(u0 - (z_dir @ phi.T)[None, :])
    M = len(seed_u)
    nc_M = nc_obs.repeat_interleave(M, dim=0)
    X_seeds = [sample_given_u_and_pattern_small(seed_u[m_i], model.part, nc_obs, gen) for m_i in range(M)]
    X_flat = torch.stack(X_seeds, dim=1).reshape(B * M, -1)
    u0_M = u0.repeat_interleave(M, dim=0)

    burn_per_chain = max(burn // 4, 25)
    for _ in range(burn_per_chain):
        z_b = model.basket_s(X_flat) + torch.randn(B * M, r, generator=gen)
        X_flat = sample_given_u_and_pattern_small(u0_M + z_b @ phi.T, model.part, nc_M, gen)

    k_per_chain = max((components + M - 1) // M, 4)
    centres = []
    for k in range(k_per_chain * thin):
        z_b = model.basket_s(X_flat) + torch.randn(B * M, r, generator=gen)
        X_flat = sample_given_u_and_pattern_small(u0_M + z_b @ phi.T, model.part, nc_M, gen)
        if (k + 1) % thin == 0:
            centres.append(model.basket_s(X_flat).view(B, M, r))
    centres = torch.cat(centres, dim=1)[:, :max(components, M * 4), :]
    K = centres.shape[1]

    diff_c = (centres[:, :, None, :] - centres[:, None, :, :]).square().sum(3)
    same_c = (diff_c < 1e-8).double()
    mult = same_c.sum(2).clamp_min(1.0)
    c_flat = centres.reshape(B * K, r)
    u_centres = (u0[:, None, :] + centres @ phi.T).reshape(B * K, -1)
    nc_K = nc_obs.repeat_interleave(K, dim=0)
    lzc_c = log_zc_pattern_small(
        u_centres, model.part, rho_c, rho0, nc_K,
        f_extra=f_ext, cat_shift=model.cat_shift(c_flat), cat_rho0=model.cat_rho0
    ).view(B, K)
    lp_c = -0.5 * centres.square().sum(2)
    log_v = lzc_c + lp_c - torch.log(mult)
    w_mode = torch.softmax(log_v, dim=1)
    w_comp = 0.70 * w_mode + 0.30 / float(K)
    log_w_comp = torch.log(w_comp.clamp_min(1e-300))

    probs_all = torch.cat([(1.0 - defensive) * w_comp, torch.full((B, 1), defensive, dtype=phi.dtype)], dim=1)
    cdf = probs_all.cumsum(dim=1)
    cdf[:, -1] = 1.0
    u_strat = (torch.arange(draws, dtype=phi.dtype)[None, :] + torch.rand(B, draws, generator=gen)) / float(draws)
    perm = torch.argsort(torch.rand(B, draws, generator=gen), dim=1)
    u_strat = u_strat.gather(1, perm)
    picks_all = torch.searchsorted(cdf, u_strat).clamp(max=K)
    centres_ext = torch.cat([centres, torch.zeros(B, 1, r, dtype=phi.dtype)], dim=1)

    half_log2pi = 0.5 * r * math.log(2 * math.pi)
    log_w = []
    batch_d = 64
    for start in range(0, draws, batch_d):
        d_cur = min(batch_d, draws - start)
        pick = picks_all[:, start:start + d_cur]
        mean = centres_ext.gather(1, pick[:, :, None].expand(B, d_cur, r))
        z = mean + torch.randn(B, d_cur, r, generator=gen)
        log_prior = -0.5 * z.square().sum(2) - half_log2pi
        comp = -0.5 * (z[:, :, None, :] - centres[:, None, :, :]).square().sum(3) - half_log2pi + log_w_comp[:, None, :]
        log_q = torch.logaddexp(
            math.log1p(-defensive) + torch.logsumexp(comp, 2),
            math.log(defensive) + log_prior,
        )
        z_flat = z.reshape(B * d_cur, r)
        u_flat = (u0[:, None, :] + z @ phi.T).reshape(B * d_cur, -1)
        nc_d = nc_obs.repeat_interleave(d_cur, dim=0)
        lzc = log_zc_pattern_small(
            u_flat, model.part, rho_c, rho0, nc_d,
            f_extra=f_ext, cat_shift=model.cat_shift(z_flat), cat_rho0=model.cat_rho0
        ).view(B, d_cur)
        log_w.append(log_prior + lzc - log_q)
    log_w = torch.cat(log_w, 1)
    est = torch.logsumexp(log_w, 1) - math.log(draws)
    w = torch.softmax(log_w, 1)
    ess = 1.0 / w.square().sum(1) / draws
    return est, ess


def run_small_world_check(
    seed: int = 101,
    panel_trips: int = 200,
    strengths: tuple[float, ...] = (1.0, 3.0, 7.0, 25.0, 50.0, 75.0),
    is_burn: int = 200,
    is_components: int = 256,
    is_thin: int = 4,
    is_draws: int = 2048,
    emit=print,
) -> dict:
    """Validate the log P(category pattern) estimator against brute-force enumeration on world.py."""
    J, H, W, K, r, nmax = 18, 100, 8, 3, 3, 5
    sup = Support(J, nmax)
    modes = {
        "baseline": dict(use_f_extra=False, presence_phi=False, cat_rho0=False),
        "m3": dict(use_f_extra=True, presence_phi=True, cat_rho0=True),
    }
    out_modes = {}
    phi0_dp_checks = {}

    for mode_name, kw in modes.items():
        rows = []
        for strength in strengths:
            truth, week_price = make_truth(seed, strength, J=J, H=H, W=W, K=K, r=r, nmax=nmax, **kw)
            rng = np.random.default_rng(seed + 101)
            trips = draw_trips(truth, sup, week_price, H, W, panel_trips, rng)
            hp = torch.as_tensor(trips["h"])
            pp = week_price[torch.as_tensor(trips["w"])]
            X_obs = sup.M[torch.as_tensor(trips["basket"])]
            nc_obs = (X_obs @ truth.part.onehot).long()
            nc_all = (sup.M @ truth.part.onehot).long()
            match = (nc_all[None, :, :] == nc_obs[:, None, :]).all(dim=2)

            if strength == strengths[0]:
                t_zero, _ = make_truth(seed, strength, J=J, H=H, W=W, K=K, r=r, nmax=nmax, **kw)
                with torch.no_grad():
                    t_zero.p["phi"].zero_()
                    if kw["presence_phi"]:
                        t_zero.p["psi"].zero_()
                        t_zero.p["delta_phi"].zero_()
                    E0_all = energy_table(t_zero, sup, hp, pp)
                    ex0_z = torch.logsumexp(E0_all, dim=1)
                    ex0_zn = torch.logsumexp(torch.where(match, E0_all, torch.full_like(E0_all, -1e30)), dim=1)
                    dp0_z = hs_log_zc(
                        t_zero.b(hp, pp), t_zero.part, t_zero.p["rho_c"], t_zero.rho0(),
                        f_extra=t_zero.f_extra(), cat_shift=None, cat_rho0=kw["cat_rho0"]
                    )
                    dp0_zn = log_zc_pattern_small(
                        t_zero.b(hp, pp), t_zero.part, t_zero.p["rho_c"], t_zero.rho0(), nc_obs,
                        f_extra=t_zero.f_extra(), cat_shift=None, cat_rho0=kw["cat_rho0"]
                    )
                    err0_z = float((ex0_z - dp0_z).abs().max())
                    err0_zn = float((ex0_zn - dp0_zn).abs().max())
                    err0_pn = float(((ex0_zn - ex0_z) - (dp0_zn - dp0_z)).abs().max())
                    phi0_dp_checks[mode_name] = {
                        "max_abs_err_log_z": err0_z,
                        "max_abs_err_log_z_pattern": err0_zn,
                        "max_abs_err_log_p_pattern": err0_pn,
                    }

            with torch.no_grad():
                E_all = energy_table(truth, sup, hp, pp)
                log_z_ex = torch.logsumexp(E_all, dim=1)
                log_zn_ex = torch.logsumexp(torch.where(match, E_all, torch.full_like(E_all, -1e30)), dim=1)
                e_obs = truth.energy(X_obs, hp, pp)
                log_p_tot_ex = e_obs - log_z_ex
                log_p_pat_ex = log_zn_ex - log_z_ex
                log_p_prod_ex = e_obs - log_zn_ex

                est_z, ess_z = hs_is_log_z(
                    truth, hp, pp, torch.Generator().manual_seed(seed + 12),
                    burn=is_burn, components=is_components, thin=is_thin, draws=is_draws
                )
                est_zn, ess_zn = is_log_z_pattern_small(
                    truth, hp, pp, nc_obs, torch.Generator().manual_seed(seed + 13),
                    burn=is_burn, components=is_components, thin=is_thin, draws=is_draws
                )
                log_p_tot_est = e_obs - est_z
                log_p_pat_est = est_zn - est_z
                log_p_prod_est = e_obs - est_zn

                err_z = (est_z - log_z_ex).abs()
                err_zn = (est_zn - log_zn_ex).abs()
                err_pat = (log_p_pat_est - log_p_pat_ex).abs()
                err_prod = (log_p_prod_est - log_p_prod_ex).abs()
                sum_mismatch = (log_p_pat_est + log_p_prod_est - log_p_tot_est).abs()

            row = {
                "strength": float(strength),
                "log_z_pattern": {
                    "mean_abs_err": round(float(err_zn.mean()), 5),
                    "max_abs_err": round(float(err_zn.max()), 5),
                    "mean_ess": round(float(ess_zn.mean()), 4),
                    "min_ess": round(float(ess_zn.min()), 4),
                    "pass_gate_E": bool(err_zn.mean() <= 0.01 and err_zn.max() <= 0.05),
                },
                "log_p_pattern": {
                    "mean_exact": round(float(log_p_pat_ex.mean()), 4),
                    "mean_est": round(float(log_p_pat_est.mean()), 4),
                    "mean_abs_err": round(float(err_pat.mean()), 5),
                    "max_abs_err": round(float(err_pat.max()), 5),
                    "pass_gate_E": bool(err_pat.mean() <= 0.01 and err_pat.max() <= 0.05),
                },
                "log_p_products_given_pattern": {
                    "mean_exact": round(float(log_p_prod_ex.mean()), 4),
                    "mean_est": round(float(log_p_prod_est.mean()), 4),
                    "mean_abs_err": round(float(err_prod.mean()), 5),
                    "max_abs_err": round(float(err_prod.max()), 5),
                    "pass_gate_E": bool(err_prod.mean() <= 0.01 and err_prod.max() <= 0.05),
                },
                "log_z_total": {
                    "mean_abs_err": round(float(err_z.mean()), 5),
                    "max_abs_err": round(float(err_z.max()), 5),
                    "mean_ess": round(float(ess_z.mean()), 4),
                    "min_ess": round(float(ess_z.min()), 4),
                },
                "max_additivity_mismatch": float(sum_mismatch.max()),
            }
            emit(json.dumps({"event": "small_world_check", "mode": mode_name, **row}))
            rows.append(row)
        out_modes[mode_name] = rows

    return {
        "panel_trips": panel_trips,
        "seed": seed,
        "is_burn": is_burn,
        "is_components": is_components,
        "is_thin": is_thin,
        "is_draws": is_draws,
        "gate_E_thresholds": {"mean_abs_err_le": 0.01, "max_abs_err_le": 0.05},
        "phi0_dp_vs_enumeration": phi0_dp_checks,
        "modes": out_modes,
    }


# ==============================================================================
# 2. General stress-world 3-part estimator & per-category attributions (all 4,096 trips)
# ==============================================================================
@torch.no_grad()
def sample_active_rows_given_counts(
    u_act: torch.Tensor,
    slot_act_row: torch.Tensor,
    nc_act: torch.Tensor,
    row_pos_act: torch.Tensor,
    gen: torch.Generator,
) -> torch.Tensor:
    """Exact draw S_{tc} ~ P(S_{tc} | u(z), |S_{tc}| = nc_act[r]) for all active rows r.
    Uses fast Gumbel-max on rows with nc_act == 1 and exact ESP backtracking on rows with nc_act >= 2.
    """
    n_act = len(nc_act)
    n_slots = len(u_act)
    picked = torch.zeros(n_slots, dtype=torch.bool)

    is_one_row = nc_act == 1
    one_slot_mask = is_one_row[slot_act_row]
    if bool(one_slot_mask.any()):
        one_slots = torch.nonzero(one_slot_mask, as_tuple=True)[0]
        r_one = slot_act_row[one_slots]
        u_one = u_act[one_slots]
        m_r = torch.full((n_act,), -float("inf"), dtype=u_act.dtype)
        m_r.index_reduce_(0, r_one, u_one, "amax", include_self=True)
        gumbel = -torch.log(-torch.log(torch.rand(len(one_slots), generator=gen).clamp(1e-15, 1.0 - 1e-15)))
        score = (u_one - m_r[r_one]) + gumbel
        max_sc = torch.full((n_act,), -float("inf"), dtype=score.dtype)
        max_sc.index_reduce_(0, r_one, score, "amax", include_self=True)
        picked[one_slots] = score == max_sc[r_one]

    multi_rows = torch.nonzero(nc_act >= 2, as_tuple=True)[0]
    if len(multi_rows) > 0:
        M = len(multi_rows)
        multi_remap = torch.full((n_act,), -1, dtype=torch.long)
        multi_remap[multi_rows] = torch.arange(M)
        multi_slot_mask = (nc_act >= 2)[slot_act_row]
        multi_slots = torch.nonzero(multi_slot_mask, as_tuple=True)[0]
        m_row = multi_remap[slot_act_row[multi_slots]]
        m_pos = row_pos_act[multi_slots]
        u_m = u_act[multi_slots]

        L_max = int(m_pos.max()) + 1
        K_max = int(nc_act[multi_rows].max())
        u_dense = torch.full((M, L_max), -float("inf"), dtype=u_act.dtype)
        slot_idx_dense = torch.full((M, L_max), -1, dtype=torch.long)
        u_dense[m_row, m_pos] = u_m
        slot_idx_dense[m_row, m_pos] = multi_slots

        u_max = u_dense.amax(dim=1, keepdim=True).clamp(min=-30.0, max=30.0)
        w_dense = torch.where(u_dense > -1e20, torch.exp(u_dense - u_max), torch.zeros_like(u_dense))

        e = torch.zeros((M, K_max + 1), dtype=u_act.dtype)
        e[:, 0] = 1.0
        steps = [e.clone()]
        for pos in range(L_max):
            wi = w_dense[:, pos:pos + 1]
            nd = min(pos + 1, K_max)
            e[:, 1:nd + 1] = e[:, 1:nd + 1] + e[:, :nd].clone() * wi
            steps.append(e.clone())

        need = nc_act[multi_rows].long().clone()
        for pos in range(L_max - 1, -1, -1):
            prev, cur = steps[pos], steps[pos + 1]
            wi = w_dense[:, pos]
            have = (need > 0) & (wi > 0)
            nk = need.clamp(min=1, max=K_max)
            num = wi * prev.gather(1, (nk - 1)[:, None]).squeeze(1)
            den = cur.gather(1, nk[:, None]).squeeze(1).clamp_min(1e-300)
            p_take = torch.where(have, (num / den).clamp(0.0, 1.0), torch.zeros(M, dtype=u_act.dtype))
            take = (torch.rand(M, generator=gen) < p_take) & have
            valid_slots = slot_idx_dense[take, pos]
            picked[valid_slots] = True
            need = need - take.long()
        if int((need != 0).sum()) != 0:
            raise RuntimeError(f"ESP backtrack left {int((need != 0).sum())} rows unfilled")

    return picked


def _row_iv_shift(model, ix, b_slot: torch.Tensor) -> torch.Tensor | None:
    """Optional M4b category inclusive-value shift (lambda_c - 1) * IV_{tc}(b) per row."""
    if hasattr(stress, "row_iv_shift"):
        return stress.row_iv_shift(model, ix, b_slot=b_slot)
    if not getattr(model, "has_iv_slope", False) or getattr(model, "iv_slope_raw", None) is None:
        return None
    m_r = torch.full((ix.n_rows,), -float("inf"), dtype=b_slot.dtype, device=b_slot.device)
    m_r.index_reduce_(0, ix.row_of, b_slot, "amax", include_self=True)
    w_s = torch.exp(b_slot - m_r[ix.row_of])
    sum_r = torch.zeros(ix.n_rows, dtype=b_slot.dtype, device=b_slot.device).index_add_(0, ix.row_of, w_s)
    iv_r = m_r + torch.log(sum_r.clamp_min(1e-300))
    lam_c = torch.sigmoid(model.iv_slope_raw)
    return (lam_c[ix.row_cat] - 1.0) * iv_r


@torch.no_grad()
def estimate_v4_three_part_batch(
    model,
    bt: Batch,
    r: int,
    gen: torch.Generator,
    draws: int = 1024,
    components: int = 64,
    burn: int = 50,
    defensive: float = 0.05,
):
    """Estimate log Z(x_obs), log Z(n_obs), and per-category decompositions for a batch of
    stress trips (supporting arbitrary category counts n_c in {0, 1, ..., R})."""
    ix = bt.ix
    B = ix.B
    C = model.C
    R = model.R
    nmax = model.nmax
    ph_full = phi_r(model, r).detach()
    ph_slot = ph_full[ix.item]
    b_slot = model.b_flat(ix).detach()
    u0_slot = b_slot - 0.5 * ph_slot.square().sum(-1)
    iv_shift_row = _row_iv_shift(model, ix, b_slot)
    if iv_shift_row is not None:
        iv_shift_row = iv_shift_row.detach()

    nc_row = torch.zeros(ix.n_rows, dtype=torch.long).index_add_(0, ix.row_of, bt.obs.long())
    active_row_mask = nc_row >= 1
    act_rows = torch.nonzero(active_row_mask, as_tuple=True)[0]
    n_act = len(act_rows)
    row_remap = torch.full((ix.n_rows,), -1, dtype=torch.long)
    row_remap[act_rows] = torch.arange(n_act)
    act_row_trip = ix.row_trip[act_rows]
    act_row_cat = ix.row_cat[act_rows]
    nc_act = nc_row[act_rows].clamp(max=R)
    r_size_act = ix.row_size[act_rows].clamp(max=R)

    act_slots = torch.nonzero(active_row_mask[ix.row_of], as_tuple=True)[0]
    slot_act_row = row_remap[ix.row_of[act_slots]]
    slot_trip = ix.item_trip[act_slots]
    row_pos_act = ix.item_pos[act_slots]
    u0_act = u0_slot[act_slots]
    b_act = b_slot[act_slots]
    ph_act = ph_slot[act_slots]
    obs_act = bt.obs[act_slots]

    k_b = torch.zeros(B, dtype=torch.long).index_add_(0, act_row_trip, torch.ones_like(act_row_trip))
    rho0 = model.rho_0().detach()
    rho0_k = rho0[k_b]

    has_m2 = getattr(model, "presence_phi", False)
    if has_m2:
        extra_row = (nc_row.double() - 1.0).clamp_min(0.0)
        m2_field_shift = torch.zeros(B, r, dtype=ph_act.dtype).index_add_(
            0, ix.row_trip, model.psi_cat[ix.row_cat].detach() * extra_row[:, None]
        )
    else:
        m2_field_shift = torch.zeros(B, r, dtype=ph_act.dtype)

    # Count-conditioned HS block-Gibbs chain to collect proposal centres s(S') with pattern n_obs
    s_cur = basket_field(model, ix, bt.obs, r)
    centres = [s_cur]
    for step in range(burn + components - 1):
        z_step = s_cur + torch.randn(B, r, generator=gen)
        u_step = u0_act + (z_step[slot_trip] * ph_act).sum(-1)
        picked = sample_active_rows_given_counts(u_step, slot_act_row, nc_act, row_pos_act, gen)
        s_items = torch.zeros((B, r), dtype=ph_act.dtype).index_add_(0, slot_trip[picked], ph_act[picked])
        s_cur = s_items - m2_field_shift
        if step >= burn - 1:
            centres.append(s_cur)
    centres = torch.stack(centres, dim=1)  # [B, Kc, r]
    Kc = centres.shape[1]

    # Stratified mixture proposal over Kc centres + defensive prior N(0, I)
    w_cent = (1.0 - defensive) / float(Kc)
    probs = torch.tensor([w_cent] * Kc + [defensive], dtype=centres.dtype)
    cdf = probs.cumsum(0)
    cdf[-1] = 1.0
    h = draws // 2
    h2 = draws - h
    u_a = (torch.arange(h, dtype=centres.dtype)[None, :] + torch.rand(B, h, generator=gen)) / float(h)
    u_a = u_a.gather(1, torch.argsort(torch.rand(B, h, generator=gen), dim=1))
    u_b = (torch.arange(h2, dtype=centres.dtype)[None, :] + torch.rand(B, h2, generator=gen)) / float(h2)
    u_b = u_b.gather(1, torch.argsort(torch.rand(B, h2, generator=gen), dim=1))
    picks_all = torch.cat([torch.searchsorted(cdf, u_a), torch.searchsorted(cdf, u_b)], dim=1).clamp(max=Kc)

    prior_all = picks_all == Kc
    c_pick = centres.gather(1, picks_all.clamp(max=Kc - 1)[:, :, None].expand(B, draws, r))
    mean_all = torch.where(prior_all[:, :, None], torch.zeros(B, draws, r), c_pick)
    z_all = mean_all + torch.randn(B, draws, r, generator=gen)

    half = 0.5 * r * math.log(2 * math.pi)
    lp_all = -0.5 * z_all.square().sum(2) - half
    c_norm2 = -0.5 * centres.square().sum(2)
    comp_all = (
        -0.5 * z_all.square().sum(2)[:, :, None]
        + c_norm2[:, None, :]
        + torch.bmm(z_all, centres.transpose(1, 2))
        - half
    )
    lq_all = torch.logaddexp(
        math.log1p(-defensive) + torch.logsumexp(comp_all, 2) - math.log(Kc),
        math.log(defensive) + lp_all,
    )
    base_lw = lp_all - lq_all  # [B, draws]

    # Category count penalties on active rows
    deg_ax = torch.arange(R + 1, dtype=u0_slot.dtype)
    fe_act = model.f_extra[act_row_cat].detach()
    fe_pad_act = torch.zeros((n_act, R + 1), dtype=u0_slot.dtype)
    fe_pad_act[:, 2] = fe_act[:, 0]
    fe_pad_act[:, 3] = fe_act[:, 1]
    fe_pad_act[:, 4:] = fe_act[:, 2:3]
    base_pen_act = model.rho_c[act_row_cat, None].detach() * model.pair_feature(deg_ax)[None, :] + fe_pad_act
    if iv_shift_row is not None:
        iv_shift_act = iv_shift_row[act_rows]
        base_pen_act[:, 1:] = base_pen_act[:, 1:] - iv_shift_act[:, None]
    valid_deg_act = (deg_ax[None, :] >= 1) & (deg_ax[None, :] <= r_size_act[:, None])

    # Dense slot layout on active rows [n_act, L_max]
    L_max = int(row_pos_act.max()) + 1
    u0_dense = torch.full((n_act, L_max), -float("inf"), dtype=u0_slot.dtype)
    u0_dense[slot_act_row, row_pos_act] = u0_act
    b_dense = torch.full((n_act, L_max), -float("inf"), dtype=b_slot.dtype)
    b_dense[slot_act_row, row_pos_act] = b_act
    ph_dense = torch.zeros((n_act, L_max, r), dtype=u0_slot.dtype)
    ph_dense[slot_act_row, row_pos_act] = ph_act
    valid_slot_dense = u0_dense > -1e20

    psi_act = model.psi_cat[act_row_cat].detach() if has_m2 else None
    psi_sq_act = -0.5 * psi_act.square().sum(-1) if has_m2 else None

    log_g_obs_rd = torch.empty((n_act, draws), dtype=u0_slot.dtype)
    log_iv_rd = torch.empty((n_act, draws), dtype=u0_slot.dtype)
    log_p_prod_rd = torch.empty((n_act, draws), dtype=u0_slot.dtype)

    u_sd_obs = u0_act[obs_act, None] + torch.einsum("tr,tdr->td", ph_act[obs_act], z_all[slot_trip[obs_act]])
    u_chosen_rd = torch.zeros((n_act, draws), dtype=u0_slot.dtype).index_add_(0, slot_act_row[obs_act], u_sd_obs)

    chunk_d = 128
    for d0 in range(0, draws, chunk_d):
        d1 = min(draws, d0 + chunk_d)
        dc = d1 - d0
        z_chunk = z_all[:, d0:d1, :]
        z_row = z_chunk[act_row_trip]
        u_chunk = u0_dense[:, :, None] + torch.einsum("nlr,ndr->nld", ph_dense, z_row)
        u_chunk = torch.where(valid_slot_dense[:, :, None], u_chunk, torch.full_like(u_chunk, -float("inf")))
        m_rc = u_chunk.amax(dim=1)
        w_chunk = torch.where(
            valid_slot_dense[:, :, None],
            torch.exp(u_chunk - m_rc[:, None, :]),
            torch.zeros_like(u_chunk),
        )
        e_tab = torch.zeros((n_act, dc, R + 1), dtype=u0_slot.dtype)
        e_tab[:, :, 0] = 1.0
        for pos in range(L_max):
            wi = w_chunk[:, pos, :, None]
            nd = min(pos + 1, R)
            e_tab[:, :, 1:nd + 1] = e_tab[:, :, 1:nd + 1] + e_tab[:, :, :nd].clone() * wi
        log_e_rc = torch.log(e_tab.clamp_min(1e-300)) + deg_ax[None, None, :] * m_rc[:, :, None]
        log_g_rc = log_e_rc - base_pen_act[:, None, :]
        if has_m2:
            cshift_rc = psi_sq_act[:, None] + torch.einsum("nr,ndr->nd", psi_act, z_row)
            log_g_rc = log_g_rc - cshift_rc[:, :, None] * (deg_ax - 1.0).clamp_min(0.0)[None, None, :]
        log_g_pos = torch.where(valid_deg_act[:, None, :], log_g_rc, torch.full_like(log_g_rc, -1e30))
        log_iv_rd[:, d0:d1] = torch.logsumexp(log_g_pos[:, :, 1:], dim=2)
        log_g_obs_rd[:, d0:d1] = log_g_rc.gather(2, nc_act[:, None, None].expand(n_act, dc, 1)).squeeze(2)
        log_e_obs_rc = log_e_rc.gather(2, nc_act[:, None, None].expand(n_act, dc, 1)).squeeze(2)
        log_p_prod_rd[:, d0:d1] = u_chosen_rd[:, d0:d1] - log_e_obs_rc

    # Additive-only (phi = 0) within-category product log-probability log P_b(S_{tc} | n_{tc})
    m_b = b_dense.amax(dim=1)
    w_b = torch.where(valid_slot_dense, torch.exp(b_dense - m_b[:, None]), torch.zeros_like(b_dense))
    e_b = torch.zeros((n_act, R + 1), dtype=b_slot.dtype)
    e_b[:, 0] = 1.0
    for pos in range(L_max):
        wi = w_b[:, pos:pos + 1]
        nd = min(pos + 1, R)
        e_b[:, 1:nd + 1] = e_b[:, 1:nd + 1] + e_b[:, :nd].clone() * wi
    log_e_b_obs = (torch.log(e_b.clamp_min(1e-300)) + deg_ax[None, :] * m_b[:, None]).gather(
        1, nc_act[:, None]
    ).squeeze(1)
    b_chosen = torch.zeros(n_act, dtype=b_slot.dtype).index_add_(0, slot_act_row[obs_act], b_act[obs_act])
    prod_b_only_bc = torch.zeros((B, C), dtype=b_slot.dtype)
    prod_b_only_bc[act_row_trip, act_row_cat] = b_chosen - log_e_b_obs

    # Joint log Z_c(n_obs; z) and log Z_c(x_obs; z)
    lzc_n = torch.zeros((B, draws), dtype=u0_slot.dtype).index_add_(0, act_row_trip, log_g_obs_rd) - rho0_k[:, None]
    lzc_x = torch.zeros((B, draws), dtype=u0_slot.dtype).index_add_(0, act_row_trip, log_iv_rd) - rho0_k[:, None]

    lw_n = base_lw + lzc_n
    lw_x = base_lw + lzc_x

    est_zn = torch.logsumexp(lw_n, dim=1) - math.log(draws)
    est_zx = torch.logsumexp(lw_x, dim=1) - math.log(draws)

    est_zn_a = torch.logsumexp(lw_n[:, :h], dim=1) - math.log(h)
    est_zn_b = torch.logsumexp(lw_n[:, h:], dim=1) - math.log(h2)
    half_diff_zn = est_zn_a - est_zn_b
    w_post_n = torch.softmax(lw_n, dim=1)
    ess_zn = 1.0 / w_post_n.square().sum(1) / draws

    est_zx_a = torch.logsumexp(lw_x[:, :h], dim=1) - math.log(h)
    est_zx_b = torch.logsumexp(lw_x[:, h:], dim=1) - math.log(h2)
    half_diff_zx = est_zx_a - est_zx_b
    w_post_x = torch.softmax(lw_x, dim=1)
    ess_zx = 1.0 / w_post_x.square().sum(1) / draws

    # Exact sequential count-given-presence telescope across c in A_t (sums identically to est_zn - est_zx)
    cnt_seq_bc = torch.zeros((B, C), dtype=u0_slot.dtype)
    cur_lw_cnt = lw_x.clone()
    cur_est_cnt = est_zx.clone()
    diff_cnt_rd = log_g_obs_rd - log_iv_rd
    for c_id in range(C):
        mask_c = act_row_cat == c_id
        if not bool(mask_c.any()):
            continue
        rows_c = torch.nonzero(mask_c, as_tuple=True)[0]
        trips_c = act_row_trip[rows_c]
        cur_lw_cnt[trips_c] = cur_lw_cnt[trips_c] + diff_cnt_rd[rows_c]
        new_est = torch.logsumexp(cur_lw_cnt[trips_c], dim=1) - math.log(draws)
        cnt_seq_bc[trips_c, c_id] = new_est - cur_est_cnt[trips_c]
        cur_est_cnt[trips_c] = new_est

    # Per-category marginal product choice log-probability: log E_{z | n_obs}[ P(S_c | n_c, z) ]
    w_row_n = w_post_n[act_row_trip]
    log_marg_prod_c = torch.logsumexp(torch.log(w_row_n.clamp_min(1e-300)) + log_p_prod_rd, dim=1)
    prod_marg_bc = torch.zeros((B, C), dtype=u0_slot.dtype)
    prod_marg_bc[act_row_trip, act_row_cat] = log_marg_prod_c

    # Exact sequential product choice telescope across c in A_t (sums identically to E(S) - est_zn)
    prod_seq_bc = torch.zeros((B, C), dtype=u0_slot.dtype)
    cur_lw_prod = lw_n.clone()
    cur_est_prod = est_zn.clone()
    for c_id in range(C):
        mask_c = act_row_cat == c_id
        if not bool(mask_c.any()):
            continue
        rows_c = torch.nonzero(mask_c, as_tuple=True)[0]
        trips_c = act_row_trip[rows_c]
        cur_lw_prod[trips_c] = cur_lw_prod[trips_c] + log_p_prod_rd[rows_c]
        new_est = torch.logsumexp(cur_lw_prod[trips_c], dim=1) - math.log(draws)
        prod_seq_bc[trips_c, c_id] = new_est - cur_est_prod[trips_c]
        cur_est_prod[trips_c] = new_est

    # Leave-one-category-out (LOO) pattern & presence conditionals log P_V4(n_{t,c} | n_{t,-c})
    # and log P_V4(x_{t,c} | n_{t,-c}) for all C categories on D_loo = 128 stratified draws
    D_loo = min(128, draws)
    idx_loo = torch.cat([torch.arange(D_loo // 2), torch.arange(h, h + D_loo // 2)])
    z_loo = z_all[:, idx_loo, :]
    base_w_loo = (lp_all[:, idx_loo] - lq_all[:, idx_loo])[:, None, :]

    fe_all = model.f_extra[ix.row_cat].detach()
    fe_pad_all = torch.zeros((ix.n_rows, R + 1), dtype=u0_slot.dtype)
    fe_pad_all[:, 2] = fe_all[:, 0]
    fe_pad_all[:, 3] = fe_all[:, 1]
    fe_pad_all[:, 4:] = fe_all[:, 2:3]
    base_pen_all = model.rho_c[ix.row_cat, None].detach() * model.pair_feature(deg_ax)[None, :] + fe_pad_all
    if iv_shift_row is not None:
        base_pen_all[:, 1:] = base_pen_all[:, 1:] - iv_shift_row[:, None]

    nc_bc = torch.zeros((B, C), dtype=torch.long)
    nc_bc[ix.row_trip, ix.row_cat] = nc_row.clamp(max=R)
    x_bc = (nc_bc > 0).long()
    k_other_bc = k_b[:, None] - x_bc
    n_items_b = nc_bc.sum(dim=1)
    n_other_bc = n_items_b[:, None] - nc_bc
    max_k_row = (nmax - n_other_bc[ix.row_trip, ix.row_cat]).clamp(min=1, max=R)
    max_k_row = torch.minimum(max_k_row, ix.row_size.clamp(max=R))

    log_g_obs_bcd = torch.full((B, C, D_loo), -1e30, dtype=u0_slot.dtype)
    log_iv_bcd = torch.full((B, C, D_loo), -1e30, dtype=u0_slot.dtype)
    nc_row_clamped = nc_row.clamp(min=1, max=R)

    for d_i in range(D_loo):
        z_d = z_loo[:, d_i]
        u_d = u0_slot + (z_d[ix.item_trip] * ph_slot).sum(-1)
        scale = seg_max(u_d.unsqueeze(0), ix.item_trip, B)[0]
        centred = u_d - scale[ix.item_trip]
        log_e = esp_log_bucketed(centred.unsqueeze(0), ix.row_of, ix.n_rows, R, ix.row_size, ix.item_pos)[0]
        log_g = log_e + deg_ax[None, :] * scale[ix.row_trip, None] - base_pen_all
        cshift = row_cat_shift(model, ix, z_d, beta=1.0)
        if cshift is not None:
            log_g = log_g - cshift[:, None] * (deg_ax - 1.0).clamp_min(0.0)[None, :]
        valid_k = (deg_ax[None, :] >= 1) & (deg_ax[None, :] <= max_k_row[:, None])
        log_g_pos = torch.where(valid_k, log_g, torch.full_like(log_g, -1e30))
        iv_r = torch.logsumexp(log_g_pos[:, 1:], dim=-1)
        g_obs_r = log_g.gather(1, nc_row_clamped[:, None]).squeeze(1)
        log_g_obs_bcd[ix.row_trip, ix.row_cat, d_i] = g_obs_r
        log_iv_bcd[ix.row_trip, ix.row_cat, d_i] = iv_r

    x_bool = x_bc == 1
    L_bd = (log_g_obs_bcd * x_bool[:, :, None].double()).sum(dim=1)
    L_other_bcd = L_bd[:, None, :] - log_g_obs_bcd * x_bool[:, :, None].double()

    rho0_0 = rho0[k_other_bc.clamp(min=1)]
    lzc_0 = torch.where(
        k_other_bc[:, :, None] >= 1,
        L_other_bcd - rho0_0[:, :, None],
        torch.full_like(L_other_bcd, -1e30),
    )
    rho0_1 = rho0[(k_other_bc + 1).clamp(max=nmax)]
    lzc_obs = L_other_bcd + log_g_obs_bcd - rho0_1[:, :, None]
    lzc_pos = L_other_bcd + log_iv_bcd - rho0_1[:, :, None]

    log_Z0 = torch.logsumexp(base_w_loo + lzc_0, dim=2) - math.log(D_loo)
    log_Zobs_pos = torch.logsumexp(base_w_loo + lzc_obs, dim=2) - math.log(D_loo)
    log_Zpos = torch.logsumexp(base_w_loo + lzc_pos, dim=2) - math.log(D_loo)
    log_Ztot_c = torch.logaddexp(log_Z0, log_Zpos)
    log_Zobs_c = torch.where(x_bool, log_Zobs_pos, log_Z0)
    log_Zpres_obs_c = torch.where(x_bool, log_Zpos, log_Z0)

    loo_pat_bc = log_Zobs_c - log_Ztot_c
    loo_pres_bc = log_Zpres_obs_c - log_Ztot_c
    cnt_loo_bc = x_bool.double() * (log_Zobs_pos - log_Zpos)

    return (
        est_zx,
        est_zn,
        half_diff_zx,
        half_diff_zn,
        ess_zx,
        ess_zn,
        cnt_seq_bc,
        cnt_loo_bc,
        prod_seq_bc,
        prod_marg_bc,
        prod_b_only_bc,
        loo_pat_bc,
        loo_pres_bc,
    )


@torch.no_grad()
def compute_unconditional_proposal_ess_diagnostic(model, bt: Batch, r: int, gen: torch.Generator, draws: int = 512):
    """Diagnostic showing why estimating log Z(n_obs) from the unconditional Gibbs chain
    proposal has low ESS, motivating the pattern-conditioned HS Gibbs proposal."""
    ix = bt.ix
    B = ix.B
    ph_full = phi_r(model, r).detach()
    X = sample_slots(model, ix, slot_utilities(model, ix, None, r, beta=0.0), gen, beta=0.0)
    X = gibbs(model, ix, X, r, gen, sweeps=50)
    centres = []
    for _ in range(64):
        X = gibbs(model, ix, X, r, gen)
        centres.append(basket_field(model, ix, X, r))
    centres = torch.stack(centres, 1)
    Kc = centres.shape[1]
    picks = torch.randint(Kc, (B, draws), generator=gen)
    z_uncond = centres.gather(1, picks[:, :, None].expand(B, draws, r)) + torch.randn(B, draws, r, generator=gen)
    half = 0.5 * r * math.log(2 * math.pi)
    lp = -0.5 * z_uncond.square().sum(2) - half
    comp = -0.5 * (z_uncond[:, :, None, :] - centres[:, None, :, :]).square().sum(3) - half
    lq = torch.logsumexp(comp, 2) - math.log(Kc)

    ph_slot = ph_full[ix.item]
    u0_slot = model.b_flat(ix).detach() - 0.5 * ph_slot.square().sum(-1)
    nc_row = torch.zeros(ix.n_rows, dtype=torch.long).index_add_(0, ix.row_of, bt.obs.long())
    act_rows = torch.nonzero(nc_row == 1, as_tuple=True)[0]
    row_remap = torch.full((ix.n_rows,), -1, dtype=torch.long)
    row_remap[act_rows] = torch.arange(len(act_rows))
    act_row_trip = ix.row_trip[act_rows]
    act_slots = torch.nonzero((nc_row == 1)[ix.row_of], as_tuple=True)[0]
    slot_act_row = row_remap[ix.row_of[act_slots]]
    slot_trip = ix.item_trip[act_slots]
    u0_act = u0_slot[act_slots]
    ph_act = ph_slot[act_slots]
    k_b = torch.zeros(B, dtype=torch.long).index_add_(0, act_row_trip, torch.ones_like(act_row_trip))
    rho0_k = model.rho_0()[k_b].detach()

    u_sd = u0_act[:, None] + torch.einsum("tr,tdr->td", ph_act, z_uncond[slot_trip])
    m_rd = torch.full((len(act_rows), draws), -float("inf"), dtype=u_sd.dtype)
    m_rd.index_reduce_(0, slot_act_row, u_sd, "amax", include_self=True)
    lse_rd = m_rd + torch.log(torch.zeros_like(m_rd).index_add_(0, slot_act_row, torch.exp(u_sd - m_rd[slot_act_row])))
    lzc_pat = torch.zeros((B, draws), dtype=u_sd.dtype).index_add_(0, act_row_trip, lse_rd) - rho0_k[:, None]
    lw_uncond = lp + lzc_pat - lq
    ess_uncond = 1.0 / torch.softmax(lw_uncond, 1).square().sum(1) / draws
    return {
        "batch_trips": int(B),
        "draws": int(draws),
        "unconditional_proposal_ess_mean": round(float(ess_uncond.mean()), 4),
        "unconditional_proposal_ess_median": round(float(ess_uncond.median()), 4),
        "unconditional_proposal_ess_min": round(float(ess_uncond.min()), 4),
        "frac_trips_ess_below_0.01": round(float((ess_uncond < 0.01).double().mean()), 4),
    }


# ==============================================================================
# 3. Exact Truth Ceiling via Latent Marginalisation of generate_stress_world.py
# ==============================================================================
def _softmax_1d(x: np.ndarray) -> np.ndarray:
    m = np.max(x)
    e = np.exp(x - m)
    return e / np.sum(e)


def _truth_cat_cond(
    S_c: np.ndarray,
    u_loy_c: np.ndarray,
    u_base_c: np.ndarray,
    sub_c: np.ndarray,
    alpha_c: np.ndarray,
    mu: float = 0.5,
) -> tuple[float, float, float, float, float, float]:
    """Exact conditional category probabilities (P(x_c|z_c=0), P(x_c|z_c=1), P(n_c|z_c=0),
    P(n_c|z_c=1), P(S_c|z_c=0), P(S_c|z_c=1)) under the stress-world generator."""
    m = len(u_loy_c)
    k_obs = int(S_c.sum())
    x_obs = 1 if k_obs > 0 else 0
    if m == 0:
        v = 1.0 if k_obs == 0 else 0.0
        return (v, v, v, v, v, v)
    om = 1.0 - alpha_c
    p_emp = float(np.prod(om))
    p_x0 = p_emp if x_obs == 0 else (1.0 - p_emp)
    poly = np.zeros(m + 1, dtype=np.float64)
    poly[0] = 1.0
    for i in range(m):
        a = alpha_c[i]
        omi = om[i]
        poly[1:i + 2] = poly[1:i + 2] * omi + poly[:i + 1] * a
        poly[0] *= omi
    p_n0 = float(poly[k_obs])
    p_S0 = float(np.prod(np.where(S_c, alpha_c, om)))
    if x_obs == 0:
        return (p_x0, 0.0, p_n0, 0.0, p_S0, 0.0)
    p_x1 = 1.0
    nests = np.unique(sub_c)
    iv = np.empty(len(nests), dtype=np.float64)
    masks = []
    for idx_s, s in enumerate(nests):
        mk = sub_c == s
        masks.append(mk)
        v = u_loy_c[mk] / mu
        vm = np.max(v)
        iv[idx_s] = mu * (vm + math.log(float(np.sum(np.exp(v - vm)))))
    p_sub = _softmax_1d(iv)
    p_prim = np.empty(m, dtype=np.float64)
    for idx_s, mk in enumerate(masks):
        p_prim[mk] = p_sub[idx_s] * _softmax_1d(u_loy_c[mk] / mu)
    if m == 1:
        v1 = 1.0 if k_obs == 1 else 0.0
        return (p_x0, p_x1, p_n0, v1, p_S0, v1)

    p_need_single = 0.9 * p_prim
    w_base = np.exp(u_base_c - np.max(u_base_c))
    sum_w_base = float(np.sum(w_base))
    denom_j1 = np.maximum(sum_w_base - w_base, 1e-300)
    p_var_cond = w_base[None, :] / denom_j1[:, None]
    np.fill_diagonal(p_var_cond, 0.0)
    p_need_pair = 0.1 * (p_prim[:, None] * p_var_cond + p_prim[None, :] * p_var_cond.T)

    q1 = np.empty((m, m), dtype=np.float64)
    for j1 in range(m):
        omi = om[j1]
        a = alpha_c[j1]
        q = np.empty(m, dtype=np.float64)
        q[0] = poly[0] / omi
        for k in range(1, m):
            q[k] = (poly[k] - a * q[k - 1]) / omi
        q1[j1] = q

    p_n1 = float(np.sum(p_need_single * q1[:, k_obs - 1]))
    if k_obs >= 2:
        for j1 in range(m - 1):
            q_j1 = q1[j1]
            for j2 in range(j1 + 1, m):
                om2 = om[j2]
                a2 = alpha_c[j2]
                val = q_j1[0] / om2
                for k in range(1, k_obs - 1):
                    val = (q_j1[k] - a2 * val) / om2
                p_n1 += p_need_pair[j1, j2] * val

    bought_idx = np.flatnonzero(S_c)
    p_S1 = 0.0
    for idx_a, j1 in enumerate(bought_idx):
        p_S1 += p_need_single[j1] * (p_S0 / alpha_c[j1])
        for j2 in bought_idx[idx_a + 1:]:
            p_S1 += p_need_pair[j1, j2] * (p_S0 / (alpha_c[j1] * alpha_c[j2]))

    return (p_x0, p_x1, p_n0, p_n1, p_S0, p_S1)


def _marg_needs_positive_tuples(
    world,
    f0_mat: np.ndarray,
    f1_mat: np.ndarray,
    lam_h: float,
    pi_h: np.ndarray,
    inv_h: np.ndarray,
    eps: float = 1e-24,
) -> np.ndarray:
    """Exact positive mission-tuple + recipe-subset marginalisation E_z[ prod_c f_c(z_c) ]
    for M rows of (f0_mat, f1_mat) of shape [M, 48]. Every term is strictly non-negative."""
    M_rows = f0_mat.shape[0]
    present = set(int(c) for c in np.flatnonzero(f1_mat[0] > 0))
    tilde_A = sorted(present)
    cat_idx = {c: i for i, c in enumerate(tilde_A)}
    K = len(tilde_A)
    q_out = np.ones(8, dtype=np.float64)
    s_mc = np.ones((8, K), dtype=np.float64)
    adm_recipes = []
    for m, tmpl in enumerate(world.templates):
        for c, p in zip(tmpl["categories"], tmpl["probs"]):
            c = int(c)
            if inv_h[c] == 0:
                if c not in cat_idx:
                    q_out[m] *= (1.0 - p)
                else:
                    s_mc[m, cat_idx[c]] *= (1.0 - p)
        rp = tmpl["recipe_prob"]
        for r in tmpl["recipes"]:
            if all(int(rc) in cat_idx for rc in r):
                rmask = [cat_idx[int(rc)] for rc in r]
                adm_recipes.append((m, rmask, rp))
            else:
                q_out[m] *= (1.0 - rp)
    active_m, inactive_m = [], []
    rec_missions = set(m for m, _, _ in adm_recipes)
    for m in range(8):
        if m in rec_missions or np.any(s_mc[m] < 1.0):
            active_m.append(m)
        else:
            inactive_m.append(m)
    lam_inact_factor = math.exp(sum(lam_h * pi_h[m] * (q_out[m] - 1.0) for m in inactive_m))
    pi_inact_q = sum(pi_h[m] * q_out[m] for m in inactive_m)
    inact_cats = [c for c in range(48) if c not in cat_idx]
    base_f0_inact = np.prod(f0_mat[:, inact_cats], axis=1) if inact_cats else np.ones(M_rows)
    if len(active_m) == 0:
        p_no_needs = lam_inact_factor * pi_inact_q
        return base_f0_inact * np.prod(f0_mat[:, tilde_A], axis=1) * p_no_needs

    k_mat = np.zeros((1, 0), dtype=np.int64)
    w_k = np.array([lam_inact_factor], dtype=np.float64)
    for m in active_m:
        mu = lam_h * pi_h[m] * q_out[m]
        probs_k = []
        p_cur = math.exp(-lam_h * pi_h[m])
        k = 0
        while True:
            probs_k.append(p_cur)
            if k >= 2 and p_cur < eps:
                break
            k += 1
            p_cur = p_cur * mu / k
        g = np.array(probs_k, dtype=np.float64)
        w_next = (w_k[:, None] * g[None, :]).ravel()
        keep = w_next > eps
        r_idx, c_idx = np.divmod(np.flatnonzero(keep), len(g))
        k_mat = np.column_stack([k_mat[r_idx], c_idx])
        w_k = w_next[keep]

    w_tot = w_k * (pi_inact_q + k_mat.sum(axis=1) / lam_h)
    keep_mask = w_tot > eps
    k_mat = k_mat[keep_mask]
    w_tot = w_tot[keep_mask]
    log_s_mc = np.log(np.clip(s_mc[active_m], 1e-300, 1.0))
    s_ck = np.exp(k_mat @ log_s_mc)
    f0_act = f0_mat[:, None, tilde_A]
    f1_act = f1_mat[:, None, tilde_A]
    term_mck = f0_act * s_ck[None, :, :] + f1_act * (1.0 - s_ck[None, :, :])
    if len(adm_recipes) == 0:
        return base_f0_inact * (np.prod(term_mck, axis=2) @ w_tot)
    m_to_pos = {m: pos for pos, m in enumerate(active_m)}
    n_rec = len(adm_recipes)
    rec_p0 = np.empty((n_rec, len(w_tot)), dtype=np.float64)
    for r_i, (m, _, rp) in enumerate(adm_recipes):
        rec_p0[r_i] = (1.0 - rp) ** k_mat[:, m_to_pos[m]]
    totals = np.zeros(M_rows, dtype=np.float64)
    for mask in range(1 << n_rec):
        w_rec = w_tot.copy()
        forced = set()
        for r_i, (_, rmask, _) in enumerate(adm_recipes):
            if (mask >> r_i) & 1:
                w_rec *= (1.0 - rec_p0[r_i])
                forced.update(rmask)
            else:
                w_rec *= rec_p0[r_i]
        t_mod = term_mck.copy()
        for c_pos in forced:
            t_mod[:, :, c_pos] = f1_act[:, 0, c_pos:c_pos + 1]
        totals += np.prod(t_mod, axis=2) @ w_rec
    return base_f0_inact * totals


def _marg_needs_mobius_single(world, f0: np.ndarray, f1: np.ndarray, lam_h: float, pi_h: np.ndarray, inv_h: np.ndarray) -> float:
    """Closed-form Fast Mobius Transform of F(U) = q_bar(U) exp(lam_h (q_bar(U) - 1)) for cross-checking."""
    tilde_A = sorted(int(c) for c in np.flatnonzero(f1 > 0))
    K = len(tilde_A)
    cat_bit = {c: i for i, c in enumerate(tilde_A)}
    N = 1 << K
    q_bar = np.zeros(N, dtype=np.float64)
    u_idx = np.arange(N, dtype=np.int64)
    for m, tmpl in enumerate(world.templates):
        base_q = 1.0
        act_bits, act_surv = [], []
        for c, p in zip(tmpl["categories"], tmpl["probs"]):
            c = int(c)
            if inv_h[c] == 0:
                if c not in cat_bit:
                    base_q *= (1.0 - p)
                else:
                    act_bits.append(cat_bit[c])
                    act_surv.append(1.0 - p)
        rec_masks = []
        rp = tmpl["recipe_prob"]
        for r in tmpl["recipes"]:
            if all(int(rc) in cat_bit for rc in r):
                mask = 0
                for rc in r:
                    mask |= (1 << cat_bit[int(rc)])
                rec_masks.append(mask)
            else:
                base_q *= (1.0 - rp)
        qm = np.full(N, base_q, dtype=np.float64)
        for bit, surv in zip(act_bits, act_surv):
            qm = np.where((u_idx >> bit) & 1, qm, qm * surv)
        for rmask in rec_masks:
            qm = np.where((u_idx & rmask) == rmask, qm, qm * (1.0 - rp))
        q_bar += pi_h[m] * qm
    F = q_bar * np.exp(lam_h * (q_bar - 1.0))
    P_U = F.copy()
    for i in range(K):
        step = 1 << i
        for start in range(0, N, step * 2):
            P_U[start + step:start + 2 * step] -= P_U[start:start + step]
    base_f0 = float(np.prod(f0))
    ratios = np.ones(N, dtype=np.float64)
    for i, c in enumerate(tilde_A):
        r_c = f1[c] / f0[c]
        step = 1 << i
        for start in range(0, N, step * 2):
            ratios[start + step:start + 2 * step] *= r_c
    return base_f0 * float(np.dot(P_U, ratios))


def extract_or_load_pre_trip_states(trips_4096: np.ndarray, cache_path: Path) -> dict[str, np.ndarray]:
    """Replay generate_stress_world.py once to extract exact pre-trip states (loyalty, inventory, taste)
    for the 4,096 test trips, or load from cache_path if already present and matching."""
    if cache_path.exists():
        st = np.load(cache_path)
        if np.array_equal(st["trips_4096"], trips_4096):
            return {k: st[k] for k in st.files}

    w_args = argparse.Namespace(
        seed=20260924, households=3000, stores=15, departments=6,
        categories_per_department=8, mission_variants=1, weeks=52,
        train_weeks=32, validation_weeks=10, scenario_products=16
    )
    rng = np.random.default_rng(w_args.seed)
    world = gsw.World(w_args, rng)
    H, S, T, J, C = w_args.households, w_args.stores, w_args.weeks, world.J, world.C
    visits = rng.random((H, T)) < world.visit_rate[:, None]
    hh, wk = np.nonzero(visits)
    store = np.where(rng.random(len(hh)) < 0.85, world.home[hh], rng.integers(0, S, len(hh)))
    day = wk * 7 + rng.integers(0, 7, len(hh))
    order = np.lexsort((hh, day))
    hh, wk, store, day = hh[order], wk[order], store[order], day[order]
    trip_id = np.arange(len(hh))
    state0 = {
        "loyalty": np.full((H, C), -1),
        "inventory": np.zeros((H, C), dtype=int),
        "taste": world.taste0.copy(),
        "week": 0,
    }
    test_start = w_args.train_weeks + w_args.validation_weeks
    before = wk < test_start
    baskets_a, state_test = gsw.simulate(
        world, (hh[before], store[before], wk[before], trip_id[before]), state0, seed_base=w_args.seed
    )

    # Step through test trips recording pre-trip state before each trip
    loyalty = state_test["loyalty"].copy()
    inventory = state_test["inventory"].copy()
    taste = state_test["taste"].copy()
    last_week = state_test["week"]
    ref = world.base_price

    test_indices = np.flatnonzero(~before)
    n_test_raw = len(test_indices)
    pre_loy_test = np.empty((n_test_raw, C), dtype=np.int64)
    pre_inv_test = np.empty((n_test_raw, C), dtype=np.int64)
    pre_taste_test = np.empty((n_test_raw, 3), dtype=np.float64)
    baskets_b = []

    for idx_b, raw_i in enumerate(test_indices):
        h_i, s_i, w_i, tid_i = int(hh[raw_i]), int(store[raw_i]), int(wk[raw_i]), int(trip_id[raw_i])
        if w_i != last_week:
            for _ in range(w_i - last_week):
                taste = 0.9 * taste + 0.1 * world.taste0 + np.random.default_rng(
                    [w_args.seed, 7, int(last_week)]
                ).normal(0, 0.08, taste.shape)
                inventory = np.maximum(inventory - 1, 0)
                last_week += 1
        pre_loy_test[idx_b] = loyalty[h_i]
        pre_inv_test[idx_b] = inventory[h_i]
        pre_taste_test[idx_b] = taste[h_i]

        r = np.random.default_rng([w_args.seed, tid_i])
        available = (world.first_stocked[:, s_i] <= w_i + 1) & ~world.stockout[:, s_i, w_i]
        log_ratio = np.log(world.price[:, w_i] / ref)
        price_term = -world.price_sens[h_i] * np.where(log_ratio > 0, 2.0 * log_ratio, log_ratio)
        utility = (
            world.quality + world.attributes @ taste[h_i] + world.idiosyncratic[h_i]
            + price_term + 0.5 * world.promo[:, w_i]
        )
        needs = set()
        n_missions = 1 + r.poisson(world.intensity[h_i])
        for m in r.choice(len(gsw.MISSIONS), size=n_missions, p=world.mission_pref[h_i]):
            t = world.templates[m]
            for c, p in zip(t["categories"], t["probs"]):
                if inventory[h_i, c] == 0 and r.random() < p:
                    needs.add(int(c))
            for recipe in t["recipes"]:
                if r.random() < t["recipe_prob"]:
                    needs.update(int(c) for c in recipe)
        basket = set()
        for c in sorted(needs):
            items = np.flatnonzero((world.category == c) & available)
            if len(items) == 0:
                continue
            u = utility[items] + 1.2 * (items == loyalty[h_i, c])
            subs = world.subcategory[items]
            mu = 0.5
            nests = np.unique(subs)
            iv = np.array([mu * np.log(np.exp(u[subs == s] / mu).sum()) for s in nests])
            s_pick = nests[r.choice(len(nests), p=np.exp(iv - iv.max()) / np.exp(iv - iv.max()).sum())]
            inside = items[subs == s_pick]
            w = np.exp(u[subs == s_pick] / mu)
            j = int(r.choice(inside, p=w / w.sum()))
            basket.add(j)
            loyalty[h_i, c] = j
            if world.promo[j, w_i] and r.random() < 0.5:
                inventory[h_i, c] = int(r.integers(2, 5))
            if r.random() < 0.10 and len(items) > 1:
                others = items[items != j]
                w2 = np.exp(utility[others] - utility[others].max())
                basket.add(int(r.choice(others, p=w2 / w2.sum())))
        for _ in range(r.poisson(0.4)):
            pool = np.flatnonzero(available)
            w3 = np.exp(world.quality[pool])
            basket.add(int(r.choice(pool, p=w3 / w3.sum())))
        baskets_b.append(sorted(basket))

    baskets = baskets_a + baskets_b
    keep = np.array([len(b) > 0 for b in baskets])
    period_all = wk + 1
    trained_households = np.unique(hh[keep & (period_all <= w_args.train_weeks)])
    keep &= (period_all <= w_args.train_weeks) | np.isin(hh, trained_households)
    train_items = np.zeros(J, dtype=bool)
    for i in np.flatnonzero(keep & (period_all <= w_args.train_weeks)):
        train_items[baskets[i]] = True
    item_map = np.full(J, -1)
    item_map[train_items] = np.arange(int(train_items.sum()))
    baskets_mod = [[int(item_map[j]) for j in b if item_map[j] >= 0] for b in baskets]
    keep &= np.array([len(b) > 0 for b in baskets_mod])
    kept_raw_indices = np.flatnonzero(keep)
    raw_4096 = kept_raw_indices[trips_4096]
    test_offset = int(before.sum())
    b_idx_4096 = raw_4096 - test_offset

    out = {
        "trips_4096": trips_4096,
        "raw_4096": raw_4096,
        "hh": hh[raw_4096],
        "store": store[raw_4096],
        "wk": wk[raw_4096],
        "tid": trip_id[raw_4096],
        "pre_loyalty": pre_loy_test[b_idx_4096],
        "pre_inventory": pre_inv_test[b_idx_4096],
        "pre_taste": pre_taste_test[b_idx_4096],
        "train_items": train_items,
        "item_map": item_map,
    }
    np.savez(cache_path, **out)
    return out


def compute_truth_ceiling(
    trips_4096: np.ndarray,
    cache_state_path: Path,
    cache_truth_npz: Path,
) -> dict[str, np.ndarray | float]:
    """Compute exact truth log-likelihood ceiling and 3-part decomposition on all 4,096 test trips,
    plus static-state and mechanism ablations."""
    if cache_truth_npz.exists():
        cached = np.load(cache_truth_npz)
        if np.array_equal(cached["trips"], trips_4096) and "mobius_max_rel_diff" in cached.files:
            return {k: cached[k] for k in cached.files}

    w_args = argparse.Namespace(
        seed=20260924, households=3000, stores=15, departments=6,
        categories_per_department=8, mission_variants=1, weeks=52,
        train_weeks=32, validation_weeks=10, scenario_products=16
    )
    world = gsw.World(w_args, np.random.default_rng(w_args.seed))
    st = extract_or_load_pre_trip_states(trips_4096, cache_state_path)
    obs = np.load(ROOT / "data/stress_world/observed.npz")
    memberships = obs["memberships"]
    train_items = st["train_items"]

    N_trips = len(trips_4096)
    truth_pres_ll = np.empty(N_trips, dtype=np.float64)
    truth_cnt_ll = np.empty(N_trips, dtype=np.float64)
    truth_prod_ll = np.empty(N_trips, dtype=np.float64)
    truth_tot_ll = np.empty(N_trips, dtype=np.float64)

    # Ablations on truth generator to separate dynamic oracle state from static mechanisms
    static_pres_ll = np.empty(N_trips, dtype=np.float64)
    static_cnt_ll = np.empty(N_trips, dtype=np.float64)
    static_prod_ll = np.empty(N_trips, dtype=np.float64)
    static_tot_ll = np.empty(N_trips, dtype=np.float64)
    no_hh_cat_pres_ll = np.empty(N_trips, dtype=np.float64)

    ref = world.base_price
    cat_items_world = [np.flatnonzero(world.category == c) for c in range(48)]
    mean_pi = world.mission_pref.mean(axis=0)
    mean_lam = float(world.intensity.mean())
    inv_zero = np.zeros(48, dtype=np.int64)

    mobius_rel_diffs = []

    for idx in range(N_trips):
        t = int(trips_4096[idx])
        hh = int(st["hh"][idx])
        store = int(st["store"][idx])
        wk = int(st["wk"][idx])
        loy = st["pre_loyalty"][idx]
        inv = st["pre_inventory"][idx]
        taste = st["pre_taste"][idx]

        avail = (world.first_stocked[:, store] <= wk + 1) & ~world.stockout[:, store, wk]
        log_ratio = np.log(world.price[:, wk] / ref)
        price_term = -world.price_sens[hh] * np.where(log_ratio > 0, 2.0 * log_ratio, log_ratio)
        utility = (
            world.quality + world.attributes @ taste + world.idiosyncratic[hh]
            + price_term + 0.5 * world.promo[:, wk]
        )

        # Static utility (no loyalty, no AR(1) taste drift, no unrecorded stockout)
        avail_static = world.first_stocked[:, store] <= wk + 1
        utility_static = (
            world.quality + world.attributes @ world.taste0[hh] + world.idiosyncratic[hh]
            + price_term + 0.5 * world.promo[:, wk]
        )

        pool = np.flatnonzero(avail)
        w3 = np.exp(world.quality[pool])
        q_pool = w3 / np.sum(w3)
        alpha_world = np.zeros(world.J, dtype=np.float64)
        alpha_world[pool] = -np.expm1(-0.4 * q_pool)

        pool_st = np.flatnonzero(avail_static)
        w3_st = np.exp(world.quality[pool_st])
        q_pool_st = w3_st / np.sum(w3_st)
        alpha_static = np.zeros(world.J, dtype=np.float64)
        alpha_static[pool_st] = -np.expm1(-0.4 * q_pool_st)

        bought_mod = memberships[t]
        bought_world = np.zeros(world.J, dtype=bool)
        bought_world[train_items] = bought_mod

        f0_mat = np.empty((3, 48), dtype=np.float64)
        f1_mat = np.empty((3, 48), dtype=np.float64)
        f0_emp = np.empty(48, dtype=np.float64)
        f1_emp = np.empty(48, dtype=np.float64)

        f0_st = np.empty((3, 48), dtype=np.float64)
        f1_st = np.empty((3, 48), dtype=np.float64)
        f0_emp_st = np.empty(48, dtype=np.float64)
        f1_emp_st = np.empty(48, dtype=np.float64)

        for c in range(48):
            items_c = cat_items_world[c]
            avail_c = items_c[avail[items_c] & train_items[items_c]]
            S_c = bought_world[avail_c]
            u_base_c = utility[avail_c]
            u_loy_c = u_base_c + 1.2 * (avail_c == loy[c])
            sub_c = world.subcategory[avail_c]
            alpha_c = alpha_world[avail_c]
            px0, px1, pn0, pn1, pS0, pS1 = _truth_cat_cond(S_c, u_loy_c, u_base_c, sub_c, alpha_c)
            f0_mat[0, c], f1_mat[0, c] = px0, px1
            f0_mat[1, c], f1_mat[1, c] = pn0, pn1
            f0_mat[2, c], f1_mat[2, c] = pS0, pS1
            f0_emp[c] = float(np.prod(1.0 - alpha_c)) if len(avail_c) > 0 else 1.0
            f1_emp[c] = 0.0 if len(avail_c) > 0 else 1.0

            avail_c_st = items_c[avail_static[items_c] & train_items[items_c]]
            S_c_st = bought_world[avail_c_st]
            u_st_c = utility_static[avail_c_st]
            sub_c_st = world.subcategory[avail_c_st]
            alpha_c_st = alpha_static[avail_c_st]
            spx0, spx1, spn0, spn1, spS0, spS1 = _truth_cat_cond(
                S_c_st, u_st_c, u_st_c, sub_c_st, alpha_c_st
            )
            f0_st[0, c], f1_st[0, c] = spx0, spx1
            f0_st[1, c], f1_st[1, c] = spn0, spn1
            f0_st[2, c], f1_st[2, c] = spS0, spS1
            f0_emp_st[c] = float(np.prod(1.0 - alpha_c_st)) if len(avail_c_st) > 0 else 1.0
            f1_emp_st[c] = 0.0 if len(avail_c_st) > 0 else 1.0

        lam_h = float(world.intensity[hh])
        pi_h = world.mission_pref[hh]
        p_x, p_n, p_S = _marg_needs_positive_tuples(world, f0_mat, f1_mat, lam_h, pi_h, inv)
        p_empty = _marg_needs_positive_tuples(world, f0_emp[None, :], f1_emp[None, :], lam_h, pi_h, inv)[0]
        log_norm = math.log1p(-p_empty)
        truth_pres_ll[idx] = math.log(p_x) - log_norm
        truth_cnt_ll[idx] = math.log(p_n) - math.log(p_x)
        truth_prod_ll[idx] = math.log(p_S) - math.log(p_n)
        truth_tot_ll[idx] = truth_pres_ll[idx] + truth_cnt_ll[idx] + truth_prod_ll[idx]

        # Cross-check against closed-form Fast Mobius Transform on first 100 trips with |tilde_A| <= 12
        if len(mobius_rel_diffs) < 100 and int((f1_mat[0] > 0).sum()) <= 12:
            p_S_mob = _marg_needs_mobius_single(world, f0_mat[2], f1_mat[2], lam_h, pi_h, inv)
            mobius_rel_diffs.append(abs(p_S - p_S_mob) / max(p_S, 1e-300))

        # Static-state truth ceiling (inv=0, loyalty=-1, taste=taste0, no unrecorded stockouts)
        sp_x, sp_n, sp_S = _marg_needs_positive_tuples(world, f0_st, f1_st, lam_h, pi_h, inv_zero)
        sp_empty = _marg_needs_positive_tuples(world, f0_emp_st[None, :], f1_emp_st[None, :], lam_h, pi_h, inv_zero)[0]
        slog_norm = math.log1p(-sp_empty)
        static_pres_ll[idx] = math.log(sp_x) - slog_norm
        static_cnt_ll[idx] = math.log(sp_n) - math.log(sp_x)
        static_prod_ll[idx] = math.log(sp_S) - math.log(sp_n)
        static_tot_ll[idx] = static_pres_ll[idx] + static_cnt_ll[idx] + static_prod_ll[idx]

        # Ablation: replace household mission_pref[hh] and intensity[hh] by population averages
        nh_x = _marg_needs_positive_tuples(world, f0_mat[0:1], f1_mat[0:1], mean_lam, mean_pi, inv)[0]
        nh_emp = _marg_needs_positive_tuples(world, f0_emp[None, :], f1_emp[None, :], mean_lam, mean_pi, inv)[0]
        no_hh_cat_pres_ll[idx] = math.log(nh_x) - math.log1p(-nh_emp)

    out = {
        "trips": trips_4096,
        "truth_total_ll": truth_tot_ll,
        "truth_presence_ll": truth_pres_ll,
        "truth_count_given_presence_ll": truth_cnt_ll,
        "truth_product_given_counts_ll": truth_prod_ll,
        "static_truth_total_ll": static_tot_ll,
        "static_truth_presence_ll": static_pres_ll,
        "static_truth_count_given_presence_ll": static_cnt_ll,
        "static_truth_product_given_counts_ll": static_prod_ll,
        "no_hh_cat_taste_presence_ll": no_hh_cat_pres_ll,
        "mobius_max_rel_diff": np.array(max(mobius_rel_diffs) if mobius_rel_diffs else 0.0),
    }
    np.savez(cache_truth_npz, **out)
    return out


# ==============================================================================
# 4. Formatting Helpers for 3-Part Splits and Top-10 Category Tables
# ==============================================================================
def build_split_summary(
    mask: np.ndarray,
    n_cats: np.ndarray,
    n_items: np.ndarray,
    chain_tot: np.ndarray,
    chain_pres: np.ndarray,
    chain_cnt: np.ndarray,
    chain_prod: np.ndarray,
    v4_tot: np.ndarray,
    v4_pres: np.ndarray,
    v4_cnt: np.ndarray,
    v4_prod: np.ndarray,
    v4_prod_marg_sum: np.ndarray,
    v4_prod_b_sum: np.ndarray,
    truth_dict: dict | None = None,
    v4_cnt_loo_sum: np.ndarray | None = None,
) -> dict:
    """Build a structured 3-part gap dictionary for a subset of trips selected by `mask`."""
    c_tot = float(chain_tot[mask].mean())
    c_pres = float(chain_pres[mask].mean())
    c_cnt = float(chain_cnt[mask].mean())
    c_prod = float(chain_prod[mask].mean())

    m_tot = float(v4_tot[mask].mean())
    m_pres = float(v4_pres[mask].mean())
    m_cnt = float(v4_cnt[mask].mean())
    m_prod = float(v4_prod[mask].mean())
    m_prod_marg = float(v4_prod_marg_sum[mask].mean())
    m_prod_b = float(v4_prod_b_sum[mask].mean())

    g_tot = c_tot - m_tot
    g_pres = c_pres - m_pres
    g_cnt = c_cnt - m_cnt
    g_prod = c_prod - m_prod
    g_pat = (c_pres + c_cnt) - (m_pres + m_cnt)

    max_trip_sum_err_chain = float(np.abs((chain_pres[mask] + chain_cnt[mask] + chain_prod[mask]) - chain_tot[mask]).max())
    max_trip_sum_err_v4 = float(np.abs((v4_pres[mask] + v4_cnt[mask] + v4_prod[mask]) - v4_tot[mask]).max())
    max_trip_sum_err_gap = float(
        np.abs(
            ((chain_pres[mask] - v4_pres[mask]) + (chain_cnt[mask] - v4_cnt[mask]) + (chain_prod[mask] - v4_prod[mask]))
            - (chain_tot[mask] - v4_tot[mask])
        ).max()
    )

    subsplit_loo = {}
    if v4_cnt_loo_sum is not None:
        m_cnt_loo = float(v4_cnt_loo_sum[mask].mean())
        m_pres_loo = (m_pres + m_cnt) - m_cnt_loo
        subsplit_loo = {
            "chain_binary_presence_ll": round(c_pres, 4),
            "m3_binary_presence_ll": round(m_pres_loo, 4),
            "binary_presence_gap_nats_per_trip": round(c_pres - m_pres_loo, 4),
            "chain_count_1_given_presence_ll": round(c_cnt, 4),
            "m3_count_1_given_presence_ll": round(m_cnt_loo, 4),
            "count_1_given_presence_gap_nats_per_trip": round(c_cnt - m_cnt_loo, 4),
        }

    out = {
        "n_trips": int(mask.sum()),
        "mean_active_categories": round(float(n_cats[mask].mean()), 4),
        "mean_basket_items": round(float(n_items[mask].mean()), 4),
        "chain_total_ll": round(c_tot, 4),
        "m3_total_ll": round(m_tot, 4),
        "total_gap_nats_per_trip": round(g_tot, 4),
        "max_per_trip_sum_mismatch_chain": max_trip_sum_err_chain,
        "max_per_trip_sum_mismatch_v4": max_trip_sum_err_v4,
        "max_per_trip_sum_mismatch_gap": max_trip_sum_err_gap,
        "sums_within_1e_10_on_every_trip": bool(
            max_trip_sum_err_chain <= 1e-10 and max_trip_sum_err_v4 <= 1e-10 and max_trip_sum_err_gap <= 1e-10
        ),
        "three_part_split": {
            "which_categories_present": {
                "chain_ll": round(c_pres, 4),
                "v4_ll": round(m_pres, 4),
                "gap_chain_minus_v4": round(g_pres, 4),
                "share_of_total_gap": round(g_pres / g_tot, 4) if abs(g_tot) > 1e-12 else 0.0,
            },
            "how_many_items_given_presence": {
                "chain_ll": round(c_cnt, 4),
                "v4_ll": round(m_cnt, 4),
                "gap_chain_minus_v4": round(g_cnt, 4),
                "share_of_total_gap": round(g_cnt / g_tot, 4) if abs(g_tot) > 1e-12 else 0.0,
            },
            "which_products_given_counts": {
                "chain_ll": round(c_prod, 4),
                "v4_ll": round(m_prod, 4),
                "v4_marginal_sum_ll": round(m_prod_marg, 4),
                "v4_b_only_ll": round(m_prod_b, 4),
                "gap_chain_minus_v4": round(g_prod, 4),
                "gap_per_item": round(g_prod / float(n_items[mask].mean()), 4),
                "share_of_total_gap": round(g_prod / g_tot, 4) if abs(g_tot) > 1e-12 else 0.0,
            },
        },
        "category_pattern": {
            "chain_pattern_ll": round(c_pres + c_cnt, 4),
            "m3_pattern_ll": round(m_pres + m_cnt, 4),
            "gap_nats_per_trip": round(g_pat, 4),
            "share_of_total_gap": round(g_pat / g_tot, 4) if abs(g_tot) > 1e-12 else 0.0,
            "subsplit_binary_presence_vs_count_1": subsplit_loo if subsplit_loo else {
                "chain_binary_presence_ll": round(c_pres, 4),
                "m3_binary_presence_ll": round(m_pres, 4),
                "binary_presence_gap_nats_per_trip": round(g_pres, 4),
                "chain_count_given_presence_ll": round(c_cnt, 4),
                "m3_count_given_presence_ll": round(m_cnt, 4),
                "count_given_presence_gap_nats_per_trip": round(g_cnt, 4),
            },
        },
        "products_given_pattern": {
            "chain_product_ll": round(c_prod, 4),
            "m3_product_ll": round(m_prod, 4),
            "m3_product_marginal_sum_ll": round(m_prod_marg, 4),
            "m3_product_b_only_ll": round(m_prod_b, 4),
            "gap_nats_per_trip": round(g_prod, 4),
            "gap_nats_per_item": round(g_prod / float(n_items[mask].mean()), 4),
            "share_of_total_gap": round(g_prod / g_tot, 4) if abs(g_tot) > 1e-12 else 0.0,
        },
    }

    if truth_dict is not None:
        t_tot = float(truth_dict["truth_total_ll"][mask].mean())
        t_pres = float(truth_dict["truth_presence_ll"][mask].mean())
        t_cnt = float(truth_dict["truth_count_given_presence_ll"][mask].mean())
        t_prod = float(truth_dict["truth_product_given_counts_ll"][mask].mean())
        out["truth_ceiling"] = {
            "truth_total_ll": round(t_tot, 4),
            "truth_presence_ll": round(t_pres, 4),
            "truth_count_given_presence_ll": round(t_cnt, 4),
            "truth_product_given_counts_ll": round(t_prod, 4),
            "truth_minus_chain": {
                "total": round(t_tot - c_tot, 4),
                "which_categories_present": round(t_pres - c_pres, 4),
                "how_many_items_given_presence": round(t_cnt - c_cnt, 4),
                "which_products_given_counts": round(t_prod - c_prod, 4),
            },
            "truth_minus_v4": {
                "total": round(t_tot - m_tot, 4),
                "which_categories_present": round(t_pres - m_pres, 4),
                "how_many_items_given_presence": round(t_cnt - m_cnt, 4),
                "which_products_given_counts": round(t_prod - m_prod, 4),
            },
        }
    return out


def build_top10_categories_for_subset(
    mask: np.ndarray,
    n_ch_np: np.ndarray,
    cat_of_np: np.ndarray,
    chain_loo_pres_bc: np.ndarray,
    v4_loo_pres_bc: np.ndarray,
    chain_loo_pat_bc: np.ndarray,
    v4_loo_pat_bc: np.ndarray,
    chain_cnt_bc: np.ndarray,
    v4_cnt_seq_bc: np.ndarray,
    v4_cnt_loo_bc: np.ndarray,
    chain_prod_bc: np.ndarray,
    v4_prod_seq_bc: np.ndarray,
    v4_prod_marg_bc: np.ndarray,
) -> dict:
    """Compute top-10 category attributions on a subset of trips (`mask`), both overall
    (combining presence + counts|presence + products|counts) and for each of the 3 parts."""
    n_sub = n_ch_np[mask]
    x_sub = n_sub > 0
    active_rate_c = x_sub.mean(axis=0)
    multi_rate_c = (n_sub >= 2).mean(axis=0)

    pres_gap_bc = chain_loo_pres_bc[mask] - v4_loo_pres_bc[mask]
    cnt_seq_gap_bc = chain_cnt_bc[mask] - v4_cnt_seq_bc[mask]
    cnt_loo_gap_bc = chain_cnt_bc[mask] - v4_cnt_loo_bc[mask]
    pat_loo_gap_bc = chain_loo_pat_bc[mask] - v4_loo_pat_bc[mask]
    prod_seq_gap_bc = chain_prod_bc[mask] - v4_prod_seq_bc[mask]
    prod_marg_gap_bc = chain_prod_bc[mask] - v4_prod_marg_bc[mask]

    # Total per-category contribution across all 3 parts:
    # Part 1 (LOO presence) + Part 2 (exact sequential count|presence) + Part 3 (exact sequential product|count)
    tot_3part_gap_bc = pres_gap_bc + cnt_seq_gap_bc + prod_seq_gap_bc

    pres_gap_c = pres_gap_bc.mean(axis=0)
    cnt_seq_gap_c = cnt_seq_gap_bc.mean(axis=0)
    cnt_loo_gap_c = cnt_loo_gap_bc.mean(axis=0)
    pat_loo_gap_c = pat_loo_gap_bc.mean(axis=0)
    prod_seq_gap_c = prod_seq_gap_bc.mean(axis=0)
    prod_marg_gap_c = prod_marg_gap_bc.mean(axis=0)
    tot_3part_gap_c = tot_3part_gap_bc.mean(axis=0)

    tot_3part_sum = float(tot_3part_gap_c.sum())
    pres_sum = float(pres_gap_c.sum())
    cnt_seq_sum = float(cnt_seq_gap_c.sum())
    pat_loo_sum = float(pat_loo_gap_c.sum())
    prod_seq_sum = float(prod_seq_gap_c.sum())

    top10_total = []
    for rank_i, c in enumerate(np.argsort(tot_3part_gap_c)[::-1][:10], start=1):
        c = int(c)
        act_m = x_sub[:, c]
        top10_total.append({
            "rank": rank_i,
            "category": c,
            "n_items_in_category": int((cat_of_np == c).sum()),
            "active_rate": round(float(active_rate_c[c]), 4),
            "multi_item_rate": round(float(multi_rate_c[c]), 4),
            "active_trips": int(act_m.sum()),
            "multi_item_trips": int((n_sub[:, c] >= 2).sum()),
            "total_gap_per_trip": round(float(tot_3part_gap_c[c]), 4),
            "share_of_total_category_gap_sum": round(float(tot_3part_gap_c[c] / tot_3part_sum), 4) if abs(tot_3part_sum) > 1e-12 else 0.0,
            "presence_gap_per_trip": round(float(pres_gap_c[c]), 4),
            "count_given_presence_gap_per_trip": round(float(cnt_seq_gap_c[c]), 4),
            "count_given_presence_loo_gap_per_trip": round(float(cnt_loo_gap_c[c]), 4),
            "product_given_counts_gap_per_trip": round(float(prod_seq_gap_c[c]), 4),
            "product_given_counts_marginal_gap_per_trip": round(float(prod_marg_gap_c[c]), 4),
        })

    top10_presence = []
    for rank_i, c in enumerate(np.argsort(pres_gap_c)[::-1][:10], start=1):
        c = int(c)
        act_m = x_sub[:, c]
        top10_presence.append({
            "rank": rank_i,
            "category": c,
            "active_rate": round(float(active_rate_c[c]), 4),
            "active_trips": int(act_m.sum()),
            "chain_loo_presence_ll": round(float(chain_loo_pres_bc[mask, c].mean()), 4),
            "v4_loo_presence_ll": round(float(v4_loo_pres_bc[mask, c].mean()), 4),
            "presence_gap_per_trip": round(float(pres_gap_c[c]), 4),
            "share_of_presence_gap_sum": round(float(pres_gap_c[c] / pres_sum), 4) if abs(pres_sum) > 1e-12 else 0.0,
        })

    top10_counts = []
    for rank_i, c in enumerate(np.argsort(cnt_seq_gap_c)[::-1][:10], start=1):
        c = int(c)
        act_m = x_sub[:, c]
        top10_counts.append({
            "rank": rank_i,
            "category": c,
            "active_rate": round(float(active_rate_c[c]), 4),
            "multi_item_rate": round(float(multi_rate_c[c]), 4),
            "active_trips": int(act_m.sum()),
            "multi_item_trips": int((n_sub[:, c] >= 2).sum()),
            "chain_count_given_presence_ll": round(float(chain_cnt_bc[mask, c].mean()), 4),
            "v4_count_given_presence_seq_ll": round(float(v4_cnt_seq_bc[mask, c].mean()), 4),
            "v4_count_given_presence_loo_ll": round(float(v4_cnt_loo_bc[mask, c].mean()), 4),
            "count_given_presence_gap_per_trip": round(float(cnt_seq_gap_c[c]), 4),
            "count_given_presence_loo_gap_per_trip": round(float(cnt_loo_gap_c[c]), 4),
        })

    top10_products = []
    for rank_i, c in enumerate(np.argsort(prod_seq_gap_c)[::-1][:10], start=1):
        c = int(c)
        act_m = x_sub[:, c]
        top10_products.append({
            "rank": rank_i,
            "category": c,
            "n_items_in_category": int((cat_of_np == c).sum()),
            "active_rate": round(float(active_rate_c[c]), 4),
            "active_trips": int(act_m.sum()),
            "chain_product_ll_per_trip": round(float(chain_prod_bc[mask, c].mean()), 4),
            "m3_product_ll_per_trip": round(float(v4_prod_seq_bc[mask, c].mean()), 4),
            "m3_product_marginal_ll_per_trip": round(float(v4_prod_marg_bc[mask, c].mean()), 4),
            "product_gap_per_trip": round(float(prod_seq_gap_c[c]), 4),
            "product_marginal_gap_per_trip": round(float(prod_marg_gap_c[c]), 4),
            "share_of_product_gap": round(float(prod_seq_gap_c[c] / prod_seq_sum), 4) if abs(prod_seq_sum) > 1e-12 else 0.0,
            "gap_per_active_instance": round(float(prod_seq_gap_bc[act_m, c].mean()), 4) if act_m.any() else 0.0,
        })

    top10_pattern = []
    for rank_i, c in enumerate(np.argsort(pat_loo_gap_c)[::-1][:10], start=1):
        c = int(c)
        act_m = x_sub[:, c]
        top10_pattern.append({
            "rank": rank_i,
            "category": c,
            "active_rate": round(float(active_rate_c[c]), 4),
            "active_trips": int(act_m.sum()),
            "chain_loo_pattern_ll": round(float(chain_loo_pat_bc[mask, c].mean()), 4),
            "m3_loo_pattern_ll": round(float(v4_loo_pat_bc[mask, c].mean()), 4),
            "pattern_gap_per_trip": round(float(pat_loo_gap_c[c]), 4),
            "share_of_loo_pattern_gap": round(float(pat_loo_gap_c[c] / pat_loo_sum), 4) if abs(pat_loo_sum) > 1e-12 else 0.0,
            "binary_presence_gap_per_trip": round(float(pres_gap_c[c]), 4),
            "count_1_given_presence_gap_per_trip": round(float(cnt_loo_gap_c[c]), 4),
            "gap_when_present": round(float(pat_loo_gap_bc[act_m, c].mean()), 4) if act_m.any() else 0.0,
            "gap_when_absent": round(float(pat_loo_gap_bc[~act_m, c].mean()), 4) if (~act_m).any() else 0.0,
        })

    return {
        "total_3part_category_gap_sum": round(tot_3part_sum, 4),
        "top10_total_sum": round(float(sum(x["total_gap_per_trip"] for x in top10_total)), 4),
        "top10_total_share": round(float(sum(x["total_gap_per_trip"] for x in top10_total) / tot_3part_sum), 4) if abs(tot_3part_sum) > 1e-12 else 0.0,
        "top10_categories_overall": top10_total,
        "presence_loo_gap_sum": round(pres_sum, 4),
        "top10_which_categories_present": top10_presence,
        "counts_given_presence_seq_gap_sum": round(cnt_seq_sum, 4),
        "top10_how_many_items_given_presence": top10_counts,
        "products_given_counts_seq_gap_sum": round(prod_seq_sum, 4),
        "top10_which_products_given_counts": top10_products,
        "category_pattern_loo_gap_sum": round(pat_loo_sum, 4),
        "top10_category_pattern": top10_pattern,
    }


def main():
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initialization", type=Path, default=A / "artifacts/initialization.pt")
    p.add_argument("--additive", type=Path, default=A / "out/v3_pipeline_additive_best.pt")
    p.add_argument("--m3-checkpoint", type=Path, default=ROOT / "runs/stress/hs_m4a.pt")
    p.add_argument("--m3-per-trip", type=Path, default=ROOT / "runs/stress/hs_m4a.per_trip.npz")
    p.add_argument("--chain-splits", type=Path, default=ROOT / "runs/stress/chain_eval_splits.npz")
    p.add_argument("--chain-run", type=Path, default=ROOT.parent / "basket_chain/runs/stress/fit_h64_p10")
    p.add_argument("--ref-test", type=Path, default=A / "reports/likelihood_test_paired_per_trip.npz")
    p.add_argument("--bundle", type=Path, default=ROOT / "data/stress_world/model_input_category")
    p.add_argument("--out-dir", type=Path, default=ROOT / "runs/stress/hs_m4a_gap_full")
    p.add_argument("--seed", type=int, default=41009)
    p.add_argument("--draws", type=int, default=1024)
    p.add_argument("--components", type=int, default=64)
    p.add_argument("--burn", type=int, default=50)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--skip-small-world", action="store_true", help="Skip world.py small-world enumeration check")
    args = p.parse_args()

    t_start = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out_dir / "run.log", "w")

    def emit(msg: str):
        print(msg, flush=True)
        log_file.write(msg + "\n")
        log_file.flush()

    stress.USE_CPP_SAMPLER = True
    emit(json.dumps({"event": "start", "out_dir": str(args.out_dir), "seed": args.seed, "draws": args.draws}))

    # ------------------------------------------------------------------
    # Step 1: Small-world brute-force check of log P(category pattern)
    # ------------------------------------------------------------------
    if not args.skip_small_world:
        t_sw = time.time()
        sw_check = run_small_world_check(emit=emit)
        sw_check["seconds"] = round(time.time() - t_sw, 2)
        (args.out_dir / "pattern_estimator_check.json").write_text(json.dumps(sw_check, indent=2) + "\n")
        emit(json.dumps({"event": "saved_pattern_estimator_check", "seconds": sw_check["seconds"]}))

    # ------------------------------------------------------------------
    # Step 2: Load stress-world models, reference test trips (1,887 + 2,209 = 4,096)
    # ------------------------------------------------------------------
    data, model, meta, batcher = setup(args.initialization, args.additive, args.threads)
    blob = torch.load(args.m3_checkpoint, map_location="cpu", weights_only=False)
    configure_model_from_blob(model, blob)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    r = int(blob["rank"])
    J, C, nmax = int(data["n_item"]), int(data["n_cat"]), int(meta["nmax"])

    ref_test = np.load(args.ref_test)
    m3_npz = np.load(args.m3_per_trip)
    chain_npz = np.load(args.chain_splits)
    trips_4096 = ref_test["trips"].astype(np.int64)
    assert np.array_equal(trips_4096, m3_npz["trips"])
    assert np.array_equal(trips_4096, chain_npz["test_trips_4096"])

    lptr = data["line_ptr"]
    line_cat = data["line_cat"]
    n_multi = np.zeros(len(trips_4096), dtype=np.int64)
    n_cats_all = np.zeros(len(trips_4096), dtype=np.int64)
    n_items_all = np.zeros(len(trips_4096), dtype=np.int64)
    for idx, t in enumerate(trips_4096):
        cats = line_cat[lptr[t]:lptr[t + 1]]
        u_cats, cnts = np.unique(cats, return_counts=True)
        n_cats_all[idx] = len(u_cats)
        n_items_all[idx] = len(cats)
        n_multi[idx] = int((cnts >= 2).sum())

    mask_1887 = n_multi == 0
    mask_2209 = n_multi > 0
    mask_all = np.ones(len(trips_4096), dtype=bool)
    n_trips_all = int(len(trips_4096))
    n_trips_1887 = int(mask_1887.sum())
    n_trips_2209 = int(mask_2209.sum())
    assert n_trips_1887 == 1887, f"Expected 1,887 one-item trips, got {n_trips_1887}"
    assert n_trips_2209 == 2209, f"Expected 2,209 multi-item trips, got {n_trips_2209}"

    # ------------------------------------------------------------------
    # Step 3: Category Chain exact 3-part decomposition on all 4,096 trips
    # ------------------------------------------------------------------
    bundle = load_bundle(args.bundle)
    chain_model, _ = load_chain_model(args.chain_run, bundle)

    with torch.no_grad():
        ctx = batch_context(bundle, trips_4096)
        mem = batch_members(bundle, trips_4096)
        b_ch = chain_model.utilities(ctx)
        log_a_ch, iv_ch = chain_model.category_terms(b_ch)  # [4096, C, R+1], [4096, C]
        n_ch = mem.sum(-1)  # [4096, C]
        x_ch = (n_ch > 0).to(torch.float64)
        base_ch = chain_model.chain_base(iv_ch, ctx["house"])
        eta, eta0 = chain_model.chain_logits(base_ch, x_ch)

        log_p0_ch = logsigmoid(-eta0).sum(-1)
        trunc_corr_ch = -torch.log(-torch.expm1(log_p0_ch))
        chain_pres_ll = ((x_ch * logsigmoid(eta) + (1.0 - x_ch) * logsigmoid(-eta)).sum(-1) + trunc_corr_ch).numpy()

        n_ch_clamped = n_ch.long().clamp(max=chain_model.R)
        log_a_obs = log_a_ch.gather(2, n_ch_clamped.unsqueeze(-1)).squeeze(-1)
        chain_cnt_bc = (x_ch * (log_a_obs - iv_ch)).numpy()
        chain_cnt_ll = chain_cnt_bc.sum(axis=1)
        chain_pattern_ll = chain_pres_ll + chain_cnt_ll

        pair_pen_ch = chain_model.rho[None, :] * (0.5 * n_ch * (n_ch - 1.0))
        chain_prod_bc = (x_ch * ((b_ch * mem).sum(-1) - pair_pen_ch - log_a_obs)).numpy()
        chain_prod_ll = chain_prod_bc.sum(axis=1)
        chain_total_ll = chain_pres_ll + chain_cnt_ll + chain_prod_ll

        # Category Chain leave-one-category-out (LOO) presence & pattern conditionals across all 4,096 trips
        chain_loo_pat_bc = np.zeros((n_trips_all, C), dtype=np.float64)
        chain_loo_pres_bc = np.zeros((n_trips_all, C), dtype=np.float64)
        log_p_x_obs = (x_ch * logsigmoid(eta) + (1.0 - x_ch) * logsigmoid(-eta)).sum(-1)
        for c in range(C):
            x_alt = x_ch.clone()
            x_alt[:, c] = 1.0 - x_ch[:, c]
            eta_alt, _ = chain_model.chain_logits(base_ch, x_alt)
            k_other = x_ch.sum(-1) - x_ch[:, c]
            log_p_x_alt = (x_alt * logsigmoid(eta_alt) + (1.0 - x_alt) * logsigmoid(-eta_alt)).sum(-1)
            log_p_x_alt = torch.where(k_other == 0, torch.full_like(log_p_x_alt, -1e30), log_p_x_alt)
            log_p_xc_cond = log_p_x_obs - torch.logaddexp(log_p_x_obs, log_p_x_alt)
            chain_loo_pres_bc[:, c] = log_p_xc_cond.numpy()
            chain_loo_pat_bc[:, c] = log_p_xc_cond.numpy() + chain_cnt_bc[:, c]

    chain_saved_all = chain_npz["chain_test_4096"]
    chain_mismatch_all = np.abs(chain_total_ll - chain_saved_all)

    # ------------------------------------------------------------------
    # Step 4: Version-4 3-part decomposition on all 4,096 trips
    # ------------------------------------------------------------------
    gen = torch.Generator().manual_seed(args.seed)
    e_obs_list = []
    est_zx_list, est_zn_list = [], []
    hdiff_zx_list, hdiff_zn_list = [], []
    ess_zx_list, ess_zn_list = [], []
    v4_cnt_seq_list, v4_cnt_loo_list = [], []
    v4_prod_seq_list, v4_prod_marg_list, v4_prod_b_list = [], [], []
    v4_loo_pat_list, v4_loo_pres_list = [], []

    # Diagnostic on unconditional proposal ESS (on first 256 one-item trips)
    bt_1887_0 = Batch(batcher, trips_4096[mask_1887][:256], J)
    bt_1887_0.bind(model)
    uncond_ess_diag = compute_unconditional_proposal_ess_diagnostic(
        model, bt_1887_0, r, torch.Generator().manual_seed(args.seed + 77)
    )

    for s in range(0, n_trips_all, 256):
        bt = Batch(batcher, trips_4096[s:s + 256], J)
        bt.bind(model)
        e_obs = obs_energy(model, bt, r).detach().numpy()
        (
            est_zx,
            est_zn,
            hdiff_zx,
            hdiff_zn,
            ess_zx,
            ess_zn,
            cnt_seq_bc,
            cnt_loo_bc,
            prod_seq_bc,
            prod_marg_bc,
            prod_b_only_bc,
            loo_pat_bc,
            loo_pres_bc,
        ) = estimate_v4_three_part_batch(
            model, bt, r, gen, draws=args.draws, components=args.components, burn=args.burn
        )
        e_obs_list.append(e_obs)
        est_zx_list.append(est_zx.numpy())
        est_zn_list.append(est_zn.numpy())
        hdiff_zx_list.append(hdiff_zx.numpy())
        hdiff_zn_list.append(hdiff_zn.numpy())
        ess_zx_list.append(ess_zx.numpy())
        ess_zn_list.append(ess_zn.numpy())
        v4_cnt_seq_list.append(cnt_seq_bc.numpy())
        v4_cnt_loo_list.append(cnt_loo_bc.numpy())
        v4_prod_seq_list.append(prod_seq_bc.numpy())
        v4_prod_marg_list.append(prod_marg_bc.numpy())
        v4_prod_b_list.append(prod_b_only_bc.numpy())
        v4_loo_pat_list.append(loo_pat_bc.numpy())
        v4_loo_pres_list.append(loo_pres_bc.numpy())
        emit(
            json.dumps({
                "event": "v4_batch",
                "done": s + len(bt.trips),
                "total": n_trips_all,
                "ess_zx_mean": round(float(ess_zx.mean()), 4),
                "ess_zn_mean": round(float(ess_zn.mean()), 4),
                "seconds": round(time.time() - t_start, 1),
            })
        )

    e_obs_all = np.concatenate(e_obs_list)
    est_zx_all = np.concatenate(est_zx_list)
    est_zn_all = np.concatenate(est_zn_list)
    hdiff_zx_all = np.concatenate(hdiff_zx_list)
    hdiff_zn_all = np.concatenate(hdiff_zn_list)
    ess_zx_all = np.concatenate(ess_zx_list)
    ess_zn_all = np.concatenate(ess_zn_list)
    v4_cnt_seq_bc = np.concatenate(v4_cnt_seq_list, axis=0)
    v4_cnt_loo_bc = np.concatenate(v4_cnt_loo_list, axis=0)
    v4_prod_seq_bc = np.concatenate(v4_prod_seq_list, axis=0)
    v4_prod_marg_bc = np.concatenate(v4_prod_marg_list, axis=0)
    v4_prod_b_bc = np.concatenate(v4_prod_b_list, axis=0)
    v4_loo_pat_bc = np.concatenate(v4_loo_pat_list, axis=0)
    v4_loo_pres_bc = np.concatenate(v4_loo_pres_list, axis=0)

    v4_saved_ll_all = m3_npz["ll"]
    v4_saved_ess_all = m3_npz["ess"]
    v4_saved_hdiff_all = m3_npz["half_diff"]
    est_z_saved_all = e_obs_all - v4_saved_ll_all

    # Exact 3-part split of Version-4 log-likelihood on every trip:
    #   Part 1 (which categories are present): log Z(x_obs) - log Z
    #   Part 2 (how many items | presence):    log Z(n_obs) - log Z(x_obs)
    #   Part 3 (which products | counts):      E(S_obs) - log Z(n_obs)
    v4_pres_ll = est_zx_all - est_z_saved_all
    v4_cnt_ll = est_zn_all - est_zx_all
    v4_pattern_ll = est_zn_all - est_z_saved_all
    v4_prod_ll = e_obs_all - est_zn_all
    v4_total_ll = v4_pres_ll + v4_cnt_ll + v4_prod_ll

    v4_exact_mismatch_all = np.abs(v4_total_ll - v4_saved_ll_all)
    v4_cnt_seq_mismatch_all = np.abs(v4_cnt_seq_bc.sum(axis=1) - v4_cnt_ll)
    v4_prod_marg_sum_ll = v4_prod_marg_bc.sum(axis=1)
    v4_prod_b_sum_ll = v4_prod_b_bc.sum(axis=1)
    v4_cnt_loo_sum_ll = v4_cnt_loo_bc.sum(axis=1)

    assert float(chain_mismatch_all.max()) <= 1e-10, f"Chain mismatch {chain_mismatch_all.max()}"
    assert float(v4_exact_mismatch_all.max()) <= 1e-10, f"V4 mismatch {v4_exact_mismatch_all.max()}"
    assert float(v4_cnt_seq_mismatch_all.max()) <= 1e-10, f"V4 cnt_seq mismatch {v4_cnt_seq_mismatch_all.max()}"

    # ------------------------------------------------------------------
    # Step 5: Exact Truth Ceiling via latent marginalisation on all 4,096 trips
    # ------------------------------------------------------------------
    t_tc = time.time()
    truth_res = compute_truth_ceiling(trips_4096, args.out_dir / "pre_trip_state.npz", args.out_dir / "truth_per_trip.npz")
    tc_seconds = round(time.time() - t_tc, 2)
    emit(json.dumps({
        "event": "truth_ceiling_done",
        "truth_total_ll_4096": round(float(truth_res["truth_total_ll"].mean()), 4),
        "mobius_max_rel_diff": float(truth_res["mobius_max_rel_diff"]),
        "seconds": tc_seconds,
    }))

    # Save per-trip arrays for all 4,096 trips
    np.savez(
        args.out_dir / "per_trip.npz",
        trips=trips_4096,
        mask_1887=mask_1887,
        mask_2209=mask_2209,
        n_cats=n_cats_all,
        n_items=n_items_all,
        n_multi_cats=n_multi,
        chain_total_ll=chain_total_ll,
        chain_pattern_ll=chain_pattern_ll,
        chain_presence_ll=chain_pres_ll,
        chain_count_given_presence_ll=chain_cnt_ll,
        chain_product_ll=chain_prod_ll,
        m3_total_ll=v4_total_ll,
        m3_pattern_ll=v4_pattern_ll,
        m3_presence_ll=v4_pres_ll,
        m3_count_given_presence_ll=v4_cnt_ll,
        m3_product_ll=v4_prod_ll,
        m3_product_marginal_sum_ll=v4_prod_marg_sum_ll,
        m3_obs_energy=e_obs_all,
        m3_log_z=est_z_saved_all,
        m3_log_z_presence=est_zx_all,
        m3_log_z_pattern=est_zn_all,
        m3_is_ess_log_z=v4_saved_ess_all,
        m3_is_half_diff_log_z=v4_saved_hdiff_all,
        m3_is_ess_log_z_presence=ess_zx_all,
        m3_is_half_diff_log_z_presence=hdiff_zx_all,
        m3_is_ess_log_z_pattern=ess_zn_all,
        m3_is_half_diff_log_z_pattern=hdiff_zn_all,
        truth_total_ll=truth_res["truth_total_ll"],
        truth_presence_ll=truth_res["truth_presence_ll"],
        truth_count_given_presence_ll=truth_res["truth_count_given_presence_ll"],
        truth_product_given_counts_ll=truth_res["truth_product_given_counts_ll"],
    )

    # ------------------------------------------------------------------
    # Step 6: Build 3-part splits and top-10 category tables for 2209, 1887, and all 4096
    # ------------------------------------------------------------------
    n_ch_np = n_ch.numpy()
    cat_of_np = bundle.item_cat

    split_1887 = build_split_summary(
        mask_1887, n_cats_all, n_items_all,
        chain_total_ll, chain_pres_ll, chain_cnt_ll, chain_prod_ll,
        v4_total_ll, v4_pres_ll, v4_cnt_ll, v4_prod_ll, v4_prod_marg_sum_ll, v4_prod_b_sum_ll,
        truth_dict=truth_res, v4_cnt_loo_sum=v4_cnt_loo_sum_ll,
    )
    split_2209 = build_split_summary(
        mask_2209, n_cats_all, n_items_all,
        chain_total_ll, chain_pres_ll, chain_cnt_ll, chain_prod_ll,
        v4_total_ll, v4_pres_ll, v4_cnt_ll, v4_prod_ll, v4_prod_marg_sum_ll, v4_prod_b_sum_ll,
        truth_dict=truth_res,
    )
    split_4096 = build_split_summary(
        mask_all, n_cats_all, n_items_all,
        chain_total_ll, chain_pres_ll, chain_cnt_ll, chain_prod_ll,
        v4_total_ll, v4_pres_ll, v4_cnt_ll, v4_prod_ll, v4_prod_marg_sum_ll, v4_prod_b_sum_ll,
        truth_dict=truth_res,
    )

    cat_attr_2209 = build_top10_categories_for_subset(
        mask_2209, n_ch_np, cat_of_np,
        chain_loo_pres_bc, v4_loo_pres_bc, chain_loo_pat_bc, v4_loo_pat_bc,
        chain_cnt_bc, v4_cnt_seq_bc, v4_cnt_loo_bc,
        chain_prod_bc, v4_prod_seq_bc, v4_prod_marg_bc,
    )
    cat_attr_1887 = build_top10_categories_for_subset(
        mask_1887, n_ch_np, cat_of_np,
        chain_loo_pres_bc, v4_loo_pres_bc, chain_loo_pat_bc, v4_loo_pat_bc,
        chain_cnt_bc, v4_cnt_seq_bc, v4_cnt_loo_bc,
        chain_prod_bc, v4_prod_seq_bc, v4_prod_marg_bc,
    )
    cat_attr_4096 = build_top10_categories_for_subset(
        mask_all, n_ch_np, cat_of_np,
        chain_loo_pres_bc, v4_loo_pres_bc, chain_loo_pat_bc, v4_loo_pat_bc,
        chain_cnt_bc, v4_cnt_seq_bc, v4_cnt_loo_bc,
        chain_prod_bc, v4_prod_seq_bc, v4_prod_marg_bc,
    )

    # ------------------------------------------------------------------
    # Step 7: Chain ablations & parameter/taste diagnostics across splits
    # ------------------------------------------------------------------
    with torch.no_grad():
        u_term = chain_model.theta[ctx["house"]] @ chain_model.U.T
        tau_sp = softplus(chain_model.tau)

        def score_chain_ablation_vec(tau_override=None, zero_U=False, zero_cross=False):
            cross = chain_model.cross_p @ chain_model.cross_q.T
            cross = cross - torch.diag(torch.diag(cross))
            base = base_ch.clone()
            if tau_override is not None:
                base = base + (tau_override - tau_sp)[None, :] * (iv_ch - iv_ch.mean(0, keepdim=True))
            if zero_U:
                base = base - (u_term - u_term.mean(0, keepdim=True))
            if zero_cross:
                c_term = iv_ch @ cross.T
                base = base - (c_term - c_term.mean(0, keepdim=True))
            eta_ab, eta0_ab = chain_model.chain_logits(base, x_ch)
            log_p0_ab = logsigmoid(-eta0_ab).sum(-1)
            pres_ab = (x_ch * logsigmoid(eta_ab) + (1.0 - x_ch) * logsigmoid(-eta_ab)).sum(-1) - torch.log(
                -torch.expm1(log_p0_ab)
            )
            return pres_ab.numpy()

        pres_full_vec = score_chain_ablation_vec()
        pres_tau1_vec = score_chain_ablation_vec(tau_override=1.0)
        pres_no_u_vec = score_chain_ablation_vec(zero_U=True)
        pres_both_vec = score_chain_ablation_vec(tau_override=1.0, zero_U=True)
        pres_no_cross_vec = score_chain_ablation_vec(zero_cross=True)

        ta_v4 = (model.theta_c() @ model.alpha.T).numpy()
        if getattr(model, "u_cat", None) is not None:
            u_v4 = (model.theta_c() @ model.u_cat.T).numpy()
            ta_v4_tot = ta_v4 + u_v4[:, cat_of_np]
            u_v4_between_cat_var = float(u_v4.var(0).mean())
        else:
            ta_v4_tot = ta_v4
            u_v4_between_cat_var = 0.0
        ta_ch = (chain_model.theta @ chain_model.alpha.T).numpy()
        ta_v4_between_cat_var = float(np.mean([ta_v4_tot[:, cat_of_np == c].mean(1).var() for c in range(C)]))
        ta_v4_within_cat_var = float(
            np.mean([
                (ta_v4_tot[:, cat_of_np == c] - ta_v4_tot[:, cat_of_np == c].mean(1, keepdims=True)).var()
                for c in range(C)
            ])
        )
        ta_ch_between_cat_var = float(np.mean([ta_ch[:, cat_of_np == c].mean(1).var() for c in range(C)]))
        ta_ch_within_cat_var = float(
            np.mean([
                (ta_ch[:, cat_of_np == c] - ta_ch[:, cat_of_np == c].mean(1, keepdims=True)).var()
                for c in range(C)
            ])
        )
        u_ch_between_cat_var = float((chain_model.theta @ chain_model.U.T).numpy().var(0).mean())
        g_v4_mean = float((softplus(model.gamma).mean(0) * softplus(model.beta)).sum(-1).mean())
        g_ch_mean = float((torch.exp(chain_model.gamma).mean() * softplus(chain_model.beta)).mean())

    # ------------------------------------------------------------------
    # Step 8: Write truth_ceiling.json and summary.json
    # ------------------------------------------------------------------
    truth_ceiling_report = {
        "numerical_verification": {
            "mobius_vs_positive_tuples_max_rel_diff": float(truth_res["mobius_max_rel_diff"]),
            "per_trip_three_part_sum_max_abs_err": float(
                np.abs(
                    (truth_res["truth_presence_ll"] + truth_res["truth_count_given_presence_ll"] + truth_res["truth_product_given_counts_ll"])
                    - truth_res["truth_total_ll"]
                ).max()
            ),
        },
        "overall_4096_trips": split_4096["truth_ceiling"],
        "split_1887_one_item_trips": split_1887["truth_ceiling"],
        "split_2209_multi_item_trips": split_2209["truth_ceiling"],
        "generator_mechanism_ablations_4096_trips": {
            "oracle_dynamic_state_contribution": {
                "full_truth_total_ll": round(float(truth_res["truth_total_ll"].mean()), 4),
                "static_state_truth_total_ll": round(float(truth_res["static_truth_total_ll"].mean()), 4),
                "dynamic_state_headroom_total": round(
                    float((truth_res["truth_total_ll"] - truth_res["static_truth_total_ll"]).mean()), 4
                ),
                "dynamic_inventory_headroom_on_presence": round(
                    float((truth_res["truth_presence_ll"] - truth_res["static_truth_presence_ll"]).mean()), 4
                ),
                "dynamic_loyalty_taste_stockout_headroom_on_products": round(
                    float((truth_res["truth_product_given_counts_ll"] - truth_res["static_truth_product_given_counts_ll"]).mean()), 4
                ),
            },
            "static_category_presence_headroom_4096": {
                "static_truth_presence_ll": round(float(truth_res["static_truth_presence_ll"].mean()), 4),
                "chain_presence_ll": round(float(chain_pres_ll.mean()), 4),
                "v4_presence_ll": round(float(v4_pres_ll.mean()), 4),
                "static_truth_minus_v4_presence": round(
                    float(truth_res["static_truth_presence_ll"].mean() - v4_pres_ll.mean()), 4
                ),
                "static_truth_minus_chain_presence": round(
                    float(truth_res["static_truth_presence_ll"].mean() - chain_pres_ll.mean()), 4
                ),
                "truth_presence_drop_when_removing_household_category_taste": round(
                    float((truth_res["truth_presence_ll"] - truth_res["no_hh_cat_taste_presence_ll"]).mean()), 4
                ),
                "chain_presence_drop_when_zeroing_U_4096": round(float((pres_full_vec - pres_no_u_vec).mean()), 4),
                "chain_presence_drop_when_zeroing_U_2209": round(
                    float((pres_full_vec[mask_2209] - pres_no_u_vec[mask_2209]).mean()), 4
                ),
                "chain_presence_drop_when_forcing_unit_iv_slope_tau_eq_1_4096": round(
                    float((pres_full_vec - pres_tau1_vec).mean()), 4
                ),
                "chain_presence_drop_when_forcing_unit_iv_slope_tau_eq_1_2209": round(
                    float((pres_full_vec[mask_2209] - pres_tau1_vec[mask_2209]).mean()), 4
                ),
                "chain_presence_drop_when_zeroing_cross_iv_4096": round(
                    float((pres_full_vec - pres_no_cross_vec).mean()), 4
                ),
            },
        },
        "headroom_comparison_step_2_vs_step_3": {
            "truth_minus_v4_category_presence_4096": round(
                float(truth_res["truth_presence_ll"].mean() - v4_pres_ll.mean()), 4
            ),
            "static_truth_minus_v4_category_presence_4096": round(
                float(truth_res["static_truth_presence_ll"].mean() - v4_pres_ll.mean()), 4
            ),
            "step_3_truth_household_category_taste_headroom_4096": round(
                float((truth_res["truth_presence_ll"] - truth_res["no_hh_cat_taste_presence_ll"]).mean()), 4
            ),
            "step_3_chain_household_category_taste_U_headroom_4096": round(
                float((pres_full_vec - pres_no_u_vec).mean()), 4
            ),
            "step_2_chain_iv_slope_tau_headroom_4096": round(
                float((pres_full_vec - pres_tau1_vec).mean()), 4
            ),
            "v4_dedicated_theta_U_cat_var": round(u_v4_between_cat_var, 4),
            "chain_dedicated_theta_U_cat_var": round(u_ch_between_cat_var, 4),
        },
    }
    (args.out_dir / "truth_ceiling.json").write_text(json.dumps(truth_ceiling_report, indent=2) + "\n")

    summary = {
        "command": " ".join([sys.executable] + sys.argv),
        "test_trips_all": n_trips_all,
        "test_trips_one_item_per_cat": n_trips_1887,
        "test_trips_multi_item_per_cat": n_trips_2209,
        "share_of_test_trips_one_item": round(n_trips_1887 / n_trips_all, 4),
        "share_of_test_trips_multi_item": round(n_trips_2209 / n_trips_all, 4),
        "mean_basket_size_1887": round(float(n_items_all[mask_1887].mean()), 4),
        "mean_basket_size_2209": round(float(n_items_all[mask_2209].mean()), 4),
        "mean_basket_size_4096": round(float(n_items_all.mean()), 4),
        "verification": {
            "chain_vs_saved_mean_abs_mismatch_4096": float(chain_mismatch_all.mean()),
            "chain_vs_saved_max_abs_mismatch_4096": float(chain_mismatch_all.max()),
            "v4_exact_3part_vs_saved_mean_abs_mismatch_4096": float(v4_exact_mismatch_all.mean()),
            "v4_exact_3part_vs_saved_max_abs_mismatch_4096": float(v4_exact_mismatch_all.max()),
            "v4_sequential_count_sum_vs_saved_max_abs_mismatch_4096": float(v4_cnt_seq_mismatch_all.max()),
            "all_4096_trips_sum_within_1e_10": bool(
                chain_mismatch_all.max() <= 1e-10 and v4_exact_mismatch_all.max() <= 1e-10
            ),
            "v4_is_log_z_ess_mean_4096": round(float(v4_saved_ess_all.mean()), 4),
            "v4_is_log_z_ess_min_4096": round(float(v4_saved_ess_all.min()), 4),
            "v4_is_log_z_half_split_abs_mean_4096": round(float(np.abs(v4_saved_hdiff_all).mean()), 4),
            "v4_is_log_z_half_split_abs_max_4096": round(float(np.abs(v4_saved_hdiff_all).max()), 4),
            "v4_is_log_z_presence_ess_mean_4096": round(float(ess_zx_all.mean()), 4),
            "v4_is_log_z_presence_ess_min_4096": round(float(ess_zx_all.min()), 4),
            "v4_is_log_z_pattern_ess_mean_4096": round(float(ess_zn_all.mean()), 4),
            "v4_is_log_z_pattern_ess_min_4096": round(float(ess_zn_all.min()), 4),
            "v4_is_log_z_pattern_half_split_abs_mean_4096": round(float(np.abs(hdiff_zn_all).mean()), 4),
            "v4_is_log_z_pattern_half_split_abs_max_4096": round(float(np.abs(hdiff_zn_all).max()), 4),
            "unconditional_proposal_diagnostic": uncond_ess_diag,
        },
        "gap_split_2209_trips": split_2209,
        "gap_split_1887_trips": split_1887,
        "gap_split_all_4096_trips": split_4096,
        "per_category_attribution_2209_trips": cat_attr_2209,
        "per_category_attribution_1887_trips": cat_attr_1887,
        "per_category_attribution_all_4096_trips": cat_attr_4096,
        "truth_ceiling_summary": truth_ceiling_report,
        "parameter_and_utility_distortion_diagnostics": {
            "chain_softplus_tau_mean": round(float(tau_sp.mean()), 4),
            "chain_softplus_tau_min": round(float(tau_sp.min()), 4),
            "chain_softplus_tau_max": round(float(tau_sp.max()), 4),
            "v4_theta_alpha_between_cat_mean_var": round(ta_v4_between_cat_var, 4),
            "v4_theta_alpha_within_cat_var": round(ta_v4_within_cat_var, 4),
            "chain_theta_alpha_between_cat_mean_var": round(ta_ch_between_cat_var, 4),
            "chain_theta_alpha_within_cat_var": round(ta_ch_within_cat_var, 4),
            "chain_dedicated_theta_U_cat_var": round(u_ch_between_cat_var, 4),
            "v4_dedicated_theta_U_cat_var": round(u_v4_between_cat_var, 4),
            "v4_mean_price_slope_frozen": round(g_v4_mean, 4),
            "chain_mean_price_slope": round(g_ch_mean, 4),
            "v4_mean_display_weight": round(float(model.w_dsp.mean()), 4),
            "chain_mean_display_weight": round(float(chain_model.w_dsp.mean()), 4),
        },
        "seconds": round(time.time() - t_start, 1),
    }

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    emit(json.dumps({
        "event": "done",
        "gap_2209_total": split_2209["total_gap_nats_per_trip"],
        "gap_2209_three_parts": split_2209["three_part_split"],
        "gap_4096_total": split_4096["total_gap_nats_per_trip"],
        "gap_4096_three_parts": split_4096["three_part_split"],
        "seconds": summary["seconds"],
    }))
    log_file.close()


if __name__ == "__main__":
    main()
