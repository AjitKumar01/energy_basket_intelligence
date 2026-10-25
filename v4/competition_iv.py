"""Instrumental-variables (Wald IV / 2SLS) estimator for within-category competition.

Why likelihood struggles to identify ``comp_raw`` (lambda_c = sigmoid(comp_raw_c)):
1. In the full basket energy, the competition term shifts each category's count energy by
   n_c * (lambda_c - 1) * Delta_IV_tc, where Delta_IV_tc = IV_tc(actual) - IV_tc(regular).
   Variation in category counts across trips is overwhelmingly driven by static household
   tastes, latent mission factors z, and count potentials (rho_c, f_extra, rho_0); the
   price-induced inclusive-value shift accounts for less than 0.002 nats per trip of
   log-likelihood across an 18-fold range of lambda_c.
2. Delta_IV_tc is evaluated at household-specific utilities b_tj, which include endogenous
   purchase history (rec, rec_item) and household tastes.  A household with recent purchases
   in category c both buys category c more often and has higher weight on its promoted items,
   creating a spurious positive correlation between n_tc and Delta_IV_tc even when price
   discounts cause zero category expansion.

How the Wald IV instrument estimator identifies lambda_c:
1. Exogenous price instrument Z_tc:
   For each offered slot j on trip t in category c, we compute a household- and history-
   invariant baseline share within the category from product popularity alone,
       w_tj^(0) = exp(lam_j) / sum_{k in c_t} exp(lam_k),
   and a population-mean price utility shock
       d_tj = -g_bar_j * [m_t + kappa * (dlp_tj - m_t) + (kappa_dsc - kappa) * dsc_tj]
              - disc_common * dsc_tj,
   where g_bar_j = mean_h(softplus(gamma_h)) . softplus(beta_j).
   Display and mailer are left out of the instrument.  Their weights are free per product and
   fitted on the same trips, so a product displayed in weeks it did not sell gets a strongly
   negative weight: weighted by them, display tracks the outcome and pushes lambda up
   (stress worlds runs/disc_c48_s{1,2,3} on v4-scale, truth 0.05: final lambda 0.17, 0.04,
   0.40 with them, 0.19, 0.01, 0.13 with the weights set to zero).  They still count in
   Delta_IV, so lambda_c applies to every price and promotion change.
   The raw instrument is the baseline-share-weighted price shock Z_tc^raw = sum_{j in c_t} w_tj^(0) d_tj.
   We double-centre Z_tc^raw by subtracting its category mean across training rows and its
   trip mean across offered categories on trip t:
       Z_tc = (Z_tc^raw - mean_t(Z_tc^raw)) - mean_{c' on trip t}(Z_tc'^raw - mean_t(Z_tc'^raw)).
   Because Z_tc depends only on store-week price/promotion variations and static product
   parameters, it is orthogonal to household taste, purchase history, and chain-wide
   store-week price inflation.

2. Exact Wald IV / 2SLS first stage via the model's exact conditional DP:
   For each training row (t, c), we evaluate the model's expected category count under
   no competition correction (lambda = 1, slot utilities b_base) and under pure switching
   (lambda = 0, slot utilities b_base - Delta_IV_tc) using the exact DP in v4/exact.py:
       n_tc^(1) = E[n_tc | lambda = 1, z = 0],
       n_tc^(0) = E[n_tc | lambda = 0, z = 0].
   Since E[n_tc | lambda_c] - n_tc^(0) approx lambda_c * (n_tc^(1) - n_tc^(0)) and Z_tc is
   mean-zero and orthogonal to regular-price demand, projecting onto Z_tc gives the Wald IV
   moment condition:
       N_c = sum_t n_tc^obs * Z_tc,
       D_c = sum_t (n_tc^(1) - n_tc^(0)) * Z_tc.
   Here N_c is the empirical price-induced category expansion covarying with Z_tc, and D_c
   is the model's exact first-stage category expansion at lambda_c = 1 covarying with Z_tc.

3. Empirical-Bayes shrinkage across categories:
   Individual category IV ratios N_c / D_c are noisy when a category has few promotions, so
   we shrink each category toward the pooled Wald IV ratio
       lambda_bar = clip(sum_c N_c / sum_c D_c, 0.01, 0.99)
   with prior weight tau_0 = prior_strength * mean_c(D_c):
       lambda_hat_c = clip((N_c + tau_0 * lambda_bar) / (D_c + tau_0), 0.01, 0.99).
   The pooled ratio's standard error is the category jackknife: leave out one category's
   N_c and D_c at a time (jackknife_se).
   When the true world has pure switching (lambda = 0), N_c approx 0 and lambda_hat_c is
   near 0; when price cuts grow category counts by a fraction lambda of the full additive
   expansion, N_c approx lambda * D_c and lambda_hat_c recovers lambda without attenuation.
"""
from __future__ import annotations

import torch
from torch.nn.functional import softplus

from .exact import incidence as exact_incidence
from .model import PRICE_KEYS, _row_logsumexp


def jackknife_se(num, den) -> float:
    """Category-jackknife standard error of the pooled ratio sum_c num_c / sum_c den_c."""
    C = num.numel()
    if C < 2:
        return float("nan")
    loo = (num.sum() - num) / (den.sum() - den)
    return float(((C - 1) / C * ((loo - loo.mean()) ** 2).sum()).sqrt())


@torch.no_grad()
def competition_moments(model, batches, batcher):
    """The Wald IV moments per category, N_c = sum_t n_tc^obs Z_tc and
    D_c = sum_t (n_tc^(1) - n_tc^(0)) Z_tc, over ``batches`` (a list of trip-ID arrays)."""
    C = model.C
    g_bar = (softplus(model.gamma).mean(0, keepdim=True) * softplus(model.beta)).sum(-1)
    kappa = softplus(model.price_kappa)
    disc_k = softplus(model.disc_kappa) if model.disc_kappa is not None else kappa
    disc_c = model.disc_common if model.disc_common is not None else torch.tensor(0.0)

    rows_z, rows_c, rows_t, rows_y, rows_dm = [], [], [], [], []
    trip_offset = 0
    for trips in batches:
        bt = batcher.make(trips)
        bt.bind(model)
        ix, c = bt.ix, bt.ctx
        it, tr = ix.item, ix.item_trip

        # 1. Exogenous household-invariant category price/promotion shock Z_tc^raw.
        u0 = model.lam[it]
        m0 = torch.full((ix.n_rows,), -float("inf")).index_reduce_(0, ix.row_of, u0, "amax", include_self=True)
        w0 = torch.exp(u0 - m0[ix.row_of])
        z0 = torch.zeros(ix.n_rows).index_add_(0, ix.row_of, w0).clamp_min(1e-300)
        share0 = w0 / z0[ix.row_of]

        mean = c["dlp_bar"][tr]
        d = -g_bar[it] * (mean + kappa * (c["dlp"] - mean))
        if "dsc" in c and model.disc_kappa is not None:
            d = d - g_bar[it] * (disc_k - kappa) * c["dsc"]
        if "dsc" in c and model.disc_common is not None:
            d = d - disc_c * c["dsc"]
        z_row = torch.zeros(ix.n_rows).index_add_(0, ix.row_of, share0 * d)
        y_row = torch.zeros(ix.n_rows).index_add_(0, ix.row_of, bt.observed.double())

        # 2. Exact structural sensitivity n_tc^(1) - n_tc^(0) from the per-category DP.
        raw = model.utilities(it, tr, c)
        o = model.history_offset(ix, raw)
        b_base = raw if o is None else raw - o[tr]
        reg_ctx = {k: (torch.zeros_like(v) if k in PRICE_KEYS else v) for k, v in c.items()}
        b_reg = model.utilities(it, tr, reg_ctx)
        div_row = _row_logsumexp(ix, raw) - _row_logsumexp(ix, b_reg)

        _, p1 = exact_incidence(model, ix, b_base, z=None)
        _, p0 = exact_incidence(model, ix, b_base - div_row[ix.row_of], z=None)
        dm_row = torch.zeros(ix.n_rows).index_add_(0, ix.row_of, p1 - p0)

        rows_z.append(z_row)
        rows_c.append(ix.row_cat)
        rows_t.append(ix.row_trip + trip_offset)
        rows_y.append(y_row)
        rows_dm.append(dm_row)
        trip_offset += ix.B

    z_all = torch.cat(rows_z)
    c_all = torch.cat(rows_c)
    t_all = torch.cat(rows_t)
    y_all = torch.cat(rows_y)
    dm_all = torch.cat(rows_dm)

    # Double-centre Z_tc by category and by trip.
    cat_cnt = torch.bincount(c_all, minlength=C).double().clamp_min(1.0)
    cat_mean = torch.zeros(C).index_add_(0, c_all, z_all) / cat_cnt
    z_c = z_all - cat_mean[c_all]
    trip_cnt = torch.bincount(t_all, minlength=trip_offset).double().clamp_min(1.0)
    trip_mean = torch.zeros(trip_offset).index_add_(0, t_all, z_c) / trip_cnt
    z_tilde = z_c - trip_mean[t_all]

    num = torch.zeros(C).index_add_(0, c_all, y_all * z_tilde)
    den = torch.zeros(C).index_add_(0, c_all, dm_all * z_tilde).clamp_min(1e-6)
    return num, den


@torch.no_grad()
def estimate_competition_iv(model, batches, batcher, *, prior_strength: float = 10.0) -> dict:
    """Estimate ``model.comp_raw`` in place using the exogenous price-instrument Wald IV
    estimator over ``batches`` (a list of trip-ID arrays).

    Returns a dictionary of summary diagnostics (pooled lambda, its raw value and category-
    jackknife standard error, per-category mean/min/max).
    """
    if model.comp_raw is None:
        raise ValueError("estimate_competition_iv requires model.comp_raw to be initialised")
    num, den = competition_moments(model, batches, batcher)

    raw_pool = num.sum() / den.sum()
    lam_pool = raw_pool.clamp(0.01, 0.99)
    tau0 = float(prior_strength) * den.mean().clamp_min(1e-4)
    lam_c = ((num + tau0 * lam_pool) / (den + tau0)).clamp(0.01, 0.99)
    model.comp_raw.copy_(torch.logit(lam_c))

    return {
        "comp_lambda_pool": round(float(lam_pool), 4),
        "comp_lambda_pool_raw": round(float(raw_pool), 4),
        "comp_lambda_pool_se": round(jackknife_se(num, den), 4),
        "comp_lambda_mean": round(float(lam_c.mean()), 4),
        "comp_lambda_min": round(float(lam_c.min()), 4),
        "comp_lambda_max": round(float(lam_c.max()), 4),
    }
