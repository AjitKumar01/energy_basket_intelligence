"""Objectives and projections for training the full Version-4 model.

The model trained here has every Version-4 block: count potentials f_c (M1), tied
interactions phi_j = psi_c(j) + delta_j with presence coupling (M2), the size potential on
categories present (M3), household taste per category (M4a); optionally purchase history
and M4c.  Two objectives:

  pseudo-likelihood   sum over slots of log P(x_j | the rest of the basket): exact and
                      cheap, used to warm up phi.
  SML                 stochastic maximum likelihood: log p(S) = E(S) - log Z, with the
                      gradient of log Z estimated by one exact DP per trip at a z drawn
                      from a persistent Gibbs chain (see scripts/fit_interactions.py).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .constraints import project_category_reward_

# Parameters of the base model that every stage trains (price is handled separately).
BASE_NAMES = ("lam", "alpha", "theta", "rho_c", "rho_0_free", "w_dsp", "w_mlr", "mu", "delta",
              "zeta", "xi")
CONTEXT_NAMES = ("alpha", "theta", "w_dsp", "w_mlr", "mu", "delta", "zeta", "xi")


def require_full_model(model):
    if model.f_extra is None or model.psi_cat is None or not model.size_on_categories:
        raise ValueError("training expects the full Version-4 model (M1, M2 and M3)")


def conditional_log_odds(model, ix, X):
    """For every slot j, d_j = E(S with j) - E(S without j), S = the basket X, so that
    P(j in S | the rest) = sigmoid(d_j).  Exact: the normaliser cancels.  Also returns the
    size of the basket without j at every slot.

        d_j = b_j + pair_j - category_j - size_j (+ M4c term if j opens its category)

    where pair_j is the change in the M2 pair term, category_j the change in
    rho_c C(n_c, 2) + f_c(n_c), and size_j the change in rho_0 of the number of categories."""
    require_full_model(model)
    x = X.double()
    ph = model.phi_r()[ix.item]
    n_row = torch.zeros(ix.n_rows).index_add(0, ix.row_of, x)
    others = n_row[ix.row_of] - x                       # same-category products besides j
    cat = ix.row_cat[ix.row_of]

    # M2 pair term: with other products of the category present, j contributes delta_j
    # (phi_j - psi_c); as the first of its category it contributes phi_j.
    s = model.basket_field(ix, X)
    psi = model.psi_cat[cat]
    dev = ph - psi
    has_other = (others >= 1.0).double()
    vec = torch.where(has_other[:, None] > 0.5, dev, ph)
    s_without = s[ix.item_trip] - x[:, None] * vec
    pair = (vec * s_without).sum(-1) - has_other * (psi * dev).sum(-1)

    f = model.f_extra[cat]
    category = (model.rho_c[cat] * (model.pair_feature(others + 1) - model.pair_feature(others))
                + ((others == 1).double() * f[:, 0]
                   + (others == 2).double() * (f[:, 1] - f[:, 0])
                   + (others == 3).double() * (f[:, 2] - f[:, 1])))

    n_without = (torch.zeros(ix.B).index_add(0, ix.item_trip, x)[ix.item_trip] - x).long()
    opens = others == 0                                 # j would be its category's first
    k_trip = torch.zeros(ix.B, dtype=torch.long).index_add(0, ix.row_trip, (n_row > 0).long())
    k_without = (k_trip[ix.item_trip] - X.long() * opens.long()).clamp(min=0)
    rho0 = model.rho_0()
    size = opens.double() * (rho0[(k_without + 1).clamp(max=model.nmax)] - rho0[k_without])

    b = model.slot_utilities(ix)
    d = b + pair - category - size
    iv = model.iv_shift(ix, b_slot=b)
    if iv is not None:
        d = d + opens.double() * iv[ix.row_of]
    return d, n_without


def pseudo_loglik(model, ix, X):
    """Mean over trips of sum_j log P(x_j | rest), exact, for slots of trips whose basket
    without j has size in 1..nmax-1 (see conditional_log_odds)."""
    d, n_without = conditional_log_odds(model, ix, X)
    valid = (n_without > 0) & (n_without < model.nmax)
    return (F.logsigmoid(torch.where(X, d, -d)) * valid).sum() / ix.B


@dataclass
class Priors:
    """Gaussian priors (precisions), divided by the number of training trips."""
    pool_prod: float          # pooling of the three product x context factorisations
    phi_prec: float           # on psi (category interaction vectors)
    delta_prec: float         # on delta (product deviations from psi)
    rho_c_prec: float
    f_extra_prec: float
    price_prec: float         # on price parameters, centred at the base model's values
    u_cat_prec: float
    rec_prec: float
    promo_dev_prec: float = 0.0   # on w_dsp, w_mlr as deviations from the common slopes


def add_pooling(loss, model, weight):
    """loss + weight * sum over the three product x context factorisations (mu, delta),
    (zeta, xi), (alpha, theta) of trace(P'P C'C) / (rows of P * rows of C): shrinks the
    products of factors, so a constant cannot hide in a bilinear term."""
    for product, context in ((model.mu, model.delta_c()), (model.zeta, model.xi_c()),
                             (model.alpha, model.theta_c())):
        loss = loss + weight * torch.trace((product.T @ product) @ (context.T @ context)) / (
            product.shape[0] * context.shape[0])
    return loss


def penalties(model, priors, n_train, psi, delta, price_init, recency):
    loss = add_pooling(0.0, model, priors.pool_prod)
    phi_prior = priors.phi_prec * psi.square().sum() + priors.delta_prec * delta.square().sum()
    extra_prior = priors.f_extra_prec * model.f_extra.square().sum()
    price_prior = 0.0
    for name, start in price_init.items():
        p = getattr(model, name)
        if p.requires_grad:
            price_prior = price_prior + priors.price_prec * (p - start).square().sum()
    u_cat_prior = priors.u_cat_prec * model.u_cat.square().sum()
    rec_prior = priors.rec_prec * (model.psi.square().sum() + model.psi_item.square().sum()) if recency else 0.0
    if model.hist_size is not None:
        rec_prior = rec_prior + priors.rec_prec * model.hist_size.square()
    promo_prior = 0.0
    if model.dsp_common is not None and priors.promo_dev_prec > 0.0:
        promo_prior = priors.promo_dev_prec * (model.w_dsp.square().sum() + model.w_mlr.square().sum())
    return loss + 0.5 * (phi_prior + priors.rho_c_prec * model.rho_c.square().sum() + extra_prior
                         + price_prior + u_cat_prior + rec_prior + promo_prior) / n_train


def project(model, capacities, optimizer):
    """Gauges and safety bounds after every update (none of them changes the law except
    the rho_c and f_c floors)."""
    model.project_context_gauges()
    with torch.no_grad():
        model.rho_0_free.sub_(model.rho_0_free[0].clone())    # M3: rho_0(1) = 0 is the gauge
        model.phi[:, model.rank:] = 0.0
    model.project_rho_c(-1.5)
    project_category_reward_(model, capacities, 1.5, optimizer=optimizer)
    with torch.no_grad():
        # Keep each count potential from undoing the rho_c floor.
        mult = torch.tensor([1.0, 3.0, 6.0], dtype=model.rho_c.dtype)
        model.f_extra.copy_(torch.maximum(model.f_extra, -1.5 - model.rho_c[:, None] * mult[None, :]))
