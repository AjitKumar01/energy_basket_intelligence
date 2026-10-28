"""Building and constraining the full model for training (scripts/fit_mixture.py).

``add_blocks`` puts every Version-4 block on the exact base model: count potentials f_c
(M1), tied interactions phi_j = psi_c(j) + delta_j with presence coupling (M2), the size
potential on categories present (M3), household taste per category (M4a); optionally
purchase history, its centring, within-category competition and common promotion slopes.
``penalties`` is the Gaussian prior on these blocks, ``project`` the gauges and bounds
applied after every update.  The objective itself, the mixture log-likelihood, is in
v4/mixture.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .constraints import project_category_reward_

# Parameters of the base model that every stage trains (price is handled separately).
BASE_NAMES = ("lam", "alpha", "theta", "rho_c", "rho_0_free", "w_dsp", "w_mlr", "mu", "delta",
              "zeta", "xi")
CONTEXT_NAMES = ("alpha", "theta", "w_dsp", "w_mlr", "mu", "delta", "zeta", "xi")


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


def add_blocks(model, rank, *, phi_init=0.05, recency=False, centre_history=None,
               within_competition=False, comp_init=0.0, gen=None):
    """Create the Version-4 parameters on the base model (phi = 0), in place; returns
    (psi_cat, category of each product).  Every new block starts at zero except psi_cat,
    which starts at phi_init * N(0, 1) so the taste directions are not all equal."""
    P = torch.nn.Parameter
    C = model.C
    model.f_extra = P(torch.zeros(C, 3))
    model.u_cat = P(torch.zeros(C, model.theta.shape[1]))
    if within_competition:
        model.comp_raw = P(torch.full((C,), float(comp_init)))
    if recency:
        model.psi_item = P(torch.zeros(C, 3))
        with torch.no_grad():
            model.psi.zero_()
    if centre_history:
        model.hist_size = P(torch.zeros(()))
        model.centre_history = centre_history
    # M3: move the linear part of rho_0 (its slope over sizes 1..10, or 1..nmax if smaller)
    # into lam, then let rho_0 act on the number of categories present.
    model.size_on_categories = True
    with torch.no_grad():
        top = min(10, model.rho_0_free.numel()) - 1
        slope = (model.rho_0_free[top] - model.rho_0_free[0]) / top
        model.lam.sub_(slope)
        model.rho_0_free.sub_(slope * torch.arange(1, model.rho_0_free.numel() + 1))
        model.rho_0_free.sub_(model.rho_0_free[0].clone())
    model.rank = rank
    category = model.cat_of.long()
    model.psi_cat = P(phi_init * torch.randn(C, rank, generator=gen))
    with torch.no_grad():
        model.phi.zero_()
        model.phi[:, :rank] = model.psi_cat[category]
    return model.psi_cat, category
