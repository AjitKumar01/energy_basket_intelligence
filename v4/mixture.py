"""The mixture form of the Version-4 law (docs/mixture_theory.pdf).

    p(S) = integral N(z; 0, I) p(S | z) dz,    p(S | z) = exp E_z(S) / Z_z,

where p(S | z) is the exact law given z that the current form already uses (slot_utilities_given_z,
the M2 presence shift, the category and size terms; Z_z by the dynamic program) and, for a basket S,
E_z(S) = E(S) - |s(S)|^2 / 2 + z . s(S) with E = model.energy and s = model.basket_field.  The current
form weights a taste z by N(z; 0, I) Z_z / Z; the mixture form by N(z; 0, I) alone.

E_z(S) is affine in z and log Z_z is convex, so the posterior

    log p(z | S) = -|z|^2 / 2 + z . s(S) - log Z_z + const,   Hessian -(I + Cov[s | z]) <= -I,

is 1-strongly concave: one peak m, found here by gradient ascent with backtracking.  The importance
proposal is 0.8 N(m, 1.2^2 Sigma) + 0.2 N(m, I), Sigma = (I + Cov[s | m])^-1 with Cov from exact draws;
the N(m, I) part bounds every importance weight.  The estimate of p(S) is unbiased, so its log is a
lower bound in expectation.  Baskets are drawn exactly: z ~ N(0, I), then S | z.
"""
from __future__ import annotations

import math

import torch

from v4 import exact
from v4.sampling import sample_baskets

LOG2PI = math.log(2 * math.pi)


class Posterior:
    """p(z | S) for the observed baskets of a batch (no gradients to the model parameters)."""

    def __init__(self, model, ix, iv, b_slot, s_obs):
        self.model, self.ix = model, ix
        self.iv = None if iv is None else iv.detach()
        self.b, self.s = b_slot.detach(), s_obs.detach()
        self.B, self.r = ix.B, model.rank

    def log_z(self, z):
        u = self.model.slot_utilities_given_z(self.ix, z, b=self.b)
        return exact.log_partition(self.model, self.ix, u, z=z, iv=self.iv)

    def f(self, z):
        """log p(z | S) up to a constant in z."""
        with torch.no_grad():
            return (z * self.s).sum(1) - 0.5 * z.square().sum(1) - self.log_z(z)

    def grad(self, z):
        """d log p(z | S) / dz = s(S) - z - E[s | z], with E[s | z] = d log Z_z / dz exact."""
        z = z.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            es, = torch.autograd.grad(self.log_z(z).sum(), [z])
        return self.s - z.detach() - es

    def mode(self, z=None, iters=30, tol=1e-6):
        """Gradient ascent with per-trip backtracking (Armijo), so log p(z | S) never decreases.
        Returns (peak, largest remaining gradient norm, iterations)."""
        z = torch.zeros(self.B, self.r) if z is None else z.detach().clone()
        eta = torch.ones(self.B)
        fz = self.f(z)
        it = 0
        for it in range(iters):
            g = self.grad(z)
            if float(g.norm(dim=1).max()) < tol:
                break
            for _ in range(20):
                zn = z + eta[:, None] * g
                fn = self.f(zn)
                ok = fn >= fz + 0.25 * eta * g.square().sum(1) - 1e-12
                z = torch.where(ok[:, None], zn, z)
                fz = torch.where(ok, fn, fz)
                if bool(ok.all()):
                    break
                eta = torch.where(ok, eta, 0.5 * eta)
                g = torch.where(ok[:, None], torch.zeros_like(g), g)
            eta = torch.clamp(eta * 1.5, max=1.0)
        return z, float(self.grad(z).norm(dim=1).max()), it + 1

    def spread(self, m, gen, draws=32):
        """Sigma = (I + Cov[s | m])^-1, the Laplace covariance at m, with Cov from exact draws."""
        with torch.no_grad():
            u = self.model.slot_utilities_given_z(self.ix, m, b=self.b)
            S = torch.stack([self.model.basket_field(self.ix, sample_baskets(self.model, self.ix, u, gen, z=m, iv=self.iv))
                             for _ in range(draws)])
        S = S - S.mean(0, keepdim=True)
        cov = torch.einsum("dbi,dbj->bij", S, S) / (draws - 1)
        return torch.linalg.inv(torch.eye(self.r)[None] + cov)

    def sample(self, m, Sigma, gen, K, inflate=1.2, wide=0.2):
        """K draws (K, B, r) from 0.8 N(m, inflate^2 Sigma) + 0.2 N(m, I) and their log density."""
        L = torch.linalg.cholesky(Sigma * inflate ** 2)
        zs, lr = [], []
        for _ in range(K):
            pick_wide = torch.rand(self.B, generator=gen) < wide
            e = torch.randn(self.B, self.r, generator=gen)
            z = m + torch.where(pick_wide[:, None], e, (L @ e[..., None])[..., 0])
            d = torch.linalg.solve_triangular(L, (z - m)[..., None], upper=False)[..., 0]
            l_narrow = -0.5 * d.square().sum(1) - torch.diagonal(L, dim1=1, dim2=2).log().sum(1)
            l_wide = -0.5 * (z - m).square().sum(1)
            lr.append(torch.logaddexp(l_narrow + math.log(1 - wide), l_wide + math.log(wide)) - 0.5 * self.r * LOG2PI)
            zs.append(z)
        return torch.stack(zs), torch.stack(lr)


def log_cond(model, ix, iv, b_slot, e_obs, s_obs, z):
    """log p(S | z) for each trip's observed basket; differentiable in the model parameters."""
    u = model.slot_utilities_given_z(ix, z, b=b_slot)
    lz = exact.log_partition(model, ix, u, z=z, iv=iv)
    return e_obs - 0.5 * s_obs.square().sum(1) + (z * s_obs).sum(1) - lz


def log_lik(model, batch, gen, *, K=64, mode_iters=30, spread_draws=32, grad=False, z0=None):
    """Importance estimate of log p(S) per trip (a lower bound in expectation).

    ``grad``: keep the graph to the model parameters (training).  ``z0``: start of the peak search
    (default zero).  Returns (log p per trip, ESS fraction per trip, info) with info["m"] the peaks,
    info["zs"] the draws (K, B, r) and info["w"] their normalised weights (K, B)."""
    ix = batch.ix
    with torch.no_grad():
        b0 = model.slot_utilities(ix)
        post = Posterior(model, ix, model.iv_shift(ix, b_slot=b0), b0, model.basket_field(ix, batch.observed))
    m, gmax, iters = post.mode(z=z0, iters=mode_iters)
    zs, lr = post.sample(m, post.spread(m, gen, draws=spread_draws), gen, K)
    with (torch.enable_grad() if grad else torch.no_grad()):
        b = model.slot_utilities(ix)
        iv = model.iv_shift(ix, b_slot=b)
        e = model.energy(batch, iv=iv)
        s = model.basket_field(ix, batch.observed)
        lw = torch.stack([log_cond(model, ix, iv, b, e, s, zs[k]) for k in range(K)])
        lw = lw - 0.5 * zs.square().sum(-1) - 0.5 * model.rank * LOG2PI - lr
        ll = torch.logsumexp(lw, 0) - math.log(K)
    w = torch.softmax(lw.detach(), 0)
    ess = 1.0 / (w.square().sum(0) * K)
    return ll, ess, {"mode_grad": gmax, "mode_iters": iters, "m": m, "zs": zs, "w": w}


def generate(model, ix, gen, iv=None, z=None):
    """Exact draw from the mixture: z ~ N(0, I) (or the given z), then S | z."""
    with torch.no_grad():
        z = torch.randn(ix.B, model.rank, generator=gen) if z is None else z
        return sample_baskets(model, ix, model.slot_utilities_given_z(ix, z), gen, z=z, iv=iv)


@torch.no_grad()
def whiten_(model, second_moment):
    """Parameter-expanded EM step: with A = second_moment^(1/2) (the pooled posterior second moment
    of z), set phi_j <- A phi_j, psi_c <- A psi_c and lam_j += (|A phi_j|^2 - |phi_j|^2) / 2.  A prior
    N(0, second_moment) with the old vectors is the same law as N(0, I) with the new ones, except for
    the |psi_c|^2 / 2 per extra product of a category, which is not re-compensated.  Returns the
    eigenvalues of second_moment, largest first."""
    r = model.rank
    ev, V = torch.linalg.eigh(0.5 * (second_moment + second_moment.T))
    A = V @ torch.diag(ev.clamp_min(1e-6).sqrt()) @ V.T
    old = model.phi[:, :r].square().sum(1)
    model.phi[:, :r] = model.phi[:, :r] @ A
    if model.psi_cat is not None:
        model.psi_cat.copy_(model.psi_cat @ A)
    model.lam.add_(0.5 * (model.phi[:, :r].square().sum(1) - old))
    return ev.flip(0)
