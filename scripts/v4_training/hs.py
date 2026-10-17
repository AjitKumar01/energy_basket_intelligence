"""Exact-gradient training of the Version-4 energy with its interactions, on small worlds.

The model is Version-4's, unchanged:
    E(S) = sum_{j in S} b_j + sum_{i<j in S} phi_i.phi_j - sum_c rho_c C(n_c, 2) - rho_0(|S|),
over nonempty baskets with |S| <= nmax.

Hubbard-Stratonovich identity.  With s = sum_{j in S} phi_j,
    sum_{i<j} phi_i.phi_j = |s|^2/2 - sum_{j in S} |phi_j|^2/2,
    exp(|s|^2/2) = E_{z ~ N(0,I)} exp(z.s),
so Z = E_{z ~ N(0,I)} Zc(z), where Zc(z) is the normalizer of the additive law with utilities
u_j(z) = b_j - |phi_j|^2/2 + z.phi_j (one category/size dynamic program).  Consequences used here:
  * z | S ~ N(s, I) and S | z ~ the additive law at u(z): an exact two-block Gibbs sampler;
  * grad log Z = E_{z ~ p(z)}[grad log Zc(z)]  (Fisher identity; p(z) = N(z;0,I) Zc(z) / Z),
    so a z drawn from a chain targeting the model gives an unbiased gradient through ONE
    dynamic program, Rao-Blackwellized over the basket;
  * p(z) = sum_S p(S) N(z; s_S, I), so a mixture of N(s_{S_k}, I) over model baskets S_k is a
    near-optimal importance proposal for Z = E_{N(0,I)} Zc(z) (positive weights).
"""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F

torch.set_default_dtype(torch.float64)
NEG = -1e30


class Partition:
    """Items grouped into categories (singletons allowed)."""

    def __init__(self, group: np.ndarray, nmax: int):
        self.group = np.asarray(group, dtype=np.int64)
        self.C = int(self.group.max()) + 1
        self.members = [np.flatnonzero(self.group == c) for c in range(self.C)]
        self.nmax = nmax
        self.J = len(self.group)
        self.onehot = F.one_hot(torch.as_tensor(self.group), self.C).double()   # J x C


def pair_feature(n):
    return n * (n - 1) / 2


def _category_polys(u, part: Partition, rho_c):
    """Per category: within-category log ESP tables (for sampling) and the category polynomial
    log sum_{|T|=k, T in c} exp(sum_T u) - rho_c C(k,2), k = 0..min(|c|, nmax).  u: B x J."""
    B = u.shape[0]
    polys, prefixes = [], []
    for c, items in enumerate(part.members):
        top = min(len(items), part.nmax)
        e = torch.full((B, top + 1), NEG, dtype=u.dtype)
        e = torch.cat([torch.zeros(B, 1), e[:, 1:]], 1)
        steps = [e]
        for i in items:
            shifted = torch.cat([torch.full((B, 1), NEG), e[:, :-1] + u[:, i:i + 1]], 1)
            e = torch.logaddexp(e, shifted)
            steps.append(e)
        prefixes.append(steps)
        k = torch.arange(top + 1, dtype=u.dtype)
        polys.append(e - rho_c[c] * pair_feature(k))
    return polys, prefixes


def _convolve(a, b, nmax):
    """log-domain polynomial product truncated at degree nmax. a: B x A, b: B x Bk."""
    width = min(nmax, a.shape[1] + b.shape[1] - 2) + 1
    out = torch.full((a.shape[0], width), NEG, dtype=a.dtype)
    for k in range(b.shape[1]):
        keep = min(a.shape[1], width - k)
        if keep <= 0:
            break
        seg = a[:, :keep] + b[:, k:k + 1]
        out = torch.cat([out[:, :k], torch.logaddexp(out[:, k:k + keep], seg), out[:, k + keep:]], 1)
    return out


def log_zc(u, part: Partition, rho_c, rho0, return_tables=False):
    """log Zc over nonempty baskets of size <= nmax, given per-trip utilities u (B x J).
    rho0: length-nmax tensor, rho0[n-1] is the size potential of size n."""
    polys, prefixes = _category_polys(u, part, rho_c)
    prefix = [torch.zeros(u.shape[0], 1)]
    for p in polys:
        prefix.append(_convolve(prefix[-1], p, part.nmax))
    tot = prefix[-1]
    n = tot.shape[1] - 1
    size_logp = tot[:, 1:] - rho0[:n][None, :]
    lz = torch.logsumexp(size_logp, 1)
    if return_tables:
        return lz, (polys, prefixes, prefix, size_logp)
    return lz


@torch.no_grad()
def sample_given_u(u, part: Partition, rho_c, rho0, gen):
    """Exact draws S ~ additive law at utilities u (B x J), nonempty, |S| <= nmax. Returns B x J 0/1."""
    B, J = u.shape
    lz, (polys, prefixes, prefix, size_logp) = log_zc(u, part, rho_c, rho0, return_tables=True)
    X = torch.zeros(B, J)
    rows = torch.arange(B)
    size = torch.multinomial(torch.softmax(size_logp, 1), 1, generator=gen).squeeze(1) + 1
    remaining = size.clone()
    for c in range(part.C - 1, -1, -1):
        before, poly = prefix[c], polys[c]
        K = poly.shape[1]
        # P(k_c = k | remaining) ∝ prefix_{c}[remaining - k] * poly_c[k]
        ks = torch.arange(K)
        idx = remaining[:, None] - ks[None, :]
        valid = (idx >= 0) & (idx < before.shape[1])
        lp = torch.where(valid, before.gather(1, idx.clamp(0, before.shape[1] - 1)) + poly, torch.full_like(poly, NEG))
        k = torch.multinomial(torch.softmax(lp, 1), 1, generator=gen).squeeze(1)
        remaining = remaining - k
        # choose k items of category c: sequential backward over its members
        items, steps = part.members[c], prefixes[c]
        need = k.clone()
        for pos in range(len(items) - 1, -1, -1):
            i = int(items[pos])
            prev, cur = steps[pos], steps[pos + 1]          # ESP over first pos items / pos+1 items
            have = need > 0
            nk = need.clamp(min=1)
            take_lp = u[:, i] + prev.gather(1, (nk - 1).clamp(max=prev.shape[1] - 1)[:, None]).squeeze(1) \
                - cur.gather(1, nk.clamp(max=cur.shape[1] - 1)[:, None]).squeeze(1)
            p_take = torch.where(have, take_lp.clamp(max=0.0).exp(), torch.zeros(B))
            take = torch.rand(B, generator=gen) < p_take
            X[rows[take], i] = 1.0
            need = need - take.long()
    assert int((remaining != 0).sum()) == 0
    return X


class V4:
    """Version-4 parameters for a small world.  b_tj = lam_j + theta_h.alpha_j - exp(delta_h) g_j price_wj."""

    def __init__(self, J, H, K, r, part: Partition, seed=0, requires_grad=True):
        g = torch.Generator().manual_seed(seed)
        self.part = part
        self.p = {
            "lam": torch.zeros(J),
            "theta": 0.1 * torch.randn(H, K, generator=g),
            "alpha": 0.1 * torch.randn(J, K, generator=g),
            "graw": torch.full((J,), math.log(math.expm1(1.0))),
            "delta": torch.zeros(H),
            "phi": 0.05 * torch.randn(J, r, generator=g),
            "rho_c": torch.zeros(part.C),
            "rho0_free": torch.zeros(part.nmax - 1),      # rho0(1) = 0 gauge
        }
        for v in self.p.values():
            v.requires_grad_(requires_grad)

    def g(self):
        return F.softplus(self.p["graw"])

    def rho0(self):
        return torch.cat([torch.zeros(1), self.p["rho0_free"]])

    def b(self, h, price):
        p = self.p
        return p["lam"] + p["theta"][h] @ p["alpha"].T \
            - torch.exp(p["delta"][h])[:, None] * self.g()[None, :] * price

    def energy(self, X, h, price):
        """E(S) for 0/1 baskets X (B x J)."""
        p = self.p
        s = X @ p["phi"]
        pair = 0.5 * (s.square().sum(1) - X @ p["phi"].square().sum(1))
        n_c = X @ self.part.onehot
        n = X.sum(1).long()
        return (X * self.b(h, price)).sum(1) + pair - pair_feature(n_c) @ p["rho_c"] - self.rho0()[n - 1]

    def utilities(self, h, price, z):
        phi = self.p["phi"]
        return self.b(h, price) - 0.5 * phi.square().sum(1)[None, :] + z @ phi.T

    def prior(self, prec):
        p = self.p
        return 0.5 * (prec["theta"] * p["theta"].square().sum() + prec["alpha"] * p["alpha"].square().sum()
                      + prec["delta"] * p["delta"].square().sum())

    def state(self):
        return {k: v.detach().clone() for k, v in self.p.items()}

    def load(self, st):
        with torch.no_grad():
            for k, v in st.items():
                self.p[k].copy_(v)


# ---- training ---------------------------------------------------------------------------------
def pseudo_likelihood(model: V4, X, h, price):
    """Sum over trips and items of log P(x_j | rest), using Version-4's energy differences."""
    p, part = model.p, model.part
    b = model.b(h, price)
    s = X @ p["phi"]
    phisq = p["phi"].square().sum(1)
    pair = (s @ p["phi"].T) - X * phisq[None, :]                          # phi_j.(s - x_j phi_j)
    n_c = X @ part.onehot
    n_c_wo = n_c[:, part.group] - X                                         # others in j's category
    n_wo = X.sum(1, keepdim=True) - X                                       # basket size without j
    R = torch.cat([torch.zeros(1), model.rho0()])                           # R[n] for n = 0..nmax
    nmax = part.nmax
    idx = n_wo.long()
    up = R[(idx + 1).clamp(max=nmax)] - R[idx]
    d = b + pair - p["rho_c"][part.group][None, :] * n_c_wo - up
    valid = (n_wo > 0) & (n_wo < nmax)                                      # both states admissible
    lp = F.logsigmoid(torch.where(X > 0, d, -d))
    return (lp * valid).sum()


def fit_pseudo_likelihood(model: V4, X, h, price, prec, rounds=8):
    params = list(model.p.values())
    opt = torch.optim.LBFGS(params, lr=1, max_iter=200, history_size=30, tolerance_grad=1e-9,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")
    N = X.shape[0]

    def closure():
        opt.zero_grad()
        loss = -pseudo_likelihood(model, X, h, price) / N + model.prior(prec) / N
        loss.backward()
        return loss

    for _ in range(rounds):
        opt.step(closure)


@torch.no_grad()
def gibbs_sweep(model: V4, X, h, price, gen, sweeps=1):
    """Exact two-block Gibbs: z | S ~ N(s, I), S | z exact.  Returns the new X and the last z."""
    phi = model.p["phi"]
    for _ in range(sweeps):
        z = X @ phi + torch.randn(X.shape[0], phi.shape[1], generator=gen)
        X = sample_given_u(model.utilities(h, price, z), model.part, model.p["rho_c"], model.rho0(), gen)
    return X, z


def fit_sml(model: V4, X_obs, h, price, prec, iters, lr, lr_final, gen, chains=1, sweeps=1,
            average_from=0.5, plug_in=False, log=None, every=100, monitor=None):
    """Stochastic maximum likelihood with persistent HS block-Gibbs chains (Younes; Tieleman).

    Each iteration (full batch): advance every trip's chain(s) by `sweeps` exact Gibbs sweeps,
    draw z | S_chain, and step on
        -mean_t [E(S_obs,t) - log Zc(z_t)] + prior / N,
    whose gradient is unbiased for the exact negative log-likelihood gradient when the chains are
    at the model's law.  plug_in=True replaces the chain's z by the observed basket's s (no noise),
    the shortcut in fit_fast_v4.py -- kept only as a comparison arm.
    Iterates are Polyak-averaged from `average_from` of the run.
    """
    N = X_obs.shape[0]
    params = list(model.p.values())
    opt = torch.optim.Adam(params, lr=lr)
    decay = (lr_final / lr) ** (1.0 / max(iters - 1, 1))
    hh = h.repeat(chains)
    pp = price.repeat(chains, 1)
    S = X_obs.repeat(chains, 1).clone()                         # data-initialized persistent chains
    avg, n_avg = None, 0
    start_avg = int(average_from * iters)
    for it in range(1, iters + 1):
        if plug_in:
            with torch.no_grad():
                z = X_obs @ model.p["phi"]
            zz, hz, pz = z, h, price
        else:
            S, z = gibbs_sweep(model, S, hh, pp, gen, sweeps)
            with torch.no_grad():
                z = S @ model.p["phi"] + torch.randn(S.shape[0], model.p["phi"].shape[1], generator=gen)
            zz, hz, pz = z, hh, pp
        lzc = log_zc(model.utilities(hz, pz, zz), model.part, model.p["rho_c"], model.rho0())
        loss = -(model.energy(X_obs, h, price).mean() - lzc.mean()) + model.prior(prec) / N
        opt.zero_grad()
        loss.backward()
        opt.step()
        for gparam in opt.param_groups:
            gparam["lr"] *= decay
        if it > start_avg:
            st = model.state()
            avg = st if avg is None else {k: avg[k] + (st[k] - avg[k]) / (n_avg + 1) for k in st}
            n_avg += 1
        if log is not None and (it % every == 0 or it == iters):
            row = {"iter": it, "objective": float(-loss), "lr": opt.param_groups[0]["lr"],
                   "phi_top_eig": float(torch.linalg.svdvals(model.p["phi"].detach())[0] ** 2),
                   "chain_mean_size": float(S.sum(1).mean()), "data_mean_size": float(X_obs.sum(1).mean())}
            if monitor is not None:
                row.update(monitor(model))
            log(row)
    if avg is not None:
        model.load(avg)
    return model


# ---- log Z estimation -------------------------------------------------------------------------
@torch.no_grad()
def is_log_z(model: V4, h, price, gen, burn=200, components=64, thin=2, draws=512, defensive=0.1,
             tempering=None):
    """Importance-sampling log Z per context (rows of h/price).

    Model baskets S_1..S_K come from an HS Gibbs chain started from an exact draw of the
    interaction-free law (z = 0 at utilities b - |phi|^2/2 + 0 -> use b directly).  Proposal:
    q(z) = (1-a) * mean_k N(z; s_k, I) + a * N(z; 0, I).  Weight: N(z;0,I) Zc(z) / q(z).
    Returns log Z estimate and the effective-sample-size fraction of the weights.
    """
    phi = model.p["phi"]
    B, r = h.shape[0], phi.shape[1]
    rho_c, rho0 = model.p["rho_c"], model.rho0()
    X = sample_given_u(model.b(h, price), model.part, rho_c, rho0, gen)
    for _ in range(burn):
        X, _ = gibbs_sweep(model, X, h, price, gen)
    centres = []
    for k in range(components * thin):
        X, _ = gibbs_sweep(model, X, h, price, gen)
        if (k + 1) % thin == 0:
            centres.append(X @ phi)
    centres = torch.stack(centres, 1)                                       # B x K x r
    K = centres.shape[1]
    log_w = []
    half_log2pi = 0.5 * r * math.log(2 * math.pi)
    for _ in range(draws):
        pick = torch.randint(K, (B,), generator=gen)
        use_prior = torch.rand(B, generator=gen) < defensive
        mean = torch.where(use_prior[:, None], torch.zeros(B, r), centres[torch.arange(B), pick])
        z = mean + torch.randn(B, r, generator=gen)
        log_prior = -0.5 * z.square().sum(1) - half_log2pi
        comp = -0.5 * (z[:, None, :] - centres).square().sum(2) - half_log2pi   # B x K
        log_q = torch.logaddexp(math.log1p(-defensive) + torch.logsumexp(comp, 1) - math.log(K),
                                math.log(defensive) + log_prior)
        lzc = log_zc(model.utilities(h, price, z), model.part, rho_c, rho0)
        log_w.append(log_prior + lzc - log_q)
    log_w = torch.stack(log_w, 1)
    est = torch.logsumexp(log_w, 1) - math.log(draws)
    w = torch.softmax(log_w, 1)
    ess = 1.0 / w.square().sum(1) / draws
    return est, ess
