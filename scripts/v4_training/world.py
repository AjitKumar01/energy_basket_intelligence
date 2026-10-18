"""Exactly enumerable worlds of Version-4's form, with strong interactions.

Every nonempty basket of size <= nmax is enumerated, so likelihoods, log Z and pair rates are
exact.  The truth is a Version-4 model (hs.V4 parameters) whose interaction strength -- the top
eigenvalue of Phi Phi' -- is set to a target (the stress world needs about 7; Version-4's
quadrature caps it at 1).  Interactions are complements across categories ("missions");
within-category substitution comes from rho_c (or M1's general count potential f_c(n_c)).
Contexts are (household, week): household taste, household price sensitivity, weekly prices.
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import torch

from hs import V4, Partition, cat_count_penalty_matrix, pair_feature

torch.set_default_dtype(torch.float64)

GROUPS = [[0, 1, 2], [3, 4], [5, 16], [10, 11, 12, 13], [14, 15]]   # substitute categories
RHO = [1.8, 1.5, 1.6, 1.2, 2.0]
MISSIONS = [[6, 7, 8, 9, 17], [0, 3, 10, 14], [1, 5, 11, 15]]         # cross-category complements


class Support:
    def __init__(self, J, nmax):
        baskets = [c for n in range(1, nmax + 1) for c in itertools.combinations(range(J), n)]
        M = torch.zeros(len(baskets), J)
        for r, b in enumerate(baskets):
            M[r, list(b)] = 1.0
        self.M, self.size = M, M.sum(1).long()
        self.row = {b: i for i, b in enumerate(baskets)}


def partition(J, nmax):
    group = np.full(J, -1)
    for k, items in enumerate(GROUPS):
        group[items] = k
    n = len(GROUPS)
    for j in range(J):
        if group[j] < 0:
            group[j] = n
            n += 1
    return Partition(group, nmax)


def make_truth(seed, strength, J=18, H=100, W=8, K=3, r=3, nmax=5, use_f_extra=False):
    g = torch.Generator().manual_seed(seed)
    part = partition(J, nmax)
    t = V4(J, H, K, r, part, requires_grad=False, use_f_extra=use_f_extra)
    phi = 0.25 * torch.randn(J, r, generator=g)
    for d, items in enumerate(MISSIONS):
        phi[items, d] += 1.0
    top = float(torch.linalg.svdvals(phi)[0] ** 2)
    phi = phi * math.sqrt(strength / top)
    rho_c = torch.zeros(part.C)
    rho_c[:len(RHO)] = torch.tensor(RHO)
    with torch.no_grad():
        t.p["lam"].copy_(-1.6 + 0.35 * torch.randn(J, generator=g))
        t.p["theta"].copy_(0.6 * torch.randn(H, K, generator=g))
        t.p["alpha"].copy_(0.6 * torch.randn(J, K, generator=g))
        gj = 0.5 + torch.rand(J, generator=g)
        t.p["graw"].copy_(torch.log(torch.expm1(gj)))
        t.p["delta"].copy_(0.4 * torch.randn(H, generator=g))
        t.p["phi"].copy_(phi)
        t.p["rho_c"].copy_(rho_c)
        t.p["rho0_free"].copy_(0.12 * (torch.arange(2, nmax + 1).double() - 1.0) ** 2)
        if use_f_extra:
            f_ext = torch.zeros(part.C, 3)
            # Non-quadratic offsets on multi-item categories (k=2, k=3, k>=4)
            f_ext[:len(RHO), 0] = torch.tensor([-0.35, 0.25, -0.20, 0.40, -0.30])
            f_ext[:len(RHO), 1] = torch.tensor([0.55, 0.0, 0.0, 0.75, 0.0])
            f_ext[:len(RHO), 2] = torch.tensor([0.0, 0.0, 0.0, 1.10, 0.0])
            t.p["f_extra"].copy_(f_ext)
    price = 0.35 * torch.randn(W, J, generator=g)
    return t, price


def energy_table(model: V4, sup: Support, h, price):
    """C x S exact energies for contexts (h, price rows) by brute-force enumeration over sup.M."""
    p = model.p
    b = model.b(h, price)
    phi = model.eff_phi()
    n_c = sup.M @ model.part.onehot
    s = model.basket_s(sup.M)
    diag = sup.M @ phi.square().sum(1)
    pair = 0.5 * (s.square().sum(1) - diag)
    rest = pair - cat_count_penalty_matrix(n_c, p["rho_c"], model.f_extra()) - model.rho0()[sup.size - 1]
    return b @ sup.M.T + rest[None, :]


def exact_log_z(model, sup, h, price, chunk=200):
    out = []
    for s in range(0, len(h), chunk):
        out.append(torch.logsumexp(energy_table(model, sup, h[s:s + chunk], price[s:s + chunk]), 1))
    return torch.cat(out)


def contexts(H, W):
    return np.repeat(np.arange(H), W), np.tile(np.arange(W), H)


def draw_trips(truth, sup, week_price, H, W, n, rng):
    ch, cw = contexts(H, W)
    with torch.no_grad():
        E = energy_table(truth, sup, torch.as_tensor(ch), week_price[torch.as_tensor(cw)])
        P = torch.softmax(E, 1).numpy()
    ctx = rng.integers(len(ch), size=n)
    basket = np.empty(n, dtype=np.int64)
    for c in np.unique(ctx):
        rows = np.flatnonzero(ctx == c)
        basket[rows] = rng.choice(P.shape[1], size=len(rows), p=P[c] / P[c].sum())
    return {"ctx": ctx, "h": ch[ctx], "w": cw[ctx], "basket": basket}


def tensors(trips, sup, week_price):
    h = torch.as_tensor(trips["h"])
    return sup.M[torch.as_tensor(trips["basket"])], h, week_price[torch.as_tensor(trips["w"])]


def exact_trip_ll(model, sup, trips, week_price, H, W):
    """Exact per-trip log-likelihood: energy table over all contexts, then index."""
    ch, cw = contexts(H, W)
    with torch.no_grad():
        out = np.empty(len(trips["ctx"]))
        for s in range(0, len(ch), 200):
            E = energy_table(model, sup, torch.as_tensor(ch[s:s + 200]), week_price[torch.as_tensor(cw[s:s + 200])])
            L = (E - torch.logsumexp(E, 1, keepdim=True)).numpy()
            rows = np.flatnonzero((trips["ctx"] >= s) & (trips["ctx"] < s + 200))
            out[rows] = L[trips["ctx"][rows] - s, trips["basket"][rows]]
    return out


def net_pair_interaction(model):
    """Identified pair interaction W_ij = phi_i.phi_j - rho_c [same category], i < j."""
    phi = model.eff_phi().detach()
    Wm = phi @ phi.T
    same = model.part.onehot @ model.part.onehot.T
    Wm = Wm - same * model.p["rho_c"].detach()[model.part.group][:, None]
    iu = torch.triu_indices(Wm.shape[0], Wm.shape[0], 1)
    return Wm[iu[0], iu[1]].numpy()


def fit_exact_mle(model, sup, trips, week_price, H, W, prec, rounds=30):
    """Reference only: exact maximum a posteriori by enumeration (the best this model class can do)."""
    ch, cw = contexts(H, W)
    counts = torch.zeros(len(ch), sup.M.shape[0])
    counts.index_put_((torch.as_tensor(trips["ctx"]), torch.as_tensor(trips["basket"])),
                      torch.ones(len(trips["ctx"])), accumulate=True)
    N = counts.sum()
    hh, pw = torch.as_tensor(ch), week_price[torch.as_tensor(cw)]
    params = list(model.p.values())
    opt = torch.optim.LBFGS(params, lr=1, max_iter=100, history_size=50, tolerance_grad=1e-10,
                            tolerance_change=1e-13, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        E = energy_table(model, sup, hh, pw)
        loss = -(counts * (E - torch.logsumexp(E, 1, keepdim=True))).sum() / N + model.prior(prec) / N
        loss.backward()
        return loss

    last = float("inf")
    for _ in range(rounds):
        v = float(opt.step(closure))
        if abs(last - v) < 1e-11:
            break
        last = v
    return model
