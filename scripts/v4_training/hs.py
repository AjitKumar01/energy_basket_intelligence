"""Exact-gradient training of the Version-4 energy with its interactions, on small worlds.

The model is Version-4's, unchanged (plus optional M1 free category-count potential f_c(n_c)):
    E(S) = sum_{j in S} b_j + 0.5 (|s(S)|^2 - diag(S)) - sum_c f_c(n_c) - rho_0(|S|),
over nonempty baskets with |S| <= nmax.

Hubbard-Stratonovich identity.  With s = s(S),
    exp(|s|^2/2) = E_{z ~ N(0,I)} exp(z.s),
so Z = E_{z ~ N(0,I)} Zc(z), where Zc(z) is the normalizer of the additive law with slot utilities
and category-count potentials (one category/size dynamic program).
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


def cat_count_penalty(c: int, ks: torch.Tensor, rho_c: torch.Tensor, f_extra: torch.Tensor | None = None):
    """Per-category count potential f_c(k) = rho_c[c] * C(k, 2) + f_extra[c](k),
    where f_extra[c] has 3 parameters for k=2, k=3, and k>=4 (pooled)."""
    pen = rho_c[c] * pair_feature(ks)
    if f_extra is not None:
        fe = f_extra[c]
        extra = torch.zeros_like(ks)
        extra = torch.where(ks == 2, fe[0], extra)
        extra = torch.where(ks == 3, fe[1], extra)
        extra = torch.where(ks >= 4, fe[2], extra)
        pen = pen + extra
    return pen


def cat_count_penalty_matrix(n_c: torch.Tensor, rho_c: torch.Tensor, f_extra: torch.Tensor | None = None):
    """Evaluate sum_c f_c(n_{b,c}) for matrix n_c (B x C)."""
    pen = pair_feature(n_c) @ rho_c
    if f_extra is not None:
        e2 = (n_c == 2).double() @ f_extra[:, 0]
        e3 = (n_c == 3).double() @ f_extra[:, 1]
        e4 = (n_c >= 4).double() @ f_extra[:, 2]
        pen = pen + e2 + e3 + e4
    return pen


def _category_polys(u, part: Partition, rho_c, f_extra=None, cat_shift=None):
    """Per category: within-category log ESP tables (for sampling) and the category polynomial
    log sum_{|T|=k, T in c} exp(sum_T u) - f_c(k) - (k-1)*cat_shift, k = 0..min(|c|, nmax).  u: B x J."""
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
        pen = cat_count_penalty(c, k, rho_c, f_extra)[None, :]
        if cat_shift is not None and top >= 2:
            km1 = (k - 1.0).clamp_min(0.0)[None, :]
            pen = pen + km1 * cat_shift[:, c:c + 1]
        polys.append(e - pen)
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


def log_zc(u, part: Partition, rho_c, rho0, return_tables=False, f_extra=None, cat_shift=None):
    """log Zc over nonempty baskets of size <= nmax, given per-trip utilities u (B x J).
    rho0: length-nmax tensor, rho0[n-1] is the size potential of size n."""
    if return_tables:
        polys, prefixes = _category_polys(u, part, rho_c, f_extra=f_extra, cat_shift=cat_shift)
        prefix = [torch.zeros(u.shape[0], 1)]
        for p in polys:
            prefix.append(_convolve(prefix[-1], p, part.nmax))
        tot = prefix[-1]
        n = tot.shape[1] - 1
        size_logp = tot[:, 1:] - rho0[:n][None, :]
        lz = torch.logsumexp(size_logp, 1)
        return lz, (polys, prefixes, prefix, size_logp)

    B = u.shape[0]
    nmax = part.nmax
    u_max = u.detach().amax(dim=1, keepdim=True).clamp(min=-20.0, max=20.0)
    w = torch.exp(u - u_max)
    tot = torch.zeros((B, nmax + 1), dtype=u.dtype)
    tot[:, 0] = 1.0
    deg_tot = 0
    for c, items in enumerate(part.members):
        m = len(items)
        if m == 1:
            wc = w[:, items[0]:items[0] + 1]
            top = min(deg_tot + 1, nmax)
            tot_prev = tot[:, :top].clone()
            tot[:, 1:top + 1] = tot[:, 1:top + 1] + tot_prev * wc
            deg_tot = top
        else:
            top_c = min(m, nmax)
            ec = torch.zeros((B, top_c + 1), dtype=u.dtype)
            ec[:, 0] = 1.0
            cur_d = 0
            for idx_i in items:
                wi = w[:, idx_i:idx_i + 1]
                nd = min(cur_d + 1, top_c)
                ec_prev = ec[:, :nd].clone()
                ec[:, 1:nd + 1] = ec[:, 1:nd + 1] + ec_prev * wi
                cur_d = nd
            ks = torch.arange(top_c + 1, dtype=u.dtype)
            pen = cat_count_penalty(c, ks, rho_c, f_extra)[None, :]
            if cat_shift is not None and top_c >= 2:
                km1 = (ks - 1.0).clamp_min(0.0)[None, :]
                pen = pen + km1 * cat_shift[:, c:c + 1]
            pc = ec * torch.exp(-pen)
            new_deg = min(deg_tot + top_c, nmax)
            out = torch.zeros((B, nmax + 1), dtype=u.dtype)
            out[:, :deg_tot + 1] = tot[:, :deg_tot + 1]
            for k in range(1, top_c + 1):
                keep = min(deg_tot + 1, new_deg + 1 - k)
                if keep <= 0:
                    break
                out[:, k:k + keep] = out[:, k:k + keep] + tot[:, :keep] * pc[:, k:k + 1]
            tot = out
            deg_tot = new_deg
    n = min(deg_tot, nmax)
    ks = torch.arange(1, n + 1, dtype=u.dtype)[None, :]
    size_logp = torch.log(tot[:, 1:n + 1].clamp_min(1e-300)) + ks * u_max - rho0[:n][None, :]
    return torch.logsumexp(size_logp, dim=1)


@torch.no_grad()
def sample_given_u(u, part: Partition, rho_c, rho0, gen, f_extra=None, cat_shift=None):
    """Exact draws S ~ additive law at utilities u (B x J), nonempty, |S| <= nmax. Returns B x J 0/1."""
    B, J = u.shape
    nmax = part.nmax
    u_max = u.amax(dim=1, keepdim=True).clamp(min=-20.0, max=20.0)
    w = torch.exp(u - u_max)

    polys, prefixes = [], []
    prefix = [torch.ones((B, 1), dtype=u.dtype)]
    deg_tot = 0
    for c, items in enumerate(part.members):
        m = len(items)
        top_c = min(m, nmax)
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
        prefixes.append(steps)
        if m == 1:
            pc = e
        else:
            ks = torch.arange(top_c + 1, dtype=u.dtype)
            pen = cat_count_penalty(c, ks, rho_c, f_extra)[None, :]
            if cat_shift is not None and top_c >= 2:
                km1 = (ks - 1.0).clamp_min(0.0)[None, :]
                pen = pen + km1 * cat_shift[:, c:c + 1]
            pc = e * torch.exp(-pen)
        polys.append(pc)

        prev_tot = prefix[-1]
        new_deg = min(deg_tot + top_c, nmax)
        out = torch.zeros((B, new_deg + 1), dtype=u.dtype)
        out[:, :deg_tot + 1] = prev_tot
        for k in range(1, top_c + 1):
            keep = min(deg_tot + 1, new_deg + 1 - k)
            if keep <= 0:
                break
            out[:, k:k + keep] = out[:, k:k + keep] + prev_tot[:, :keep] * pc[:, k:k + 1]
        prefix.append(out)
        deg_tot = new_deg

    tot = prefix[-1]
    n = tot.shape[1] - 1
    ks = torch.arange(1, n + 1, dtype=u.dtype)[None, :]
    size_logp = torch.log(tot[:, 1:].clamp_min(1e-300)) + ks * u_max - rho0[:n][None, :]
    X = torch.zeros(B, J, dtype=u.dtype)
    size = torch.multinomial(torch.softmax(size_logp, 1), 1, generator=gen).squeeze(1) + 1
    remaining = size.clone()
    for c in range(part.C - 1, -1, -1):
        before, poly = prefix[c], polys[c]
        K = poly.shape[1]
        ks_c = torch.arange(K)
        idx = remaining[:, None] - ks_c[None, :]
        valid = (idx >= 0) & (idx < before.shape[1])
        probs = torch.where(valid, before.gather(1, idx.clamp(0, before.shape[1] - 1)) * poly, torch.zeros_like(poly))
        k = torch.multinomial(probs.clamp_min(0.0) + 1e-300 * valid.double(), 1, generator=gen).squeeze(1)
        remaining = remaining - k
        items, steps = part.members[c], prefixes[c]
        if len(items) == 1:
            X[:, items[0]] = k.double()
        else:
            need = k.clone()
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
    assert int((remaining != 0).sum()) == 0
    return X


class V4:
    """Version-4 parameters for a small world.  b_tj = lam_j + theta_h.alpha_j - exp(delta_h) g_j price_wj.
    Supports M1 (f_extra: C x 3 count potential offsets) and M2 (presence_phi=True with psi: C x r, delta_phi: J x r)."""

    def __init__(self, J, H, K, r, part: Partition, seed=0, requires_grad=True, use_f_extra=False):
        g = torch.Generator().manual_seed(seed)
        self.part = part
        self.use_f_extra = use_f_extra
        self.presence_phi = False
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
        if use_f_extra:
            self.p["f_extra"] = torch.zeros(part.C, 3)
        for v in self.p.values():
            v.requires_grad_(requires_grad)

    def f_extra(self):
        return self.p.get("f_extra", None) if self.use_f_extra else None

    def eff_phi(self):
        if self.presence_phi:
            return self.p["psi"][self.part.group] + self.p["delta_phi"]
        return self.p["phi"]

    def basket_s(self, X):
        """s(S) for 0/1 baskets X (B x J)."""
        if self.presence_phi:
            n_c = X @ self.part.onehot
            pres_c = (n_c > 0).double()
            return pres_c @ self.p["psi"] + X @ self.p["delta_phi"]
        return X @ self.p["phi"]

    def cat_shift(self, z):
        """Category presence shift a_c(z) = z . psi_c - 0.5 |psi_c|^2 for M2 (or None for count-linear Phi)."""
        if not self.presence_phi or z is None:
            return None
        psi = self.p["psi"]
        return z @ psi.T - 0.5 * psi.square().sum(1)[None, :]

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
        phi = self.eff_phi()
        n_c = X @ self.part.onehot
        s = self.basket_s(X)
        diag = X @ phi.square().sum(1)
        if self.presence_phi:
            extra_c = (n_c - 1.0).clamp_min(0.0)
            diag = diag - extra_c @ p["psi"].square().sum(1)
        pair = 0.5 * (s.square().sum(1) - diag)
        n = X.sum(1).long()
        return (X * self.b(h, price)).sum(1) + pair - cat_count_penalty_matrix(n_c, p["rho_c"], self.f_extra()) - self.rho0()[n - 1]

    def utilities(self, h, price, z):
        phi = self.eff_phi()
        return self.b(h, price) - 0.5 * phi.square().sum(1)[None, :] + z @ phi.T

    def prior(self, prec):
        """Gaussian prior including weak minimum-norm regularisation on Phi and rho_c."""
        p = self.p
        val = (
            prec["theta"] * p["theta"].square().sum()
            + prec["alpha"] * p["alpha"].square().sum()
            + prec["delta"] * p["delta"].square().sum()
            + prec.get("rho_c", 0.2) * p["rho_c"].square().sum()
        )
        if self.presence_phi:
            val = val + prec.get("phi", 2.0) * p["psi"].square().sum() + prec.get("delta_phi", 20.0) * p["delta_phi"].square().sum()
        else:
            val = val + prec.get("phi", 2.0) * p["phi"].square().sum()
        if self.use_f_extra:
            val = val + prec.get("f_extra", 2.0) * p["f_extra"].square().sum()
        return 0.5 * val

    def state(self):
        return {k: v.detach().clone() for k, v in self.p.items()}

    def load(self, st):
        with torch.no_grad():
            for k, v in st.items():
                if k in self.p:
                    self.p[k].copy_(v)


# ---- training ---------------------------------------------------------------------------------
def pseudo_likelihood(model: V4, X, h, price):
    """Sum over trips and items of log P(x_j | rest), using Version-4's energy differences."""
    p, part = model.p, model.part
    b = model.b(h, price)
    phi = model.eff_phi()
    s = model.basket_s(X)
    phisq = phi.square().sum(1)
    pair = (s @ phi.T) - X * phisq[None, :]                                # phi_j.(s - x_j phi_j)
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
    N = float(X.shape[0])
    J = X.shape[1]
    key = torch.cat([h[:, None].double(), price, X], dim=1)
    uniq, counts = torch.unique(key, dim=0, return_counts=True)
    hu = uniq[:, 0].long()
    pu = uniq[:, 1:1 + J]
    Xu = uniq[:, 1 + J:]
    wu = counts.double()[:, None]

    part = model.part
    n_c = Xu @ part.onehot
    n_c_wo = n_c[:, part.group] - Xu
    n_wo = Xu.sum(1, keepdim=True) - Xu
    nmax = part.nmax
    idx = n_wo.long()
    idx_up = (idx + 1).clamp(max=nmax)
    valid_w = ((n_wo > 0) & (n_wo < nmax)).double() * wu

    def closure():
        opt.zero_grad()
        p = model.p
        b = model.b(hu, pu)
        phi = model.eff_phi()
        s = Xu @ phi
        phisq = phi.square().sum(1)
        pair = (s @ phi.T) - Xu * phisq[None, :]
        R = torch.cat([torch.zeros(1), model.rho0()])
        up = R[idx_up] - R[idx]
        d = b + pair - p["rho_c"][part.group][None, :] * n_c_wo - up
        lp = F.logsigmoid(torch.where(Xu > 0, d, -d))
        loss = -(lp * valid_w).sum() / N + model.prior(prec) / N
        loss.backward()
        return loss

    last = float("inf")
    for _ in range(rounds):
        v = float(opt.step(closure))
        if abs(last - v) < 1e-12:
            break
        last = v


@torch.no_grad()
def gibbs_sweep(model: V4, X, h, price, gen, sweeps=1):
    """Exact two-block Gibbs: z | S ~ N(s, I), S | z exact.  Returns the new X and the last z."""
    phi = model.eff_phi()
    for _ in range(sweeps):
        z = model.basket_s(X) + torch.randn(X.shape[0], phi.shape[1], generator=gen)
        X = sample_given_u(model.utilities(h, price, z), model.part, model.p["rho_c"], model.rho0(), gen,
                           f_extra=model.f_extra(), cat_shift=model.cat_shift(z))
    return X, z


def fit_sml(model: V4, X_obs, h, price, prec, iters, lr, lr_final, gen, chains=1, sweeps=1,
            average_from=0.5, plug_in=False, log=None, every=100, monitor=None):
    """Stochastic maximum likelihood with persistent HS block-Gibbs chains (Younes; Tieleman)."""
    N = X_obs.shape[0]
    J = X_obs.shape[1]
    params = list(model.p.values())
    opt = torch.optim.Adam(params, lr=lr)
    decay = (lr_final / lr) ** (1.0 / max(iters - 1, 1))

    ctx_key = torch.cat([h[:, None].double(), price], dim=1)
    uniq_ctx, ctx_inv, ctx_counts = torch.unique(ctx_key, dim=0, return_inverse=True, return_counts=True)
    C_ctx = uniq_ctx.shape[0]
    h_ctx = uniq_ctx[:, 0].long()
    p_ctx = uniq_ctx[:, 1:]
    w_ctx = ctx_counts.double() / float(N)

    X_ctx_mean = torch.zeros((C_ctx, J), dtype=X_obs.dtype)
    X_ctx_mean.index_add_(0, ctx_inv, X_obs / float(N))
    C_xx = (X_obs.T @ X_obs) / float(N)
    C_xx.diagonal().zero_()
    n_c_obs = X_obs @ model.part.onehot
    pair_c_mean = pair_feature(n_c_obs).mean(0)
    obs_sizes = X_obs.sum(1).long()
    size_mean = torch.bincount(obs_sizes - 1, minlength=model.part.nmax).double() / float(N)
    data_mean_size = float(X_obs.sum(1).mean())

    def mean_obs_energy():
        if model.use_f_extra or model.presence_phi:
            return model.energy(X_obs, h, price).mean()
        p = model.p
        b_c = model.b(h_ctx, p_ctx)
        e_b = (X_ctx_mean * b_c).sum()
        e_pair = 0.5 * (p["phi"] * (C_xx @ p["phi"])).sum()
        e_cat = (pair_c_mean * p["rho_c"]).sum()
        e_size = (size_mean * model.rho0()).sum()
        return e_b + e_pair - e_cat - e_size

    if plug_in:
        ukey = torch.cat([ctx_inv[:, None].double(), X_obs], dim=1)
        u_rows, u_counts = torch.unique(ukey, dim=0, return_counts=True)
        u_cidx = u_rows[:, 0].long()
        Xu = u_rows[:, 1:]
        wu_plug = u_counts.double() / float(N)
        S = X_obs
    else:
        K_per_ctx = max(8 * int(chains), 1)
        if C_ctx * K_per_ctx < N:
            order = torch.argsort(ctx_inv)
            ptr = torch.cat([torch.zeros(1, dtype=torch.long), ctx_counts.cumsum(0)])
            starts = ptr[:-1][:, None]
            offs = (torch.arange(K_per_ctx)[None, :] % ctx_counts[:, None])
            pick_rows = order[(starts + offs).reshape(-1)]
            S = X_obs[pick_rows].clone()
            chain_cidx = torch.arange(C_ctx).repeat_interleave(K_per_ctx)
            chain_w = (w_ctx / float(K_per_ctx)).repeat_interleave(K_per_ctx)
        else:
            S = X_obs.repeat(chains, 1).clone()
            chain_cidx = ctx_inv.repeat(chains)
            chain_w = torch.full((S.shape[0],), 1.0 / float(S.shape[0]), dtype=X_obs.dtype)

    avg, n_avg = None, 0
    start_avg = int(average_from * iters)
    for it in range(1, iters + 1):
        if plug_in:
            with torch.no_grad():
                z = model.basket_s(Xu)
            phi = model.eff_phi()
            b_c = model.b(h_ctx, p_ctx)
            u_all = b_c[u_cidx] - 0.5 * phi.square().sum(1)[None, :] + z @ phi.T
            lzc = log_zc(u_all, model.part, model.p["rho_c"], model.rho0(),
                         f_extra=model.f_extra(), cat_shift=model.cat_shift(z))
            lzc_mean = (lzc * wu_plug).sum()
        else:
            with torch.no_grad():
                phi = model.eff_phi()
                b_c = model.b(h_ctx, p_ctx)
                b_chain = b_c[chain_cidx] - 0.5 * phi.square().sum(1)[None, :]
                rho_c, rho0, f_ext = model.p["rho_c"], model.rho0(), model.f_extra()
                for _ in range(sweeps):
                    z = model.basket_s(S) + torch.randn(S.shape[0], phi.shape[1], generator=gen)
                    S = sample_given_u(b_chain + z @ phi.T, model.part, rho_c, rho0, gen,
                                       f_extra=f_ext, cat_shift=model.cat_shift(z))
                z = model.basket_s(S) + torch.randn(S.shape[0], phi.shape[1], generator=gen)
            phi = model.eff_phi()
            b_c = model.b(h_ctx, p_ctx)
            u_all = b_c[chain_cidx] - 0.5 * phi.square().sum(1)[None, :] + z @ phi.T
            lzc = log_zc(u_all, model.part, model.p["rho_c"], model.rho0(),
                         f_extra=model.f_extra(), cat_shift=model.cat_shift(z))
            lzc_mean = (lzc * chain_w).sum()

        loss = -(mean_obs_energy() - lzc_mean) + model.prior(prec) / N
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
            c_size = data_mean_size if plug_in else float((S.sum(1) * chain_w).sum())
            row = {"iter": it, "objective": float(-loss), "lr": opt.param_groups[0]["lr"],
                   "phi_top_eig": float(torch.linalg.svdvals(model.eff_phi().detach())[0] ** 2),
                   "chain_mean_size": c_size, "data_mean_size": data_mean_size}
            if monitor is not None:
                row.update(monitor(model))
            log(row)
    if avg is not None:
        model.load(avg)
    return model


# ---- log Z estimation -------------------------------------------------------------------------
@torch.no_grad()
def is_log_z_orig(model: V4, h, price, gen, burn=200, components=64, thin=2, draws=512, defensive=0.1):
    """Original single-chain importance-sampling log Z per context."""
    phi = model.eff_phi()
    B, r = h.shape[0], phi.shape[1]
    rho_c, rho0, f_ext = model.p["rho_c"], model.rho0(), model.f_extra()
    b0 = model.b(h, price)
    u0 = b0 - 0.5 * phi.square().sum(1)[None, :]
    X = sample_given_u(b0, model.part, rho_c, rho0, gen, f_extra=f_ext)
    for _ in range(burn):
        z_b = model.basket_s(X) + torch.randn(B, r, generator=gen)
        X = sample_given_u(u0 + z_b @ phi.T, model.part, rho_c, rho0, gen,
                           f_extra=f_ext, cat_shift=model.cat_shift(z_b))
    centres = []
    for k in range(components * thin):
        z_b = model.basket_s(X) + torch.randn(B, r, generator=gen)
        X = sample_given_u(u0 + z_b @ phi.T, model.part, rho_c, rho0, gen,
                           f_extra=f_ext, cat_shift=model.cat_shift(z_b))
        if (k + 1) % thin == 0:
            centres.append(model.basket_s(X))
    centres = torch.stack(centres, 1)                                       # B x K x r
    K = centres.shape[1]
    log_w = []
    half_log2pi = 0.5 * r * math.log(2 * math.pi)
    batch_d = 64
    for start in range(0, draws, batch_d):
        d_cur = min(batch_d, draws - start)
        pick = torch.randint(K, (B, d_cur), generator=gen)
        use_prior = torch.rand(B, d_cur, generator=gen) < defensive
        c_pick = centres.gather(1, pick[:, :, None].expand(B, d_cur, r))
        mean = torch.where(use_prior[:, :, None], torch.zeros(B, d_cur, r), c_pick)
        z = mean + torch.randn(B, d_cur, r, generator=gen)
        log_prior = -0.5 * z.square().sum(2) - half_log2pi
        comp = -0.5 * (z[:, :, None, :] - centres[:, None, :, :]).square().sum(3) - half_log2pi
        log_q = torch.logaddexp(math.log1p(-defensive) + torch.logsumexp(comp, 2) - math.log(K),
                                math.log(defensive) + log_prior)
        z_flat = z.reshape(B * d_cur, r)
        u_flat = (u0[:, None, :] + z @ phi.T).reshape(B * d_cur, -1)
        lzc = log_zc(u_flat, model.part, rho_c, rho0,
                     f_extra=f_ext, cat_shift=model.cat_shift(z_flat)).view(B, d_cur)
        log_w.append(log_prior + lzc - log_q)
    log_w = torch.cat(log_w, 1)
    est = torch.logsumexp(log_w, 1) - math.log(draws)
    w = torch.softmax(log_w, 1)
    ess = 1.0 / w.square().sum(1) / draws
    return est, ess


@torch.no_grad()
def is_log_z(model: V4, h, price, gen, burn=200, components=64, thin=2, draws=512, defensive=0.05,
             tempering=None):
    """Multi-seed mode-weighted importance-sampling log Z per context (rows of h/price).

    Starts M parallel chains across +/- singular directions of Phi, the beta=0 additive law,
    and z=0 so isolated interaction modes at large Phi are all visited. Weights proposal
    centres by exact target log-mass at each unique basket centre + uniform floor, and uses
    stratified mixture component draws to eliminate discrete mode-sampling variance.
    """
    phi = model.eff_phi()
    B, r = h.shape[0], phi.shape[1]
    rho_c, rho0, f_ext = model.p["rho_c"], model.rho0(), model.f_extra()
    b0 = model.b(h, price)
    phi_sq = phi.square().sum(1)[None, :]
    u0 = b0 - 0.5 * phi_sq

    _, sv, Vt = torch.linalg.svd(phi, full_matrices=False)
    seed_u = [b0, u0]
    seed_z = [None, torch.zeros(B, r, dtype=phi.dtype)]
    for d in range(r):
        if float(sv[d]) > 0.5:
            for mult in (1.5, 3.0):
                z_dir = (mult * sv[d]) * Vt[d]
                seed_u.append(u0 + (z_dir @ phi.T)[None, :])
                seed_z.append(z_dir[None, :].expand(B, r))
                seed_u.append(u0 - (z_dir @ phi.T)[None, :])
                seed_z.append((-z_dir)[None, :].expand(B, r))
    M = len(seed_u)
    X_seeds = []
    for m_i in range(M):
        X_seeds.append(sample_given_u(seed_u[m_i], model.part, rho_c, rho0, gen,
                                      f_extra=f_ext, cat_shift=model.cat_shift(seed_z[m_i])))
    X_flat = torch.stack(X_seeds, dim=1).reshape(B * M, -1)
    u0_M = u0.repeat_interleave(M, dim=0)

    burn_per_chain = max(burn // 4, 25)
    for _ in range(burn_per_chain):
        z_b = model.basket_s(X_flat) + torch.randn(B * M, r, generator=gen)
        X_flat = sample_given_u(u0_M + z_b @ phi.T, model.part, rho_c, rho0, gen,
                                f_extra=f_ext, cat_shift=model.cat_shift(z_b))

    k_per_chain = max((components + M - 1) // M, 4)
    centres = []
    for k in range(k_per_chain * thin):
        z_b = model.basket_s(X_flat) + torch.randn(B * M, r, generator=gen)
        X_flat = sample_given_u(u0_M + z_b @ phi.T, model.part, rho_c, rho0, gen,
                                f_extra=f_ext, cat_shift=model.cat_shift(z_b))
        if (k + 1) % thin == 0:
            centres.append(model.basket_s(X_flat).view(B, M, r))
    centres = torch.cat(centres, dim=1)[:, :max(components, M * 4), :]
    K = centres.shape[1]

    # Weight centres by exact target density at s_k (deduplicating identical centres) + uniform floor
    diff_c = (centres[:, :, None, :] - centres[:, None, :, :]).square().sum(3)
    same_c = (diff_c < 1e-8).double()
    mult = same_c.sum(2).clamp_min(1.0)
    c_flat = centres.reshape(B * K, r)
    u_centres = (u0[:, None, :] + centres @ phi.T).reshape(B * K, -1)
    lzc_c = log_zc(u_centres, model.part, rho_c, rho0,
                   f_extra=f_ext, cat_shift=model.cat_shift(c_flat)).view(B, K)
    lp_c = -0.5 * centres.square().sum(2)
    log_v = lzc_c + lp_c - torch.log(mult)
    w_mode = torch.softmax(log_v, dim=1)
    w_comp = 0.70 * w_mode + 0.30 / float(K)
    log_w_comp = torch.log(w_comp.clamp_min(1e-300))

    # Stratified component selection across the K centres + 1 defensive prior component (index K)
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
        log_q = torch.logaddexp(math.log1p(-defensive) + torch.logsumexp(comp, 2),
                                math.log(defensive) + log_prior)
        z_flat = z.reshape(B * d_cur, r)
        u_flat = (u0[:, None, :] + z @ phi.T).reshape(B * d_cur, -1)
        lzc = log_zc(u_flat, model.part, rho_c, rho0,
                     f_extra=f_ext, cat_shift=model.cat_shift(z_flat)).view(B, d_cur)
        log_w.append(log_prior + lzc - log_q)
    log_w = torch.cat(log_w, 1)
    est = torch.logsumexp(log_w, 1) - math.log(draws)
    w = torch.softmax(log_w, 1)
    ess = 1.0 / w.square().sum(1) / draws
    return est, ess
