"""SHOPPER (Ruiz, Athey and Blei, arXiv 1711.03560) in PyTorch: a sequential basket model.

A trip t (household u, week w) builds its basket one item at a time.  At step i, with the
items y_1..y_{i-1} already chosen, the next choice is a softmax over the trip's offered items
not yet chosen plus a checkout item:

    Psi_tci = psi_tc + rho_c . (1 / (i - 1)) sum_{j < i} alpha_{y_j}
    psi_tc  = lambda_c + theta_u . alpha_c - (gamma_u . beta_g(c)) x_tc + delta_w . mu_g(c) + o_tc

x_tc is the item's log price relative to its own average (the paper's normPrice=1), o_tc a
fixed log-availability offset (0 unless the data supply one), and g(c) the item's group, which
shares beta and mu as in the paper's itemGroup file.  gamma_u and beta_g are positive
(softplus).  Checkout has its own lambda, alpha and rho, no price or season, and is not
available at step 1, so every basket is nonempty.  The look-ahead variant is not implemented.

Differences from the released C++/CUDA code, all deliberate:
  * point (MAP) estimates with Adam instead of variational inference; the priors become
    penalties with the same variances (Gaussian) and shapes/rates (gamma) as the defaults;
  * the softmax is exact over the trip's offered items (no one-vs-each bound, no negative
    sampling), so the step probabilities are exact;
  * a basket's probability sums over orderings: log P(S) = log n! + log E_perm P(S, perm),
    estimated with random permutations (exact enumeration for small baskets).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn.functional import softplus


@dataclass
class Trips:
    """A padded batch of trips.

    offered  [B, M] long  item ids of each trip's offered items (padding: any id, masked)
    valid    [B, M] bool  offered-slot mask
    price    [B, M] float log relative price x_tc at each slot
    offset   [B, M] float fixed log-availability offset o_tc
    user     [B] long; week [B] long (season index, -1 for none)
    basket   [B, N] long  slot positions (into M) of the purchased items, -1 padded
    size     [B] long     number of purchased items
    """
    offered: torch.Tensor
    valid: torch.Tensor
    price: torch.Tensor
    offset: torch.Tensor
    user: torch.Tensor
    week: torch.Tensor
    basket: torch.Tensor
    size: torch.Tensor


class Shopper(torch.nn.Module):
    def __init__(self, n_item, n_user, n_week, item_group, K=50, P=10, D=10, init_scale=0.1, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        r = lambda *s: torch.randn(*s, generator=g) * init_scale
        item_group = torch.as_tensor(item_group, dtype=torch.long)
        G = int(item_group.max()) + 1
        self.register_buffer("item_group", item_group)
        self.K, self.P, self.D = K, P, D
        self.lam = torch.nn.Parameter(torch.zeros(n_item))
        self.alpha = torch.nn.Parameter(r(n_item, K))
        self.rho = torch.nn.Parameter(r(n_item, K))
        self.theta = torch.nn.Parameter(r(n_user, K))
        # softplus^-1(0.1) start for the price factors, the prior mean shp/rte = 0.1
        sp_inv = math.log(math.expm1(0.1))
        self.gamma_raw = torch.nn.Parameter(torch.full((n_user, P), sp_inv) + r(n_user, P)) if P else None
        self.beta_raw = torch.nn.Parameter(torch.full((G, P), sp_inv) + r(G, P)) if P else None
        self.delta = torch.nn.Parameter(r(n_week, D)) if D else None
        self.mu = torch.nn.Parameter(r(G, D)) if D else None
        self.ck_lam = torch.nn.Parameter(torch.zeros(()))
        self.ck_alpha = torch.nn.Parameter(r(K))
        self.ck_rho = torch.nn.Parameter(r(K))

    # ---- utilities -------------------------------------------------------------------
    def base(self, t: Trips) -> torch.Tensor:
        """psi_tc at every offered slot [B, M] (padding at -inf)."""
        c = t.offered
        psi = self.lam[c] + (self.theta[t.user][:, None, :] * self.alpha[c]).sum(-1)
        if self.P:
            g = (softplus(self.gamma_raw[t.user])[:, None, :]
                 * softplus(self.beta_raw[self.item_group[c]])).sum(-1)
            psi = psi - g * t.price
        if self.D:
            wk = t.week.clamp(min=0)
            season = (self.delta[wk][:, None, :] * self.mu[self.item_group[c]]).sum(-1)
            psi = psi + torch.where((t.week >= 0)[:, None], season, torch.zeros_like(season))
        psi = psi + t.offset
        return psi.masked_fill(~t.valid, -math.inf)

    def checkout_base(self, t: Trips) -> torch.Tensor:
        return self.ck_lam + self.theta[t.user] @ self.ck_alpha

    # ---- sequence log-probability -----------------------------------------------------
    def sequence_logprob(self, t: Trips, order: torch.Tensor, psi=None, psi_ck=None) -> torch.Tensor:
        """log P(y_1..y_n, checkout) for the basket taken in `order` ([B, N] slot positions,
        -1 padded, first `size` entries used).  Returns [B]."""
        B, M = t.offered.shape
        N = order.shape[1]
        S = N + 1                                            # steps: n items then checkout
        psi = self.base(t) if psi is None else psi
        psi_ck = self.checkout_base(t) if psi_ck is None else psi_ck
        ar = torch.arange(S, device=psi.device)
        live = ar[None, :] <= t.size[:, None]               # steps 0..n are real
        is_item_step = ar[None, :] < t.size[:, None]
        pos = torch.cat([order, order.new_full((B, 1), -1)], 1).clamp(min=0)   # [B, S]
        chosen_alpha = self.alpha[t.offered.gather(1, pos)]  # [B, S, K]
        chosen_alpha = chosen_alpha * is_item_step[..., None]
        csum = torch.cumsum(chosen_alpha, 1) - chosen_alpha  # sum of items before step i
        ctx = csum / ar.clamp(min=1)[None, :, None]          # mean; zero at step 0
        rho = self.rho[t.offered]                            # [B, M, K]
        util = psi[:, None, :] + torch.einsum("bsk,bmk->bsm", ctx, rho)       # [B, S, M]
        # items chosen at earlier steps are no longer available
        onehot = torch.zeros(B, S, M, dtype=torch.bool, device=psi.device)
        onehot.scatter_(2, pos[..., None], is_item_step[..., None])
        taken = (torch.cumsum(onehot.int(), 1) - onehot.int()) > 0
        util = util.masked_fill(taken, -math.inf)
        ck = psi_ck[:, None] + ctx @ self.ck_rho             # [B, S]
        ck = torch.where(ar[None, :] == 0, torch.full_like(ck, -math.inf), ck)
        full = torch.cat([util, ck[..., None]], 2)           # [B, S, M + 1]
        lse = torch.logsumexp(full, 2)
        target = torch.where(is_item_step, pos, torch.full_like(pos, M))
        chosen = full.gather(2, target[..., None]).squeeze(-1)
        step_lp = torch.where(live, chosen - lse, torch.zeros_like(lse))
        return step_lp.sum(1)

    def random_order(self, t: Trips, gen=None) -> torch.Tensor:
        B, N = t.basket.shape
        key = torch.rand(B, N, generator=gen)
        key = key.masked_fill(torch.arange(N)[None, :] >= t.size[:, None], 2.0)
        return t.basket.gather(1, torch.argsort(key, 1))

    @torch.no_grad()
    def basket_logprob(self, t: Trips, perms=256, gen=None, exact_up_to=5) -> torch.Tensor:
        """log P(S) = log n! + log mean over uniformly random orderings of P(S, ordering).
        Exact (all n! orderings) for baskets with n <= exact_up_to."""
        psi, psi_ck = self.base(t), self.checkout_base(t)
        out = torch.empty(t.size.shape[0], dtype=psi.dtype)
        small = t.size <= exact_up_to
        if bool(small.any()):
            import itertools
            for n in torch.unique(t.size[small]).tolist():
                rows = torch.nonzero(small & (t.size == n)).squeeze(1)
                sub = _rows(t, rows)
                lps = [self.sequence_logprob(sub, _apply_perm(sub.basket, p), psi[rows], psi_ck[rows])
                       for p in itertools.permutations(range(n))]
                out[rows] = torch.logsumexp(torch.stack(lps, 1), 1)
        big = torch.nonzero(~small).squeeze(1)
        if len(big):
            sub = _rows(t, big)
            lps = torch.stack([self.sequence_logprob(sub, self.random_order(sub, gen), psi[big], psi_ck[big])
                               for _ in range(perms)], 1)
            out[big] = (torch.logsumexp(lps, 1) - math.log(perms)
                        + torch.lgamma(sub.size.to(psi.dtype) + 1))
        return out

    @torch.no_grad()
    def basket_logprob_is(self, t: Trips, perms=64, gen=None) -> torch.Tensor:
        """log P(S) by importance sampling over orderings.  The proposal adds the basket's
        remaining items in proportion to their current utilities, so the weight of one
        ordering is prod_i [sum_{remaining basket items} e^u / sum_{available + checkout} e^u]
        times P(checkout | full basket); P(S) = E_q[weight] exactly."""
        B, M = t.offered.shape
        R = perms
        rep = lambda x: x.repeat_interleave(R, 0)
        tt = Trips(*(rep(getattr(t, f)) for f in Trips.__dataclass_fields__))
        psi, psi_ck = self.base(tt), self.checkout_base(tt)
        rho = self.rho[tt.offered]
        BR = B * R
        in_basket = torch.zeros(BR, M, dtype=torch.bool)
        rows = torch.arange(BR)[:, None].expand_as(tt.basket)
        ok = tt.basket >= 0
        in_basket[rows[ok], tt.basket[ok]] = True
        taken = torch.zeros(BR, M, dtype=torch.bool)
        asum = torch.zeros(BR, self.K, dtype=psi.dtype)
        logw = torch.zeros(BR, dtype=psi.dtype)
        for i in range(int(tt.size.max()) + 1):
            ctx = asum / max(i, 1)
            util = (psi + (rho * ctx[:, None, :]).sum(-1)).masked_fill(taken, -math.inf)
            ck = psi_ck + ctx @ self.ck_rho
            if i == 0:
                ck = torch.full_like(ck, -math.inf)
            denom = torch.logsumexp(torch.cat([util, ck[:, None]], 1), 1)
            item_step = tt.size > i
            end_step = tt.size == i
            rem = util.masked_fill(~in_basket | taken, -math.inf)
            lse_rem = torch.logsumexp(rem, 1)
            logw = logw + torch.where(item_step, lse_rem - denom, torch.zeros_like(denom))
            logw = logw + torch.where(end_step, ck - denom, torch.zeros_like(denom))
            if not bool(item_step.any()):
                break
            r_idx = torch.nonzero(item_step).squeeze(1)
            probs = torch.softmax(rem[r_idx], 1)
            pick = torch.multinomial(probs, 1, generator=gen).squeeze(1)
            taken[r_idx, pick] = True
            asum[r_idx] += self.alpha[tt.offered[r_idx, pick]]
        logw = logw.view(B, R)
        return torch.logsumexp(logw, 1) - math.log(R)

    # ---- generation ----------------------------------------------------------------------
    @torch.no_grad()
    def sample(self, t: Trips, seed: int, max_size=None) -> torch.Tensor:
        """Draw one basket per trip by Gumbel-max at each step; returns a [B, M] bool mask of
        chosen slots.  The same seed gives the same noise, so two price settings drawn with
        one seed share random numbers."""
        B, M = t.offered.shape
        S = max_size or M
        gen = torch.Generator().manual_seed(int(seed))
        psi, psi_ck = self.base(t), self.checkout_base(t)
        rho = self.rho[t.offered]
        chosen = torch.zeros(B, M, dtype=torch.bool)
        asum = torch.zeros(B, self.K, dtype=psi.dtype)
        n = torch.zeros(B, dtype=psi.dtype)
        done = torch.zeros(B, dtype=torch.bool)
        for i in range(S):
            ctx = asum / n.clamp(min=1)[:, None]
            util = (psi + (rho * ctx[:, None, :]).sum(-1)).masked_fill(chosen, -math.inf)
            ck = psi_ck + ctx @ self.ck_rho
            ck = torch.full_like(ck, -math.inf) if i == 0 else ck
            noise = -torch.log(-torch.log(torch.rand(B, M + 1, generator=gen, dtype=psi.dtype)
                                          .clamp(1e-300, 1 - 1e-16)))
            full = torch.cat([util, ck[:, None]], 1) + noise
            pick = full.argmax(1)
            stop = (pick == M) | done | torch.isinf(full.max(1).values)
            take = ~stop
            rows = torch.nonzero(take).squeeze(1)
            chosen[rows, pick[rows]] = True
            asum[rows] += self.alpha[t.offered[rows, pick[rows]]]
            n[rows] += 1
            done = done | stop
            if bool(done.all()):
                break
        return chosen

    # ---- priors (MAP penalties), per training basket -------------------------------------
    def neg_log_prior(self, s2=1.0, s2_season=0.01, shp=100.0, rte=1000.0):
        gauss = sum((p.square().sum() / (2 * s2)) for p in
                    (self.lam, self.alpha, self.rho, self.theta, self.ck_lam, self.ck_alpha, self.ck_rho))
        if self.D:
            gauss = gauss + (self.delta.square().sum() + self.mu.square().sum()) / (2 * s2_season)
        gam = 0.0
        if self.P:
            for raw in (self.gamma_raw, self.beta_raw):
                x = softplus(raw)
                gam = gam - ((shp - 1.0) * torch.log(x) - rte * x).sum()
        return gauss + gam


def _apply_perm(basket, perm):
    idx = torch.as_tensor(list(perm) + list(range(len(perm), basket.shape[1])), dtype=torch.long)
    return basket[:, idx]


def _rows(t: Trips, rows):
    return Trips(*(getattr(t, f)[rows] for f in Trips.__dataclass_fields__))
