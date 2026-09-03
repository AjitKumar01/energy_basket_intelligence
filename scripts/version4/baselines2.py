"""
Three more baselines: nonsymmetric DPP, SHOPPER, and a BEMB-style multinomial set model.

Same rules as baselines.py -- exactly normalised P(S) over the store's assortment,
conditioned on non-empty -- so the numbers are comparable with each other and with
version 4's set component.

NONSYMMETRIC DPP (Gartrell et al. 2019, 2021).  The important addition, and the strongest
competitor to version 4's central claim. A symmetric DPP scores det(L_S) with L positive
semi-definite, which represents REPULSION only; that is why the symmetric variant finished
below a frequency baseline.  A nonsymmetric kernel

    L = D + V V' + B C B',      C block-diagonal with 2x2 blocks [[0, l], [-l, 0]]

adds a skew-symmetric part, and a skew part is exactly what lets a DPP express ATTRACTION
as well.  The normaliser stays exact, by the same determinant identity applied to the
stacked low-rank factor W = [V, B] with middle matrix M = blockdiag(I, C):

    det(I + D + W M W') = det(I + D) . det(I + M W'(I + D)^{-1} W)

If this matches version 4, the case for the enumerated within-category term is much weaker,
because a nonsymmetric DPP reaches both signs of interaction with an exact normaliser and no
Monte Carlo at all.

SHOPPER (Ruiz, Athey, Blei 2020).  The model this whole line of work descends from.  It is
sequential: items are chosen one at a time, each from a softmax over what is left, ending
with a checkout item, and the interaction enters as rho_c'(mean of alpha over the items so
far).  Its likelihood is over ORDERED baskets, so scoring a SET requires summing over
orderings:

    P(S) = sum over the n! orderings of P(ordering)
         = n! * E_{pi uniform}[ P(pi) ],  estimated by sampling orderings

That estimator is unbiased for P(S) and therefore biased LOW for log P(S) by Jensen, so
SHOPPER's number here is a lower bound on what it would score with the sum done exactly.
The bias is reported alongside it rather than hidden: it falls as the number of sampled
orderings rises, and both are printed.

BEMB-STYLE MULTINOMIAL.  BEMB models which item is chosen on a purchase occasion, not which
SET is bought, so it has no set likelihood of its own.  The faithful adaptation is a
conditional draw of n distinct items with weights w_j -- P(S | n) proportional to
prod_{j in S} w_j, normalised by the elementary symmetric polynomial e_n(w) -- with P(n)
from the empirical size distribution. Conditional on n, this is version 4's additive
family (phi = rho_c = 0).  It is NOT literally a nested joint model: the baseline is handed
a trip-independent empirical P(n), whereas version 4 induces a trip-dependent P(n|x)
through one global rho_0(n) and the trip-specific ESP e_n(exp b(x)).  Equality would require
rho_0(n, x) = log e_n(exp b(x)) - log P_emp(n) + const(x), but rho_0 has no x argument.
Therefore the reported size and composition likelihoods must be separated, and a true
nested ablation must zero the interaction blocks inside RaggedModel and re-optimise it.

"""
import itertools
import math

import numpy as np
import torch

from baselines import LinearIndex
from ragged import esp_bucketed, poly_tree, seg_max


class NDPP(torch.nn.Module):
    """Nonsymmetric low-rank DPP: L = diag(d) + V V' + B C B', C skew-symmetric."""

    def __init__(self, J, N, S, rank=16, srank=8, seed=0,
                 interaction_init=0.1, **kw):
        super().__init__()
        self.idx = LinearIndex(J, N, S, seed=seed, **kw)
        g = torch.Generator().manual_seed(seed + 2)
        self.V = torch.nn.Parameter(torch.randn(J, rank, generator=g) * interaction_init)
        self.B = torch.nn.Parameter(torch.randn(J, 2 * srank, generator=g) * interaction_init)
        self.lam = torch.nn.Parameter(torch.ones(srank) * 0.5)
        self.rank, self.srank = rank, srank

    def _C(self):
        """Block-diagonal skew-symmetric middle matrix."""
        C = torch.zeros(2 * self.srank, 2 * self.srank, dtype=self.lam.dtype,
                        device=self.lam.device)
        for i in range(self.srank):
            C[2 * i, 2 * i + 1] = self.lam[i]
            C[2 * i + 1, 2 * i] = -self.lam[i]
        return C

    def loglik(self, d):
        q = self.idx(d["item"], d["st"], d["house"], d["ctx"])
        C = self._C()
        k, m = self.rank, 2 * self.srank
        M = torch.zeros(k + m, k + m, dtype=q.dtype)
        M[:k, :k] = torch.eye(k, dtype=q.dtype)
        M[k:, k:] = C
        out = []
        for b in range(d["B"]):
            msk = d["st"] == b
            dg = torch.exp(q[msk].clamp(-12, 6))
            W = torch.cat([self.V[d["item"][msk]], self.B[d["item"][msk]]], dim=1)
            s = 1.0 + dg
            A = torch.eye(k + m, dtype=q.dtype) + M @ (W / s.unsqueeze(-1)).T @ W
            sign_norm, logdet_norm = torch.linalg.slogdet(A)
            if float(sign_norm.detach()) <= 0:
                raise RuntimeError("NDPP normalizer determinant is not positive")
            log_norm = torch.log(s).sum() + logdet_norm
            sl = d["lslot"][d["lt"] == b] - int(d["off"][b])
            Ws, ds = W[sl], dg[sl]
            L_S = Ws @ M @ Ws.T + torch.diag(ds)
            sign_num, log_num = torch.linalg.slogdet(L_S)
            if float(sign_num.detach()) <= 0:
                raise RuntimeError("NDPP observed principal minor is not positive")
            out.append(log_num - (log_norm + torch.log1p(-torch.exp(-log_norm))))
        return torch.stack(out)


def esp_tree(W, nmax):
    """e_0..e_nmax over a padded [B, P] weight matrix, by a balanced product tree.

    The generating polynomial is prod_j (1 + w_j u) and e_n is its degree-n coefficient.
    The sequential recursion needs P steps -- 5,420 at a dunnhumby store -- which in torch
    means 5,420 kernel launches per batch and was 12x slower than every other baseline.
    Multiplying the degree-1 factors pairwise in a tree needs log2(P) ~ 13 rounds instead,
    at somewhat more arithmetic but far fewer launches.  Padded slots carry w = 0, whose
    factor is (1, 0), the identity -- so no masking is needed.
    """
    B, P = W.shape
    poly = torch.stack([torch.ones_like(W), W], dim=-1)          # [B, P, 2]
    while poly.shape[1] > 1:
        n, d = poly.shape[1], poly.shape[-1]
        if n % 2:                                                 # pad with the identity
            idp = torch.zeros(B, 1, d, dtype=W.dtype, device=W.device)
            idp[:, :, 0] = 1.0
            poly = torch.cat([poly, idp], dim=1)
            n += 1
        a, b = poly[:, 0::2], poly[:, 1::2]
        nd = min(2 * d - 1, nmax + 1)
        out = torch.zeros(B, n // 2, nd, dtype=W.dtype, device=W.device)
        for r in range(min(d, nd)):
            take = min(d, nd - r)
            out[..., r:r + take] = out[..., r:r + take] + a[..., :take] * b[..., r:r + 1]
        poly = out
    return poly[:, 0, :]


class Multinomial(torch.nn.Module):
    """BEMB-style: n distinct items drawn with weights w, size from the empirical law.

    P(S) = P(n) * prod_{j in S} w_j / e_n(w),  e_n the elementary symmetric polynomial.
    """

    def __init__(self, J, N, S, size_law, seed=0, **kw):
        super().__init__()
        self.idx = LinearIndex(J, N, S, seed=seed, **kw)
        p = torch.as_tensor(size_law, dtype=torch.get_default_dtype())
        if p.ndim != 1 or len(p) < 2 or float(p[0]) != 0.0:
            raise ValueError("size law must be a vector on 0..nmax with P(n=0)=0")
        if bool((p < 0).any()) or not torch.isfinite(p).all() or abs(float(p.sum()) - 1) > 1e-10:
            raise ValueError("size law must be finite, non-negative and normalized")
        self.register_buffer("log_pn", torch.where(p > 0, torch.log(p),
                                                   torch.full_like(p, -float("inf"))))

    def loglik(self, d, category_cap=23):
        w = self.idx(d["item"], d["st"], d["house"], d["ctx"])
        wl = self.idx(d["li"], d["lt"], d["house"], d["lctx"])
        B = d["B"]
        n = torch.bincount(d["lt"], minlength=B)
        if bool((n <= 0).any()) or int(n.max()) >= len(self.log_pn):
            raise ValueError("multinomial received a basket outside its non-empty size support")
        num = torch.zeros(B, dtype=w.dtype).index_add_(0, d["lt"], wl)
        M = seg_max(w, d["st"], B)
        wn = torch.exp(w - M[d["st"]])
        if category_cap is None:
            cnt = torch.bincount(d["st"], minlength=B)
            P = int(cnt.max())
            off = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(cnt, 0)[:-1]])
            pos = torch.arange(len(w), device=w.device) - off[d["st"]]
            W = torch.zeros(B * P, dtype=w.dtype, device=w.device).index_copy(
                0, d["st"] * P + pos, wn).view(B, P)
            e = esp_tree(W, int(n.max()))
            coef = e.gather(1, n.unsqueeze(1)).squeeze(1)
        else:
            # Exact denominator on the MAIN MODEL'S support: at most category_cap items
            # from each affinity category.  Form each category's ESP, truncate its degree,
            # then multiply the category polynomials.  This uses the same coverage-complete
            # ragged kernel as the main model, including the 1,774-product residual row.
            ix = d["rix"]
            degree = min(int(category_cap), int(n.max()))
            e = esp_bucketed(wn, ix.row_of, ix.n_rows, degree,
                             ix.row_size, ix.item_pos, parallel=True)
            G = torch.zeros(B * ix.Cpad, degree + 1, dtype=w.dtype, device=w.device)
            G[:, 0] = 1.0
            G = G.index_copy(0, ix.flat_slot, e).view(B, ix.Cpad, degree + 1)
            A = poly_tree(G, int(n.max()))
            coef = A.gather(1, n.unsqueeze(1)).squeeze(1)
        le = torch.log(coef.clamp_min(1e-300)) + n.to(w.dtype) * M
        return self.log_pn[n] + num - le


class Shopper(torch.nn.Module):
    """Sequential choice with an interaction on the running mean of alpha, plus checkout.

    Set probability by averaging over sampled orderings: P(S) = n! E_pi[P(pi)].  Unbiased
    for P(S), so biased low for log P(S); the script reports the estimate at two ordering
    counts so the size of that bias is visible.
    """

    def __init__(self, J, N, S, K=32, Ki=16, seed=0,
                 interaction_init=0.1, **kw):
        super().__init__()
        self.idx = LinearIndex(J, N, S, K=K, seed=seed, **kw)
        g = torch.Generator().manual_seed(seed + 3)
        self.rho = torch.nn.Parameter(torch.randn(J, Ki, generator=g) * interaction_init)
        self.alpha_i = torch.nn.Parameter(torch.randn(J, Ki, generator=g) * interaction_init)
        self.checkout = torch.nn.Parameter(torch.zeros(1))
        self.Ki = Ki

    def _ordered_logprob(self, ps, rho, al, order, max_size=0):
        """Log probability of each explicit ordering, including final checkout."""
        n_orders, n = order.shape
        alive = torch.ones(n_orders, len(ps), dtype=torch.bool, device=ps.device)
        run = torch.zeros(n_orders, self.Ki, dtype=ps.dtype, device=ps.device)
        tot = torch.zeros(n_orders, dtype=ps.dtype, device=ps.device)
        co = self.checkout.to(dtype=ps.dtype, device=ps.device).expand(n_orders, 1)
        for i in range(n):
            u = ps.unsqueeze(0) + (run @ rho.T if i else
                                   torch.zeros_like(alive, dtype=ps.dtype))
            um = u.masked_fill(~alive, -float("inf"))
            den = torch.logsumexp(torch.cat([um, co], dim=1), dim=1)
            j = order[:, i]
            tot = tot + u.gather(1, j.unsqueeze(1)).squeeze(1) - den
            run = (run * i + al[j]) / (i + 1)
            alive.scatter_(1, j.unsqueeze(1), False)
        # A hard maximum is an exact generative support restriction: after max_size items,
        # checkout is forced and therefore has log probability zero.  For shorter baskets
        # the ordinary learned checkout probability remains part of the sequence law.
        if max_size and n == max_size:
            return tot
        u = ps.unsqueeze(0) + run @ rho.T
        den = torch.logsumexp(torch.cat([u.masked_fill(~alive, -float("inf")), co], 1), 1)
        return tot + self.checkout[0] - den

    def loglik(self, d, n_orders=4, gen=None, exact_max_n=0, max_size=120):
        """Monte Carlo set likelihood with deterministic, vectorized orderings.

        The previous implementation used NumPy's process-global RNG and ignored ``gen``;
        the same checkpoint therefore produced a different number on every invocation.
        Orders now share an explicit torch generator, and all orders at one sequence step
        are evaluated in one matrix multiply.
        """
        if n_orders < 1:
            raise ValueError("n_orders must be positive")
        psi = self.idx(d["item"], d["st"], d["house"], d["ctx"])
        out = []
        for b in range(d["B"]):
            msk = d["st"] == b
            ps, it = psi[msk], d["item"][msk]
            rho, al = self.rho[it], self.alpha_i[it]
            sl = d["lslot"][d["lt"] == b] - int(d["off"][b])
            n = len(sl)
            if n == 0 or len(torch.unique(sl)) != n:
                raise ValueError("SHOPPER requires a non-empty set of distinct products")
            if max_size and n > max_size:
                raise ValueError("SHOPPER basket lies outside its forced-checkout support")
            # The sequential model can checkout before selecting an item.  Condition on a
            # non-empty basket, as every other row in the comparison does.
            logden0 = torch.logsumexp(torch.cat([ps, self.checkout.to(ps)]), 0)
            log_nonempty = torch.log1p(-torch.exp((self.checkout[0] - logden0).clamp(max=-1e-12)))
            if exact_max_n and n <= exact_max_n:
                # Exact sum over all n! orderings.  Kept deliberately below n=7: the
                # vectorized utility matrix is [n!, assortment], so exactness is cheap for
                # small baskets and explicitly bounded for large ones.
                order = torch.as_tensor(list(itertools.permutations(sl.tolist())),
                                        dtype=torch.long, device=ps.device)
                lp = torch.logsumexp(
                    self._ordered_logprob(ps, rho, al, order, max_size=max_size), 0)
            else:
                # Random-key permutations are uniform and vectorize cleanly.  Using one
                # order matrix makes a 32/128/512 ladder nested when callers reset the seed.
                keys = torch.rand(n_orders, n, generator=gen, dtype=ps.dtype, device=ps.device)
                order = sl[torch.argsort(keys, dim=1)]
                per = self._ordered_logprob(ps, rho, al, order, max_size=max_size)
                lp = (torch.logsumexp(per, 0) - math.log(n_orders)
                      + float(math.lgamma(n + 1)))
            out.append(lp - log_nonempty)
        return torch.stack(out)


def size_law(D, nmax=120, category_cap=23, prior=0.5):
    """Empirical P(n) on exactly the non-empty support used by the main model."""
    tr = np.flatnonzero(D["trip_split"] == 0)
    lp, lc = D["line_ptr"], D["line_cat"]
    kept = []
    for t in tr:
        lo, hi = int(lp[t]), int(lp[t + 1])
        n = hi - lo
        if 1 <= n <= nmax and (hi <= lo or np.bincount(lc[lo:hi]).max() <= category_cap):
            kept.append(n)
    c = np.zeros(nmax + 1, dtype=np.float64)
    c[1:] = prior
    c += np.bincount(np.asarray(kept), minlength=nmax + 1)[:nmax + 1]
    c[0] = 0.0
    return c / c.sum()
