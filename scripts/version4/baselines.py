"""
Baselines that produce a properly normalised P(S) on the same support.

The comparison has to be like for like or it means nothing, so every model here obeys the
same three rules as the version-4 model:

  * the support is the store's assortment from data.py -- the same set of products
  * the probability is over the SET, exactly normalised, not a per-item score or a top-k
  * it is conditioned on a non-empty basket, and the conditioning is exact

That rules out most of what is usually reported for basket data (top-k recall, next-item
accuracy, per-line likelihood against sampled negatives) and leaves models whose
normaliser is tractable in closed form. Two are implemented in this module.

INDEPENDENT BERNOULLI.  pi_j = sigmoid(b_j), with the same broad covariate families as
version 4 -- intercept, household taste, price, display, mailer, seasonality, store.  This
is a separately parameterised baseline, not an exact nested ablation: the current main
model also constrains the price factors, separates common and relative price, and applies
identifiability/pooling transforms.  An exact nested comparison must use RaggedModel itself
with phi and rho_c zeroed and frozen (fit.py --zero-phi --zero-rho-c).

LOW-RANK DPP.  P(S) proportional to det(L_S) with L = V V' + diag(d).  The normaliser is
det(L + I), exact, and computable in O(J k^2) by the identity

    det(I + D + V V') = det(I + D) * det(I_k + V' (I + D)^{-1} V)

This is the strongest principled competitor and the interesting one: a DPP models
REPULSION natively and cannot represent complementarity, which is the exact complement of
version 4's Gaussian latent, whose interaction is positive semi-definite and so bounded in
how much repulsion it can express.  Whichever wins, the direction is informative.

The empty-set conditioning is the same identity in both families: det(L_empty) = 1, so
P(empty) = 1 / det(L + I) and the conditional normaliser is det(L + I) - 1, exactly as
version 4's is Z - 1.

Reported per basket AND per line, because the two families disagree about what a trip is
and quoting only one invites the wrong comparison.
"""
import numpy as np
import torch

from ragged import RaggedIndex, esp_bucketed, poly_tree, seg_max


class Batches:
    """Assortment slots and purchased lines with the same observed version-4 features."""

    def __init__(self, D, F):
        self.D, self.F = D, F
        self.C = int(D["n_cat"])
        self.J, self.S = int(D["n_item"]), int(D["n_store"])
        self.ptr, self.items, self.lptr = (D["store_cat_ptr"], D["store_items"],
                                           D["line_ptr"])
        # Store assortments and their category rows do not change between minibatches.
        # Building them through C Python loops per trip and reconstructing a
        # (trip,item)->slot dictionary over ~130k slots consumed about half of every
        # baseline update.  Cache the local layouts and a compact inverse position map;
        # observed features remain gathered afresh for each trip/context.
        self.store_layout = []
        self.store_position = np.full((self.S, self.J), -1, dtype=np.int32)
        for store in range(self.S):
            item_parts, row_parts, categories = [], [], []
            row = 0
            base = store * self.C
            for category in range(self.C):
                lo, hi = int(self.ptr[base + category]), int(self.ptr[base + category + 1])
                if hi <= lo:
                    continue
                values = np.asarray(self.items[lo:hi], dtype=np.int64)
                item_parts.append(values)
                row_parts.append(np.full(len(values), row, dtype=np.int64))
                categories.append(category)
                row += 1
            items = np.concatenate(item_parts)
            row_of = np.concatenate(row_parts)
            row_cat = np.asarray(categories, dtype=np.int64)
            self.store_position[store, items] = np.arange(len(items), dtype=np.int32)
            self.store_layout.append((items, row_of, row_cat))

    def make(self, trips):
        D, C = self.D, self.C
        it_l, slot_trip, row_of, row_trip, row_cat = [], [], [], [], []
        nrow = 0
        stores = np.asarray(D["trip_store"][trips], dtype=np.int64)
        for bi, t in enumerate(trips):
            items, local_rows, categories = self.store_layout[stores[bi]]
            it_l.append(items)
            slot_trip.append(np.full(len(items), bi, dtype=np.int64))
            row_of.append(local_rows + nrow)
            row_trip.append(np.full(len(categories), bi, dtype=np.int64))
            row_cat.append(categories)
            nrow += len(categories)
        item = torch.as_tensor(np.concatenate(it_l), dtype=torch.long)
        st = torch.as_tensor(np.concatenate(slot_trip), dtype=torch.long)
        # Keep the category rows as part of the batch contract.  The main model caps the
        # number of products selected from one category, so a baseline that discards this
        # structure is normalized on a different support even when it sees the same items.
        rix = RaggedIndex(item, np.concatenate(row_of), np.concatenate(row_trip),
                          np.concatenate(row_cat), len(trips))
        if not torch.equal(rix.item, item) or not torch.equal(rix.item_trip, st):
            raise RuntimeError("baseline assortment and ragged category index disagree")
        store = torch.as_tensor(stores, dtype=torch.long)
        day = torch.as_tensor(D["trip_day"][trips], dtype=torch.long)
        week = torch.as_tensor(D["trip_week"][trips], dtype=torch.long)
        house = torch.as_tensor(D["trip_user"][trips], dtype=torch.long)
        dlp, dsp, mlr = self.F.gather(item, store[st], day[st], week[st])
        ctx = dict(dlp=dlp.double(), disp=dsp.double(), mail=mlr.double(),
                   week=(week[st] - 1) % 52, store=store[st])
        li, lt = [], []
        for bi, t in enumerate(trips):
            a, b = int(self.lptr[t]), int(self.lptr[t + 1])
            li.append(D["line_item"][a:b])
            lt.append(np.full(b - a, bi, np.int64))
        li_array, lt_array = np.concatenate(li), np.concatenate(lt)
        LI = torch.as_tensor(li_array, dtype=torch.long)
        LT = torch.as_tensor(lt_array, dtype=torch.long)
        dl2, ds2, ml2 = self.F.gather(LI, store[LT], day[LT], week[LT])
        lctx = dict(dlp=dl2.double(), disp=ds2.double(), mail=ml2.double(),
                    week=(week[LT] - 1) % 52, store=store[LT])
        # position of each purchased line within its trip's slot block
        sizes = np.asarray([len(self.store_layout[s][0]) for s in stores], dtype=np.int64)
        off_array = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(sizes)])
        off = torch.as_tensor(off_array, dtype=torch.long)
        local_slot = self.store_position[stores[lt_array], li_array].astype(np.int64)
        if bool(np.any(local_slot < 0)):
            raise RuntimeError("purchased baseline item is absent from its assortment")
        lslot = torch.as_tensor(off_array[lt_array] + local_slot, dtype=torch.long)
        return dict(item=item, st=st, ctx=ctx, house=house, B=len(trips),
                    li=LI, lt=LT, lctx=lctx, lslot=lslot, off=off, rix=rix)


class LinearIndex(torch.nn.Module):
    """A common additive utility layer shared by all external basket baselines.

    It covers the same observed covariates as the no-recency main runs, but deliberately
    remains the baseline parameterisation.  It must not be called byte-for-byte identical
    to the current RaggedModel index; that model has additional price constraints and
    centring/pooling transforms.
    """

    def __init__(self, J, N, S, K=32, Kp=8, Kt=8, Ks=4, seed=0,
                 taste_init=0.3):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.lam = torch.nn.Parameter(torch.zeros(J))
        self.alpha = torch.nn.Parameter(torch.randn(J, K, generator=g) * taste_init)
        self.theta = torch.nn.Parameter(torch.randn(N, K, generator=g) * taste_init)
        self.gamma = torch.nn.Parameter(torch.randn(N, Kp, generator=g) * 0.1)
        self.beta = torch.nn.Parameter(torch.randn(J, Kp, generator=g) * 0.1)
        self.w_dsp = torch.nn.Parameter(torch.zeros(J))
        self.w_mlr = torch.nn.Parameter(torch.zeros(J))
        self.mu = torch.nn.Parameter(torch.randn(J, Kt, generator=g) * 0.1)
        self.delta = torch.nn.Parameter(torch.randn(52, Kt, generator=g) * 0.1)
        self.zeta = torch.nn.Parameter(torch.randn(J, Ks, generator=g) * 0.1)
        self.xi = torch.nn.Parameter(torch.randn(S, Ks, generator=g) * 0.1)

    def forward(self, it, trip, house, c):
        hh = house[trip]
        b = self.lam[it] + (self.theta[hh] * self.alpha[it]).sum(-1)
        b = b - (self.gamma[hh] * self.beta[it]).sum(-1) * c["dlp"]
        b = b + self.w_dsp[it] * c["disp"] + self.w_mlr[it] * c["mail"]
        b = b + (self.mu[it] * self.delta[c["week"]]).sum(-1)
        b = b + (self.zeta[it] * self.xi[c["store"]]).sum(-1)
        return b


class Bernoulli(torch.nn.Module):
    """Independent purchase, exactly normalized on a bounded non-empty support.

    Conditional on ``1 <= |S| <= nmax``, an independent Bernoulli model has

        P(S) = prod_{j in S} odds_j / sum_{r=1}^{nmax} e_r(odds).

    The factors ``prod_j (1-p_j)`` cancel.  Computing the bounded denominator matters for
    stores with few training visits: Laplace smoothing can otherwise assign material mass
    above the version-4 limit even though no observed basket is that large.
    """

    def __init__(self, J, N, S, **kw):
        super().__init__()
        self.idx = LinearIndex(J, N, S, **kw)

    def loglik(self, d, nmax=120, category_cap=120):
        b_slot = self.idx(d["item"], d["st"], d["house"], d["ctx"])
        b_line = self.idx(d["li"], d["lt"], d["house"], d["lctx"])
        n = torch.bincount(d["lt"], minlength=d["B"])
        if bool((n <= 0).any()) or int(n.max()) > nmax:
            raise ValueError("Bernoulli basket lies outside its bounded non-empty support")
        pos = torch.zeros(d["B"], dtype=b_slot.dtype, device=b_slot.device).index_add_(
            0, d["lt"], b_line)

        # Shift every trip so all odds passed to the polynomial recursion are <= 1.
        # The degree-r coefficient then receives exp(r*M) on the log scale.
        M = seg_max(b_slot, d["st"], d["B"])
        odds = torch.exp(b_slot - M[d["st"]])
        degree = int(nmax)
        if category_cap is None or int(category_cap) >= degree:
            # With |S|<=degree, a per-category cap >=degree is algebraically redundant:
            # prod_c prod_{j in c}(1+w_j x) = prod_j(1+w_j x).  Evaluate that product
            # once over each trip's catalogue.  The native tree has the same
            # subtraction-free forward polynomial and exact reverse-mode derivative as
            # the category-factorized path, but avoids a second polynomial tree.
            row_size = d["off"][1:] - d["off"][:-1]
            item_pos = (torch.arange(len(d["item"]), device=b_slot.device)
                        - d["off"][d["st"]])
            try:
                A = esp_bucketed(
                    odds.unsqueeze(0), d["st"], d["B"], degree,
                    row_size, item_pos, native=True)[0]
            except (ImportError, RuntimeError):
                # A fresh clone has no extension before the pipeline build.  A worktree
                # whose PyTorch environment changed can also contain an ABI-stale local
                # binary.  The subtraction-free tree is exact, so falling back here
                # preserves both the objective and gradients until --force rebuilds it.
                A = esp_bucketed(
                    odds, d["st"], d["B"], degree,
                    row_size, item_pos, parallel=True)
        else:
            ix = d["rix"]
            per_row_degree = min(int(category_cap), degree)
            e = esp_bucketed(odds, ix.row_of, ix.n_rows, per_row_degree,
                             ix.row_size, ix.item_pos, parallel=True)
            G = torch.zeros(d["B"] * ix.Cpad, degree + 1, dtype=b_slot.dtype,
                            device=b_slot.device)
            G[:, 0] = 1.0
            G[:, :per_row_degree + 1] = G[:, :per_row_degree + 1].index_copy(
                0, ix.flat_slot, e)
            A = poly_tree(G.view(d["B"], ix.Cpad, degree + 1), degree)
        degrees = torch.arange(degree + 1, dtype=b_slot.dtype, device=b_slot.device)
        log_terms = torch.log(A.clamp_min(1e-300)) + M[:, None] * degrees[None, :]
        log_norm = torch.logsumexp(log_terms[:, 1:], dim=1)
        return pos - log_norm


class DPP(torch.nn.Module):
    """Low-rank determinantal point process.  L = V V' + diag(d), exact normaliser."""

    def __init__(self, J, N, S, rank=16, interaction_init=0.1, **kw):
        super().__init__()
        self.idx = LinearIndex(J, N, S, **kw)
        g = torch.Generator().manual_seed(kw.get("seed", 0) + 1)
        self.V = torch.nn.Parameter(torch.randn(J, rank, generator=g) * interaction_init)
        self.rank = rank

    def loglik(self, d):
        q_slot = self.idx(d["item"], d["st"], d["house"], d["ctx"])
        out = []
        for b in range(d["B"]):
            m = d["st"] == b
            dg = torch.exp(q_slot[m])                       # diag of L, positive
            Vb = self.V[d["item"][m]]
            # log det(I + D + V V') = sum log(1+d) + logdet(I_k + V'(I+D)^-1 V)
            s = 1.0 + dg
            A = torch.eye(self.rank, dtype=dg.dtype) + Vb.T @ (Vb / s.unsqueeze(-1))
            log_norm = torch.log(s).sum() + torch.linalg.slogdet(A)[1]
            sl = d["lslot"][d["lt"] == b] - int(d["off"][b])
            Vs, ds = Vb[sl], dg[sl]
            L_S = Vs @ Vs.T + torch.diag(ds)
            log_num = torch.linalg.slogdet(L_S)[1]
            # P(empty) = 1/det(L+I), so the non-empty normaliser is det(L+I) - 1
            out.append(log_num - (log_norm + torch.log1p(-torch.exp(-log_norm))))
        return torch.stack(out)
