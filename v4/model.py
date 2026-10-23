"""Parameters, product utilities and the energy of a basket.

Utility of product j on a trip of household h at store s on day d (week w):

    b_j = lam_j                                   product popularity
        + theta_h . alpha_j                       household taste (low rank K)
        - g_hj * price term                       price, g_hj = softplus(gamma_h) . softplus(beta_j) >= 0
        + w_dsp_j * display + w_mlr_j * mailer    promotions
        + mu_j . delta_w                          season
        + zeta_j . xi_s                           store
        + psi_j . rec                             sub-commodity recency (optional)
        + log availability                        fixed offset (optional, canonical bundles)
        + theta_h . u_c(j)                        household taste per category (M4a, optional)
        + psi_item_c(j) . rec_item                product purchase history (optional)

Price term.  With dlp = log price deviation and m = the trip's mean dlp over its assortment,

    g_hj * [m + kappa (dlp - m)],   kappa = softplus(price_kappa)

so kappa scales the *relative* price.  With a discount panel, dsc = centred discount depth,
the discount part of dlp gets its own multiplier and a slope shared by all households and
products (the "discount split"):

    + g_hj * (softplus(disc_kappa) - kappa) * dsc  +  disc_common * dsc

With disc_kappa = price_kappa and disc_common = 0 this is exactly the one-slope model.

Energy of a basket S (n products, n_c of them in category c, k categories present):

    E(S) = sum_{j in S} b_j + sum_{i<j in S} phi_i . phi_j
           - sum_c [rho_c C(n_c, 2) + f_c(n_c)]                 f_c: M1, free counts 2, 3, >=4
           + M2 presence correction                               see energy()
           - rho_0(n)       or  - rho_0(k) with M3 (size potential on categories present)
           + sum_c n_c * (lambda_c - 1) * IV_price_c              M4c, see iv_shift()

Version-4 blocks are optional parameters (None when absent): f_extra (M1), psi_cat (M2),
u_cat (M4a), psi_item (purchase history), iv_slope_raw (M4c), hist_size (history size
term); M3 is the flag ``size_on_categories``.  The exact additive base model has none of
them and phi = 0.

History centring (``centre_history`` = None, "mean" or "inclusive").  The history part of
the utility, h_tj = psi_j . rec + psi_item_c(j) . rec_item, grows with the household's
history (more products "ever bought"), and raising many utilities also raises basket size,
so without centring the predicted size drifts upward with calendar time.  With centring every
utility of trip t is shifted by the same number

    o_t = c_t + hist_size * log(1 + M_t),   M_t = products ever bought before the trip,

    "mean"       c_t = mean_{j offered on t} h_tj
    "inclusive"  c_t = log sum_j exp(b0_tj + h_tj) - log sum_j exp(b0_tj),  b0 = b without h

"inclusive" removes the history's effect on the trip's total attraction (its inclusive
value), so a wider spread of h across products does not raise size either; "mean" removes
only its average.

so the history decides *which* products are bought, while its effect on *how many* is the
learned term n * hist_size * log(1 + M_t).  A shift shared by all products of a trip is a
term -n o_t in E(S): the law given the basket size and all within-trip comparisons are
unchanged, and the normaliser stays exact.
"""
from __future__ import annotations

import math

import torch
from torch.nn.functional import softplus

PRICE_KEYS = ("dlp", "dlp_bar", "disp", "mail", "dsc")
OPTIONAL_PARAMETERS = ("disc_kappa", "disc_common", "f_extra", "psi_cat", "u_cat",
                       "psi_item", "iv_slope_raw", "hist_size")
HISTORY_KEYS = ("rec", "rec_item")
# Parameters of an old quantity model and size factorisation that older checkpoints carry;
# the basket law does not use them.
LEGACY_KEYS = ("a_q", "gamma_q", "beta_q", "log_r", "factored_size_enabled",
               "factored_size_log_p")


class BasketModel(torch.nn.Module):
    """J products, N households, C categories, S stores.

    K   taste rank (theta, alpha); the last taste coordinate is a household size shift
    Kz  columns of phi (only the first ``rank`` are used; the rest stay zero)
    Kp  rank of the household x product price slopes
    nmax largest basket size; R largest count per category (here R = nmax)
    """

    def __init__(self, J, N, C, *, K, Kz, Kp, nmax, R, S, seed=0, taste_init=0.3,
                 n_week=53, Kt=8, Ks=4):
        super().__init__()
        if torch.get_default_dtype() != torch.float64:
            raise RuntimeError("set torch.set_default_dtype(torch.float64) before building the model")
        if K < 2:
            raise ValueError("K must be at least 2 (one coordinate is the household size shift)")
        g = torch.Generator().manual_seed(seed)
        self.J, self.N, self.C, self.S = J, N, C, S
        self.K, self.Kz, self.nmax, self.R = K, Kz, nmax, R
        # The random draws below keep the order of the original implementation so that a
        # given seed gives the same starting values.
        self.lam = torch.nn.Parameter(torch.zeros(J))
        self.alpha = torch.nn.Parameter(torch.randn(J, K, generator=g) * taste_init)
        self.theta = torch.nn.Parameter(torch.randn(N, K, generator=g) * taste_init)
        self.phi = torch.nn.Parameter(torch.randn(J, Kz, generator=g) * 0.0)
        self.rho_c = torch.nn.Parameter(torch.zeros(C))
        self.rho_0_free = torch.nn.Parameter(torch.zeros(nmax))
        self.price_kappa = torch.nn.Parameter(torch.tensor(0.5413))   # softplus = 1.0
        self.gamma = torch.nn.Parameter(torch.randn(N, Kp, generator=g) * 0.1 - 3.3)
        self.beta = torch.nn.Parameter(torch.randn(J, Kp, generator=g) * 0.1 - 3.3)
        self.w_dsp = torch.nn.Parameter(torch.zeros(J))
        self.w_mlr = torch.nn.Parameter(torch.zeros(J))
        self.mu = torch.nn.Parameter(torch.randn(J, Kt, generator=g) * 0.1)
        self.delta = torch.nn.Parameter(torch.randn(n_week, Kt, generator=g) * 0.1)
        self.psi = torch.nn.Parameter(torch.zeros(J, 4))
        self.zeta = torch.nn.Parameter(torch.randn(J, Ks, generator=g) * 0.1)
        self.xi = torch.nn.Parameter(torch.randn(S, Ks, generator=g) * 0.1)
        self.register_buffer("cat_of", torch.zeros(J, dtype=torch.long))
        for name in OPTIONAL_PARAMETERS:
            setattr(self, name, None)
        self.size_on_categories = False      # M3
        self.centre_history = None           # None, "mean" or "inclusive" (module docstring)
        self.rank = 0                        # active columns of phi
        self.house = None                    # [B] households of the current batch
        self.ctx = None                      # features of the current batch (Batch.bind)

    # ---- small pieces ---------------------------------------------------------------------
    def rho_0(self):
        """Size potential rho_0(n), n = 0..nmax, with rho_0(0) = 0."""
        return torch.cat([torch.zeros(1, dtype=self.rho_0_free.dtype), self.rho_0_free])

    def pair_feature(self, count):
        """C(n, 2) = n (n - 1) / 2, the number of product pairs among n (n capped at nmax)."""
        k = count.clamp(max=self.nmax)
        return k * (k - 1) / 2.0

    def phi_r(self):
        return self.phi[:, :self.rank]

    def price_coefficients(self, it, trip):
        """g_hj >= 0 for product ``it`` on batch trip ``trip``."""
        return (softplus(self.gamma[self.house[trip]]) * softplus(self.beta[it])).sum(-1)

    # Context factors are centred so that the per-product constant lives only in lam: a
    # constant in delta, xi or theta could otherwise move into mu, zeta or alpha (the energy
    # is unchanged) and the penalties would decide where.  The mean is detached so a
    # minibatch gradient does not touch households outside it; project_context_gauges()
    # re-centres the stored values after every step.
    def theta_c(self):
        return self.theta - self.theta.mean(0, keepdim=True).detach()

    def delta_c(self):
        return self.delta - self.delta.mean(0, keepdim=True).detach()

    def xi_c(self):
        return self.xi - self.xi.mean(0, keepdim=True).detach()

    @torch.no_grad()
    def project_context_gauges(self):
        self.theta.sub_(self.theta.mean(0, keepdim=True))
        self.delta.sub_(self.delta.mean(0, keepdim=True))
        self.xi.sub_(self.xi.mean(0, keepdim=True))
        self.alpha[:, :-1].sub_(self.alpha[:, :-1].mean(0, keepdim=True))
        self.alpha[:, -1].fill_(1.0)

    @torch.no_grad()
    def project_rho_c(self, floor=-1.5):
        """Bound within-category attraction: exp(-rho_c C(n,2)) overflows for large
        negative rho_c."""
        self.rho_c.clamp_(min=floor)

    # ---- utilities ---------------------------------------------------------------------------
    def utilities(self, it, trip, c):
        """b at products ``it`` of batch trips ``trip`` with features ``c`` (None: taste only)."""
        hh = self.house[trip]
        theta = self.theta_c()
        # theta_h . alpha_j = size shift theta_h[-1] + centred composition taste.
        alpha = self.alpha[:, :-1]
        alpha = alpha - alpha.mean(0, keepdim=True).detach()
        b = self.lam[it] + theta[hh, -1] + (theta[hh, :-1] * alpha[it]).sum(-1)
        if c is not None:
            g = self.price_coefficients(it, trip)
            mean = c["dlp_bar"][trip]
            kappa = softplus(self.price_kappa)
            b = b - g * (mean + kappa * (c["dlp"] - mean))
            if "dsc" in c and self.disc_kappa is not None:
                b = b - g * (softplus(self.disc_kappa) - kappa) * c["dsc"]
            if "dsc" in c and self.disc_common is not None:
                b = b - self.disc_common * c["dsc"]
            b = b + self.w_dsp[it] * c["disp"] + self.w_mlr[it] * c["mail"]
            b = b + (self.mu[it] * self.delta_c()[c["week"]]).sum(-1)
            b = b + (self.zeta[it] * self.xi_c()[c["store"]]).sum(-1)
            if "log_avail" in c:
                b = b + c["log_avail"]
        if self.u_cat is not None:
            b = b + (theta[hh] * self.u_cat[self.cat_of[it]]).sum(-1)
        h = self.history_utilities(it, c)
        return b if h is None else b + h

    def history_utilities(self, it, c):
        """h = psi_j . rec + psi_item_c(j) . rec_item at products ``it`` (None without history)."""
        h = None
        if c is not None and "rec" in c:
            h = (self.psi[it] * c["rec"]).sum(-1)
        if c is not None and "rec_item" in c and self.psi_item is not None:
            part = (self.psi_item[self.cat_of[it]] * c["rec_item"]).sum(-1)
            h = part if h is None else h + part
        return h

    def history_offset(self, ix, b=None):
        """o_t per trip of the bound batch (module docstring), or None without centring.
        ``b``: the slot utilities before centring, if already computed ("inclusive" needs them)."""
        if not self.centre_history:
            return None
        h = self.history_utilities(ix.item, self.ctx)
        if h is None:
            return None
        if self.centre_history == "mean":
            count = torch.bincount(ix.item_trip, minlength=ix.B).double()
            o = torch.zeros(ix.B).index_add_(0, ix.item_trip, h) / count
        elif self.centre_history == "inclusive":
            b = self.utilities(ix.item, ix.item_trip, self.ctx) if b is None else b
            o = _trip_logsumexp(ix, b) - _trip_logsumexp(ix, b - h)
        else:
            raise ValueError(f"unknown centre_history {self.centre_history!r}")
        if self.hist_size is not None and "rec_item" in self.ctx:
            ever = torch.zeros(ix.B).index_add_(0, ix.item_trip, self.ctx["rec_item"][:, 0])
            o = o + self.hist_size * torch.log1p(ever)
        return o

    def slot_utilities(self, ix):
        """b at every offered slot of the bound batch (minus the trip's history offset)."""
        b = self.utilities(ix.item, ix.item_trip, self.ctx)
        o = self.history_offset(ix, b)
        return b if o is None else b - o[ix.item_trip]

    def slot_utilities_given_z(self, ix, z, beta=1.0, b=None):
        """Utilities of the law of S given the latent z (B x rank):

            b - beta |phi_j|^2 / 2 + sqrt(beta) z . phi_j.

        Writing exp(sum_{i<j} phi_i.phi_j) = E_z exp(z . sum_j phi_j - sum_j |phi_j|^2 / 2),
        z ~ N(s, I) given S, removes the pairwise term.  beta in [0, 1] tempers it.
        ``b``: the slot utilities, if already computed."""
        ph = self.phi_r()[ix.item]
        u = (self.slot_utilities(ix) if b is None else b) - 0.5 * beta * ph.square().sum(-1)
        if z is not None:
            u = u + math.sqrt(beta) * (z[ix.item_trip] * ph).sum(-1)
        return u

    # ---- Version-4 category terms ------------------------------------------------------------
    def presence_shift(self, ix, z, beta=1.0):
        """M2, per row: sqrt(beta) z . psi_c - beta |psi_c|^2 / 2, charged per *extra*
        product of category c (see basket_field)."""
        if self.psi_cat is None or beta == 0.0:
            return None
        psi = self.psi_cat[ix.row_cat]
        shift = -0.5 * beta * psi.square().sum(-1)
        if z is not None:
            shift = shift + math.sqrt(beta) * (z[ix.row_trip] * psi).sum(-1)
        return shift

    def iv_lambda(self):
        """M4c lambda_c = sigmoid(iv_slope_raw) in (0, 1), or None."""
        return None if self.iv_slope_raw is None else torch.sigmoid(self.iv_slope_raw)

    @torch.no_grad()
    def iv_instrument(self, ix):
        """Household-invariant category price/promotion inclusive-value shift per row,
        centred within each category of the batch so learning lambda_c is orthogonal to
        category intercepts and free of in-sample household taste/history selection bias."""
        g_bar = (softplus(self.gamma).mean(0, keepdim=True) * softplus(self.beta[ix.item])).sum(-1)
        mean = self.ctx["dlp_bar"][ix.item_trip]
        kappa = softplus(self.price_kappa)
        delta = -g_bar * (mean + kappa * (self.ctx["dlp"] - mean))
        if "dsc" in self.ctx and self.disc_kappa is not None:
            delta = delta - g_bar * (softplus(self.disc_kappa) - kappa) * self.ctx["dsc"]
        if "dsc" in self.ctx and self.disc_common is not None:
            delta = delta - self.disc_common * self.ctx["dsc"]
        delta = delta + self.w_dsp[ix.item] * self.ctx["disp"] + self.w_mlr[ix.item] * self.ctx["mail"]
        b0 = self.lam[ix.item]
        div = _row_logsumexp(ix, b0 + delta) - _row_logsumexp(ix, b0)
        count = torch.bincount(ix.row_cat, minlength=self.C).clamp_min(1).double()
        mean_c = torch.zeros(self.C).index_add_(0, ix.row_cat, div) / count
        return div - mean_c[ix.row_cat]

    def iv_shift(self, ix, b_slot=None):
        """M4c, per row: (lambda_c - 1) * [IV_c(actual) - IV_c(regular)].

        IV_c = log sum_{j in c} exp b_j, the category's inclusive value; "regular" is the
        same trip with no price deviation, display, mailer or discount.  The term is charged
        per bought product of the category (equivalent to shifting every slot utility in
        category c by iv_c), so each degree k >= 1 of the category polynomial scales with
        k * lambda_c: a price cut moves choice *within* its category while only a fraction
        lambda_c of its inclusive-value gain expands the category.  It depends on the
        context only, so the normaliser stays exact."""
        lam = self.iv_lambda()
        if lam is None:
            return None
        if b_slot is None:
            b_slot = self.slot_utilities(ix)
        regular = {k: (torch.zeros_like(v) if k in PRICE_KEYS else v) for k, v in self.ctx.items()}
        b_regular = self.utilities(ix.item, ix.item_trip, regular)
        o = self.history_offset(ix)                        # the same for both, so it cancels
        if o is not None:
            b_regular = b_regular - o[ix.item_trip]
        iv = _row_logsumexp(ix, b_slot) - _row_logsumexp(ix, b_regular)
        lam_c = lam[ix.row_cat]
        if not self.iv_slope_raw.requires_grad:
            return (lam_c - 1.0) * iv
        inst = self.iv_instrument(ix)
        return (lam_c - 1.0).detach() * iv + (lam_c - lam_c.detach()) * inst

    def basket_field(self, ix, X):
        """s(S) = sum_{j in S} phi_j, minus psi_c per extra product of a category with M2.
        The Gibbs chain draws z ~ N(s(S), I)."""
        x = X.double()
        s = torch.zeros(ix.B, self.rank).index_add_(0, ix.item_trip, self.phi_r()[ix.item] * x[:, None])
        if self.psi_cat is not None:
            extra = (torch.zeros(ix.n_rows).index_add_(0, ix.row_of, x) - 1.0).clamp_min(0.0)
            s = s - torch.zeros(ix.B, self.rank).index_add_(
                0, ix.row_trip, self.psi_cat[ix.row_cat] * extra[:, None])
        return s

    # ---- energy of observed baskets -----------------------------------------------------------
    def additive_energy(self, line_item, line_trip, line_cat, B, line_ctx):
        """sum b + sum_{i<j} phi_i.phi_j - sum_c rho_c C(n_c, 2) - rho_0(n)."""
        b = self.utilities(line_item, line_trip, line_ctx)
        lin = torch.zeros(B).index_add_(0, line_trip, b)
        v = torch.zeros(B, self.Kz).index_add_(0, line_trip, self.phi[line_item])
        sq = torch.zeros(B).index_add_(0, line_trip, (self.phi[line_item] ** 2).sum(-1))
        pair = 0.5 * ((v * v).sum(-1) - sq)
        n_c = torch.bincount(line_trip * self.C + line_cat, minlength=B * self.C).view(B, self.C).double()
        category = (self.rho_c.unsqueeze(0) * self.pair_feature(n_c)).sum(-1)
        n = torch.bincount(line_trip, minlength=B)
        if B and int(n.max()) > self.nmax:
            raise ValueError(f"a basket of size {int(n.max())} is outside 1..{self.nmax}")
        return lin + pair - category - self.rho_0()[n]

    def energy(self, batch, iv=None):
        """E(S) of each observed basket in a bound batch, with every Version-4 term present.
        ``iv``: the M4c shift of the batch, if already computed."""
        e = self.additive_energy(batch.line_item, batch.line_trip, batch.line_cat,
                                 batch.ix.B, batch.line_ctx)
        o = self.history_offset(batch.ix)
        if o is not None:                                  # -n o_t: every bought product shifted by o_t
            e = e - torch.bincount(batch.line_trip, minlength=batch.ix.B).double() * o
        iv = self.iv_shift(batch.ix) if iv is None else iv
        if (self.f_extra is None and self.psi_cat is None and not self.size_on_categories
                and iv is None):
            return e
        ix = batch.ix
        n_row = torch.zeros(ix.n_rows).index_add_(0, ix.row_of, batch.observed.double())
        if self.f_extra is not None:                       # M1: count 2, 3, >= 4
            f = self.f_extra[ix.row_cat]
            pen = ((n_row == 2).double() * f[:, 0] + (n_row == 3).double() * f[:, 1]
                   + (n_row >= 4).double() * f[:, 2])
            e = e - torch.zeros(ix.B).index_add_(0, ix.row_trip, pen)
        if self.psi_cat is not None:
            # M2: the pair term uses s(S) (basket_field) instead of sum phi.  Replace
            # |sum phi|^2 / 2 by |s|^2 / 2 and add back |psi_c|^2 / 2 per extra product,
            # the diagonal that the Gaussian identity leaves.
            ph = self.phi_r()
            v = torch.zeros(ix.B, self.rank).index_add_(0, batch.line_trip, ph[batch.line_item])
            extra = (n_row - 1.0).clamp_min(0.0)
            psi = self.psi_cat[ix.row_cat]
            s = v - torch.zeros(ix.B, self.rank).index_add_(0, ix.row_trip, psi * extra[:, None])
            diagonal = torch.zeros(ix.B).index_add_(0, ix.row_trip, extra * psi.square().sum(-1))
            e = e + 0.5 * (s.square().sum(-1) - v.square().sum(-1) + diagonal)
        if self.size_on_categories:                        # M3: rho_0 on categories present
            n = torch.zeros(ix.B, dtype=torch.long).index_add_(
                0, batch.line_trip, torch.ones_like(batch.line_item))
            k = torch.zeros(ix.B, dtype=torch.long).index_add_(0, ix.row_trip, (n_row > 0).long())
            rho0 = self.rho_0()
            e = e + rho0[n] - rho0[k]
        if iv is not None:                                 # M4c: per product of the category
            e = e + torch.zeros(ix.B).index_add_(0, ix.row_trip, n_row * iv)
        return e

    # ---- loading ------------------------------------------------------------------------------
    def load_fitted_state(self, state):
        """Load a state dict, creating the optional blocks it contains and dropping legacy
        keys.  Flags that are not parameters (rank, size_on_categories) are set by the caller."""
        state = dict(state)
        if bool(state.pop("factored_size_enabled", torch.tensor(False))):
            raise ValueError("checkpoint uses the factored size law, which this model does not have")
        for key in LEGACY_KEYS:
            state.pop(key, None)
        for name in OPTIONAL_PARAMETERS:
            if name in state and state[name] is not None and getattr(self, name) is None:
                setattr(self, name, torch.nn.Parameter(torch.zeros_like(state[name])))
        self.load_state_dict(state)


def _trip_logsumexp(ix, b_slot):
    """log sum exp of slot values within each trip."""
    m = torch.full((ix.B,), -float("inf")).index_reduce_(
        0, ix.item_trip, b_slot, "amax", include_self=True).detach()
    w = torch.zeros(ix.B).index_add_(0, ix.item_trip, torch.exp(b_slot - m[ix.item_trip]))
    return m + torch.log(w)


def _row_logsumexp(ix, b_slot):
    """log sum exp of slot values within each row."""
    m = torch.full((ix.n_rows,), -float("inf")).index_reduce_(
        0, ix.row_of, b_slot, "amax", include_self=True).detach()
    w = torch.zeros(ix.n_rows).index_add_(0, ix.row_of, torch.exp(b_slot - m[ix.row_of]))
    return m + torch.log(w.clamp_min(1e-300))
