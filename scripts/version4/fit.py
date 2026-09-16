"""Shared batching, initialization and incidence-ranking helpers for Version-4 stages.

The historical Version-3 trainer that used to live here was superseded by the staged
pipeline (exact additive fit, spectral rank selection, size-stratified natural interaction
solve).  What remains is the code those stages share:

* :class:`Batcher` gathers the ragged assortment index and the per-slot/per-line context
  for a set of trips, so the energy and the normaliser score every product identically;
* :func:`popularity_logits`, :func:`initialize_taste_moments`,
  :func:`initialize_size_potential` and :func:`calibrate_size_ipf` build the untrained
  initialization artifact from training trips only; and
* :func:`rec_eval` ranks a hidden basket item by the model's marginal (optionally
  conditional) incidence ``pi_j = d log Z_+ / d b_j``.
"""
import numpy as np
import torch

from ragged import RaggedIndex


def log(m):
    print(f"[fit] {m}", flush=True)


@torch.no_grad()
def initialize_taste_moments(model, D, trips, strength=1.0, prior=100.0,
                             clip=3.0, seed=0):
    """Initialize the existing theta_h'alpha_j term from training-only log shares.

    At weak interaction, conditioning on basket size gives the multinomial moment

        log P(j | h) - log P(j) = theta_h' alpha_j + household constant.

    We smooth each household share toward the global product share, remove the household
    constant and residual product level, then take the weighted rank-K least-squares
    approximation.  This initializes the unchanged version-4 parameter block; every factor
    remains free during exact joint-likelihood training.
    """
    if strength <= 0:
        return None
    try:
        from sklearn.utils.extmath import randomized_svd
    except ImportError as exc:
        raise RuntimeError("--moment-taste-init requires scikit-learn") from exc

    N, J = model.theta.shape[0], model.alpha.shape[0]
    keep_trip = np.zeros(len(D["trip_split"]), dtype=bool)
    keep_trip[np.asarray(trips, dtype=np.int64)] = True
    line_trip = np.repeat(np.arange(len(D["line_ptr"]) - 1, dtype=np.int64),
                          np.diff(D["line_ptr"]))
    line_keep = keep_trip[line_trip]
    h = D["trip_user"][line_trip[line_keep]].astype(np.int64, copy=False)
    j = D["line_item"][line_keep].astype(np.int64, copy=False)

    count = np.zeros((N, J), dtype=np.float32)
    np.add.at(count, (h, j), 1.0)
    lines_h = count.sum(1)
    share = count.sum(0) + np.float32(0.5)
    share /= share.sum()
    residual = np.log((count + np.float32(prior) * share[None, :])
                      / (lines_h[:, None] + np.float32(prior))) - np.log(share[None, :])

    # Remove the two additive directions already represented by the household size gauge
    # and lam_j.  Household weights match uniform sampling of training trips.
    trip_weight = np.bincount(D["trip_user"][trips], minlength=N).astype(np.float32)
    trip_weight /= max(float(trip_weight.mean()), 1e-12)
    residual -= (residual @ share)[:, None]
    residual -= ((trip_weight[:, None] * residual).sum(0)
                 / np.maximum(trip_weight.sum(), 1e-12))[None, :]
    np.clip(residual, -clip, clip, out=residual)

    # One existing taste dimension is reserved for the household common offset below.
    # Without it, a zero-mean conditional utility still raises logsumexp by Jensen's
    # inequality and the initializer changes basket size even though its moment equation is
    # conditional on size.
    conditional_rank = model.K - 1
    if conditional_rank < 1:
        raise ValueError("moment taste initialization requires K >= 2")
    weighted = residual * np.sqrt(trip_weight[:, None])
    U, singular, Vt = randomized_svd(
        weighted, n_components=conditional_rank, n_iter=4, random_state=seed)
    root = np.sqrt(np.maximum(singular, 0.0) * strength)
    theta_cond = U * root[None, :] / np.sqrt(trip_weight[:, None]).clip(min=1e-12)
    alpha_cond = Vt.T * root[None, :]
    fitted_cond = theta_cond @ alpha_cond.T

    # Compute the common offset against the products each household was actually exposed
    # to, weighting each by the popularity utility already in lam.  This is the discrete
    # log-partition gauge of the conditional multinomial moment: subtracting it changes no
    # within-household product odds, while preserving the additive assortment intensity
    # that the empirical rho_0 initialization expects.
    try:
        from scipy import sparse
    except ImportError as exc:
        raise RuntimeError("--moment-taste-init requires scipy") from exc
    S = int(D["n_store"])
    us = sparse.coo_matrix(
        (np.ones(len(trips), dtype=np.float32),
         (D["trip_user"][trips], D["trip_store"][trips])),
        shape=(N, S)).tocsr()
    av_row, av_col = [], []
    ptr, items, Ccat = D["store_cat_ptr"], D["store_items"], int(D["n_cat"])
    for store in range(S):
        lo, hi = int(ptr[store * Ccat]), int(ptr[(store + 1) * Ccat])
        av_row.extend([store] * (hi - lo))
        av_col.extend(items[lo:hi].tolist())
    availability = sparse.coo_matrix(
        (np.ones(len(av_row), dtype=np.float32), (av_row, av_col)),
        shape=(S, J)).tocsr()
    exposure = (us @ availability).tocoo()
    base = np.exp(model.lam.detach().cpu().numpy().clip(-30.0, 30.0))
    ew = exposure.data * base[exposure.col]
    den = np.bincount(exposure.row, weights=ew, minlength=N).clip(min=1e-30)
    num = np.bincount(
        exposure.row,
        weights=ew * np.exp(fitted_cond[exposure.row, exposure.col].clip(-30.0, 30.0)),
        minlength=N).clip(min=1e-30)
    common = np.log(num / den)

    theta = np.zeros((N, model.K), dtype=theta_cond.dtype)
    alpha = np.zeros((J, model.K), dtype=alpha_cond.dtype)
    if model.household_size_rank1:
        # Fix the final product loading to one.  The remaining item loadings are centred,
        # and their removed household-common value is transferred into kappa.  This is an
        # exact reparameterization of fitted_cond-common:
        #
        # theta_cond alpha_cond' - common
        #   = theta_cond (alpha_cond-alpha_bar)' +
        #     (theta_cond alpha_bar-common) 1'.
        alpha_bar = alpha_cond.mean(0)
        alpha_comp = alpha_cond - alpha_bar[None, :]
        theta[:, :conditional_rank], alpha[:, :conditional_rank] = (
            theta_cond, alpha_comp)
        theta[:, -1] = theta_cond @ alpha_bar - common
        alpha[:, -1] = 1.0
        const_scale = 1.0
    else:
        # Balance the constant factor's raw scale for weight decay; only its product
        # matters in the unconstrained factorization.
        const_scale = max(float((np.square(common).sum() / J) ** 0.25), 1e-4)
        theta[:, :conditional_rank], alpha[:, :conditional_rank] = (
            theta_cond, alpha_cond)
        theta[:, -1], alpha[:, -1] = -common / const_scale, const_scale

    # theta_c() removes the raw unweighted mean.  Transfer that exact product-specific
    # offset to lam so the initialized utility is unchanged by the gauge convention.
    theta_mean = theta.mean(0)
    theta -= theta_mean[None, :]
    model.theta.copy_(torch.as_tensor(theta, dtype=model.theta.dtype,
                                      device=model.theta.device))
    model.alpha.copy_(torch.as_tensor(alpha, dtype=model.alpha.dtype,
                                      device=model.alpha.device))
    model.lam.add_(torch.as_tensor(alpha @ theta_mean, dtype=model.lam.dtype,
                                   device=model.lam.device))
    fitted = theta @ alpha.T
    observed = fitted[h, j]
    return dict(all_sd=float(fitted.std()), conditional_sd=float(fitted_cond.std()),
                observed_mean=float(observed.mean()),
                observed_sd=float(observed.std()),
                common_mean=float(common.mean()), common_max=float(common.max()),
                theta_norm=float(np.linalg.norm(theta, axis=1).mean()),
                alpha_norm=float(np.linalg.norm(alpha, axis=1).mean()))


class Batcher:
    """Builds the ragged index and the per-slot features for a set of trips."""

    def __init__(self, D, F, nmax, include_recency=True):
        self.D, self.F, self.nmax = D, F, nmax
        self.include_recency = bool(include_recency)
        if self.include_recency and not getattr(F, "include_recency", True):
            raise ValueError("Batcher requests recency but Features did not load it")
        self.C = int(D["n_cat"])
        self.ptr = D["store_cat_ptr"]
        self.items = D["store_items"]
        self.lptr = D["line_ptr"]

    def make(self, trips):
        D, C = self.D, self.C
        it_l, row_of, row_trip, row_cat = [], [], [], []
        nrow = 0
        for bi, t in enumerate(trips):
            s = int(D["trip_store"][t]) * C
            for c in range(C):
                lo, hi = int(self.ptr[s + c]), int(self.ptr[s + c + 1])
                if hi <= lo:
                    continue
                it_l.append(self.items[lo:hi])
                row_of.append(np.full(hi - lo, nrow, np.int64))
                row_trip.append(bi)
                row_cat.append(c)
                nrow += 1
        item = np.concatenate(it_l)
        ix = RaggedIndex(item, np.concatenate(row_of),
                         np.array(row_trip, np.int64), np.array(row_cat, np.int64),
                         len(trips))
        store = torch.as_tensor(D["trip_store"][trips], dtype=torch.long)
        day = torch.as_tensor(D["trip_day"][trips], dtype=torch.long)
        week = torch.as_tensor(D["trip_week"][trips], dtype=torch.long)
        st_i, dy_i, wk_i = store[ix.item_trip], day[ix.item_trip], week[ix.item_trip]
        dlp, disp, mail = self.F.gather(ix.item, st_i, dy_i, wk_i)
        # week-of-year, per the spec: (WEEK_NO - 1) mod 52.  Clamping instead, as an earlier
        # version did, collapsed 54.6% of trips onto one seasonal parameter.
        user = torch.as_tensor(D["trip_user"][trips], dtype=torch.long)
        # Per-trip mean price deviation over the ASSORTMENT.  It is the reference for
        # splitting dlp into a common level and an idiosyncratic deviation, and it must be
        # the same number whether b_at is called on assortment slots or on purchased lines
        # -- so it is computed once here, from the assortment, and carried in both views.
        _dbar = torch.zeros(ix.B, dtype=torch.float64).index_add_(
            0, ix.item_trip, dlp.double())
        _dcnt = torch.zeros(ix.B, dtype=torch.float64).index_add_(
            0, ix.item_trip, torch.ones_like(dlp, dtype=torch.float64))
        _dbar = _dbar / _dcnt.clamp_min(1.0)
        ctx = dict(dlp_bar=_dbar, dlp=dlp.double(), disp=disp.double(), mail=mail.double(),
                   week=(wk_i - 1) % 52, store=st_i)
        if self.include_recency:
            ctx["rec"] = self.F.recency(ix.item, user[ix.item_trip], dy_i)
        li, lt, lc, lu = [], [], [], []
        for bi, t in enumerate(trips):
            a, b = int(self.lptr[t]), int(self.lptr[t + 1])
            li.append(D["line_item"][a:b])
            lc.append(D["line_cat"][a:b])
            lu.append(D["line_units"][a:b])
            lt.append(np.full(b - a, bi, np.int64))
        LI = torch.as_tensor(np.concatenate(li), dtype=torch.long)
        LT = torch.as_tensor(np.concatenate(lt), dtype=torch.long)
        # the SAME features, gathered at the purchased lines, so energy() and log_Z score
        # each product identically
        dlp_l, disp_l, mail_l = self.F.gather(LI, store[LT], day[LT], week[LT])
        lctx = dict(dlp_bar=_dbar, dlp=dlp_l.double(), disp=disp_l.double(), mail=mail_l.double(),
                    week=(week[LT] - 1) % 52, store=store[LT])
        if self.include_recency:
            lctx["rec"] = self.F.recency(LI, user[LT], day[LT])
        house = torch.as_tensor(D["trip_user"][trips], dtype=torch.long)
        return (ix, ctx, lctx, house,
                LI, LT, torch.as_tensor(np.concatenate(lc), dtype=torch.long),
                torch.as_tensor(np.concatenate(lu), dtype=torch.long))


def popularity_logits(D, trips):
    """Exposure-corrected product incidence initializer on the training window.

    A zero lam makes the first model nearly uniform over about 5,000 available products,
    even though one pass over the training data already identifies their marginal rates.
    This is initialization, not target leakage: only training trips and the assortments in
    which each item was available enter the estimate.
    """
    J, C, S = (int(D[k]) for k in ("n_item", "n_cat", "n_store"))
    count = np.zeros(J, dtype=np.float64)
    for t in trips:
        lo, hi = int(D["line_ptr"][t]), int(D["line_ptr"][t + 1])
        count[D["line_item"][lo:hi]] += 1.0
    exposure = np.zeros(J, dtype=np.float64)
    store_n = np.bincount(D["trip_store"][trips], minlength=S)
    ptr, items = D["store_cat_ptr"], D["store_items"]
    for store in range(S):
        lo, hi = int(ptr[store * C]), int(ptr[(store + 1) * C])
        exposure[items[lo:hi]] += store_n[store]
    seen = exposure > 0
    value = np.empty(J, dtype=np.float64)
    value[seen] = np.log((count[seen] + 0.5) / (exposure[seen] + 1.0))
    value[~seen] = np.median(value[seen])
    value -= value.mean()       # common shifts are exactly a linear rho_0 gauge direction
    return torch.as_tensor(value, dtype=torch.float64)


def _size_coeffs(m, z, ix):
    """mean_b log A_n(z) at the given z -- the combinatorial part of the size law, before
    rho_0 tilts it."""
    from ragged import esp_bucketed, poly_mul_trunc, seg_max
    phi_i = m.phi[ix.item]
    bt = m.b_flat(ix) - 0.5 * (phi_i ** 2).sum(-1)
    proj = (z[ix.item_trip] * phi_i.unsqueeze(1)).sum(-1)
    logw = (bt.unsqueeze(1) + proj).transpose(0, 1)
    M = seg_max(logw, ix.item_trip, ix.B)
    w = torch.exp(logw - M.index_select(-1, ix.item_trip))
    e = esp_bucketed(w, ix.row_of, ix.n_rows, m.R, ix.row_size, ix.item_pos)
    r = torch.arange(m.R + 1, dtype=w.dtype)
    G = torch.exp(-m.rho_c[ix.row_cat].unsqueeze(-1) * m.pair_feature(r)).unsqueeze(0) * e
    Gp = torch.zeros(1, ix.B * ix.Cpad, m.R + 1, dtype=w.dtype)
    Gp[:, :, 0] = 1.0
    Gp = Gp.index_copy(1, ix.flat_slot, G).view(1, ix.B, ix.Cpad, m.R + 1)
    A = Gp[:, :, 0, :]
    for c in range(1, ix.Cpad):
        A = poly_mul_trunc(A, Gp[:, :, c, :], m.nmax)
    n_ax = torch.arange(A.shape[-1], dtype=w.dtype)
    return (torch.log(A.clamp_min(1e-300)) + n_ax * M.unsqueeze(-1))[0].mean(0)


def initialize_size_potential(model, data, training, batcher, nmax):
    """Initialize rho_0 against the empirical training size law at reference contexts."""
    n_train = data["trip_nlines"][training]
    count = np.bincount(np.clip(n_train, 0, nmax), minlength=nmax + 1) + 0.5
    target = torch.log(torch.as_tensor(count / count.sum(), dtype=model.lam.dtype))
    sub = training[np.random.default_rng(0).choice(len(training), size=64, replace=False)]
    ix, ctx, _, house, *_ = batcher.make(sub)
    model.house, model.ctx = house, ctx
    z = torch.zeros(ix.B, 1, model.Kz, dtype=model.lam.dtype)
    with torch.no_grad():
        coeff = _size_coeffs(model, z, ix)
        rho = coeff[:nmax + 1] - target
        model.rho_0_free.copy_((rho - rho[0])[1:])
    return float((count / count.sum() * np.arange(nmax + 1)).sum())


def calibrate_size_ipf(model, data, training, batcher, nmax, steps=6,
                       n_trips=256, chunk=24, damp=0.5, progress=None):
    """Fit the aggregate size law by deterministic iterative proportional fitting.

    For a common size tilt, ``P(n|x)`` is multiplied by ``exp(-Delta rho(n))``.  Updating
    ``rho(n) += log[p_model(n)/p_target(n)]`` is therefore the exact one-table IPF update;
    averaging context-specific normalized laws makes repeated damped updates necessary.
    Composition is invariant to rho_0, so this calibration cannot undo a utility transfer.
    """
    if steps <= 0 or n_trips <= 0 or not 0 < damp <= 1:
        raise ValueError("size IPF requires positive steps/trips and damp in (0,1]")
    count = np.bincount(np.clip(data["trip_nlines"][training], 1, nmax),
                        minlength=nmax + 1)[1:].astype(np.float64) + 0.5
    target = torch.as_tensor(count / count.sum(), dtype=model.lam.dtype)
    sample = training[np.random.default_rng(1729).choice(
        len(training), size=min(n_trips, len(training)), replace=False)]
    history = []
    for iteration in range(steps):
        total = torch.zeros(nmax, dtype=model.lam.dtype)
        seen = 0
        for start in range(0, len(sample), chunk):
            sub = sample[start:start + chunk]
            ix, ctx, _, house, *_ = batcher.make(sub)
            model.house, model.ctx = house, ctx
            with torch.no_grad():
                _, pn = model.log_Z(ix, drop_empty=True, return_size=True)
            total += pn.sum(0)
            seen += ix.B
        pbar = (total / seen).clamp_min(1e-300)
        pbar = pbar / pbar.sum()
        grid = torch.arange(1, nmax + 1, dtype=pbar.dtype)
        history.append(dict(step=iteration,
                            mean=float((pbar * grid).sum()),
                            kl=float((target * (target.log() - pbar.log())).sum()),
                            max_log_ratio=float((pbar.log() - target.log()).abs().max())))
        delta = pbar.log() - target.log()
        delta = delta - delta[0]                 # fix size-one as the non-empty gauge
        with torch.no_grad():
            model.rho_0_free.add_(damp * delta)
        if progress is not None:
            progress(history[-1])
    return history


def midrank(score, position):
    """1-based rank of ``score[position]`` with ties sharing the average rank."""
    score = np.asarray(score)
    target = score[position]
    return 1.0 + np.count_nonzero(score > target) + 0.5 * (
        np.count_nonzero(score == target) - 1)


def rec_eval(m, B, trips, seed=0, chunk=24, return_ranks=False, conditioned=True):
    """Complete-the-basket MRR and median rank, at every checkpoint.

    Version 4 defines the recommendation score as marginal incidence

        pi_j = d log(Z - 1) / d b_j,

    conditioned on the revealed remainder of the basket.  Conditioning is done exactly:
    remove the revealed items, shift candidate utilities by their interaction with that
    set, and shift the category and total-size potentials by its counts.  Differentiate the
    SAME normaliser used by training once per batch.  Ranking on the raw energy increment
    ``b + phi + rho_c`` is a different
    exactly-one-completion task; version4.html measured that scorer at about 0.023 versus
    about 0.082 for incidence.  Logging it as version-4 MRR therefore masks the quantity the
    experiment actually declares.

    It earns its place because nothing else in the log can see a ranking failure.  Measured on
    run68, the model scored MRR 0.0036 against a popularity baseline's 0.0467 -- WORSE than
    ranking by raw frequency -- while every pre-declared goal, the normaliser check and the
    distributional KL all looked unremarkable.  They score the joint distribution or its
    moments; none of them scores the ordering.

    The holdout is drawn with a fixed seed so the same items are hidden at every checkpoint
    and the series is comparable across a run.
    """
    rng = np.random.default_rng(seed)
    # Save and restore the model's batch context: this runs INSIDE the checkpoint block,
    # before the normaliser check, and leaving m.ctx pointing at the recommendation batch
    # makes the next b_flat mismatch its index (127,999 slots against 128,546).
    _sh, _sc = m.house, m.ctx
    ranks = []
    try:
        for k in range(0, len(trips), chunk):
            ix, ctx, lctx, hh, LI, LT, LC, LU = B.make(trips[k:k + chunk])
            m.house, m.ctx = hh, ctx
            with torch.no_grad():
                bf = m.b_flat(ix)

            # Draw holdouts once, then construct all conditioned trips together.  Saving
            # these explicitly avoids fragile RNG rewind/replay logic and guarantees that
            # the rank pass uses exactly the item that was hidden for the pi pass.
            holdout = {}
            remainder = {}
            b0 = bf.detach().clone()
            for b in range(ix.B):
                basket = LI[LT == b]
                if len(basket) < 2:
                    continue
                hid = int(basket[rng.integers(len(basket))])
                rest = torch.as_tensor([int(x) for x in basket if int(x) != hid],
                                       dtype=torch.long)
                if len(rest) == 0:
                    continue
                holdout[b], remainder[b] = hid, rest

            score_ix = ix
            if conditioned:
                # For S = R union T, the version-4 energy becomes, up to a constant,
                #
                #   sum_{j in T} [b_j + phi_j' Phi_R]
                #   + pair_phi(T)
                #   - sum_c rho_c [g(r_c+t_c)-g(r_c)]
                #   - [rho_0(r+t)-rho_0(r)].
                #
                # Thus the exact conditional law is another polynomial normaliser over
                # the UNREVEALED slots.  It needs no arbitrary finite utility pin.
                fixed_phi = torch.zeros(ix.B, m.Kz, dtype=m.phi.dtype,
                                        device=m.phi.device)
                fixed_cat = torch.zeros(ix.B, m.C, dtype=torch.long,
                                        device=m.phi.device)
                fixed_size = torch.zeros(ix.B, dtype=torch.long, device=m.phi.device)
                remove = torch.zeros(len(ix.item), dtype=torch.bool, device=ix.item.device)
                for b, rest in remainder.items():
                    fixed_size[b] = len(rest)
                    fixed_phi[b] = m.phi[rest].sum(0).detach()
                    fixed_cat[b] = torch.bincount(m.cat_of[rest], minlength=m.C)
                    sel = (ix.item_trip == b).nonzero().flatten()
                    remove[sel[torch.isin(ix.item[sel], rest)]] = True
                keep = ~remove
                score_ix = RaggedIndex(ix.item[keep], ix.row_of[keep], ix.row_trip,
                                       ix.row_cat, ix.B)
                b0 = (bf[keep] + (m.phi[score_ix.item]
                                  * fixed_phi[score_ix.item_trip]).sum(-1)).detach()
                m._condition_cat_count = fixed_cat
                m._condition_size = fixed_size

            b0 = b0.requires_grad_(True)
            m._b_override = b0
            try:
                with torch.enable_grad():
                    # In the exact conditional law T may be empty because the revealed set
                    # itself is a valid basket.  The unconditioned law still conditions the
                    # original model on a non-empty basket.
                    logz = m.log_Z(score_ix, drop_empty=not conditioned)
                pi = torch.autograd.grad(logz.sum(), b0)[0].detach()
            finally:
                m._b_override = None
                m._condition_cat_count = None
                m._condition_size = None

            for b, hid in holdout.items():
                sel = score_ix.item_trip == b
                items = score_ix.item[sel]
                pos = (items == hid).nonzero().flatten()
                if len(pos) == 0:
                    continue
                sc = pi[sel].clone()
                if not conditioned:
                    sc[torch.isin(items, remainder[b])] = -float("inf")
                ranks.append(midrank(sc.numpy(), int(pos[0])))
    finally:
        m._b_override = None
        m._condition_cat_count = None
        m._condition_size = None
        m.house, m.ctx = _sh, _sc
    if return_ranks:
        return np.asarray(ranks, dtype=float)
    if not ranks:
        return float("nan"), float("nan")
    r = np.asarray(ranks, dtype=float)
    return float((1.0 / r).mean()), float(np.median(r))
