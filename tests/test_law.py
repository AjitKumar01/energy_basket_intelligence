"""Exact checks of the basket law against enumeration of every basket (tests/tiny.py)."""
import math

import pytest
import torch

from tiny import (NMAX, RANK, baskets_of, energy_given_z, full_energy, log_sum_exp, make_batch,
                  make_model)
from v4 import exact, sampling
from v4.competition_iv import competition_moments, estimate_competition_iv, jackknife_se
from v4.price import changed_price_context, own_price_tilt
from v4.training import Priors, penalties, pseudo_loglik

BLOCKS = {
    "additive": {},
    "M1": {"f_extra": True},
    "M2": {"presence": True},
    "M3": {"categories": True},
    "M4c": {"iv": True},
    "history-mean": {"history": "mean"},
    "history-inclusive": {"history": "inclusive"},
    "all": {"f_extra": True, "presence": True, "categories": True, "u_cat": True, "iv": True,
            "discount": True, "history": "inclusive"},
    "all-mean": {"f_extra": True, "presence": True, "categories": True, "u_cat": True, "iv": True,
                 "discount": True, "history": "mean"},
    "competition": {"competition": True},
    "all-competition": {"f_extra": True, "presence": True, "categories": True, "u_cat": True,
                        "discount": True, "history": "inclusive", "competition": True},
}


@pytest.mark.parametrize("name", BLOCKS)
def test_log_partition_and_incidence_given_z(name):
    model = make_model(**BLOCKS[name])
    batch = make_batch()
    batch.bind(model)
    ix = batch.ix
    z = torch.randn(2, RANK, generator=torch.Generator().manual_seed(3))
    u = model.slot_utilities_given_z(ix, z).detach()
    iv = model.iv_shift(ix)
    lz, p = exact.incidence(model, ix, u, z=z, iv=iv)
    for t in range(2):
        energies = {S: energy_given_z(model, ix, u, z, iv, S) for S in baskets_of(ix, t)}
        lz_t = log_sum_exp(list(energies.values()))
        assert abs(float(lz[t]) - lz_t) < 1e-10
        for s in (s for s in range(len(ix.item)) if int(ix.item_trip[s]) == t):
            prob = sum(math.exp(e - lz_t) for S, e in energies.items() if s in S)
            assert abs(float(p[s]) - prob) < 1e-10


def test_additive_size_law():
    model = make_model()
    batch = make_batch()
    batch.bind(model)
    ix = batch.ix
    u = model.slot_utilities(ix).detach()
    law = exact.additive_log_size(model, ix, u).softmax(-1)
    for t in range(2):
        energies = {S: energy_given_z(model, ix, u, None, None, S) for S in baskets_of(ix, t)}
        total = log_sum_exp(list(energies.values()))
        for n in range(1, NMAX + 1):
            exact_p = sum(math.exp(e - total) for S, e in energies.items() if len(S) == n)
            assert abs(float(law[t, n - 1]) - exact_p) < 1e-10


@pytest.mark.parametrize("name", ["additive", "all", "all-mean", "all-competition"])
def test_energy_matches_gaussian_identity(name):
    model = make_model(**BLOCKS[name])
    batch = make_batch()
    batch.bind(model)
    ix = batch.ix
    b, iv = model.slot_utilities(ix), model.iv_shift(ix)
    observed = [tuple(s for s in range(len(ix.item)) if batch.observed[s] and int(ix.item_trip[s]) == t)
                for t in range(2)]
    e = model.energy(batch)
    for t in range(2):
        assert abs(float(e[t]) - full_energy(model, ix, b, iv, observed[t])) < 1e-10


@pytest.mark.parametrize("cap", [NMAX, 2])
def test_sampler_matches_exact_marginals(cap):
    model = make_model(**BLOCKS["all"])
    model.R = cap
    base = make_batch()
    reps = 20000
    batch = make_batch()
    # replicate each trip many times by repeating the index
    from v4.batches import RaggedIndex
    ix0 = base.ix
    T = len(ix0.item)
    item = ix0.item.repeat(reps)
    row_of = (ix0.row_of[None, :] + ix0.n_rows * torch.arange(reps)[:, None]).reshape(-1)
    row_trip = (ix0.row_trip[None, :] + 2 * torch.arange(reps)[:, None]).reshape(-1)
    row_cat = ix0.row_cat.repeat(reps)
    ix = RaggedIndex(item, row_of, row_trip, row_cat, 2 * reps)
    batch.ix = ix
    batch.house = torch.tensor([0, 1]).repeat(reps)
    batch.ctx = {k: v.repeat(reps, *[1] * (v.dim() - 1)) for k, v in base.ctx.items()}
    batch.bind(model)
    z = torch.randn(2, RANK, generator=torch.Generator().manual_seed(4)).repeat(reps, 1)
    u = model.slot_utilities_given_z(ix, z).detach()
    iv = model.iv_shift(ix)
    X = sampling.sample_baskets(model, ix, u, torch.Generator().manual_seed(5), z=z, iv=iv)
    _, p = exact.incidence(model, ix, u, z=z, iv=iv)
    freq = X.double().view(reps, T).mean(0)
    exact_p = p.view(reps, T)[0]
    assert float((freq - exact_p).abs().max()) < 4 * math.sqrt(0.25 / reps)
    sizes = torch.zeros(2 * reps).index_add_(0, ix.item_trip, X.double())
    assert int(sizes.min()) >= 1 and int(sizes.max()) <= NMAX
    assert int(torch.zeros(ix.n_rows).index_add_(0, ix.row_of, X.double()).max()) <= cap


@pytest.mark.parametrize("name", ["all", "all-competition"])
def test_gibbs_and_importance_sampling_match_full_law(name):
    model = make_model(**BLOCKS[name])
    base = make_batch()
    base.bind(model)
    ix0 = base.ix
    b0, iv0 = model.slot_utilities(ix0).detach(), model.iv_shift(ix0)
    iv0 = None if iv0 is None else iv0.detach()
    exact_lz, exact_inc = [], torch.zeros(len(ix0.item))
    for t in range(2):
        energies = {S: full_energy(model, ix0, b0, iv0, S) for S in baskets_of(ix0, t)}
        lz = log_sum_exp(list(energies.values()))
        exact_lz.append(lz)
        for S, e in energies.items():
            for s in S:
                exact_inc[s] += math.exp(e - lz)
    est, _, _, _ = sampling.importance_log_z(model, ix0, torch.Generator().manual_seed(6), burn=50,
                                             components=16, thin=2, draws=512)
    assert max(abs(float(est[t]) - exact_lz[t]) for t in range(2)) < 0.02

    reps = 4000
    from v4.batches import RaggedIndex
    T = len(ix0.item)
    ix = RaggedIndex(ix0.item.repeat(reps),
                     (ix0.row_of[None, :] + ix0.n_rows * torch.arange(reps)[:, None]).reshape(-1),
                     (ix0.row_trip[None, :] + 2 * torch.arange(reps)[:, None]).reshape(-1),
                     ix0.row_cat.repeat(reps), 2 * reps)
    base.ix, base.house = ix, torch.tensor([0, 1]).repeat(reps)
    base.ctx = {k: v.repeat(reps, *[1] * (v.dim() - 1)) for k, v in base.ctx.items()}
    base.bind(model)
    X = sampling.draw_from_model(model, ix, torch.Generator().manual_seed(7), burn=60)
    freq = X.double().view(reps, T).mean(0)
    assert float((freq - exact_inc).abs().max()) < 0.03


@pytest.mark.parametrize("name", ["all", "all-competition"])
def test_pseudo_likelihood_is_exact_conditionals(name):
    model = make_model(**BLOCKS[name])
    batch = make_batch(basket=((0, 1, 3), (0, 5)))
    batch.bind(model)
    ix = batch.ix
    b, iv = model.slot_utilities(ix), model.iv_shift(ix)
    total = 0.0
    for t in range(2):
        S = {s for s in range(len(ix.item)) if batch.observed[s] and int(ix.item_trip[s]) == t}
        for s in (s for s in range(len(ix.item)) if int(ix.item_trip[s]) == t):
            without = S - {s}
            if not 0 < len(without) < NMAX:
                continue
            e_in = full_energy(model, ix, b, iv, tuple(sorted(without | {s})))
            e_out = full_energy(model, ix, b, iv, tuple(sorted(without)))
            e_obs = e_in if s in S else e_out
            total += e_obs - log_sum_exp([e_in, e_out])
    assert abs(float(pseudo_loglik(model, ix, batch.observed)) - total / 2) < 1e-10


def test_discount_split_starts_at_one_slope_model():
    model = make_model()
    batch = make_batch()
    batch.bind(model)
    before = model.slot_utilities(batch.ix).detach()
    model.disc_kappa = torch.nn.Parameter(model.price_kappa.detach().clone())
    model.disc_common = torch.nn.Parameter(torch.zeros(()))
    assert torch.allclose(model.slot_utilities(batch.ix), before, atol=1e-14, rtol=0)


def test_price_context_and_single_utility_tilt():
    model = make_model()
    batch = make_batch()
    batch.bind(model)
    ix = batch.ix
    change = torch.where(ix.item == 3, -0.2, 0.0)
    ctx = changed_price_context(batch.ctx, ix.item_trip, change)
    assert torch.allclose(ctx["dlp_bar"][0], batch.ctx["dlp_bar"][0] - 0.2 / 5)
    # With kappa = 1 a price change moves only the product's own utility, so the tilt is exact.
    with torch.no_grad():
        model.price_kappa.fill_(math.log(math.expm1(1.0)))
    u0 = model.slot_utilities(ix).detach()
    _, p0 = exact.incidence(model, ix, u0)
    batch.bind(model, ctx)
    u1 = model.slot_utilities(ix).detach()
    _, p1 = exact.incidence(model, ix, u1)
    s = int(torch.nonzero(ix.item == 3)[0])
    assert abs(float(own_price_tilt(p0[s], u1[s] - u0[s])) - float(p1[s])) < 1e-12


@pytest.mark.parametrize("name", ["additive", "all"])
def test_cap_on_products_per_category(name):
    """With R < nmax the law is the same law restricted to at most R products per category."""
    model = make_model(**BLOCKS[name])
    model.R = 2
    batch = make_batch()
    batch.bind(model)
    ix = batch.ix
    z = torch.randn(2, RANK, generator=torch.Generator().manual_seed(8))
    u = model.slot_utilities_given_z(ix, z).detach()
    iv = model.iv_shift(ix)
    lz, p = exact.incidence(model, ix, u, z=z, iv=iv)
    for t in range(2):
        energies = {S: energy_given_z(model, ix, u, z, iv, S) for S in baskets_of(ix, t, max_per_category=2)}
        lz_t = log_sum_exp(list(energies.values()))
        assert abs(float(lz[t]) - lz_t) < 1e-10
        for s in (s for s in range(len(ix.item)) if int(ix.item_trip[s]) == t):
            assert abs(float(p[s]) - sum(math.exp(e - lz_t) for S, e in energies.items() if s in S)) < 1e-10


@pytest.mark.parametrize("mode", ["mean", "inclusive"])
def test_history_offset_is_the_intended_trip_shift(mode):
    """Centring shifts every product of a trip by o_t = c_t + hist_size log(1 + ever), with
    c_t the mean of h ("mean") or the change of log sum exp b ("inclusive"); without centring
    the utilities are the raw ones."""
    model = make_model(history=mode)
    batch = make_batch()
    batch.bind(model)
    ix = batch.ix
    raw = model.utilities(ix.item, ix.item_trip, batch.ctx)
    h = model.history_utilities(ix.item, batch.ctx)
    b = model.slot_utilities(ix)
    for t in range(2):
        on = ix.item_trip == t
        ever = float(batch.ctx["rec_item"][on, 0].sum())
        if mode == "mean":
            c = float(h[on].mean())
        else:
            c = float(torch.logsumexp(raw[on], 0) - torch.logsumexp(raw[on] - h[on], 0))
        o = c + float(model.hist_size) * math.log1p(ever)
        assert float((raw[on] - o - b[on]).abs().max()) < 1e-12
    model.centre_history = None
    assert float((model.slot_utilities(ix) - raw).abs().max()) == 0.0


def test_competition_moves_purchases_within_the_category():
    """With lambda_c -> 0, at most one product per category and no interaction term, a price
    cut on one product leaves every category's expected count unchanged (pure switching to it
    from its rivals); with lambda_c -> 1 the block has no effect.  (Given z the products also
    carry z . phi_j, which the shift does not see, so switching is then close but not exact.)"""
    model = make_model(competition=True)
    model.R = 1
    model.rank = 0
    with torch.no_grad():                                         # price slopes near 3, so a cut matters
        model.gamma.fill_(1.0)
        model.beta.fill_(1.0)
    batch = make_batch()
    ix = batch.ix
    z = None
    cut = torch.where(ix.item == 0, math.log(0.7), 0.0)
    after = changed_price_context(batch.ctx, ix.item_trip, cut)

    def per_category(ctx):
        batch.bind(model, ctx)
        _, p = exact.incidence(model, ix, model.slot_utilities_given_z(ix, z).detach(), z=z)
        return torch.zeros(ix.n_rows).index_add_(0, ix.row_of, p), p

    with torch.no_grad():
        model.comp_raw.fill_(-40.0)
    rows0, p0 = per_category(batch.ctx)
    rows1, p1 = per_category(after)
    own = (ix.item == 0) & (ix.item_trip == 0)
    assert float(p1[own].sum() - p0[own].sum()) > 0.01          # the cut product gains
    assert float((rows1 - rows0).abs().max()) < 1e-10            # no category grows or shrinks

    with torch.no_grad():
        model.comp_raw.fill_(40.0)
    with_block = per_category(after)[1]
    model.comp_raw = None
    without = per_category(after)[1]
    assert float((with_block - without).abs().max()) < 1e-12


def test_existing_m4c_checkpoint_loads_and_scores_unchanged():
    """An existing M4c state dict (with iv_slope_raw and no comp_raw) loads into BasketModel
    and scores identically to the original once-per-category M4c law, including multi-item
    category baskets where M4c charges iv_shift once per present category."""
    src = make_model(**BLOCKS["all"])
    state = {k: v.detach().clone() for k, v in src.state_dict().items()}
    assert "iv_slope_raw" in state and "comp_raw" not in state

    dst = make_model()                                            # bare model, optional blocks created on load
    dst.size_on_categories = True
    dst.centre_history = "inclusive"
    dst.rank = RANK
    dst.load_fitted_state(state)
    assert dst.iv_slope_raw is not None and dst.comp_raw is None

    # Use a basket with 2 items in category 0 ((0, 1, 3)) so once-per-category M4c vs per-item
    # competition would differ if M4c were altered.
    batch = make_batch(basket=((0, 1, 3), (0, 5)))
    batch.bind(dst)
    ix = batch.ix
    b, iv = dst.slot_utilities(ix), dst.iv_shift(ix)
    e = dst.energy(batch)
    observed = [tuple(s for s in range(len(ix.item)) if batch.observed[s] and int(ix.item_trip[s]) == t)
                for t in range(2)]
    for t in range(2):
        assert abs(float(e[t]) - full_energy(dst, ix, b, iv, observed[t])) < 1e-12
    pl_dst = float(pseudo_loglik(dst, ix, batch.observed))
    batch.bind(src)
    assert abs(pl_dst - float(pseudo_loglik(src, ix, batch.observed))) < 1e-12
    assert torch.allclose(e, src.energy(batch), atol=1e-12, rtol=0)


def test_instrument_estimator_updates_comp_raw():
    """The exogenous price-instrument Wald IV estimator runs on a competition model and sets
    finite comp_raw logits in (0.01, 0.99) without touching M4c."""
    model = make_model(**BLOCKS["all-competition"])
    batch = make_batch(basket=((0, 1, 3), (0, 5)))

    class DummyBatcher:
        def make(self, trips):
            return batch

    stats = estimate_competition_iv(model, [[0, 1]], DummyBatcher())
    lam = model.comp_lambda()
    assert model.iv_slope_raw is None
    assert torch.isfinite(model.comp_raw).all()
    assert float(lam.min()) >= 0.01 - 1e-12 and float(lam.max()) <= 0.99 + 1e-12
    assert abs(stats["comp_lambda_mean"] - round(float(lam.mean()), 4)) < 1e-12


def test_instrument_leaves_out_display_and_mailer():
    """The instrument is built from price and discount only: the display and mailer weights,
    fitted on the same trips, do not move N_c, while the discount slope does.  The returned
    raw pooled lambda is sum N_c / sum D_c."""
    model = make_model(**BLOCKS["all-competition"])
    batch = make_batch(basket=((0, 1, 3), (0, 5)))

    class DummyBatcher:
        def make(self, trips):
            return batch

    num0, den0 = competition_moments(model, [[0, 1]], DummyBatcher())
    g = torch.Generator().manual_seed(7)
    with torch.no_grad():
        model.w_dsp.copy_(3.0 * torch.randn(model.w_dsp.shape, generator=g))
        model.w_mlr.copy_(3.0 * torch.randn(model.w_mlr.shape, generator=g))
    num1, den1 = competition_moments(model, [[0, 1]], DummyBatcher())
    assert torch.allclose(num0, num1, atol=1e-12, rtol=0)
    if model.disc_common is not None:
        with torch.no_grad():
            model.disc_common.add_(1.0)
        num2, _ = competition_moments(model, [[0, 1]], DummyBatcher())
        assert not torch.allclose(num1, num2, atol=1e-9, rtol=0)
        with torch.no_grad():
            model.disc_common.sub_(1.0)
    stats = estimate_competition_iv(model, [[0, 1]], DummyBatcher())
    assert abs(stats["comp_lambda_pool_raw"] - round(float(num1.sum() / den1.sum()), 4)) < 1e-12


def test_jackknife_se_matches_leave_one_category_out():
    g = torch.Generator().manual_seed(3)
    num, den = torch.randn(7, generator=g), 1.0 + torch.rand(7, generator=g)
    loo = [float((num.sum() - num[c]) / (den.sum() - den[c])) for c in range(7)]
    mean = sum(loo) / 7
    brute = math.sqrt(6 / 7 * sum((v - mean) ** 2 for v in loo))
    assert abs(jackknife_se(num, den) - brute) < 1e-12


def test_promo_common_is_a_reparameterisation():
    """Common display and mailer slopes plus per-product deviations give the same utilities,
    energies and log Z as the per-product weights they add up to, and survive a checkpoint
    round trip."""
    model = make_model(**BLOCKS["all-competition"])
    batch = make_batch(basket=((0, 1, 3), (0, 5)))
    batch.bind(model)
    ix = batch.ix
    b0, e0 = model.slot_utilities(ix), model.energy(batch)
    lz0 = exact.log_partition(model, ix, b0)
    with torch.no_grad():
        model.dsp_common = torch.nn.Parameter(torch.tensor(0.7))
        model.mlr_common = torch.nn.Parameter(torch.tensor(-0.2))
        model.w_dsp.sub_(0.7)
        model.w_mlr.add_(0.2)
    batch.bind(model)
    b1 = model.slot_utilities(ix)
    assert torch.allclose(b0, b1, atol=1e-12, rtol=0)
    assert torch.allclose(e0, model.energy(batch), atol=1e-12, rtol=0)
    assert torch.allclose(lz0, exact.log_partition(model, ix, b1), atol=1e-12, rtol=0)
    fresh = make_model(**BLOCKS["all-competition"])
    fresh.load_fitted_state(model.state_dict())
    for a, b in zip(fresh.promo_weights(), model.promo_weights()):
        assert torch.allclose(a, b, atol=1e-12, rtol=0)


def test_promo_deviation_prior():
    """With common slopes the prior adds 0.5 * prec * (|w_dsp|^2 + |w_mlr|^2) / n_train."""
    model = make_model(**BLOCKS["all-competition"])
    model.dsp_common = torch.nn.Parameter(torch.tensor(0.3))
    model.mlr_common = torch.nn.Parameter(torch.tensor(0.1))
    psi, delta = torch.randn(3, RANK), torch.randn(len(model.lam), RANK)
    off = Priors(0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0)
    on = Priors(0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 3.0)
    gap = float(penalties(model, on, 50, psi, delta, {}, True) - penalties(model, off, 50, psi, delta, {}, True))
    expected = 0.5 * 3.0 * float(model.w_dsp.square().sum() + model.w_mlr.square().sum()) / 50
    assert abs(gap - expected) < 1e-12


@pytest.mark.parametrize("name", ["additive", "M3", "all", "all-competition"])
def test_fast_log_partition_and_batch_z_against_enumeration(name):
    """Fused C++ log_partition_cpp and log_partition_batch_z_cpp (including dynamic item-axis
    truncation with trunc_min_cap=1 and fallback at large z) match basket enumeration."""
    model = make_model(**BLOCKS[name])
    model.trunc_min_cap = 1
    batch = make_batch()
    batch.bind(model)
    ix = batch.ix
    b = model.slot_utilities(ix).detach()
    iv = model.iv_shift(ix, b_slot=b)
    ph_slot = model.phi_r()[ix.item].detach().contiguous()
    u0 = (b - 0.5 * ph_slot.square().sum(-1)).contiguous()

    # Check both moderate z (where truncation can trigger if tail is tiny) and extreme z
    # (where mass near nmax forces fallback to full cap).
    g = torch.Generator().manual_seed(42)
    z_draws = torch.cat([
        torch.randn(2, 4, RANK, generator=g) * 0.5,
        torch.randn(2, 4, RANK, generator=g) * 6.0,
        torch.full((2, 2, RANK), -8.0),
    ], dim=1).contiguous()
    D = z_draws.shape[1]

    from v4.kernels import check, ptr, sampler
    a = sampling._arrays(model, iv=iv, psi=model.psi_cat)
    K_max = sampling._max_count(model, ix)
    out_batch = torch.empty((2, D), dtype=torch.float64)
    with torch.no_grad():
        check(sampler().log_partition_batch_z_cpp(
            2, D, ix.n_rows, len(ix.item), ix.Cpad, K_max, int(model.nmax), RANK,
            int(model.size_on_categories), 1, 1.0,
            ptr(u0), ptr(ph_slot), ptr(z_draws), ptr(ix.row_of), ptr(ix.item_pos),
            ptr(ix.row_size), ptr(ix.row_cat), ptr(ix.row_trip), ptr(ix.flat_slot),
            ptr(a["rho_c"]), ptr(a["rho_0"]), ptr(a["f_extra"]), ptr(a["psi"]),
            ptr(a["iv"]), ptr(out_batch)), "log_partition_batch_z")

        for d in range(D):
            zd = z_draws[:, d]
            ud = model.slot_utilities_given_z(ix, zd, b=b).detach()
            lz_fast = exact.log_partition(model, ix, ud, z=zd, iv=iv)
            for t in range(2):
                energies = [energy_given_z(model, ix, ud, zd, iv, S) for S in baskets_of(ix, t)]
                lz_enum = log_sum_exp(energies)
                assert abs(float(lz_fast[t]) - lz_enum) <= 1e-8
                assert abs(float(out_batch[t, d]) - lz_enum) <= 1e-8


def test_single_basket_four_chains_and_truncated_sampler_against_enumeration():
    """Single-basket (B=1) parallel 4-chain importance_log_z and truncated sampler match
    exact enumeration of the full model."""
    model = make_model(**BLOCKS["all"])
    model.trunc_min_cap = 1
    base = make_batch()
    base.bind(model)
    ix0 = base.ix
    b0, iv0 = model.slot_utilities(ix0).detach(), model.iv_shift(ix0)
    iv0 = None if iv0 is None else iv0.detach()
    energies0 = {S: full_energy(model, ix0, b0, iv0, S) for S in baskets_of(ix0, 0)}
    exact_lz0 = log_sum_exp(list(energies0.values()))

    # Build a single-trip (B=1) Batch for trip 0 to exercise gibbs_four_chains_cpp.
    from v4.batches import RaggedIndex
    slot_mask0 = ix0.item_trip == 0
    row_mask0 = ix0.row_trip == 0
    ix1 = RaggedIndex(ix0.item[slot_mask0], ix0.row_of[slot_mask0],
                      ix0.row_trip[row_mask0], ix0.row_cat[row_mask0], 1)
    base.ix = ix1
    base.house = base.house[:1]
    base.ctx = {k: (v[:1] if k == "dlp_bar" else v[slot_mask0]) for k, v in base.ctx.items()}
    base.bind(model)

    est1, _, _, X_out = sampling.importance_log_z(
        model, ix1, torch.Generator().manual_seed(19), burn=50, components=16, thin=2, draws=512)
    assert abs(float(est1[0]) - exact_lz0) < 0.02
    assert 1 <= int(X_out.sum()) <= NMAX


@pytest.mark.parametrize("name", ["additive", "M1", "M2", "M3", "all", "all-competition"])
def test_log_partition_with_sizes_beyond_double_range(name):
    """Basket weights of different sizes that differ by far more than a double can hold:
    one product at utility 1000 and four at 740, so after centring at the trip's largest
    utility a size-4 basket is e^-780 in the table, yet e^2220 heavier than size 1 once n * 1000
    is added back.  A single tilt per trip rounded size 4 to zero and lost about 740 nats of log Z
    here (and 123 nats on a real Dunnhumby trip at a chain-end z).  Both the training path
    (with gradient: CategoryProduct2D for M3, CategoryProduct otherwise) and the fast path
    (log_partition_cpp) must match enumeration, and so must the incidence."""
    model = make_model(**BLOCKS[name])
    batch = make_batch()
    batch.bind(model)
    ix = batch.ix
    z = torch.randn(2, RANK, generator=torch.Generator().manual_seed(3))
    u = model.slot_utilities_given_z(ix, z).detach().clone()
    iv = model.iv_shift(ix)
    trip0 = [s for s in range(len(ix.item)) if int(ix.item_trip[s]) == 0]
    for s in trip0:
        j = int(ix.item[s])
        u[s] = 1000.0 if j == 0 else 740.0
    lz_grad, p = exact.incidence(model, ix, u, z=z, iv=iv)
    with torch.no_grad():
        lz_fast = exact.log_partition(model, ix, u, z=z, iv=iv)
    for t in range(2):
        energies = {S: energy_given_z(model, ix, u, z, iv, S) for S in baskets_of(ix, t)}
        lz_t = log_sum_exp(list(energies.values()))
        assert abs(float(lz_grad[t]) - lz_t) < 1e-8
        assert abs(float(lz_fast[t]) - lz_t) < 1e-8
        for s in (s for s in range(len(ix.item)) if int(ix.item_trip[s]) == t):
            prob = sum(math.exp(e - lz_t) for S, e in energies.items() if s in S)
            assert abs(float(p[s]) - prob) < 1e-10


@pytest.mark.parametrize("name", ["additive", "M3", "all"])
def test_log_partition_with_categories_far_below_the_trip(name):
    """Category 0 at utilities 800, -400, -400 and the two products of category 1 at 400:
    after centring at the trip's largest utility, category 1's pair term is e^-800, below the
    double range, yet the basket {800, 400, 400} is e^400 heavier than any basket without
    that pair.  The
    within-category sums must keep it (each category centred at its own largest weight), on
    the training path (with gradient) and on the fast path, and so must the incidence.
    (A spread of more than about 745 nats inside one category is still not supported.)"""
    model = make_model(**BLOCKS[name])
    batch = make_batch()
    batch.bind(model)
    ix = batch.ix
    z = torch.randn(2, RANK, generator=torch.Generator().manual_seed(3))
    u = model.slot_utilities_given_z(ix, z).detach().clone()
    iv = model.iv_shift(ix)
    for s in (s for s in range(len(ix.item)) if int(ix.item_trip[s]) == 0):
        u[s] = {0: 800.0, 1: -400.0, 2: -400.0, 3: 400.0, 4: 400.0}[int(ix.item[s])]
    lz_grad, p = exact.incidence(model, ix, u, z=z, iv=iv)
    with torch.no_grad():
        lz_fast = exact.log_partition(model, ix, u, z=z, iv=iv)
    for t in range(2):
        energies = {S: energy_given_z(model, ix, u, z, iv, S) for S in baskets_of(ix, t)}
        lz_t = log_sum_exp(list(energies.values()))
        assert abs(float(lz_grad[t]) - lz_t) < 1e-8
        assert abs(float(lz_fast[t]) - lz_t) < 1e-8
        for s in (s for s in range(len(ix.item)) if int(ix.item_trip[s]) == t):
            prob = sum(math.exp(e - lz_t) for S, e in energies.items() if s in S)
            assert abs(float(p[s]) - prob) < 1e-10
