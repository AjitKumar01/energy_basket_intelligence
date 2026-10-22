"""Exact checks of the basket law against enumeration of every basket (tests/tiny.py)."""
import math

import pytest
import torch

from tiny import (NMAX, RANK, baskets_of, energy_given_z, full_energy, log_sum_exp, make_batch,
                  make_model)
from v4 import exact, sampling
from v4.price import changed_price_context, own_price_tilt
from v4.training import pseudo_loglik

BLOCKS = {
    "additive": {},
    "M1": {"f_extra": True},
    "M2": {"presence": True},
    "M3": {"categories": True},
    "M4c": {"iv": True},
    "history": {"history": True},
    "all": {"f_extra": True, "presence": True, "categories": True, "u_cat": True, "iv": True,
            "discount": True, "history": True},
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


@pytest.mark.parametrize("name", ["additive", "all"])
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


def test_gibbs_and_importance_sampling_match_full_law():
    model = make_model(**BLOCKS["all"])
    base = make_batch()
    base.bind(model)
    ix0 = base.ix
    b0, iv0 = model.slot_utilities(ix0).detach(), model.iv_shift(ix0).detach()
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


def test_pseudo_likelihood_is_exact_conditionals():
    model = make_model(**{k: v for k, v in BLOCKS["all"].items()})
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


def test_history_offset_is_the_trip_mean_plus_size_term():
    """Centring shifts every product of a trip by o_t = mean h + hist_size log(1 + ever);
    without centring the utilities are the raw ones."""
    model = make_model(history=True)
    batch = make_batch()
    batch.bind(model)
    ix = batch.ix
    raw = model.utilities(ix.item, ix.item_trip, batch.ctx)
    h = model.history_utilities(ix.item, batch.ctx)
    b = model.slot_utilities(ix)
    for t in range(2):
        on = ix.item_trip == t
        ever = float(batch.ctx["rec_item"][on, 0].sum())
        o = float(h[on].mean()) + float(model.hist_size) * math.log1p(ever)
        assert float((raw[on] - o - b[on]).abs().max()) < 1e-12
    model.centre_history = False
    assert float((model.slot_utilities(ix) - raw).abs().max()) == 0.0
