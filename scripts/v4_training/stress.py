#!/usr/bin/env python3
"""Train the unchanged Version-4 model WITH its interactions on the stress world, and score it.

Same method that passed Step 1 on exact worlds (scripts/v4_training/hs.py), on the real
RaggedModel and its native exact kernels:
  start   the certified exact Phi = 0 checkpoint (artifacts/stress_clean_confirm)
  stage A pseudo-likelihood (exact conditionals log P(x_j | rest), all assortment slots)
  stage B stochastic maximum likelihood: persistent per-trip HS block-Gibbs chains,
          gradient through ONE exact dynamic program per trip at z drawn from the chain
          (differentiable_logz_beta0 at slot utilities b - |phi|^2/2 + z.phi)
  prior   weak Gaussian prior on Phi (2.0) and rho_c (0.2), divided by N -- Step 1's fix
  frozen  supported price (gamma, beta, price_kappa) and recency, as in the certified run
Phi's rank is 8 (the certified active rank) and is NOT capped.

eval: held-out log-likelihood on the certified reference trips by importance-sampled log Z
(proposal = mixture of N(s_k, I) over model baskets from the HS chain; Step 1's estimator),
paired with certified Version-4 (capped Phi, Smolyak) and with the Phi = 0 refit; the
classifier two-sample test; the 32 price scenarios against the world's causal truth.

Pass rules (declared before any stress-world result):
  1. paired held-out LL vs certified Version-4: household-clustered 95% interval above 0
  2. category-presence classifier AUC <= 0.80
  3. own-price scenario correlation >= 0.85
Run with ENERGY_MODEL_DATA_ROOT set to the stress bundle.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/version4"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

torch.set_default_dtype(torch.float64)
NEG = -float("inf")
BASE_NAMES = ("lam", "alpha", "theta", "rho_c", "rho_0_free", "w_dsp", "w_mlr", "mu", "delta", "zeta", "xi")
OTHER_BASE_NAMES = ("lam", "alpha", "theta", "w_dsp", "w_mlr", "mu", "delta", "zeta", "xi")
USE_CPP_SAMPLER = os.environ.get("STRESS_CPP_SAMPLER", "0") == "1"


# ---- setup ------------------------------------------------------------------------------------
def setup(initialization, additive, threads):
    from data import build
    from features import Features
    from fit import Batcher
    from fit_exact_additive import load_model
    torch.set_num_threads(threads)
    data = build()
    model, meta, _ = load_model(Path(initialization), data)
    raw = torch.load(additive, map_location="cpu", weights_only=False)
    model.load_state_dict(raw["model"])
    with torch.no_grad():
        model.phi.zero_()
    features = Features(int(data["n_item"]), int(data["n_store"]), include_recency=False,
                        include_store_price=True)
    batcher = Batcher(data, features, int(meta["nmax"]), include_recency=False)
    return data, model, meta, batcher


class Batch:
    """One Version-4 batch plus the observed basket as a slot mask."""

    def __init__(self, batcher, trips, J):
        self.trips = trips
        ix, ctx, line_ctx, house, li, lt, lc, _ = batcher.make(trips)
        self.ix, self.ctx, self.line_ctx, self.house, self.li, self.lt, self.lc = ix, ctx, line_ctx, house, li, lt, lc
        key = ix.item_trip * J + ix.item
        order = torch.argsort(key)
        pos = torch.searchsorted(key[order], lt * J + li)
        slot = order[pos.clamp(max=len(order) - 1)]
        if not bool((key[slot] == lt * J + li).all()):
            raise RuntimeError("a basket line is outside its trip's assortment")
        self.obs = torch.zeros(len(ix.item), dtype=torch.bool)
        self.obs[slot] = True

    def bind(self, model, ctx=None):
        model.house, model.ctx = self.house, (self.ctx if ctx is None else ctx)


def phi_r(model, r):
    return model.phi[:, :r]


def slot_utilities(model, ix, z, r, beta=1.0):
    """b - beta |phi|^2/2 + sqrt(beta) z.phi at every assortment slot (z: B x r)."""
    ph = phi_r(model, r)[ix.item]
    u = model.b_flat(ix) - 0.5 * beta * ph.square().sum(-1)
    if z is not None:
        u = u + math.sqrt(beta) * (z[ix.item_trip] * ph).sum(-1)
    return u


def basket_field(model, ix, X, r):
    """s = sum of phi over the basket (X: slot mask), B x r."""
    ph = phi_r(model, r)[ix.item]
    return torch.zeros(ix.B, r).index_add_(0, ix.item_trip, ph * X.double()[:, None])


def log_zc(model, ix, u):
    from interaction_particles import differentiable_logz_beta0
    return differentiable_logz_beta0(model, ix, u)


# ---- exact vectorized sampler of S | u -------------------------------------------------------
@torch.no_grad()
def sample_slots(model, ix, u, gen, use_cpp: bool | None = None):
    """One exact draw per trip from the additive Version-4 law at slot utilities u (category
    penalty rho_c, size potential rho_0, 1 <= |S| <= nmax).  Returns a slot mask."""
    B, nmax = ix.B, model.nmax
    L = int(ix.row_size.max())
    K = min(model.R, L, nmax)
    if USE_CPP_SAMPLER if use_cpp is None else use_cpp:
        from build_stress_sampler import load_stress_sampler
        T = len(ix.item)
        u_c = u.detach().contiguous()
        rho_c = model.rho_c.detach().contiguous()
        rho_0 = model.rho_0().detach().contiguous()
        seed = int(torch.randint(1, 2**62, (1,), generator=gen).item())
        out = torch.empty(T, dtype=torch.uint8)
        rc = load_stress_sampler().sample_slots_cpp(
            B, ix.n_rows, T, ix.Cpad, K, nmax, seed,
            u_c.data_ptr(), ix.row_of.data_ptr(), ix.item_pos.data_ptr(),
            ix.row_size.data_ptr(), ix.row_cat.data_ptr(), ix.flat_slot.data_ptr(),
            rho_c.data_ptr(), rho_0.data_ptr(), out.data_ptr())
        if rc != 0:
            raise RuntimeError(f"native sample_slots_cpp failed with status {rc}")
        return out.bool()
    scale = torch.full((B,), NEG).index_reduce_(0, ix.item_trip, u, "amax", include_self=True)
    w = u - scale[ix.item_trip]
    Wp = torch.full((ix.n_rows, L), NEG)
    Wp[ix.row_of, ix.item_pos] = w
    e = torch.full((ix.n_rows, K + 1), NEG)
    e[:, 0] = 0.0
    E = [e]
    for p in range(L):
        nxt = e.clone()
        nxt[:, 1:] = torch.logaddexp(e[:, 1:], e[:, :-1] + Wp[:, p:p + 1])
        e = nxt
        E.append(e)
    k_axis = torch.arange(K + 1, dtype=u.dtype)
    g = e - model.rho_c[ix.row_cat][:, None] * model.pair_feature(k_axis)[None, :]
    G = torch.full((B * ix.Cpad, K + 1), NEG)
    G[:, 0] = 0.0
    G[ix.flat_slot] = g
    G = G.view(B, ix.Cpad, K + 1)
    f = torch.full((B, nmax + 1), NEG)
    f[:, 0] = 0.0
    Fs = [f]
    for c in range(ix.Cpad):
        cur = torch.full((B, nmax + 1), NEG)
        for k in range(K + 1):
            cur[:, k:] = torch.logaddexp(cur[:, k:], f[:, :nmax + 1 - k] + G[:, c, k:k + 1])
        f = cur
        Fs.append(f)
    n_axis = torch.arange(1, nmax + 1, dtype=u.dtype)
    log_size = f[:, 1:] + n_axis[None, :] * scale[:, None] - model.rho_0()[1:nmax + 1][None, :]
    rem = torch.multinomial(torch.softmax(log_size, 1), 1, generator=gen).squeeze(1) + 1
    take_c = torch.zeros(B, ix.Cpad, dtype=torch.long)
    ks = torch.arange(K + 1)
    for c in range(ix.Cpad - 1, -1, -1):
        idx = rem[:, None] - ks[None, :]
        ok = idx >= 0
        lp = torch.where(ok, Fs[c].gather(1, idx.clamp(min=0)) + G[:, c, :], torch.full((B, K + 1), NEG))
        k = torch.multinomial(torch.softmax(lp, 1), 1, generator=gen).squeeze(1)
        take_c[:, c] = k
        rem = rem - k
    if bool((rem != 0).any()):
        raise RuntimeError("category backtrack left slots unfilled")
    need = take_c.view(-1)[ix.flat_slot]
    take = torch.zeros(ix.n_rows, L, dtype=torch.bool)
    for p in range(L - 1, -1, -1):
        have = need > 0
        nk = need.clamp(min=1)
        lp = Wp[:, p] + E[p].gather(1, (nk - 1)[:, None]).squeeze(1) - E[p + 1].gather(1, nk[:, None]).squeeze(1)
        prob = torch.where(have, lp.exp().clamp(0.0, 1.0), torch.zeros(ix.n_rows))
        t = torch.rand(ix.n_rows, generator=gen) < prob
        take[:, p] = t
        need = need - t.long()
    if bool((need != 0).any()):
        raise RuntimeError("item backtrack left slots unfilled")
    return take[ix.row_of, ix.item_pos]


@torch.no_grad()
def gibbs(model, ix, X, r, gen, sweeps=1, beta=1.0, use_cpp: bool | None = None):
    if USE_CPP_SAMPLER if use_cpp is None else use_cpp:
        from build_stress_sampler import load_stress_sampler
        B, nmax, T = ix.B, model.nmax, len(ix.item)
        L = int(ix.row_size.max())
        K = min(model.R, L, nmax)
        ph = phi_r(model, r)[ix.item].detach().contiguous()
        b_base = (model.b_flat(ix).detach() - 0.5 * beta * ph.square().sum(-1)).contiguous()
        rho_c = model.rho_c.detach().contiguous()
        rho_0 = model.rho_0().detach().contiguous()
        seed = int(torch.randint(1, 2**62, (1,), generator=gen).item())
        mask = X.to(torch.uint8).contiguous().clone()
        rc = load_stress_sampler().gibbs_sweeps_cpp(
            B, ix.n_rows, T, ix.Cpad, K, nmax, r, int(sweeps), math.sqrt(float(beta)), seed,
            b_base.data_ptr(), ph.data_ptr(), ix.item_trip.data_ptr(), ix.row_of.data_ptr(),
            ix.item_pos.data_ptr(), ix.row_size.data_ptr(), ix.row_cat.data_ptr(),
            ix.row_trip.data_ptr(), ix.flat_slot.data_ptr(), rho_c.data_ptr(),
            rho_0.data_ptr(), mask.data_ptr())
        if rc != 0:
            raise RuntimeError(f"native gibbs_sweeps_cpp failed with status {rc}")
        return mask.bool()
    for _ in range(sweeps):
        z = basket_field(model, ix, X, r) + torch.randn(ix.B, r, generator=gen)
        X = sample_slots(model, ix, slot_utilities(model, ix, z, r, beta), gen, use_cpp=False)
    return X


# ---- objectives -------------------------------------------------------------------------------
def pseudo_loglik(model, ix, X, r):
    """Mean over trips of sum_j log P(x_j | rest) (exact Version-4 conditionals)."""
    x = X.double()
    ph = phi_r(model, r)[ix.item]
    s = torch.zeros(ix.B, r).index_add(0, ix.item_trip, ph * x[:, None])
    pair = (ph * s[ix.item_trip]).sum(-1) - x * ph.square().sum(-1)
    n_row = torch.zeros(ix.n_rows).index_add(0, ix.row_of, x)
    nc_wo = n_row[ix.row_of] - x
    pf = model.pair_feature
    dcat = model.rho_c[ix.row_cat[ix.row_of]] * (pf(nc_wo + 1) - pf(nc_wo))
    n_trip = torch.zeros(ix.B).index_add(0, ix.item_trip, x)
    n_wo = (n_trip[ix.item_trip] - x).long()
    rho0 = model.rho_0()
    dsize = rho0[(n_wo + 1).clamp(max=model.nmax)] - rho0[n_wo]
    d = model.b_flat(ix) + pair - dcat - dsize
    valid = (n_wo > 0) & (n_wo < model.nmax)
    lp = F.logsigmoid(torch.where(X, d, -d))
    return (lp * valid).sum() / ix.B


def penalties(model, args, n_train, r, tied=None):
    loss = 0.0
    for product, context in ((model.mu, model.delta_c()), (model.zeta, model.xi_c()),
                             (model.alpha, model.theta_c())):
        loss = loss + args.pool_prod * torch.trace((product.T @ product) @ (context.T @ context)) / (
            product.shape[0] * context.shape[0])
    if tied is None:
        phi_prior = args.phi_prec * phi_r(model, r).square().sum()
    else:                       # phi_j = psi_c(j) + delta_j: category vector plus a shrunk product deviation
        psi, dlt, _ = tied
        phi_prior = args.phi_prec * psi.square().sum() + args.delta_prec * dlt.square().sum()
    loss = loss + 0.5 * (phi_prior + args.rho_c_prec * model.rho_c.square().sum()) / n_train
    return loss


def project(model, caps, optimizer, r):
    from category_safety import project_category_reward_
    model.project_context_gauges()
    with torch.no_grad():
        mean_lam = model.lam.mean().clone()
        model.lam.sub_(mean_lam)
        model.rho_0_free.sub_(mean_lam * torch.arange(1, model.rho_0_free.numel() + 1))
        model.phi[:, r:] = 0.0
    model.project_rho_c(-1.5)
    project_category_reward_(model, caps, 1.5, optimizer=optimizer)


# ---- train ------------------------------------------------------------------------------------
def train(args):
    from category_safety import category_capacities
    from pipeline_support import supported_trips
    data, model, meta, batcher = setup(args.initialization, args.additive, args.threads)
    J, C, nmax, r = int(data["n_item"]), int(data["n_cat"]), int(meta["nmax"]), args.rank
    train_trips = supported_trips(data, 0, nmax)
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(train_trips)
    t0 = time.time()
    batches = [Batch(batcher, np.sort(order[s:s + args.batch]), J) for s in range(0, len(order), args.batch)]
    log = open(args.out.with_suffix(".log"), "a")

    def emit(row):
        line = json.dumps(row)
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    emit({"event": "start", "train_trips": len(train_trips), "batches": len(batches),
          "batching_seconds": round(time.time() - t0, 1), "args": {k: str(v) for k, v in vars(args).items()}})
    torch.manual_seed(args.seed)
    tied = None
    if args.tie_phi:
        cat = model.cat_of.long()
        psi = (args.phi_init * torch.randn(C, r)).requires_grad_(True)
        dlt = torch.zeros(J, r, requires_grad=True)
        tied = (psi, dlt, cat)
        with torch.no_grad():
            model.phi.zero_()
            model.phi[:, :r] = psi[cat] + dlt
    else:
        with torch.no_grad():
            model.phi[:, :r] = args.phi_init * torch.randn(J, r)
    for n in ("gamma", "beta", "price_kappa", "psi"):
        getattr(model, n).requires_grad_(False)
    base = [getattr(model, n) for n in BASE_NAMES]
    other_base = [getattr(model, n) for n in OTHER_BASE_NAMES]
    model.phi.requires_grad_(True)
    caps = category_capacities(data, C, nmax)
    N = len(train_trips)
    gen = torch.Generator().manual_seed(args.seed + 1)
    started = time.time()

    # Empirical training incidence for natural-gradient / moment-matching preconditioning of rho_0 and rho_c
    with torch.no_grad():
        size_counts = torch.zeros(nmax + 1)
        cat_pair_sum = torch.zeros(C)
        for bt in batches:
            n_b = torch.zeros(bt.ix.B, dtype=torch.long).index_add_(0, bt.lt, torch.ones_like(bt.li))
            size_counts.index_add_(0, n_b.clamp(max=nmax), torch.ones_like(n_b, dtype=torch.float64))
            nc_row = torch.zeros(bt.ix.n_rows).index_add_(0, bt.ix.row_of, bt.obs.double())
            cat_pair_sum.index_add_(0, bt.ix.row_cat, model.pair_feature(nc_row))
        p_size_prec = (size_counts[1:] / N).clamp_min(2e-3)   # rho_0_free[k] is the size-(k+1) potential
        d_cat_prec = (cat_pair_sum / N).clamp_min(5e-3)

    def run_stage(name, epochs, lr_base, lr_rho, lr_rho0, lr_phi, step_fn, average, mm_rho=0.0, mm_rho0=0.0):
        opt = torch.optim.Adam([
            {"params": other_base, "lr": lr_base},
            {"params": [model.rho_c], "lr": lr_rho},
            {"params": [model.rho_0_free], "lr": lr_rho0},
            {"params": [model.phi] if tied is None else [tied[0], tied[1]], "lr": lr_phi},
        ])
        phi_params = opt.param_groups[3]["params"]
        total = epochs * len(batches)
        decay = (args.lr_final_ratio) ** (1.0 / max(total - 1, 1))
        avg, n_avg, it = None, 0, 0
        for epoch in range(1, epochs + 1):
            t_ep = time.time()
            acc, sizes = [], []
            g_r0_sum = torch.zeros_like(model.rho_0_free)
            g_rc_sum = torch.zeros_like(model.rho_c)
            for bi in rng.permutation(len(batches)):
                it += 1
                bt = batches[bi]
                bt.bind(model)
                value, extra = step_fn(bt)
                loss = -value + penalties(model, args, N, r, tied)
                opt.zero_grad(set_to_none=True)
                if tied is not None:
                    model.phi.grad = None
                loss.backward()
                if tied is not None:                         # chain rule phi -> (psi, delta)
                    with torch.no_grad():
                        g = model.phi.grad[:, :r]
                        tied[0].grad = tied[0].grad + torch.zeros_like(tied[0]).index_add_(0, tied[2], g)
                        tied[1].grad = tied[1].grad + g
                        model.phi.grad = None
                g_r0 = model.rho_0_free.grad.detach().clone()
                g_rc = model.rho_c.grad.detach().clone()
                g_r0_sum.add_(g_r0)
                g_rc_sum.add_(g_rc)
                torch.nn.utils.clip_grad_norm_(base + list(phi_params), 10.0)
                opt.step()
                if tied is not None:
                    with torch.no_grad():
                        model.phi[:, :r] = tied[0][tied[2]] + tied[1]
                if mm_rho > 0.0 or mm_rho0 > 0.0:
                    scale_lr = opt.param_groups[3]["lr"] / max(lr_phi, 1e-12) if lr_phi > 0 else 1.0
                    with torch.no_grad():
                        if mm_rho0 > 0.0:
                            model.rho_0_free.sub_(mm_rho0 * scale_lr * g_r0)
                        if mm_rho > 0.0:
                            model.rho_c.sub_((mm_rho * scale_lr / len(batches)) * (g_rc / d_cat_prec))
                project(model, caps, opt, r)
                for gp in opt.param_groups:
                    gp["lr"] *= decay
                acc.append(float(value.detach()))
                if extra is not None:
                    sizes.append(extra)
                if average and it > total // 2:
                    st = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    avg = st if avg is None else {k: (avg[k] + (st[k] - avg[k]) / (n_avg + 1)) if st[k].is_floating_point()
                                                   else st[k] for k in st}
                    n_avg += 1
            if mm_rho > 0.0 or mm_rho0 > 0.0:
                with torch.no_grad():
                    g_r0_bar = g_r0_sum / len(batches)
                    g_rc_bar = g_rc_sum / len(batches)
                    if mm_rho0 > 0.0:
                        model.rho_0_free.sub_(0.5 * (g_r0_bar / p_size_prec).clamp(-0.5, 0.5))
                    if mm_rho > 0.0:
                        model.rho_c.sub_(0.5 * (g_rc_bar / d_cat_prec).clamp(-0.5, 0.5))
                project(model, caps, opt, r)
            sv = torch.linalg.svdvals(phi_r(model, r).detach())
            row = {"stage": name, "epoch": epoch, "objective": float(np.mean(acc)),
                   "phi_eig_top3": [round(float(v), 3) for v in (sv[:3] ** 2)],
                   "lr_phi": opt.param_groups[3]["lr"],
                   "epoch_seconds": round(time.time() - t_ep, 2),
                   "seconds": round(time.time() - started, 1)}
            if sizes:
                row["chain_mean_size"] = float(np.mean([s[0] for s in sizes]))
                row["data_mean_size"] = float(np.mean([s[1] for s in sizes]))
            emit(row)
        if avg is not None:
            model.load_state_dict(avg)
            if tied is not None:                             # keep phi = psi[cat] + delta at the averaged phi
                with torch.no_grad():
                    tied[1].copy_(model.phi[:, :r] - tied[0][tied[2]])

    # stage A: pseudo-likelihood (gentle Phi warm-up with separate rho_0 / rho_c learning rates)
    run_stage("pseudo-likelihood", args.pl_epochs, args.lr_base, args.lr_rho, args.lr_rho0, args.lr_phi_pl,
              lambda bt: (pseudo_loglik(model, bt.ix, bt.obs, r), None), average=False)
    torch.save({"model": {k: v.detach().clone() for k, v in model.state_dict().items()}, "rank": r,
                "stage": "pseudo-likelihood"}, args.out.with_name(args.out.stem + "_pl.pt"))

    # stage B: stochastic maximum likelihood with persistent per-trip chains (data-initialized)
    chains = [bt.obs.clone() for bt in batches]
    index = {id(bt): i for i, bt in enumerate(batches)}

    def sml_step(bt):
        i = index[id(bt)]
        ix = bt.ix
        X = gibbs(model, ix, chains[i], r, gen, sweeps=args.sweeps)
        chains[i] = X
        with torch.no_grad():
            z = basket_field(model, ix, X, r) + torch.randn(ix.B, r, generator=gen)
        e_obs = model.energy(bt.li, bt.lt, bt.lc, ix.B, bt.line_ctx)
        lz = log_zc(model, ix, slot_utilities(model, ix, z, r))
        size_chain = float(torch.zeros(ix.B).index_add_(0, ix.item_trip, X.double()).mean())
        return (e_obs - lz).mean(), (size_chain, float(bt.obs.sum()) / ix.B)

    run_stage("sml", args.sml_epochs, args.lr_base, args.lr_rho, args.lr_rho0, args.lr_phi_sml, sml_step,
              average=True, mm_rho=args.mm_rho, mm_rho0=args.mm_rho0)
    if args.cal_epochs > 0:
        run_stage("calibrate", args.cal_epochs, 0.0, args.lr_rho, args.lr_rho0, 0.0, sml_step,
                  average=False, mm_rho=args.mm_rho, mm_rho0=args.mm_rho0)
    torch.save({"model": {k: v.detach().clone() for k, v in model.state_dict().items()}, "rank": r,
                "stage": "sml", "seconds": time.time() - started, "args": {k: str(v) for k, v in vars(args).items()}},
               args.out.with_suffix(".pt"))
    emit({"event": "trained", "seconds": round(time.time() - started, 1)})


# ---- evaluate ---------------------------------------------------------------------------------
@torch.no_grad()
def is_log_z(model, bt, r, gen, burn, components, thin, draws, defensive=0.1):
    """Importance-sampled log Z per trip (Step 1's estimator) and the final chain state."""
    ix = bt.ix
    X = sample_slots(model, ix, slot_utilities(model, ix, None, r, beta=0.0), gen)
    X = gibbs(model, ix, X, r, gen, sweeps=burn)
    centres = []
    for k in range(components * thin):
        X = gibbs(model, ix, X, r, gen)
        if (k + 1) % thin == 0:
            centres.append(basket_field(model, ix, X, r))
    centres = torch.stack(centres, 1)
    B, Kc = ix.B, centres.shape[1]
    half = 0.5 * r * math.log(2 * math.pi)
    log_w = []
    for _ in range(draws):
        pick = torch.randint(Kc, (B,), generator=gen)
        prior = torch.rand(B, generator=gen) < defensive
        mean = torch.where(prior[:, None], torch.zeros(B, r), centres[torch.arange(B), pick])
        z = mean + torch.randn(B, r, generator=gen)
        lp = -0.5 * z.square().sum(1) - half
        comp = -0.5 * (z[:, None, :] - centres).square().sum(2) - half
        lq = torch.logaddexp(math.log1p(-defensive) + torch.logsumexp(comp, 1) - math.log(Kc),
                             math.log(defensive) + lp)
        log_w.append(lp + log_zc(model, ix, slot_utilities(model, ix, z, r)) - lq)
    log_w = torch.stack(log_w, 1)
    est = torch.logsumexp(log_w, 1) - math.log(draws)
    h = draws // 2
    est_a = torch.logsumexp(log_w[:, :h], 1) - math.log(h)
    est_b = torch.logsumexp(log_w[:, h:], 1) - math.log(draws - h)
    ess = 1.0 / torch.softmax(log_w, 1).square().sum(1) / draws
    return est, est_a - est_b, ess, X


def clustered_ci(delta, cluster, reps=2000, seed=0):
    codes, idx = np.unique(cluster, return_inverse=True)
    sums, counts = np.bincount(idx, delta), np.bincount(idx).astype(float)
    draws = np.random.default_rng(seed).integers(len(codes), size=(reps, len(codes)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def classifier_auc(real, generated, groups, n_item, item_cat, n_cat, kind):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    def features(baskets):
        X = np.zeros((len(baskets), (n_item if kind == "full" else 0) + n_cat + (1 if kind == "full" else 0)),
                     np.float32)
        for i, items in enumerate(baskets):
            if kind == "full":
                X[i, items] = 1.0
                np.add.at(X[i], n_item + item_cat[items], 1.0)
                X[i, -1] = len(items)
            else:
                X[i, item_cat[items]] = 1.0
        return X
    X = np.r_[features(real), features(generated)]
    y = np.r_[np.ones(len(real)), np.zeros(len(generated))]
    aucs = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        clf = HistGradientBoostingClassifier(max_iter=300, early_stopping=True, random_state=k).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))


def evaluate(args):
    from pipeline_support import supported_trips
    from price_response import changed_price_context
    data, model, meta, batcher = setup(args.initialization, args.additive, args.threads)
    blob = torch.load(args.out.with_suffix(".pt") if args.checkpoint is None else args.checkpoint,
                      map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    r = int(blob["rank"])
    J, C, nmax = int(data["n_item"]), int(data["n_cat"]), int(meta["nmax"])
    item_cat = np.zeros(J, dtype=np.int64)
    item_cat[data["line_item"]] = data["line_cat"]
    gen = torch.Generator().manual_seed(args.seed + 5)
    tag = args.out.stem if args.checkpoint is None else Path(args.checkpoint).stem
    report = {"checkpoint": str(args.checkpoint or args.out.with_suffix(".pt")),
              "phi_eigenvalues": (torch.linalg.svdvals(phi_r(model, r)) ** 2).tolist(),
              "rho_c_min": float(model.rho_c.min())}
    t0 = time.time()

    # 1. held-out log-likelihood on the certified reference trips
    ref = np.load(args.reference)
    trips = ref["trips"].astype(np.int64)
    if args.ll_trips:
        trips = trips[:args.ll_trips]
    ll, half_diff, ess_all, gen_baskets = [], [], [], {}
    for s in range(0, len(trips), args.eval_batch):
        bt = Batch(batcher, trips[s:s + args.eval_batch], J)
        bt.bind(model)
        est, dhalf, ess, X = is_log_z(model, bt, r, gen, args.burn, args.components, args.thin, args.draws)
        e = model.energy(bt.li, bt.lt, bt.lc, bt.ix.B, bt.line_ctx)
        ll.append((e - est).numpy()); half_diff.append(dhalf.numpy()); ess_all.append(ess.numpy())
        items = bt.ix.item.numpy(); xt = bt.ix.item_trip.numpy(); Xn = X.numpy()
        for b, t in enumerate(bt.trips):
            gen_baskets[int(t)] = np.sort(items[Xn & (xt == b)])
        print(f"[eval {tag}] LL {s + len(bt.trips)}/{len(trips)} running mean {np.concatenate(ll).mean():.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
    ll, half_diff, ess_all = map(np.concatenate, (ll, half_diff, ess_all))
    n = len(ll)
    report["likelihood"] = {
        "trips": n, "mean_ll": float(ll.mean()),
        "certified_v4_mean_ll": float(ref["target_child"][:n].mean()),
        "vs_certified_v4": float((ll - ref["target_child"][:n]).mean()),
        "vs_certified_v4_household_95": clustered_ci(ll - ref["target_child"][:n], ref["household"][:n]),
        "vs_certified_phi0_parent": float((ll - ref["exact_parent"][:n]).mean()),
        "is_half_split_abs_mean": float(np.abs(half_diff).mean()), "is_half_split_abs_max": float(np.abs(half_diff).max()),
        "is_ess_fraction_mean": float(ess_all.mean()), "is_ess_fraction_min": float(ess_all.min())}
    if args.phi0_refit is not None:
        o = np.load(args.phi0_refit)
        m = dict(zip(o["trips"].tolist(), o["ll"].tolist()))
        d = ll - np.array([m[int(t)] for t in trips])
        report["likelihood"]["vs_phi0_refit"] = float(d.mean())
        report["likelihood"]["vs_phi0_refit_household_95"] = clustered_ci(d, ref["household"][:n])
    np.savez(args.out.with_name(tag + ".per_trip.npz"), trips=trips, ll=ll)
    print(json.dumps(report["likelihood"]), flush=True)

    # 2. classifier two-sample test on all test trips (one chain basket per trip after burn-in)
    test = supported_trips(data, 2, nmax)
    lptr = data["line_ptr"]
    real = [np.sort(data["line_item"][lptr[t]:lptr[t + 1]]) for t in test]
    generated = []
    for s in range(0, len(test), args.eval_batch):
        part = test[s:s + args.eval_batch]
        if all(int(t) in gen_baskets for t in part):
            generated += [gen_baskets[int(t)] for t in part]
            continue
        bt = Batch(batcher, part, J)
        bt.bind(model)
        X = sample_slots(model, bt.ix, slot_utilities(model, bt.ix, None, r, beta=0.0), gen)
        X = gibbs(model, bt.ix, X, r, gen, sweeps=args.burn).numpy()
        items, xt = bt.ix.item.numpy(), bt.ix.item_trip.numpy()
        generated += [np.sort(items[X & (xt == b)]) for b in range(len(part))]
    groups = np.r_[data["trip_user"][test], data["trip_user"][test]]
    full = classifier_auc(real, generated, groups, J, item_cat, C, "full")
    cats = classifier_auc(real, generated, groups, J, item_cat, C, "category")
    report["classifier"] = {"full_basket_auc": full[0], "category_presence_auc": cats[0],
                            "size_mean_real": float(np.mean([len(x) for x in real])),
                            "size_mean_generated": float(np.mean([len(x) for x in generated]))}
    print(json.dumps(report["classifier"]), flush=True)

    # 3. price scenarios: base chains give z draws; a changed price reweights them exactly
    #    (p_new(z) / p_base(z) is proportional to Zc_new(z) / Zc_base(z)), Rao-Blackwellized incidence
    o = np.load(args.world / "observed.npz")
    truth_meta = json.loads((args.world / "truth.json").read_text())
    period = o["trip_period"]
    split = np.where(period <= truth_meta["train_weeks"], 0,
                     np.where(period <= truth_meta["train_weeks"] + truth_meta["validation_weeks"], 1, 2))
    rows = np.sort(np.random.default_rng(77).choice(np.flatnonzero(split == 2), size=args.scenario_trips,
                                                    replace=False))
    to_model = {int(orig): i for i, orig in enumerate(o["original_item"])}
    truth = json.loads((args.world / "counterfactual_truth.json").read_text())
    cat, sub = o["category"], o["subcategory"]
    sets = []
    for s in range(0, len(rows), args.eval_batch):
        bt = Batch(batcher, rows[s:s + args.eval_batch], J)
        bt.bind(model)
        X = sample_slots(model, bt.ix, slot_utilities(model, bt.ix, None, r, beta=0.0), gen)
        X = gibbs(model, bt.ix, X, r, gen, sweeps=args.burn)
        zs = []
        for k in range(args.scenario_draws * args.thin):
            X = gibbs(model, bt.ix, X, r, gen)
            if (k + 1) % args.thin == 0:
                zs.append(basket_field(model, bt.ix, X, r) + torch.randn(bt.ix.B, r, generator=gen))
        sets.append((bt, zs))

    def incidence(product=None, multiplier=1.0):
        inc, size = np.zeros(J), 0.0
        for bt, zs in sets:
            ix = bt.ix
            ctx = bt.ctx
            if product is not None:
                change = torch.where(ix.item == product, math.log(multiplier), 0.0)
                ctx = changed_price_context(bt.ctx, ix.item_trip, change)
            lzs, grads, lzb = [], [], []
            for z in zs:
                bt.bind(model, ctx)
                u = slot_utilities(model, ix, z, r).detach().requires_grad_(True)
                with torch.enable_grad():
                    lz = log_zc(model, ix, u)
                    g, = torch.autograd.grad(lz.sum(), u)
                lzs.append(lz.detach()); grads.append(g)
                if product is not None:
                    bt.bind(model)
                    lzb.append(log_zc(model, ix, slot_utilities(model, ix, z, r)))
            lzs = torch.stack(lzs, 1)                                # B x K
            logw = (lzs - torch.stack(lzb, 1)) if product is not None else torch.zeros_like(lzs)
            w = torch.softmax(logw, 1)                               # per-trip self-normalized weights
            G = torch.stack(grads, 1)                                # T x K
            m = (G * w[ix.item_trip]).sum(1)
            np.add.at(inc, ix.item.numpy(), m.numpy())
            size += float(m.sum())
        bt.bind(model)
        return inc, size / len(rows)

    base_inc, base_size = incidence()
    rel = lambda a, b: float(b / a - 1.0) if a > 0 else None  # noqa: E731
    results = []
    for sc in truth["scenarios"]:
        k = to_model.get(int(sc["product"]))
        if k is None:
            continue
        alt, alt_size = incidence(k, sc["price_multiplier"])
        same = np.flatnonzero((cat == cat[k]) & (sub == sub[k]) & (np.arange(J) != k))
        rest = np.flatnonzero((cat == cat[k]) & (sub != sub[k]))
        results.append({"product": int(sc["product"]), "price_multiplier": sc["price_multiplier"],
                        "own": {"truth": rel(*sc["own_incidence"]), "model": rel(base_inc[k], alt[k])},
                        "same_subcategory": {"truth": rel(*sc["same_subcategory_incidence"]),
                                             "model": rel(base_inc[same].sum(), alt[same].sum())},
                        "rest_of_category": {"truth": rel(*sc["rest_of_category_incidence"]),
                                             "model": rel(base_inc[rest].sum(), alt[rest].sum())},
                        "basket_size_change": {"truth": sc["mean_basket_size"][1] - sc["mean_basket_size"][0],
                                               "model": alt_size - base_size}})

    def agreement(key):
        pairs = np.array([(x[key]["truth"], x[key]["model"]) for x in results
                          if x[key]["truth"] is not None and x[key]["model"] is not None])
        t_, m_ = pairs.T
        return {"scenarios": len(pairs), "correlation": float(np.corrcoef(t_, m_)[0, 1]),
                "mean_abs_error": float(np.abs(t_ - m_).mean()),
                "sign_agreement": float(np.mean(np.sign(t_) == np.sign(m_))),
                "mean_truth": float(t_.mean()), "mean_model": float(m_.mean())}
    report["price"] = {k: agreement(k) for k in ("own", "same_subcategory", "rest_of_category", "basket_size_change")}
    report["price_scenarios"] = results
    print(json.dumps(report["price"]), flush=True)

    L = report["likelihood"]
    report["pass_rules"] = {
        "1_ll_vs_certified_ci_above_0": L["vs_certified_v4_household_95"][0] > 0,
        "2_category_auc_le_0.80": report["classifier"]["category_presence_auc"] <= 0.80,
        "3_own_price_corr_ge_0.85": report["price"]["own"]["correlation"] >= 0.85}
    report["eval_seconds"] = time.time() - t0
    args.out.with_name(tag + ".eval.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["pass_rules"]), flush=True)


def main():
    global USE_CPP_SAMPLER
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=("train", "eval"))
    p.add_argument("--initialization", type=Path, default=A / "artifacts/initialization.pt")
    p.add_argument("--additive", type=Path, default=A / "out/v3_pipeline_additive_best.pt")
    p.add_argument("--reference", type=Path, default=A / "reports/likelihood_test_paired_per_trip.npz")
    p.add_argument("--phi0-refit", type=Path,
                   default=ROOT.parent / "ebi_v4x/runs/stress/base.per_trip.npz")
    p.add_argument("--world", type=Path, default=ROOT / "data/stress_world")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--phi-init", type=float, default=0.05)
    p.add_argument("--pl-epochs", type=int, default=2)
    p.add_argument("--sml-epochs", type=int, default=16)
    p.add_argument("--cal-epochs", type=int, default=1)
    p.add_argument("--sweeps", type=int, default=4)
    p.add_argument("--lr-base", type=float, default=3e-4)
    p.add_argument("--lr-rho", type=float, default=3e-2)
    p.add_argument("--lr-rho0", type=float, default=5e-2)
    p.add_argument("--lr-phi-pl", type=float, default=1e-3)
    p.add_argument("--lr-phi-sml", type=float, default=3e-3)
    p.add_argument("--mm-rho", type=float, default=1.0)
    p.add_argument("--mm-rho0", type=float, default=0.5)
    p.add_argument("--lr-final-ratio", type=float, default=0.1)
    p.add_argument("--pool-prod", type=float, default=1.45)
    p.add_argument("--phi-prec", type=float, default=2.0)
    p.add_argument("--rho-c-prec", type=float, default=0.2)
    p.add_argument("--tie-phi", action="store_true", help="phi_j = psi_category(j) + delta_j")
    p.add_argument("--delta-prec", type=float, default=200.0, help="prior precision on delta (with --tie-phi)")
    p.add_argument("--cpp-sampler", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--ll-trips", type=int, default=0)
    p.add_argument("--burn", type=int, default=200)
    p.add_argument("--components", type=int, default=128)
    p.add_argument("--thin", type=int, default=4)
    p.add_argument("--draws", type=int, default=1024)
    p.add_argument("--scenario-trips", type=int, default=400)
    p.add_argument("--scenario-draws", type=int, default=64)
    p.add_argument("--seed", type=int, default=41001)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    USE_CPP_SAMPLER = bool(args.cpp_sampler)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    (train if args.command == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
