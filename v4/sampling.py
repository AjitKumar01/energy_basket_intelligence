"""Sampling baskets and estimating log Z when phi != 0.

The pairwise term is a Gaussian integral over a latent z (model.slot_utilities_given_z):

    p(S | x) proportional to  E_{z ~ N(0, I)} [ exp E_z(S) ],

where E_z has no pairwise term.  So:

  * given z, S is sampled exactly (``sample_baskets``, the C++ sampler);
  * given S, z ~ N(s(S), I) with s(S) = model.basket_field;
  * alternating the two is a block Gibbs chain on (z, S) whose S-marginal is the model
    (``gibbs``);
  * log Z(x) = log E_z Z_z(x) is estimated by importance sampling over z with a proposal
    built from the chain's s(S) values (``importance_log_z``).

All random numbers come from a torch.Generator; the C++ code takes one seed drawn from it
per call, so a generator seed fixes every result.
"""
from __future__ import annotations

import math
import os

import torch

from .exact import log_partition
from .kernels import check, ptr, sampler


def _seed(gen):
    return int(torch.randint(1, 2 ** 62, (1,), generator=gen).item())


def _arrays(model, **optional):
    """Detached contiguous copies of the model arrays the C++ code reads, plus the given
    optional ones (None stays None and is passed as a null pointer)."""
    out = {"rho_c": model.rho_c.detach().contiguous(),
           "rho_0": model.rho_0().detach().contiguous(),
           "f_extra": None if model.f_extra is None else model.f_extra.detach().contiguous()}
    for name, value in optional.items():
        out[name] = None if value is None else value.detach().contiguous()
    return out


def _max_count(model, ix):
    """Largest number of products the DP lets one category hold in this batch."""
    return min(model.R, int(ix.row_size.max()), model.nmax)


def _min_cap(model):
    """Smallest per-(trip, z) item cap of the sampler (the cap is certified: <= 1e-8 of the
    mass is cut).  V4_LEGACY=1 turns it off; V4_EXACT_SWEEPS=1 only restores the old sweep
    schedule of importance_log_z and leaves the cap on."""
    if os.environ.get("V4_LEGACY") == "1":
        return int(model.nmax)
    return int(getattr(model, "trunc_min_cap", 32))


def _cached_trip_statics(model, ix, iv=None):
    """Cache (b, iv, ph, half_phi_sq) on ix keyed by model parameter/context versions."""
    ctx_key = None if model.ctx is None else tuple((k, id(v), v._version) for k, v in model.ctx.items())
    param_key = tuple((id(p), p._version) for p in model.parameters())
    key = (id(model), id(model.house), None if model.house is None else model.house._version,
           ctx_key, param_key, model.centre_history, model.rank, model.nmax, model.R)
    cache = getattr(ix, "_gibbs_cache", None)
    if cache is not None and cache[0] == key:
        b, cached_iv, ph, half_phi_sq = cache[1]
        return b, (cached_iv if iv is None else iv), ph, half_phi_sq
    b = model.slot_utilities(ix).detach().contiguous()
    cached_iv = model.iv_shift(ix, b_slot=b)
    if cached_iv is not None:
        cached_iv = cached_iv.detach().contiguous()
    ph = model.phi_r()[ix.item].detach().contiguous()
    half_phi_sq = (0.5 * ph.square().sum(-1)).contiguous()
    ix._gibbs_cache = (key, (b, cached_iv, ph, half_phi_sq))
    return b, (cached_iv if iv is None else iv), ph, half_phi_sq


@torch.no_grad()
def sample_baskets(model, ix, u, gen, *, z=None, beta=1.0, iv=None):
    """One exact draw of S per trip from the law given slot utilities u (and z for M2).
    Returns a [T] bool slot mask."""
    shift = model.presence_shift(ix, z, beta=beta)
    iv = model.iv_shift(ix) if iv is None else iv
    # Every array handed to C++ is held in a local variable until the call returns.
    a = _arrays(model, u=u, shift=shift, iv=iv)
    out = torch.empty(len(ix.item), dtype=torch.uint8)
    seed = _seed(gen)
    check(sampler().sample_slots_cpp(
        ix.B, ix.n_rows, len(ix.item), ix.Cpad, _max_count(model, ix), model.nmax,
        int(model.size_on_categories), _min_cap(model), seed,
        ptr(a["u"]), ptr(ix.row_of), ptr(ix.item_pos), ptr(ix.row_size), ptr(ix.row_cat),
        ptr(ix.flat_slot), ptr(a["rho_c"]), ptr(a["rho_0"]), ptr(a["f_extra"]),
        ptr(a["shift"]), ptr(a["iv"]), ptr(out)), "sampler")
    return out.bool()


@torch.no_grad()
def gibbs(model, ix, X, gen, *, sweeps=1, beta=1.0, iv=None):
    """``sweeps`` block-Gibbs sweeps from slot mask X: z ~ N(s(X), I), then X ~ S | z."""
    if os.environ.get("V4_LEGACY") == "1":
        b = model.slot_utilities(ix).detach()
        iv = model.iv_shift(ix, b_slot=b) if iv is None else iv
        ph = model.phi_r()[ix.item].detach().contiguous()
        b_base = (b - 0.5 * beta * ph.square().sum(-1)).contiguous()
    else:
        b, iv, ph, half_phi_sq = _cached_trip_statics(model, ix, iv=iv)
        b_base = (b - float(beta) * half_phi_sq).contiguous()
    a = _arrays(model, iv=iv, psi=model.psi_cat)
    seed = _seed(gen)
    mask = X.to(torch.uint8).contiguous().clone()
    check(sampler().gibbs_sweeps_cpp(
        ix.B, ix.n_rows, len(ix.item), ix.Cpad, _max_count(model, ix), model.nmax, model.rank,
        int(sweeps), int(model.size_on_categories), _min_cap(model), math.sqrt(float(beta)), seed,
        ptr(b_base), ptr(ph), ptr(ix.item_trip), ptr(ix.row_of), ptr(ix.item_pos),
        ptr(ix.row_size), ptr(ix.row_cat), ptr(ix.row_trip), ptr(ix.flat_slot),
        ptr(a["rho_c"]), ptr(a["rho_0"]), ptr(a["f_extra"]), ptr(a["psi"]), ptr(a["iv"]),
        ptr(mask)), "Gibbs")
    return mask.bool()


@torch.no_grad()
def draw_from_model(model, ix, gen, burn):
    """A basket per trip from the full model: start from the law without pairwise terms
    (beta = 0), then ``burn`` Gibbs sweeps."""
    if os.environ.get("V4_LEGACY") == "1":
        X = sample_baskets(model, ix, model.slot_utilities_given_z(ix, None, beta=0.0), gen, beta=0.0)
        return gibbs(model, ix, X, gen, sweeps=burn)
    b, iv, _, _ = _cached_trip_statics(model, ix)
    X = sample_baskets(model, ix, b, gen, beta=0.0, iv=iv)
    return gibbs(model, ix, X, gen, sweeps=burn, iv=iv)


@torch.no_grad()
def importance_log_z(model, ix, gen, *, burn, components, thin, draws, defensive=0.1):
    """Importance-sampled log Z per trip.

    Proposal for z: a mixture of N(s_k, I) over chain states s_k = s(S_k), plus a
    defensive wide Gaussian N(0, diag(sig^2)) with weight ``defensive`` (sig = 1.35 along
    the strong directions of phi, eigenvalue > 4).  Chain states come from four chains:
    one from beta = 0 with ``burn`` sweeps, and three started apart (z = 0 and
    z = +-1.5 sv_1 v_1) and annealed through beta = 0.25, 0.5, 0.75, 1.  Each half of the
    draws uses its own stratified component picks, so the two half estimates are
    independent (their difference is reported as a convergence check).

    Returns (log Z, half-split difference, ESS fraction, a model basket per trip).
    """
    B, r = ix.B, model.rank
    ph = model.phi_r().detach()
    _, sv, Vt = torch.linalg.svd(ph, full_matrices=False)
    strong = sv.square() > 4.0
    sig = torch.where(strong, torch.full_like(sv, 1.35), torch.ones_like(sv))
    log_det = sig.log().sum()

    if os.environ.get("V4_LEGACY") == "1":
        X = draw_from_model(model, ix, gen, burn)
        centres, per_chain, thin_c = [], max(components // 2, 64), max(thin // 2, 1)
        X_out = None
        for k in range(max(components * thin, per_chain * thin_c)):
            X = gibbs(model, ix, X, gen)
            if (k + 1) % thin_c == 0 and len(centres) < per_chain:
                centres.append(model.basket_field(ix, X))
            if k + 1 == components * thin:
                X_out = X.clone()
        X_out = X.clone() if X_out is None else X_out

        anneal = max(burn // 8, 25)
        for chain in range(1, 4):
            if chain == 1:
                z0 = torch.zeros(B, r)
            elif float(sv[0]) > 1.0:
                z0 = ((1.0 if chain == 2 else -1.0) * 1.5 * sv[0] * Vt[0])[None, :].expand(B, r)
            else:
                z0 = None
            if z0 is None:
                Xc = sample_baskets(model, ix, model.slot_utilities_given_z(ix, None, beta=0.0), gen, beta=0.0)
            else:
                Xc = sample_baskets(model, ix, model.slot_utilities_given_z(ix, z0), gen, z=z0)
            for b_temp in (0.25, 0.5, 0.75, 1.0):
                Xc = gibbs(model, ix, Xc, gen, sweeps=anneal, beta=b_temp)
            for k in range(per_chain * thin_c):
                Xc = gibbs(model, ix, Xc, gen)
                if (k + 1) % thin_c == 0:
                    centres.append(model.basket_field(ix, Xc))
        centres = torch.stack(centres, 1) @ Vt.T
    elif not model.size_on_categories:
        b_slot, iv, ph_slot, _ = _cached_trip_statics(model, ix)
        X = sample_baskets(model, ix, b_slot, gen, beta=0.0, iv=iv)
        X = gibbs(model, ix, X, gen, sweeps=burn, iv=iv)
        centres, per_chain, thin_c = [], max(components // 2, 48), max(thin // 2, 1)
        X_out = None
        for k in range(per_chain * thin_c):
            X = gibbs(model, ix, X, gen, iv=iv)
            if (k + 1) % thin_c == 0 and len(centres) < per_chain:
                centres.append(model.basket_field(ix, X))
            if k + 1 == components * thin:
                X_out = X.clone()
        X_out = X.clone() if X_out is None else X_out
        anneal = max(burn // 12, 12)
        for chain in range(1, 4):
            if chain == 1:
                z0 = torch.zeros(B, r)
            elif float(sv[0]) > 1.0:
                z0 = ((1.0 if chain == 2 else -1.0) * 1.5 * sv[0] * Vt[0])[None, :].expand(B, r)
            else:
                z0 = None
            if z0 is None:
                Xc = sample_baskets(model, ix, b_slot, gen, beta=0.0, iv=iv)
            else:
                Xc = sample_baskets(model, ix, model.slot_utilities_given_z(ix, z0, b=b_slot), gen, z=z0, iv=iv)
            for b_temp in (0.25, 0.5, 0.75, 1.0):
                Xc = gibbs(model, ix, Xc, gen, sweeps=anneal, beta=b_temp, iv=iv)
            for k in range(per_chain * thin_c):
                Xc = gibbs(model, ix, Xc, gen, iv=iv)
                if (k + 1) % thin_c == 0:
                    centres.append(model.basket_field(ix, Xc))
        centres = torch.stack(centres, 1) @ Vt.T
    else:
        b_slot, iv, ph_slot, half_phi_sq = _cached_trip_statics(model, ix)
        exact_sweeps = os.environ.get("V4_EXACT_SWEEPS") == "1"
        per_chain = max(components // 2, 64 if exact_sweeps else 48)
        thin_c = max(thin // 2, 1)
        anneal = max(burn // 8, 25) if exact_sweeps else max(burn // 12, 12)
        Kc = 4 * per_chain
        centres_raw = torch.empty((Kc, B, r), dtype=torch.float64)
        a = _arrays(model, iv=iv, psi=model.psi_cat)
        K_max = _max_count(model, ix)
        min_cap = _min_cap(model)
        T_slots = len(ix.item)

        # Chain 0 stage parameters
        X0 = sample_baskets(model, ix, b_slot, gen, beta=0.0, iv=iv)
        n_k0 = max(components * thin, per_chain * thin_c) if exact_sweeps else (per_chain * thin_c)
        st_sweeps0 = [int(burn)] + [1] * n_k0
        st_sbeta0 = [1.0] * (1 + n_k0)
        st_seeds0 = [_seed(gen) for _ in range(1 + n_k0)]
        st_cslot0 = [-1]
        c_idx = 0
        save_stage0 = n_k0
        for k in range(n_k0):
            if (k + 1) % thin_c == 0 and c_idx < per_chain:
                st_cslot0.append(c_idx)
                c_idx += 1
            else:
                st_cslot0.append(-1)
            if k + 1 == components * thin:
                save_stage0 = k + 1

        t_sweeps0 = torch.tensor(st_sweeps0, dtype=torch.int64)
        t_sbeta0 = torch.tensor(st_sbeta0, dtype=torch.float64)
        t_seeds0 = torch.tensor(st_seeds0, dtype=torch.int64)
        t_cslot0 = torch.tensor(st_cslot0, dtype=torch.int64)

        # Chains 1-3 stage parameters
        n_kc = per_chain * thin_c
        st_sweeps_c = [int(anneal)] * 4 + [1] * n_kc
        st_sbeta_c = [math.sqrt(0.25), math.sqrt(0.5), math.sqrt(0.75), 1.0] + [1.0] * n_kc
        t_sweeps_c = torch.tensor(st_sweeps_c, dtype=torch.int64)
        t_sbeta_c = torch.tensor(st_sbeta_c, dtype=torch.float64)

        if B == 1:
            masks_all = torch.empty((4, T_slots), dtype=torch.uint8)
            masks_all[0] = X0.to(torch.uint8)
            t_seeds_list = []
            t_cslot_list = []
            for chain in range(1, 4):
                if chain == 1:
                    z0 = torch.zeros(B, r)
                elif float(sv[0]) > 1.0:
                    z0 = ((1.0 if chain == 2 else -1.0) * 1.5 * sv[0] * Vt[0])[None, :].expand(B, r)
                else:
                    z0 = None
                if z0 is None:
                    Xc = sample_baskets(model, ix, b_slot, gen, beta=0.0, iv=iv)
                else:
                    u_z0 = b_slot - half_phi_sq + (z0[ix.item_trip] * ph_slot).sum(-1)
                    Xc = sample_baskets(model, ix, u_z0, gen, z=z0, iv=iv)
                masks_all[chain] = Xc.to(torch.uint8)
                st_seeds_c = [_seed(gen) for _ in range(4 + n_kc)]
                st_cslot_c = [-1] * 4
                for k in range(n_kc):
                    if (k + 1) % thin_c == 0:
                        st_cslot_c.append(c_idx)
                        c_idx += 1
                    else:
                        st_cslot_c.append(-1)
                t_seeds_list.append(torch.tensor(st_seeds_c, dtype=torch.int64))
                t_cslot_list.append(torch.tensor(st_cslot_c, dtype=torch.int64))
            check(sampler().gibbs_four_chains_cpp(
                B, ix.n_rows, T_slots, ix.Cpad, K_max, model.nmax, r,
                int(model.size_on_categories), min_cap,
                len(st_sweeps0), save_stage0,
                ptr(t_sweeps0), ptr(t_sbeta0), ptr(t_seeds0), ptr(t_cslot0),
                len(st_sweeps_c), ptr(t_sweeps_c), ptr(t_sbeta_c),
                ptr(t_seeds_list[0]), ptr(t_cslot_list[0]),
                ptr(t_seeds_list[1]), ptr(t_cslot_list[1]),
                ptr(t_seeds_list[2]), ptr(t_cslot_list[2]),
                ptr(b_slot), ptr(half_phi_sq), ptr(ph_slot), ptr(ix.item_trip), ptr(ix.row_of),
                ptr(ix.item_pos), ptr(ix.row_size), ptr(ix.row_cat), ptr(ix.row_trip),
                ptr(ix.flat_slot), ptr(a["rho_c"]), ptr(a["rho_0"]), ptr(a["f_extra"]),
                ptr(a["psi"]), ptr(a["iv"]), ptr(masks_all), ptr(centres_raw)), "gibbs_four_chains")
            X_out = masks_all[0].bool()
        else:
            mask0 = X0.to(torch.uint8).contiguous().clone()
            check(sampler().gibbs_stages_cpp(
                B, ix.n_rows, T_slots, ix.Cpad, K_max, model.nmax, r,
                int(model.size_on_categories), min_cap, len(st_sweeps0), save_stage0,
                ptr(t_sweeps0), ptr(t_sbeta0), ptr(t_seeds0), ptr(t_cslot0),
                ptr(b_slot), ptr(half_phi_sq), ptr(ph_slot), ptr(ix.item_trip), ptr(ix.row_of),
                ptr(ix.item_pos), ptr(ix.row_size), ptr(ix.row_cat), ptr(ix.row_trip),
                ptr(ix.flat_slot), ptr(a["rho_c"]), ptr(a["rho_0"]), ptr(a["f_extra"]),
                ptr(a["psi"]), ptr(a["iv"]), ptr(mask0), ptr(centres_raw)), "gibbs_stages_c0")
            X_out = mask0.bool()

            for chain in range(1, 4):
                if chain == 1:
                    z0 = torch.zeros(B, r)
                elif float(sv[0]) > 1.0:
                    z0 = ((1.0 if chain == 2 else -1.0) * 1.5 * sv[0] * Vt[0])[None, :].expand(B, r)
                else:
                    z0 = None
                if z0 is None:
                    Xc = sample_baskets(model, ix, b_slot, gen, beta=0.0, iv=iv)
                else:
                    u_z0 = b_slot - half_phi_sq + (z0[ix.item_trip] * ph_slot).sum(-1)
                    Xc = sample_baskets(model, ix, u_z0, gen, z=z0, iv=iv)
                st_seeds_c = [_seed(gen) for _ in range(4 + n_kc)]
                st_cslot_c = [-1] * 4
                for k in range(n_kc):
                    if (k + 1) % thin_c == 0:
                        st_cslot_c.append(c_idx)
                        c_idx += 1
                    else:
                        st_cslot_c.append(-1)
                mask_c = Xc.to(torch.uint8).contiguous()
                t_seeds_c = torch.tensor(st_seeds_c, dtype=torch.int64)
                t_cslot_c = torch.tensor(st_cslot_c, dtype=torch.int64)
                check(sampler().gibbs_stages_cpp(
                    B, ix.n_rows, T_slots, ix.Cpad, K_max, model.nmax, r,
                    int(model.size_on_categories), min_cap, len(st_sweeps_c), -1,
                    ptr(t_sweeps_c), ptr(t_sbeta_c), ptr(t_seeds_c), ptr(t_cslot_c),
                    ptr(b_slot), ptr(half_phi_sq), ptr(ph_slot), ptr(ix.item_trip), ptr(ix.row_of),
                    ptr(ix.item_pos), ptr(ix.row_size), ptr(ix.row_cat), ptr(ix.row_trip),
                    ptr(ix.flat_slot), ptr(a["rho_c"]), ptr(a["rho_0"]), ptr(a["f_extra"]),
                    ptr(a["psi"]), ptr(a["iv"]), ptr(mask_c), ptr(centres_raw)), "gibbs_stages_c")
        centres = centres_raw.transpose(0, 1).contiguous() @ Vt.T

    # Rotate into phi's singular directions, where the defensive component is diagonal.
    Kc = centres.shape[1]
    c_norm = -0.5 * centres.square().sum(2)
    cdf = torch.tensor([(1.0 - defensive) / Kc] * Kc + [defensive]).cumsum(0)
    cdf[-1] = 1.0
    half = 0.5 * r * math.log(2 * math.pi)
    h1, h2 = draws // 2, draws - draws // 2

    def stratified(n):
        u = (torch.arange(n, dtype=centres.dtype)[None, :] + torch.rand(B, n, generator=gen)) / float(n)
        return u.gather(1, torch.argsort(torch.rand(B, n, generator=gen), dim=1))

    pick = torch.cat([torch.searchsorted(cdf, stratified(h1)),
                      torch.searchsorted(cdf, stratified(h2))], dim=1).clamp(max=Kc)
    wide = pick == Kc                                                  # B x draws
    mean = torch.where(wide[:, :, None], torch.zeros(B, draws, r),
                       centres.gather(1, pick.clamp(max=Kc - 1)[:, :, None].expand(B, draws, r)))
    spread = torch.where(wide[:, :, None], sig[None, None, :], torch.ones(B, draws, r))
    z_rot = mean + spread * torch.randn(B, draws, r, generator=gen)
    z_all = (z_rot @ Vt).contiguous()
    z_norm = -0.5 * z_rot.square().sum(2)
    log_prior = z_norm - half
    log_comp = z_norm[:, :, None] + c_norm[:, None, :] + torch.bmm(z_rot, centres.transpose(1, 2).contiguous()) - half
    log_wide = -0.5 * (z_rot / sig[None, None, :]).square().sum(2) - half - log_det
    log_q = torch.logaddexp(math.log1p(-defensive) + torch.logsumexp(log_comp, 2) - math.log(Kc),
                            math.log(defensive) + log_wide)

    if os.environ.get("V4_LEGACY") == "1":
        ph_slot = ph[ix.item]
        b = model.slot_utilities(ix)
        u0 = b - 0.5 * ph_slot.square().sum(-1)
        iv = model.iv_shift(ix, b_slot=b)
        log_zz = torch.stack([log_partition(model, ix, u0 + (z_all[:, d][ix.item_trip] * ph_slot).sum(-1),
                                            z=z_all[:, d], iv=iv) for d in range(draws)], 1)
    else:
        b_slot, iv, ph_slot, half_phi_sq = _cached_trip_statics(model, ix)
        u0 = (b_slot - half_phi_sq).contiguous()
        a = _arrays(model, iv=iv, psi=model.psi_cat)
        K_max = _max_count(model, ix)
        min_cap = _min_cap(model)
        log_zz = torch.empty((B, draws), dtype=torch.float64)
        check(sampler().log_partition_batch_z_cpp(
            B, int(draws), ix.n_rows, len(ix.item), ix.Cpad, K_max, int(model.nmax), r,
            int(model.size_on_categories), min_cap, 1.0,
            ptr(u0), ptr(ph_slot), ptr(z_all), ptr(ix.row_of), ptr(ix.item_pos),
            ptr(ix.row_size), ptr(ix.row_cat), ptr(ix.row_trip), ptr(ix.flat_slot),
            ptr(a["rho_c"]), ptr(a["rho_0"]), ptr(a["f_extra"]), ptr(a["psi"]),
            ptr(a["iv"]), ptr(log_zz)), "log_partition_batch_z")

    log_w = log_prior + log_zz - log_q
    est = torch.logsumexp(log_w, 1) - math.log(draws)
    diff = (torch.logsumexp(log_w[:, :h1], 1) - math.log(h1)) - (torch.logsumexp(log_w[:, h1:], 1) - math.log(h2))
    ess = 1.0 / torch.softmax(log_w, 1).square().sum(1) / draws
    return est, diff, ess, X_out
