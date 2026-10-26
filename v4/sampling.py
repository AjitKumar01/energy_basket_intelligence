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

import torch

from . import exact
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
        exact.size_mode(model), seed,
        ptr(a["u"]), ptr(ix.row_of), ptr(ix.item_pos), ptr(ix.row_size), ptr(ix.row_cat),
        ptr(ix.flat_slot), ptr(a["rho_c"]), ptr(a["rho_0"]), ptr(a["f_extra"]),
        ptr(a["shift"]), ptr(a["iv"]), ptr(out)), "sampler")
    return out.bool()


@torch.no_grad()
def gibbs(model, ix, X, gen, *, sweeps=1, beta=1.0, iv=None):
    """``sweeps`` block-Gibbs sweeps from slot mask X: z ~ N(s(X), I), then X ~ S | z."""
    b = model.slot_utilities(ix).detach()
    iv = model.iv_shift(ix, b_slot=b) if iv is None else iv
    ph = model.phi_r()[ix.item].detach().contiguous()
    b_base = (b - 0.5 * beta * ph.square().sum(-1)).contiguous()
    a = _arrays(model, iv=iv, psi=model.psi_cat)
    seed = _seed(gen)
    mask = X.to(torch.uint8).contiguous().clone()
    check(sampler().gibbs_sweeps_cpp(
        ix.B, ix.n_rows, len(ix.item), ix.Cpad, _max_count(model, ix), model.nmax, model.rank,
        int(sweeps), exact.size_mode(model), math.sqrt(float(beta)), seed,
        ptr(b_base), ptr(ph), ptr(ix.item_trip), ptr(ix.row_of), ptr(ix.item_pos),
        ptr(ix.row_size), ptr(ix.row_cat), ptr(ix.row_trip), ptr(ix.flat_slot),
        ptr(a["rho_c"]), ptr(a["rho_0"]), ptr(a["f_extra"]), ptr(a["psi"]), ptr(a["iv"]),
        ptr(mask)), "Gibbs")
    return mask.bool()


@torch.no_grad()
def draw_from_model(model, ix, gen, burn):
    """A basket per trip from the full model: start from the law without pairwise terms
    (beta = 0), then ``burn`` Gibbs sweeps."""
    X = sample_baskets(model, ix, model.slot_utilities_given_z(ix, None, beta=0.0), gen, beta=0.0)
    return gibbs(model, ix, X, gen, sweeps=burn)


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

    # Chain 0, from beta = 0; its state after components*thin sweeps is returned.
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

    # Chains 1-3: different starting points, then annealed to beta = 1.
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

    # Rotate into phi's singular directions, where the defensive component is diagonal.
    centres = torch.stack(centres, 1) @ Vt.T                        # B x Kc x r
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
    z_all = z_rot @ Vt
    z_norm = -0.5 * z_rot.square().sum(2)
    log_prior = z_norm - half
    log_comp = z_norm[:, :, None] + c_norm[:, None, :] + torch.bmm(z_rot, centres.transpose(1, 2).contiguous()) - half
    log_wide = -0.5 * (z_rot / sig[None, None, :]).square().sum(2) - half - log_det
    log_q = torch.logaddexp(math.log1p(-defensive) + torch.logsumexp(log_comp, 2) - math.log(Kc),
                            math.log(defensive) + log_wide)

    ph_slot = ph[ix.item]
    b = model.slot_utilities(ix)
    u0 = b - 0.5 * ph_slot.square().sum(-1)
    iv = model.iv_shift(ix, b_slot=b)
    log_zz = [log_partition(model, ix, u0 + (z_all[:, d][ix.item_trip] * ph_slot).sum(-1),
                            z=z_all[:, d], iv=iv) for d in range(draws)]
    log_w = log_prior + torch.stack(log_zz, 1) - log_q
    est = torch.logsumexp(log_w, 1) - math.log(draws)
    diff = (torch.logsumexp(log_w[:, :h1], 1) - math.log(h1)) - (torch.logsumexp(log_w[:, h1:], 1) - math.log(h2))
    ess = 1.0 / torch.softmax(log_w, 1).square().sum(1) / draws
    return est, diff, ess, X_out
