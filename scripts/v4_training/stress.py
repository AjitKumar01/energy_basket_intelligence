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
import pathlib
import time
from pathlib import Path

sys.modules.setdefault("pathlib._local", pathlib)
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
OTHER_BASE_NAMES = ("alpha", "theta", "w_dsp", "w_mlr", "mu", "delta", "zeta", "xi")
USE_CPP_SAMPLER = os.environ.get("STRESS_CPP_SAMPLER", "0") == "1"
# Run as a script, this module is __main__; helpers that `import stress` must get this same module
# (and its USE_CPP_SAMPLER), not a second copy with the pure-Python sampler.
sys.modules.setdefault("stress", sys.modules[__name__])


# ---- setup ------------------------------------------------------------------------------------
class PurchaseHistory:
    """Product-level purchase history strictly before each trip's day, from the observed baskets.

    Three features per (household, product, day), all from earlier trips of the same household:
      ever    the household bought this product before
      last    this product was in the household's most recent purchase of its category
      decay   exp(-days since the household last bought this product / 28)
    Keys pack (group, day) as group * 1024 + day, so a strictly-before lookup is one searchsorted,
    as in Features.recency.  They are fixed data: they shift b linearly and leave the normaliser exact.
    """

    STRIDE = 1024

    def __init__(self, data):
        J, C = int(data["n_item"]), int(data["n_cat"])
        line_trip = np.repeat(np.arange(len(data["trip_user"])), np.diff(data["line_ptr"]))
        user = data["trip_user"][line_trip].astype(np.int64)
        day = data["trip_day"][line_trip].astype(np.int64)
        if int(day.max()) >= self.STRIDE:
            raise RuntimeError("trip days exceed the purchase-history key stride")
        item = data["line_item"].astype(np.int64)
        cat = data["line_cat"].astype(np.int64)
        self.J, self.C = J, C
        self.item_keys = torch.from_numpy(np.unique((user * J + item) * self.STRIDE + day))
        self.cat_keys = torch.from_numpy(np.unique((user * C + cat) * self.STRIDE + day))

    @staticmethod
    def _previous(keys, group, day, stride):
        idx = torch.searchsorted(keys, group * stride + day)
        prev = keys[(idx - 1).clamp(min=0)]
        return (idx > 0) & (torch.div(prev, stride, rounding_mode="floor") == group), prev % stride

    def features(self, item, cat, user, day):
        S = self.STRIDE
        item, cat, user, day = item.long(), cat.long(), user.long(), day.long()
        g_item = user * self.J + item
        seen, last_item_day = self._previous(self.item_keys, g_item, day, S)
        decay = torch.where(seen, torch.exp(-(day - last_item_day).double() / 28.0),
                            torch.zeros((), dtype=torch.float64))
        cat_seen, last_cat_day = self._previous(self.cat_keys, user * self.C + cat, day, S)
        q = g_item * S + last_cat_day
        hit = self.item_keys[torch.searchsorted(self.item_keys, q).clamp(max=len(self.item_keys) - 1)] == q
        return torch.stack([seen.double(), (cat_seen & hit).double(), decay], dim=-1)


def attach_purchase_history(batcher, history, cat_of):
    """Add ctx['rec_item'] at every assortment slot and every purchased line of each batch."""
    make = batcher.make

    def make_with_history(trips):
        out = make(trips)
        ix, ctx, lctx, _, LI, LT = out[:6]
        user = torch.as_tensor(batcher.D["trip_user"][trips], dtype=torch.long)
        day = torch.as_tensor(batcher.D["trip_day"][trips], dtype=torch.long)
        ctx["rec_item"] = history.features(ix.item, cat_of[ix.item], user[ix.item_trip], day[ix.item_trip])
        lctx["rec_item"] = history.features(LI, cat_of[LI], user[LT], day[LT])
        return out

    batcher.make = make_with_history


def setup(initialization, additive, threads, recency=False, iv_slope=False):
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
    model.presence_phi = False
    model.cat_rho0 = False
    model.has_iv_slope = False
    model.iv_price_only = False
    orig_b_at = model.b_at

    def b_at_with_u_cat(it, trip, c):
        b = orig_b_at(it, trip, c)
        if getattr(model, "u_cat", None) is not None:
            hh = model.house[trip]
            theta = model.theta_c()
            b = b + (theta[hh] * model.u_cat[model.cat_of[it]]).sum(-1)
        if c is not None and "rec_item" in c and getattr(model, "psi_item", None) is not None:
            b = b + (model.psi_item[model.cat_of[it]] * c["rec_item"]).sum(-1)
        return b

    model.b_at = b_at_with_u_cat
    features = Features(int(data["n_item"]), int(data["n_store"]), include_recency=recency,
                        include_store_price=True)
    batcher = Batcher(data, features, int(meta["nmax"]), include_recency=recency)
    model.recency = bool(recency)
    if recency:
        attach_purchase_history(batcher, PurchaseHistory(data), model.cat_of.long())
    return data, model, meta, batcher


def checkpoint_uses_recency(path):
    return bool(torch.load(path, map_location="cpu", weights_only=False).get("recency", False))


def configure_model_from_blob(model, blob):
    """Attach optional M1 (f_extra), M2 (psi_cat, presence_phi), M3 (cat_rho0), and M4a (u_cat) parameters from checkpoint blob and load state."""
    st = blob["model"]
    if bool(blob.get("recency", False)) != bool(getattr(model, "recency", False)):
        raise RuntimeError("the checkpoint and setup() disagree on recency features; "
                           "call setup(..., recency=checkpoint_uses_recency(path))")
    if "psi_item" in st and getattr(model, "psi_item", None) is None:
        model.psi_item = torch.nn.Parameter(torch.zeros_like(st["psi_item"]))
    if "f_extra" in st and st["f_extra"] is not None:
        if getattr(model, "f_extra", None) is None:
            model.f_extra = torch.nn.Parameter(torch.zeros_like(st["f_extra"]))
    if "psi_cat" in st and st["psi_cat"] is not None:
        model.presence_phi = bool(blob.get("presence_phi", True))
        if getattr(model, "psi_cat", None) is None:
            model.psi_cat = torch.nn.Parameter(torch.zeros_like(st["psi_cat"]))
    else:
        model.presence_phi = False
    if "u_cat" in st and st["u_cat"] is not None:
        if getattr(model, "u_cat", None) is None:
            model.u_cat = torch.nn.Parameter(torch.zeros_like(st["u_cat"]))
    if "iv_slope_raw" in st and st["iv_slope_raw"] is not None:
        model.has_iv_slope = bool(blob.get("iv_slope", True))
        if getattr(model, "iv_slope_raw", None) is None:
            model.iv_slope_raw = torch.nn.Parameter(torch.zeros_like(st["iv_slope_raw"]))
    else:
        model.has_iv_slope = False
    model.iv_price_only = bool(blob.get("iv_price_only", False))
    model.cat_rho0 = bool(blob.get("cat_rho0", False))
    model.load_state_dict(st)


class _LogDegree2DProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, log_coefficients, degree, nmax):
        from build_stress_sampler import load_stress_sampler
        lc = log_coefficients.contiguous()
        deg = degree.contiguous()
        B, Cpad, stride = lc.shape
        nmax_i = int(nmax)
        log_out = torch.empty((B, nmax_i + 1, nmax_i + 1), dtype=lc.dtype, device=lc.device)
        q_all = torch.empty_like(lc)
        states = torch.empty((B, Cpad + 1, nmax_i + 1, nmax_i + 1), dtype=lc.dtype, device=lc.device)
        rc = load_stress_sampler().log_degree_2d_forward_cpp(
            B, Cpad, stride, nmax_i,
            lc.data_ptr(), deg.data_ptr(),
            log_out.data_ptr(), q_all.data_ptr(), states.data_ptr())
        if rc != 0:
            raise RuntimeError(f"native log_degree_2d_forward_cpp failed with status {rc}")
        ctx.save_for_backward(q_all, deg, states)
        ctx.nmax = nmax_i
        return log_out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        from build_stress_sampler import load_stress_sampler
        q_all, deg, states = ctx.saved_tensors
        B, Cpad, stride = q_all.shape
        grad_c = grad_output.contiguous()
        grad_in = torch.empty_like(q_all)
        rc = load_stress_sampler().log_degree_2d_backward_cpp(
            B, Cpad, stride, ctx.nmax,
            grad_c.data_ptr(), q_all.data_ptr(), deg.data_ptr(),
            states.data_ptr(), grad_in.data_ptr())
        if rc != 0:
            raise RuntimeError(f"native log_degree_2d_backward_cpp failed with status {rc}")
        return grad_in, None, None


def log_poly_2d_native(log_coefficients, degree, nmax):
    if not torch.is_grad_enabled() or not log_coefficients.requires_grad:
        from build_stress_sampler import load_stress_sampler
        lc = log_coefficients.contiguous()
        deg = degree.contiguous()
        B, Cpad, stride = lc.shape
        nmax_i = int(nmax)
        log_out = torch.empty((B, nmax_i + 1, nmax_i + 1), dtype=lc.dtype, device=lc.device)
        rc = load_stress_sampler().log_degree_2d_forward_cpp(
            B, Cpad, stride, nmax_i,
            lc.data_ptr(), deg.data_ptr(),
            log_out.data_ptr(), 0, 0)
        if rc != 0:
            raise RuntimeError(f"native log_degree_2d_forward_cpp failed with status {rc}")
        return log_out
    return _LogDegree2DProduct.apply(log_coefficients, degree, int(nmax))


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


def row_cat_shift(model, ix, z, beta=1.0):
    """Per-row category presence shift a_row(z) = sqrt(beta) z_b.psi_c - 0.5 beta |psi_c|^2 for M2."""
    if not getattr(model, "presence_phi", False) or beta == 0.0:
        return None
    psi_r = model.psi_cat[ix.row_cat]
    shift = -0.5 * beta * psi_r.square().sum(-1)
    if z is not None:
        shift = shift + math.sqrt(beta) * (z[ix.row_trip] * psi_r).sum(-1)
    return shift


def iv_slope_lambda(model):
    if getattr(model, "iv_slope_override", None) is not None:
        return model.iv_slope_override
    if getattr(model, "has_iv_slope", False) and getattr(model, "iv_slope_raw", None) is not None:
        return torch.sigmoid(model.iv_slope_raw)
    return None


PRICE_KEYS = ("dlp", "dlp_bar", "disp", "mail")


def row_logsumexp(ix, b_slot):
    from ragged import seg_max, seg_sum
    m_r = seg_max(b_slot.unsqueeze(0), ix.row_of, ix.n_rows)[0].detach()
    w_s = torch.exp(b_slot - m_r[ix.row_of])
    return m_r + torch.log(seg_sum(w_s.unsqueeze(0), ix.row_of, ix.n_rows)[0].clamp_min(1e-300))


def row_iv_shift(model, ix, b_slot=None):
    """Per-row category inclusive-value presence shift (lambda_c - 1) * IV_{tc}(b) (M4b), or with
    model.iv_price_only (M4c) (lambda_c - 1) * [IV_{tc}(b) - IV_{tc}(b at regular price)], the
    regular price being no price deviation, display or mailer.  Either way a function of the
    context only, so the normaliser stays exact."""
    lam_c = iv_slope_lambda(model)
    if lam_c is None:
        return None
    if b_slot is None:
        b_slot = model.b_flat(ix)
    iv_r = row_logsumexp(ix, b_slot)
    if getattr(model, "iv_price_only", False):
        regular = {k: (torch.zeros_like(v) if k in PRICE_KEYS else v) for k, v in model.ctx.items()}
        iv_r = iv_r - row_logsumexp(ix, model.b_at(ix.item, ix.item_trip, regular))
    return (lam_c[ix.row_cat] - 1.0) * iv_r


def basket_field(model, ix, X, r):
    """s = sum of phi over the basket (or M2 presence-based coupling), B x r."""
    x = X.double()
    ph = phi_r(model, r)[ix.item]
    s = torch.zeros(ix.B, r).index_add_(0, ix.item_trip, ph * x[:, None])
    if getattr(model, "presence_phi", False):
        nc_row = torch.zeros(ix.n_rows).index_add_(0, ix.row_of, x)
        extra_row = (nc_row - 1.0).clamp_min(0.0)
        s = s - torch.zeros(ix.B, r).index_add_(0, ix.row_trip, model.psi_cat[ix.row_cat] * extra_row[:, None])
    return s


def obs_energy(model, bt, r):
    """Exact energy of observed baskets in batch bt, including optional M1 (f_extra), M2 (presence_phi), M3 (cat_rho0), and M4b (iv_slope)."""
    e = model.energy(bt.li, bt.lt, bt.lc, bt.ix.B, bt.line_ctx)
    has_f = getattr(model, "f_extra", None) is not None
    has_m2 = getattr(model, "presence_phi", False)
    has_m3 = getattr(model, "cat_rho0", False)
    iv_sh = row_iv_shift(model, bt.ix)
    has_m4b = iv_sh is not None
    if not has_f and not has_m2 and not has_m3 and not has_m4b:
        return e
    ix = bt.ix
    nc_row = torch.zeros(ix.n_rows, dtype=e.dtype).index_add_(0, ix.row_of, bt.obs.to(e.dtype))
    if has_f:
        fe = model.f_extra[ix.row_cat]
        pen_row = ((nc_row == 2).to(e.dtype) * fe[:, 0]
                   + (nc_row == 3).to(e.dtype) * fe[:, 1]
                   + (nc_row >= 4).to(e.dtype) * fe[:, 2])
        e = e - torch.zeros(ix.B, dtype=e.dtype).index_add_(0, ix.row_trip, pen_row)
    if has_m2:
        ph = phi_r(model, r)
        v_lin = torch.zeros(ix.B, r, dtype=e.dtype).index_add_(0, bt.lt, ph[bt.li])
        extra_row = (nc_row - 1.0).clamp_min(0.0)
        psi_row = model.psi_cat[ix.row_cat]
        shift_b = torch.zeros(ix.B, r, dtype=e.dtype).index_add_(0, ix.row_trip, psi_row * extra_row[:, None])
        s_m2 = v_lin - shift_b
        diag_corr = torch.zeros(ix.B, dtype=e.dtype).index_add_(0, ix.row_trip, extra_row * psi_row.square().sum(-1))
        e = e + 0.5 * (s_m2.square().sum(-1) - v_lin.square().sum(-1) + diag_corr)
    if has_m3:
        n_b = torch.zeros(ix.B, dtype=torch.long).index_add_(0, bt.lt, torch.ones_like(bt.li))
        k_b = torch.zeros(ix.B, dtype=torch.long).index_add_(0, ix.row_trip, (nc_row > 0).long())
        rho0 = model.rho_0()
        e = e + rho0[n_b] - rho0[k_b]
    if has_m4b:
        pres_row = (nc_row > 0).to(e.dtype)
        e = e + torch.zeros(ix.B, dtype=e.dtype).index_add_(0, ix.row_trip, pres_row * iv_sh)
    return e


def log_zc(model, ix, u, z=None, beta=1.0, b_slot=None, iv_shift=None):
    has_f = getattr(model, "f_extra", None) is not None
    has_m2 = getattr(model, "presence_phi", False) and beta != 0.0
    has_m3 = getattr(model, "cat_rho0", False)
    iv_sh = row_iv_shift(model, ix, b_slot=b_slot) if iv_shift is None else iv_shift
    has_m4b = iv_sh is not None
    if not has_f and not has_m2 and not has_m3 and not has_m4b:
        from interaction_particles import differentiable_logz_beta0
        return differentiable_logz_beta0(model, ix, u)
    from poly_degree_native import log_poly_tree_degree_native
    from ragged import esp_log_bucketed, seg_max
    scale = seg_max(u.unsqueeze(0), ix.item_trip, ix.B)[0]
    centred = u - scale[ix.item_trip]
    log_e = esp_log_bucketed(
        centred.unsqueeze(0), ix.row_of, ix.n_rows, model.R,
        ix.row_size, ix.item_pos)[0]
    degree_axis = torch.arange(model.R + 1, dtype=u.dtype, device=u.device)
    log_g = (log_e - model.rho_c[ix.row_cat].unsqueeze(-1)
             * model.pair_feature(degree_axis))
    if has_f:
        fe = model.f_extra[ix.row_cat]
        fe_pad = torch.zeros(ix.n_rows, model.R + 1, dtype=u.dtype, device=u.device)
        if model.R >= 2:
            fe_pad[:, 2] = fe[:, 0]
        if model.R >= 3:
            fe_pad[:, 3] = fe[:, 1]
        if model.R >= 4:
            fe_pad[:, 4:] = fe[:, 2:3]
        log_g = log_g - fe_pad
    if has_m2:
        cshift = row_cat_shift(model, ix, z, beta=beta)
        if cshift is not None:
            km1 = (degree_axis - 1.0).clamp_min(0.0)
            log_g = log_g - cshift[:, None] * km1[None, :]
    if has_m4b:
        log_g = torch.cat([log_g[:, :1], log_g[:, 1:] + iv_sh[:, None]], dim=1)
    category = torch.full((1, ix.B * ix.Cpad, model.R + 1), -float("inf"),
                          dtype=u.dtype, device=u.device)
    category[..., 0] = 0.0
    category = category.index_copy(1, ix.flat_slot, log_g.unsqueeze(0)).view(
        1, ix.B, ix.Cpad, model.R + 1)
    degrees = torch.zeros(ix.B * ix.Cpad, dtype=torch.long, device=u.device)
    degrees[ix.flat_slot] = ix.row_size.clamp(max=model.R)
    degrees = degrees.view(ix.B, ix.Cpad)
    slope = (category[..., 1:] / degree_axis[1:]).amax(
        dim=(2, 3)).clamp_min(0.0)
    tilted = category - slope[:, :, None, None] * degree_axis
    if has_m3:
        log_a2d = log_poly_2d_native(tilted[0].contiguous(), degrees.contiguous(), model.nmax)
        size_axis = torch.arange(model.nmax + 1, dtype=u.dtype, device=u.device)
        log_a2d = log_a2d + slope[0, :, None, None] * size_axis[None, :, None]
        log_joint = (
            log_a2d[:, 1:, 1:]
            + size_axis[None, 1:, None] * scale[:, None, None]
            - model.rho_0()[None, None, 1:model.nmax + 1]
        )
        return torch.logsumexp(log_joint.reshape(ix.B, -1), dim=-1)
    log_a = log_poly_tree_degree_native(
        tilted.contiguous(), degrees.contiguous(), model.nmax)
    log_a = log_a + slope.unsqueeze(-1) * degree_axis
    size_axis = torch.arange(log_a.shape[-1], dtype=u.dtype, device=u.device)
    log_size = (log_a[0] + size_axis * scale.unsqueeze(-1)
                - model.rho_0()[:log_a.shape[-1]])[:, 1:]
    return torch.logsumexp(log_size, dim=-1)


# ---- exact vectorized sampler of S | u -------------------------------------------------------
@torch.no_grad()
def sample_slots(model, ix, u, gen, use_cpp: bool | None = None, z=None, beta=1.0, iv_shift=None):
    """One exact draw per trip from the additive Version-4 law at slot utilities u (category
    penalty rho_c + f_extra, M2 category presence shift, M4b IV shift, size/category potential rho_0, 1 <= |S| <= nmax)."""
    B, nmax = ix.B, model.nmax
    L = int(ix.row_size.max())
    K = min(model.R, L, nmax)
    cshift = row_cat_shift(model, ix, z, beta=beta)
    iv_sh = row_iv_shift(model, ix) if iv_shift is None else iv_shift
    has_m3 = bool(getattr(model, "cat_rho0", False))
    if USE_CPP_SAMPLER if use_cpp is None else use_cpp:
        from build_stress_sampler import load_stress_sampler
        T = len(ix.item)
        u_c = u.detach().contiguous()
        rho_c = model.rho_c.detach().contiguous()
        rho_0 = model.rho_0().detach().contiguous()
        fe_c = model.f_extra.detach().contiguous() if getattr(model, "f_extra", None) is not None else None
        cs_c = cshift.detach().contiguous() if cshift is not None else None
        iv_c = iv_sh.detach().contiguous() if iv_sh is not None else None
        fe_ptr = fe_c.data_ptr() if fe_c is not None else 0
        cs_ptr = cs_c.data_ptr() if cs_c is not None else 0
        iv_ptr = iv_c.data_ptr() if iv_c is not None else 0
        seed = int(torch.randint(1, 2**62, (1,), generator=gen).item())
        out = torch.empty(T, dtype=torch.uint8)
        rc = load_stress_sampler().sample_slots_cpp(
            B, ix.n_rows, T, ix.Cpad, K, nmax, int(has_m3), seed,
            u_c.data_ptr(), ix.row_of.data_ptr(), ix.item_pos.data_ptr(),
            ix.row_size.data_ptr(), ix.row_cat.data_ptr(), ix.flat_slot.data_ptr(),
            rho_c.data_ptr(), rho_0.data_ptr(), fe_ptr, cs_ptr, iv_ptr, out.data_ptr())
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
    if getattr(model, "f_extra", None) is not None:
        fe = model.f_extra[ix.row_cat]
        fe_pad = torch.zeros(ix.n_rows, K + 1, dtype=u.dtype)
        if K >= 2:
            fe_pad[:, 2] = fe[:, 0]
        if K >= 3:
            fe_pad[:, 3] = fe[:, 1]
        if K >= 4:
            fe_pad[:, 4:] = fe[:, 2:3]
        g = g - fe_pad
    if cshift is not None:
        km1 = (k_axis - 1.0).clamp_min(0.0)
        g = g - cshift[:, None] * km1[None, :]
    if iv_sh is not None:
        g[:, 1:] = g[:, 1:] + iv_sh[:, None]
    G = torch.full((B * ix.Cpad, K + 1), NEG)
    G[:, 0] = 0.0
    G[ix.flat_slot] = g
    G = G.view(B, ix.Cpad, K + 1)
    take_c = torch.zeros(B, ix.Cpad, dtype=torch.long)
    if has_m3:
        f2d = torch.full((B, nmax + 1, nmax + 1), NEG, dtype=u.dtype)
        f2d[:, 0, 0] = 0.0
        Fs2d = [f2d]
        for c in range(ix.Cpad):
            cur = f2d + G[:, c, 0:1, None]
            for k in range(1, K + 1):
                cur[:, k:, 1:] = torch.logaddexp(cur[:, k:, 1:], f2d[:, :nmax + 1 - k, :nmax] + G[:, c, k:k + 1, None])
            f2d = cur
            Fs2d.append(f2d)
        n_axis = torch.arange(1, nmax + 1, dtype=u.dtype)
        log_joint = (
            f2d[:, 1:, 1:]
            + n_axis[None, :, None] * scale[:, None, None]
            - model.rho_0()[None, None, 1:nmax + 1]
        )
        flat_idx = torch.multinomial(torch.softmax(log_joint.reshape(B, -1), 1), 1, generator=gen).squeeze(1)
        rem_n = (flat_idx // nmax) + 1
        rem_m = (flat_idx % nmax) + 1
        b_idx = torch.arange(B)
        for c in range(ix.Cpad - 1, -1, -1):
            f_prev = Fs2d[c]
            lp = torch.full((B, K + 1), NEG, dtype=u.dtype)
            lp[:, 0] = f_prev[b_idx, rem_n, rem_m] + G[:, c, 0]
            for k in range(1, K + 1):
                ok = (rem_n >= k) & (rem_m >= 1)
                rn = (rem_n - k).clamp(min=0)
                rm = (rem_m - 1).clamp(min=0)
                lp[:, k] = torch.where(ok, f_prev[b_idx, rn, rm] + G[:, c, k], torch.full((B,), NEG, dtype=u.dtype))
            k = torch.multinomial(torch.softmax(lp, 1), 1, generator=gen).squeeze(1)
            take_c[:, c] = k
            rem_n = rem_n - k
            rem_m = rem_m - (k > 0).long()
        if bool((rem_n != 0).any()) or bool((rem_m != 0).any()):
            raise RuntimeError("2D category backtrack left slots unfilled")
    else:
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
def gibbs(model, ix, X, r, gen, sweeps=1, beta=1.0, use_cpp: bool | None = None, iv_shift=None):
    b_slot = model.b_flat(ix).detach()
    iv_sh = row_iv_shift(model, ix, b_slot=b_slot) if iv_shift is None else iv_shift
    if USE_CPP_SAMPLER if use_cpp is None else use_cpp:
        from build_stress_sampler import load_stress_sampler
        B, nmax, T = ix.B, model.nmax, len(ix.item)
        L = int(ix.row_size.max())
        K = min(model.R, L, nmax)
        ph = phi_r(model, r)[ix.item].detach().contiguous()
        b_base = (b_slot - 0.5 * beta * ph.square().sum(-1)).contiguous()
        rho_c = model.rho_c.detach().contiguous()
        rho_0 = model.rho_0().detach().contiguous()
        fe_c = model.f_extra.detach().contiguous() if getattr(model, "f_extra", None) is not None else None
        psi_c = model.psi_cat.detach().contiguous() if getattr(model, "presence_phi", False) else None
        iv_c = iv_sh.detach().contiguous() if iv_sh is not None else None
        fe_ptr = fe_c.data_ptr() if fe_c is not None else 0
        psi_ptr = psi_c.data_ptr() if psi_c is not None else 0
        iv_ptr = iv_c.data_ptr() if iv_c is not None else 0
        has_m3 = bool(getattr(model, "cat_rho0", False))
        seed = int(torch.randint(1, 2**62, (1,), generator=gen).item())
        mask = X.to(torch.uint8).contiguous().clone()
        rc = load_stress_sampler().gibbs_sweeps_cpp(
            B, ix.n_rows, T, ix.Cpad, K, nmax, r, int(sweeps), int(has_m3), math.sqrt(float(beta)), seed,
            b_base.data_ptr(), ph.data_ptr(), ix.item_trip.data_ptr(), ix.row_of.data_ptr(),
            ix.item_pos.data_ptr(), ix.row_size.data_ptr(), ix.row_cat.data_ptr(),
            ix.row_trip.data_ptr(), ix.flat_slot.data_ptr(), rho_c.data_ptr(),
            rho_0.data_ptr(), fe_ptr, psi_ptr, iv_ptr, mask.data_ptr())
        if rc != 0:
            raise RuntimeError(f"native gibbs_sweeps_cpp failed with status {rc}")
        return mask.bool()
    for _ in range(sweeps):
        z = basket_field(model, ix, X, r) + torch.randn(ix.B, r, generator=gen)
        X = sample_slots(model, ix, slot_utilities(model, ix, z, r, beta), gen, use_cpp=False, z=z, beta=beta, iv_shift=iv_sh)
    return X


# ---- objectives -------------------------------------------------------------------------------
def pseudo_loglik(model, ix, X, r):
    """Mean over trips of sum_j log P(x_j | rest) (exact conditionals including M1/M2/M3)."""
    x = X.double()
    ph = phi_r(model, r)[ix.item]
    n_row = torch.zeros(ix.n_rows).index_add(0, ix.row_of, x)
    nc_wo = n_row[ix.row_of] - x
    slot_cat = ix.row_cat[ix.row_of]
    if getattr(model, "presence_phi", False):
        s = basket_field(model, ix, X, r)
        psi_slot = model.psi_cat[slot_cat]
        dlt_slot = ph - psi_slot
        has_other = (nc_wo >= 1.0).double()
        eff_vec = torch.where(has_other[:, None] > 0.5, dlt_slot, ph)
        self_sub = torch.where(has_other > 0.5, dlt_slot.square().sum(-1), ph.square().sum(-1))
        s_wo = s[ix.item_trip] - x[:, None] * eff_vec
        pair = (eff_vec * s_wo).sum(-1) - has_other * (psi_slot * dlt_slot).sum(-1)
    else:
        s = torch.zeros(ix.B, r).index_add(0, ix.item_trip, ph * x[:, None])
        pair = (ph * s[ix.item_trip]).sum(-1) - x * ph.square().sum(-1)
    pf = model.pair_feature
    dcat = model.rho_c[slot_cat] * (pf(nc_wo + 1) - pf(nc_wo))
    if getattr(model, "f_extra", None) is not None:
        fe_slot = model.f_extra[slot_cat]
        dfe = (
            (nc_wo == 1).double() * fe_slot[:, 0]
            + (nc_wo == 2).double() * (fe_slot[:, 1] - fe_slot[:, 0])
            + (nc_wo == 3).double() * (fe_slot[:, 2] - fe_slot[:, 1])
        )
        dcat = dcat + dfe
    n_trip = torch.zeros(ix.B).index_add(0, ix.item_trip, x)
    n_wo = (n_trip[ix.item_trip] - x).long()
    rho0 = model.rho_0()
    new_cat = (nc_wo == 0)
    if getattr(model, "cat_rho0", False):
        k_trip = torch.zeros(ix.B, dtype=torch.long).index_add(0, ix.row_trip, (n_row > 0).long())
        k_wo = (k_trip[ix.item_trip] - X.long() * new_cat.long()).clamp(min=0)
        dsize = new_cat.double() * (rho0[(k_wo + 1).clamp(max=model.nmax)] - rho0[k_wo])
    else:
        dsize = rho0[(n_wo + 1).clamp(max=model.nmax)] - rho0[n_wo]
    b_slot = model.b_flat(ix)
    d = b_slot + pair - dcat - dsize
    iv_sh = row_iv_shift(model, ix, b_slot=b_slot)
    if iv_sh is not None:
        d = d + new_cat.double() * iv_sh[ix.row_of]
    valid = (n_wo > 0) & (n_wo < model.nmax)
    lp = F.logsigmoid(torch.where(X, d, -d))
    return (lp * valid).sum() / ix.B


def penalties(model, args, n_train, r, tied=None, price_init=None):
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
    extra_prior = 0.0
    if getattr(model, "f_extra", None) is not None:
        extra_prior = args.f_extra_prec * model.f_extra.square().sum()
    price_prior = 0.0
    if price_init is not None:
        for n, v0 in price_init.items():
            p = getattr(model, n)
            if p.requires_grad:
                price_prior = price_prior + args.price_prec * (p - v0).square().sum()
    u_cat_prior = 0.0
    if getattr(model, "u_cat", None) is not None:
        u_cat_prior = args.u_cat_prec * model.u_cat.square().sum()
    rec_prior = 0.0
    if getattr(args, "recency", False):
        rec_prior = args.rec_prec * (model.psi.square().sum() + model.psi_item.square().sum())
    loss = loss + 0.5 * (phi_prior + args.rho_c_prec * model.rho_c.square().sum() + extra_prior + price_prior
                         + u_cat_prior + rec_prior) / n_train
    return loss


def project(model, caps, optimizer, r):
    from category_safety import project_category_reward_
    model.project_context_gauges()
    with torch.no_grad():
        if getattr(model, "cat_rho0", False):
            model.rho_0_free.sub_(model.rho_0_free[0].clone())
        else:
            mean_lam = model.lam.mean().clone()
            model.lam.sub_(mean_lam)
            model.rho_0_free.sub_(mean_lam * torch.arange(1, model.rho_0_free.numel() + 1))
        model.phi[:, r:] = 0.0
    model.project_rho_c(-1.5)
    project_category_reward_(model, caps, 1.5, optimizer=optimizer)
    if getattr(model, "f_extra", None) is not None:
        with torch.no_grad():
            mult = torch.tensor([1.0, 3.0, 6.0], dtype=model.rho_c.dtype, device=model.rho_c.device)
            floor_fe = -1.5 - model.rho_c[:, None] * mult[None, :]
            model.f_extra.copy_(torch.maximum(model.f_extra, floor_fe))


# ---- train ------------------------------------------------------------------------------------
def train(args):
    from category_safety import category_capacities
    from pipeline_support import supported_trips
    data, model, meta, batcher = setup(args.initialization, args.additive, args.threads, recency=args.recency)
    J, C, nmax, r = int(data["n_item"]), int(data["n_cat"]), int(meta["nmax"]), args.rank
    all_train_trips = supported_trips(data, 0, nmax)
    rng = np.random.default_rng(args.seed)
    t0 = time.time()
    val_batches = []
    if args.val_trips > 0:
        v_perm = np.random.default_rng(args.seed + 99).permutation(all_train_trips)
        val_trips = np.sort(v_perm[:args.val_trips])
        train_trips = np.sort(v_perm[args.val_trips:])
        val_batches = [Batch(batcher, val_trips[s:s + args.batch], J) for s in range(0, len(val_trips), args.batch)]
    else:
        train_trips = all_train_trips
    order = rng.permutation(train_trips)
    batches = [Batch(batcher, np.sort(order[s:s + args.batch]), J) for s in range(0, len(order), args.batch)]
    log = open(args.out.with_suffix(".log"), "a")

    def emit(row):
        line = json.dumps(row)
        print(line, flush=True)
        log.write(line + "\n"); log.flush()

    emit({"event": "start", "train_trips": len(train_trips), "val_trips": len(val_batches) * args.batch if val_batches else 0,
          "batches": len(batches), "batching_seconds": round(time.time() - t0, 1),
          "args": {k: str(v) for k, v in vars(args).items()}})
    torch.manual_seed(args.seed)
    if args.f_extra:
        model.register_parameter("f_extra", torch.nn.Parameter(torch.zeros(C, 3)))
    if args.u_cat:
        model.register_parameter("u_cat", torch.nn.Parameter(torch.zeros(C, model.theta.shape[1])))
    if args.iv_slope:
        model.register_parameter("iv_slope_raw", torch.nn.Parameter(torch.full((C,), float(args.iv_slope_init))))
        model.has_iv_slope = True
        model.iv_price_only = bool(args.iv_price_only)
    if args.recency:
        # sub-commodity recency (psi, J x 4) and product purchase history (psi_item, C x 3), from zero
        model.register_parameter("psi_item", torch.nn.Parameter(torch.zeros(C, 3)))
        with torch.no_grad():
            model.psi.zero_()
    if args.presence_phi:
        args.tie_phi = True
        model.presence_phi = True
    if args.cat_rho0:
        model.cat_rho0 = True
        with torch.no_grad():
            s0 = (model.rho_0_free[9] - model.rho_0_free[0]) / 9.0
            model.lam.sub_(s0)
            model.rho_0_free.sub_(s0 * torch.arange(1, model.rho_0_free.numel() + 1, dtype=model.rho_0_free.dtype))
            model.rho_0_free.sub_(model.rho_0_free[0].clone())
    tied = None
    if args.tie_phi:
        cat = model.cat_of.long()
        psi_init = args.phi_init * torch.randn(C, r)
        if args.presence_phi:
            model.register_parameter("psi_cat", torch.nn.Parameter(psi_init))
            psi = model.psi_cat
        else:
            psi = psi_init.requires_grad_(True)
        dlt = torch.zeros(J, r, requires_grad=True)
        tied = (psi, dlt, cat)
        with torch.no_grad():
            model.phi.zero_()
            model.phi[:, :r] = psi[cat] + dlt
    else:
        with torch.no_grad():
            model.phi[:, :r] = args.phi_init * torch.randn(J, r)
    # with --recency, psi is trained in its own group (from stage A) instead of with the T1 price block
    price_names = ("gamma", "beta", "price_kappa") + (() if args.recency else ("psi",))
    price_init = {n: getattr(model, n).detach().clone() for n in price_names}
    for n in price_names:
        getattr(model, n).requires_grad_(False)
    rec_params = [model.psi, model.psi_item] if args.recency else []
    for q in rec_params:
        q.requires_grad_(True)
    base = [getattr(model, n) for n in BASE_NAMES]
    other_base = [getattr(model, n) for n in OTHER_BASE_NAMES]
    model.phi.requires_grad_(True)
    caps = category_capacities(data, C, nmax)
    N = len(train_trips)
    gen = torch.Generator().manual_seed(args.seed + 1)
    started = time.time()
    lr_lam = args.lr_base if args.lr_lam is None else args.lr_lam

    # Empirical training incidence for natural-gradient / moment-matching preconditioning of rho_0, rho_c, and lam
    with torch.no_grad():
        size_counts = torch.zeros(nmax + 1)
        cat_pair_sum = torch.zeros(C)
        item_counts = torch.zeros(J)
        obs_cat_pres = torch.zeros(C)
        for bt in batches:
            nc_row = torch.zeros(bt.ix.n_rows).index_add_(0, bt.ix.row_of, bt.obs.double())
            if args.cat_rho0:
                k_b = torch.zeros(bt.ix.B, dtype=torch.long).index_add_(0, bt.ix.row_trip, (nc_row > 0).long())
                size_counts.index_add_(0, k_b.clamp(max=nmax), torch.ones_like(k_b, dtype=torch.float64))
            else:
                n_b = torch.zeros(bt.ix.B, dtype=torch.long).index_add_(0, bt.lt, torch.ones_like(bt.li))
                size_counts.index_add_(0, n_b.clamp(max=nmax), torch.ones_like(n_b, dtype=torch.float64))
            cat_pair_sum.index_add_(0, bt.ix.row_cat, model.pair_feature(nc_row))
            obs_cat_pres.index_add_(0, bt.ix.row_cat, (nc_row > 0).double())
            item_counts.index_add_(0, bt.ix.item, bt.obs.double())
        p_size_prec = (size_counts[1:] / N).clamp_min(2e-3)
        d_cat_prec = (cat_pair_sum / N).clamp_min(5e-3)
        p_item_prec = (item_counts / N).clamp_min(2e-3)
        p_obs_cat_pres = obs_cat_pres / N

    @torch.no_grad()
    def eval_val_ll():
        if not val_batches:
            return None, None
        vgen = torch.Generator().manual_seed(args.seed + 777)
        lls, ess_list = [], []
        for vbt in val_batches:
            vbt.bind(model)
            est, _, ess, _ = is_log_z(model, vbt, r, vgen, burn=80, components=32, thin=2, draws=256)
            e = obs_energy(model, vbt, r)
            lls.append((e - est).numpy())
            ess_list.append(ess.numpy())
        return float(np.concatenate(lls).mean()), float(np.concatenate(ess_list).mean())

    def run_stage(name, epochs, lr_b, lr_l, lr_rho, lr_rho0, lr_phi, step_fn, average, mm_rho=0.0, mm_rho0=0.0, mm_lam=0.0,
                  monitor_val=False, lr_price=0.0, lr_u_cat=0.0, lr_rec=0.0, lr_iv_slope=0.0):
        param_groups = [
            {"params": other_base, "lr": lr_b},
            {"params": [model.lam], "lr": lr_l},
            {"params": [model.rho_c], "lr": lr_rho},
            {"params": [model.rho_0_free], "lr": lr_rho0},
            {"params": [model.phi] if tied is None else [tied[0], tied[1]], "lr": lr_phi},
        ]
        extra_params = []
        if getattr(model, "f_extra", None) is not None:
            param_groups.append({"params": [model.f_extra], "lr": 0.2 * lr_rho})
            extra_params.append(model.f_extra)
        if getattr(model, "u_cat", None) is not None and lr_u_cat > 0.0:
            param_groups.append({"params": [model.u_cat], "lr": lr_u_cat})
            extra_params.append(model.u_cat)
        if getattr(model, "iv_slope_raw", None) is not None and lr_iv_slope > 0.0:
            param_groups.append({"params": [model.iv_slope_raw], "lr": lr_iv_slope})
            extra_params.append(model.iv_slope_raw)
        if rec_params and lr_rec > 0.0:
            param_groups.append({"params": rec_params, "lr": lr_rec})
            extra_params.extend(rec_params)
        price_params = [getattr(model, n) for n in price_names if getattr(model, n).requires_grad]
        if price_params and lr_price > 0.0:
            param_groups.append({"params": price_params, "lr": lr_price})
            extra_params.extend(price_params)
        opt = torch.optim.Adam(param_groups)
        phi_params = opt.param_groups[4]["params"]
        total = epochs * len(batches)
        decay = (args.lr_final_ratio) ** (1.0 / max(total - 1, 1))
        avg, n_avg, it = None, 0, 0
        best_val_ll, best_state, bad_checks = -float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            t_ep = time.time()
            acc, sizes = [], []
            g_r0_sum = torch.zeros_like(model.rho_0_free)
            g_rc_sum = torch.zeros_like(model.rho_c)
            g_lam_sum = torch.zeros_like(model.lam)
            mod_cat_pres = torch.zeros(C)
            for bi in rng.permutation(len(batches)):
                it += 1
                bt = batches[bi]
                bt.bind(model)
                value, extra = step_fn(bt)
                loss = -value + penalties(model, args, N, r, tied, price_init=price_init)
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
                g_lam = model.lam.grad.detach().clone()
                g_r0_sum.add_(g_r0)
                g_rc_sum.add_(g_rc)
                g_lam_sum.add_(g_lam)
                torch.nn.utils.clip_grad_norm_(base + extra_params + list(phi_params), 10.0)
                opt.step()
                if tied is not None:
                    with torch.no_grad():
                        model.phi[:, :r] = tied[0][tied[2]] + tied[1]
                if mm_rho > 0.0 or mm_rho0 > 0.0 or mm_lam > 0.0:
                    scale_lr = opt.param_groups[4]["lr"] / max(lr_phi, 1e-12) if lr_phi > 0 else 1.0
                    with torch.no_grad():
                        if mm_rho0 > 0.0:
                            model.rho_0_free.sub_(mm_rho0 * scale_lr * g_r0)
                        if mm_rho > 0.0:
                            model.rho_c.sub_((mm_rho * scale_lr / len(batches)) * (g_rc / d_cat_prec))
                        if mm_lam > 0.0:
                            model.lam.sub_((mm_lam * scale_lr / len(batches)) * (g_lam / p_item_prec))
                project(model, caps, opt, r)
                for gp in opt.param_groups:
                    gp["lr"] *= decay
                acc.append(float(value.detach()))
                if extra is not None:
                    sizes.append(extra[:2])
                    if len(extra) > 2:
                        mod_cat_pres.add_(extra[2])
                if average and it > total // 3:
                    st = {k: v.detach().clone() for k, v in model.state_dict().items()}
                    avg = st if avg is None else {k: (avg[k] + (st[k] - avg[k]) / (n_avg + 1)) if st[k].is_floating_point()
                                                   else st[k] for k in st}
                    n_avg += 1
            if mm_rho > 0.0 or mm_rho0 > 0.0 or mm_lam > 0.0:
                with torch.no_grad():
                    g_r0_bar = g_r0_sum / len(batches)
                    g_rc_bar = g_rc_sum / len(batches)
                    g_lam_bar = g_lam_sum / len(batches)
                    if mm_rho0 > 0.0:
                        model.rho_0_free.sub_(0.5 * (g_r0_bar / p_size_prec).clamp(-0.5, 0.5))
                    if mm_rho > 0.0:
                        model.rho_c.sub_(0.5 * (g_rc_bar / d_cat_prec).clamp(-0.5, 0.5))
                    if mm_lam > 0.0:
                        model.lam.sub_(0.35 * (g_lam_bar / p_item_prec).clamp(-0.35, 0.35))
                project(model, caps, opt, r)
            sv = torch.linalg.svdvals(phi_r(model, r).detach())
            row = {"stage": name, "epoch": epoch, "objective": float(np.mean(acc)),
                   "phi_eig_top3": [round(float(v), 3) for v in (sv[:3] ** 2)],
                   "lr_phi": opt.param_groups[4]["lr"],
                   "epoch_seconds": round(time.time() - t_ep, 2),
                   "seconds": round(time.time() - started, 1)}
            if getattr(model, "has_iv_slope", False) and getattr(model, "iv_slope_raw", None) is not None:
                lam_c = torch.sigmoid(model.iv_slope_raw.detach())
                row["iv_slope_mean"] = round(float(lam_c.mean()), 4)
                row["iv_slope_min"] = round(float(lam_c.min()), 4)
                row["iv_slope_max"] = round(float(lam_c.max()), 4)
            if sizes:
                row["chain_mean_size"] = float(np.mean([s[0] for s in sizes]))
                row["data_mean_size"] = float(np.mean([s[1] for s in sizes]))
                cat_gap = (mod_cat_pres / N - p_obs_cat_pres).abs()
                row["cat_pres_gap_mean"] = round(float(cat_gap.mean()), 4)
                row["cat_pres_gap_max"] = round(float(cat_gap.max()), 4)
            if monitor_val and val_batches and (epoch % args.val_every == 0 or epoch == epochs):
                cur_st = {k: v.detach().clone() for k, v in model.state_dict().items()}
                eval_st = avg if avg is not None else cur_st
                model.load_state_dict(eval_st)
                v_ll, v_ess = eval_val_ll()
                row["val_ll"] = round(v_ll, 4)
                row["val_ess"] = round(v_ess, 4)
                if v_ll > best_val_ll + args.val_tol:
                    best_val_ll = v_ll
                    best_state = {k: v.clone() for k, v in eval_st.items()}
                    bad_checks = 0
                else:
                    if v_ll > best_val_ll:
                        best_val_ll = v_ll
                        best_state = {k: v.clone() for k, v in eval_st.items()}
                    if epoch >= args.min_sml_epochs:
                        bad_checks += 1
                model.load_state_dict(cur_st)
                emit(row)
                if bad_checks >= args.patience and epoch >= args.min_sml_epochs:
                    emit({"event": "early_stop", "stage": name, "epoch": epoch, "best_val_ll": round(best_val_ll, 4)})
                    break
            else:
                emit(row)
        if best_state is not None:
            model.load_state_dict(best_state)
        elif avg is not None:
            model.load_state_dict(avg)
        if tied is not None:
            with torch.no_grad():
                tied[1].copy_(model.phi[:, :r] - tied[0][tied[2]])

    lr_u_pl = args.lr_u_cat if args.u_cat else 0.0
    lr_u_sml = args.lr_u_cat if args.u_cat else 0.0
    lr_u_cal = 0.5 * args.lr_u_cat if args.u_cat else 0.0
    lr_iv_pl = args.lr_iv_slope if args.iv_slope else 0.0
    lr_iv_sml = args.lr_iv_slope if args.iv_slope else 0.0
    lr_iv_cal = 0.5 * args.lr_iv_slope if args.iv_slope else 0.0
    # stage A: pseudo-likelihood (gentle Phi warm-up with separate rho_0 / rho_c / lam learning rates)
    run_stage("pseudo-likelihood", args.pl_epochs, args.lr_base, lr_lam, args.lr_rho, args.lr_rho0, args.lr_phi_pl,
              lambda bt: (pseudo_loglik(model, bt.ix, bt.obs, r), None), average=False, lr_u_cat=lr_u_pl,
              lr_rec=args.lr_rec, lr_iv_slope=lr_iv_pl)
    torch.save({"model": {k: v.detach().clone() for k, v in model.state_dict().items()}, "rank": r,
                "use_f_extra": bool(args.f_extra), "presence_phi": bool(args.presence_phi),
                "cat_rho0": bool(args.cat_rho0), "u_cat": bool(args.u_cat), "recency": bool(args.recency),
                "iv_slope": bool(args.iv_slope), "iv_price_only": bool(args.iv_price_only),
                "stage": "pseudo-likelihood"}, args.out.with_name(args.out.stem + "_pl.pt"))

    # stage B: stochastic maximum likelihood with persistent per-trip chains (data-initialized)
    if args.unfreeze_price:
        for n in price_names:
            getattr(model, n).requires_grad_(True)
    lr_price_sml = args.lr_price if args.unfreeze_price else 0.0
    lr_price_cal = 0.5 * args.lr_price if args.unfreeze_price else 0.0
    chains = [bt.obs.clone() for bt in batches]
    index = {id(bt): i for i, bt in enumerate(batches)}

    def sml_step(bt):
        i = index[id(bt)]
        ix = bt.ix
        X = gibbs(model, ix, chains[i], r, gen, sweeps=args.sweeps)
        chains[i] = X
        with torch.no_grad():
            z = basket_field(model, ix, X, r) + torch.randn(ix.B, r, generator=gen)
            nc_mod = torch.zeros(ix.n_rows).index_add_(0, ix.row_of, X.double())
            cat_pres_b = torch.zeros(C).index_add_(0, ix.row_cat, (nc_mod > 0).double())
        e_obs = obs_energy(model, bt, r)
        lz = log_zc(model, ix, slot_utilities(model, ix, z, r), z=z)
        size_chain = float(torch.zeros(ix.B).index_add_(0, ix.item_trip, X.double()).mean())
        return (e_obs - lz).mean(), (size_chain, float(bt.obs.sum()) / ix.B, cat_pres_b)

    run_stage("sml", args.sml_epochs, args.lr_base, lr_lam, args.lr_rho, args.lr_rho0, args.lr_phi_sml, sml_step,
              average=True, mm_rho=args.mm_rho, mm_rho0=args.mm_rho0, mm_lam=args.mm_lam, monitor_val=True,
              lr_price=lr_price_sml, lr_u_cat=lr_u_sml, lr_rec=args.lr_rec, lr_iv_slope=lr_iv_sml)
    if args.cal_epochs > 0:
        run_stage("calibrate", args.cal_epochs, 0.0, 0.5 * lr_lam, args.lr_rho, args.lr_rho0, 0.0, sml_step,
                  average=False, mm_rho=args.mm_rho, mm_rho0=args.mm_rho0, mm_lam=args.mm_lam,
                  lr_price=lr_price_cal, lr_u_cat=lr_u_cal, lr_rec=0.5 * args.lr_rec, lr_iv_slope=lr_iv_cal)
    if val_batches:
        v_ll, v_ess = eval_val_ll()
        emit({"event": "final_val", "val_ll": round(v_ll, 4), "val_ess": round(v_ess, 4)})
    torch.save({"model": {k: v.detach().clone() for k, v in model.state_dict().items()}, "rank": r,
                "use_f_extra": bool(args.f_extra), "presence_phi": bool(args.presence_phi),
                "cat_rho0": bool(args.cat_rho0), "unfreeze_price": bool(args.unfreeze_price),
                "u_cat": bool(args.u_cat), "recency": bool(args.recency), "iv_slope": bool(args.iv_slope), "iv_price_only": bool(args.iv_price_only),
                "stage": "sml", "seconds": time.time() - started, "args": {k: str(v) for k, v in vars(args).items()}},
               args.out.with_suffix(".pt"))
    emit({"event": "trained", "seconds": round(time.time() - started, 1)})


# ---- evaluate ---------------------------------------------------------------------------------
@torch.no_grad()
def is_log_z_orig(model, bt, r, gen, burn, components, thin, draws, defensive=0.1):
    """Original single-chain importance-sampled log Z per trip."""
    ix = bt.ix
    X = sample_slots(model, ix, slot_utilities(model, ix, None, r, beta=0.0), gen, beta=0.0)
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
        log_w.append(lp + log_zc(model, ix, slot_utilities(model, ix, z, r), z=z) - lq)
    log_w = torch.stack(log_w, 1)
    est = torch.logsumexp(log_w, 1) - math.log(draws)
    h = draws // 2
    est_a = torch.logsumexp(log_w[:, :h], 1) - math.log(h)
    est_b = torch.logsumexp(log_w[:, h:], 1) - math.log(draws - h)
    ess = 1.0 / torch.softmax(log_w, 1).square().sum(1) / draws
    return est, est_a - est_b, ess, X


@torch.no_grad()
def is_log_z(model, bt, r, gen, burn, components, thin, draws, defensive=0.1):
    """Multi-chain tempered + active-subspace overdispersed importance-sampled log Z per trip."""
    if os.environ.get("STRESS_ORIG_IS", "0") == "1":
        return is_log_z_orig(model, bt, r, gen, burn, components, thin, draws, defensive)
    ix = bt.ix
    B = ix.B
    ph_full = phi_r(model, r).detach()
    _, sv, Vt = torch.linalg.svd(ph_full, full_matrices=False)
    eigs = sv.square()
    active = eigs > 4.0
    sig_def = torch.where(active, torch.full_like(sv, 1.35), torch.ones_like(sv))
    log_det_def = sig_def.log().sum()

    # Chain 0: exact standard burn-in from beta=0 so returned X has unchanged law for classifier test
    X = sample_slots(model, ix, slot_utilities(model, ix, None, r, beta=0.0), gen, beta=0.0)
    X = gibbs(model, ix, X, r, gen, sweeps=burn)
    centres = []
    n_ch = 4
    k_per_ch = max(components // 2, 64)
    thin_c = max(thin // 2, 1)
    X_out = None
    steps_0 = max(components * thin, k_per_ch * thin_c)
    for k in range(steps_0):
        X = gibbs(model, ix, X, r, gen)
        if (k + 1) % thin_c == 0 and len(centres) < k_per_ch:
            centres.append(basket_field(model, ix, X, r))
        if (k + 1) == components * thin:
            X_out = X.clone()
    if X_out is None:
        X_out = X.clone()

    # Chains 1..3: diverse initializations with tempered annealing (beta = 0.25 -> 0.5 -> 0.75 -> 1.0)
    t_sweeps = max(burn // 8, 25)
    for ch in range(1, n_ch):
        if ch == 1:
            z0 = torch.zeros(B, r)
            Xc = sample_slots(model, ix, slot_utilities(model, ix, z0, r, beta=1.0), gen, z=z0, beta=1.0)
        elif ch in (2, 3) and float(sv[0]) > 1.0:
            sgn = 1.0 if ch == 2 else -1.0
            z_seed = (sgn * 1.5 * sv[0] * Vt[0])[None, :].expand(B, r)
            Xc = sample_slots(model, ix, slot_utilities(model, ix, z_seed, r, beta=1.0), gen, z=z_seed, beta=1.0)
        else:
            Xc = sample_slots(model, ix, slot_utilities(model, ix, None, r, beta=0.0), gen, beta=0.0)
        for b_temp in (0.25, 0.5, 0.75, 1.0):
            Xc = gibbs(model, ix, Xc, r, gen, sweeps=t_sweeps, beta=b_temp)
        for k in range(k_per_ch * thin_c):
            Xc = gibbs(model, ix, Xc, r, gen)
            if (k + 1) % thin_c == 0:
                centres.append(basket_field(model, ix, Xc, r))

    centres = torch.stack(centres, 1)                       # B x Kc x r
    Kc = centres.shape[1]
    centres_rot = centres @ Vt.T                            # B x Kc x r
    centres_rot_T = centres_rot.transpose(1, 2).contiguous()  # B x r x Kc
    c_norm2 = -0.5 * centres_rot.square().sum(2)            # B x Kc

    # Mixture weights: (1 - defensive) uniform over Kc centres + defensive overdispersed N(0, diag(sig_def^2))
    w_cent = (1.0 - defensive) / float(Kc)
    probs = torch.tensor([w_cent] * Kc + [defensive], dtype=centres.dtype)
    cdf = probs.cumsum(0)
    cdf[-1] = 1.0

    half = 0.5 * r * math.log(2 * math.pi)
    h = draws // 2
    h2 = draws - h
    # Independent stratified component indices for each half [0:h] and [h:draws]
    u_a = (torch.arange(h, dtype=centres.dtype)[None, :] + torch.rand(B, h, generator=gen)) / float(h)
    u_a = u_a.gather(1, torch.argsort(torch.rand(B, h, generator=gen), dim=1))
    u_b = (torch.arange(h2, dtype=centres.dtype)[None, :] + torch.rand(B, h2, generator=gen)) / float(h2)
    u_b = u_b.gather(1, torch.argsort(torch.rand(B, h2, generator=gen), dim=1))
    picks_all = torch.cat([torch.searchsorted(cdf, u_a), torch.searchsorted(cdf, u_b)], dim=1).clamp(max=Kc)

    # Vectorized proposal sampling and log_q evaluation across all draws at once
    prior_all = picks_all == Kc                                                        # B x draws
    c_pick_all = centres_rot.gather(1, picks_all.clamp(max=Kc - 1)[:, :, None].expand(B, draws, r))
    mean_rot_all = torch.where(prior_all[:, :, None], torch.zeros(B, draws, r), c_pick_all)
    scale_rot_all = torch.where(prior_all[:, :, None], sig_def[None, None, :], torch.ones(B, draws, r))
    z_rot_all = mean_rot_all + scale_rot_all * torch.randn(B, draws, r, generator=gen)
    z_all = z_rot_all @ Vt                                                             # B x draws x r
    z_norm2_all = -0.5 * z_rot_all.square().sum(2)                                     # B x draws
    lp_all = z_norm2_all - half
    comp_all = z_norm2_all[:, :, None] + c_norm2[:, None, :] + torch.bmm(z_rot_all, centres_rot_T) - half
    lp_def_all = -0.5 * (z_rot_all / sig_def[None, None, :]).square().sum(2) - half - log_det_def
    lq_all = torch.logaddexp(math.log1p(-defensive) + torch.logsumexp(comp_all, 2) - math.log(Kc),
                             math.log(defensive) + lp_def_all)
    ph_slot = ph_full[ix.item]
    b_slot = model.b_flat(ix)
    u0_slot = b_slot - 0.5 * ph_slot.square().sum(-1)
    iv_sh = row_iv_shift(model, ix, b_slot=b_slot)
    lzc_list = []
    for d_idx in range(draws):
        z_d = z_all[:, d_idx]
        u_d = u0_slot + (z_d[ix.item_trip] * ph_slot).sum(-1)
        lzc_list.append(log_zc(model, ix, u_d, z=z_d, iv_shift=iv_sh))
    log_w = lp_all + torch.stack(lzc_list, 1) - lq_all
    est = torch.logsumexp(log_w, 1) - math.log(draws)
    est_a = torch.logsumexp(log_w[:, :h], 1) - math.log(h)
    est_b = torch.logsumexp(log_w[:, h:], 1) - math.log(h2)
    ess = 1.0 / torch.softmax(log_w, 1).square().sum(1) / draws
    return est, est_a - est_b, ess, X_out


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
    from price_response import changed_price_context
    blob = torch.load(args.out.with_suffix(".pt") if args.checkpoint is None else args.checkpoint,
                      map_location="cpu", weights_only=False)
    data, model, meta, batcher = setup(args.initialization, args.additive, args.threads,
                                       recency=bool(blob.get("recency", False)))
    configure_model_from_blob(model, blob)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    r = int(blob["rank"])
    J = int(data["n_item"])
    item_cat = np.zeros(J, dtype=np.int64)
    item_cat[data["line_item"]] = data["line_cat"]
    gen = torch.Generator().manual_seed(args.seed + 5)
    tag = args.out.stem if args.checkpoint is None else Path(args.checkpoint).stem
    report = {"checkpoint": str(args.checkpoint or args.out.with_suffix(".pt")),
              "phi_eigenvalues": (torch.linalg.svdvals(phi_r(model, r)) ** 2).tolist(),
              "rho_c_min": float(model.rho_c.min()),
              "use_f_extra": getattr(model, "f_extra", None) is not None,
              "presence_phi": bool(getattr(model, "presence_phi", False)),
              "cat_rho0": bool(getattr(model, "cat_rho0", False)),
              "iv_slope": bool(getattr(model, "has_iv_slope", False)),
              "iv_price_only": bool(getattr(model, "iv_price_only", False))}
    if getattr(model, "has_iv_slope", False) and getattr(model, "iv_slope_raw", None) is not None:
        lam_c = torch.sigmoid(model.iv_slope_raw.detach())
        report["iv_slope_lambda"] = {
            "mean": float(lam_c.mean()),
            "min": float(lam_c.min()),
            "max": float(lam_c.max()),
            "values": [round(float(v), 4) for v in lam_c],
        }
    t0 = time.time()

    # 1. held-out log-likelihood on the certified reference trips
    ref = np.load(args.reference)
    trips = ref["trips"].astype(np.int64)
    if args.ll_trips:
        trips = trips[:args.ll_trips]
    ll, half_diff, ess_all = [], [], []
    for s in range(0, len(trips), args.eval_batch):
        bt = Batch(batcher, trips[s:s + args.eval_batch], J)
        bt.bind(model)
        est, dhalf, ess, _ = is_log_z(model, bt, r, gen, args.burn, args.components, args.thin, args.draws)
        e = obs_energy(model, bt, r)
        ll.append((e - est).numpy()); half_diff.append(dhalf.numpy()); ess_all.append(ess.numpy())
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
    if args.phi0_refit is not None and str(args.phi0_refit) != "none":
        o = np.load(args.phi0_refit)
        m = dict(zip(o["trips"].tolist(), o["ll"].tolist()))
        d = ll - np.array([m[int(t)] for t in trips])
        report["likelihood"]["vs_phi0_refit"] = float(d.mean())
        report["likelihood"]["vs_phi0_refit_household_95"] = clustered_ci(d, ref["household"][:n])
    np.savez(args.out.with_name(tag + ".per_trip.npz"), trips=trips, ll=ll, half_diff=half_diff, ess=ess_all)
    print(json.dumps(report["likelihood"]), flush=True)

    # 2. classifier two-sample test on all test trips: one fresh basket per trip for each of --auc-seeds
    from eval_3seed_auc import seed_aucs
    aucs = seed_aucs(model, data, batcher, r, args.auc_seeds, args.burn, args.eval_batch)
    aucs = {"checkpoint": report["checkpoint"], **aucs}
    args.out.with_name(tag + "_3seed_auc.json").write_text(json.dumps(aucs, indent=2) + "\n")
    first = aucs["seeds"][0]
    report["classifier"] = {"full_basket_auc": first["full_basket_auc"], "category_presence_auc": first["category_presence_auc"],
                            "size_mean_real": first["size_mean_real"], "size_mean_generated": first["size_mean_generated"],
                            "full_basket_auc_mean": aucs["full_basket_auc_mean"],
                            "category_presence_auc_mean": aucs["category_presence_auc_mean"],
                            "category_presence_auc_sd": aucs["category_presence_auc_sd"],
                            "size_mean_generated_mean": aucs["size_mean_generated_mean"]}
    print(json.dumps(report["classifier"]), flush=True)
    if args.no_price_scenarios:   # real data (ERIM, Dunnhumby): no causal price truth to score against
        report["pass_rules"] = {
            "1_ll_vs_certified_ci_above_0": report["likelihood"]["vs_certified_v4_household_95"][0] > 0,
            "2_category_auc_le_0.80": report["classifier"]["category_presence_auc_mean"] <= 0.80}
        report["eval_seconds"] = time.time() - t0
        args.out.with_name(tag + ".eval.json").write_text(json.dumps(report, indent=1) + "\n")
        print(json.dumps(report["pass_rules"]), flush=True)
        return

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
        X = sample_slots(model, bt.ix, slot_utilities(model, bt.ix, None, r, beta=0.0), gen, beta=0.0)
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
            bt.bind(model, ctx)
            iv_sh_alt = row_iv_shift(model, ix)
            if iv_sh_alt is not None:
                iv_sh_alt = iv_sh_alt.detach()
            iv_sh_base = None
            if product is not None:
                bt.bind(model)
                iv_sh_base = row_iv_shift(model, ix)
                if iv_sh_base is not None:
                    iv_sh_base = iv_sh_base.detach()
            lzs, grads, lzb = [], [], []
            for z in zs:
                bt.bind(model, ctx)
                u = slot_utilities(model, ix, z, r).detach().requires_grad_(True)
                with torch.enable_grad():
                    lz = log_zc(model, ix, u, z=z, iv_shift=iv_sh_alt)
                    g, = torch.autograd.grad(lz.sum(), u)
                lzs.append(lz.detach()); grads.append(g)
                if product is not None:
                    bt.bind(model)
                    lzb.append(log_zc(model, ix, slot_utilities(model, ix, z, r), z=z, iv_shift=iv_sh_base))
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
        "2_category_auc_le_0.80": report["classifier"]["category_presence_auc_mean"] <= 0.80,
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
    p.add_argument("--auc-seeds", type=int, nargs="+", default=[41006, 41007, 41008],
                   help="classifier seeds; eval writes <run>_3seed_auc.json and pass rule 2 uses their mean")
    p.add_argument("--no-price-scenarios", action="store_true",
                   help="skip the price scenarios (real data has no causal truth); pass rules 1-2 only")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=None)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--phi-init", type=float, default=0.05)
    p.add_argument("--pl-epochs", type=int, default=2)
    p.add_argument("--sml-epochs", type=int, default=16)
    p.add_argument("--min-sml-epochs", type=int, default=0)
    p.add_argument("--cal-epochs", type=int, default=1)
    p.add_argument("--val-trips", type=int, default=0)
    p.add_argument("--val-every", type=int, default=4)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--val-tol", type=float, default=0.005)
    p.add_argument("--sweeps", type=int, default=4)
    p.add_argument("--lr-base", type=float, default=3e-4)
    p.add_argument("--lr-lam", type=float, default=None)
    p.add_argument("--lr-rho", type=float, default=3e-2)
    p.add_argument("--lr-rho0", type=float, default=5e-2)
    p.add_argument("--lr-phi-pl", type=float, default=1e-3)
    p.add_argument("--lr-phi-sml", type=float, default=3e-3)
    p.add_argument("--mm-rho", type=float, default=1.0)
    p.add_argument("--mm-rho0", type=float, default=0.5)
    p.add_argument("--mm-lam", type=float, default=0.0)
    p.add_argument("--lr-final-ratio", type=float, default=0.1)
    p.add_argument("--pool-prod", type=float, default=1.45)
    p.add_argument("--phi-prec", type=float, default=2.0)
    p.add_argument("--rho-c-prec", type=float, default=0.2)
    p.add_argument("--tie-phi", action="store_true", help="phi_j = psi_category(j) + delta_j")
    p.add_argument("--delta-prec", type=float, default=200.0, help="prior precision on delta (with --tie-phi)")
    p.add_argument("--f-extra", action="store_true", help="M1: free per-category count potential f_c(n_c) with >=4 pooled")
    p.add_argument("--f-extra-prec", type=float, default=2.0, help="prior precision on f_extra deviations from quadratic")
    p.add_argument("--presence-phi", action="store_true", help="M2: presence-based category coupling s(S) = sum_c 1(n_c>0) psi_c + sum_j delta_j")
    p.add_argument("--cat-rho0", action="store_true", help="M3: apply size potential rho_0 to active category count K_cat instead of |S|")
    p.add_argument("--unfreeze-price", action="store_true", help="T1: unfreeze gamma, beta, price_kappa and psi in SML and calibration stages")
    p.add_argument("--lr-price", type=float, default=3e-3, help="T1: learning rate for unfrozen price/recency parameters in SML (halved in calibrate)")
    p.add_argument("--price-prec", type=float, default=2.0, help="T1: prior precision on gamma, beta, price_kappa, psi centred at initial Phi=0 values")
    p.add_argument("--u-cat", action="store_true", help="M4a: household taste per category b_tj += theta_h . U_c(j)")
    p.add_argument("--lr-u-cat", type=float, default=5e-3, help="M4a: learning rate for U_c in PL and SML (halved in calibrate)")
    p.add_argument("--u-cat-prec", type=float, default=2.0, help="M4a: weak prior precision on U_c centred at 0")
    p.add_argument("--recency", action="store_true",
                   help="purchase history: sub-commodity recency (psi) and product history (psi_item), trained from stage A")
    p.add_argument("--lr-rec", type=float, default=5e-3, help="learning rate for psi and psi_item (halved in calibrate)")
    p.add_argument("--rec-prec", type=float, default=2.0, help="weak prior precision on psi and psi_item centred at 0")
    p.add_argument("--iv-slope", action="store_true",
                   help="M4b: per-category presence term (lambda_c - 1) * IV_c(x) with lambda_c = sigmoid(raw) in (0, 1]")
    p.add_argument("--lr-iv-slope", type=float, default=0.05,
                   help="M4b: learning rate for iv_slope_raw in PL and SML (halved in calibrate)")
    p.add_argument("--iv-price-only", action="store_true",
                   help="M4c: with --iv-slope, discount only the price part of IV (actual minus regular price)")
    p.add_argument("--iv-slope-init", type=float, default=0.0,
                   help="M4b: initial raw logit for lambda_c = sigmoid(iv_slope_raw)")
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
