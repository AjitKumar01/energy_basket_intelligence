"""The exact normaliser: log Z of the basket law given slot utilities u.

Given z (or with phi = 0) the energy has no pairwise product term, so baskets factor over
categories.  For each (trip, category) row:

    e_k        = sum over k-subsets of the row's products of prod exp(u_j)   (ESP, R >= k)
    g_c(k)     = e_k * exp(-rho_c C(k, 2) - f_c(k) - k_extra * presence_shift)
                 * exp(iv_shift) if k >= 1                                     (category polynomial)

and the trip's total weight of baskets with n products is the coefficient of t^n in
prod_c g_c(t), weighted by exp(-rho_0(n)).  With M3 the product also tracks the number of
categories present, m, and the weight is exp(-rho_0(m)) instead (the 2-D program).

Numerics: utilities are centred at each trip's largest value (``scale``; added back as
n * scale), and each category polynomial is tilted by its largest per-degree slope before
the product (added back as n * slope), so every intermediate stays in range.

The gradient of log Z with respect to u is the incidence P(j in S) of every slot.
"""
from __future__ import annotations

import torch

from .kernels import CategoryProduct, RowESP, category_product_2d

ESP_BUCKETS = (8, 32, 96, 256)


def trip_max(u, ix):
    return torch.full((ix.B,), -float("inf")).index_reduce_(0, ix.item_trip, u, "amax", include_self=True)


def row_esp(log_w, ix, R):
    """log e_0..e_R for every row, [rows, R+1].

    Rows are grouped by size into buckets (8, 32, 96, 256, then doubling) and each bucket
    is padded only to its own largest row; padded items have weight 0 (log -inf)."""
    out = torch.full((ix.n_rows, R + 1), -float("inf"))
    out[:, 0] = 0.0
    largest = int(ix.row_size.max()) if ix.row_size.numel() else 0
    limits = list(ESP_BUCKETS)
    while limits[-1] < largest:
        limits.append(min(largest, 2 * limits[-1]))
    lo = 0
    for hi in limits:
        in_bucket = (ix.row_size > lo) & (ix.row_size <= hi)
        lo = hi
        if not bool(in_bucket.any()):
            continue
        rows = torch.nonzero(in_bucket, as_tuple=True)[0]
        local = torch.full((ix.n_rows,), -1, dtype=torch.long)
        local[rows] = torch.arange(len(rows))
        slots = in_bucket[ix.row_of]
        padded = torch.full((len(rows) * hi,), -float("inf")).index_copy(
            -1, local[ix.row_of[slots]] * hi + ix.item_pos[slots], log_w[slots]).view(len(rows), hi)
        degree = min(R, hi)                    # a row of <= hi products has e_k = 0 for k > hi
        e = RowESP.apply(padded, ix.row_size.index_select(0, rows).clamp(max=hi), degree, 32)
        if degree < R:
            e = torch.nn.functional.pad(e, (0, R - degree), value=-float("inf"))
        out = out.index_copy(-2, rows, e)
    return out


def category_polynomials(model, ix, u, *, z=None, beta=1.0, iv=None):
    """log g_c(k) for every row, [rows, R+1], and the per-trip centring ``scale``."""
    scale = trip_max(u, ix)
    log_e = row_esp(u - scale[ix.item_trip], ix, model.R)
    k = torch.arange(model.R + 1, dtype=u.dtype)
    log_g = log_e - model.rho_c[ix.row_cat].unsqueeze(-1) * model.pair_feature(k)
    if model.f_extra is not None:                     # M1: free potential at counts 2, 3, >= 4
        f = model.f_extra[ix.row_cat]
        pad = torch.zeros(ix.n_rows, model.R + 1)
        if model.R >= 2:
            pad[:, 2] = f[:, 0]
        if model.R >= 3:
            pad[:, 3] = f[:, 1]
        if model.R >= 4:
            pad[:, 4:] = f[:, 2:3]
        log_g = log_g - pad
    shift = model.presence_shift(ix, z, beta=beta)    # M2: per extra product of the category
    if shift is not None:
        log_g = log_g - shift[:, None] * (k - 1.0).clamp_min(0.0)[None, :]
    if iv is not None:                                # M4c: once per category present
        log_g = torch.cat([log_g[:, :1], log_g[:, 1:] + iv[:, None]], dim=1)
    return log_g, scale


def _trip_grid(model, ix, log_g):
    """Rows placed into a [1, B, Cpad, R+1] grid (missing categories are the polynomial 1),
    tilted by each trip's largest per-degree slope; also the degree cap of every cell."""
    R = model.R
    grid = torch.full((1, ix.B * ix.Cpad, R + 1), -float("inf"))
    grid[..., 0] = 0.0
    grid = grid.index_copy(1, ix.flat_slot, log_g.unsqueeze(0)).view(1, ix.B, ix.Cpad, R + 1)
    degree = torch.zeros(ix.B * ix.Cpad, dtype=torch.long)
    degree[ix.flat_slot] = ix.row_size.clamp(max=R)
    k = torch.arange(R + 1, dtype=log_g.dtype)
    slope = (grid[..., 1:] / k[1:]).amax(dim=(2, 3)).clamp_min(0.0)        # [1, B]
    return grid - slope[:, :, None, None] * k, degree.view(ix.B, ix.Cpad), slope


def log_mass_by_size(model, ix, log_g, scale):
    """log total weight of baskets of each size n = 1..nmax, [B, nmax] (size potential on n)."""
    tilted, degree, slope = _trip_grid(model, ix, log_g)
    log_a = CategoryProduct.apply(tilted.contiguous(), degree.contiguous(), model.nmax)
    n = torch.arange(model.nmax + 1, dtype=log_g.dtype)
    log_a = log_a + slope.unsqueeze(-1) * n
    return (log_a[0] + n * scale.unsqueeze(-1) - model.rho_0()[:model.nmax + 1])[:, 1:]


def log_mass_by_size_and_categories(model, ix, log_g, scale):
    """log total weight of baskets with n products in m categories, n, m = 1..nmax,
    [B, nmax, nmax], with the size potential on m (M3)."""
    tilted, degree, slope = _trip_grid(model, ix, log_g)
    log_a = category_product_2d(tilted[0].contiguous(), degree.contiguous(), model.nmax)
    n = torch.arange(model.nmax + 1, dtype=log_g.dtype)
    log_a = log_a + slope[0, :, None, None] * n[None, :, None]
    return (log_a[:, 1:, 1:] + n[None, 1:, None] * scale[:, None, None]
            - model.rho_0()[None, None, 1:model.nmax + 1])


def log_partition(model, ix, u, *, z=None, beta=1.0, iv=None):
    """log Z per trip of the law of S given slot utilities u (and z for M2)."""
    log_g, scale = category_polynomials(model, ix, u, z=z, beta=beta, iv=iv)
    if model.size_on_categories:
        return torch.logsumexp(log_mass_by_size_and_categories(model, ix, log_g, scale).reshape(ix.B, -1), -1)
    return torch.logsumexp(log_mass_by_size(model, ix, log_g, scale), -1)


def additive_log_size(model, ix, u=None):
    """log mass by size 1..nmax of the additive model (phi = 0, no Version-4 terms)."""
    u = model.slot_utilities(ix) if u is None else u
    log_g, scale = category_polynomials(model, ix, u)
    return log_mass_by_size(model, ix, log_g, scale)


def incidence(model, ix, u, **kw):
    """log Z and P(j in S) at every slot (the gradient of log Z with respect to u)."""
    u = u.detach().requires_grad_(True)
    with torch.enable_grad():
        lz = log_partition(model, ix, u, **kw)
        p, = torch.autograd.grad(lz.sum(), u)
    return lz.detach(), p
