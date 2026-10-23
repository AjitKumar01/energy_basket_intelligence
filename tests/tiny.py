"""A tiny world where every basket can be listed, for exact checks against enumeration.

Two trips at one store; trip 0 is offered categories {0: 3 products, 1: 2 products},
trip 1 categories {0: 3 products, 2: 1 product}; baskets have 1..nmax = 4 products.
Every Version-4 block can be switched on: f_extra (M1), psi_cat (M2), size on categories
(M3), u_cat (M4a), iv_slope_raw (M4c), the discount split, and purchase history with
centring and its size term (``history``).
"""
from __future__ import annotations

import itertools
import math

import torch

from v4.batches import Batch, RaggedIndex
from v4.model import BasketModel

torch.set_default_dtype(torch.float64)

J, C, NMAX, RANK = 6, 3, 4, 2
PRODUCT_CAT = [0, 0, 0, 1, 1, 2]
TRIPS = [[0, 1, 2, 3, 4], [0, 1, 2, 5]]            # offered products per trip


def make_model(seed=0, *, f_extra=False, presence=False, categories=False, u_cat=False,
               iv=False, discount=False, history=None):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    m = BasketModel(J, 2, C, K=3, Kz=4, Kp=2, nmax=NMAX, R=NMAX, S=1, seed=seed)
    r = lambda *shape, s=1.0: s * torch.randn(*shape, generator=g)  # noqa: E731
    with torch.no_grad():
        m.cat_of.copy_(torch.tensor(PRODUCT_CAT))
        m.lam.copy_(r(J, s=0.5))
        m.rho_c.copy_(torch.tensor([-0.3, 0.4, 0.1]))
        m.rho_0_free.copy_(r(NMAX, s=0.5))
        m.w_dsp.copy_(r(J, s=0.3))
        m.w_mlr.copy_(r(J, s=0.3))
        m.phi[:, :RANK] = r(J, RANK, s=0.5)
    m.rank = RANK
    P = torch.nn.Parameter
    if f_extra:
        m.f_extra = P(r(C, 3, s=0.3))
    if presence:
        m.psi_cat = P(r(C, RANK, s=0.4))
    m.size_on_categories = categories
    if u_cat:
        m.u_cat = P(r(C, 3, s=0.3))
    if iv:
        m.iv_slope_raw = P(r(C, s=1.0))
    if discount:
        m.disc_kappa = P(torch.tensor(0.9))
        m.disc_common = P(torch.tensor(0.4))
    if history:
        with torch.no_grad():
            m.psi.copy_(r(J, 4, s=0.4))
        m.psi_item = P(r(C, 3, s=0.5))
        m.hist_size = P(torch.tensor(0.3))
        m.centre_history = history                      # "mean" or "inclusive"
    return m


def make_batch(basket=((0, 3), (1, 5)), seed=1):
    """A Batch for both trips with observed baskets ``basket`` (product ids per trip)."""
    g = torch.Generator().manual_seed(seed)
    item, row_of, row_trip, row_cat = [], [], [], []
    for t, offered in enumerate(TRIPS):
        for c in sorted({PRODUCT_CAT[j] for j in offered}):
            for j in offered:
                if PRODUCT_CAT[j] == c:
                    item.append(j)
                    row_of.append(len(row_trip))
            row_trip.append(t)
            row_cat.append(c)
    ix = RaggedIndex(item, row_of, row_trip, row_cat, len(TRIPS))

    def context(items, trips):
        n = len(items)
        return {"dlp": torch.randn(n, generator=g) * 0.2, "disp": (torch.rand(n, generator=g) < 0.4).double(),
                "mail": (torch.rand(n, generator=g) < 0.3).double(), "dsc": -torch.rand(n, generator=g) * 0.2,
                "week": torch.zeros(n, dtype=torch.long), "store": torch.zeros(n, dtype=torch.long)}

    ctx = context(ix.item, ix.item_trip)
    ctx["dlp_bar"] = torch.zeros(2).index_add_(0, ix.item_trip, ctx["dlp"]) / torch.bincount(ix.item_trip).double()
    n = len(ix.item)                                     # purchase history (zero effect unless history=True)
    ctx["rec"] = torch.rand(n, 4, generator=g)
    ctx["rec_item"] = torch.stack([(torch.rand(n, generator=g) < 0.5).double(),
                                   (torch.rand(n, generator=g) < 0.3).double(), torch.rand(n, generator=g)], -1)
    line_item = torch.tensor([j for b in basket for j in b])
    line_trip = torch.tensor([t for t, b in enumerate(basket) for _ in b])
    slot = {(int(t), int(j)): s for s, (t, j) in enumerate(zip(ix.item_trip, ix.item))}
    line_slots = [slot[(int(t), int(j))] for t, j in zip(line_trip, line_item)]
    line_ctx = {k: (v[line_slots] if k != "dlp_bar" else v) for k, v in ctx.items()}
    observed = torch.zeros(len(ix.item), dtype=torch.bool)
    observed[line_slots] = True
    return Batch(None, ix, ctx, line_ctx, torch.tensor([0, 1]), line_item, line_trip,
                 torch.tensor([PRODUCT_CAT[j] for j in line_item.tolist()]), observed)


def baskets_of(ix, trip, max_per_category=NMAX):
    """All baskets (tuples of slot indices) of a trip with 1..nmax products and at most
    ``max_per_category`` from any one category."""
    slots = [s for s in range(len(ix.item)) if int(ix.item_trip[s]) == trip]
    for n in range(1, NMAX + 1):
        for S in itertools.combinations(slots, n):
            rows = [int(ix.row_of[s]) for s in S]
            if max(rows.count(r) for r in set(rows)) <= max_per_category:
                yield S


def energy_given_z(model, ix, u, z, iv, S):
    """E_z(S): the law of S given z, written out term by term."""
    e = sum(float(u[s]) for s in S)
    counts = {}
    for s in S:
        counts[int(ix.row_of[s])] = counts.get(int(ix.row_of[s]), 0) + 1
    for row, n in counts.items():
        c = int(ix.row_cat[row])
        e -= float(model.rho_c[c]) * n * (n - 1) / 2
        if model.f_extra is not None and n >= 2:
            e -= float(model.f_extra[c, min(n, 4) - 2])
        if model.psi_cat is not None and z is not None:
            psi = model.psi_cat[c]
            e -= (n - 1) * (float(z[int(ix.row_trip[row])] @ psi) - 0.5 * float(psi @ psi))
        if iv is not None:
            e += n * float(iv[row])
    size = len(counts) if model.size_on_categories else len(S)
    return e - float(model.rho_0()[size])


def full_energy(model, ix, b, iv, S):
    """E(S) of the full model, from the Gaussian identity (see model.basket_field)."""
    phi = model.phi_r()
    s = sum((phi[int(ix.item[j])] for j in S), torch.zeros(model.rank))
    e = sum(float(b[j]) for j in S) - 0.5 * sum(float(phi[int(ix.item[j])] @ phi[int(ix.item[j])]) for j in S)
    counts = {}
    for j in S:
        counts[int(ix.row_of[j])] = counts.get(int(ix.row_of[j]), 0) + 1
    for row, n in counts.items():
        c = int(ix.row_cat[row])
        e -= float(model.rho_c[c]) * n * (n - 1) / 2
        if model.f_extra is not None and n >= 2:
            e -= float(model.f_extra[c, min(n, 4) - 2])
        if model.psi_cat is not None:
            s = s - (n - 1) * model.psi_cat[c]
            e += 0.5 * (n - 1) * float(model.psi_cat[c] @ model.psi_cat[c])
        if iv is not None:
            e += n * float(iv[row])
    size = len(counts) if model.size_on_categories else len(S)
    return e + 0.5 * float(s @ s) - float(model.rho_0()[size])


def log_sum_exp(values):
    m = max(values)
    return m + math.log(sum(math.exp(v - m) for v in values))
