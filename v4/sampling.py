"""Exact basket sampling given a taste z.

Given z, the law of the basket has no pairwise term (model.slot_utilities_given_z and the
M2 presence shift), so the C++ sampler draws S | z exactly, category by category, from the
same dynamic-program tables that give Z_z (``sample_baskets``).  Drawing z ~ N(0, I) first
gives an exact draw from the mixture law (v4/mixture.py).

All random numbers come from a torch.Generator; the C++ code takes one seed drawn from it
per call, so a generator seed fixes every result.
"""
from __future__ import annotations

import torch

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
    """Smallest per-(trip, z) item cap of the sampler.  Each trip's cap K is chosen so that
    the baskets with more than K products carry at most 1e-8 of the mass (a certified bound;
    the cap grows to nmax whenever z needs it)."""
    return int(getattr(model, "trunc_min_cap", 32))


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
