"""How the cost depends on the item cap, and how much log Z each trip loses when cut at K.

For held-out trips at a fixed z per trip (z = s(observed basket) + N(0, I)), for each K:
model.nmax = K, then the median time of exact log Z and of one Gibbs sweep, and the per-trip
loss  log Z(nmax) - log Z(K)  (share of trips below 1e-12, 1e-8, 1e-4, and the largest).
The cap is restored afterwards.  The loss is at one typical z only: inside a Gibbs chain z
moves, so a cap chosen this way is not safe without a bound that holds at every z.

    python bench/cap_profile.py --world real [--trips 256] [--caps 120 96 80 64 48 40 32 24 16]
"""
from __future__ import annotations

import time

import numpy as np
import torch

from common import batch_of, emit, load, parser
from v4.exact import log_partition
from v4.sampling import draw_from_model, gibbs


def main():
    p = parser(__doc__)
    p.add_argument("--trips", type=int, default=256)
    p.add_argument("--caps", type=int, nargs="+")
    a = p.parse_args()
    model, batcher, held = load(a)
    nmax = int(model.nmax)
    caps = sorted({k for k in (a.caps or [nmax, 96, 80, 64, 48, 40, 32, 24, 16]) if k <= nmax}, reverse=True)
    if caps[0] != nmax:
        caps = [nmax] + caps
    with torch.no_grad():
        b = batch_of(model, batcher, held[:a.trips])
        ix, iv = b.ix, model.iv_shift(b.ix)
        gen = torch.Generator().manual_seed(11)
        z = model.basket_field(ix, b.observed) + torch.randn(ix.B, model.rank, generator=gen)
        u = model.slot_utilities_given_z(ix, z)
        ref = None
        try:
            for K in caps:
                model.nmax = K
                ts = []
                for _ in range(3):
                    t = time.time()
                    lz = log_partition(model, ix, u, z=z, iv=iv)
                    ts.append(time.time() - t)
                X = draw_from_model(model, ix, gen, 0)
                t = time.time()
                gibbs(model, ix, X, gen, sweeps=5, iv=iv)
                t_sweep = (time.time() - t) / 5
                ref = lz.clone() if ref is None else ref
                d = (ref - lz).numpy()
                emit(K=K, logz_seconds=round(float(np.median(ts)), 4), sweep_seconds=round(t_sweep, 4),
                     share_loss_below_1e_12=round(float((d < 1e-12).mean()), 4),
                     share_loss_below_1e_8=round(float((d < 1e-8).mean()), 4),
                     share_loss_below_1e_4=round(float((d < 1e-4).mean()), 4), loss_max=float(d.max()))
        finally:
            model.nmax = nmax


if __name__ == "__main__":
    main()
