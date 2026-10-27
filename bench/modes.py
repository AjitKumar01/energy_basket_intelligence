"""Does the model's law hide a heavy giant-basket mode?

For held-out trips, Gibbs chains are started from four points:
  observed   the trip's real basket
  beta0      an exact draw from the model without interactions (as importance_log_z's chain 0)
  +far, -far an exact draw given z = +-1.5 sigma_1 v_1 (sigma_1 v_1: the leading singular
             direction of the pair vectors; as importance_log_z's chains 2 and 3)
and run for --sweeps sweeps.  At each chain's end, z ~ N(s(S), I) is drawn and its z-marginal
weight log w = log Z_z - |z|^2 / 2 is computed (the z-marginal is q(z) ~ N(z; 0, I) Z_z, so a
region with higher log w holds more of the model's probability).

Per start: mean basket size, share of trips at the size cap, mean log w.  Per trip: the gap
between the largest log w over all four chain ends and the log w at the end of the chain from
the observed basket.  A gap of many nats means the model puts most of its probability in a
region the chain from the data does not reach (the training chains of SML start there).

    python bench/modes.py --world real [--trips 64] [--sweeps 200] [--seed 3]
"""
from __future__ import annotations

import numpy as np
import torch

from common import batch_of, emit, load, parser
from v4 import exact
from v4.sampling import draw_from_model, gibbs, sample_baskets


def main():
    p = parser(__doc__)
    p.add_argument("--trips", type=int, default=64)
    p.add_argument("--sweeps", type=int, default=200)
    p.add_argument("--seed", type=int, default=3)
    a = p.parse_args()
    model, batcher, held = load(a)
    b = batch_of(model, batcher, held[:a.trips])
    ix, iv = b.ix, model.iv_shift(b.ix)
    gen = torch.Generator().manual_seed(a.seed)
    nmax = int(model.nmax)

    def size(X):
        return torch.zeros(ix.B).index_add_(0, ix.item_trip, X.double())

    with torch.no_grad():
        _, sv, Vt = torch.linalg.svd(model.phi_r().detach(), full_matrices=False)
        starts = {"observed": b.observed.clone(), "beta0": draw_from_model(model, ix, gen, 0)}
        for sign, name in ((1.0, "+far"), (-1.0, "-far")):
            z0 = (sign * 1.5 * sv[0] * Vt[0])[None, :].expand(ix.B, -1).contiguous()
            starts[name] = sample_baskets(model, ix, model.slot_utilities_given_z(ix, z0), gen, z=z0, iv=iv)
        logw = {}
        for name, X in starts.items():
            X = gibbs(model, ix, X, gen, sweeps=a.sweeps, iv=iv)
            z = model.basket_field(ix, X) + torch.randn(ix.B, model.rank, generator=gen)
            lz = exact.log_partition(model, ix, model.slot_utilities_given_z(ix, z), z=z, iv=iv)
            logw[name] = (lz - 0.5 * z.square().sum(1)).numpy()
            n = size(X)
            emit(start=name, sweeps=a.sweeps, mean_size=round(float(n.mean()), 2),
                 share_at_cap=round(float((n >= nmax).double().mean()), 3),
                 mean_log_weight=round(float(logw[name].mean()), 2))
    best = np.max(np.stack(list(logw.values())), 0)
    gap = best - logw["observed"]
    emit(trips=int(ix.B), sigma1_squared=round(float(sv[0] ** 2), 1),
         gap_median=round(float(np.median(gap)), 2), gap_max=round(float(gap.max()), 2),
         trips_gap_over_5=int((gap > 5).sum()), trips_gap_over_20=int((gap > 20).sum()),
         note="gap = largest log weight over the four chain ends minus the observed-start chain's")


if __name__ == "__main__":
    main()
