"""Timings of the three costly operations on held-out trips (median of repeats):

  logz        exact log Z given z for a batch of trips (v4.exact.log_partition)
  sweep       one block-Gibbs sweep for a batch of trips (v4.sampling.gibbs)
  likelihood  log-likelihood of baskets by importance sampling (v4.sampling.importance_log_z),
              one basket at a time and a whole batch, at the validation setting
              (burn 80, components 32, thin 2, draws 128)

    python bench/timing.py --world syn [--trips 256] [--one 8] [--quick]
--quick: 32 trips, 2 single baskets, fewer repeats (a smoke test, under a minute on syn).
"""
from __future__ import annotations

import time

import numpy as np
import torch

from common import batch_of, emit, load, parser
from v4.exact import log_partition
from v4.sampling import draw_from_model, gibbs, importance_log_z

VAL = dict(burn=80, components=32, thin=2, draws=128)


def median_time(fn, reps):
    ts = []
    for _ in range(reps):
        t = time.time()
        fn()
        ts.append(time.time() - t)
    return float(np.median(ts))


def main():
    p = parser(__doc__)
    p.add_argument("--trips", type=int, default=256)
    p.add_argument("--one", type=int, default=8, help="single baskets to time")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if a.quick:
        a.trips, a.one = 32, 2
    reps = 2 if a.quick else 5
    model, batcher, held = load(a)
    with torch.no_grad():
        b = batch_of(model, batcher, held[:a.trips])
        ix, iv = b.ix, model.iv_shift(b.ix)
        emit(trips=ix.B, offered_products_per_trip=round(len(ix.item) / ix.B), categories=int(ix.Cpad))
        gen = torch.Generator().manual_seed(1)
        z = model.basket_field(ix, b.observed) + torch.randn(ix.B, model.rank, generator=gen)
        u = model.slot_utilities_given_z(ix, z)
        emit(part="logz", trips=ix.B, seconds=round(median_time(lambda: log_partition(model, ix, u, z=z, iv=iv), reps), 4))
        X = draw_from_model(model, ix, gen, 10)
        sweeps = 2 if a.quick else 10
        t = median_time(lambda: gibbs(model, ix, X, gen, sweeps=sweeps, iv=iv), 3)
        emit(part="sweep", trips=ix.B, seconds_per_sweep=round(t / sweeps, 4))
        one = []
        for i in range(a.one):
            bi = batch_of(model, batcher, held[a.trips + i:a.trips + i + 1])
            t = time.time()
            importance_log_z(model, bi.ix, torch.Generator().manual_seed(10 + i), **VAL)
            one.append(time.time() - t)
        b.bind(model)
        t = time.time()
        importance_log_z(model, ix, torch.Generator().manual_seed(20), **VAL)
        t_batch = time.time() - t
        emit(part="likelihood", setting="burn 80, components 32, thin 2, draws 128",
             seconds_one_basket_median=round(float(np.median(one)), 3), seconds_one_basket_min=round(min(one), 3),
             seconds_one_basket_max=round(max(one), 3), batch=ix.B, seconds_per_basket_in_batch=round(t_batch / ix.B, 4))
    emit(done=True)


if __name__ == "__main__":
    main()
