"""Checks that a computation still gives the same law, on held-out trips of a fitted model.

  draws    exact basket draws given z (v4.sampling.sample_baskets) against the exact incidence
           (v4.exact.incidence): mean basket size, largest per-trip size gap, largest
           item-frequency gap.  Run it with z near the observed basket and with z taken from
           the end of long Gibbs chains (where z can be far from typical).
  chain    block-Gibbs chains on (z, S) started at the observed baskets: mean and largest basket
           size and the number of trips at the item cap after 0, 10, 50, 100, 200, 400 sweeps
           (the check that caught the uncapped variant running away to ~2,660 items).
  logz     exact log Z at the same (trip, z) pairs, written to --save (.npz) so two code
           versions can be compared trip by trip with  bench/compare_logz.py.
  ll       log-likelihood of held-out baskets at draws 1024 (burn 200, components 128, thin 4),
           saved per basket to --save for a paired comparison (bench/compare_ll.py).

    python bench/law_checks.py --world syn --check chain [--trips 64] [--seed 9]
"""
from __future__ import annotations

import os

import numpy as np
import torch

from common import batch_of, emit, load, parser
from v4 import exact
from v4.sampling import gibbs, importance_log_z, sample_baskets


def main():
    p = parser(__doc__)
    p.add_argument("--check", choices=("draws", "chain", "logz", "ll"), required=True)
    p.add_argument("--trips", type=int, default=64)
    p.add_argument("--seed", type=int, default=9)
    p.add_argument("--draws", type=int, default=400, help="draws: baskets per trip")
    p.add_argument("--sweeps", type=int, nargs="+", default=[0, 10, 50, 100, 200, 400])
    p.add_argument("--save", help="logz / ll: .npz to write")
    a = p.parse_args()
    if a.save:
        a.save = os.path.abspath(a.save)               # load() changes directory
    model, batcher, held = load(a)
    b = batch_of(model, batcher, held[:a.trips])
    ix, iv = b.ix, model.iv_shift(b.ix)
    gen = torch.Generator().manual_seed(a.seed)
    nmax = int(model.nmax)

    def sizes(X):
        return torch.zeros(ix.B).index_add_(0, ix.item_trip, X.double())

    with torch.no_grad():
        if a.check == "chain":
            X, done = b.observed.clone(), 0
            for target in sorted(a.sweeps):
                if target > done:
                    X = gibbs(model, ix, X, gen, sweeps=target - done, iv=iv)
                    done = target
                n = sizes(X)
                emit(check="chain", seed=a.seed, sweep=target, mean_size=round(float(n.mean()), 3),
                     max_size=int(n.max()), trips_at_cap=int((n >= nmax).sum()), trips_over_cap=int((n > nmax).sum()))
            return
        zs = {"observed": model.basket_field(ix, b.observed) + torch.randn(ix.B, model.rank, generator=gen)}
        X = gibbs(model, ix, b.observed.clone(), gen, sweeps=200, iv=iv)
        zs["chain200"] = model.basket_field(ix, X) + torch.randn(ix.B, model.rank, generator=gen)
        if a.check == "draws":
            for name, z in zs.items():
                u = model.slot_utilities_given_z(ix, z)
                with torch.enable_grad():
                    _, p_ex = exact.incidence(model, ix, u, z=z, iv=iv)
                acc = torch.zeros(len(ix.item))
                for _ in range(a.draws):
                    acc += sample_baskets(model, ix, u, gen, z=z, iv=iv).double()
                freq = acc / a.draws
                emit(check="draws", z=name, draws_per_trip=a.draws,
                     exact_mean_size=round(float(sizes(p_ex).mean()), 3), drawn_mean_size=round(float(sizes(freq).mean()), 3),
                     max_trip_size_gap=round(float((sizes(freq) - sizes(p_ex)).abs().max()), 3),
                     max_item_freq_gap=round(float((freq - p_ex).abs().max()), 3),
                     note="sd of one frequency <= 0.5/sqrt(draws)")
        elif a.check == "logz":
            out = {}
            for name, z in zs.items():
                out[name] = exact.log_partition(model, ix, model.slot_utilities_given_z(ix, z), z=z, iv=iv).numpy()
                emit(check="logz", z=name, mean=float(out[name].mean()))
            np.savez(a.save, trips=b.trips, **out)
            emit(saved=a.save)
        else:
            est, half, ess, _ = importance_log_z(model, ix, gen, burn=200, components=128, thin=4, draws=1024)
            ll = (model.energy(b) - est).numpy()
            np.savez(a.save, trips=b.trips, ll=ll, half=half.numpy(), ess=ess.numpy())
            emit(check="ll", mean_ll=float(ll.mean()), half_split_abs_mean=float(half.abs().mean()),
                 ess_mean=float(ess.mean()), saved=a.save)


if __name__ == "__main__":
    main()
