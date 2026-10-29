"""Generated basket size with the taste z drawn three ways (docs/training_theory.pdf, Section 6):

    prior      z ~ N(0, I)                          (the model's generation)
    posterior  z ~ p(z | S_t) of the trip itself    (importance draws resampled by weight)
    pooled     z ~ N(mean, cov) of the pooled posterior

    ENERGY_MODEL_DATA_ROOT=<bundle> python3 -u research/size_by_taste.py CK1.pt [CK2.pt ...]

At a maximum-likelihood point the size averaged over each trip's posterior matches the data; the
prior's size matches only if the pooled posterior equals N(0, I).  Trips: --trips training trips
(the 1,024 that research/score_checkpoints.py scores as "training").  One JSON line per checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from v4 import mixture  # noqa: E402
from v4.bundle import data_root, load_bundle, supported_trips  # noqa: E402
from v4.checkpoints import load_version4  # noqa: E402


def n_categories(ix, X):
    """Number of categories present in each trip's basket."""
    n_row = torch.zeros(ix.n_rows).index_add_(0, ix.row_of, X.double())
    return torch.zeros(ix.B).index_add_(0, ix.row_trip, (n_row > 0).double())


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--trips", type=int, default=512)
    p.add_argument("--K", type=int, default=64)
    p.add_argument("--draws", type=int, default=4, help="baskets per trip and way")
    p.add_argument("--seed", type=int, default=41001)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    root = data_root(None)
    data = load_bundle(root)
    for ck in a.checkpoints:
        model, batcher, blob, _ = load_version4(ck, data, root)
        for q in model.parameters():
            q.requires_grad_(False)
        held = np.random.default_rng(a.seed + 99).permutation(supported_trips(data, 0, int(model.nmax)))
        trips = held[1024:1024 + a.trips]
        size = {"prior": [], "posterior": [], "pooled": []}
        cats = {"prior": [], "posterior": [], "pooled": []}
        obs, obs_k, post_z = [], [], []
        batches = []
        for b in range(0, len(trips), 256):
            bt = batcher.make(np.sort(trips[b:b + 256]))
            bt.bind(model)
            g = torch.Generator().manual_seed(700 + b)
            _, _, info = mixture.log_lik(model, bt, g, K=a.K)
            pick = torch.multinomial(info["w"].T, a.draws, replacement=True, generator=g)     # (B, draws)
            zs = info["zs"].permute(1, 0, 2)                                                  # (B, K, r)
            post_z.append(torch.stack([zs[torch.arange(bt.ix.B), pick[:, d]] for d in range(a.draws)]))
            obs.append(torch.zeros(bt.ix.B).index_add_(0, bt.ix.item_trip, bt.observed.double()))
            obs_k.append(n_categories(bt.ix, bt.observed))
            batches.append(bt)
        Z = torch.cat(post_z, 1)                                                   # (draws, trips, r)
        flat = Z.reshape(-1, Z.shape[-1])
        mu, cov = flat.mean(0), torch.cov(flat.T)
        L = torch.linalg.cholesky(cov)
        start = 0
        for bt in batches:
            bt.bind(model)                   # binding sets per-batch state: bind before every use
            B = bt.ix.B
            g = torch.Generator().manual_seed(900 + start)
            iv = model.iv_shift(bt.ix)
            for d in range(a.draws):
                for way, z in (("prior", torch.randn(B, model.rank, generator=g)),
                               ("posterior", Z[d, start:start + B]),
                               ("pooled", mu + torch.randn(B, model.rank, generator=g) @ L.T)):
                    X = mixture.generate(model, bt.ix, g, iv=iv, z=z)
                    size[way].append(torch.zeros(B).index_add_(0, bt.ix.item_trip, X.double()))
                    cats[way].append(n_categories(bt.ix, X))
            start += B
        ev = torch.linalg.eigvalsh(cov)
        print(json.dumps({"checkpoint": str(ck), "data_size": round(float(torch.cat(obs).mean()), 3),
                          **{f"gen_{k}": round(float(torch.cat(v).mean()), 3) for k, v in size.items()},
                          "data_categories": round(float(torch.cat(obs_k).mean()), 3),
                          **{f"gen_categories_{k}": round(float(torch.cat(v).mean()), 3) for k, v in cats.items()},
                          "pooled_mean_norm": round(float(mu.norm()), 3),
                          "pooled_cov_eig": [round(float(ev.min()), 3), round(float(ev.max()), 3)]}), flush=True)


if __name__ == "__main__":
    main()
