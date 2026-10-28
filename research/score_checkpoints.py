"""Score mixture checkpoints on 1,024 trips of the fit's held-out split or of its training split.

    ENERGY_MODEL_DATA_ROOT=<bundle> python3 research/score_checkpoints.py --split heldout \
        --out scores_heldout.pt CK1.pt CK2.pt ...

Per checkpoint: log p(S) per trip (K = 256 importance draws, a lower bound in expectation), its mean
and standard error, the importance ESS, and the size of 4 exact draws per trip against the data.
Every checkpoint uses the same trips and the same random numbers, so the differences printed at the
end are paired, trip by trip.  The trips are those of scripts/fit_mixture.py with its default seed:
held-out = the first 1,024 of the seeded order (its validation trips), training = the next 1,024.
Writes OUT (torch): {checkpoint: {"stage", "trips", "ll"}}.  One JSON line per result on stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from v4 import mixture  # noqa: E402
from v4.bundle import data_root, load_bundle, supported_trips  # noqa: E402
from v4.checkpoints import load_version4  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoints", nargs="+", type=Path)
    p.add_argument("--split", choices=("heldout", "training"), required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trips", type=int, default=1024)
    p.add_argument("--K", type=int, default=256)
    p.add_argument("--seed", type=int, default=41001, help="the fit's --seed (fixes its held-out split)")
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    root = data_root(None)
    data = load_bundle(root)
    res = {}
    for ck in a.checkpoints:
        t0 = time.time()
        model, batcher, blob, _ = load_version4(ck, data, root)
        for q in model.parameters():
            q.requires_grad_(False)
        nmax = int(model.nmax)
        held = np.random.default_rng(a.seed + 99).permutation(supported_trips(data, 0, nmax))
        trips = held[:a.trips] if a.split == "heldout" else held[a.trips:2 * a.trips]
        lls, esss, gen_n, obs_n, order = [], [], [], [], []
        for b in range(0, len(trips), 256):
            chunk = np.sort(trips[b:b + 256])
            order.append(chunk)
            bt = batcher.make(chunk)
            bt.bind(model)
            ll, ess, _ = mixture.log_lik(model, bt, torch.Generator().manual_seed(500 + b // 256), K=a.K)
            lls.append(ll)
            esss.append(ess)
            obs_n.append(torch.zeros(bt.ix.B).index_add_(0, bt.ix.item_trip, bt.observed.double()))
            g = torch.Generator().manual_seed(600 + b // 256)
            for _ in range(4):
                X = mixture.generate(model, bt.ix, g, iv=model.iv_shift(bt.ix))
                gen_n.append(torch.zeros(bt.ix.B).index_add_(0, bt.ix.item_trip, X.double()))
        ll, ess, gn, on = torch.cat(lls), torch.cat(esss), torch.cat(gen_n), torch.cat(obs_n)
        res[str(ck)] = {"stage": blob.get("stage"), "trips": np.concatenate(order), "ll": ll.tolist()}
        print(json.dumps({"checkpoint": str(ck), "split": a.split, "stage": blob.get("stage"),
                          "ll": round(float(ll.mean()), 4), "se": round(float(ll.std() / len(ll) ** 0.5), 3),
                          "ess_mean": round(float(ess.mean()), 3), "ess_min": round(float(ess.min()), 4),
                          "gen_size": round(float(gn.mean()), 3), "data_size": round(float(on.mean()), 3),
                          "gen_q99": float(gn.quantile(0.99)), "data_q99": float(on.quantile(0.99)),
                          "gen_at_cap": int((gn >= nmax).sum()), "seconds": round(time.time() - t0)}), flush=True)
    names = list(res)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            d = torch.tensor(res[names[j]]["ll"]) - torch.tensor(res[names[i]]["ll"])
            print(json.dumps({"diff": f"{Path(names[j]).name} - {Path(names[i]).name}",
                              "mean": round(float(d.mean()), 4), "paired_se": round(float(d.std() / len(d) ** 0.5), 4)}),
                  flush=True)
    torch.save(res, a.out)


if __name__ == "__main__":
    main()
