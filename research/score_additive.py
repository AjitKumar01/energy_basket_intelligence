"""Exact log p(S) of the additive base model (phi = 0) on the same 1,024 trips that
research/score_checkpoints.py scores, so the gain of the later stages is a paired difference.

    ENERGY_MODEL_DATA_ROOT=<bundle> python3 -u research/score_additive.py --initialization INIT.pt \
        --additive ADD.pt --like MIXTURE.pt --split heldout --out OUT.pt [--compare SCORES.pt]

log p(S) = E(S) - log Z with log Z = logsumexp over sizes of the exact dynamic program
(v4/exact.additive_log_size).  The trips are taken with the nmax of --like, as the mixture scores
were.  --compare: a file written by score_checkpoints.py on the same split; prints the paired
difference of each of its checkpoints against the additive model.
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

from v4.bundle import data_root, load_bundle, supported_trips  # noqa: E402
from v4.checkpoints import batcher_for, freeze, load_additive, load_version4  # noqa: E402
from v4.exact import additive_log_size  # noqa: E402


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--initialization", type=Path, required=True)
    p.add_argument("--additive", type=Path, required=True)
    p.add_argument("--like", type=Path, required=True, help="a mixture checkpoint: its nmax fixes the trips")
    p.add_argument("--split", choices=("heldout", "training"), required=True)
    p.add_argument("--trips", type=int, default=1024)
    p.add_argument("--seed", type=int, default=41001)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--compare", type=Path)
    a = p.parse_args()
    root = data_root(None)
    data = load_bundle(root)
    nmax = int(load_version4(a.like, data, root)[0].nmax)
    model, meta, blob = load_additive(a.initialization, a.additive, data, root)
    freeze(model)
    batcher = batcher_for(data, root, blob)
    held = np.random.default_rng(a.seed + 99).permutation(supported_trips(data, 0, nmax))
    trips = held[:a.trips] if a.split == "heldout" else held[a.trips:2 * a.trips]
    lls, order = [], []
    for b in range(0, len(trips), 256):
        chunk = np.sort(trips[b:b + 256])
        bt = batcher.make(chunk)
        bt.bind(model)
        ll = model.energy(bt) - torch.logsumexp(additive_log_size(model, bt.ix), -1)
        lls.append(ll)
        order.append(chunk)
    ll = torch.cat(lls)
    trip_order = np.concatenate(order)
    print(json.dumps({"additive": str(a.additive), "split": a.split, "nmax": nmax, "model_nmax": int(model.nmax),
                      "ll": round(float(ll.mean()), 4), "se": round(float(ll.std() / len(ll) ** 0.5), 3)}), flush=True)
    torch.save({"trips": trip_order, "ll": ll.tolist()}, a.out)
    if a.compare is not None:
        other = torch.load(a.compare, weights_only=False)
        for ck, r in other.items():
            assert np.array_equal(np.asarray(r["trips"]), trip_order), "different trips"
            d = torch.tensor(r["ll"]) - ll
            print(json.dumps({"diff": f"{Path(ck).name} - additive", "mean": round(float(d.mean()), 4),
                              "paired_se": round(float(d.std() / len(d) ** 0.5), 4)}), flush=True)


if __name__ == "__main__":
    main()
