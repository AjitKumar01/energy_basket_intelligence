"""The EM update of the prior covariance of z, applied after training with every other parameter fixed
(docs/training_theory.pdf, Section 7).

    ENERGY_MODEL_DATA_ROOT=<bundle> python3 -u research/final_whiten.py CK.pt --out CK_white.pt --iters 3

Each iteration estimates the pooled posterior second moment M on --trips training trips (K = --K
draws) and applies mixture.whiten_ (phi, psi <- M^(1/2) phi, M^(1/2) psi, with lam compensated):
the model with prior N(0, I) and the new vectors is the old model with prior N(0, M), apart from the
|psi_c|^2 / 2 per extra product.  Training trips only (the fit's split: held-out = the first
--val-trips of the seeded order).  One JSON line per iteration.
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


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--trips", type=int, default=512)
    p.add_argument("--K", type=int, default=256)
    p.add_argument("--val-trips", type=int, default=1024)
    p.add_argument("--seed", type=int, default=41001)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    root = data_root(None)
    data = load_bundle(root)
    model, batcher, blob, _ = load_version4(a.checkpoint, data, root)
    for q in model.parameters():
        q.requires_grad_(False)
    held = np.random.default_rng(a.seed + 99).permutation(supported_trips(data, 0, int(model.nmax)))
    train = held[a.val_trips:]
    # Trips after the 1,024 that research/score_checkpoints.py scores as "training", so they are not reused.
    trips = np.sort(train[1024:1024 + a.trips])
    for it in range(a.iters):
        M, n = 0.0, 0
        for b in range(0, len(trips), 256):
            bt = batcher.make(trips[b:b + 256])
            bt.bind(model)
            _, _, info = mixture.log_lik(model, bt, torch.Generator().manual_seed(a.seed + 31 + b), K=a.K)
            M = M + torch.einsum("kb,kbi,kbj->ij", info["w"], info["zs"], info["zs"])
            n += bt.ix.B
        M = M / n
        with torch.no_grad():
            eig = mixture.whiten_(model, M)
        print(json.dumps({"iter": it, "trace_over_r": round(float(eig.sum()) / model.rank, 4),
                          "eig_max": round(float(eig[0]), 4), "eig_min": round(float(eig[-1]), 4)}), flush=True)
    out = dict(blob)
    out["model"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    out["stage"] = f"{blob.get('stage')} + final whitening ({a.iters} EM steps of the prior covariance)"
    torch.save(out, a.out)
    print(a.out, flush=True)


if __name__ == "__main__":
    main()
