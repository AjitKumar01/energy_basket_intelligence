"""Block swap: copies of checkpoint A with one block of parameters taken from checkpoint B.

    python3 research/swap_blocks.py --base A.pt --donor B.pt --out-dir DIR

Writes DIR/<A>_with_<B>_<block>.pt for each block below.  Score them together with A and B with
research/score_checkpoints.py: if putting a block back to B's values gains on held-out trips and
loses on training trips, A's changes to that block since B only memorised the training trips.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

BLOCKS = {
    "household": ["theta", "gamma"],
    "product_context": ["alpha", "beta", "mu", "zeta", "w_dsp", "w_mlr", "psi", "psi_item", "u_cat",
                        "dsp_common", "mlr_common", "delta", "xi", "price_kappa", "disc_kappa", "disc_common",
                        "hist_size"],
    "interaction": ["phi", "psi_cat", "lam"],
    "size": ["rho_0_free", "rho_c", "f_extra"],
}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--donor", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    base = torch.load(a.base, map_location="cpu", weights_only=False)
    donor = torch.load(a.donor, map_location="cpu", weights_only=False)["model"]
    for name, keys in BLOCKS.items():
        blob = dict(base)
        blob["model"] = {k: v.clone() for k, v in base["model"].items()}
        hit = [k for k in keys if k in blob["model"] and k in donor]
        for k in hit:
            blob["model"][k] = donor[k].clone()
        blob["stage"] = f"{base.get('stage')} with {name} from {a.donor.name}"
        out = a.out_dir / f"{a.base.stem}_with_{a.donor.stem}_{name}.pt"
        torch.save(blob, out)
        print(out, hit)


if __name__ == "__main__":
    main()
