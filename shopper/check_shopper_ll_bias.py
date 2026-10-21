#!/usr/bin/env python3
"""How far is SHOPPER's reported log-likelihood from the truth because of its ordering estimate?

log of an importance-sampling mean is biased low.  On a random subset of the reference trips,
rescore with many more orderings and report the mean shift (reference estimate minus the
high-ordering estimate), by basket size, plus the agreement of two independent high-ordering
estimates (how converged they are).  The fit itself is not changed.
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from evaluate_shopper import load  # noqa: E402
from shopper_data import BundleSource  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--per-trip", type=Path, required=True, help="the evaluation's per-trip file (256 orderings)")
p.add_argument("--trips", type=int, default=256)
p.add_argument("--perms", type=int, default=8192)
p.add_argument("--chunk", type=int, default=4)
p.add_argument("--threads", type=int, default=4)
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
torch.set_num_threads(a.threads)
torch.set_default_dtype(torch.float64)
model, _ = load(a.checkpoint)
model.double()
cfg = torch.load(a.checkpoint, map_location="cpu", weights_only=False)["config"]
src = BundleSource(price_contract=cfg.get("price_contract"), discount_panel=cfg.get("discount_panel"))
ref = np.load(a.per_trip)
pick = np.sort(np.random.default_rng(11).choice(len(ref["trips"]), a.trips, replace=False))
trips, ll_ref = ref["trips"][pick], ref["ll"][pick]
hi1, hi2, sizes = [], [], []
for s in range(0, len(trips), a.chunk):
    t = src.make(trips[s:s + a.chunk])
    h1 = model.basket_logprob_is(t, perms=a.perms // 2, gen=torch.Generator().manual_seed(70000 + s))
    h2 = model.basket_logprob_is(t, perms=a.perms // 2, gen=torch.Generator().manual_seed(90000 + s))
    hi1.append(h1.numpy()); hi2.append(h2.numpy()); sizes.append(t.size.numpy())
    if (s // a.chunk) % 8 == 0:
        print(f"{s + len(t.size)}/{len(trips)}", flush=True)
hi1, hi2, sizes = map(np.concatenate, (hi1, hi2, sizes))
hi = np.logaddexp(hi1, hi2) - math.log(2)
shift = ll_ref - hi
res = {"trips": int(len(trips)), "reference_perms": 256, "high_perms": a.perms,
       "mean_ll_reference": float(ll_ref.mean()), "mean_ll_high": float(hi.mean()),
       "mean_shift_reference_minus_high": float(shift.mean()),
       "shift_se": float(shift.std(ddof=1) / math.sqrt(len(shift))),
       "high_half_split_abs_mean": float(np.abs(hi1 - hi2).mean()),
       "by_size": {}}
for lo, hi_ in ((1, 5), (6, 10), (11, 20), (21, 200)):
    m = (sizes >= lo) & (sizes <= hi_)
    if m.any():
        res["by_size"][f"{lo}-{hi_}"] = {"trips": int(m.sum()), "mean_shift": float(shift[m].mean()),
                                         "high_half_split_abs_mean": float(np.abs(hi1 - hi2)[m].mean())}
a.output.write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
