#!/usr/bin/env python3
"""Run 3-seed classifier two-sample AUC evaluation (full basket and category presence)
on all supported test trips for a given Version-4 checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from pathlib import Path

sys.modules.setdefault("pathlib._local", pathlib)
os.environ.setdefault("V3_AFFINITY", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/version4"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import stress  # noqa: E402
from stress import Batch, classifier_auc, gibbs, sample_slots, setup, slot_utilities  # noqa: E402
from pipeline_support import supported_trips  # noqa: E402

torch.set_default_dtype(torch.float64)


def main():
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--initialization", type=Path, default=A / "artifacts/initialization.pt")
    p.add_argument("--additive", type=Path, default=A / "out/v3_pipeline_additive_best.pt")
    p.add_argument("--checkpoint", type=Path, default=ROOT / "runs/stress/hs_conv.pt")
    p.add_argument("--seeds", type=int, nargs="+", default=[41006, 41007, 41008])
    p.add_argument("--burn", type=int, default=200)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, default=ROOT / "runs/stress/hs_conv_3seed_auc.json")
    args = p.parse_args()

    stress.USE_CPP_SAMPLER = True
    data, model, meta, batcher = setup(args.initialization, args.additive, args.threads)
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if hasattr(stress, "load_checkpoint_into_model"):
        stress.load_checkpoint_into_model(model, blob)
    else:
        model.load_state_dict(blob["model"])
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    r = int(blob["rank"])
    J, C, nmax = int(data["n_item"]), int(data["n_cat"]), int(meta["nmax"])
    item_cat = np.zeros(J, dtype=np.int64)
    item_cat[data["line_item"]] = data["line_cat"]

    test = supported_trips(data, 2, nmax)
    lptr = data["line_ptr"]
    real = [np.sort(data["line_item"][lptr[t]:lptr[t + 1]]) for t in test]
    groups = np.r_[data["trip_user"][test], data["trip_user"][test]]
    test_batches = [Batch(batcher, test[s:s + args.eval_batch], J) for s in range(0, len(test), args.eval_batch)]

    seed_results = []
    for sd in args.seeds:
        ts = time.time()
        gen = torch.Generator().manual_seed(sd)
        generated = []
        with torch.no_grad():
            for bt in test_batches:
                bt.bind(model)
                X = sample_slots(model, bt.ix, slot_utilities(model, bt.ix, None, r, beta=0.0), gen)
                X = gibbs(model, bt.ix, X, r, gen, sweeps=args.burn).numpy()
                items, xt = bt.ix.item.numpy(), bt.ix.item_trip.numpy()
                generated += [np.sort(items[X & (xt == b)]) for b in range(len(bt.trips))]
        full_m, full_sd = classifier_auc(real, generated, groups, J, item_cat, C, "full")
        cat_m, cat_sd = classifier_auc(real, generated, groups, J, item_cat, C, "category")
        row = {
            "seed": int(sd),
            "full_basket_auc": round(full_m, 4),
            "full_basket_fold_sd": round(full_sd, 4),
            "category_presence_auc": round(cat_m, 4),
            "category_presence_fold_sd": round(cat_sd, 4),
            "size_mean_real": round(float(np.mean([len(x) for x in real])), 4),
            "size_mean_generated": round(float(np.mean([len(x) for x in generated])), 4),
            "seconds": round(time.time() - ts, 1),
        }
        print(json.dumps(row), flush=True)
        seed_results.append(row)

    summary = {
        "checkpoint": str(args.checkpoint),
        "test_trips": int(len(test)),
        "burn": args.burn,
        "seeds": seed_results,
        "full_basket_auc_mean": round(float(np.mean([x["full_basket_auc"] for x in seed_results])), 4),
        "full_basket_auc_sd": round(float(np.std([x["full_basket_auc"] for x in seed_results], ddof=1)), 4),
        "category_presence_auc_mean": round(float(np.mean([x["category_presence_auc"] for x in seed_results])), 4),
        "category_presence_auc_sd": round(float(np.std([x["category_presence_auc"] for x in seed_results], ddof=1)), 4),
        "size_mean_generated_mean": round(float(np.mean([x["size_mean_generated"] for x in seed_results])), 4),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print("SUMMARY:", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
