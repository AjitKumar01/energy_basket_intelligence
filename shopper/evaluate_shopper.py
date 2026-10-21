#!/usr/bin/env python3
"""Score a SHOPPER fit exactly as scripts/evaluate.py scores the energy model (stress world or Dunnhumby).

1. Held-out basket log-likelihood on the certified reference trips (the 4,096 test trips the
   energy models use), log P(S) by importance sampling over orderings with a half-split check,
   paired per trip with an energy-model run's per-trip log-likelihoods (e.g. hs_rec).
2. Classifier two-sample AUCs (full basket, category presence) on all supported test trips,
   one generated basket per trip per seed (generation capped at the energy model's nmax).
3. Stress world only (--world): the 32 single-product price scenarios against the world's
   causal truth, on the same 400 test trips: incidences P(j in S) by Monte Carlo with common
   random numbers.

The bundle is $ENERGY_MODEL_DATA_ROOT; the price contract is the one the fit used.  The
protocol (docs/SHOPPER_BASELINE_PROTOCOL.md) needs at least 8,192 orderings for log P(S).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from shopper import Shopper, Trips  # noqa: E402
from shopper_data import BundleSource  # noqa: E402


def load(ckpt):
    b = torch.load(ckpt, map_location="cpu", weights_only=False)
    c = b["config"]
    m = Shopper(b["n_item"], b["n_user"], b["n_week"], b["item_group"], K=c["K"], P=c["P"], D=c["D"])
    m.load_state_dict(b["model"])
    m.eval()
    for q in m.parameters():
        q.requires_grad_(False)
    return m, b


def rep(t, R):
    return Trips(*(getattr(t, f).repeat_interleave(R, 0) for f in Trips.__dataclass_fields__))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--reference", type=Path, required=True, help="reference trips (.npz with trips, household)")
    p.add_argument("--paired", type=Path, help="an energy model's per-trip log-likelihoods on the same trips")
    p.add_argument("--world", type=Path, help="synthetic world with counterfactual truth (price scenarios)")
    p.add_argument("--perms", type=int, default=8192, help="orderings per basket for log P(S)")
    p.add_argument("--auc-seeds", type=int, nargs="+", default=[41006, 41007, 41008])
    p.add_argument("--scenario-trips", type=int, default=400)
    p.add_argument("--scenario-draws", type=int, default=256)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--skip", nargs="*", default=[])
    p.add_argument("--placebo", action="store_true",
                   help="price scenarios only: leave every price unchanged and redraw with other seeds, "
                        "so the reported 'model' changes are pure Monte Carlo noise")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_default_dtype(torch.float64)
    model, blob = load(a.checkpoint)
    model.double()
    if a.world is None and "price" not in a.skip:
        a.skip.append("price")                       # no causal truth on real data
    cfg = blob["config"]
    src = BundleSource(price_contract=cfg.get("price_contract"), discount_panel=cfg.get("discount_panel"))
    data = src.data
    J = src.n_item
    report = {"checkpoint": str(a.checkpoint), "config": {k: str(v) for k, v in blob["config"].items()},
              "best_val_basket_ll": blob["best_val_basket_ll"]}
    t0 = time.time()
    from v4.reporting import classifier_auc, clustered_ci

    if "ll" not in a.skip:
        ref = np.load(a.reference)
        trips = ref["trips"].astype(np.int64)
        ll, half = [], []
        for s in range(0, len(trips), a.chunk):
            t = src.make(trips[s:s + a.chunk])
            g1 = torch.Generator().manual_seed(1000 + s)
            g2 = torch.Generator().manual_seed(5000 + s)
            h1 = model.basket_logprob_is(t, perms=a.perms // 2, gen=g1)
            h2 = model.basket_logprob_is(t, perms=a.perms // 2, gen=g2)
            ll.append((torch.logaddexp(h1, h2) - math.log(2)).numpy())
            half.append((h1 - h2).numpy())
        ll, half = np.concatenate(ll), np.concatenate(half)
        rep_ll = {"trips": int(len(ll)), "mean_ll": float(ll.mean()),
                  "se": float(ll.std(ddof=1) / math.sqrt(len(ll))),
                  "half_split_abs_mean": float(np.abs(half).mean()), "half_split_abs_max": float(np.abs(half).max())}
        if a.paired is not None and a.paired.is_file():
            other = np.load(a.paired)["ll"][:len(ll)]
            d = ll - other
            rep_ll["paired_with"] = str(a.paired)
            rep_ll["paired_mean_ll"] = float(other.mean())
            rep_ll["shopper_minus_paired"] = float(d.mean())
            rep_ll["shopper_minus_paired_household_95"] = clustered_ci(d, ref["household"][:len(ll)])
        report["likelihood"] = rep_ll
        np.savez(a.output.with_suffix(".per_trip.npz"), trips=trips, ll=ll, half_diff=half)
        print(json.dumps(rep_ll), flush=True)

    nmax = int(np.asarray(data["trip_nlines"])[src.split(0)].max())
    if "auc" not in a.skip:
        test = src.split(2)
        test = test[np.asarray(data["trip_nlines"])[test] <= nmax]
        lptr = data["line_ptr"]
        real = [np.sort(data["line_item"][lptr[t]:lptr[t + 1]]) for t in test]
        groups = np.r_[data["trip_user"][test], data["trip_user"][test]]
        C = int(data["n_cat"])
        seeds = []
        for sd in a.auc_seeds:
            gen_b = []
            for s in range(0, len(test), a.chunk):
                t = src.make(test[s:s + a.chunk])
                X = model.sample(t, seed=sd * 100003 + s, max_size=nmax)
                gen_b += [np.sort(t.offered[b][X[b]].numpy()) for b in range(len(X))]
            full_m, _ = classifier_auc(real, gen_b, groups, J, src.item_cat, C, "full")
            cat_m, _ = classifier_auc(real, gen_b, groups, J, src.item_cat, C, "category")
            seeds.append({"seed": sd, "full_basket_auc": round(full_m, 4), "category_presence_auc": round(cat_m, 4),
                          "size_mean_real": round(float(np.mean([len(x) for x in real])), 4),
                          "size_mean_generated": round(float(np.mean([len(x) for x in gen_b])), 4)})
            print(json.dumps(seeds[-1]), flush=True)
        report["classifier"] = {
            "seeds": seeds, "test_trips": int(len(test)), "max_size": nmax,
            "full_basket_auc_mean": float(np.mean([x["full_basket_auc"] for x in seeds])),
            "category_presence_auc_mean": float(np.mean([x["category_presence_auc"] for x in seeds])),
            "category_presence_auc_sd": float(np.std([x["category_presence_auc"] for x in seeds], ddof=1))}

    if "price" not in a.skip:
        from v4 import scenarios
        rows = scenarios.test_rows(a.world, a.scenario_trips)
        R = a.scenario_draws
        base_trips = [src.make(rows[s:s + 16]) for s in range(0, len(rows), 16)]

        def incidence(product, multiplier):
            inc, size = np.zeros(J), 0.0
            offset = 777
            if a.placebo and product is not None:    # same prices, different random numbers
                offset = 100000 * (1 + product) + int(1000 * multiplier)
                product = None
            for bi, t in enumerate(base_trips):
                seed = offset + bi
                if product is not None:
                    t = Trips(**{f: getattr(t, f) for f in Trips.__dataclass_fields__})
                    t.price = t.price + torch.where(t.offered == product, math.log(multiplier), 0.0) * t.valid
                tr = rep(t, R)
                X = model.sample(tr, seed=seed, max_size=nmax)
                w = X.double().view(len(t.size), R, -1).mean(1)          # P(slot chosen) per trip
                np.add.at(inc, t.offered[t.valid].numpy(), w[t.valid].numpy())
                size += float(w.sum())
            return inc, size / len(rows)

        def progress(i, k, mult):
            print(f"[scenario] {i}: product {k} x{mult} (total {time.time() - t0:.0f}s)", flush=True)

        report["price"], report["price_scenarios"] = scenarios.run(a.world, J, incidence, progress)
        print(json.dumps(report["price"]), flush=True)
    report["eval_seconds"] = time.time() - t0
    a.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
