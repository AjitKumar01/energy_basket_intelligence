#!/usr/bin/env python3
"""Score a SHOPPER fit on the stress world exactly as scripts/evaluate.py scores the energy model.

1. Held-out basket log-likelihood on the certified reference trips (the 4,096 test trips the
   energy models use), log P(S) by importance sampling over orderings with a half-split check,
   paired per trip with an energy-model run's per-trip log-likelihoods (e.g. hs_rec).
2. Classifier two-sample AUCs (full basket, category presence) on all supported test trips,
   one generated basket per trip per seed (generation capped at the energy model's nmax).
3. The 32 single-product price scenarios against the world's causal truth, on the same 400
   test trips: incidences P(j in S) by Monte Carlo with common random numbers.
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
    p.add_argument("--world", type=Path, required=True, help="synthetic world with counterfactual truth")
    p.add_argument("--perms", type=int, default=256)
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
    src = BundleSource()
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
        o = np.load(a.world / "observed.npz")
        tm = json.loads((a.world / "truth.json").read_text())
        period = o["trip_period"]
        split = np.where(period <= tm["train_weeks"], 0, np.where(period <= tm["train_weeks"] + tm["validation_weeks"], 1, 2))
        rows = np.sort(np.random.default_rng(77).choice(np.flatnonzero(split == 2), size=a.scenario_trips, replace=False))
        to_model = {int(orig): i for i, orig in enumerate(o["original_item"])}
        truth = json.loads((a.world / "counterfactual_truth.json").read_text())
        cat, sub = o["category"], o["subcategory"]
        R = a.scenario_draws
        base_trips = [src.make(rows[s:s + 16]) for s in range(0, len(rows), 16)]

        def incidence(product=None, multiplier=1.0):
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

        ts = time.time()
        base_inc, base_size = incidence()
        print(f"[scenario] base ({time.time() - ts:.0f}s)", flush=True)
        rel = lambda x, y: float(y / x - 1.0) if x > 0 else None  # noqa: E731
        results = []
        for sc in truth["scenarios"]:
            k = to_model.get(int(sc["product"]))
            if k is None:
                continue
            ts = time.time()
            alt, alt_size = incidence(k, sc["price_multiplier"])
            print(f"[scenario] {len(results) + 1}: product {k} x{sc['price_multiplier']} "
                  f"({time.time() - ts:.0f}s, total {time.time() - t0:.0f}s)", flush=True)
            same = np.flatnonzero((cat == cat[k]) & (sub == sub[k]) & (np.arange(J) != k))
            rest = np.flatnonzero((cat == cat[k]) & (sub != sub[k]))
            results.append({"product": int(sc["product"]), "price_multiplier": sc["price_multiplier"],
                            "own": {"truth": rel(*sc["own_incidence"]), "model": rel(base_inc[k], alt[k])},
                            "same_subcategory": {"truth": rel(*sc["same_subcategory_incidence"]),
                                                 "model": rel(base_inc[same].sum(), alt[same].sum())},
                            "rest_of_category": {"truth": rel(*sc["rest_of_category_incidence"]),
                                                 "model": rel(base_inc[rest].sum(), alt[rest].sum())},
                            "basket_size_change": {"truth": sc["mean_basket_size"][1] - sc["mean_basket_size"][0],
                                                   "model": alt_size - base_size}})

        def agreement(key):
            pairs = np.array([(x[key]["truth"], x[key]["model"]) for x in results
                              if x[key]["truth"] is not None and x[key]["model"] is not None])
            t_, m_ = pairs.T
            return {"scenarios": len(pairs), "correlation": float(np.corrcoef(t_, m_)[0, 1]),
                    "mean_abs_error": float(np.abs(t_ - m_).mean()),
                    "sign_agreement": float(np.mean(np.sign(t_) == np.sign(m_))),
                    "mean_truth": float(t_.mean()), "mean_model": float(m_.mean())}
        report["price"] = {k: agreement(k) for k in ("own", "same_subcategory", "rest_of_category", "basket_size_change")}
        report["price_scenarios"] = results
        print(json.dumps(report["price"]), flush=True)
    report["eval_seconds"] = time.time() - t0
    a.output.write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
