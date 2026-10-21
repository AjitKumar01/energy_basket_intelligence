#!/usr/bin/env python3
"""Stage 5: score a trained Version-4 model on held-out trips.

1. Likelihood.  log p(S) = E(S) - log Z on the reference trips of --reference (an .npz
   with ``trips`` and ``household``); log Z by importance sampling (sampling.py), with the
   half-split difference and ESS reported.  Any other per-trip column of the reference
   (e.g. another model's log-likelihood on the same trips) is compared pairwise, with a
   household-clustered 95% interval.
2. Realism.  For every supported test trip, one basket is generated per seed; a classifier
   tries to tell real from generated baskets (full basket, and category presence).
   AUC 0.5 means indistinguishable.
3. Price scenarios (synthetic worlds with known truth, --world).  For each scenario
   (one product's price times a multiplier) the change in incidence of the product, of the
   rest of its subcategory and of the rest of its category is compared with the truth.
   Model incidences use z draws from chains at the factual prices; a changed price
   reweights each draw exactly by Z_new(z) / Z_old(z) and incidences are the exact
   d log Z / d u at each draw (Rao-Blackwellised).

Writes TAG.eval.json, TAG.per_trip.npz and TAG_3seed_auc.json next to the checkpoint
(or in --out-dir).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from v4.bundle import data_root, item_categories, load_bundle, supported_trips  # noqa: E402
from v4.checkpoints import freeze, load_version4  # noqa: E402
from v4.exact import incidence as exact_incidence, log_partition  # noqa: E402
from v4.price import changed_price_context  # noqa: E402
from v4.reporting import classifier_auc, clustered_ci  # noqa: E402
from v4.sampling import draw_from_model, gibbs, importance_log_z  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--initialization", type=Path, help="only for checkpoints that do not record it")
    p.add_argument("--additive", type=Path, help="only for checkpoints that do not record it")
    p.add_argument("--reference", type=Path, required=True, help=".npz with trips and household")
    p.add_argument("--world", type=Path, help="synthetic world directory with counterfactual truth")
    p.add_argument("--auc-seeds", type=int, nargs="+", default=[41006, 41007, 41008])
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--burn", type=int, default=200)
    p.add_argument("--components", type=int, default=128)
    p.add_argument("--thin", type=int, default=4)
    p.add_argument("--draws", type=int, default=1024)
    p.add_argument("--scenario-trips", type=int, default=400)
    p.add_argument("--scenario-draws", type=int, default=64)
    p.add_argument("--seed", type=int, default=41001)
    p.add_argument("--threads", type=int, default=4)
    return p.parse_args()


def likelihood(model, batcher, ref, args, gen, tag):
    trips = ref["trips"].astype(np.int64)
    ll, half, ess = [], [], []
    started = time.time()
    for s in range(0, len(trips), args.eval_batch):
        batch = batcher.make(trips[s:s + args.eval_batch])
        batch.bind(model)
        est, diff, e, _ = importance_log_z(model, batch.ix, gen, burn=args.burn, components=args.components,
                                           thin=args.thin, draws=args.draws)
        ll.append((model.energy(batch) - est).numpy())
        half.append(diff.numpy())
        ess.append(e.numpy())
        print(f"[eval {tag}] LL {s + len(batch.trips)}/{len(trips)} running mean "
              f"{np.concatenate(ll).mean():.4f} ({time.time() - started:.0f}s)", flush=True)
    ll, half, ess = map(np.concatenate, (ll, half, ess))
    out = {"trips": len(ll), "mean_ll": float(ll.mean()),
           "is_half_split_abs_mean": float(np.abs(half).mean()), "is_half_split_abs_max": float(np.abs(half).max()),
           "is_ess_fraction_mean": float(ess.mean()), "is_ess_fraction_min": float(ess.min())}
    for name in ref.files:                     # paired comparison with other models' scores
        if name in ("trips", "household") or ref[name].shape != ll.shape or ref[name].dtype.kind != "f":
            continue
        d = ll - ref[name]
        out[f"vs_{name}"] = float(d.mean())
        out[f"vs_{name}_household_95"] = clustered_ci(d, ref["household"])
    return out, (trips, ll, half, ess)


def realism(model, data, batcher, args):
    """Classifier AUCs of real vs generated test baskets, one generated basket per trip per seed."""
    J, C = int(data["n_item"]), int(data["n_cat"])
    item_cat = item_categories(data)
    test = supported_trips(data, 2, model.nmax)
    lptr = data["line_ptr"]
    real = [np.sort(data["line_item"][lptr[t]:lptr[t + 1]]) for t in test]
    groups = np.r_[data["trip_user"][test], data["trip_user"][test]]
    batches = [batcher.make(test[s:s + args.eval_batch]) for s in range(0, len(test), args.eval_batch)]
    rows = []
    for seed in args.auc_seeds:
        gen, generated = torch.Generator().manual_seed(seed), []
        for batch in batches:
            batch.bind(model)
            X = draw_from_model(model, batch.ix, gen, args.burn).numpy()
            items, trip = batch.ix.item.numpy(), batch.ix.item_trip.numpy()
            generated += [np.sort(items[X & (trip == b)]) for b in range(len(batch.trips))]
        full = classifier_auc(real, generated, groups, J, item_cat, C, "full")
        cat = classifier_auc(real, generated, groups, J, item_cat, C, "category")
        rows.append({"seed": seed, "full_basket_auc": round(full[0], 4), "full_basket_fold_sd": round(full[1], 4),
                     "category_presence_auc": round(cat[0], 4), "category_presence_fold_sd": round(cat[1], 4),
                     "size_mean_real": round(float(np.mean([len(x) for x in real])), 4),
                     "size_mean_generated": round(float(np.mean([len(x) for x in generated])), 4)})
        print(json.dumps(rows[-1]), flush=True)
    mean = lambda k: round(float(np.mean([r[k] for r in rows])), 4)  # noqa: E731
    sd = lambda k: round(float(np.std([r[k] for r in rows], ddof=1)), 4) if len(rows) > 1 else 0.0  # noqa: E731
    return {"test_trips": len(test), "burn": args.burn, "seeds": rows,
            "full_basket_auc_mean": mean("full_basket_auc"), "full_basket_auc_sd": sd("full_basket_auc"),
            "category_presence_auc_mean": mean("category_presence_auc"),
            "category_presence_auc_sd": sd("category_presence_auc"),
            "size_mean_generated_mean": mean("size_mean_generated")}


def price_scenarios(model, data, batcher, args, gen):
    world = args.world
    o = np.load(world / "observed.npz")
    meta = json.loads((world / "truth.json").read_text())
    period = o["trip_period"]
    split = np.where(period <= meta["train_weeks"], 0,
                     np.where(period <= meta["train_weeks"] + meta["validation_weeks"], 1, 2))
    rows = np.sort(np.random.default_rng(77).choice(np.flatnonzero(split == 2), size=args.scenario_trips,
                                                    replace=False))
    to_model = {int(orig): i for i, orig in enumerate(o["original_item"])}
    truth = json.loads((world / "counterfactual_truth.json").read_text())
    cat, sub, J = o["category"], o["subcategory"], int(data["n_item"])

    # z draws at factual prices, shared by every scenario.
    sets = []
    for s in range(0, len(rows), args.eval_batch):
        batch = batcher.make(rows[s:s + args.eval_batch])
        batch.bind(model)
        X = draw_from_model(model, batch.ix, gen, args.burn)
        zs = []
        for k in range(args.scenario_draws * args.thin):
            X = gibbs(model, batch.ix, X, gen)
            if (k + 1) % args.thin == 0:
                zs.append(model.basket_field(batch.ix, X) + torch.randn(batch.ix.B, model.rank, generator=gen))
        sets.append((batch, zs))

    def incidence(product=None, multiplier=1.0):
        """Expected purchases of every product (summed over trips) and mean basket size."""
        inc, size = np.zeros(J), 0.0
        for batch, zs in sets:
            ix, ctx = batch.ix, batch.ctx
            if product is not None:
                ctx = changed_price_context(batch.ctx, ix.item_trip,
                                            torch.where(ix.item == product, math.log(multiplier), 0.0))
            batch.bind(model, ctx)
            iv_new = model.iv_shift(ix)
            iv_new = None if iv_new is None else iv_new.detach()
            iv_old = None
            if product is not None:
                batch.bind(model)
                iv_old = model.iv_shift(ix)
                iv_old = None if iv_old is None else iv_old.detach()
            lz_new, grads, lz_old = [], [], []
            for z in zs:
                batch.bind(model, ctx)
                lz, p = exact_incidence(model, ix, model.slot_utilities_given_z(ix, z), z=z, iv=iv_new)
                lz_new.append(lz)
                grads.append(p)
                if product is not None:
                    batch.bind(model)
                    lz_old.append(log_partition(model, ix, model.slot_utilities_given_z(ix, z), z=z, iv=iv_old))
            lz_new = torch.stack(lz_new, 1)
            log_w = (lz_new - torch.stack(lz_old, 1)) if product is not None else torch.zeros_like(lz_new)
            w = torch.softmax(log_w, 1)                      # self-normalised weight per draw
            m = (torch.stack(grads, 1) * w[ix.item_trip]).sum(1)
            np.add.at(inc, ix.item.numpy(), m.numpy())
            size += float(m.sum())
        batch.bind(model)
        return inc, size / len(rows)

    base_inc, base_size = incidence()
    rel = lambda a, b: float(b / a - 1.0) if a > 0 else None  # noqa: E731
    results = []
    for sc in truth["scenarios"]:
        k = to_model.get(int(sc["product"]))
        if k is None:
            continue
        alt, alt_size = incidence(k, sc["price_multiplier"])
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
        t, m = pairs.T
        return {"scenarios": len(pairs), "correlation": float(np.corrcoef(t, m)[0, 1]),
                "mean_abs_error": float(np.abs(t - m).mean()),
                "sign_agreement": float(np.mean(np.sign(t) == np.sign(m))),
                "mean_truth": float(t.mean()), "mean_model": float(m.mean())}

    return {k: agreement(k) for k in ("own", "same_subcategory", "rest_of_category", "basket_size_change")}, results


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    root = data_root(args.data_root)
    data = load_bundle(root)
    model, batcher, blob, _ = load_version4(args.checkpoint, data, root, initialization=args.initialization,
                                            additive=args.additive)
    freeze(model)
    tag = args.checkpoint.stem
    out_dir = args.out_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator().manual_seed(args.seed + 5)
    report = {"checkpoint": str(args.checkpoint),
              "phi_eigenvalues": (torch.linalg.svdvals(model.phi_r()) ** 2).tolist(),
              "rho_c_min": float(model.rho_c.min())}
    if model.iv_slope_raw is not None:
        lam = model.iv_lambda()
        report["iv_slope_lambda"] = {"mean": float(lam.mean()), "min": float(lam.min()), "max": float(lam.max()),
                                     "values": [round(float(v), 4) for v in lam]}
    started = time.time()

    report["likelihood"], (trips, ll, half, ess) = likelihood(model, batcher, np.load(args.reference), args, gen, tag)
    np.savez(out_dir / f"{tag}.per_trip.npz", trips=trips, ll=ll, half_diff=half, ess=ess)
    print(json.dumps(report["likelihood"]), flush=True)

    aucs = realism(model, data, batcher, args)
    (out_dir / f"{tag}_3seed_auc.json").write_text(json.dumps({"checkpoint": str(args.checkpoint), **aucs},
                                                              indent=2) + "\n")
    report["classifier"] = {k: v for k, v in aucs.items() if k != "seeds"}

    if args.world is not None:
        report["price"], report["price_scenarios"] = price_scenarios(model, data, batcher, args, gen)
        print(json.dumps(report["price"]), flush=True)
    report["eval_seconds"] = time.time() - started
    (out_dir / f"{tag}.eval.json").write_text(json.dumps(report, indent=1) + "\n")


if __name__ == "__main__":
    main()
