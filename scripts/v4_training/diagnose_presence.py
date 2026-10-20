#!/usr/bin/env python3
"""Where do generated baskets differ from real ones in category presence?

One generated basket per supported test trip (as in the classifier test), then real vs generated:
number of categories present, per-category presence rates, pairwise co-occurrence (phi correlation
of category indicators), between-household spread of category rates, and the category-presence
classifier AUC with restricted inputs (number of categories only; linear in the indicators; boosted
trees on the indicators, and depth-limited trees that can only use pairs or triples), which separates
marginal-rate errors from interaction errors and pairwise from higher-order ones.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
from pathlib import Path

sys.modules.setdefault("pathlib._local", pathlib)
os.environ.setdefault("V3_AFFINITY", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/version4"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import stress  # noqa: E402
from stress import Batch, gibbs, sample_slots, setup, slot_utilities  # noqa: E402
from pipeline_support import supported_trips  # noqa: E402

torch.set_default_dtype(torch.float64)


def auc(X, y, groups, kind, depth=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    out = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        clf = (HistGradientBoostingClassifier(max_iter=300, early_stopping=True, random_state=k, max_depth=depth)
               if kind == "trees"
               else LogisticRegression(max_iter=2000))
        out.append(roc_auc_score(y[te], clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
    return round(float(np.mean(out)), 4)


def household_spread(P, house):
    """Mean over categories of the between-household variance of per-household presence rates."""
    ids, inv = np.unique(house, return_inverse=True)
    n = np.bincount(inv)
    keep = n >= 3
    rates = np.stack([np.bincount(inv, P[:, c]) / n for c in range(P.shape[1])], 1)[keep]
    return round(float(rates.var(0).mean()), 5)


def main():
    A = ROOT.parent / "energy_basket_intelligence/artifacts/stress_clean_confirm"
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--initialization", type=Path, default=A / "artifacts/initialization.pt")
    p.add_argument("--additive", type=Path, default=A / "out/v3_pipeline_additive_best.pt")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, default=41006)
    p.add_argument("--burn", type=int, default=200)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    stress.USE_CPP_SAMPLER = True
    blob = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    data, model, meta, batcher = setup(args.initialization, args.additive, args.threads,
                                       recency=bool(blob.get("recency", False)))
    stress.configure_model_from_blob(model, blob)
    model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    r = int(blob["rank"])
    J, C = int(data["n_item"]), int(data["n_cat"])
    item_cat = np.zeros(J, dtype=np.int64)
    item_cat[data["line_item"]] = data["line_cat"]
    test = supported_trips(data, 2, int(model.nmax))
    lptr = data["line_ptr"]
    real = [data["line_item"][lptr[t]:lptr[t + 1]] for t in test]
    gen = torch.Generator().manual_seed(args.seed)
    generated = []
    with torch.no_grad():
        for s in range(0, len(test), 256):
            bt = Batch(batcher, test[s:s + 256], J)
            bt.bind(model)
            X = sample_slots(model, bt.ix, slot_utilities(model, bt.ix, None, r, beta=0.0), gen, beta=0.0)
            X = gibbs(model, bt.ix, X, r, gen, sweeps=args.burn).numpy()
            items, xt = bt.ix.item.numpy(), bt.ix.item_trip.numpy()
            generated += [items[X & (xt == b)] for b in range(len(bt.trips))]

    def presence(baskets):
        P = np.zeros((len(baskets), C), np.float32)
        for i, b in enumerate(baskets):
            P[i, item_cat[b]] = 1.0
        return P
    Pr, Pg = presence(real), presence(generated)
    house = data["trip_user"][test]
    Kr, Kg = Pr.sum(1), Pg.sum(1)
    rate_r, rate_g = Pr.mean(0), Pg.mean(0)
    Cr, Cg = np.corrcoef(Pr.T), np.corrcoef(Pg.T)
    iu = np.triu_indices(C, 1)
    dc = (Cg - Cr)[iu]
    worst = np.argsort(-np.abs(dc))[:10]
    X = np.r_[Pr, Pg]
    y = np.r_[np.ones(len(Pr)), np.zeros(len(Pg))]
    groups = np.r_[house, house]
    K = X.sum(1, keepdims=True)
    out = {
        "checkpoint": str(args.checkpoint), "test_trips": int(len(test)), "seed": args.seed,
        "n_categories": {"real_mean": round(float(Kr.mean()), 3), "gen_mean": round(float(Kg.mean()), 3),
                         "real_var": round(float(Kr.var()), 3), "gen_var": round(float(Kg.var()), 3),
                         "real_hist": np.bincount(Kr.astype(int), minlength=20)[:20].tolist(),
                         "gen_hist": np.bincount(Kg.astype(int), minlength=20)[:20].tolist()},
        "category_rate": {"mean_abs_diff": round(float(np.abs(rate_g - rate_r).mean()), 4),
                          "max_abs_diff": round(float(np.abs(rate_g - rate_r).max()), 4),
                          "worst": [[int(c), round(float(rate_r[c]), 3), round(float(rate_g[c]), 3)]
                                    for c in np.argsort(-np.abs(rate_g - rate_r))[:8]]},
        "pair_correlation": {"real_mean_abs": round(float(np.abs(Cr[iu]).mean()), 4),
                             "gen_mean_abs": round(float(np.abs(Cg[iu]).mean()), 4),
                             "mean_abs_diff": round(float(np.abs(dc).mean()), 4),
                             "corr_of_corrs": round(float(np.corrcoef(Cr[iu], Cg[iu])[0, 1]), 4),
                             "worst": [[int(iu[0][k]), int(iu[1][k]), round(float(Cr[iu][k]), 3), round(float(Cg[iu][k]), 3)]
                                       for k in worst]},
        "household_spread": {"real": household_spread(Pr, house), "gen": household_spread(Pg, house)},
        "auc": {"n_categories_only": auc(K, y, groups, "trees"), "linear_indicators": auc(X, y, groups, "linear"),
                "trees_indicators": auc(X, y, groups, "trees"),
                "trees_depth2_pairs": auc(X, y, groups, "trees", depth=2),
                "trees_depth3_triples": auc(X, y, groups, "trees", depth=3),
                "trees_indicators_given_K": auc(np.c_[X, K], y, groups, "trees")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output.with_suffix(".npz"), trips=test, real=Pr.astype(bool), generated=Pg.astype(bool))
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
