#!/usr/bin/env python3
"""Do the baskets support discrete trip types?  A latent-class check on category presence.

Fits a mixture of K independent-Bernoulli classes to the category-presence vectors of the training
trips by EM (several restarts, Beta(0.5, 0.5) smoothing), for each K in --ks, and reports:
  val_ll        mean log-likelihood per validation trip (validation split), and its gain over K = 1
  auc           the category-presence classifier test (same classifier as stress.py) of real test
                baskets against baskets sampled from the fitted mixture: all depths, and trees
                limited to pairs (depth 2) and triples (depth 3)
The model ignores households and prices, so it shows only whether presence has discrete-type
structure and how many types the data support; it is not a basket model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.special import logsumexp


def presence(cache, trips):
    C = int(cache["n_cat"])
    lptr, lcat = cache["line_ptr"], cache["line_cat"]
    X = np.zeros((len(trips), C), bool)
    for i, t in enumerate(trips):
        X[i, lcat[lptr[t]:lptr[t + 1]]] = True
    return X


def loglik_rows(X, w, theta):
    lt, l1t = np.log(theta), np.log1p(-theta)
    return logsumexp(np.log(w)[None, :] + X @ lt.T + (~X) @ l1t.T, axis=1)


def fit(X, K, rng, iters=500, tol=1e-7, a=0.5):
    N, C = X.shape
    Xf = X.astype(np.float64)
    resp = rng.dirichlet(np.ones(K), N)
    prev = -np.inf
    for _ in range(iters):
        nk = resp.sum(0)
        w = (nk + 1.0) / (N + K)
        theta = (resp.T @ Xf + a) / (nk[:, None] + 2 * a)
        lj = np.log(w)[None, :] + Xf @ np.log(theta).T + (1 - Xf) @ np.log1p(-theta).T
        ll = logsumexp(lj, axis=1)
        resp = np.exp(lj - ll[:, None])
        cur = ll.mean()
        if cur - prev < tol:
            break
        prev = cur
    return w, theta, cur


def sample(w, theta, n, rng):
    k = rng.choice(len(w), n, p=w)
    return rng.random((n, theta.shape[1])) < theta[k]


def auc(X, y, groups, depth=None):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold
    out = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        clf = HistGradientBoostingClassifier(max_iter=300, early_stopping=True, random_state=k, max_depth=depth)
        out.append(roc_auc_score(y[te], clf.fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]))
    return round(float(np.mean(out)), 4)


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--cache", type=Path, required=True, help="bundle basket_input/v3_index_affinity.npz")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 2, 3, 4, 6, 8, 12, 16])
    p.add_argument("--restarts", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    cache = np.load(a.cache)
    split = cache["trip_split"]
    nonempty = np.diff(cache["line_ptr"]) > 0
    tr, va, te = (np.flatnonzero((split == s) & nonempty) for s in (0, 1, 2))
    Xtr, Xva, Xte = presence(cache, tr), presence(cache, va), presence(cache, te)
    house_te = cache["trip_user"][te]
    rng = np.random.default_rng(a.seed)
    out = {"cache": str(a.cache), "categories": int(Xtr.shape[1]), "trips": {"train": len(tr), "validation": len(va), "test": len(te)},
           "mean_categories_per_trip": round(float(Xtr.sum(1).mean()), 3), "by_k": []}
    base = None
    for K in a.ks:
        best = max((fit(Xtr, K, np.random.default_rng([a.seed, K, r])) for r in range(a.restarts)), key=lambda f: f[2])
        w, theta, train_ll = best
        val_ll = float(loglik_rows(Xva, w, theta).mean())
        base = val_ll if base is None else base
        G = sample(w, theta, len(Xte), rng)
        X = np.r_[Xte, G].astype(np.float32)
        y = np.r_[np.ones(len(Xte)), np.zeros(len(G))]
        groups = np.r_[house_te, house_te]
        row = {"K": K, "train_ll": round(train_ll, 4), "val_ll": round(val_ll, 4), "val_gain_over_k1": round(val_ll - base, 4),
               "weights": np.round(np.sort(w)[::-1], 3).tolist(),
               "auc": {"all_depths": auc(X, y, groups), "depth2_pairs": auc(X, y, groups, 2), "depth3_triples": auc(X, y, groups, 3)}}
        print(json.dumps(row), flush=True)
        out["by_k"].append(row)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
