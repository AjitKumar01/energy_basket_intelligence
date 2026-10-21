"""Small helpers for reports: JSON without NaN, household-clustered intervals, the
classifier two-sample test."""
from __future__ import annotations

import json
import math
import numbers
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def json_safe(value: Any) -> Any:
    """Plain JSON types, with non-finite numbers as null."""
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    return value


def strict_json_dumps(value: Any, *, indent: int = 2) -> str:
    return json.dumps(json_safe(value), indent=indent, allow_nan=False) + "\n"


def clustered_ci(delta, cluster, reps=2000, seed=0):
    """95% interval of the mean of ``delta`` by bootstrap over clusters (households)."""
    codes, idx = np.unique(cluster, return_inverse=True)
    sums, counts = np.bincount(idx, delta), np.bincount(idx).astype(float)
    draws = np.random.default_rng(seed).integers(len(codes), size=(reps, len(codes)))
    means = sums[draws].sum(1) / counts[draws].sum(1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def classifier_auc(real, generated, groups, n_item, item_cat, n_cat, kind):
    """How well a gradient-boosted classifier tells real baskets from generated ones
    (AUC 0.5 = indistinguishable).  kind "full": products, category counts and size;
    kind "category": which categories are present.  5-fold, folds grouped by household.
    Returns (mean AUC, sd over folds)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold

    def features(baskets):
        width = (n_item + n_cat + 1) if kind == "full" else n_cat
        X = np.zeros((len(baskets), width), np.float32)
        for i, items in enumerate(baskets):
            if kind == "full":
                X[i, items] = 1.0
                np.add.at(X[i], n_item + item_cat[items], 1.0)
                X[i, -1] = len(items)
            else:
                X[i, item_cat[items]] = 1.0
        return X

    X = np.r_[features(real), features(generated)]
    y = np.r_[np.ones(len(real)), np.zeros(len(generated))]
    aucs = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=5).split(X, y, groups)):
        clf = HistGradientBoostingClassifier(max_iter=300, early_stopping=True, random_state=k).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], clf.predict_proba(X[te])[:, 1]))
    return float(np.mean(aucs)), float(np.std(aucs, ddof=1))
