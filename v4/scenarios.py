"""The stress world's price scenarios: one product's price times a multiplier, compared with
the generator's causal truth.  Shared by scripts/evaluate.py and the SHOPPER evaluator; each
supplies its own ``incidence(product, multiplier)``.

For every scenario the relative change in expected purchases of the product ("own"), of the
rest of its subcategory, and of the rest of its category, and the change in mean basket
size, are compared with the truth over all scenarios: correlation, mean absolute error, sign
agreement and means.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KEYS = ("own", "same_subcategory", "rest_of_category", "basket_size_change")


def test_rows(world: Path, n_trips: int) -> np.ndarray:
    """The ``n_trips`` test trips the scenarios are scored on (fixed seed)."""
    o = np.load(world / "observed.npz")
    meta = json.loads((world / "truth.json").read_text())
    period = o["trip_period"]
    split = np.where(period <= meta["train_weeks"], 0,
                     np.where(period <= meta["train_weeks"] + meta["validation_weeks"], 1, 2))
    return np.sort(np.random.default_rng(77).choice(np.flatnonzero(split == 2), size=n_trips, replace=False))


def run(world: Path, n_item: int, incidence, progress=None):
    """Score every scenario; ``incidence(product, multiplier)`` returns (expected purchases
    of every product summed over the test trips, mean basket size); product None = factual."""
    o = np.load(world / "observed.npz")
    to_model = {int(orig): i for i, orig in enumerate(o["original_item"])}
    truth = json.loads((world / "counterfactual_truth.json").read_text())
    cat, sub = o["category"], o["subcategory"]
    base_inc, base_size = incidence(None, 1.0)
    rel = lambda a, b: float(b / a - 1.0) if a > 0 else None  # noqa: E731
    results = []
    for sc in truth["scenarios"]:
        k = to_model.get(int(sc["product"]))
        if k is None:
            continue
        alt, alt_size = incidence(k, sc["price_multiplier"])
        if progress is not None:
            progress(len(results) + 1, k, sc["price_multiplier"])
        same = np.flatnonzero((cat == cat[k]) & (sub == sub[k]) & (np.arange(n_item) != k))
        rest = np.flatnonzero((cat == cat[k]) & (sub != sub[k]))
        results.append({"product": int(sc["product"]), "price_multiplier": sc["price_multiplier"],
                        "own": {"truth": rel(*sc["own_incidence"]), "model": rel(base_inc[k], alt[k])},
                        "same_subcategory": {"truth": rel(*sc["same_subcategory_incidence"]),
                                             "model": rel(base_inc[same].sum(), alt[same].sum())},
                        "rest_of_category": {"truth": rel(*sc["rest_of_category_incidence"]),
                                             "model": rel(base_inc[rest].sum(), alt[rest].sum())},
                        "basket_size_change": {"truth": sc["mean_basket_size"][1] - sc["mean_basket_size"][0],
                                               "model": alt_size - base_size}})
    return {k: agreement(results, k) for k in KEYS}, results


def agreement(results, key):
    pairs = np.array([(x[key]["truth"], x[key]["model"]) for x in results
                      if x[key]["truth"] is not None and x[key]["model"] is not None])
    t, m = pairs.T
    return {"scenarios": len(pairs), "correlation": float(np.corrcoef(t, m)[0, 1]),
            "mean_abs_error": float(np.abs(t - m).mean()),
            "sign_agreement": float(np.mean(np.sign(t) == np.sign(m))),
            "mean_truth": float(t.mean()), "mean_model": float(m.mean())}
