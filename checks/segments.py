#!/usr/bin/env python3
"""Customer segments from a fitted model's household vectors, described from the data.

Features per household: the centred taste vector theta_h (its composition part, and the
size shift kappa_h, the last coordinate) and the log of its mean price slope
g_h = mean_j softplus(gamma_h) . softplus(beta_j).  Composition directions are weighted by
how much they vary over products (the alpha scale), so a direction the model barely uses
counts little.  Features are standardised and clustered with k-means; k is chosen by
silhouette over --ks unless --k is given.

Each segment is described from the baskets, not the model: households, trips, products
per trip, share bought on discount (chain promo_depth >= 5%), the commodities it buys most
above the population's share (training weeks), and, with --demographics, household size,
children and income level.  Checks:
    held-out   the same commodity index on validation weeks (never used in fitting);
               correlation with the training index, per segment
    stability  adjusted Rand index between k-means runs with different seeds

    segments.py --checkpoint CK --out FILE [--k 6] [--demographics CSV]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402

from v4.bundle import data_root  # noqa: E402

sp = lambda x: np.logaddexp(0.0, x)  # noqa: E731


def household_features(st):
    theta = st["theta"] - st["theta"].mean(0)
    alpha = st["alpha"][:, :-1] - st["alpha"][:, :-1].mean(0)
    scale = alpha.std(0)                                  # how much each direction varies over products
    composition = theta[:, :-1] * scale
    slope = (sp(st["gamma"]) @ sp(st["beta"]).T).mean(1)
    X = np.column_stack([composition, theta[:, -1], np.log(slope)])
    return (X - X.mean(0)) / X.std(0).clip(1e-12), theta[:, -1], slope


def commodity_index(rows, commodity, labels, k):
    """For each segment: share of its purchases in each commodity divided by everyone's share."""
    c = commodity[rows.item_id.to_numpy()]
    seg = labels[rows.user_id.to_numpy()]
    table = pd.crosstab(seg, c)
    share = table.div(table.sum(1), axis=0)
    overall = table.sum(0) / table.values.sum()
    return share.div(overall, axis=1).reindex(range(k))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--k", type=int, help="number of segments (default: best silhouette over --ks)")
    p.add_argument("--ks", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    p.add_argument("--demographics", type=Path, help="raw hh_demographic.csv (Dunnhumby)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = data_root(a.data_root)
    st = {k: v.numpy() for k, v in torch.load(a.checkpoint, map_location="cpu", weights_only=False)["model"].items()
          if torch.is_tensor(v)}
    X, kappa, slope = household_features(st)
    N = len(X)

    scores = {}
    for k in ([a.k] if a.k else a.ks):
        lab = KMeans(k, n_init=10, random_state=a.seed).fit_predict(X)
        scores[k] = float(silhouette_score(X, lab, sample_size=min(N, 2000), random_state=a.seed))
    k = a.k or max(scores, key=scores.get)
    labels = KMeans(k, n_init=10, random_state=a.seed).fit_predict(X)
    stability = [float(adjusted_rand_score(labels, KMeans(k, n_init=10, random_state=s).fit_predict(X)))
                 for s in (1, 2, 3)]

    items = pd.read_parquet(root / "basket_input/items.parquet").sort_values("item_id").reset_index(drop=True)
    commodity = items.COMMODITY_DESC.str.title().to_numpy()
    b = pd.read_parquet(root / "basket_input/baskets.parquet", columns=["BASKET_ID", "user_id", "WEEK_NO", "item_id", "split"])
    week = pd.read_parquet(root / "data/price_week.parquet", columns=["PRODUCT_ID", "WEEK_NO", "promo_depth"])
    week = week.merge(items[["PRODUCT_ID", "item_id"]], on="PRODUCT_ID")
    deal = week[week.promo_depth >= 0.05][["item_id", "WEEK_NO"]].assign(deal=1)
    b = b.merge(deal, on=["item_id", "WEEK_NO"], how="left").fillna({"deal": 0})
    train, valid = b[b.split == "train"], b[b.split == "validation"]
    per_trip = train.groupby(["user_id", "BASKET_ID"]).size()
    hh = pd.DataFrame({"products_per_trip": per_trip.groupby("user_id").mean(),
                       "trips": per_trip.groupby("user_id").size(),
                       "share_on_discount": train.groupby("user_id").deal.mean()}).reindex(range(N))
    idx_train = commodity_index(train, commodity, labels, k)
    idx_valid = commodity_index(valid, commodity, labels, k)
    counts = pd.Series(commodity[train.item_id]).value_counts()
    big = counts[counts >= 5000].index                  # enough purchases for a stable index
    demo = None
    if a.demographics is not None:
        tx = pd.read_parquet(root / "data/tx.parquet", columns=["BASKET_ID", "household_key"]).drop_duplicates("BASKET_ID")
        key = b[["BASKET_ID", "user_id"]].drop_duplicates().merge(tx, on="BASKET_ID").groupby("user_id").household_key.first()
        demo = pd.DataFrame({"household_key": key.reindex(range(N)).to_numpy(), "segment": labels}).merge(
            pd.read_csv(a.demographics), on="household_key")

    segments = []
    for s in range(k):
        m = labels == s
        top = idx_train.loc[s, big].sort_values(ascending=False)
        common = [c for c in big if c in idx_valid.columns]          # sold in both periods
        row = {"segment": s, "households": int(m.sum()),
               "products_per_trip": round(float(hh.products_per_trip[m].mean()), 2),
               "trips_in_training": round(float(hh.trips[m].mean()), 1),
               "share_bought_on_discount": round(float(hh.share_on_discount[m].mean()), 3),
               "model_price_slope": round(float(slope[m].mean()), 4),
               "model_size_shift": round(float(kappa[m].mean()), 3),
               "buys_most_above_average": top.head(6).round(2).to_dict(),
               "buys_least": top.tail(4).round(2).to_dict(),
               "held_out_index_corr": round(float(np.corrcoef(np.log(idx_train.loc[s, common].clip(1e-3)),
                                                              np.log(idx_valid.loc[s, common].clip(1e-3)))[0, 1]), 3)}
        if demo is not None:
            d = demo[demo.segment == s]
            row["with_demographics"] = int(len(d))
            if len(d):
                row["household_size_5plus"] = round(float((d.classification_4 == "5+").mean()), 3)
                row["single_person"] = round(float((d.classification_4 == "1").mean()), 3)
                row["with_children"] = round(float((d.KID_CATEGORY_DESC != "None/Unknown").mean()), 3)
        segments.append(row)
    out = {"checkpoint": str(a.checkpoint), "k": k, "silhouette_by_k": scores,
           "stability_adjusted_rand_vs_other_seeds": np.round(stability, 3).tolist(),
           "population": {"products_per_trip": round(float(hh.products_per_trip.mean()), 2),
                          "share_bought_on_discount": round(float(hh.share_on_discount.mean()), 3),
                          "model_price_slope": round(float(slope.mean()), 4)},
           "segments": segments}
    if demo is not None:
        out["population"]["household_size_5plus"] = round(float((demo.classification_4 == "5+").mean()), 3)
        out["population"]["with_children"] = round(float((demo.KID_CATEGORY_DESC != "None/Unknown").mean()), 3)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    np.save(a.out.with_suffix(".labels.npy"), labels)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
