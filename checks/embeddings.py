#!/usr/bin/env python3
"""What has a fitted model learned?  Interaction, taste and price vectors, read against the data.

Interactions (Version 4; phi_j = psi_c(j) + delta_j):
    spectrum of Phi'Phi; the leading direction (which commodities load on it, and whether it
    follows popularity or basket size); the strongest category pairs once it is removed; and
    the correlation of learned pair strength phi_i.phi_j with real log co-purchase lift,
    log P(i, j) / (P(i) P(j)), over popular product pairs from different model categories.
Taste (theta_h . alpha_j; the last coordinate is the household size shift kappa_h):
    whether products close in taste space share a commodity, against chance; whether kappa_h
    follows how many products the household really buys per trip.
Price (g_hj = softplus(gamma_h) . softplus(beta_j) >= 0):
    whether households with larger slopes buy a larger share on discount; the most and least
    price-sensitive commodities; private label against national brands; and whether product
    sensitivity follows how often the product is discounted.
With --demographics (the raw hh_demographic.csv): price sensitivity and size shift by
household size, children and income level.

    embeddings.py --checkpoint CK [--initialization INIT] --out FILE [--demographics CSV]
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

torch.set_default_dtype(torch.float64)

from v4.bundle import data_root  # noqa: E402

sp = lambda x: np.logaddexp(0.0, x)  # noqa: E731   softplus


def corr(a, b):
    return round(float(np.corrcoef(a, b)[0, 1]), 3)


def load_state(path, initialization):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = {k: v.numpy() for k, v in blob["model"].items() if torch.is_tensor(v)}
    if "rank" not in blob:                                       # additive: phi = 0
        return state, 0
    return state, int(blob["rank"])


def interactions(st, r, items, train, n_baskets, out):
    phi, psi, cat = st["phi"][:, :r], st["psi_cat"], st["cat_of"]
    J, C = len(items), psi.shape[0]
    commodity = items.COMMODITY_DESC.str.title().to_numpy()
    name = (items.COMMODITY_DESC.str.title() + " / " + items.SUB_COMMODITY_DESC.str.title()).to_numpy()
    count = np.bincount(train.item_id, minlength=J)
    size = train.groupby("BASKET_ID").item_id.transform("size").to_numpy()
    size_when_bought = np.bincount(train.item_id, weights=size, minlength=J) / np.maximum(count, 1)

    ev = np.linalg.eigvalsh(phi.T @ phi)[::-1]
    _, _, Vt = np.linalg.svd(phi, full_matrices=False)
    v = Vt[0] * np.sign((phi @ Vt[0]).sum())
    load = phi @ v
    seen = count > 50
    by_commodity = pd.Series(load).groupby(commodity).mean().sort_values()
    out["interactions"] = {
        "eigenvalues": np.round(ev, 2).tolist(), "share": np.round(ev / ev.sum(), 3).tolist(),
        "category_part_share": round(float((psi[cat] ** 2).sum() / (phi ** 2).sum()), 3),
        "leading_direction": {
            "share_positive": round(float((load > 0).mean()), 3),
            "corr_with_log_purchases": corr(load[seen], np.log(count[seen])),
            "corr_with_size_of_baskets_containing_it": corr(load[seen], size_when_bought[seen]),
            "highest_commodities": by_commodity.tail(8).round(2).to_dict(),
            "lowest_commodities": by_commodity.head(8).round(2).to_dict()}}

    label = pd.Series(commodity).groupby(cat).agg(lambda s: s.value_counts().index[0])
    sizes = np.bincount(cat, minlength=C)
    rest = psi - np.outer(psi @ v, v)
    G = rest @ rest.T
    iu = np.triu_indices(C, 1)
    ok = (sizes[iu[0]] >= 3) & (sizes[iu[1]] >= 3)
    order = np.argsort(-G[iu][ok])
    pick = lambda ts: [[round(float(G[iu[0][ok][t], iu[1][ok][t]]), 2), label[iu[0][ok][t]],  # noqa: E731
                        label[iu[1][ok][t]]] for t in ts]
    out["interactions"]["category_pairs_without_leading_direction"] = {
        "strongest": pick(order[:15]), "most_negative": pick(order[-8:][::-1])}

    top = np.argsort(-count)[:400]
    where = {j: k for k, j in enumerate(top)}
    co = np.zeros((len(top), len(top)))
    for basket in train[train.item_id.isin(top)].groupby("BASKET_ID").item_id.apply(list):
        k = [where[j] for j in basket]
        co[np.ix_(k, k)] += 1
    p = count[top] / n_baskets
    lift = np.log((co / n_baskets + 1e-12) / np.outer(p, p))
    pair = phi[top] @ phi[top].T
    residual = pair - np.outer(load[top], load[top])
    ui = np.triu_indices(len(top), 1)
    keep = (cat[top][ui[0]] != cat[top][ui[1]]) & (co[ui] >= 20)
    best = np.argsort(-residual[ui][keep])[:10]
    out["interactions"]["learned_pairs_vs_real_lift"] = {
        "pairs": int(keep.sum()), "corr_with_log_lift": corr(pair[ui][keep], lift[ui][keep]),
        "corr_without_leading_direction": corr(residual[ui][keep], lift[ui][keep]),
        "strongest_learned_pairs": [[round(float(np.exp(lift[ui][keep][t])), 1), name[top[ui[0][keep][t]]],
                                     name[top[ui[1][keep][t]]]] for t in best]}


def taste(st, items, train, baskets_per_household, out):
    alpha = st["alpha"][:, :-1]
    alpha = alpha - alpha.mean(0)
    theta = st["theta"] - st["theta"].mean(0)
    commodity = items.COMMODITY_DESC.to_numpy()
    count = np.bincount(train.item_id, minlength=len(items))
    popular = np.flatnonzero(count >= 100)
    a = alpha[popular] / np.linalg.norm(alpha[popular], axis=1, keepdims=True).clip(1e-12)
    sim = a @ a.T
    np.fill_diagonal(sim, -np.inf)
    nearest = np.argsort(-sim, axis=1)[:, :5]
    same = (commodity[popular][nearest] == commodity[popular][:, None]).mean()
    chance = (pd.Series(commodity[popular]).value_counts(normalize=True) ** 2).sum()
    kappa = theta[:, -1]
    out["taste"] = {
        "products_compared": int(len(popular)),
        "share_of_5_nearest_in_same_commodity": round(float(same), 3),
        "same_commodity_by_chance": round(float(chance), 3),
        "size_shift_corr_with_products_per_trip": corr(kappa, baskets_per_household["mean_size"]),
        "size_shift_corr_with_trips": corr(kappa, np.log(baskets_per_household["trips"]))}
    return kappa


def price(st, items, train, root, out):
    g_h = sp(st["gamma"]).sum(1)                       # household scale
    G = sp(st["gamma"]) @ sp(st["beta"]).T             # [N, J]
    household, product = G.mean(1), G.mean(0)
    kappa = float(sp(st["price_kappa"]))
    extra = {"relative_multiplier_kappa": round(kappa, 3)}
    if "disc_kappa" in st:
        extra["discount_multiplier"] = round(float(sp(st["disc_kappa"])), 3)
    if "disc_common" in st:
        extra["common_discount_slope"] = round(float(st["disc_common"]), 4)
    week = pd.read_parquet(root / "data/price_week.parquet", columns=["PRODUCT_ID", "WEEK_NO", "promo_depth"])
    week = week.merge(items[["PRODUCT_ID", "item_id"]], on="PRODUCT_ID")
    on_deal = week[week.promo_depth >= 0.05][["item_id", "WEEK_NO"]].assign(deal=1)
    rows = train.merge(on_deal, on=["item_id", "WEEK_NO"], how="left").fillna({"deal": 0})
    deal_share = rows.groupby("user_id").deal.mean().reindex(range(len(household))).to_numpy()
    deal_weeks = week.assign(d=week.promo_depth >= 0.05).groupby("item_id").d.mean().reindex(
        range(len(items))).fillna(0).to_numpy()
    commodity = items.COMMODITY_DESC.str.title().to_numpy()
    count = np.bincount(train.item_id, minlength=len(items))
    big = pd.Series(count).groupby(commodity).sum() >= 2000
    by_comm = pd.Series(product).groupby(commodity).mean()[big].sort_values()
    brand = items.BRAND.to_numpy() if "BRAND" in items else None
    out["price"] = {
        **extra,
        "own_price_slope_mean": round(float(product.mean() * kappa), 4),
        "household_slope_spread_p10_p90": np.round(np.quantile(household, [.1, .9]), 4).tolist(),
        "household_slope_corr_with_share_bought_on_discount": corr(household, deal_share),
        "product_slope_corr_with_share_of_weeks_discounted": corr(product, deal_weeks),
        "most_price_sensitive_commodities": by_comm.tail(8).round(4).to_dict(),
        "least_price_sensitive_commodities": by_comm.head(8).round(4).to_dict()}
    if brand is not None:
        out["price"]["mean_product_slope_by_brand_type"] = pd.Series(product).groupby(brand).mean().round(4).to_dict()
    return household, g_h


def demographics(path, root, kappa, household_slope, out):
    demo = pd.read_csv(path)
    tx = pd.read_parquet(root / "data/tx.parquet", columns=["BASKET_ID", "household_key"]).drop_duplicates("BASKET_ID")
    b = pd.read_parquet(root / "basket_input/baskets.parquet", columns=["BASKET_ID", "user_id"]).drop_duplicates()
    key = b.merge(tx, on="BASKET_ID").groupby("user_id").household_key.first()
    frame = pd.DataFrame({"household_key": key.reindex(range(len(kappa))).to_numpy(),
                          "size_shift": kappa, "price_slope": household_slope}).merge(demo, on="household_key")
    columns = {"household_size": "classification_4", "children": "KID_CATEGORY_DESC", "income_level": "classification_3"}
    out["demographics"] = {"households_with_demographics": int(len(frame))}
    for label, column in columns.items():
        if column in frame:
            out["demographics"][label] = frame.groupby(column)[["size_shift", "price_slope"]].mean().round(4).to_dict()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--initialization", type=Path)
    p.add_argument("--data-root", type=Path)
    p.add_argument("--demographics", type=Path, help="raw hh_demographic.csv (Dunnhumby)")
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    root = data_root(a.data_root)
    st, r = load_state(a.checkpoint, a.initialization)
    items = pd.read_parquet(root / "basket_input/items.parquet").sort_values("item_id").reset_index(drop=True)
    baskets = pd.read_parquet(root / "basket_input/baskets.parquet", columns=["BASKET_ID", "user_id", "WEEK_NO", "item_id", "split"])
    train = baskets[baskets.split == "train"]
    per_trip = train.groupby(["user_id", "BASKET_ID"]).size()
    per_household = pd.DataFrame({"mean_size": per_trip.groupby("user_id").mean(),
                                  "trips": per_trip.groupby("user_id").size()}).reindex(range(st["theta"].shape[0]))
    out = {"checkpoint": str(a.checkpoint), "rank": r}
    if r > 0:
        interactions(st, r, items, train, train.BASKET_ID.nunique(), out)
    kappa = taste(st, items, train, per_household, out)
    household_slope, _ = price(st, items, train, root, out)
    if a.demographics is not None:
        demographics(a.demographics, root, kappa, household_slope, out)
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    main()
