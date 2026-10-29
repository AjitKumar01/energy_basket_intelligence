"""What the learned embeddings of a real-data mixture checkpoint contain.

    ENERGY_MODEL_DATA_ROOT=<bundle> python3 -u research/inspect_embeddings.py CK.pt --raw <Complete Journey CSV dir> --out OUT.json

interaction  phi_j (rank r): coupling phi_i . phi_j in nats.  Agreement with the training baskets'
             log lift of product pairs from different categories; each product's strongest
             partners in other categories; the commodities at the two ends of the leading
             directions of phi.
customer     theta_h with product loadings alpha_j: the size shift theta_h[-1] against the
             household's mean basket size; how well theta_h . alpha_j ranks each household's own
             department shares (training weeks); demographic differences along the leading
             directions of theta.
price        g_hj = softplus(gamma_h) . softplus(beta_j) >= 0: spread over households and products;
             household sensitivity by income level and children; product sensitivity by department
             and private label against national brands.
Training weeks only.
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
from torch.nn.functional import softplus  # noqa: E402

torch.set_default_dtype(torch.float64)

from v4.bundle import data_root, load_bundle  # noqa: E402
from v4.checkpoints import freeze, load_version4  # noqa: E402


def spearman(a, b):
    ra, rb = pd.Series(a).rank().to_numpy(), pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


@torch.no_grad()
def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--raw", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top-products", type=int, default=600)
    a = p.parse_args()
    root = data_root(None)
    data = load_bundle(root)
    model, _, blob, _ = load_version4(a.checkpoint, data, root)
    freeze(model)
    items = pd.read_parquet(root / "basket_input/items.parquet").sort_values("item_id").reset_index(drop=True)
    J = len(items)
    cat = items.cat_id.to_numpy()
    name = (items.COMMODITY_DESC.str.strip() + " / " + items.SUB_COMMODITY_DESC.str.strip()).to_numpy()
    comm = items.COMMODITY_DESC.str.strip().to_numpy()
    dept = items.DEPARTMENT.str.strip().to_numpy()
    train = np.flatnonzero(data["trip_split"] == 0)
    lp = data["line_ptr"]
    baskets = [np.unique(data["line_item"][lp[t]:lp[t + 1]]) for t in train]
    report = {"checkpoint": str(a.checkpoint)}

    # ---- interaction embeddings ----------------------------------------------------------------
    phi = model.phi_r().numpy()
    counts = np.bincount(data["line_item"][np.concatenate([np.arange(lp[t], lp[t + 1]) for t in train])], minlength=J)
    top = np.argsort(-counts)[:a.top_products]
    pos = {j: i for i, j in enumerate(top)}
    T = len(baskets)
    single = np.zeros(len(top))
    pair = np.zeros((len(top), len(top)))
    for bk in baskets:
        idx = np.array([pos[j] for j in bk if j in pos], dtype=np.int64)
        single[idx] += 1
        if len(idx) > 1:
            pair[np.ix_(idx, idx)] += 1
    iu = np.triu_indices(len(top), 1)
    diff_cat = cat[top][iu[0]] != cat[top][iu[1]]
    obs = pair[iu]
    expect = single[iu[0]] * single[iu[1]] / T
    ok = diff_cat & (obs >= 5)
    lift = np.log(obs[ok] / expect[ok])
    coupling = (phi[top] @ phi[top].T)[iu][ok]
    report["interaction"] = {
        "rank": int(phi.shape[1]),
        "coupling_vs_log_lift_other_category_pairs": {"pairs": int(ok.sum()),
                                                      "pearson": float(np.corrcoef(coupling, lift)[0, 1]),
                                                      "spearman": spearman(coupling, lift)},
        "coupling_sd_nats": float(coupling.std()),
        "norm_by_department": {d: round(float(np.linalg.norm(phi[dept == d], axis=1).mean()), 3)
                               for d in pd.Series(dept).value_counts().index[:8]}}
    examples = {}
    for query in ("HOT DOG BUNS", "FRZN MEAT/MEAT DINNERS", "BABY FOODS", "DIAPERS & DISPOSABLES", "BEERS/ALES",
                  "DRY NOODLES/PASTA", "PASTA SAUCE", "TORTILLA/NACHO CHIPS", "CAT FOOD", "SALAD MIX"):
        cand = [j for j in top if comm[j] == query]
        if not cand:
            continue
        j = cand[0]
        c = phi @ phi[j]
        others = [k for k in np.argsort(-c) if cat[k] != cat[j] and counts[k] >= 50][:5]
        examples[name[j]] = [[name[k], round(float(c[k]), 3)] for k in others]
    report["interaction"]["strongest_partners_in_other_categories"] = examples
    U, S, Vt = np.linalg.svd(phi - phi.mean(0), full_matrices=False)
    w = counts / counts.sum()
    directions = []
    for d in range(3):
        score = pd.Series(U[:, d] * S[d]).groupby(comm).apply(lambda x: float(np.average(x, weights=w[x.index] + 1e-12)))
        directions.append({"share_of_variance": round(float(S[d] ** 2 / (S ** 2).sum()), 3),
                           "positive": score.sort_values(ascending=False).index[:6].tolist(),
                           "negative": score.sort_values().index[:6].tolist()})
    report["interaction"]["leading_directions"] = directions

    # ---- customer embeddings -------------------------------------------------------------------
    theta = model.theta_c().numpy()
    alpha = model.alpha[:, :-1].numpy()
    alpha = alpha - alpha.mean(0)
    users = data["trip_user"][train]
    size = np.array([len(b) for b in baskets])
    H = theta.shape[0]
    mean_size = np.bincount(users, size, minlength=H) / np.maximum(np.bincount(users, minlength=H), 1)
    have = np.bincount(users, minlength=H) > 0
    depts = pd.Series(dept).value_counts().index[:15].tolist()
    dcode = np.array([depts.index(x) if x in depts else -1 for x in dept])
    share = np.zeros((H, len(depts)))
    for u, bk in zip(users, baskets):
        dd = dcode[bk]
        np.add.at(share[u], dd[dd >= 0], 1.0)
    share = share / np.maximum(share.sum(1, keepdims=True), 1)
    base_share = share[have].mean(0)
    taste = theta[:, :-1] @ alpha.T                                    # H x J
    dtaste = np.stack([taste[:, dcode == k].mean(1) for k in range(len(depts))], 1)
    rank_corr = [spearman(dtaste[h], share[h] / base_share) for h in np.flatnonzero(have)]
    report["customer"] = {
        "households": int(have.sum()), "dimension": int(theta.shape[1]),
        "size_shift_vs_mean_basket_size_spearman": spearman(theta[have, -1], mean_size[have]),
        "taste_ranks_own_department_overindex_spearman_median": float(np.median(rank_corr)),
        "taste_ranks_own_department_overindex_share_positive": float(np.mean(np.array(rank_corr) > 0))}

    # demographics
    raw = a.raw
    tx = pd.read_csv(raw / "transaction_data.csv", usecols=["BASKET_ID", "household_key"]).drop_duplicates()
    bmap = pd.read_parquet(root / "basket_input/baskets.parquet", columns=["BASKET_ID", "user_id"]).drop_duplicates()
    umap = bmap.merge(tx, on="BASKET_ID").drop_duplicates("user_id").set_index("user_id").household_key
    demo = pd.read_csv(raw / "hh_demographic.csv").set_index("household_key")
    hh = pd.Series(np.arange(H)).map(umap)
    dem = demo.reindex(hh.to_numpy())
    dem.index = np.arange(H)
    income = dem["classification_3"].str.extract(r"(\d+)")[0].astype(float)
    kids = dem["KID_CATEGORY_DESC"].map(lambda x: np.nan if pd.isna(x) else (0.0 if "None" in str(x) else 1.0))
    known = income.notna() & have
    Ut, St, Vtt = np.linalg.svd(theta[:, :-1] - theta[:, :-1].mean(0), full_matrices=False)
    demo_dirs = []
    for d in range(3):
        sc = Ut[:, d] * St[d]
        prod_score = alpha @ Vtt[d]                                     # products' loading on the direction
        csc = pd.Series(prod_score).groupby(comm).mean()
        demo_dirs.append({"share_of_variance": round(float(St[d] ** 2 / (St ** 2).sum()), 3),
                          "products_high": csc.sort_values(ascending=False).index[:6].tolist(),
                          "products_low": csc.sort_values().index[:6].tolist(),
                          "corr_income": float(np.corrcoef(sc[known], income[known])[0, 1]),
                          "corr_children": float(np.corrcoef(sc[known & kids.notna()], kids[known & kids.notna()])[0, 1]),
                          "corr_mean_basket_size": float(np.corrcoef(sc[have], mean_size[have])[0, 1])})
    report["customer"]["leading_directions"] = demo_dirs
    report["customer"]["households_with_demographics"] = int(known.sum())

    # ---- price embeddings ----------------------------------------------------------------------
    G = softplus(model.gamma).numpy()
    B = softplus(model.beta).numpy()
    g_mean_h = G @ B.mean(0)                                           # household's mean g over products
    g_mean_j = B @ G[have].mean(0)                                     # product's mean g over households
    kappa = float(softplus(model.price_kappa))
    gm = G[have] @ B[top[:200]].T
    report["price"] = {
        "kappa_relative_price": kappa,
        "g_household_mean_quantiles": [float(np.quantile(g_mean_h[have], q)) for q in (0.1, 0.5, 0.9)],
        "g_product_mean_quantiles": [float(np.quantile(g_mean_j, q)) for q in (0.1, 0.5, 0.9)],
        "share_of_g_variance_household_vs_product": {
            "household": float(gm.mean(1).var() / gm.var()), "product": float(gm.mean(0).var() / gm.var())},
        "household_g_by_income_level": {int(k): round(float(v), 3) for k, v in
                                        pd.Series(g_mean_h[known]).groupby(income[known].to_numpy()).mean().items()},
        "household_g_corr_income": float(np.corrcoef(g_mean_h[known], income[known])[0, 1]),
        "household_g_children_vs_none": [float(g_mean_h[known & (kids == 1)].mean()), float(g_mean_h[known & (kids == 0)].mean())],
        "household_g_corr_mean_basket_size": float(np.corrcoef(g_mean_h[have], mean_size[have])[0, 1]),
        "product_g_by_department": {d: round(float(g_mean_j[dept == d].mean()), 3) for d in depts[:10]},
        "product_g_private_vs_national": {b: round(float(g_mean_j[items.BRAND.str.strip().to_numpy() == b].mean()), 3)
                                         for b in ("Private", "National")},
        "most_sensitive_commodities": pd.Series(g_mean_j).groupby(comm).mean()[pd.Series(comm).value_counts() >= 10]
                                      .sort_values(ascending=False).index[:8].tolist(),
        "least_sensitive_commodities": pd.Series(g_mean_j).groupby(comm).mean()[pd.Series(comm).value_counts() >= 10]
                                       .sort_values().index[:8].tolist()}
    a.out.write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
