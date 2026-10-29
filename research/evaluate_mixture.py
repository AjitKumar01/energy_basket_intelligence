#!/usr/bin/env python3
"""Evaluate mixture checkpoints: recommendation, basket generation and price counterfactuals.

The protocols are those of the evaluators removed in d716f95 (checks/recommendation.py,
scripts/evaluate.py, checks/discount_promotion.py), so results compare with earlier ones; only
the treatment of the taste z changes, to the mixture form p(S) = int N(z; 0, I) p(S | z) dz.
Validation weeks only (trip_split == 1); the test weeks are never read.

    recommend   hide one product of each basket (>= 2 products) and rank every offered product
                by log p(rest + j) - log p(rest) = log E[exp(dE_z(j)) | rest], with
                dE_z(j) = d_j - s.ds_j - |ds_j|^2 / 2 + z.ds_j: d_j is the exact change of E (the
                normaliser cancels), ds_j the change of the basket field, and the expectation is
                over the posterior of z given the rest (importance draws).  MRR, recall@k and
                median rank with household-clustered standard errors; popularity on the same cases.
    generate    one basket per validation trip (z ~ N(0, I), then S | z exactly), per seed; a
                classifier tells real from generated baskets (full basket and category presence,
                AUC 0.5 = indistinguishable), and the size and category-count distributions.
    price       synthetic world with causal truth (--world): each scenario multiplies one
                product's price; the relative change of the product's purchases, the rest of its
                subcategory, the rest of its category, and the basket-size change, against the
                truth.  Incidence = mean over common z ~ N(0, I) draws of the exact P_z(j in S).
    discount    real data: discount events of the validation weeks (--events, built by the old
                events step); the model's within-category share response to the discount-depth
                change (depth slope), to display and mailer, and its agreement with the observed
                share changes.  Per-z exact incidences, averaged over common z ~ N(0, I) draws.

    ENERGY_MODEL_DATA_ROOT=<bundle> python3 -u research/evaluate_mixture.py recommend CK.pt --out OUT.json
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
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)

from v4 import mixture  # noqa: E402
from v4.bundle import data_root, item_categories, load_bundle, supported_trips  # noqa: E402
from v4.checkpoints import freeze, load_version4  # noqa: E402
from v4.exact import incidence as exact_incidence  # noqa: E402
from v4.price import changed_price_context, own_price_tilt  # noqa: E402
from v4.reporting import classifier_auc, strict_json_dumps  # noqa: E402

VALIDATION = 1


def log(msg):
    print(f"[eval] {msg}", flush=True)


def load(ck, data, root):
    model, batcher, blob, meta = load_version4(ck, data, root)
    return freeze(model), batcher, blob, meta


def household_se(values, households):
    values, households = np.asarray(values, float), np.asarray(households)
    codes, idx = np.unique(households, return_inverse=True)
    resid = np.bincount(idx, values - values.mean())
    G = len(codes)
    return float(math.sqrt(G / max(G - 1, 1) * np.sum(resid ** 2)) / len(values))


# ---- recommendation ------------------------------------------------------------------------------
def conditional_log_odds(model, ix, X):
    """For every slot j: d_j = E(S with j) - E(S without j) for the basket X (the old exact formula,
    v4/training.py before d716f95), and the change ds_j of the basket field s when j is added."""
    x = X.double()
    ph = model.phi_r()[ix.item]
    n_row = torch.zeros(ix.n_rows).index_add(0, ix.row_of, x)
    others = n_row[ix.row_of] - x
    cat = ix.row_cat[ix.row_of]
    s = model.basket_field(ix, X)
    psi = model.psi_cat[cat]
    dev = ph - psi
    has_other = (others >= 1.0).double()
    vec = torch.where(has_other[:, None] > 0.5, dev, ph)                # ds_j
    s_without = s[ix.item_trip] - x[:, None] * vec
    pair = (vec * s_without).sum(-1) - has_other * (psi * dev).sum(-1)
    f = model.f_extra[cat]
    category = (model.rho_c[cat] * (model.pair_feature(others + 1) - model.pair_feature(others))
                + ((others == 1).double() * f[:, 0] + (others == 2).double() * (f[:, 1] - f[:, 0])
                   + (others == 3).double() * (f[:, 2] - f[:, 1])))
    opens = others == 0
    k_trip = torch.zeros(ix.B, dtype=torch.long).index_add(0, ix.row_trip, (n_row > 0).long())
    k_without = (k_trip[ix.item_trip] - X.long() * opens.long()).clamp(min=0)
    rho0 = model.rho_0()
    size = opens.double() * (rho0[(k_without + 1).clamp(max=model.nmax)] - rho0[k_without])
    b = model.slot_utilities(ix)
    d = b + pair - category - size
    iv = model.iv_shift(ix, b_slot=b)
    if iv is not None:
        d = d + opens.double() * iv[ix.row_of]
    return d, vec, s_without


def posterior_draws(model, ix, X, gen, K):
    """Importance draws (K, B, r) and normalised weights (K, B) of p(z | basket X)."""
    b0 = model.slot_utilities(ix)
    iv = model.iv_shift(ix, b_slot=b0)
    s = model.basket_field(ix, X)
    post = mixture.Posterior(model, ix, iv, b0, s)
    m, _, _ = post.mode(iters=30)
    zs, lr = post.sample(m, post.spread(m, gen), gen, K)
    lw = torch.stack([(zs[k] * s).sum(1) - post.log_z(zs[k]) - 0.5 * zs[k].square().sum(1) - lr[k]
                      for k in range(K)])
    return zs, torch.softmax(lw, 0)


def midrank(score, position):
    target = score[position]
    return 1.0 + np.count_nonzero(score > target) + 0.5 * (np.count_nonzero(score == target) - 1)


def rank_metrics(ranks, households):
    ranks = np.asarray(ranks, float)
    rr = 1.0 / ranks
    out = {"cases": int(len(ranks)), "mrr": float(rr.mean()), "mrr_se": household_se(rr, households),
           "median_rank": float(np.median(ranks)), "mean_rank": float(ranks.mean())}
    for k in (5, 10, 20, 100):
        out[f"recall_at_{k}"] = float((ranks <= k).mean())
    return out


@torch.no_grad()
def recommend(a, data, root):
    from initialize import popularity_logits
    model, batcher, _, _ = load(a.checkpoint, data, root)
    population = np.flatnonzero((data["trip_split"] == VALIDATION) & (data["trip_nlines"] <= model.nmax))
    trips = population[np.random.default_rng(a.seed).permutation(len(population))[:a.trips]]
    popularity = popularity_logits(data, np.flatnonzero(data["trip_split"] == 0)).numpy()
    rng = np.random.default_rng(a.seed + 17)
    gen = torch.Generator().manual_seed(a.seed + 23)
    ranks, plain_ranks, pop_ranks, households = [], [], [], []
    t0 = time.time()
    for start in range(0, len(trips), 24):
        batch = batcher.make(trips[start:start + 24])
        batch.bind(model)
        ix = batch.ix
        cases = []
        rest_mask = torch.zeros(len(ix.item), dtype=torch.bool)
        for b in range(ix.B):
            observed = torch.unique(batch.line_item[batch.line_trip == b])
            if observed.numel() < 2:
                continue
            hidden = int(observed[int(rng.integers(observed.numel()))])
            rest = observed[observed != hidden]
            rest_mask |= (ix.item_trip == b) & torch.isin(ix.item, rest)
            cases.append((b, hidden))
        d, ds, s_w = conditional_log_odds(model, ix, rest_mask)
        zs, w = posterior_draws(model, ix, rest_mask, gen, a.K)
        base = d - (s_w * ds).sum(-1) - 0.5 * ds.square().sum(-1)
        lin = torch.einsum("kti,ti->kt", zs[:, ix.item_trip], ds)             # z_k . ds_j, (K, slots)
        score = base + torch.logsumexp(lin + w[:, ix.item_trip].clamp_min(1e-300).log(), 0)
        for b, hidden in cases:
            slots = torch.nonzero((ix.item_trip == b) & ~rest_mask, as_tuple=True)[0]
            position = torch.nonzero(ix.item[slots] == hidden, as_tuple=True)[0]
            if position.numel() != 1:
                continue
            pos = int(position[0])
            ranks.append(midrank(score[slots].numpy(), pos))
            plain_ranks.append(midrank(d[slots].numpy(), pos))
            pop_ranks.append(midrank(popularity[ix.item[slots].numpy()], pos))
            households.append(int(data["trip_user"][trips[start + b]]))
        if (start // 24) % 20 == 0:
            log(f"recommend {start + len(batch.trips)}/{len(trips)} trips, MRR so far "
                f"{np.mean(1.0 / np.asarray(ranks)):.4f} ({time.time() - t0:.0f}s)")
    return {"checkpoint": str(a.checkpoint), "split": "validation", "K": a.K,
            "model": rank_metrics(ranks, households),
            "model_without_taste_integral": rank_metrics(plain_ranks, households),
            "popularity": rank_metrics(pop_ranks, households),
            "per_case": {"ranks": ranks, "popularity_ranks": pop_ranks, "households": households}}


# ---- generation ----------------------------------------------------------------------------------
@torch.no_grad()
def generate(a, data, root):
    model, batcher, _, _ = load(a.checkpoint, data, root)
    J, C = int(data["n_item"]), int(data["n_cat"])
    item_cat = item_categories(data)
    trips = supported_trips(data, VALIDATION, model.nmax)
    if a.trips and len(trips) > a.trips:
        trips = np.sort(np.random.default_rng(a.seed).choice(trips, a.trips, replace=False))
    lptr = data["line_ptr"]
    real = [np.sort(data["line_item"][lptr[t]:lptr[t + 1]]) for t in trips]
    groups = np.r_[data["trip_user"][trips], data["trip_user"][trips]]
    batches = [batcher.make(trips[s:s + 256]) for s in range(0, len(trips), 256)]
    rows = []
    for seed in a.seeds:
        gen, generated = torch.Generator().manual_seed(seed), []
        for batch in batches:
            batch.bind(model)
            X = mixture.generate(model, batch.ix, gen, iv=model.iv_shift(batch.ix)).numpy()
            items, trip = batch.ix.item.numpy(), batch.ix.item_trip.numpy()
            generated += [np.sort(items[X & (trip == b)]) for b in range(len(batch.trips))]
        full = classifier_auc(real, generated, groups, J, item_cat, C, "full")
        cat = classifier_auc(real, generated, groups, J, item_cat, C, "category")
        n_real, n_gen = np.array([len(x) for x in real]), np.array([len(x) for x in generated])
        k_real = np.array([len(np.unique(item_cat[x])) for x in real])
        k_gen = np.array([len(np.unique(item_cat[x])) for x in generated])
        row = {"seed": seed, "full_basket_auc": round(full[0], 4), "full_basket_fold_sd": round(full[1], 4),
               "category_presence_auc": round(cat[0], 4), "category_presence_fold_sd": round(cat[1], 4),
               "size_mean": [round(float(n_real.mean()), 3), round(float(n_gen.mean()), 3)],
               "size_q50_q90_q99": [[float(np.quantile(v, q)) for q in (0.5, 0.9, 0.99)] for v in (n_real, n_gen)],
               "categories_mean": [round(float(k_real.mean()), 3), round(float(k_gen.mean()), 3)],
               "share_size_1": [round(float((n_real == 1).mean()), 4), round(float((n_gen == 1).mean()), 4)],
               "share_size_20plus": [round(float((n_real >= 20).mean()), 4), round(float((n_gen >= 20).mean()), 4)]}
        rows.append(row)
        log(json.dumps(row))
    mean = lambda k: round(float(np.mean([r[k] for r in rows])), 4)  # noqa: E731
    return {"checkpoint": str(a.checkpoint), "split": "validation", "trips": int(len(trips)), "seeds": rows,
            "full_basket_auc_mean": mean("full_basket_auc"), "category_presence_auc_mean": mean("category_presence_auc")}


# ---- price scenarios with causal truth (synthetic) -----------------------------------------------
@torch.no_grad()
def price(a, data, root):
    model, batcher, _, _ = load(a.checkpoint, data, root)
    world = a.world
    o = np.load(world / "observed.npz")
    to_model = {int(orig): i for i, orig in enumerate(o["original_item"])}
    truth = json.loads((world / "counterfactual_truth.json").read_text())
    cat, sub = o["category"], o["subcategory"]
    J = int(data["n_item"])
    pool = supported_trips(data, VALIDATION, model.nmax)
    trips = np.sort(np.random.default_rng(77).choice(pool, size=min(a.trips, len(pool)), replace=False))
    gen = torch.Generator().manual_seed(a.seed)
    sets = []
    for s in range(0, len(trips), 256):
        batch = batcher.make(trips[s:s + 256])
        sets.append((batch, [torch.randn(batch.ix.B, model.rank, generator=gen) for _ in range(a.draws)]))

    def incidence(product, multiplier):
        inc, size = np.zeros(J), 0.0
        for batch, zs in sets:
            ix, ctx = batch.ix, batch.ctx
            if product is not None:
                ctx = changed_price_context(batch.ctx, ix.item_trip,
                                            torch.where(ix.item == product, math.log(multiplier), 0.0))
            batch.bind(model, ctx)
            iv = model.iv_shift(ix)
            m = sum(exact_incidence(model, ix, model.slot_utilities_given_z(ix, z), z=z, iv=iv)[1] for z in zs) / len(zs)
            np.add.at(inc, ix.item.numpy(), m.numpy())
            size += float(m.sum())
        return inc, size / len(trips)

    base_inc, base_size = incidence(None, 1.0)
    rel = lambda x, y: float(y / x - 1.0) if x > 0 else None  # noqa: E731
    results = []
    t0 = time.time()
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
        if len(results) % 10 == 0:
            log(f"price scenario {len(results)} ({time.time() - t0:.0f}s)")

    def agreement(key):
        pairs = np.array([(x[key]["truth"], x[key]["model"]) for x in results
                          if x[key]["truth"] is not None and x[key]["model"] is not None])
        t, m = pairs.T
        return {"scenarios": len(pairs), "correlation": float(np.corrcoef(t, m)[0, 1]),
                "mean_abs_error": float(np.abs(t - m).mean()),
                "sign_agreement": float(np.mean(np.sign(t) == np.sign(m))),
                "mean_truth": float(t.mean()), "mean_model": float(m.mean())}

    keys = ("own", "same_subcategory", "rest_of_category", "basket_size_change")
    return {"checkpoint": str(a.checkpoint), "trips": int(len(trips)), "split": "validation", "draws": a.draws,
            "agreement": {k: agreement(k) for k in keys}, "scenarios": results}


# ---- discount and promotion response (real data) -------------------------------------------------
KEY_WEEKS = 128


def weekly_chain_prices(root, items, n_periods):
    import pandas as pd
    weekly = pd.read_parquet(root / "data/price_week.parquet",
                             columns=["PRODUCT_ID", "WEEK_NO", "price", "base_price", "promo_depth"])
    weekly = weekly.merge(items[["PRODUCT_ID", "item_id"]], on="PRODUCT_ID", how="inner", validate="many_to_one")
    weekly = weekly[weekly.WEEK_NO.between(0, n_periods)]
    J = int(items.item_id.max()) + 1
    out = {}
    for name in ("price", "base_price"):
        panel = np.full((J, n_periods + 1), np.nan)
        panel[weekly.item_id.to_numpy(np.int64), weekly.WEEK_NO.to_numpy(np.int64)] = weekly[name].to_numpy(np.float64)
        out[name] = pd.DataFrame(panel).ffill(axis=1).bfill(axis=1).to_numpy()
    return out["price"], out["base_price"]


def representative_trips(data, events, nmax, per_cell, seed):
    week, store, n = data["trip_week"], data["trip_store"], data["trip_nlines"]
    cur = set(zip(events.store_id.tolist(), events.WEEK_NO.tolist()))
    needed = cur | set(zip(events.store_id.tolist(), events.previous_WEEK_NO.tolist()))
    cur_keys = {s * KEY_WEEKS + w for s, w in cur}
    lo, hi = int(events.previous_WEEK_NO.min()), int(events.WEEK_NO.max())
    window = np.flatnonzero((week >= lo) & (week <= hi) & (n >= 1) & (n <= nmax))
    cell = store[window].astype(np.int64) * KEY_WEEKS + week[window].astype(np.int64)
    rng = np.random.default_rng(seed)
    first, second = [], []
    for key in sorted(s * KEY_WEEKS + w for s, w in needed):
        trips = window[cell == key]
        if len(trips) == 0:
            raise RuntimeError(f"no eligible trip for store-week {key}")
        if len(trips) > per_cell:
            trips = np.sort(rng.choice(trips, size=per_cell, replace=False))
        (first if key in cur_keys else second).append(trips)
    return np.concatenate([np.sort(np.concatenate(first)), np.sort(np.concatenate(second)) if second
                           else np.zeros(0, dtype=np.int64)]), cur_keys


@torch.no_grad()
def discount(a, data, root):
    import pandas as pd
    model, batcher, _, meta = load(a.checkpoint, data, root)
    events = pd.read_parquet(a.events / "events.parquet")
    events = events[events.split == "validation"].reset_index(drop=True)
    empirical = json.loads((a.events / "empirical.json").read_text())["validation"]
    items = pd.read_parquet(root / "basket_input/items.parquet", columns=["PRODUCT_ID", "item_id", "cat_id"]
                            ).sort_values("item_id")
    merch_cat = items.cat_id.to_numpy(np.int64)
    n_weeks = int(data["trip_week"].max()) + 1
    price_w, base_w = weekly_chain_prices(root, items, n_weeks - 1)
    log_price = np.log(price_w)
    disc = np.minimum(np.log(price_w / base_w), 0.0)
    trips, cur_keys = representative_trips(data, events, int(meta["nmax"]), a.per_cell, a.seed + 101)
    log(f"{len(events)} validation events, {len(trips)} trips")
    J, S, Cm = int(data["n_item"]), int(data["n_store"]), int(merch_cat.max()) + 1
    w0 = int(events.previous_WEEK_NO.min())
    W = int(events.WEEK_NO.max()) - w0 + 1
    acc = {n: np.zeros((J, S, W)) for n in ("fact", "single", "chain", "disp_on", "disp_off", "mail_on", "mail_off")}
    acc_cat = {n: np.zeros((Cm, S, W)) for n in ("fact", "chain")}
    trips_per_cell = np.zeros((S, W), dtype=np.int64)
    gen = torch.Generator().manual_seed(a.seed + 17)
    kappa = float(torch.nn.functional.softplus(model.price_kappa))
    t0 = time.time()
    for s in range(0, len(trips), 128):
        batch = batcher.make(trips[s:s + 128])
        ix = batch.ix
        st = data["trip_store"][batch.trips].astype(np.int64)
        wk = data["trip_week"][batch.trips].astype(np.int64)
        np.add.at(trips_per_cell, (st, wk - w0), 1)
        slot_item = ix.item.numpy()
        slot_store, slot_week = st[ix.item_trip.numpy()], wk[ix.item_trip.numpy()]
        where = (slot_item, slot_store, slot_week - w0)
        where_cat = (merch_cat[slot_item], slot_store, slot_week - w0)
        zs = [torch.randn(ix.B, model.rank, generator=gen) for _ in range(a.draws)]
        batch.bind(model)
        iv = model.iv_shift(ix)
        P0 = torch.stack([exact_incidence(model, ix, model.slot_utilities_given_z(ix, z), z=z, iv=iv)[1] for z in zs])
        p0 = P0.mean(0)
        np.add.at(acc["fact"], where, p0.numpy())
        np.add.at(acc_cat["fact"], where_cat, p0.numpy())
        for name, weight in zip(("disp", "mail"), model.promo_weights()):
            w_slot, state = weight[ix.item].double(), batch.ctx[name].double()
            np.add.at(acc[f"{name}_on"], where, own_price_tilt(P0, w_slot * (1.0 - state)).mean(0).numpy())
            np.add.at(acc[f"{name}_off"], where, own_price_tilt(P0, -w_slot * state).mean(0).numpy())
        if not any(int(x) * KEY_WEEKS + int(y) in cur_keys for x, y in zip(st, wk)):
            continue
        prev = np.maximum(slot_week - 1, 1)
        d_price = torch.from_numpy(log_price[slot_item, prev] - log_price[slot_item, slot_week])
        d_disc = torch.from_numpy(disc[slot_item, prev] - disc[slot_item, slot_week])
        g = model.price_coefficients(ix.item, ix.item_trip)
        count = torch.bincount(ix.item_trip, minlength=ix.B).double()
        du = -g * (kappa + (1.0 - kappa) / count[ix.item_trip]) * d_price
        if "dsc" in batch.ctx:
            if model.disc_kappa is not None:
                du = du - g * (float(torch.nn.functional.softplus(model.disc_kappa)) - kappa) * d_disc
            if model.disc_common is not None:
                du = du - float(model.disc_common) * d_disc
        np.add.at(acc["single"], where, own_price_tilt(P0, du).mean(0).numpy())
        ctx = changed_price_context(batch.ctx, ix.item_trip, d_price, discount_change=d_disc)
        batch.bind(model, ctx)
        iv_c = model.iv_shift(ix)
        p_chain = sum(exact_incidence(model, ix, model.slot_utilities_given_z(ix, z), z=z, iv=iv_c)[1] for z in zs) / len(zs)
        batch.bind(model)
        np.add.at(acc["chain"], where, p_chain.numpy())
        np.add.at(acc_cat["chain"], where_cat, p_chain.numpy())
        log(f"discount {min(s + 128, len(trips))}/{len(trips)} trips ({time.time() - t0:.0f}s)")

    i, c = events.item_id.to_numpy(np.int64), events.cat_id.to_numpy(np.int64)
    s_, w, wp = (events.store_id.to_numpy(np.int64), events.WEEK_NO.to_numpy(np.int64) - w0,
                 events.previous_WEEK_NO.to_numpy(np.int64) - w0)
    depth = events.promotion_depth_change.to_numpy(np.float64)
    item, catv = acc["fact"][i, s_, w], acc_cat["fact"][c, s_, w]
    share = np.where(catv > 0, item / np.maximum(catv, 1e-30), 0.0)
    single = acc["single"][i, s_, w]
    rest = np.maximum(catv - item, 0.0)
    share_single = np.where(rest + single > 0, single / np.maximum(rest + single, 1e-30), 0.0)
    chain_cat = acc_cat["chain"][c, s_, w]
    share_chain = np.where(chain_cat > 0, acc["chain"][i, s_, w] / np.maximum(chain_cat, 1e-30), 0.0)

    def slope(after, before):
        ok = (after > 0) & (before > 0)
        y, x = np.log(after[ok]) - np.log(before[ok]), depth[ok]
        return {"events": int(ok.sum()), "slope": float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))}

    def promotion(name):
        on, off = acc[f"{name}_on"][i, s_, w], acc[f"{name}_off"][i, s_, w]
        ok = (on > 0) & (off > 0) & (rest > 0)
        share_ratio = np.log(on[ok] / (rest[ok] + on[ok])) - np.log(off[ok] / (rest[ok] + off[ok]))
        weight = item[ok] / item[ok].sum()
        return {"events": int(ok.sum()), "choice_share_log_ratio_mean": float(share_ratio.mean()),
                "choice_share_log_ratio_purchase_weighted": float((weight * share_ratio).sum())}

    def agreement(pred):
        informative = events.total_purchases.to_numpy() > 0
        p, t = pred[informative], events.observed_choice_share_change.to_numpy(np.float64)[informative]
        nonzero = (np.abs(p) > 0) & (np.abs(t) > 0)
        up = depth[informative] >= 0.05
        return {"events": int(informative.sum()), "pearson": float(np.corrcoef(p, t)[0, 1]),
                "sign_agreement_nonzero_observed": float(np.mean(np.sign(p[nonzero]) == np.sign(t[nonzero]))),
                "mean_predicted_change_discount_increase": float(p[up].mean()),
                "mean_observed_change_discount_increase": float(t[up].mean())}

    emp = {k: v["estimate"] for k, v in empirical["effects"].items()}
    emp_ci = {k: v.get("product_clustered_95_interval") for k, v in empirical["effects"].items()}
    return {"checkpoint": str(a.checkpoint), "split": "validation", "trips": int(len(trips)), "draws": a.draws,
            "empirical_validation": emp, "empirical_validation_product_95": emp_ci,
            "depth_slope": {"single_item_choice_share": slope(share, share_single),
                            "single_item_purchase_rate": slope(item, single),
                            "chain_week_choice_share": slope(share, share_chain)},
            "promotion_on_vs_off": {"display": promotion("disp"), "mailer": promotion("mail")},
            "share_change_agreement": {"single_item": agreement(share - share_single),
                                       "chain_week": agreement(share - share_chain)},
            "seconds": time.time() - t0}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("command", choices=("recommend", "generate", "price", "discount"))
    p.add_argument("checkpoint", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--trips", type=int, default=None, help="recommend 2000, generate 0 (all), price 400")
    p.add_argument("--K", type=int, default=32, help="recommend: posterior draws per basket")
    p.add_argument("--seeds", type=int, nargs="+", default=[41006, 41007, 41008])
    p.add_argument("--draws", type=int, default=64, help="price, discount: z draws per trip")
    p.add_argument("--world", type=Path, help="price: synthetic world directory")
    p.add_argument("--events", type=Path, help="discount: directory with events.parquet and empirical.json")
    p.add_argument("--per-cell", type=int, default=1, help="discount: trips per store-week")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    a.trips = a.trips if a.trips is not None else {"recommend": 2000, "generate": 0, "price": 400, "discount": 0}[a.command]
    a.seed = a.seed if a.seed is not None else {"recommend": 2560202, "generate": 41001, "price": 41005,
                                                "discount": 68101}[a.command]
    torch.set_num_threads(a.threads)
    root = data_root(None)
    data = load_bundle(root)
    report = {"recommend": recommend, "generate": generate, "price": price, "discount": discount}[a.command](a, data, root)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(strict_json_dumps(report))
    summary = {k: v for k, v in report.items() if k not in ("per_case", "scenarios", "seeds")}
    log(json.dumps(summary)[:3000])


if __name__ == "__main__":
    main()
