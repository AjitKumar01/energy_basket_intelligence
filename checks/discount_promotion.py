#!/usr/bin/env python3
"""Real-data check: does the model respond to discounts, displays and mailers as shoppers do?

Needs a bundle with ``data/price_week.parquet`` (chain-wide weekly price, regular price
and promo_depth per product), ``basket_input/promo.npz`` and ``basket_input/items.parquet``.

Step 1, ``events`` (observed data only, no model):
    An event is (product, store, week w) where the product's chain discount depth changed by
    at least --min-depth-change from week w-1, the store sold something in both weeks and
    at least --min-trips trips both weeks.  Per event: purchases in w-1 and w, display and
    mailer in both weeks, and the store's baskets containing the product's category.
    Empirical effects: a conditional binomial per split.  Given the event's purchases in the
    two weeks, m = y0 + y1, the week-w share is
        y1 ~ Binomial(m, sigmoid(offset + b_depth * d_depth + b_disp * d_display + b_mail * d_mailer)),
    offset = log(category baskets w / category baskets w-1).  Conditioning on m removes the
    product's baseline; the offset removes the category's week-to-week change, so the
    coefficients are effects on the product's share *within* its category.  Standard
    errors are clustered by product, store and week.

Step 2, ``model``:
    For the events of one split (--split, validation by default; test is opened once per
    final model), the model's incidence of every product is computed on one trip per
    (store, week) involved, using chain-wide prices only (store prices exist only where
    the product sold).  Then:
      depth slope        slope of the model's log(share at w / share at w-1 prices) on the
                         depth change, comparable to b_depth.  Three versions: only the
                         product's own price moved back to w-1 ("single item", share and
                         purchase rate) and every chain price moved back ("chain week").
      promotion on/off   log ratio of the product's within-category share with its display
                         (or mailer) switched on vs off, comparable to b_disp and b_mail.
      agreement          correlation and sign agreement of the predicted and observed share
                         changes on events with purchases.
    A change to one product's utility tilts its incidence exactly:
        p -> p e^du / (1 - p + p e^du);
    the single-item and promotion results use this.  For the single item, du includes the
    product's share of the trip-mean price shift but not the small shift of the other
    products (nor, with M4c, of the category's inclusive value).

Writes events.parquet and empirical.json (step 1) and the model report (step 2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

torch.set_default_dtype(torch.float64)
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402

from v4.bundle import data_root, file_sha256, load_bundle  # noqa: E402
from v4.checkpoints import batcher_for, freeze, load_model  # noqa: E402
from v4.exact import incidence as exact_incidence  # noqa: E402
from v4.price import changed_price_context, own_price_tilt  # noqa: E402
from v4.reporting import strict_json_dumps  # noqa: E402
from v4.sampling import draw_from_model, gibbs  # noqa: E402

COVARIATES = ("promotion_depth_change", "display_change", "mailer_change")
KEY_WEEKS = 128


def log(message):
    print(f"[discount] {message}", flush=True)


def lookup(keys, values, query):
    """values at sorted keys, zero where the key is absent."""
    pos = np.searchsorted(keys, query)
    safe = np.minimum(pos, len(keys) - 1)
    found = (pos < len(keys)) & (keys[safe] == query)
    out = np.zeros(len(query), dtype=values.dtype)
    out[found] = values[pos[found]]
    return out


def weekly_chain_prices(root, items, n_periods, fill_all=False):
    """price, regular price and promo depth per [product, week], carried forward then back
    (for products with all three series, or every product with ``fill_all``)."""
    weekly = pd.read_parquet(root / "data/price_week.parquet",
                             columns=["PRODUCT_ID", "WEEK_NO", "price", "base_price", "promo_depth"])
    weekly = weekly.merge(items[["PRODUCT_ID", "item_id"]], on="PRODUCT_ID", how="inner", validate="many_to_one")
    weekly = weekly[weekly.WEEK_NO.between(0, n_periods)]
    J = int(items.item_id.max()) + 1
    out = {}
    for name in ("price", "base_price", "promo_depth"):
        panel = np.full((J, n_periods + 1), np.nan)
        panel[weekly.item_id.to_numpy(np.int64), weekly.WEEK_NO.to_numpy(np.int64)] = weekly[name].to_numpy(np.float64)
        out[name] = panel
    reliable = np.all([np.isfinite(p).any(axis=1) for p in out.values()], axis=0)
    rows = np.ones(J, dtype=bool) if fill_all else reliable
    for name, panel in out.items():
        panel[rows] = pd.DataFrame(panel[rows]).ffill(axis=1).bfill(axis=1).to_numpy()
    return out["price"], out["base_price"], out["promo_depth"], reliable


# ---- step 1: events and empirical effects ----------------------------------------------------
def build_events(root, min_depth_change=0.05, min_trips=16):
    bi = root / "basket_input"
    baskets = pd.read_parquet(bi / "baskets.parquet", columns=["BASKET_ID", "WEEK_NO", "item_id", "store_id"])
    items = pd.read_parquet(bi / "items.parquet", columns=["PRODUCT_ID", "item_id", "n_train_lines", "cat_id"]
                            ).sort_values("item_id").reset_index(drop=True)
    meta = json.loads((bi / "meta.json").read_text())
    C = int(items.cat_id.max()) + 1
    item_cat, item_lines = items.cat_id.to_numpy(np.int64), items.n_train_lines.to_numpy(np.int64)
    with np.load(bi / "store_price.npz") as sp:
        carried, S = sp["carried"].astype(bool), int(sp["n_stores"])

    # Trips per (store, week); baskets containing each category per (category, store, week).
    trips = baskets[["BASKET_ID", "store_id", "WEEK_NO"]].drop_duplicates().groupby(["store_id", "WEEK_NO"]).size()
    trip_count = np.zeros((S, KEY_WEEKS), dtype=np.int64)
    trip_count[trips.index.get_level_values(0), trips.index.get_level_values(1)] = trips.to_numpy()
    with_cat = baskets.assign(cat_id=item_cat[baskets.item_id.to_numpy(np.int64)])
    cat_trips = with_cat[["BASKET_ID", "cat_id", "store_id", "WEEK_NO"]].drop_duplicates().groupby(
        ["cat_id", "store_id", "WEEK_NO"]).size()
    cat_count = np.zeros((C, S, KEY_WEEKS), dtype=np.int64)
    cat_count[tuple(cat_trips.index.get_level_values(i) for i in range(3))] = cat_trips.to_numpy()
    # Baskets buying each (product, store, week).
    bought = baskets.groupby(["item_id", "store_id", "WEEK_NO"]).BASKET_ID.nunique().sort_index()
    bought_key = ((bought.index.get_level_values(0).to_numpy(np.int64) * S
                   + bought.index.get_level_values(1).to_numpy(np.int64)) * KEY_WEEKS
                  + bought.index.get_level_values(2).to_numpy(np.int64))
    order = np.argsort(bought_key)
    bought_key, bought_value = bought_key[order], bought.to_numpy(np.int64)[order]

    n_periods = int(meta["analysis_last_week"])
    price, base, depth, reliable = weekly_chain_prices(root, items, n_periods)
    with np.load(bi / "promo.npz") as promo:
        order = np.argsort(promo["keys"].astype(np.int64))
        promo_key = promo["keys"].astype(np.int64)[order]
        display, mailer = promo["disp"][order], promo["mail"][order]

    # Candidate (product, week): depth changed enough between w-1 and w.
    first_week = max(2, int(meta.get("analysis_first_week", 1)), 10)
    last_week = min(n_periods, 101)
    week = np.arange(1, n_periods + 1)[None, :]
    cand = (reliable[:, None] & (week >= first_week) & (week <= last_week)
            & (np.abs(depth[:, 1:] - depth[:, :-1]) >= min_depth_change))
    cand_item, cand_week = np.nonzero(cand)
    cand_week = cand_week + 1
    parts = []
    for item, w in zip(cand_item, cand_week):
        stores = np.flatnonzero(carried[item] & (trip_count[:, w] > 0) & (trip_count[:, w - 1] > 0))
        if len(stores):
            parts.append(np.column_stack([np.full(len(stores), item), stores, np.full(len(stores), w)]))
    item, store, w = np.concatenate(parts).T.astype(np.int64)
    prev = w - 1
    cat = item_cat[item]
    key, prev_key = (item * S + store) * KEY_WEEKS + w, (item * S + store) * KEY_WEEKS + prev

    ev = pd.DataFrame({
        "item_id": item, "cat_id": cat, "store_id": store, "WEEK_NO": w, "previous_WEEK_NO": prev,
        "n_train_lines": item_lines[item],
        "trips": trip_count[store, w], "previous_trips": trip_count[store, prev],
        "category_baskets": cat_count[cat, store, w], "previous_category_baskets": cat_count[cat, store, prev],
        "purchases": lookup(bought_key, bought_value, key),
        "previous_purchases": lookup(bought_key, bought_value, prev_key),
        "display": lookup(promo_key, display, key).astype(np.int8),
        "previous_display": lookup(promo_key, display, prev_key).astype(np.int8),
        "mailer": lookup(promo_key, mailer, key).astype(np.int8),
        "previous_mailer": lookup(promo_key, mailer, prev_key).astype(np.int8)})
    ev["log_price_change"] = np.log(price[item, w]) - np.log(price[item, prev])
    ev["promotion_depth_change"] = depth[item, w] - depth[item, prev]
    ev["display_change"] = ev.display.astype(np.float64) - ev.previous_display.astype(np.float64)
    ev["mailer_change"] = ev.mailer.astype(np.float64) - ev.previous_mailer.astype(np.float64)
    ev["total_purchases"] = ev.previous_purchases + ev.purchases
    ev["split"] = np.where(ev.WEEK_NO >= int(meta["test_from_week"]), "test",
                           np.where(ev.WEEK_NO >= int(meta["val_from_week"]), "validation", "train"))

    # Keep events whose category was bought in both weeks in a store with enough trips, in
    # categories with at least one training event with purchases.
    keep = ((ev.category_baskets > 0) & (ev.previous_category_baskets > 0)
            & (ev.trips >= min_trips) & (ev.previous_trips >= min_trips))
    ev = ev[keep].copy()
    train_cats = np.sort(ev.loc[(ev.split == "train") & (ev.total_purchases > 0), "cat_id"].unique())
    ev = ev[ev.cat_id.isin(train_cats)].copy()
    ev["choice_offset"] = np.log(ev.category_baskets / ev.previous_category_baskets)
    ev["observed_choice_share_change"] = ev.purchases / ev.category_baskets - ev.previous_purchases / ev.previous_category_baskets
    ev["observed_incidence_change"] = ev.purchases / ev.trips - ev.previous_purchases / ev.previous_trips
    log(f"events: {ev.split.value_counts().to_dict()}")
    return ev


def empirical_effects(frame):
    """Conditional-binomial estimates of the three effects, with clustered intervals."""
    X = frame[list(COVARIATES)].to_numpy(np.float64)
    offset = frame.choice_offset.to_numpy(np.float64)
    y, m = frame.purchases.to_numpy(np.float64), frame.total_purchases.to_numpy(np.float64)
    denom = max(float(m.sum()), 1.0)

    def objective(b):
        lin = offset + X @ b
        return (float(np.sum(m * np.logaddexp(0.0, lin) - y * lin) / denom),
                (X.T @ (m * expit(lin) - y)) / denom)

    fit = minimize(objective, np.zeros(3), method="L-BFGS-B", jac=True, bounds=[(-10.0, 10.0)] * 3,
                   options={"maxiter": 2000, "ftol": 1e-15, "gtol": 1e-11})
    beta = fit.x.copy()
    p = expit(offset + X @ beta)
    hess_inv = np.linalg.inv((X.T * (m * p * (1.0 - p))) @ X)
    scores = X * (y - m * p)[:, None]
    effects = {name: {"estimate": float(beta[i])} for i, name in enumerate(COVARIATES)}
    for cluster, labels in (("product", frame.item_id), ("store", frame.store_id), ("week", frame.WEEK_NO)):
        codes, unique = pd.factorize(labels.to_numpy(np.int64), sort=True)
        G = len(unique)
        summed = np.zeros((G, 3))
        np.add.at(summed, codes, scores)
        se = np.sqrt(np.maximum(np.diag(hess_inv @ (G / max(G - 1, 1) * summed.T @ summed) @ hess_inv), 0.0))
        for i, name in enumerate(COVARIATES):
            effects[name][f"{cluster}_clustered_95_interval"] = [float(beta[i] - 1.96 * se[i]),
                                                                 float(beta[i] + 1.96 * se[i])]
    return {"events": int(len(frame)), "effects": effects, "conditional_log_loss_per_purchase": float(fit.fun)}


# ---- step 2: the model's responses -----------------------------------------------------------
def load_any(path, data, root, initialization, panel=None):
    """An additive or Version-4 checkpoint with chain-only prices (store prices exist only
    where the product sold); ``panel`` overrides the recorded discount panel."""
    model, meta, parent, flags, _ = load_model(path, data, root, initialization=initialization)
    batcher = batcher_for(data, root, parent, recency=flags["recency"], purchase_history=flags["recency"],
                          contract="chain_product_week_only", panel=panel)
    return freeze(model), meta, batcher


def representative_trips(data, events, nmax, per_cell, seed):
    """Up to ``per_cell`` trips in every (store, week) the events need; current-week cells first."""
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


class SlotIncidence:
    """P(j in S) at every slot of a batch: exact for the additive model; for a Version-4
    model, averaged over z draws from chains at the factual context, with a changed context
    reweighting each draw by Z_new(z) / Z_old(z)."""

    def __init__(self, model, batch, gen, burn, draws, thin):
        self.model, self.batch = model, batch
        batch.bind(model)
        self.zs = None
        if model.rank > 0:
            X = draw_from_model(model, batch.ix, gen, burn)
            self.zs = []
            for k in range(draws * thin):
                X = gibbs(model, batch.ix, X, gen)
                if (k + 1) % thin == 0:
                    self.zs.append(model.basket_field(batch.ix, X) + torch.randn(batch.ix.B, model.rank, generator=gen))
        self.base_lz, self.base = self._at(batch.ctx)

    def _at(self, ctx):
        m, ix = self.model, self.batch.ix
        self.batch.bind(m, ctx)
        if self.zs is None:
            lz, p = exact_incidence(m, ix, m.slot_utilities(ix))
            return lz, p
        iv = m.iv_shift(ix)
        out = [exact_incidence(m, ix, m.slot_utilities_given_z(ix, z), z=z, iv=iv) for z in self.zs]
        return torch.stack([o[0] for o in out], 1), torch.stack([o[1] for o in out], 1)

    def factual(self):
        return self.base if self.zs is None else self.base.mean(1)

    def under(self, ctx):
        lz, p = self._at(ctx)
        self.batch.bind(self.model)
        if self.zs is None:
            return p
        w = torch.softmax(lz - self.base_lz, dim=1)
        return (p * w[self.batch.ix.item_trip]).sum(1)


def model_check(args):
    torch.set_num_threads(args.threads)
    root = data_root(args.data_root)
    data = load_bundle(root)
    model, meta, batcher = load_any(args.checkpoint, data, root, args.initialization, args.discount_panel)
    events = pd.read_parquet(args.events / "events.parquet")
    events = events[events.split == args.split].reset_index(drop=True)
    empirical = json.loads((args.events / "empirical.json").read_text())
    items = pd.read_parquet(root / "basket_input/items.parquet", columns=["PRODUCT_ID", "item_id", "cat_id"]
                            ).sort_values("item_id")
    merch_cat = items.cat_id.to_numpy(np.int64)                 # the events' category, not the model's
    n_weeks = int(data["trip_week"].max()) + 1
    price, base_price, _, _ = weekly_chain_prices(root, items, n_weeks - 1, fill_all=True)
    log_price = np.log(price)
    discount = np.minimum(np.log(price / base_price), 0.0)

    trips, cur_keys = representative_trips(data, events, int(meta["nmax"]), args.trips_per_store_week, args.seed + 101)
    log(f"{len(events)} {args.split} events, {len(trips)} trips, rank {model.rank}")
    J, S, Cm = int(data["n_item"]), int(data["n_store"]), int(merch_cat.max()) + 1
    # Accumulators cover only the weeks the events span (week w is stored at w - w0).
    w0 = int(events.previous_WEEK_NO.min())
    W = int(events.WEEK_NO.max()) - w0 + 1
    acc = {name: np.zeros((J, S, W)) for name in
           ("fact", "single", "chain", "disp_on", "disp_off", "mail_on", "mail_off")}
    acc_cat = {name: np.zeros((Cm, S, W)) for name in ("fact", "chain")}
    trips_per_cell = np.zeros((S, W), dtype=np.int64)
    gen = torch.Generator().manual_seed(args.seed + 17)
    kappa = float(torch.nn.functional.softplus(model.price_kappa))
    started = time.time()

    for s in range(0, len(trips), 128):
        batch = batcher.make(trips[s:s + 128])
        ix = batch.ix
        st = data["trip_store"][batch.trips].astype(np.int64)
        wk = data["trip_week"][batch.trips].astype(np.int64)
        np.add.at(trips_per_cell, (st, wk - w0), 1)
        slot_item, slot_store, slot_week = ix.item.numpy(), st[ix.item_trip.numpy()], wk[ix.item_trip.numpy()]
        where = (slot_item, slot_store, slot_week - w0)
        where_cat = (merch_cat[slot_item], slot_store, slot_week - w0)
        inc = SlotIncidence(model, batch, gen, args.burn, args.draws, args.thin)
        p0 = inc.factual()
        np.add.at(acc["fact"], where, p0.numpy())
        np.add.at(acc_cat["fact"], where_cat, p0.numpy())
        for name, weight in zip(("disp", "mail"), model.promo_weights()):
            w_slot, state = weight[ix.item].double(), batch.ctx[name].double()
            np.add.at(acc[f"{name}_on"], where, own_price_tilt(p0, w_slot * (1.0 - state)).numpy())
            np.add.at(acc[f"{name}_off"], where, own_price_tilt(p0, -w_slot * state).numpy())
        if not any(int(a) * KEY_WEEKS + int(b) in cur_keys for a, b in zip(st, wk)):
            continue
        # Move prices (and discount depths) back to the previous week.
        prev = np.maximum(slot_week - 1, 1)
        d_price = torch.from_numpy(log_price[slot_item, prev] - log_price[slot_item, slot_week])
        d_disc = torch.from_numpy(discount[slot_item, prev] - discount[slot_item, slot_week])
        batch.bind(model)
        g = model.price_coefficients(ix.item, ix.item_trip)
        count = torch.bincount(ix.item_trip, minlength=ix.B).double()
        du = -g * (kappa + (1.0 - kappa) / count[ix.item_trip]) * d_price
        if "dsc" in batch.ctx:
            if model.disc_kappa is not None:
                du = du - g * (float(torch.nn.functional.softplus(model.disc_kappa)) - kappa) * d_disc
            if model.disc_common is not None:
                du = du - float(model.disc_common) * d_disc
        np.add.at(acc["single"], where, own_price_tilt(p0, du).numpy())
        p_chain = inc.under(changed_price_context(batch.ctx, ix.item_trip, d_price, discount_change=d_disc))
        np.add.at(acc["chain"], where, p_chain.numpy())
        np.add.at(acc_cat["chain"], where_cat, p_chain.numpy())
        log(f"{min(s + 128, len(trips))}/{len(trips)} trips ({time.time() - started:.0f}s)")

    report = summarize(events, acc, acc_cat, trips_per_cell, model, empirical, w0)
    report.update(checkpoint=str(args.checkpoint), checkpoint_sha256=file_sha256(args.checkpoint),
                  split=args.split, rank=model.rank, trips=len(trips), seconds=time.time() - started)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(strict_json_dumps(report))
    log(f"wrote {args.out}")


def summarize(events, acc, acc_cat, trips_per_cell, model, empirical, w0):
    i, c = events.item_id.to_numpy(np.int64), events.cat_id.to_numpy(np.int64)
    s, w, wp = (events.store_id.to_numpy(np.int64), events.WEEK_NO.to_numpy(np.int64) - w0,
                events.previous_WEEK_NO.to_numpy(np.int64) - w0)
    if (trips_per_cell[s, w] == 0).any() or (trips_per_cell[s, wp] == 0).any():
        raise RuntimeError("an event's store-week has no evaluated trip")
    depth = events.promotion_depth_change.to_numpy(np.float64)
    item, cat = acc["fact"][i, s, w], acc_cat["fact"][c, s, w]
    share = np.where(cat > 0, item / np.maximum(cat, 1e-30), 0.0)
    single = acc["single"][i, s, w]
    rest = np.maximum(cat - item, 0.0)
    share_single = np.where(rest + single > 0, single / np.maximum(rest + single, 1e-30), 0.0)
    chain_cat = acc_cat["chain"][c, s, w]
    share_chain = np.where(chain_cat > 0, acc["chain"][i, s, w] / np.maximum(chain_cat, 1e-30), 0.0)

    def slope(after, before):
        ok = (after > 0) & (before > 0)
        y, x = np.log(after[ok]) - np.log(before[ok]), depth[ok]
        return {"events": int(ok.sum()), "slope": float(np.cov(x, y)[0, 1] / np.var(x, ddof=1))}

    def promotion(name):
        on, off = acc[f"{name}_on"][i, s, w], acc[f"{name}_off"][i, s, w]
        ok = (on > 0) & (off > 0) & (rest > 0)
        share_ratio = np.log(on[ok] / (rest[ok] + on[ok])) - np.log(off[ok] / (rest[ok] + off[ok]))
        rate_ratio = np.log(on[ok]) - np.log(off[ok])
        weight = item[ok] / item[ok].sum()
        coef = model.promo_weights()[0 if name == "disp" else 1].detach().numpy()
        return {"events": int(ok.sum()),
                "choice_share_log_ratio_mean": float(share_ratio.mean()),
                "choice_share_log_ratio_purchase_weighted": float((weight * share_ratio).sum()),
                "purchase_rate_log_ratio_mean": float(rate_ratio.mean()),
                "purchase_rate_log_ratio_purchase_weighted": float((weight * rate_ratio).sum()),
                "share_of_events_with_negative_effect": float((share_ratio < 0).mean()),
                "catalogue_weight_mean": float(coef.mean()),
                "catalogue_share_negative_weight": float((coef < 0).mean())}

    def agreement(pred):
        informative = events.total_purchases.to_numpy() > 0
        p, t = pred[informative], events.observed_choice_share_change.to_numpy(np.float64)[informative]
        nonzero = (np.abs(p) > 0) & (np.abs(t) > 0)
        up = depth[informative] >= 0.05
        return {"events": int(informative.sum()), "pearson": float(np.corrcoef(p, t)[0, 1]),
                "sign_agreement_nonzero_observed": float(np.mean(np.sign(p[nonzero]) == np.sign(t[nonzero]))),
                "mean_predicted_change_discount_increase": float(p[up].mean()),
                "mean_observed_change_discount_increase": float(t[up].mean())}

    emp = {split: {k: v["estimate"] for k, v in e["effects"].items()} for split, e in empirical.items()}
    return {"empirical_effects": emp,
            "depth_slope": {"single_item_choice_share": slope(share, share_single),
                            "single_item_purchase_rate": slope(item, single),
                            "chain_week_choice_share": slope(share, share_chain)},
            "promotion_on_vs_off": {"display": promotion("disp"), "mailer": promotion("mail")},
            "share_change_agreement": {"single_item": agreement(share - share_single),
                                       "chain_week": agreement(share - share_chain)}}


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    e = sub.add_parser("events", help="build events and empirical effects")
    e.add_argument("--data-root", type=Path)
    e.add_argument("--out-dir", type=Path, required=True)
    e.add_argument("--min-depth-change", type=float, default=0.05)
    e.add_argument("--min-trips", type=int, default=16)
    m = sub.add_parser("model", help="score a model on one split's events")
    m.add_argument("--data-root", type=Path)
    m.add_argument("--events", type=Path, required=True, help="directory written by the events step")
    m.add_argument("--checkpoint", type=Path, required=True, help="additive or Version-4 checkpoint")
    m.add_argument("--initialization", type=Path, help="for checkpoints that do not record it")
    m.add_argument("--discount-panel", type=Path, help="for checkpoints that record it as a relative path")
    m.add_argument("--split", choices=("validation", "test"), default="validation")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--trips-per-store-week", type=int, default=1)
    m.add_argument("--burn", type=int, default=200)
    m.add_argument("--draws", type=int, default=64)
    m.add_argument("--thin", type=int, default=2)
    m.add_argument("--seed", type=int, default=68101)
    m.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.command == "events":
        root = data_root(args.data_root)
        events = build_events(root, args.min_depth_change, args.min_trips)
        args.out_dir.mkdir(parents=True, exist_ok=True)
        events.to_parquet(args.out_dir / "events.parquet", index=False)
        empirical = {split: empirical_effects(events[events.split == split])
                     for split in ("train", "validation", "test")}
        (args.out_dir / "empirical.json").write_text(strict_json_dumps(empirical))
        log(json.dumps({k: {n: round(v["estimate"], 4) for n, v in e["effects"].items()}
                        for k, e in empirical.items()}))
    else:
        model_check(args)


if __name__ == "__main__":
    main()
