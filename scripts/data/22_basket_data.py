"""
Stage 22 -- Build the dataset for a basket model with household state.

This replaces 02/03 rather than extending them.  Those scripts exist to serve the
paper's model, and every one of their major filters is there to protect an
assumption this model drops:

  paper's pipeline                        here
  --------------------------------------  ------------------------------------------
  at most one item per category per trip  a basket is a multiset; buy several items,
  -> 132 of 307 categories dropped        and several units of each
  categories independent                  items interact through a shared embedding
  Sunday + Monday only (172 sessions)     all 711 days, because state needs time
  -> 560 items, 56 categories             -> ~5,500 items, ~750 sub-commodities
  price = the only time-varying input     price + household state per sub-commodity
  prices pooled to chain level            weekly modal chain/store price from stage 01
  no stock feed                           declared chain catalogue at every modelled store;
                                          sales are not mislabelled as availability

What it writes to basket_input/:

  baskets.parquet   one row per (basket, item) purchase: units, store, split label
  items.parquet     item_id <-> PRODUCT_ID and the held-out labels (sub-commodity,
                    brand, manufacturer, department) that the model never sees
  log_price*.npy    [J, D] weekly modal log price, carried between observed weeks
  store_price.npz   store-level price deviations where observed, plus the
                    descriptive training-sale mask (not likelihood support)
  state.npz         the structure that answers "how long since this household last
                    bought this sub-commodity, as of this day"
  meta.json         sizes, split boundaries, coverage and cohort policy

The state lookup deserves a note.  The obvious implementations are both wrong at
this scale: materialising (household x day x sub-commodity) is ~32 million rows, and
a per-sample Python lookup is ~35 million dict hits per epoch.  Instead the purchase
days are stored once as a globally sorted key array, key = group_id * 1024 + day,
where group_id indexes a (household, sub-commodity) pair.  A single vectorised
np.searchsorted then answers the question for a whole batch, and because the offset
is larger than any day the ordering never crosses a group boundary.

Splitting is by household x week, as the paper does, so a household contributes to
several splits and the model is never asked to predict a household it has not seen.
The last weeks are held out rather than a random slice: a model meant to answer
"what happens next" should be tested on what happened next.
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "data")
OUT = os.path.join(HERE, "..", "..", "basket_input")
RAW = os.path.join(os.environ.get(
    "NF_RAW_DIR",
    os.path.join(HERE, "..", "..", "..", "dunnhumby_The-Complete-Journey",
                 "dunnhumby_The-Complete-Journey CSV")), "")

DAY_STRIDE = 1024          # > 711, so group_id * DAY_STRIDE + day never collides


def log(m):
    print(f"[22] {m}", flush=True)


def main(a):
    global OUT
    if a.outdir:
        OUT = os.path.join(HERE, "..", "..", a.outdir)
    os.makedirs(OUT, exist_ok=True)
    tx0 = pd.read_parquet(os.path.join(DATA, "tx.parquet"),
                          columns=["household_key", "DAY", "WEEK_NO", "PRODUCT_ID",
                                   "BASKET_ID", "QUANTITY", "STORE_ID",
                                   "unit_price", "base_price"])
    calendar_days = int(tx0.DAY.max()) + 1
    day_week_frame = tx0[["DAY", "WEEK_NO"]].drop_duplicates()
    if day_week_frame.groupby("DAY").WEEK_NO.nunique().max() != 1:
        raise SystemExit("DAY maps to more than one WEEK_NO")
    prod = pd.read_csv(RAW + "product.csv",
                       usecols=["PRODUCT_ID", "COMMODITY_DESC", "SUB_COMMODITY_DESC",
                                "BRAND", "MANUFACTURER", "DEPARTMENT"])
    tx = tx0.merge(prod, on="PRODUCT_ID", how="left").dropna(subset=["COMMODITY_DESC"])
    log(f"raw: {len(tx):,} lines, {tx.PRODUCT_ID.nunique():,} products, "
        f"{tx.BASKET_ID.nunique():,} baskets")

    # ---------------------------------------------------- the split boundary, first
    # The week boundaries have to be known BEFORE any filter is applied, because with
    # --train-only-features the filters themselves must be computed from training weeks
    # alone.  Deciding which items and households exist using the full two years is a
    # sample-selection leak: an item kept because it sold well in the test period is an
    # item the model would never have known to model.
    wmax0 = int(tx.WEEK_NO.max())
    test_from0 = wmax0 - a.test_weeks + 1
    val_from0 = test_from0 - a.val_weeks
    if not (int(tx.WEEK_NO.min()) <= a.first_week < val_from0 < test_from0 <=
            a.last_week <= wmax0):
        raise SystemExit("invalid analysis window or split boundaries")
    # causal_data.csv covers weeks 9..101.  Outside that interval an absent row is
    # "unmeasured", not "not promoted".  Exclude those boundary weeks before defining
    # the cohort; silently encoding them as zero is a covariate error.
    tx = tx[(tx.WEEK_NO >= a.first_week) & (tx.WEEK_NO <= a.last_week)].copy()
    log(f"analysis window: weeks {a.first_week}-{a.last_week}; fixed split boundaries "
        f"validation={val_from0}, test={test_from0}")

    # ------------------------------------------------------------- item universe
    # Fix the computational catalogue budget at 5,455, but choose it using TRAINING
    # observations only.  Ordering by (-frequency, PRODUCT_ID) makes ties reproducible.
    # The former full-panel >=100 rule let future sales decide which embeddings existed
    # and even admitted a product with no training observations.
    freq = (tx[tx.WEEK_NO < val_from0].groupby("PRODUCT_ID").size().rename("n_train")
            .reset_index().sort_values(["n_train", "PRODUCT_ID"],
                                       ascending=[False, True], kind="mergesort"))
    if len(freq) < a.n_items:
        raise SystemExit(f"only {len(freq)} products occur in training; cannot select "
                         f"the requested {a.n_items}")
    selected = freq.head(a.n_items).copy()
    keep = set(selected.PRODUCT_ID.astype(np.int64))
    boundary = int(selected.n_train.min())
    boundary_ties = int((freq.n_train == boundary).sum())
    tx = tx[tx.PRODUCT_ID.isin(keep)].copy()
    log(f"kept top {len(keep):,} items by training frequency (boundary {boundary} "
        f"lines; {boundary_ties} products tied there) -> {len(tx):,} lines "
        f"({tx.BASKET_ID.nunique():,} baskets, {tx.SUB_COMMODITY_DESC.nunique():,} "
        f"sub-commodities, {tx.COMMODITY_DESC.nunique()} commodities)")

    # Households need enough trips for a taste vector to be estimable.  This is the
    # paper's own 20-300 trip rule, kept because it is about statistical support
    # rather than about the model's structure.
    trips_per_hh = tx[tx.WEEK_NO < val_from0].groupby("household_key").DAY.nunique()
    hh_keep = trips_per_hh[(trips_per_hh >= a.min_trips) & (trips_per_hh <= a.max_trips)].index
    tx = tx[tx.household_key.isin(set(hh_keep))].copy()
    log(f"kept {len(hh_keep):,} households with {a.min_trips}-{a.max_trips} TRAINING "
        f"shopping days "
        f"-> {len(tx):,} lines")

    # ------------------------------------------------------------------- ids
    items = (tx.groupby("PRODUCT_ID")
             .agg(n_lines=("QUANTITY", "size"),
                  n_households=("household_key", "nunique"))
             .reset_index()
             .merge(prod, on="PRODUCT_ID", how="left")
             .sort_values("PRODUCT_ID").reset_index(drop=True))
    train_item_stats = (tx[tx.WEEK_NO < val_from0].groupby("PRODUCT_ID")
                        .agg(n_train_lines=("QUANTITY", "size"),
                             n_train_households=("household_key", "nunique")))
    items = items.merge(train_item_stats, on="PRODUCT_ID", how="left", validate="one_to_one")
    if items[["n_train_lines", "n_train_households"]].isna().any().any():
        raise SystemExit("selected catalogue contains an item without training support")
    items["item_id"] = np.arange(len(items))
    iid = items.set_index("PRODUCT_ID").item_id
    for col, name in [("SUB_COMMODITY_DESC", "sub_id"), ("COMMODITY_DESC", "cat_id"),
                      ("BRAND", "brand_id"), ("MANUFACTURER", "mfr_id"),
                      ("DEPARTMENT", "dept_id")]:
        items[name] = pd.factorize(items[col])[0]
    users = np.sort(tx.household_key.unique())
    uid = pd.Series(np.arange(len(users)), index=users)

    tx["item_id"] = tx.PRODUCT_ID.map(iid).astype(np.int32)
    tx["user_id"] = tx.household_key.map(uid).astype(np.int32)
    tx["sub_id"] = tx.item_id.map(items.set_index("item_id").sub_id).astype(np.int32)

    J, N, D = len(items), len(users), calendar_days
    S = int(items.sub_id.max()) + 1
    log(f"ids: {N:,} households, {J:,} items, {S:,} sub-commodities, {D} days")

    # ------------------------------------------------------------ basket table
    # One row per (basket, item), with units as a count.
    # A trip happens at one store: only 2.4% of household-days touch more than one,
    # so the store is a clean per-basket attribute rather than a within-trip choice.
    basket_invariants = tx.groupby("BASKET_ID").agg(
        n_store=("STORE_ID", "nunique"), n_user=("household_key", "nunique"),
        n_day=("DAY", "nunique"), n_week=("WEEK_NO", "nunique"))
    if (basket_invariants[["n_store", "n_user", "n_day", "n_week"]] != 1).any().any():
        raise SystemExit("BASKET_ID is not a unique checkout at one store/day/household")
    store_of_basket = tx.groupby("BASKET_ID").STORE_ID.first().rename("store_raw")
    stores = np.sort(store_of_basket.unique())
    sid = pd.Series(np.arange(len(stores)), index=stores)
    bk = (tx.groupby(["BASKET_ID", "user_id", "DAY", "WEEK_NO", "item_id", "sub_id"],
                     as_index=False)
          .agg(units=("QUANTITY", "sum"), price=("unit_price", "median")))
    bk["store_id"] = bk.BASKET_ID.map(store_of_basket).map(sid).astype(np.int32)
    # Units are kept as counts.  22.3% of (basket, item) rows buy more than one, and
    # those rows carry 42.7% of all units, so collapsing to binary would discard
    # nearly half the volume and one of the two channels a price cut works through
    # (more buyers, and more units each -- the units-per-buyer elasticity is -0.22).
    bk["units"] = bk.units.clip(lower=1, upper=a.max_units).astype(np.int16)
    log(f"stores: {len(stores)};  units>1 on {(bk.units > 1).mean():.1%} of rows, "
        f"carrying {bk.units[bk.units > 1].sum() / bk.units.sum():.1%} of all units "
        f"(clipped at {a.max_units})")
    bk = bk.sort_values(["user_id", "DAY", "BASKET_ID", "item_id"]).reset_index(drop=True)
    log(f"basket table: {len(bk):,} (basket, item) rows over {bk.BASKET_ID.nunique():,} baskets")

    # ----------------------------------------------------------------- splits
    # Hold out the final weeks, not a random slice.  A model whose point is to answer
    # "what happens next week if we change price" should be scored on later weeks,
    # and a random split would let it see the future of every household.
    wmax = int(bk.WEEK_NO.max())
    test_from, val_from = test_from0, val_from0
    bk["split"] = np.where(bk.WEEK_NO >= test_from, "test",
                           np.where(bk.WEEK_NO >= val_from, "validation", "train"))
    log(f"splits by week: train <{val_from}, validation {val_from}-{test_from - 1}, "
        f"test {test_from}-{wmax}; weeks outside promotion coverage were excluded")
    log("  " + "  ".join(f"{k} {v:,}" for k, v in bk.split.value_counts().items()))

    # The cohort was defined from training data, so cold-start rows are now an error,
    # never something the evaluator is allowed to delete.
    tr = bk[bk.split == "train"]
    known_i, known_u = set(tr.item_id), set(tr.user_id)
    bad = (~bk.item_id.isin(known_i) | ~bk.user_id.isin(known_u)) & (bk.split != "train")
    if bad.any() or len(known_i) != J or len(known_u) != N:
        raise SystemExit(f"training-defined cohort has cold support: {int(bad.sum())} "
                         "held-out rows")
    log("  cold-start integrity: every modelled item and household occurs in training")

    # Everything below this line is a FEATURE derived from the data.  With
    # --train-only-features every one of them is computed from training weeks alone,
    # because a feature built from the whole panel leaks held-out information into the
    # inputs the model sees at training time.  See NESTED_MODEL.md 11.
    # WHAT MUST BE TRAIN-ONLY, AND WHAT MUST NOT.
    #
    # A feature is a leak only if it could not have been known at the moment of the
    # decision it helps predict.  That splits three ways and conflating them is easy:
    #
    #   contemporaneous  the posted price on day t, and a store's price that week.  A
    #                    shopper standing at the shelf in week 95 sees the week-95 tag.
    #                    Restricting these to training weeks does not remove a leak, it
    #                    DELETES THE EXPERIMENT: held-out prices then carry forward from
    #                    week 82 and the test period has no price variation at all.
    #   past lookups     "days since this household last bought this sub-commodity".
    #                    searchsorted is strictly-before, so all events are safe to use.
    #   time-aggregates  the within-item mean that centres the price, the per-
    #                    sub-commodity median repurchase gap, and "this store ever sold
    #                    this item".  These summarise a span that includes the future.
    #                    THESE are the leaks, and only these are restricted below.
    tx_agg = tx[tx.WEEK_NO < val_from]
    log(f"aggregates from weeks < {val_from} ({len(tx_agg):,} of {len(tx):,} lines); "
        f"prices and store deviations stay contemporaneous")

    # ------------------------------------------------------------- price panel
    # Stage 01 already estimated a deterministic modal faced price in each product-week.
    # Rebuilding a daily median here was redundant and statistically different: it let
    # the mix of buyers on a day move the model's price.  Carry the modal tag only across
    # missing weeks, then broadcast that weekly price to the calendar days.
    pw = pd.read_parquet(os.path.join(DATA, "price_week.parquet"),
                         columns=["PRODUCT_ID", "WEEK_NO", "price"])
    pw = pw[pw.PRODUCT_ID.isin(keep)].copy()
    pw["item_id"] = pw.PRODUCT_ID.map(iid).astype(np.int32)
    week_grid = np.full((J, 128), np.nan, dtype=np.float64)
    week_grid[pw.item_id.to_numpy(), pw.WEEK_NO.to_numpy()] = pw.price.to_numpy()
    week_grid = pd.DataFrame(week_grid).ffill(axis=1).bfill(axis=1).to_numpy()
    if not np.isfinite(week_grid).all():
        raise SystemExit("modal weekly price grid has holes")
    day_week = np.full(D, -1, dtype=np.int16)
    day_week[day_week_frame.DAY.to_numpy()] = day_week_frame.WEEK_NO.to_numpy()
    day_week = (pd.Series(day_week).replace(-1, np.nan).ffill().bfill()
                .to_numpy(dtype=np.int16))
    price = week_grid[:, day_week].astype(np.float32)
    obs_share = float(len(pw[(pw.WEEK_NO >= a.first_week) &
                             (pw.WEEK_NO <= a.last_week)]) /
                      (J * (a.last_week - a.first_week + 1)))
    log_price = np.log(np.clip(price, 1e-3, None)).astype(np.float32)
    # Centre per item: the level is absorbed by the item's own intercept, and what
    # identifies price response is deviation from that item's normal price.
    # Centre on the TRAINING days only when asked.  Using all D days makes the centring
    # constant depend on held-out prices; it is a per-item shift, but it is still a
    # number the model could not have known at training time.
    train_days = (day_week >= a.first_week) & (day_week < val_from)
    log_price_dev = (log_price - log_price[:, train_days].mean(axis=1, keepdims=True)
                     ).astype(np.float32)
    log(f"  centred on calendar days whose weeks are {a.first_week}-{val_from - 1}")
    log(f"price panel: {J} x {D}, {obs_share:.1%} of in-window item-weeks observed, "
        f"sd of within-item log price {log_price_dev.std():.4f}")

    # ------------------------------------------------ store prices and assortment
    # Two things the chain-level panel gets wrong.
    #
    # 1. Price.  15.8% of store-item-weeks sit more than a cent from the chain price
    #    (sd across stores $0.12), so pooling puts measurement error straight into the
    #    regressor that the elasticity is read off.  Store-level cells are sparse
    #    though -- only 2.3% of the item x store x week grid is ever observed -- so
    #    the deviation is stored sparsely and the model falls back to the chain price
    #    wherever a store-week was not observed.  That is honest about what is known
    #    rather than imputing a store price from nothing.
    #
    # 2. Assortment.  The median store carries 63% of the catalogue and the 10th
    #    percentile store carries 39%.  Scoring an item a store has never stocked as
    #    a *rejected* alternative is a specification error: it was not available to
    #    reject.  dunnhumby has no stock-out feed (the paper's data did), so
    #    availability is proxied by "this store sold this item at some point in a
    #    window around this week", which is the standard proxy and is a proxy.
    tx["store_id"] = tx.BASKET_ID.map(store_of_basket).map(sid).astype(np.int32)
    # store_id is assigned here, after tx_agg was first taken, so re-slice it
    tx_agg = tx[tx.WEEK_NO < val_from]
    n_stores = len(stores)
    psw = pd.read_parquet(os.path.join(DATA, "price_store_week.parquet"),
                          columns=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "price"])
    psw = psw[(psw.PRODUCT_ID.isin(keep)) & (psw.STORE_ID.isin(set(stores))) &
              (psw.WEEK_NO >= a.first_week) & (psw.WEEK_NO <= a.last_week)].copy()
    psw["item_id"] = psw.PRODUCT_ID.map(iid).astype(np.int32)
    psw["store_id"] = psw.STORE_ID.map(sid).astype(np.int32)
    cp = pw[["PRODUCT_ID", "WEEK_NO", "price"]].rename(columns={"price": "p_chain"})
    sp = psw.merge(cp, on=["PRODUCT_ID", "WEEK_NO"], validate="many_to_one")
    sp = sp.rename(columns={"price": "p_store"})
    eps = 1e-3
    sp["dev"] = (np.log(sp.p_store.clip(lower=eps)) - np.log(sp.p_chain.clip(lower=eps)))
    sp = sp[sp.dev.abs() > 1e-6]
    log(f"store prices: {len(sp):,} item x store x week deviations kept "
        f"({len(sp) / max(len(cp) * n_stores, 1):.2%} of the grid); "
        f"median |dev| {sp.dev.abs().median():.4f} in logs")

    # availability: item x store, plus the first and last week each store sold it
    av = (tx_agg.groupby(["item_id", "store_id"])
          .WEEK_NO.agg(["min", "max", "size"]).reset_index())
    av = av[av["size"] >= a.min_store_lines]
    carried = np.zeros((J, n_stores), dtype=bool)
    carried[av.item_id.to_numpy(), av.store_id.to_numpy()] = True
    log(f"assortment: a store carries {carried.sum(0).mean():.0f} of {J} items on "
        f"average (median {np.median(carried.sum(0)):.0f}); "
        f"{carried.mean():.1%} of the item x store grid")

    # --------------------------------------------------------------- state
    # For every (household, sub-commodity) pair, the sorted days on which that
    # household bought that sub-commodity.  Encoded as one globally sorted key array
    # so a batch query is a single searchsorted (see the module docstring).
    # NOT tx_agg.  The state lookup is strictly-before -- searchsorted finds the last
    # purchase with day < t -- so using every purchase event is correct even for a test
    # row: a household's week-90 purchase is genuinely in the past when predicting week
    # 95.  Restricting these keys to training weeks would delete real history and make
    # the feature worse for no gain.  What must be train-only is the AGGREGATE below,
    # sub_gap, because a median over the whole panel is a number computed from the
    # future.
    ev = tx[["user_id", "sub_id", "DAY"]].drop_duplicates()
    ev = ev.sort_values(["user_id", "sub_id", "DAY"]).reset_index(drop=True)
    ev["group"] = ev.user_id.astype(np.int64) * S + ev.sub_id
    keys = (ev.group.to_numpy() * DAY_STRIDE + ev.DAY.to_numpy()).astype(np.int64)
    assert (np.diff(keys) > 0).all(), "state keys must be strictly increasing"
    # group -> first index, so a query can tell "no earlier purchase in this group"
    # from "the previous key belongs to a different group".
    gstart = {}
    uniq, first = np.unique(ev.group.to_numpy(), return_index=True)
    gstart_keys, gstart_vals = uniq.astype(np.int64), first.astype(np.int64)
    log(f"state: {len(ev):,} (household, sub-commodity, day) purchase events across "
        f"{len(uniq):,} (household, sub-commodity) pairs")

    # Median repurchase gap per sub-commodity: the natural time scale for that
    # sub-commodity, used to normalise recency so "a long time" means the right
    # number of days for milk and for shampoo.
    g = ev[ev.DAY <= bk[bk.split == "train"].DAY.max()].copy()
    g["prev"] = g.groupby(["user_id", "sub_id"]).DAY.shift()
    gaps = (g.DAY - g.prev).dropna()
    gap_by_sub = (g.assign(gap=g.DAY - g.prev).dropna(subset=["gap"])
                  .groupby("sub_id").gap.median())
    sub_gap = np.full(S, float(gaps.median()), dtype=np.float32)
    sub_gap[gap_by_sub.index.to_numpy()] = gap_by_sub.to_numpy()
    sub_gap = np.clip(sub_gap, 3.0, 180.0)
    log(f"  repurchase gap: overall median {gaps.median():.0f} days; per sub-commodity "
        f"median ranges {sub_gap.min():.0f}-{sub_gap.max():.0f}")

    # ------------------------------------------------ category-level recency
    # The incidence head needs "how long since this household bought anything from
    # category c".  Until now it borrowed the sub-commodity state of whichever item in
    # the category happened to have the lowest PRODUCT_ID -- an arbitrary pick that
    # covered a median of 50% of its category's items and under 20% in a fifth of them
    # (for BAKED BREAD/BUNS/ROLLS it asked about HOT DOG BUNS).  Built properly here,
    # with the same sorted-key trick one level up.
    tx["cat_id"] = tx.item_id.map(items.set_index("item_id").cat_id).astype(np.int32)
    evc = tx[["user_id", "cat_id", "DAY"]].drop_duplicates()
    evc = evc.sort_values(["user_id", "cat_id", "DAY"]).reset_index(drop=True)
    C = int(items.cat_id.max()) + 1
    evc["group"] = evc.user_id.astype(np.int64) * C + evc.cat_id
    cat_keys = (evc.group.to_numpy() * DAY_STRIDE + evc.DAY.to_numpy()).astype(np.int64)
    assert (np.diff(cat_keys) > 0).all(), "category state keys must be strictly increasing"
    gc = evc[evc.DAY <= bk[bk.split == "train"].DAY.max()].copy()
    gc["prev"] = gc.groupby(["user_id", "cat_id"]).DAY.shift()
    gaps_c = (gc.assign(g=gc.DAY - gc.prev).dropna(subset=["g"])
              .groupby("cat_id").g.median())
    cat_gap = np.full(C, float((gc.DAY - gc.prev).dropna().median()), dtype=np.float32)
    cat_gap[gaps_c.index.to_numpy()] = gaps_c.to_numpy()
    cat_gap = np.clip(cat_gap, 3.0, 180.0)
    log(f"category state: {len(evc):,} (household, category, day) events across "
        f"{evc.group.nunique():,} (household, category) pairs; "
        f"gap median {np.median(cat_gap):.0f} days, range {cat_gap.min():.0f}-{cat_gap.max():.0f}")

    # ------------------------------------------------------------------ write
    bk.to_parquet(os.path.join(OUT, "baskets.parquet"), index=False)
    items.to_parquet(os.path.join(OUT, "items.parquet"), index=False)
    np.save(os.path.join(OUT, "log_price_dev.npy"), log_price_dev)
    np.save(os.path.join(OUT, "log_price.npy"), log_price)
    np.savez(os.path.join(OUT, "store_price.npz"),
             item=sp.item_id.to_numpy(np.int32), store=sp.store_id.to_numpy(np.int32),
             week=sp.WEEK_NO.to_numpy(np.int32), dev=sp.dev.to_numpy(np.float32),
             carried=carried, n_stores=np.int32(n_stores))
    np.savez(os.path.join(OUT, "state.npz"),
             keys=keys, gstart_keys=gstart_keys, gstart_vals=gstart_vals,
             sub_gap=sub_gap, item_sub=items.sub_id.to_numpy().astype(np.int32),
             cat_keys=cat_keys, cat_gap=cat_gap,
             item_cat=items.cat_id.to_numpy().astype(np.int32))
    meta = {
        "n_users": int(N), "n_items": int(J), "n_subs": int(S), "n_days": int(D),
        "n_baskets": int(bk.BASKET_ID.nunique()),
        "n_rows": int(len(bk)),
        "cohort_policy": "top_n_products_by_training_lines_then_training_trip_households",
        "n_items_requested": int(a.n_items),
        "item_training_line_boundary": boundary,
        "item_boundary_ties": boundary_ties,
        "min_trips": a.min_trips, "max_trips": a.max_trips,
        "val_from_week": int(val_from), "test_from_week": int(test_from),
        "analysis_first_week": int(a.first_week),
        "analysis_last_week": int(a.last_week),
        "promotion_coverage_required": [int(a.first_week), int(a.last_week)],
        "day_stride": DAY_STRIDE,
        "split_rows": {k: int(v) for k, v in bk.split.value_counts().items()},
        "n_commodities": int(items.cat_id.nunique()),
        "n_stores": int(n_stores),
        "max_units": a.max_units,
        "share_rows_units_gt1": float((bk.units > 1).mean()),
        "share_units_in_multi_rows": float(bk.units[bk.units > 1].sum() / bk.units.sum()),
        "store_price_cells": int(len(sp)),
        "assortment_share": float(carried.mean()),
        "price_obs_share": obs_share,
        "median_repurchase_gap_days": float(gaps.median()),
    }
    with open(os.path.join(OUT, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log(f"wrote {os.path.basename(OUT)}/ -> {len(bk):,} rows, {J:,} items, {N:,} households, "
        f"{S:,} sub-commodities")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-items", type=int, default=5455,
                   help="number of products, selected by training-line frequency")
    p.add_argument("--min-trips", type=int, default=20)
    p.add_argument("--max-trips", type=int, default=300)
    p.add_argument("--val-weeks", type=int, default=8)
    p.add_argument("--test-weeks", type=int, default=12)
    p.add_argument("--first-week", type=int, default=9,
                   help="first week with causal/promotion coverage")
    p.add_argument("--last-week", type=int, default=101,
                   help="last week with causal/promotion coverage")
    p.add_argument("--max-units", type=int, default=12,
                   help="cap on units per (basket, item); dunnhumby's QUANTITY is "
                        "unreliable for weighed goods and the tail is bulk lines")
    p.add_argument("--outdir", default=None,
                   help="write somewhere other than basket_input/")
    p.add_argument("--min-store-lines", type=int, default=1,
                   help="sales needed before a store counts as carrying an item; 1 "
                        "because an item a store sold was by definition available "
                        "there, and anything higher marks real purchases unavailable")
    main(p.parse_args())
