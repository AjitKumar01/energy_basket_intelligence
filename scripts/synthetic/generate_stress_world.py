#!/usr/bin/env python3
"""A misspecified retail world for model evaluation: shaped like neither basket model.

Neither the staged Version-4 model nor the single-stage model generated these baskets.
There is no energy function, pair matrix, category penalty or size potential here; baskets
come from a procedural shopping story built from standard marketing-science mechanisms:

* missions: a trip is 1 + Poisson(intensity_h) missions (breakfast, recipe, snacks, ...),
  each a sparse list of needed categories; recipes add all-or-none category triplets, so
  complementarity is higher order and mission-driven;
* nested choice: each need picks a subcategory, then a product (nested logit, nest
  parameter 0.5); substitution comes from "one product per need", nested, never uniform;
* long-tailed catalogue: Zipf quality inside categories, uneven subcategory sizes;
* heterogeneity and dynamics: household taste over three product attributes drifting
  weekly (AR(1)); brand loyalty to the last product bought in a category; pantry
  inventory: a promotion purchase suppresses that need for a few weeks (stockpiling);
  variety seeking (occasionally two products for one need); impulse additions;
* prices: chain-level weekly prices with promotions (25% off plus a display); reference-
  price response with loss aversion (a rise hurts 2x more than an equal cut helps), and
  household-specific price sensitivity;
* observation: heavy-tailed visit rates (many households with a handful of trips), store
  launches seen through a retail feed, and unrecorded stockouts (3% of cells).

Every trip has its own random stream, so re-simulating test weeks with changed prices from
the same state gives the causal truth with common random numbers.  The declared price
scenarios and their true effects are written before any model is fitted.

Outputs: <output>/canonical/* (the canonical contract), observed.npz (the same observed
data as arrays), truth.json, and counterfactual_truth.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

MISSIONS = ("breakfast", "lunch", "dinner_recipe", "snacks", "cleaning", "baby", "party", "top_up")


def build_catalogue(rng, departments, categories_per_department):
    C = departments * categories_per_department
    cat_dept = np.repeat(np.arange(departments), categories_per_department)
    rows = []
    for c in range(C):
        n_sub = int(rng.integers(1, 5))
        for s in range(n_sub):
            size = int(np.clip(rng.zipf(1.8), 1, 12)) + 1
            for k in range(size):
                rows.append((c, s))
    category = np.array([r[0] for r in rows]); subcategory = np.array([r[1] for r in rows])
    J = len(rows)
    quality = np.zeros(J)
    for c in range(C):
        items = np.flatnonzero(category == c)
        ranks = rng.permutation(len(items)) + 1
        quality[items] = -1.1 * np.log(ranks) + rng.normal(0, 0.3, len(items))     # Zipf-like
    attributes = rng.normal(0, 1, (J, 3))                    # premium, organic, value
    base_price = np.round(np.exp(rng.normal(1.0, 0.6, J)), 2)
    return {"C": C, "J": J, "category": category, "subcategory": subcategory, "cat_dept": cat_dept,
            "quality": quality, "attributes": attributes, "base_price": base_price}


def build_missions(rng, C, cat_dept, departments):
    templates = []
    for m in range(len(MISSIONS)):
        focus = rng.choice(departments, size=int(rng.integers(1, 3)), replace=False)
        pool = np.flatnonzero(np.isin(cat_dept, focus))
        chosen = rng.choice(pool, size=min(len(pool), int(rng.integers(4, 9))), replace=False)
        extra = rng.choice(C, size=2, replace=False)          # cross-department needs
        cats = np.unique(np.concatenate([chosen, extra]))
        probs = rng.uniform(0.2, 0.8, len(cats))
        recipes = [rng.choice(cats, size=3, replace=False) for _ in range(int(rng.integers(0, 3)))]
        templates.append({"categories": cats, "probs": probs, "recipes": recipes,
                          "recipe_prob": float(rng.uniform(0.2, 0.5))})
    return templates


class World:
    def __init__(self, args, rng):
        self.args = args
        cat = build_catalogue(rng, args.departments, args.categories_per_department)
        self.__dict__.update(cat)
        self.templates = build_missions(rng, self.C, self.cat_dept, args.departments)
        H, S, T, J = args.households, args.stores, args.weeks, self.J
        # larger catalogues: each mission gets `mission_variants` category templates (variant 0 is
        # the template above), from a separate stream so the default world is unchanged.  A
        # household has a usual variant per mission and uses it on 60% of those missions.
        self.variants = int(args.mission_variants)
        if self.variants > 1:
            vrng = np.random.default_rng([args.seed, 7919])
            extra = [build_missions(vrng, self.C, self.cat_dept, args.departments)
                     for _ in range(self.variants - 1)]
            self.variant_templates = [[self.templates[m]] + [e[m] for e in extra] for m in range(len(MISSIONS))]
            self.usual_variant = vrng.integers(0, self.variants, (H, len(MISSIONS)))
        # households
        segment = rng.integers(0, 4, H)
        seg_pref = rng.dirichlet(np.full(len(MISSIONS), 0.7), 4)
        self.mission_pref = np.array([rng.dirichlet(20 * seg_pref[s] + 0.1) for s in segment])
        self.segment = segment
        self.intensity = rng.gamma(2.0, 0.6, H)
        self.visit_rate = rng.beta(1.1, 3.0, H)
        self.home = rng.integers(0, S, H)
        self.price_sens = np.exp(rng.normal(0.4, 0.5, H))      # household price sensitivity
        self.taste0 = rng.normal(0, 0.6, (H, 3))
        self.idiosyncratic = rng.normal(0, 0.5, (H, J)) * (rng.random((H, J)) < 0.15)
        # prices and promotions (chain level), weekly
        log_mult = rng.normal(0, 0.08, (J, T))
        promo = rng.random((J, T)) < 0.07
        self.price = np.round(self.base_price[:, None] * np.exp(log_mult) * np.where(promo, 0.75, 1.0), 2)
        self.promo = promo
        # assortment: launches seen in the retail feed, and unrecorded stockouts
        first = np.where(rng.random((J, S)) < 0.85, 1, T + 1).astype(np.int16)
        launched = rng.choice(J, size=J // 10, replace=False)
        for j in launched:
            first[j] = np.where(rng.random(S) < 0.4, 1, rng.integers(10, T - 2, S))
        for j in range(J):
            if not (first[j] <= args.train_weeks // 2).any():
                first[j, rng.integers(S)] = 1
        self.first_stocked = first
        self.stockout = rng.random((J, S, T)) < 0.03


def simulate(world, trips, week_slice_state, price=None, seed_base=0):
    """Simulate the given trips in chronological order from a household state.

    trips: arrays (household, store, week_index, trip_id); state: dict of loyalty (H x C),
    inventory (H x C, weeks of pantry left) and taste (H x 3) at the first trip's week.
    Each trip draws from its own stream seeded by (seed_base, trip_id).
    """
    args = world.args
    price = world.price if price is None else price
    loyalty = week_slice_state["loyalty"].copy()
    inventory = week_slice_state["inventory"].copy()
    taste = week_slice_state["taste"].copy()
    last_week = week_slice_state["week"]
    baskets = []
    ref = world.base_price
    for hh, st, wk, tid in zip(*trips):
        if wk != last_week:                                 # weekly dynamics for everyone
            steps = wk - last_week
            for _ in range(steps):
                taste = 0.9 * taste + 0.1 * world.taste0 + np.random.default_rng(
                    [seed_base, 7, int(last_week)]).normal(0, 0.08, taste.shape)
                inventory = np.maximum(inventory - 1, 0)
                last_week += 1
        r = np.random.default_rng([seed_base, int(tid)])
        available = (world.first_stocked[:, st] <= wk + 1) & ~world.stockout[:, st, wk]
        log_ratio = np.log(price[:, wk] / ref)
        price_term = -world.price_sens[hh] * np.where(log_ratio > 0, 2.0 * log_ratio, log_ratio)
        utility = (world.quality + world.attributes @ taste[hh] + world.idiosyncratic[hh]
                   + price_term + 0.5 * world.promo[:, wk])
        needs = set()
        n_missions = 1 + r.poisson(world.intensity[hh])
        for m in r.choice(len(MISSIONS), size=n_missions, p=world.mission_pref[hh]):
            if world.variants == 1:
                t = world.templates[m]
            else:
                v = world.usual_variant[hh, m] if r.random() < 0.6 else r.integers(world.variants)
                t = world.variant_templates[m][v]
            for c, p in zip(t["categories"], t["probs"]):
                if inventory[hh, c] == 0 and r.random() < p:
                    needs.add(int(c))
            for recipe in t["recipes"]:
                if r.random() < t["recipe_prob"]:
                    needs.update(int(c) for c in recipe)
        basket = set()
        for c in sorted(needs):
            items = np.flatnonzero((world.category == c) & available)
            if len(items) == 0:
                continue
            u = utility[items] + 1.2 * (items == loyalty[hh, c])
            subs = world.subcategory[items]
            # nested logit: subcategory inclusive values with nest parameter 0.5
            mu = 0.5
            nests = np.unique(subs)
            iv = np.array([mu * np.log(np.exp(u[subs == s] / mu).sum()) for s in nests])
            s_pick = nests[r.choice(len(nests), p=np.exp(iv - iv.max()) / np.exp(iv - iv.max()).sum())]
            inside = items[subs == s_pick]; w = np.exp(u[subs == s_pick] / mu)
            j = int(r.choice(inside, p=w / w.sum()))
            basket.add(j)
            loyalty[hh, c] = j
            if world.promo[j, wk] and r.random() < 0.5:
                inventory[hh, c] = int(r.integers(2, 5))  # stockpile
            if r.random() < 0.10 and len(items) > 1:      # variety seeking
                others = items[items != j]; w2 = np.exp(utility[others] - utility[others].max())
                basket.add(int(r.choice(others, p=w2 / w2.sum())))
        for _ in range(r.poisson(0.4)):                      # impulse purchases
            pool = np.flatnonzero(available)
            w3 = np.exp(world.quality[pool]); basket.add(int(r.choice(pool, p=w3 / w3.sum())))
        baskets.append(sorted(basket))
    state = {"loyalty": loyalty, "inventory": inventory, "taste": taste, "week": last_week}
    return baskets, state


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=20260924)
    p.add_argument("--households", type=int, default=3000)
    p.add_argument("--stores", type=int, default=15)
    p.add_argument("--departments", type=int, default=6)
    p.add_argument("--categories-per-department", type=int, default=8)
    p.add_argument("--mission-variants", type=int, default=1,
                   help="category templates per mission (scale with the category count for large catalogues)")
    p.add_argument("--weeks", type=int, default=52)
    p.add_argument("--train-weeks", type=int, default=32)
    p.add_argument("--validation-weeks", type=int, default=10)
    p.add_argument("--scenario-products", type=int, default=16)
    args = p.parse_args()
    started = time.time()
    rng = np.random.default_rng(args.seed)
    world = World(args, rng)
    H, S, T, J, C = args.households, args.stores, args.weeks, world.J, world.C

    visits = rng.random((H, T)) < world.visit_rate[:, None]
    hh, wk = np.nonzero(visits)
    store = np.where(rng.random(len(hh)) < 0.85, world.home[hh], rng.integers(0, S, len(hh)))
    day = wk * 7 + rng.integers(0, 7, len(hh))
    order = np.lexsort((hh, day)); hh, wk, store, day = hh[order], wk[order], store[order], day[order]
    trip_id = np.arange(len(hh))
    state0 = {"loyalty": np.full((H, C), -1), "inventory": np.zeros((H, C), dtype=int),
              "taste": world.taste0.copy(), "week": 0}
    test_start = args.train_weeks + args.validation_weeks
    before = wk < test_start
    baskets_a, state_test = simulate(world, (hh[before], store[before], wk[before], trip_id[before]), state0,
                                     seed_base=args.seed)
    test_trips = (hh[~before], store[~before], wk[~before], trip_id[~before])
    baskets_b, _ = simulate(world, test_trips, state_test, seed_base=args.seed)
    baskets = baskets_a + baskets_b
    keep = np.array([len(b) > 0 for b in baskets])            # the modeled experiment: nonempty baskets
    print(f"[stress] simulated {len(baskets)} trips in {time.time() - started:.0f}s; "
          f"{keep.mean():.3f} nonempty", flush=True)

    # ---- causal price truth on the test weeks (common random numbers, same starting state) ----
    counts = np.zeros(J)
    for i in np.flatnonzero(before):
        counts[baskets[i]] += 1
    eligible = np.flatnonzero(counts >= 30)
    chosen = np.sort(rng.choice(eligible, size=min(args.scenario_products, len(eligible)), replace=False))
    base_inc = np.zeros(J); base_size = []
    for b in baskets_b:
        if b:
            base_inc[b] += 1; base_size.append(len(b))
    base_n = len(base_size)
    scenarios = []
    for k in chosen:
        for multiplier in (0.8, 1.2):
            changed = world.price.copy()
            changed[k, test_start:] = np.round(changed[k, test_start:] * multiplier, 2)
            alt, _ = simulate(world, test_trips, state_test, price=changed, seed_base=args.seed)
            inc = np.zeros(J); size = []
            for b in alt:
                if b:
                    inc[b] += 1; size.append(len(b))
            alt_n = len(size)
            same_sub = np.flatnonzero((world.category == world.category[k])
                                      & (world.subcategory == world.subcategory[k]) & (np.arange(J) != k))
            same_cat = np.flatnonzero((world.category == world.category[k]) & (np.arange(J) != k))
            rest = np.setdiff1d(same_cat, same_sub)
            scenarios.append({
                "product": int(k), "price_multiplier": multiplier,
                "nonempty_test_baskets": [base_n, alt_n],
                "own_incidence": [float(base_inc[k] / base_n), float(inc[k] / alt_n)],
                "same_subcategory_incidence": [float(base_inc[same_sub].sum() / base_n), float(inc[same_sub].sum() / alt_n)],
                "rest_of_category_incidence": [float(base_inc[rest].sum() / base_n), float(inc[rest].sum() / alt_n)],
                "mean_basket_size": [float(np.mean(base_size)), float(np.mean(size))],
                "incidence_change_all_products": (inc / alt_n - base_inc / base_n).tolist()})
        print(f"[stress] counterfactual truth for product {k} done", flush=True)

    # ---- canonical outputs ----
    # contract: held-out customers need training trips; products need training purchases
    period_all = wk + 1
    trained_households = np.unique(hh[keep & (period_all <= args.train_weeks)])
    keep &= (period_all <= args.train_weeks) | np.isin(hh, trained_households)
    train_items = np.zeros(J, dtype=bool)
    for i in np.flatnonzero(keep & (period_all <= args.train_weeks)):
        train_items[baskets[i]] = True
    item_map = np.full(J, -1); item_map[train_items] = np.arange(int(train_items.sum()))
    baskets = [[int(item_map[j]) for j in b if item_map[j] >= 0] for b in baskets]
    keep &= np.array([len(b) > 0 for b in baskets])
    hh, wk, store, day, trip_id = hh[keep], wk[keep], store[keep], day[keep], trip_id[keep]
    baskets = [b for b, k in zip(baskets, keep) if k]
    kept = np.flatnonzero(train_items)                      # original ids of modeled products
    J_all, J = J, len(kept)
    period = wk + 1
    split = np.where(period <= args.train_weeks, "train",
                     np.where(period <= test_start, "validation", "test"))
    trip_row = np.repeat(np.arange(len(baskets)), [len(b) for b in baskets])
    item = np.concatenate(baskets).astype(np.int64)
    product_id = np.array([f"sku{j:05d}" for j in range(J)])        # modeled products only
    cat_names = np.array([f"cat{c:03d}" for c in range(C)])
    transactions = pd.DataFrame({
        "basket_id": trip_row.astype(np.int64), "customer_id": hh[trip_row].astype(np.int32),
        "store_id": store[trip_row].astype(np.int32), "period": period[trip_row].astype(np.int16),
        "day": day[trip_row].astype(np.int16), "product_id": product_id[item],
        "item_id": item.astype(np.int32), "category": cat_names[world.category[kept[item]]],
        "quantity": (1 + rng.poisson(0.3, len(item))).astype(np.float64),
        "unit_price": world.price[kept[item], wk[trip_row]], "split": split[trip_row]})
    # contiguous customers (some households never shop)
    customers = np.unique(transactions.customer_id)
    remap = np.full(H, -1); remap[customers] = np.arange(len(customers))
    transactions["customer_id"] = remap[transactions.customer_id].astype(np.int32)
    transactions["pre_coupon_value"] = transactions.quantity * transactions.unit_price
    products = pd.DataFrame({
        "item_id": np.arange(J, dtype=np.int32), "product_id": product_id,
        "category": cat_names[world.category[kept]],
        "subcategory": [f"cat{c:03d}_sub{s}" for c, s in zip(world.category[kept], world.subcategory[kept])],
        "label": [f"product {j}" for j in range(J)],
        "brand": [f"brand{j % 23}" for j in range(J)],
        "department": [f"dept{d}" for d in world.cat_dept[world.category[kept]]],
        "manufacturer": [f"maker{j % 9}" for j in range(J)]})
    stocked = world.first_stocked[kept][:, :, None] <= np.arange(1, T + 1)[None, None, :]
    popularity = np.exp(world.quality[kept] - 1.5)[:, None, None]
    non_panel = rng.poisson(200 * popularity / (1 + popularity) * np.ones((1, S, T)))
    non_panel = np.where(stocked & ~world.stockout[kept], non_panel, 0)
    panel_units = transactions.groupby(["item_id", "store_id", "period"]).quantity.sum()
    cells = []
    for j, s, t in zip(*np.nonzero(stocked)):
        units = float(non_panel[j, s, t]) + float(panel_units.get((j, s, t + 1), 0.0))
        if units > 0:
            cells.append((product_id[j], s, t + 1, units, units * world.price[kept[j], t]))
    store_week_prices = pd.DataFrame(cells, columns=["product_id", "store_id", "period", "units", "revenue"])
    store_week_prices["price_source"] = "retail_aggregate"
    pj, pt = np.nonzero(world.promo[kept])
    promotions = pd.DataFrame({"product_id": np.repeat(product_id[pj], S),
                               "store_id": np.tile(np.arange(S), len(pj)).astype(np.int32),
                               "period": np.repeat(pt + 1, S).astype(np.int16),
                               "display": True, "advertised": True, "special_price": True})
    opportunities = pd.DataFrame({"customer_id": remap[hh].astype(np.int32), "store_id": store.astype(np.int32),
                                  "period": period.astype(np.int16)}).drop_duplicates()
    canonical = args.output / "canonical"
    canonical.mkdir(parents=True, exist_ok=False)
    digests = {}
    for name, frame in {"transactions": transactions, "products": products,
                        "store_week_prices": store_week_prices, "promotions": promotions,
                        "shopping_opportunities": opportunities}.items():
        path = canonical / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    sizes = np.array([len(b) for b in baskets])
    (canonical / "build_audit.json").write_text(json.dumps({
        "schema_version": 1, "status": "passed", "generator": "scripts/synthetic/generate_stress_world.py",
        "source_sha256": digests,
        "audit": {"cohort_policy": {"minimum_product_training_lines": 1}, "trips": int(len(baskets)),
                  "lines": int(len(transactions)), "mean_basket_size": float(sizes.mean())}}, indent=2) + "\n")
    memberships = np.zeros((len(baskets), J), dtype=bool)
    memberships[trip_row, item] = True
    np.savez_compressed(args.output / "observed.npz", trip_household=remap[hh], trip_store=store,
                        trip_period=period, memberships=memberships, price=world.price[kept],
                        promo=world.promo[kept], category=world.category[kept],
                        subcategory=world.subcategory[kept], base_price=world.base_price[kept],
                        first_stocked=world.first_stocked[kept], original_item=kept)
    (args.output / "counterfactual_truth.json").write_text(json.dumps(
        {"test_weeks": [test_start + 1, T], "item_ids": "original product ids; observed.npz original_item maps "
         "modeled item ids to them", "scenarios": scenarios}) + "\n")
    summary = {"seed": args.seed, "households_with_trips": int(len(customers)), "stores": S, "products": J,
               "products_simulated": J_all,
               "categories": C, "weeks": T, "train_weeks": args.train_weeks,
               "validation_weeks": args.validation_weeks, "trips": int(len(baskets)),
               "lines": int(len(transactions)), "mean_basket_size": float(sizes.mean()),
               "max_basket_size": int(sizes.max()), "size_quantiles": np.quantile(sizes, [0.5, 0.9, 0.99]).tolist(),
               "trips_per_household_quantiles": np.quantile(np.bincount(remap[hh]), [0.1, 0.5, 0.9]).tolist(),
               "scenario_products": chosen.tolist(), "runtime_seconds": round(time.time() - started, 1)}
    (args.output / "truth.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
