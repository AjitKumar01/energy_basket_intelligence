#!/usr/bin/env python3
"""Generate a known-truth retail world in the canonical basket input format.

The world exercises every capability the pipeline claims:

* household taste in three latent segments plus a household basket-size propensity;
* randomized chain prices (i.i.d. weekly log multipliers) with category price sensitivity;
* store assortments and store-level product launches, including launches after training;
* cross-category complement interactions (rank-2 phi), within-category substitution
  (rho_c) and a basket-size potential;
* an independent store sales feed (non-panel shoppers plus the panel's own units).

Outputs: ``<output>/canonical/*`` (the canonical contract) and ``<output>/truth.npz``
plus ``truth.json`` (all generating parameters, stock calendar and segments).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from basket_world import World, gibbs_sample, utilities  # noqa: E402

CATEGORIES = ("bakery", "dairy", "produce", "snacks", "beverages", "pantry", "frozen", "household")


def make_world(rng: np.random.Generator, products_per_category: int, nmax: int):
    C = len(CATEGORIES)
    J = C * products_per_category
    category = np.repeat(np.arange(C), products_per_category)
    lam = rng.normal(-3.5, 0.7, J)
    Kt = 3
    alpha = rng.normal(0.0, 0.6, (J, Kt))
    # Two shopping missions across categories: breakfast (bakery, dairy, beverages) and
    # party (snacks, beverages, frozen). Complements have aligned phi.
    phi = rng.normal(0.0, 0.08, (J, 2))
    missions = {0: (0, 1, 4), 1: (3, 4, 6)}
    for axis, members in missions.items():
        for c in members:
            chosen = rng.choice(np.flatnonzero(category == c), size=products_per_category // 2,
                                replace=False)
            phi[chosen, axis] += rng.uniform(0.45, 0.75, len(chosen))
    rho_c = rng.uniform(0.4, 1.2, C)
    n = np.arange(nmax + 1, dtype=np.float64)
    rho0 = 0.06 * (n - 1) ** 2
    rho0[0] = 0.0
    sensitivity = np.round(np.linspace(0.6, 2.4, C)[rng.permutation(C)], 3)
    return World(lam=lam, alpha=alpha, phi=phi, category=category, rho_c=rho_c, rho0=rho0,
                 price_sensitivity=sensitivity)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--households", type=int, default=3000)
    parser.add_argument("--stores", type=int, default=20)
    parser.add_argument("--products-per-category", type=int, default=15)
    parser.add_argument("--weeks", type=int, default=52)
    parser.add_argument("--train-weeks", type=int, default=32)
    parser.add_argument("--validation-weeks", type=int, default=10)
    parser.add_argument("--visit-probability", type=float, default=0.55)
    parser.add_argument("--nmax", type=int, default=12)
    parser.add_argument("--sweeps", type=int, default=30)
    parser.add_argument("--chunk", type=int, default=8000)
    args = parser.parse_args()
    started = time.time()
    rng = np.random.default_rng(args.seed)
    canonical = args.output / "canonical"
    canonical.mkdir(parents=True, exist_ok=False)
    world = make_world(rng, args.products_per_category, args.nmax)
    J, C, S, H, T = len(world.lam), len(CATEGORIES), args.stores, args.households, args.weeks

    # ---- stock calendar: first stocked week per (product, store); weeks+1 = never ----
    first_stocked = np.where(rng.random((J, S)) < 0.8, 1, T + 1).astype(np.int16)
    launched = rng.choice(J, size=J // 8, replace=False)
    for j in launched:
        initial = rng.random(S) < 0.4
        initial[rng.integers(S)] = True              # stocked somewhere from week 1
        first_stocked[j] = np.where(initial, 1, rng.integers(12, T - 2, S))
    for j in range(J):                               # every product is sold during training
        if not (first_stocked[j] <= args.train_weeks // 2).any():
            first_stocked[j, rng.integers(S)] = 1

    # ---- households ----
    segment = rng.integers(0, 3, H)
    centers = rng.normal(0.0, 1.0, (3, world.alpha.shape[1]))
    taste = centers[segment] + rng.normal(0.0, 0.35, (H, world.alpha.shape[1]))
    size_shift = rng.normal(0.0, 0.3, H)
    home = rng.integers(0, S, H)

    # ---- randomized chain prices ----
    base_price = np.round(np.exp(rng.normal(1.0, 0.5, J)), 2)
    log_multiplier = rng.uniform(-0.25, 0.25, (J, T))
    price = np.round(base_price[:, None] * np.exp(log_multiplier), 2)
    log_price = np.log(price)
    dlp = log_price - log_price[:, :args.train_weeks].mean(1, keepdims=True)

    # ---- trips ----
    visits = rng.random((H, T)) < args.visit_probability
    hh, week_index = np.nonzero(visits)
    store = np.where(rng.random(len(hh)) < 0.85, home[hh], rng.integers(0, S, len(hh)))
    period = week_index + 1
    day = week_index * 7 + rng.integers(0, 7, len(hh))
    order = np.lexsort((store, hh, day))
    hh, store, period, day, week_index = (a[order] for a in (hh, store, period, day, week_index))
    trips = len(hh)
    memberships = np.zeros((trips, J), dtype=bool)
    for start in range(0, trips, args.chunk):
        stop = min(trips, start + args.chunk)
        rows = slice(start, stop)
        available = first_stocked[:, store[rows]].T <= period[rows, None]
        b = utilities(world, taste[hh[rows]], size_shift[hh[rows]],
                      dlp[:, week_index[rows]].T, available)
        memberships[rows] = gibbs_sample(world, b, rng, sweeps=args.sweeps)
        print(f"[world] sampled trips {stop}/{trips}", flush=True)

    train_last = args.train_weeks
    validation_last = args.train_weeks + args.validation_weeks
    split = np.where(period <= train_last, "train",
                     np.where(period <= validation_last, "validation", "test"))
    trip_row, item = np.nonzero(memberships)
    quantity = 1 + rng.poisson(0.3, len(item))
    transactions = pd.DataFrame({
        "basket_id": trip_row.astype(np.int64),
        "customer_id": hh[trip_row].astype(np.int32),
        "store_id": store[trip_row].astype(np.int32),
        "period": period[trip_row].astype(np.int16),
        "day": day[trip_row].astype(np.int16),
        "product_id": [f"sku{j:04d}" for j in item],
        "item_id": item.astype(np.int32),
        "category": [CATEGORIES[c] for c in world.category[item]],
        "quantity": quantity.astype(np.float64),
        "unit_price": price[item, week_index[trip_row]],
        "split": split[trip_row],
    })
    transactions["pre_coupon_value"] = transactions.quantity * transactions.unit_price
    products = pd.DataFrame({
        "item_id": np.arange(J, dtype=np.int32),
        "product_id": [f"sku{j:04d}" for j in range(J)],
        "category": [CATEGORIES[c] for c in world.category],
        "label": [f"{CATEGORIES[c]} product {j:03d}" for j, c in enumerate(world.category)],
        "brand": [f"brand{j % 11}" for j in range(J)],
        "department": "SYNTHETIC",
        "manufacturer": [f"maker{j % 7}" for j in range(J)],
    })

    # ---- independent store sales feed: non-panel shoppers plus the panel's own units ----
    shoppers = rng.integers(150, 400, S)
    marginal = 1.0 / (1.0 + np.exp(-(world.lam[:, None] - 1.0
                                     - world.price_sensitivity[world.category][:, None] * dlp)))
    non_panel = rng.poisson(shoppers[None, :, None] * marginal[:, None, :])        # [J, S, T]
    stocked = first_stocked[:, :, None] <= np.arange(1, T + 1)[None, None, :]
    non_panel = np.where(stocked, non_panel, 0)
    panel_units = transactions.groupby(["item_id", "store_id", "period"]).quantity.sum()
    cells = []
    for j, s, t in zip(*np.nonzero(stocked)):
        units = float(non_panel[j, s, t]) + float(panel_units.get((j, s, t + 1), 0.0))
        if units > 0:
            cells.append((f"sku{j:04d}", s, t + 1, units, units * price[j, t]))
    store_week_prices = pd.DataFrame(cells, columns=["product_id", "store_id", "period",
                                                     "units", "revenue"])
    store_week_prices["price_source"] = "retail_aggregate"
    promotions = pd.DataFrame({"product_id": pd.Series(dtype=str),
                               "store_id": pd.Series(dtype=np.int32),
                               "period": pd.Series(dtype=np.int16),
                               "display": pd.Series(dtype=bool),
                               "advertised": pd.Series(dtype=bool),
                               "special_price": pd.Series(dtype=bool)})
    opportunities = pd.DataFrame({"customer_id": hh.astype(np.int32),
                                  "store_id": store.astype(np.int32),
                                  "period": period.astype(np.int16)}).drop_duplicates()

    outputs = {"transactions": transactions, "products": products,
               "store_week_prices": store_week_prices, "promotions": promotions,
               "shopping_opportunities": opportunities}
    digests = {}
    for name, frame in outputs.items():
        path = canonical / f"{name}.parquet"
        frame.to_parquet(path, index=False)
        digests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    audit = {
        "schema_version": 1, "status": "passed", "generator": "scripts/synthetic/generate_canonical_world.py",
        "source_sha256": digests,
        "audit": {"cohort_policy": {"minimum_product_training_lines": 1},
                  "trips": int(trips), "lines": int(len(transactions)),
                  "mean_basket_size": float(memberships.sum(1).mean())},
    }
    (canonical / "build_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    np.savez_compressed(
        args.output / "truth.npz", lam=world.lam, alpha=world.alpha, phi=world.phi,
        category=world.category, rho_c=world.rho_c, rho0=world.rho0,
        price_sensitivity=world.price_sensitivity, first_stocked=first_stocked,
        segment=segment, taste=taste, size_shift=size_shift, home=home, price=price,
        dlp=dlp, trip_household=hh, trip_store=store, trip_period=period, trip_day=day,
        memberships=memberships)
    summary = {
        "seed": args.seed, "households": H, "stores": S, "products": J, "categories": C,
        "weeks": T, "train_weeks": args.train_weeks, "validation_weeks": args.validation_weeks,
        "nmax": args.nmax, "trips": int(trips), "lines": int(len(transactions)),
        "mean_basket_size": float(memberships.sum(1).mean()),
        "size_distribution": np.bincount(memberships.sum(1), minlength=args.nmax + 1).tolist(),
        "price_sensitivity_by_category": dict(zip(CATEGORIES, world.price_sensitivity.tolist())),
        "rho_c": dict(zip(CATEGORIES, np.round(world.rho_c, 4).tolist())),
        "stocked_pair_share_week1": float((first_stocked <= 1).mean()),
        "launched_products": int(len(launched)),
        "launch_cells_after_training": int(((first_stocked > args.train_weeks)
                                            & (first_stocked <= T)).sum()),
        "runtime_seconds": round(time.time() - started, 1),
    }
    (args.output / "truth.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
