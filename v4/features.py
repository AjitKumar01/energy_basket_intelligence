"""Context features gathered at every offered product of a trip.

The normaliser scores every product a store offers, not only the purchased ones, so every
feature is looked up at every (trip, offered product) slot.  Panels, all from the bundle's
``basket_input/``:

    log_price_dev.npy   dense [product, day]: log price minus the product's training-mean
                        log price, from the chain-wide price (same in every store)
    store_price.npz     sparse (product, store, week) store deviations.  Only for the old
                        "chain_and_store" contract: they exist only where the product sold,
                        so they leak purchases into the feature.  The pipeline uses
                        "chain_product_week_only", which leaves them out.
    discount panel      optional dense [product, day] discount depth min(log(price/regular), 0),
                        centred on each product's training days
                        (scripts/build_discount_panel.py)
    promo.npz           sparse (product, store, week) display and mailer indicators
    state.npz           purchase events for the sub-commodity recency features (optional)
    availability.npz    optional: the week each product is first confirmed at each store

Sparse keys pack (product, store, week) as (product * n_store + store) * 128 + week, and
recency keys pack (household group, day) as group * 1024 + day; the constructor checks the
data fits these strides.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

WEEK_STRIDE = 128
DAY_STRIDE = 1024
PRICE_CONTRACTS = ("chain_and_store", "chain_product_week_only")


def log(message):
    print(f"[features] {message}", flush=True)


def sparse_lookup(keys, values, query):
    """Value where the key exists, zero where it does not (keys sorted)."""
    if len(keys) == 0:
        return torch.zeros_like(query, dtype=values.dtype)
    i = torch.searchsorted(keys, query).clamp(max=len(keys) - 1)
    return torch.where(keys[i] == query, values[i], torch.zeros((), dtype=values.dtype))


class Features:
    """All panels held in memory once and gathered per batch."""

    def __init__(self, root: Path, n_item: int, n_store: int, *, recency: bool,
                 price_contract: str, discount_panel: str | Path | None = None):
        folder = Path(root) / "basket_input"
        if price_contract not in PRICE_CONTRACTS:
            raise ValueError(f"unknown price contract {price_contract!r}")
        self.S = n_store
        self.include_recency = bool(recency)

        # ---- price --------------------------------------------------------------------
        self.dev = torch.from_numpy(np.load(folder / "log_price_dev.npy").astype(np.float32))
        if self.dev.shape[0] != n_item or self.dev.shape[1] > DAY_STRIDE:
            raise RuntimeError(f"price panel shape {tuple(self.dev.shape)} does not fit "
                               f"{n_item} products and {DAY_STRIDE} days")
        self.sp_key = torch.empty(0, dtype=torch.int64)
        self.sp_val = torch.empty(0, dtype=torch.float32)
        if price_contract == "chain_and_store":
            sp = np.load(folder / "store_price.npz")
            if len(sp["week"]) and int(sp["week"].max()) >= WEEK_STRIDE:
                raise RuntimeError("store price weeks exceed the key stride")
            key = (sp["item"].astype(np.int64) * n_store + sp["store"]) * WEEK_STRIDE + sp["week"]
            order = np.argsort(key)
            self.sp_key = torch.from_numpy(key[order])
            self.sp_val = torch.from_numpy(sp["dev"][order].astype(np.float32))
        log(f"price panel {tuple(self.dev.shape)}, contract {price_contract}")

        # ---- discount depth (optional) ---------------------------------------------------
        self.discount_panel = str(discount_panel) if discount_panel else None
        self.dsc = None
        if self.discount_panel:
            self.dsc = torch.from_numpy(np.load(self.discount_panel).astype(np.float32))
            if self.dsc.shape != self.dev.shape:
                raise RuntimeError("discount panel and price panel shapes differ")
            log(f"discount panel {self.discount_panel}: sd {float(self.dsc.std()):.4f}")

        # ---- promotions ---------------------------------------------------------------------
        promo = np.load(folder / "promo.npz")
        meta = json.loads((folder / "meta.json").read_text())
        week_range = [int(promo["coverage_min_week"]), int(promo["coverage_max_week"])]
        if not 1 <= week_range[0] <= week_range[1] < WEEK_STRIDE:
            raise RuntimeError("promotion weeks must lie in 1..127")
        if meta.get("promotion_coverage_required") != week_range:
            raise RuntimeError(f"promotion coverage {week_range} does not match the baskets")
        order = np.argsort(promo["keys"])
        self.promo_key = torch.from_numpy(promo["keys"][order])
        self.display = torch.from_numpy(promo["disp"][order].astype(np.float32))
        self.mailer = torch.from_numpy(promo["mail"][order].astype(np.float32))

        # ---- sub-commodity recency (optional) ------------------------------------------------
        if self.include_recency:
            state = np.load(folder / "state.npz")
            self.event_keys = torch.from_numpy(state["keys"])
            self.item_sub = torch.from_numpy(state["item_sub"].astype(np.int64))
            self.sub_gap = torch.from_numpy(state["sub_gap"])
            self.n_sub = int(self.item_sub.max()) + 1

        # ---- availability (optional; canonical bundles such as ERIM) ------------------------
        contract = meta.get("availability_contract") or {"enabled": False}
        self.availability = bool(contract.get("enabled", False))
        if self.availability:
            with np.load(folder / "availability.npz") as panel:
                self.first_week = torch.from_numpy(panel["first_period"].astype(np.int64))
                self.log_floor = float(panel["log_floor"])
            if tuple(self.first_week.shape) != (n_item, n_store) or self.log_floor > 0:
                raise RuntimeError("availability panel does not match the catalogue")
            log(f"availability floor {math.exp(self.log_floor):.4g}")

    def gather(self, item, store, day, week):
        """Price deviation, display and mailer at each (product, store, day, week) slot."""
        key = (item * self.S + store) * WEEK_STRIDE + week
        dlp = self.dev[item, day] + sparse_lookup(self.sp_key, self.sp_val, key)
        return (dlp, sparse_lookup(self.promo_key, self.display, key),
                sparse_lookup(self.promo_key, self.mailer, key))

    def discount(self, item, day):
        return self.dsc[item, day]

    def log_availability(self, item, store, week):
        """0 once a product is confirmed at a store, log(epsilon) before."""
        confirmed = week >= self.first_week[item, store]
        return torch.where(confirmed, torch.zeros((), dtype=torch.float64),
                           torch.full((), self.log_floor, dtype=torch.float64))

    def recency(self, item, user, day):
        """Four features of the household's last purchase in the product's sub-commodity,
        strictly before this day: never bought, exp(-days/7), exp(-days/typical gap),
        log(1 + days) / log(100)."""
        sub = self.item_sub[item]
        group = user.to(torch.int64) * self.n_sub + sub
        idx = torch.searchsorted(self.event_keys, group * DAY_STRIDE + day)
        prev = self.event_keys[(idx - 1).clamp(0, len(self.event_keys) - 1)]
        seen = (idx > 0) & (torch.div(prev, DAY_STRIDE, rounding_mode="floor") == group)
        since = torch.where(seen, (day - prev % DAY_STRIDE).double(),
                            torch.zeros(1, dtype=torch.float64))
        gap = self.sub_gap[sub].double().clamp_min(1.0)
        zero = torch.zeros(1, dtype=torch.float64)
        return torch.stack([(~seen).double(),
                            torch.where(seen, torch.exp(-since / 7.0), zero),
                            torch.where(seen, torch.exp(-since / gap), zero),
                            torch.where(seen, torch.log1p(since) / math.log(100.0), zero)], dim=-1)
