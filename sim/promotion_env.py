"""A Gymnasium environment for testing promotion policies on a fitted basket model.

One step is one week.  The action sets a discount level for each of ``n_categories``
promotable commodities (all their products); the environment re-runs a fixed sample of
that week's real trips (their households, stores, prices and promotions as observed) with
the discounted prices, and returns the expected revenue.

    expected revenue = sum over trips and offered products of
                       P(product in basket | trip, prices) x average units x paid price

Paid price is the week's chain price, times (1 - discount) for promoted products.  So the
reward is revenue at regular prices minus the markdown given away.  ``info`` reports both
parts, the revenue per commodity, and the uplift over the same week without promotion.

Purchase probabilities come from the model (THEORY.md section 7):
    simulator="base"  the exact base model (fast: about a second per step)
    simulator="v4"    a Version-4 model: latent draws are taken once per week at the
                      factual prices and reused for every action, each reweighted exactly
                      by Z_new(z) / Z_old(z) (minutes per step; for evaluating policies)

A discount lowers the product's log price by log(1 - d) and, with a discount panel, its
discount depth by the same amount, so the model's discount slope applies.

Limits: the model is conditional on a trip, so a promotion cannot bring in extra shoppers;
units per product are the training average (the model predicts incidence, not quantity);
simulated purchases do not feed back into later weeks' purchase history.

Caution: the current models add an invented gain to the rest of the basket when one
commodity is discounted (the halo; docs/PIPELINE.md section 7), so the revenue numbers are
too high.  Use the environment for plumbing and policy-code tests until that is fixed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from v4.bundle import data_root, load_bundle  # noqa: E402
from v4.checkpoints import batcher_for, freeze, load_additive, load_version4  # noqa: E402
from v4.exact import incidence, log_partition  # noqa: E402
from v4.price import changed_price_context  # noqa: E402
from v4.sampling import draw_from_model, gibbs  # noqa: E402


class PromotionEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, checkpoint, *, simulator="base", data_root_path=None, initialization=None,
                 weeks=None, n_categories=12, levels=(0.0, 0.1, 0.2, 0.3), trips_per_week=500,
                 draws=32, burn=100, batch=128, seed=0, threads=4):
        super().__init__()
        torch.set_default_dtype(torch.float64)
        torch.set_num_threads(threads)
        self.root = data_root(data_root_path)
        self.data = load_bundle(self.root)
        self.simulator = simulator
        if simulator == "base":
            self.model, _, blob = load_additive(initialization, checkpoint, self.data, self.root)
            self.batcher = batcher_for(self.data, self.root, blob)
        elif simulator == "v4":
            self.model, self.batcher, _, _ = load_version4(checkpoint, self.data, self.root,
                                                           initialization=initialization)
        else:
            raise ValueError("simulator must be 'base' or 'v4'")
        freeze(self.model)
        self.levels = np.asarray(levels, dtype=float)
        self.draws, self.burn, self.batch = draws, burn, batch

        meta_split = self.data["trip_split"]
        weeks_all = self.data["trip_week"]
        self.weeks = list(weeks or sorted(np.unique(weeks_all[meta_split == 1])))
        items = pd.read_parquet(self.root / "basket_input/items.parquet").sort_values("item_id").reset_index(drop=True)
        lines = pd.read_parquet(self.root / "basket_input/baskets.parquet",
                                columns=["item_id", "units", "price", "split"])
        train = lines[lines.split == "train"]
        J = int(self.data["n_item"])
        self.units = (train.groupby("item_id").units.mean().reindex(range(J)).fillna(1.0).to_numpy())
        revenue = (train.units * train.price).groupby(items.COMMODITY_DESC.to_numpy()[train.item_id]).sum()
        self.categories = list(revenue.sort_values(ascending=False).index[:n_categories])
        self.item_category = np.full(J, -1)
        for k, c in enumerate(self.categories):
            self.item_category[(items.COMMODITY_DESC == c).to_numpy()] = k
        self.price = self._weekly_prices(items, J)

        rng = np.random.default_rng(seed)
        nmax = self.model.nmax
        self.week_trips = {}
        for w in self.weeks:
            pool = np.flatnonzero((weeks_all == w) & (self.data["trip_nlines"] <= nmax))
            self.week_trips[w] = np.sort(rng.choice(pool, size=min(trips_per_week, len(pool)), replace=False))
        self._cache = {}                                    # per week: batches, latent draws, baseline
        K = len(self.categories)
        self.action_space = gym.spaces.MultiDiscrete([len(self.levels)] * K)
        self.observation_space = gym.spaces.Box(0.0, np.inf, shape=(1 + 2 * K,), dtype=np.float64)
        self._t = 0
        self._last = (np.zeros(K), np.zeros(K))

    # ---- data ----------------------------------------------------------------------------------
    def _weekly_prices(self, items, J):
        week = pd.read_parquet(self.root / "data/price_week.parquet", columns=["PRODUCT_ID", "WEEK_NO", "price"])
        week = week.merge(items[["PRODUCT_ID", "item_id"]], on="PRODUCT_ID")
        n = int(self.data["trip_week"].max()) + 1
        grid = np.full((J, n), np.nan)
        grid[week.item_id.to_numpy(), week.WEEK_NO.clip(upper=n - 1).to_numpy()] = week.price.to_numpy()
        return pd.DataFrame(grid).ffill(axis=1).bfill(axis=1).to_numpy()

    def _week(self, w):
        """Batches of the week's trips and, for Version 4, latent draws at factual prices."""
        if w not in self._cache:
            trips = self.week_trips[w]
            gen = torch.Generator().manual_seed(int(w))
            parts = []
            for s in range(0, len(trips), self.batch):
                batch = self.batcher.make(trips[s:s + self.batch])
                batch.bind(self.model)
                zs, lz0 = None, None
                if self.simulator == "v4":
                    ix = batch.ix
                    X = draw_from_model(self.model, ix, gen, self.burn)
                    zs = []
                    for _ in range(self.draws):
                        X = gibbs(self.model, ix, X, gen)
                        zs.append(self.model.basket_field(ix, X) + torch.randn(ix.B, self.model.rank, generator=gen))
                    iv = self.model.iv_shift(ix)
                    with torch.no_grad():
                        lz0 = torch.stack([log_partition(self.model, ix, self.model.slot_utilities_given_z(ix, z),
                                                         z=z, iv=iv) for z in zs], 1)
                parts.append((batch, zs, lz0))
            self._cache[w] = {"parts": parts}
            self._cache[w]["baseline"] = self._simulate(w, np.zeros(len(self.categories), dtype=int))
        return self._cache[w]

    # ---- the model -----------------------------------------------------------------------------
    def _incidence(self, batch, zs, lz0, ctx):
        m, ix = self.model, batch.ix
        batch.bind(m, ctx)
        if zs is None:
            return incidence(m, ix, m.slot_utilities(ix))[1]
        iv = m.iv_shift(ix)
        out = [incidence(m, ix, m.slot_utilities_given_z(ix, z), z=z, iv=iv) for z in zs]
        w = torch.softmax(torch.stack([o[0] for o in out], 1) - lz0, 1)
        return (torch.stack([o[1] for o in out], 1) * w[ix.item_trip]).sum(1)

    def _simulate(self, w, action):
        """Expected revenue at regular prices, markdown, and revenue per category."""
        depth = self.levels[np.asarray(action)]
        parts = self._week(w)["parts"] if w in self._cache else None
        if parts is None:
            raise RuntimeError("week not prepared")
        K = len(self.categories)
        regular, markdown, by_cat = 0.0, 0.0, np.zeros(K)
        for batch, zs, lz0 in parts:
            ix = batch.ix
            item = ix.item.numpy()
            k = self.item_category[item]
            d = np.where(k >= 0, depth[np.maximum(k, 0)], 0.0)
            change = torch.from_numpy(np.log1p(-d))
            ctx = changed_price_context(batch.ctx, ix.item_trip, change, discount_change=change) if d.any() else batch.ctx
            p = self._incidence(batch, zs, lz0, ctx).numpy()
            value = p * self.units[item] * self.price[item, w]
            regular += float(value.sum())
            markdown += float((value * d).sum())
            np.add.at(by_cat, k[k >= 0], (value * (1 - d))[k >= 0])
        batch.bind(self.model)
        return {"regular_revenue": regular, "markdown": markdown, "revenue": regular - markdown,
                "revenue_by_category": by_cat}

    # ---- the Gymnasium interface ------------------------------------------------------------------
    def _obs(self):
        return np.concatenate([[self._t / len(self.weeks)], self._last[0], self._last[1]]).astype(np.float64)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        self._last = (np.zeros(len(self.categories)), np.zeros(len(self.categories)))
        return self._obs(), {"week": self.weeks[0], "categories": self.categories}

    def step(self, action):
        w = self.weeks[self._t]
        base = self._week(w)["baseline"]
        out = self._simulate(w, action)
        scale = max(base["revenue"], 1.0)
        reward = out["revenue"] / scale                    # revenue relative to the no-promotion week
        info = {"week": w, **{k: v for k, v in out.items() if k != "revenue_by_category"},
                "baseline_revenue": base["revenue"], "uplift": out["revenue"] - base["revenue"],
                "revenue_by_category": dict(zip(self.categories, np.round(out["revenue_by_category"], 2)))}
        self._last = (self.levels[np.asarray(action)], out["revenue_by_category"] / np.maximum(
            base["revenue_by_category"], 1e-9))
        self._t += 1
        terminated = self._t >= len(self.weeks)
        return self._obs(), reward, terminated, False, info
