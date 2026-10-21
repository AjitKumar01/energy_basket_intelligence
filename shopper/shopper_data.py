"""Data sources for SHOPPER: the paper's TSV format and the Version-4 model-data bundles.

Each source exposes n_item, n_user, n_week, item_group, split trip indices (train /
validation / test) and make(trip_indices) -> shopper.Trips.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from shopper import Trips


def _read_tsv(path, names):
    text = Path(path).read_text().replace("\r\n", "\n").replace("\r", "\n")
    rows = [r.split("\t") for r in text.split("\n") if r.strip()]
    return pd.DataFrame(rows, columns=names[:len(rows[0])]).apply(pd.to_numeric)


class TsvSource:
    """The released code's format (see its README).  Trips are (user, session) pairs; every
    item is offered on every trip; price is log(price) (normPrice 0) or log(price / item
    mean) (normPrice 1).  A random share of training trips is held out for validation."""

    def __init__(self, directory, norm_price=1, validation_share=0.1, seed=0):
        d = Path(directory)
        tr = _read_tsv(d / "train.tsv", ["user", "item", "session", "qty"]).assign(split=0)
        te = _read_tsv(d / "test.tsv", ["user", "item", "session", "qty"]).assign(split=2)
        lines = pd.concat([tr, te])
        price = _read_tsv(d / "item_sess_price.tsv", ["item", "session", "price"])
        group = _read_tsv(d / "itemGroup.tsv", ["item", "group"])
        self.items = np.sort(lines.item.unique())
        self.users = np.sort(lines.user.unique())
        imap = {v: i for i, v in enumerate(self.items)}
        umap = {v: i for i, v in enumerate(self.users)}
        # A trip is (user, session, source file): test trips are independent of training trips
        # even when they share a session id.
        trips = lines[["user", "session", "split"]].drop_duplicates().reset_index(drop=True)
        key = {(u, s, sp): k for k, (u, s, sp) in enumerate(zip(trips.user, trips.session, trips.split))}
        self.basket = [[] for _ in range(len(trips))]
        for u, i, s, sp in zip(lines.user, lines.item, lines.session, lines.split):
            self.basket[key[(u, s, sp)]].append(imap[i])
        rng = np.random.default_rng(seed)
        val = (trips.split == 0) & (rng.random(len(trips)) < validation_share)
        trips.loc[val, "split"] = 1
        self.trip_split = trips.split.to_numpy()
        self.trip_user = np.array([umap[u] for u in trips.user])
        self.trip_session = trips.session.to_numpy()
        J = len(self.items)
        sessions = np.sort(price.session.unique())
        smap = {v: i for i, v in enumerate(sessions)}
        grid = np.ones((len(sessions), J))
        ok = price.item.isin(imap)
        grid[price.session[ok].map(smap), price.item[ok].map(imap)] = price.price[ok]
        lp = np.log(grid)
        if norm_price == 1:
            lp = lp - np.log(grid.mean(0, keepdims=True))
        self.session_price = lp
        self.session_index = np.array([smap[s] for s in self.trip_session])
        gmap = dict(zip(group.item, group.group))
        gid = np.array([gmap.get(v, v) for v in self.items])
        self.item_group = np.unique(gid, return_inverse=True)[1]
        self.n_item, self.n_user, self.n_week = J, len(self.users), 1

    def split(self, s):
        return np.flatnonzero(self.trip_split == s)

    def make(self, trips) -> Trips:
        B, J = len(trips), self.n_item
        sizes = [len(self.basket[t]) for t in trips]
        N = max(1, max(sizes))
        basket = torch.full((B, N), -1, dtype=torch.long)
        for b, t in enumerate(trips):
            basket[b, :sizes[b]] = torch.as_tensor(self.basket[t])
        return Trips(offered=torch.arange(J).expand(B, J).clone(), valid=torch.ones(B, J, dtype=torch.bool),
                     price=torch.as_tensor(self.session_price[self.session_index[trips]], dtype=torch.get_default_dtype()),
                     offset=torch.zeros(B, J), user=torch.as_tensor(self.trip_user[trips]),
                     week=torch.full((B,), -1), basket=basket, size=torch.as_tensor(sizes))


class BundleSource:
    """A Version-4 model-data bundle through the same Batcher the energy model uses: each
    trip's offered items are its store's assortment, the price is the energy model's log
    price deviation (dlp, from the item's training mean), the offset its log-availability,
    and the season the week of year.  Item groups are subcategories."""

    def __init__(self, root=None, price_contract="chain_and_store", discount_panel=None):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from v4.batches import Batcher
        from v4.bundle import data_root, item_categories, load_bundle
        from v4.features import Features
        root = data_root(root)
        data = load_bundle(root)
        self.data = data
        J = int(data["n_item"])
        self.batcher = Batcher(data, Features(root, J, int(data["n_store"]), recency=False,
                                              price_contract=price_contract, discount_panel=discount_panel))
        items = pd.read_parquet(root / "basket_input/items.parquet").sort_values("item_id")
        col = "sub_id" if "sub_id" in items.columns else "cat_id"
        self.item_group = np.unique(items[col].to_numpy(), return_inverse=True)[1]
        self.item_cat = item_categories(data)
        self.n_item, self.n_user, self.n_week = J, int(data["n_user"]), 52
        self.trip_split = np.asarray(data["trip_split"])
        self.trip_user = np.asarray(data["trip_user"])

    def split(self, s):
        return np.flatnonzero((self.trip_split == s) & (np.asarray(self.data["trip_nlines"]) >= 1))

    def make(self, trips, ctx_override=None) -> Trips:
        batch = self.batcher.make(np.asarray(trips))
        ix, ctx, house, li, lt = batch.ix, batch.ctx, batch.house, batch.line_item, batch.line_trip
        if ctx_override is not None:
            ctx = ctx_override(ix, ctx)
        B = len(trips)
        it = ix.item_trip
        counts = torch.bincount(it, minlength=B)
        M = int(counts.max())
        start = torch.cumsum(counts, 0) - counts
        pos = torch.arange(len(it)) - start[it]
        dt = torch.get_default_dtype()
        offered = torch.zeros(B, M, dtype=torch.long)
        valid = torch.zeros(B, M, dtype=torch.bool)
        price = torch.zeros(B, M, dtype=dt)
        offset = torch.zeros(B, M, dtype=dt)
        offered[it, pos] = ix.item
        valid[it, pos] = True
        price[it, pos] = ctx["dlp"].to(dt)
        if "log_avail" in ctx:
            offset[it, pos] = ctx["log_avail"].to(dt)
        # purchased lines -> slot positions
        key = it * self.n_item + ix.item
        order = torch.argsort(key)
        q = lt * self.n_item + li
        hit = order[torch.searchsorted(key[order], q).clamp(max=len(order) - 1)]
        if not bool((key[hit] == q).all()):
            raise RuntimeError("a purchased item is outside its trip's assortment")
        sizes = torch.bincount(lt, minlength=B)
        N = max(1, int(sizes.max()))
        lstart = torch.cumsum(sizes, 0) - sizes
        lpos = torch.arange(len(lt)) - lstart[lt]
        basket = torch.full((B, N), -1, dtype=torch.long)
        basket[lt, lpos] = pos[hit]
        week = ctx["week"].new_zeros(B)
        week[it] = ctx["week"]
        return Trips(offered=offered, valid=valid, price=price, offset=offset,
                     user=house.long(), week=week.long(), basket=basket, size=sizes)
