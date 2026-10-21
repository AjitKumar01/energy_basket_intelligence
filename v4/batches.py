"""One batch of trips laid out for the model.

Every offered product of every trip in the batch is one *slot*.  Slots are kept in a flat
array grouped into *rows*, one row per (trip, category); only the category axis is padded
(to the largest number of categories in any trip of the batch, ``Cpad``).  A store offers
up to ~1,800 products in one category and ~5,000 in total, so padding the product axis
would waste most of the work.

A ``Batch`` carries this layout (``ix``), the features at every slot (``ctx``), the same
features at the purchased lines (``line_ctx``), and the observed basket as a slot mask.
The energy and the normaliser read the same features, so a product is scored identically
whether it is bought or only offered.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .bundle import item_categories


class RaggedIndex:
    """Integer bookkeeping of one batch (no parameters).

        item      [T]       product id of each slot
        row_of    [T]       row (trip, category) of each slot
        item_pos  [T]       position of the slot inside its row
        item_trip [T]       trip of each slot (0..B-1 within the batch)
        row_trip  [rows]    trip of each row
        row_cat   [rows]    category of each row
        row_pos   [rows]    position of the row inside its trip (0..Cpad-1)
        row_size  [rows]    number of offered products in the row
        flat_slot [rows]    row_trip * Cpad + row_pos, for padding rows into a [B, Cpad] grid
    """

    def __init__(self, item, row_of, row_trip, row_cat, n_trips):
        self.item = torch.as_tensor(item, dtype=torch.long)
        self.row_of = torch.as_tensor(row_of, dtype=torch.long)
        self.row_trip = torch.as_tensor(row_trip, dtype=torch.long)
        self.row_cat = torch.as_tensor(row_cat, dtype=torch.long)
        self.n_rows = len(self.row_trip)
        self.B = n_trips
        counts = torch.bincount(self.row_trip, minlength=n_trips)
        self.Cpad = int(counts.max())
        order = torch.argsort(self.row_trip, stable=True)
        first_row = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(counts, 0)[:-1]])
        self.row_pos = torch.zeros(self.n_rows, dtype=torch.long)
        self.row_pos[order] = torch.arange(self.n_rows) - first_row[self.row_trip[order]]
        self.row_size = torch.bincount(self.row_of, minlength=self.n_rows)
        first_slot = torch.cat([torch.zeros(1, dtype=torch.long), torch.cumsum(self.row_size, 0)[:-1]])
        self.item_pos = torch.arange(len(self.row_of)) - first_slot[self.row_of]
        self.item_trip = self.row_trip[self.row_of]
        self.flat_slot = self.row_trip * self.Cpad + self.row_pos


@dataclass
class Batch:
    trips: np.ndarray               # bundle trip ids
    ix: RaggedIndex
    ctx: dict                       # features at every slot
    line_ctx: dict                  # the same features at every purchased line
    house: torch.Tensor             # [B] household of each trip
    line_item: torch.Tensor         # purchased lines: product, trip in batch, category
    line_trip: torch.Tensor
    line_cat: torch.Tensor
    observed: torch.Tensor          # [T] bool, slot was bought

    def bind(self, model, ctx=None):
        """Point the model at this batch (its households and a context, by default ctx)."""
        model.house, model.ctx = self.house, (self.ctx if ctx is None else ctx)


class PurchaseHistory:
    """Product-level purchase history strictly before each trip's day.

    Three features per (household, product, day), all from the household's earlier trips:
        ever    bought this product before
        last    this product was in the household's latest purchase of its category
        decay   exp(-days since last bought this product / 28)
    They are fixed data that shift utilities, so the normaliser stays exact.
    """

    STRIDE = 1024

    def __init__(self, data):
        J, C = int(data["n_item"]), int(data["n_cat"])
        line_trip = np.repeat(np.arange(len(data["trip_user"])), np.diff(data["line_ptr"]))
        user = data["trip_user"][line_trip].astype(np.int64)
        day = data["trip_day"][line_trip].astype(np.int64)
        if int(day.max()) >= self.STRIDE:
            raise RuntimeError("trip days exceed the purchase-history key stride")
        item = data["line_item"].astype(np.int64)
        cat = data["line_cat"].astype(np.int64)
        self.J, self.C = J, C
        self.item_keys = torch.from_numpy(np.unique((user * J + item) * self.STRIDE + day))
        self.cat_keys = torch.from_numpy(np.unique((user * C + cat) * self.STRIDE + day))

    @staticmethod
    def _previous(keys, group, day, stride):
        """Was there an event of this group before this day, and on which day."""
        idx = torch.searchsorted(keys, group * stride + day)
        prev = keys[(idx - 1).clamp(min=0)]
        return (idx > 0) & (torch.div(prev, stride, rounding_mode="floor") == group), prev % stride

    def features(self, item, cat, user, day):
        S = self.STRIDE
        item, cat, user, day = item.long(), cat.long(), user.long(), day.long()
        g_item = user * self.J + item
        seen, last_item_day = self._previous(self.item_keys, g_item, day, S)
        decay = torch.where(seen, torch.exp(-(day - last_item_day).double() / 28.0),
                            torch.zeros((), dtype=torch.float64))
        cat_seen, last_cat_day = self._previous(self.cat_keys, user * self.C + cat, day, S)
        q = g_item * S + last_cat_day
        hit = self.item_keys[torch.searchsorted(self.item_keys, q).clamp(max=len(self.item_keys) - 1)] == q
        return torch.stack([seen.double(), (cat_seen & hit).double(), decay], dim=-1)


class Batcher:
    """Builds a Batch for any set of trips.

    recency          add the sub-commodity recency features ``rec`` (needs state.npz)
    purchase_history add the product purchase-history features ``rec_item``
    """

    def __init__(self, data, features, *, recency=False, purchase_history=None):
        if recency and not features.include_recency:
            raise ValueError("recency requested but the features did not load it")
        self.D, self.F = data, features
        self.C = int(data["n_cat"])
        self.recency = bool(recency)
        self.history = purchase_history
        self.cat_of = torch.as_tensor(item_categories(data))

    def make(self, trips) -> Batch:
        D, C = self.D, self.C
        trips = np.asarray(trips)
        ptr, items = D["store_cat_ptr"], D["store_items"]
        slot_item, row_of, row_trip, row_cat = [], [], [], []
        for b, t in enumerate(trips):
            base = int(D["trip_store"][t]) * C
            for c in range(C):
                lo, hi = int(ptr[base + c]), int(ptr[base + c + 1])
                if hi > lo:
                    slot_item.append(items[lo:hi])
                    row_of.append(np.full(hi - lo, len(row_trip), np.int64))
                    row_trip.append(b)
                    row_cat.append(c)
        ix = RaggedIndex(np.concatenate(slot_item), np.concatenate(row_of),
                         np.array(row_trip, np.int64), np.array(row_cat, np.int64), len(trips))

        store = torch.as_tensor(D["trip_store"][trips], dtype=torch.long)
        day = torch.as_tensor(D["trip_day"][trips], dtype=torch.long)
        week = torch.as_tensor(D["trip_week"][trips], dtype=torch.long)
        user = torch.as_tensor(D["trip_user"][trips], dtype=torch.long)

        lines = [np.arange(D["line_ptr"][t], D["line_ptr"][t + 1]) for t in trips]
        line_item = torch.as_tensor(np.concatenate([D["line_item"][s] for s in lines]), dtype=torch.long)
        line_cat = torch.as_tensor(np.concatenate([D["line_cat"][s] for s in lines]), dtype=torch.long)
        line_trip = torch.as_tensor(np.concatenate([np.full(len(s), b, np.int64)
                                                    for b, s in enumerate(lines)]))

        ctx = self._context(ix.item, ix.item_trip, store, day, week, user)
        # The trip's mean price deviation over its whole assortment: the reference point that
        # splits a price into a common level and a relative part.  It must be the same number
        # for offered slots and purchased lines, so it is computed once from the assortment.
        mean_dlp = (torch.zeros(ix.B, dtype=torch.float64).index_add_(0, ix.item_trip, ctx["dlp"])
                    / torch.bincount(ix.item_trip, minlength=ix.B).double().clamp_min(1.0))
        ctx["dlp_bar"] = mean_dlp
        line_ctx = self._context(line_item, line_trip, store, day, week, user)
        line_ctx["dlp_bar"] = mean_dlp

        observed = torch.zeros(len(ix.item), dtype=torch.bool)
        key = ix.item_trip * int(D["n_item"]) + ix.item
        order = torch.argsort(key)
        line_key = line_trip * int(D["n_item"]) + line_item
        slot = order[torch.searchsorted(key[order], line_key).clamp(max=len(order) - 1)]
        if not bool((key[slot] == line_key).all()):
            raise RuntimeError("a purchased product is outside its trip's assortment")
        observed[slot] = True
        return Batch(trips, ix, ctx, line_ctx, user, line_item, line_trip, line_cat, observed)

    def _context(self, item, trip, store, day, week, user):
        """Features of products ``item`` in batch trips ``trip``."""
        st, dy, wk = store[trip], day[trip], week[trip]
        dlp, disp, mail = self.F.gather(item, st, dy, wk)
        ctx = dict(dlp=dlp.double(), disp=disp.double(), mail=mail.double(),
                   week=(wk - 1) % 52, store=st)          # week of year
        if self.F.availability:
            ctx["log_avail"] = self.F.log_availability(item, st, wk)
        if self.F.dsc is not None:
            ctx["dsc"] = self.F.discount(item, dy).double()
        if self.recency:
            ctx["rec"] = self.F.recency(item, user[trip], dy)
        if self.history is not None:
            ctx["rec_item"] = self.history.features(item, self.cat_of[item], user[trip], dy)
        return ctx


