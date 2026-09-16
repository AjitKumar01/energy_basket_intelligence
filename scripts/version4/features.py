"""
Conditioning features gathered at assortment slots for any model-data bundle.

The model scores every product a store carries, not just the purchased ones, so every
feature is gathered at every offered slot of a trip rather than joined at purchase lines.
Panels are read from ``$ENERGY_MODEL_DATA_ROOT/basket_input`` (the repository root when
unset), so Dunnhumby and canonical external bundles share one implementation.

    price        Delta log p_jst = Delta log p_jt + Delta^s_jsw: a dense (product x day)
                 deviation from each product's training-mean log price plus an optional
                 sparse (product, store, week) deviation
    promotion    display and mailer indicators, sparse over (product, store, week)
    seasonality  mu_j' delta_w
    store        zeta_j' xi_s
    recency      optional strictly-before purchase-history features from state.npz

Sparse keys pack (product, store, week) as (item * n_store + store) * 128 + week, and
recency keys pack (household-group, day) as group * 1024 + day, so a bundle must satisfy
week < 128 and day < 1024.  The constructor checks both rather than letting a collision
silently return another cell's value.
"""
import json
import math
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_ROOT = os.path.join(HERE, "..", "..")
MODEL_DATA_ROOT = os.path.abspath(os.environ.get("ENERGY_MODEL_DATA_ROOT", DEFAULT_ROOT))
BI = os.path.join(MODEL_DATA_ROOT, "basket_input")


def log(m):
    print(f"[fea] {m}", flush=True)


class Features:
    """Panels held once, gathered per batch."""

    WEEK_KEY_STRIDE = 128
    DAY_KEY_STRIDE = 1024

    def __init__(self, n_item, n_store, include_recency=True, include_store_price=None):
        self.S = n_store
        self.include_recency = bool(include_recency)
        if include_store_price is None:
            contract = os.environ.get(
                "ENERGY_PRICE_FEATURE_CONTRACT", "chain_and_store")
            if contract not in {"chain_and_store", "chain_product_week_only"}:
                raise ValueError(f"unknown price feature contract {contract!r}")
            include_store_price = contract == "chain_and_store"
        self.include_store_price = bool(include_store_price)
        self.dev = torch.from_numpy(
            np.load(os.path.join(BI, "log_price_dev.npy")).astype(np.float32))
        if self.dev.shape[0] != n_item:
            raise RuntimeError(f"price panel covers {self.dev.shape[0]} products, "
                               f"not the {n_item}-product catalogue")
        if self.dev.shape[1] > self.DAY_KEY_STRIDE:
            raise RuntimeError("price panel has more days than the recency key stride")
        log(f"price deviation panel {tuple(self.dev.shape)}, "
            f"|mean| {float(self.dev.abs().mean()):.4f}")

        # Key conventions are NOT guessed.  The promotion panel is built by
        # pipeline/23_promo_data.py as (item * n_stores + store) * 128 + WEEK_NO with the
        # RAW week, and store_price.npz stores item/store/week directly over weeks 1..102.
        # An earlier draft of this file guessed a different multiplier and a modulo; the
        # lookup would have silently returned zeros for almost every cell.
        if self.include_store_price:
            sp = np.load(os.path.join(BI, "store_price.npz"))
            if len(sp["week"]) and int(sp["week"].max()) >= self.WEEK_KEY_STRIDE:
                raise RuntimeError("store price weeks exceed the sparse key stride")
            key = ((sp["item"].astype(np.int64) * self.S
                    + sp["store"].astype(np.int64)) * self.WEEK_KEY_STRIDE
                   + sp["week"].astype(np.int64))
            o = np.argsort(key)
            self.sp_key = torch.from_numpy(key[o])
            self.sp_val = torch.from_numpy(sp["dev"][o].astype(np.float32))
            log(f"store price deviations: {len(key):,} cells "
                f"({len(key) / (n_item * n_store * 52):.2%} of the grid)")
        else:
            # ERIM store/product/week rows exist only when the product sold.  Using that
            # sparse deviation would make the feature itself outcome-selected.  The
            # direct-price experiment therefore uses the complete, carried-forward chain
            # product/week panel and explicitly disables this lookup.
            self.sp_key = torch.empty(0, dtype=torch.int64)
            self.sp_val = torch.empty(0, dtype=torch.float32)
            log("store price deviations disabled; using chain product/week price only")

        if self.include_recency:
            st = np.load(os.path.join(BI, "state.npz"))
            self.st_keys = torch.from_numpy(st["keys"])
            self.item_sub = torch.from_numpy(st["item_sub"].astype(np.int64))
            self.sub_gap = torch.from_numpy(st["sub_gap"])
            self.n_sub = int(self.item_sub.max()) + 1
            log(f"recency: {len(self.st_keys):,} purchase events over {self.n_sub} "
                f"sub-commodities; median gap {float(self.sub_gap.median()):.1f} days")
        else:
            self.st_keys = self.item_sub = self.sub_gap = None
            self.n_sub = 0
            log("recency disabled by the fitted feature contract")

        pr = np.load(os.path.join(BI, "promo.npz"))
        self.promo_week_min = int(pr["coverage_min_week"])
        self.promo_week_max = int(pr["coverage_max_week"])
        meta = json.load(open(os.path.join(BI, "meta.json")))
        self.promotion_enabled = bool(
            meta.get("promotion_contract", {}).get("enabled", True))
        if not 1 <= self.promo_week_min <= self.promo_week_max < self.WEEK_KEY_STRIDE:
            raise RuntimeError("promotion weeks must lie in 1..127 for the sparse key stride")
        required = meta.get("promotion_coverage_required")
        if required != [self.promo_week_min, self.promo_week_max]:
            raise RuntimeError(f"promotion coverage {self.promo_week_min}-"
                               f"{self.promo_week_max} does not match basket window "
                               f"{required}")
        o = np.argsort(pr["keys"])
        self.pk = torch.from_numpy(pr["keys"][o])
        self.pd_ = torch.from_numpy(pr["disp"][o].astype(np.float32))
        self.pm = torch.from_numpy(pr["mail"][o].astype(np.float32))
        display_rate = float(self.pd_.mean()) if len(self.pd_) else 0.0
        mail_rate = float(self.pm.mean()) if len(self.pm) else 0.0
        log(f"promotion cells: {len(self.pk):,}, coverage weeks "
            f"{self.promo_week_min}-{self.promo_week_max}; "
            f"display rate {display_rate:.3f}, mailer {mail_rate:.3f}; "
            f"enabled={self.promotion_enabled}")

    def recency(self, item, user, day):
        """The four recency functions of the specification, at every assortment slot.

        Purchase events are stored as sorted keys (user * n_sub + sub) * 1024 + DAY, so a
        strictly-before lookup is one searchsorted.  The previous key belongs to the same
        (household, sub-commodity) group only if its group field matches, which is what
        separates "no earlier purchase" from "the previous event is someone else's".

        This is the block version 2 measures as its largest single ablation, and it was the
        largest thing missing here.
        """
        if not self.include_recency:
            raise RuntimeError("recency was not loaded for this feature contract")
        sub = self.item_sub[item]
        group = user.to(torch.int64) * self.n_sub + sub
        key = group * self.DAY_KEY_STRIDE + day
        idx = torch.searchsorted(self.st_keys, key)
        prev = (idx - 1).clamp(0, len(self.st_keys) - 1)
        pk = self.st_keys[prev]
        same = (idx > 0) & (torch.div(pk, self.DAY_KEY_STRIDE, rounding_mode="floor") == group)
        since = torch.where(same, (day - pk % self.DAY_KEY_STRIDE).double(),
                            torch.zeros(1, dtype=torch.float64))
        gap = self.sub_gap[sub].double().clamp_min(1.0)
        z = torch.zeros(1, dtype=torch.float64)
        return torch.stack([
            (~same).double(),
            torch.where(same, torch.exp(-since / 7.0), z),
            torch.where(same, torch.exp(-since / gap), z),
            torch.where(same, torch.log1p(since) / math.log(100.0), z)], dim=-1)

    @staticmethod
    def _lookup(keys, vals, q):
        """Sparse gather: value where the key exists, zero where it does not."""
        if len(keys) == 0:
            return torch.zeros_like(q, dtype=vals.dtype)
        i = torch.searchsorted(keys, q)
        i = i.clamp(max=len(keys) - 1)
        hit = keys[i] == q
        return torch.where(hit, vals[i], torch.zeros((), dtype=vals.dtype))

    def gather(self, item, store, day, week):
        """Features at every assortment slot.  item/store/day/week are [T] longs."""
        q = (item * self.S + store) * self.WEEK_KEY_STRIDE + week          # one convention for both panels
        dlp = self.dev[item, day] + self._lookup(self.sp_key, self.sp_val, q)
        disp = self._lookup(self.pk, self.pd_, q)
        mail = self._lookup(self.pk, self.pm, q)
        return dlp, disp, mail

