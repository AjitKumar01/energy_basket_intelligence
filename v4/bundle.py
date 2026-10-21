"""The model-data bundle: trips, baskets and each store's assortment as flat arrays.

A bundle is a directory with ``basket_input/`` (and ``data/``) written by a dataset adapter.
The files this module reads:

    basket_input/meta.json               n_items, n_stores, n_users, ...
    basket_input/baskets.parquet         one row per purchased product line:
                                         BASKET_ID, user_id, store_id, DAY, WEEK_NO,
                                         item_id, units, split (train/validation/test)
    basket_input/items_affinity.parquet  the category of every product (item_id, cat_id),
                                         built from training co-purchases by
                                         scripts/build_affinity_partition.py

Every store is taken to offer the whole training catalogue (transaction logs record sales,
not stock, so there is no honest narrower support).  The products a store offers are kept
in one flat array grouped by (store, category); a trip's assortment is a contiguous slice.

The arrays are built once and cached in ``basket_input/v3_index_affinity.npz`` (the file
name is kept so existing bundles and their fingerprints stay valid).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

INDEX_FILE = "v3_index_affinity.npz"
SPLITS = {"train": 0, "validation": 1, "test": 2}


def data_root(root: str | Path | None = None) -> Path:
    """The bundle directory: the argument, else $ENERGY_MODEL_DATA_ROOT."""
    if root is None:
        root = os.environ.get("ENERGY_MODEL_DATA_ROOT")
        if not root:
            raise RuntimeError("no model-data bundle given; pass --data-root or set "
                               "ENERGY_MODEL_DATA_ROOT")
    return Path(root).expanduser().resolve()


def load_bundle(root: str | Path | None = None) -> dict[str, np.ndarray]:
    """Trips, purchase lines and assortments of a bundle (built and cached on first use).

    Returned arrays (T = trips, L = purchase lines, S = stores, C = categories):

        n_item, n_user, n_store, n_cat    sizes
        store_cat_ptr [S*C+1]             slice of store_items for each (store, category)
        store_items                       product ids grouped by (store, category)
        item_slot [S, J]                  position of a product inside its block
        trip_user/store/day/week/split    per trip; split is 0/1/2 = train/validation/test
        trip_nlines [T]                   basket size
        line_ptr [T+1]                    purchase lines of trip t are line_ptr[t]:line_ptr[t+1]
        line_item, line_cat, line_slot, line_units   per purchase line
    """
    folder = data_root(root) / "basket_input"
    cache = folder / INDEX_FILE
    if cache.exists():
        with np.load(cache, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}
    data = _build_index(folder)
    np.savez_compressed(cache, **data)
    print(f"[bundle] wrote {cache}", flush=True)
    return data


def _build_index(folder: Path) -> dict[str, np.ndarray]:
    meta = json.loads((folder / "meta.json").read_text())
    J, S = meta["n_items"], meta["n_stores"]
    baskets = pd.read_parquet(folder / "baskets.parquet")
    partition = pd.read_parquet(folder / "items_affinity.parquet")[["item_id", "cat_id"]]
    C = int(partition.cat_id.max()) + 1
    baskets = baskets.merge(partition, on="item_id", how="left")

    if baskets[baskets.split == "train"].item_id.nunique() != J:
        raise SystemExit("every catalogue product must appear in training")
    # Declared support: every store offers the whole catalogue, so no basket can fall outside.
    offer = pd.DataFrame({"store_id": np.repeat(np.arange(S, dtype=np.int32), J),
                          "item_id": np.tile(np.arange(J, dtype=np.int32), S)})
    offer = offer.merge(partition, on="item_id", how="left").sort_values(
        ["store_id", "cat_id", "item_id"], kind="mergesort")

    store_items = offer.item_id.to_numpy(np.int32)
    key = offer.store_id.to_numpy(np.int64) * C + offer.cat_id.to_numpy(np.int64)
    ptr = np.zeros(S * C + 1, np.int64)
    np.cumsum(np.bincount(key, minlength=S * C), out=ptr[1:])
    item_slot = np.full((S, J), -1, np.int32)
    item_slot[key // C, store_items] = np.arange(len(offer)) - ptr[key]

    trips = (baskets.groupby("BASKET_ID")
             .agg(user=("user_id", "first"), store=("store_id", "first"),
                  day=("DAY", "first"), week=("WEEK_NO", "first"),
                  split=("split", "first"), n=("item_id", "size"))
             .reset_index().sort_values("BASKET_ID", kind="mergesort"))
    trip_of = {v: i for i, v in enumerate(trips.BASKET_ID.to_numpy())}
    lines = baskets.assign(trip=baskets.BASKET_ID.map(trip_of)).sort_values(
        ["trip", "cat_id", "item_id"], kind="mergesort")
    line_trip = lines.trip.to_numpy(np.int32)
    line_item = lines.item_id.to_numpy(np.int32)
    line_slot = item_slot[lines.store_id.to_numpy(np.int64), line_item]
    if (line_slot < 0).any():
        raise RuntimeError("a purchased product is missing from its store's assortment")
    line_ptr = np.zeros(len(trips) + 1, np.int64)
    np.cumsum(np.bincount(line_trip, minlength=len(trips)), out=line_ptr[1:])

    return dict(store_cat_ptr=ptr, store_items=store_items, item_slot=item_slot,
                n_store=np.int64(S), n_cat=np.int64(C), n_item=np.int64(J),
                n_user=np.int64(meta["n_users"]),
                trip_user=trips.user.to_numpy(np.int32),
                trip_store=trips.store.to_numpy(np.int32),
                trip_day=trips.day.to_numpy(np.int32),
                trip_week=trips.week.to_numpy(np.int32),
                trip_split=trips.split.map(SPLITS).to_numpy(np.int8),
                trip_nlines=trips.n.to_numpy(np.int32),
                line_ptr=line_ptr, line_item=line_item,
                line_cat=lines.cat_id.to_numpy(np.int32),
                line_slot=line_slot.astype(np.int32),
                line_units=lines.units.to_numpy(np.int16))


def supported_trips(data, split: int, nmax: int) -> np.ndarray:
    """Trips of one split whose basket size lies in the model's support 1..nmax."""
    n = data["trip_nlines"]
    return np.flatnonzero((data["trip_split"] == int(split)) & (n >= 1) & (n <= int(nmax)))


def item_categories(data) -> np.ndarray:
    """Category of every product, from the purchase lines."""
    cat = np.zeros(int(data["n_item"]), dtype=np.int64)
    cat[data["line_item"]] = data["line_cat"]
    return cat


# ---- data fingerprint -------------------------------------------------------------------
# A bundle carries basket_input/model_data_fingerprint.json: hashes of its model-facing
# files and one digest over them.  Checkpoints record the digest, so a checkpoint can only
# be used with the data it was fitted on.

def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _digest(body: dict) -> str:
    text = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


FINGERPRINT_FILES = {
    "preprocessing_manifest": "basket_input/preprocessing_manifest.json",
    "affinity_manifest": "basket_input/affinity_manifest.json",
    "affinity_partition": "basket_input/items_affinity.parquet",
    "basket_meta": "basket_input/meta.json",
    "base_build_meta": "data/build_meta.json",
    "ragged_index": "basket_input/" + INDEX_FILE,
    # optional panels, included when the fingerprint recorded them
    "observed_promotion_panel": "basket_input/promo_observed.npz",
    "price_evidence_panel": "data/price_week.parquet",
    "availability_panel": "basket_input/availability.npz",
}


def data_fingerprint(root: str | Path | None = None) -> str:
    """Verify the bundle's fingerprint against its files and return the digest."""
    root = data_root(root)
    recorded = json.loads((root / "basket_input/model_data_fingerprint.json").read_text())
    digest = recorded.pop("fingerprint_sha256")
    if _digest(recorded) != digest:
        raise ValueError("model_data_fingerprint.json fails its own digest")
    for name, expected in recorded["files"].items():
        if file_sha256(root / FINGERPRINT_FILES[name]) != expected:
            raise ValueError(f"bundle file {FINGERPRINT_FILES[name]} differs from its fingerprint")
    return digest


def require_fingerprint(recorded: str | None, root: str | Path | None = None) -> str:
    """Stop unless a checkpoint's recorded fingerprint matches the bundle."""
    current = data_fingerprint(root)
    if recorded != current:
        raise ValueError("the checkpoint was fitted on different data than this bundle")
    return current
