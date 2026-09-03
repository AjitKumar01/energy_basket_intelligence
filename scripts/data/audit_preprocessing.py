"""Fail-closed audit for the Version-4 modelling dataset.

This script independently recomputes the cohort, split and price-panel invariants from
``data/tx.parquet``.  It does not repair anything: one mismatch exits non-zero before a
model can be initialized.  On success it writes an immutable, hash-addressed manifest.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
BI = Path(os.environ.get("NF_BASKET_INPUT", ROOT / "basket_input"))
RAW = Path(os.environ.get(
    "NF_RAW_DIR",
    ROOT.parent / "dunnhumby_The-Complete-Journey" /
    "dunnhumby_The-Complete-Journey CSV",
))

EXPECTED_RAW_SHA256 = {
    "transaction_data.csv": "3a685c0729cef664d634486189f774518b84f53cde7cbf701a5963238692b476",
    "product.csv": "7ecbcec41e0f1e5a51b43a359965cc50dc5586a2b900a378b64032750fedc949",
    "causal_data.csv": "60ed7021fefc209c0caf36fcf95d1e93693775acfdac01fc9ee38273da88937a",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require(ok: bool, message: str) -> None:
    if not ok:
        raise SystemExit(f"[data-audit] FAIL: {message}")


def main() -> None:
    meta = json.loads((BI / "meta.json").read_text())
    items = pd.read_parquet(BI / "items.parquet")
    baskets = pd.read_parquet(BI / "baskets.parquet")
    tx = pd.read_parquet(
        DATA / "tx.parquet",
        columns=["household_key", "DAY", "WEEK_NO", "PRODUCT_ID", "BASKET_ID",
                 "QUANTITY", "STORE_ID"],
    )

    first = int(meta["analysis_first_week"])
    last = int(meta["analysis_last_week"])
    val_from = int(meta["val_from_week"])
    test_from = int(meta["test_from_week"])
    n_items = int(meta["n_items_requested"])
    require(meta["cohort_policy"] ==
            "top_n_products_by_training_lines_then_training_trip_households",
            "unrecognized cohort policy")
    require(first < val_from < test_from <= last, "invalid time ordering")
    require(len(items) == n_items == 5455, "catalogue is not exactly 5,455 products")
    require(np.array_equal(items.item_id.to_numpy(), np.arange(n_items)),
            "item_id is not contiguous")

    # Independently reconstruct the training-only catalogue and household cohort.
    window = tx[(tx.WEEK_NO >= first) & (tx.WEEK_NO <= last)].copy()
    freq = (window[window.WEEK_NO < val_from].groupby("PRODUCT_ID").size()
            .rename("n_train").reset_index()
            .sort_values(["n_train", "PRODUCT_ID"], ascending=[False, True],
                         kind="mergesort"))
    selected = freq.head(n_items)
    expected_products = np.sort(selected.PRODUCT_ID.to_numpy())
    require(np.array_equal(expected_products, np.sort(items.PRODUCT_ID.to_numpy())),
            "items.parquet is not the training-only top-5,455 catalogue")
    boundary = int(selected.n_train.min())
    require(boundary == int(meta["item_training_line_boundary"]),
            "recorded product-frequency boundary is wrong")

    window = window[window.PRODUCT_ID.isin(set(expected_products))]
    train_days = window[window.WEEK_NO < val_from].groupby("household_key").DAY.nunique()
    expected_hh = train_days[(train_days >= int(meta["min_trips"])) &
                             (train_days <= int(meta["max_trips"]))].index
    window = window[window.household_key.isin(set(expected_hh))].copy()
    actual_hh = baskets.user_id.nunique()
    require(actual_hh == len(expected_hh) == int(meta["n_users"]),
            "household cohort differs from the training-period trip rule")
    retained_item_train = (window[window.WEEK_NO < val_from].groupby("PRODUCT_ID").size()
                           .reindex(items.PRODUCT_ID).to_numpy())
    require(np.array_equal(retained_item_train, items.n_train_lines.to_numpy()),
            "stored per-item training counts are wrong")

    # No outcome line may disappear after cohort definition.  Compare the complete set
    # of basket-product keys and the clipped unit counts, not just aggregate row counts.
    actual = baskets.merge(items[["item_id", "PRODUCT_ID"]], on="item_id", validate="many_to_one")
    expected = (window.groupby(["BASKET_ID", "PRODUCT_ID"], as_index=False)
                .QUANTITY.sum().rename(columns={"QUANTITY": "units_expected"}))
    expected.units_expected = expected.units_expected.clip(1, int(meta["max_units"])).astype(np.int16)
    joined = expected.merge(actual[["BASKET_ID", "PRODUCT_ID", "units"]],
                            on=["BASKET_ID", "PRODUCT_ID"], how="outer",
                            indicator=True, validate="one_to_one")
    require((joined._merge == "both").all(), "basket-product outcomes were added or deleted")
    require(np.array_equal(joined.units_expected.to_numpy(), joined.units.to_numpy()),
            "basket unit aggregation/clipping is inconsistent")
    invariant = baskets.groupby("BASKET_ID").agg(
        user=("user_id", "nunique"), store=("store_id", "nunique"),
        day=("DAY", "nunique"), week=("WEEK_NO", "nunique"), split=("split", "nunique"))
    require((invariant == 1).all().all(), "a basket crosses user/store/day/week/split")
    require(not baskets.duplicated(["BASKET_ID", "item_id"]).any(),
            "duplicate basket-product rows")

    split_expected = np.where(baskets.WEEK_NO >= test_from, "test",
                              np.where(baskets.WEEK_NO >= val_from, "validation", "train"))
    require(np.array_equal(split_expected, baskets.split.to_numpy()), "split labels are wrong")
    require(int(baskets.WEEK_NO.min()) >= first and int(baskets.WEEK_NO.max()) <= last,
            "basket lies outside promotion coverage")
    train = baskets[baskets.split == "train"]
    require(train.item_id.nunique() == len(items), "some catalogue item has no training outcome")
    require(train.user_id.nunique() == actual_hh, "some household has no training outcome")

    # Reconstruct the modal weekly price broadcast and its training-only centring.
    pw = pd.read_parquet(DATA / "price_week.parquet",
                         columns=["PRODUCT_ID", "WEEK_NO", "price"])
    iid = items.set_index("PRODUCT_ID").item_id
    pw = pw[pw.PRODUCT_ID.isin(set(expected_products))].copy()
    require(not pw.duplicated(["PRODUCT_ID", "WEEK_NO"]).any(),
            "chain modal price keys are not unique")
    pw["item_id"] = pw.PRODUCT_ID.map(iid).astype(np.int32)
    wg = np.full((n_items, 128), np.nan, dtype=np.float64)
    wg[pw.item_id.to_numpy(), pw.WEEK_NO.to_numpy()] = pw.price.to_numpy()
    wg = pd.DataFrame(wg).ffill(axis=1).bfill(axis=1).to_numpy()
    day_map = tx[["DAY", "WEEK_NO"]].drop_duplicates()
    require(day_map.groupby("DAY").WEEK_NO.nunique().max() == 1,
            "DAY does not map uniquely to WEEK_NO")
    D = int(meta["n_days"])
    dw = np.full(D, -1, dtype=np.int16)
    dw[day_map.DAY.to_numpy()] = day_map.WEEK_NO.to_numpy()
    dw = pd.Series(dw).replace(-1, np.nan).ffill().bfill().to_numpy(dtype=np.int16)
    price = wg[:, dw].astype(np.float32)
    lp = np.log(np.clip(price, 1e-3, None)).astype(np.float32)
    expected_dev = lp - lp[:, (dw >= first) & (dw < val_from)].mean(axis=1, keepdims=True)
    require(np.array_equal(lp, np.load(BI / "log_price.npy")),
            "log_price.npy is not the stage-01 modal weekly panel")
    require(np.allclose(expected_dev, np.load(BI / "log_price_dev.npy"), atol=2e-7),
            "price centring is not training-only")

    # Store deviations must be modal store-week price minus the same modal chain-week
    # price.  This catches the former second, inconsistent median reconstruction.
    basket_store = baskets[["BASKET_ID", "store_id"]].drop_duplicates()
    raw_basket_store = tx[["BASKET_ID", "STORE_ID"]].drop_duplicates()
    sm = basket_store.merge(raw_basket_store, on="BASKET_ID", validate="one_to_one")
    require(sm.groupby("STORE_ID").store_id.nunique().max() == 1,
            "raw store to model store map is not unique")
    smap = sm.groupby("STORE_ID").store_id.first()
    psw = pd.read_parquet(DATA / "price_store_week.parquet",
                          columns=["PRODUCT_ID", "STORE_ID", "WEEK_NO", "price"])
    psw = psw[(psw.PRODUCT_ID.isin(set(expected_products))) &
              (psw.STORE_ID.isin(set(smap.index))) &
              (psw.WEEK_NO >= first) & (psw.WEEK_NO <= last)].copy()
    psw["item_id"] = psw.PRODUCT_ID.map(iid).astype(np.int32)
    psw["store_id"] = psw.STORE_ID.map(smap).astype(np.int32)
    cp = pw[["PRODUCT_ID", "WEEK_NO", "price"]].rename(columns={"price": "p_chain"})
    expected_sp = psw.merge(cp, on=["PRODUCT_ID", "WEEK_NO"], validate="many_to_one")
    expected_sp["dev"] = (np.log(expected_sp.price.clip(lower=1e-3)) -
                           np.log(expected_sp.p_chain.clip(lower=1e-3)))
    expected_sp = expected_sp[expected_sp.dev.abs() > 1e-6]
    expected_key = ((expected_sp.item_id.to_numpy(np.int64) * int(meta["n_stores"]) +
                     expected_sp.store_id.to_numpy(np.int64)) * 128 +
                    expected_sp.WEEK_NO.to_numpy(np.int64))
    sp = np.load(BI / "store_price.npz")
    actual_key = ((sp["item"].astype(np.int64) * int(meta["n_stores"]) +
                   sp["store"].astype(np.int64)) * 128 + sp["week"].astype(np.int64))
    eo, ao = np.argsort(expected_key), np.argsort(actual_key)
    require(np.array_equal(expected_key[eo], actual_key[ao]),
            "store modal-price sparse keys differ")
    require(np.allclose(expected_sp.dev.to_numpy()[eo], sp["dev"][ao], atol=2e-7),
            "store modal-price deviations differ")

    promo = np.load(BI / "promo.npz")
    coverage = [int(promo["coverage_min_week"]), int(promo["coverage_max_week"])]
    require(coverage == [first, last] == meta["promotion_coverage_required"],
            "promotion coverage and basket window differ")
    require((np.diff(promo["keys"]) > 0).all(), "promotion sparse keys are not unique/sorted")

    raw_hashes = {}
    for name, expected_hash in EXPECTED_RAW_SHA256.items():
        path = RAW / name
        require(path.exists(), f"raw source is missing: {path}")
        raw_hashes[name] = sha256(path)
        require(raw_hashes[name] == expected_hash, f"unexpected raw digest for {name}")

    derived_names = ["baskets.parquet", "items.parquet", "log_price.npy",
                     "log_price_dev.npy", "store_price.npz", "state.npz", "promo.npz"]
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "raw_sha256": raw_hashes,
        "derived_sha256": {name: sha256(BI / name) for name in derived_names},
        "cohort": {
            "n_items": len(items), "n_users": actual_hh,
            "n_basket_product_rows": len(baskets),
            "n_baskets": int(baskets.BASKET_ID.nunique()),
            "weeks": [first, last], "validation_from": val_from,
            "test_from": test_from, "minimum_selected_item_training_lines": boundary,
        },
        "checks": [
            "training-only deterministic item catalogue",
            "training-only household eligibility",
            "no post-cohort outcome deletion",
            "unique checkout invariants",
            "time split and promotion coverage",
            "modal weekly price reconstruction",
            "modal store-week price reconstruction",
            "training-only price centring",
            "raw source digests",
        ],
    }
    (BI / "preprocessing_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[data-audit] PASS: {len(items):,} products, {actual_hh:,} households, "
          f"{len(baskets):,} basket-product rows; manifest written", flush=True)


if __name__ == "__main__":
    main()
