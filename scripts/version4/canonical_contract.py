"""The dataset-neutral canonical basket input contract and its validator.

Any dataset enters the energy-basket pipeline by writing one canonical directory:

    transactions.parquet            one row per purchased product in a basket
    products.parquet                the catalogue
    store_week_prices.parquet       independent store-period price/sales feed
    promotions.parquet              store-period promotion flags (may be empty)
    shopping_opportunities.parquet  optional; household-store-period visits
    build_audit.json                adapter provenance and cohort policy

Dataset-specific choices (price basis, promotion feature, availability rule, affinity
partition sizes, catalogue metadata defaults) are parameters in a dataset configuration,
not code. See ``paper/CANONICAL_INPUT_CONTRACT.md``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

MAX_PERIODS = 127          # features.Features.WEEK_KEY_STRIDE - 1
MAX_DAYS = 1023            # features.Features.DAY_KEY_STRIDE - 1
SPLITS = ("train", "validation", "test")
PRICE_SOURCES = ("retail_aggregate", "purchase_aggregate")

SCHEMA: dict[str, dict[str, Any]] = {
    "transactions": {
        "required": {
            "basket_id": "integer; one shopping trip",
            "customer_id": "integer 0..N-1, contiguous",
            "store_id": "integer 0..S-1, contiguous",
            "period": f"integer week 1..{MAX_PERIODS}",
            "day": "integer day 0.., with period == day // 7 + 1",
            "product_id": "string/integer key into products.product_id",
            "item_id": "integer model index of the product",
            "category": "category label (must match products.category)",
            "quantity": "positive integer units",
            "unit_price": "positive paid price per unit",
            "split": "train | validation | test, chronological by period",
        },
    },
    "products": {
        "required": {
            "item_id": "integer 0..J-1, contiguous",
            "product_id": "unique product key",
            "category": "category label",
            "label": "readable product description",
        },
        "optional": {
            "subcategory": "sub-category label (defaults to category)",
            "brand": "brand label (defaults to label)",
            "manufacturer": "manufacturer (defaults to metadata_defaults.MANUFACTURER)",
            "department": "department (defaults to metadata_defaults.DEPARTMENT)",
        },
    },
    "store_week_prices": {
        "required": {
            "product_id": "product key",
            "store_id": "store index",
            "period": "week",
            "units": "units sold in the cell (> 0)",
            "revenue": "revenue in the cell (> 0)",
            "price_source": "retail_aggregate (independent feed) | purchase_aggregate (panel-derived)",
        },
    },
    "promotions": {
        "required": {
            "product_id": "product key", "store_id": "store index", "period": "week",
            "display": "bool", "advertised": "bool", "special_price": "bool",
        },
    },
    "shopping_opportunities": {
        "required": {"customer_id": "customer index", "store_id": "store index",
                     "period": "week"},
        "optional_table": True,
    },
}
BUILD_AUDIT_REQUIRED = ("source_sha256", "audit.cohort_policy.minimum_product_training_lines")


class CanonicalContractError(ValueError):
    """The canonical directory violates the declared input contract."""


def _fail(message: str) -> None:
    raise CanonicalContractError(message)


def _require_columns(name: str, frame: pd.DataFrame) -> None:
    missing = sorted(set(SCHEMA[name]["required"]).difference(frame.columns))
    if missing:
        _fail(f"{name}.parquet is missing required columns: {missing}")


def _contiguous(values: pd.Series, label: str) -> int:
    unique = np.sort(pd.unique(values))
    if len(unique) == 0 or not np.array_equal(unique, np.arange(len(unique))):
        _fail(f"{label} must be contiguous integers starting at 0")
    return int(len(unique))


def load_canonical_directory(directory: Path) -> dict[str, Any]:
    directory = Path(directory)
    tables: dict[str, Any] = {}
    for name, spec in SCHEMA.items():
        path = directory / f"{name}.parquet"
        if not path.is_file():
            if spec.get("optional_table"):
                tables[name] = None
                continue
            _fail(f"canonical directory lacks {path.name}")
        tables[name] = pd.read_parquet(path)
    audit_path = directory / "build_audit.json"
    if not audit_path.is_file():
        _fail("canonical directory lacks build_audit.json")
    tables["build_audit"] = json.loads(audit_path.read_text())
    return tables


def validate_canonical_tables(tables: dict[str, Any]) -> dict[str, Any]:
    """Check every contract invariant; return a summary of the validated dataset."""
    tx, products = tables["transactions"], tables["products"]
    prices, promotions = tables["store_week_prices"], tables["promotions"]
    for name in ("transactions", "products", "store_week_prices", "promotions"):
        _require_columns(name, tables[name])
    if tables.get("shopping_opportunities") is not None:
        _require_columns("shopping_opportunities", tables["shopping_opportunities"])
    if tx.empty or products.empty:
        _fail("transactions and products must be nonempty")
    required_tx = list(SCHEMA["transactions"]["required"])
    if tx[required_tx].isna().any().any():
        _fail("transactions contain nulls in required columns")

    # catalogue
    n_items = _contiguous(products.item_id, "products.item_id")
    if products.product_id.duplicated().any():
        _fail("products.product_id must be unique")
    product_item = products.set_index("product_id").item_id
    if not set(tx.product_id).issubset(product_item.index):
        _fail("transactions reference products absent from products.parquet")
    if not (tx.item_id.to_numpy() == product_item.loc[tx.product_id].to_numpy()).all():
        _fail("transactions.item_id disagrees with products.item_id")
    category = products.set_index("product_id").category
    if not (tx.category.astype(str).to_numpy()
            == category.loc[tx.product_id].astype(str).to_numpy()).all():
        _fail("transactions.category disagrees with products.category")

    # identifiers and time
    n_customers = _contiguous(tx.customer_id, "transactions.customer_id")
    n_stores = _contiguous(tx.store_id, "transactions.store_id")
    period = tx.period.to_numpy()
    if not np.issubdtype(period.dtype, np.integer) or period.min() < 1 or period.max() > MAX_PERIODS:
        _fail(f"transactions.period must be integers in 1..{MAX_PERIODS}")
    day = tx.day.to_numpy()
    if not np.issubdtype(day.dtype, np.integer) or day.min() < 0 or day.max() > MAX_DAYS:
        _fail(f"transactions.day must be integers in 0..{MAX_DAYS}")
    if not (period == day // 7 + 1).all():
        _fail("transactions.period must equal day // 7 + 1")

    # baskets
    if tx.duplicated(["basket_id", "item_id"]).any():
        _fail("a basket lists the same product twice; aggregate quantities first")
    per_basket = tx.groupby("basket_id").agg(
        c=("customer_id", "nunique"), s=("store_id", "nunique"), d=("day", "nunique"),
        p=("split", "nunique"))
    if bool((per_basket != 1).any().any()):
        _fail("a basket spans more than one customer, store, day or split")
    quantity = tx.quantity.to_numpy(dtype=np.float64)
    if (quantity <= 0).any() or (np.abs(quantity - np.rint(quantity)) > 1e-9).any():
        _fail("transactions.quantity must be positive integers")
    if (tx.unit_price <= 0).any():
        _fail("transactions.unit_price must be positive")

    # chronological splits and training support
    if set(tx.split.unique()) != set(SPLITS):
        _fail(f"transactions.split must contain exactly {SPLITS}")
    if tx[["period", "split"]].drop_duplicates().groupby("period").split.nunique().max() != 1:
        _fail("each period must belong to exactly one split")
    bounds = tx.groupby("split").period.agg(["min", "max"])
    if not (bounds.loc["train", "max"] < bounds.loc["validation", "min"]
            and bounds.loc["validation", "max"] < bounds.loc["test", "min"]):
        _fail("splits must be chronological: train < validation < test")
    training = tx[tx.split == "train"]
    untrained = sorted(set(products.item_id).difference(training.item_id))
    if untrained:
        _fail(f"{len(untrained)} products have no training purchases (e.g. item_id "
              f"{untrained[:5]}); drop them in the adapter")
    heldout = tx[tx.split != "train"]
    if not set(heldout.customer_id).issubset(training.customer_id):
        _fail("held-out baskets contain customers without training baskets")

    # store-period feeds
    for name, frame in (("store_week_prices", prices), ("promotions", promotions)):
        if frame.empty:
            continue
        if not set(frame.product_id).issubset(product_item.index):
            _fail(f"{name} references unknown products")
        if frame.store_id.min() < 0 or frame.store_id.max() >= n_stores:
            _fail(f"{name}.store_id must lie in 0..{n_stores - 1}")
        if frame.period.min() < 1 or frame.period.max() > int(period.max()):
            _fail(f"{name}.period must lie within the transaction periods")
    if not prices.empty:
        if not set(prices.price_source).issubset(PRICE_SOURCES):
            _fail(f"store_week_prices.price_source must be one of {PRICE_SOURCES}")
        if (prices.units <= 0).any() or (prices.revenue <= 0).any():
            _fail("store_week_prices units and revenue must be positive")

    audit = tables["build_audit"]
    if "source_sha256" not in audit:
        _fail("build_audit.json must record source_sha256")
    try:
        minimum = int(audit["audit"]["cohort_policy"]["minimum_product_training_lines"])
    except (KeyError, TypeError, ValueError):
        _fail("build_audit.json must record audit.cohort_policy.minimum_product_training_lines")
    return {
        "n_items": n_items, "n_customers": n_customers, "n_stores": n_stores,
        "n_periods": int(period.max()), "n_baskets": int(tx.basket_id.nunique()),
        "n_lines": int(len(tx)), "categories": int(products.category.nunique()),
        "retail_price_rows": int((prices.price_source == "retail_aggregate").sum())
        if not prices.empty else 0,
        "promotion_rows": int(len(promotions)),
        "minimum_product_training_lines": minimum,
        "split_periods": {split: [int(bounds.loc[split, "min"]), int(bounds.loc[split, "max"])]
                          for split in SPLITS},
    }


def validate_canonical_directory(directory: Path) -> dict[str, Any]:
    return validate_canonical_tables(load_canonical_directory(directory))
