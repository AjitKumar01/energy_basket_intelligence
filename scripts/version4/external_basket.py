"""Canonical multi-product basket contracts and the ERIM source adapter."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from external_choice_data import (
    read_erim_products,
    read_erim_purchase,
    read_erim_retail,
    read_erim_shopping,
)


BASKET_KEYS = ["market", "household", "store", "source_week", "dow", "trip"]
OPPORTUNITY_KEYS = ["market", "store", "household", "source_week"]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class BasketCategorySource:
    name: str
    purchase: Path
    shopping: tuple[Path, ...]
    products: Path
    retail: Path | None = None

    @property
    def files(self) -> tuple[Path, ...]:
        return (self.purchase, *self.shopping, self.products) + (
            () if self.retail is None else (self.retail,))


@dataclass(frozen=True)
class BasketBuildPolicy:
    train_fraction: float = .6
    validation_fraction: float = .2
    minimum_product_training_lines: int = 20
    minimum_customer_training_baskets: int = 10
    maximum_customer_training_baskets: int = 300

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1 or not 0 < self.validation_fraction < 1:
            raise ValueError("split fractions must be between zero and one")
        if self.train_fraction + self.validation_fraction >= 1:
            raise ValueError("test fraction must be positive")
        if min(self.minimum_product_training_lines,
               self.minimum_customer_training_baskets) < 1:
            raise ValueError("training support thresholds must be positive")
        if self.maximum_customer_training_baskets < self.minimum_customer_training_baskets:
            raise ValueError("maximum customer baskets is below the minimum")


@dataclass(frozen=True)
class CanonicalBasketDataset:
    transactions: pd.DataFrame
    products: pd.DataFrame
    opportunities: pd.DataFrame
    store_week_prices: pd.DataFrame
    promotions: pd.DataFrame
    audit: Mapping[str, Any]
    source_files: Sequence[Path]


class BasketDatasetAdapter(ABC):
    @abstractmethod
    def load(self) -> CanonicalBasketDataset:
        raise NotImplementedError


class CanonicalBasketValidator:
    """Format-independent invariants required by downstream basket models."""

    @staticmethod
    def validate(dataset: CanonicalBasketDataset) -> None:
        tx = dataset.transactions
        products = dataset.products
        required_tx = {
            "basket_id", "customer_id", "store_id", "period", "day", "product_id",
            "category", "quantity", "pre_coupon_value", "unit_price", "split",
        }
        missing = sorted(required_tx.difference(tx.columns))
        if missing:
            raise ValueError(f"canonical transactions are missing columns: {missing}")
        if tx.empty or products.empty:
            raise ValueError("canonical basket data cannot be empty")
        if tx[list(required_tx)].isna().any().any():
            raise ValueError("canonical transaction fields contain nulls")
        if bool((tx.quantity <= 0).any() or (tx.pre_coupon_value < 0).any()
                or (tx.unit_price <= 0).any()):
            raise ValueError(
                "canonical quantities/prices must be positive and values nonnegative")
        if products.product_id.duplicated().any():
            raise ValueError("canonical product identifiers are not unique")
        if not set(tx.product_id).issubset(set(products.product_id)):
            raise ValueError("a transaction references an unknown product")
        if tx.duplicated(["basket_id", "product_id"]).any():
            raise ValueError("canonical data contain duplicate basket-product rows")
        invariants = tx.groupby("basket_id").agg(
            customers=("customer_id", "nunique"), stores=("store_id", "nunique"),
            periods=("period", "nunique"), days=("day", "nunique"),
            splits=("split", "nunique"))
        if bool((invariants != 1).any().any()):
            raise ValueError("a basket crosses customer, store, period, day, or split")
        if set(tx.split.unique()) != {"train", "validation", "test"}:
            raise ValueError("canonical dataset requires nonempty train/validation/test splits")
        training = tx[tx.split == "train"]
        heldout = tx[tx.split != "train"]
        if not set(heldout.product_id).issubset(set(training.product_id)):
            raise ValueError("held-out transactions contain cold products")
        if not set(heldout.customer_id).issubset(set(training.customer_id)):
            raise ValueError("held-out transactions contain cold customers")
        period_split = tx[["period", "split"]].drop_duplicates()
        if period_split.groupby("period").split.nunique().max() != 1:
            raise ValueError("a period appears in more than one split")
        maxima = tx.groupby("split").period.agg(["min", "max"])
        if not maxima.loc["train", "max"] < maxima.loc["validation", "min"]:
            raise ValueError("validation does not follow training")
        if not maxima.loc["validation", "max"] < maxima.loc["test", "min"]:
            raise ValueError("test does not follow validation")


class ERIMMultiCategoryBasketAdapter(BasketDatasetAdapter):
    """Join ERIM modules only on their common household-store-week observation support."""

    def __init__(self, categories: Sequence[BasketCategorySource],
                 policy: BasketBuildPolicy = BasketBuildPolicy()):
        if len(categories) < 2:
            raise ValueError("a multi-category basket adapter needs at least two categories")
        names = [category.name for category in categories]
        if len(set(names)) != len(names):
            raise ValueError("ERIM category names must be unique")
        self.categories = tuple(categories)
        self.policy = policy

    def load(self) -> CanonicalBasketDataset:
        shopping_support: dict[str, set[tuple[int, int, int, int]]] = {}
        purchases: list[pd.DataFrame] = []
        product_rows: list[dict[str, Any]] = []
        retail_rows: list[pd.DataFrame] = []
        source_files: list[Path] = []
        raw_audit: dict[str, Any] = {}

        for source in self.categories:
            for path in source.files:
                if not Path(path).is_file():
                    raise FileNotFoundError(path)
                source_files.append(Path(path).resolve())
            shopping = pd.concat(
                [read_erim_shopping(path) for path in source.shopping], ignore_index=True)
            shopping = shopping.rename(columns={"week": "source_week"})
            support_frame = shopping[OPPORTUNITY_KEYS].drop_duplicates()
            shopping_support[source.name] = set(map(
                tuple, support_frame.itertuples(index=False, name=None)))

            purchase = read_erim_purchase(source.purchase).rename(
                columns={"week": "source_week"})
            purchase["category"] = source.name
            purchase["product_id"] = source.name + ":" + purchase.upc.map(
                lambda value: f"{int(value):013d}")
            purchases.append(purchase)

            label_map = read_erim_products(source.products)
            for upc in sorted(purchase.upc.unique()):
                product_rows.append({
                    "product_id": f"{source.name}:{int(upc):013d}",
                    "category": source.name,
                    "source_upc": int(upc),
                    "label": label_map.get(int(upc), f"{source.name} {int(upc):013d}"),
                    "label_found": int(upc) in label_map,
                })

            raw_audit[source.name] = {
                "purchase_rows": int(len(purchase)),
                "purchase_households": int(purchase.household.nunique()),
                "purchase_products": int(purchase.upc.nunique()),
                "purchase_week_range": [
                    int(purchase.source_week.min()), int(purchase.source_week.max())],
                "shopping_rows": int(len(shopping)),
                "shopping_support_cells": int(len(support_frame)),
                "retail_file_available": source.retail is not None,
                "product_labels_found": int(sum(
                    int(upc) in label_map for upc in purchase.upc.unique())),
            }

        common_support = set.intersection(*shopping_support.values())
        if not common_support:
            raise ValueError("ERIM categories have no common shopping observation support")
        opportunities = pd.DataFrame(
            sorted(common_support), columns=OPPORTUNITY_KEYS)
        purchase_all = pd.concat(purchases, ignore_index=True)
        supported = purchase_all.merge(
            opportunities, on=OPPORTUNITY_KEYS, how="inner", validate="many_to_one")
        if supported.empty:
            raise ValueError("no purchases remain on common ERIM observation support")

        weeks = np.sort(opportunities.source_week.unique())
        period_map = {int(week): index + 1 for index, week in enumerate(weeks)}
        allowed_weeks = set(map(int, weeks))
        for source in self.categories:
            if source.retail is None:
                continue
            retail = read_erim_retail(
                source.retail, allowed_weeks=allowed_weeks).rename(
                columns={"week": "source_week"})
            retail["category"] = source.name
            retail["product_id"] = source.name + ":" + retail.upc.map(
                lambda value: f"{int(value):013d}")
            retail_rows.append(retail)
        supported["period"] = supported.source_week.map(period_map).astype(np.int16)
        opportunities["period"] = opportunities.source_week.map(period_map).astype(np.int16)
        train_end, validation_end = self._split_boundaries(len(weeks))
        supported["split"] = np.where(
            supported.period <= train_end, "train",
            np.where(supported.period <= validation_end, "validation", "test"))

        grouped = (supported.groupby(
            BASKET_KEYS + ["period", "split", "category", "product_id"], as_index=False)
            .agg(quantity=("units", "sum"), pre_coupon_value=("expenditure", "sum"),
                 display=("end_display", self._any_yes),
                 advertised=("advertised", self._any_yes),
                 special_price=("special_price", self._any_yes)))
        grouped["unit_price"] = np.where(
            grouped.pre_coupon_value > 0,
            grouped.pre_coupon_value / grouped.quantity, np.nan)
        grouped["price_imputed"] = grouped.unit_price.isna()
        reference = supported[
            (supported.units > 0) & (supported.expenditure > 0)].copy()
        reference["reference_price"] = reference.expenditure / reference.units
        reference = (reference.groupby(
            ["category", "product_id", "store", "source_week"], as_index=False)
            .reference_price.median())
        grouped = grouped.merge(
            reference, on=["category", "product_id", "store", "source_week"],
            how="left", validate="many_to_one")
        missing_price = grouped.unit_price.isna()
        grouped.loc[missing_price, "unit_price"] = grouped.loc[
            missing_price, "reference_price"]
        if grouped.unit_price.isna().any():
            fallback = (supported[
                (supported.units > 0) & (supported.expenditure > 0)]
                .assign(reference_price=lambda frame: frame.expenditure / frame.units)
                .groupby(["category", "product_id"], as_index=False)
                .reference_price.median().rename(
                    columns={"reference_price": "product_reference_price"}))
            grouped = grouped.merge(
                fallback, on=["category", "product_id"], how="left",
                validate="many_to_one")
            missing_price = grouped.unit_price.isna()
            grouped.loc[missing_price, "unit_price"] = grouped.loc[
                missing_price, "product_reference_price"]
            grouped = grouped.drop(columns="product_reference_price")
        grouped = grouped.drop(columns="reference_price")
        if grouped.unit_price.isna().any() or bool((grouped.unit_price <= 0).any()):
            raise ValueError("nonpositive purchase values could not be assigned a valid reference price")
        training = grouped[grouped.split == "train"]
        product_support = training.groupby("product_id").size()
        kept_products = set(product_support[
            product_support >= self.policy.minimum_product_training_lines].index)
        training_baskets = training[BASKET_KEYS].drop_duplicates()
        customer_support = training_baskets.groupby("household").size()
        kept_customers = set(customer_support[
            (customer_support >= self.policy.minimum_customer_training_baskets)
            & (customer_support <= self.policy.maximum_customer_training_baskets)].index)
        before_cohort_rows = len(grouped)
        grouped = grouped[
            grouped.product_id.isin(kept_products)
            & grouped.household.isin(kept_customers)].copy()
        if grouped.empty:
            raise ValueError("the training-defined ERIM cohort is empty")

        stores = np.sort(grouped.store.unique())
        store_map = {int(value): index for index, value in enumerate(stores)}
        customers = np.sort(grouped.household.unique())
        customer_map = {int(value): index for index, value in enumerate(customers)}
        grouped["store_id"] = grouped.store.map(store_map).astype(np.int32)
        grouped["customer_id"] = grouped.household.map(customer_map).astype(np.int32)
        grouped["day"] = ((grouped.period - 1) * 7 + grouped.dow - 1).astype(np.int16)
        basket_table = (grouped[BASKET_KEYS].drop_duplicates()
                        .sort_values(BASKET_KEYS).reset_index(drop=True))
        basket_table["basket_id"] = np.arange(len(basket_table), dtype=np.int64)
        grouped = grouped.merge(
            basket_table, on=BASKET_KEYS, how="left", validate="many_to_one")
        grouped = grouped.rename(columns={"basket_id": "basket_id"})

        products = pd.DataFrame(product_rows).drop_duplicates("product_id")
        products = products[products.product_id.isin(kept_products)].copy()
        product_support_frame = product_support.rename("training_lines").reset_index()
        products = products.merge(product_support_frame, on="product_id", how="left")
        products = products.sort_values(["category", "product_id"]).reset_index(drop=True)
        products["item_id"] = np.arange(len(products), dtype=np.int32)
        product_map = products.set_index("product_id").item_id
        grouped["item_id"] = grouped.product_id.map(product_map).astype(np.int32)

        # Observed store-week price cells use the retail aggregate when available and
        # purchase aggregation only where the archive omitted retail data or a cell.
        purchase_prices = (supported.groupby(
            ["category", "product_id", "store", "source_week", "period"], as_index=False)
            .agg(units=("units", "sum"), revenue=("expenditure", "sum")))
        invalid_purchase_price_cells = int(
            ((purchase_prices.units <= 0) | (purchase_prices.revenue <= 0)).sum())
        purchase_prices = purchase_prices[
            (purchase_prices.units > 0) & (purchase_prices.revenue > 0)].copy()
        purchase_prices["average_price"] = purchase_prices.revenue / purchase_prices.units
        purchase_prices["price_source"] = "purchase_aggregate"
        if retail_rows:
            retail = pd.concat(retail_rows, ignore_index=True)
            retail = retail[retail.source_week.isin(period_map)].copy()
            retail["period"] = retail.source_week.map(period_map).astype(np.int16)
            retail = retail[[
                "category", "product_id", "store", "source_week", "period", "units",
                "revenue", "average_price", "end_display", "front_display",
                "aisle_display", "other_display", "advertised", "special_price"]]
            retail["price_source"] = "retail_aggregate"
            price_key = ["category", "product_id", "store", "source_week", "period"]
            retail_keys = set(map(tuple, retail[price_key].itertuples(index=False, name=None)))
            fallback_mask = np.fromiter(
                (tuple(row) not in retail_keys
                 for row in purchase_prices[price_key].itertuples(index=False, name=None)),
                dtype=bool, count=len(purchase_prices))
            store_prices = pd.concat(
                [retail, purchase_prices.loc[fallback_mask]], ignore_index=True, sort=False)
            display_columns = [
                "end_display", "front_display", "aisle_display", "other_display"]
            promotions = retail[[
                "product_id", "store", "source_week", "period", "advertised",
                "special_price", *display_columns]].copy()
            promotions["display"] = promotions[display_columns].eq("Y").any(axis=1)
            promotions["advertised"] = promotions.advertised.eq("Y")
            promotions["special_price"] = promotions.special_price.eq("Y")
            promotions["promotion_observed"] = True
            promotions = promotions[[
                "product_id", "store", "source_week", "period", "display",
                "advertised", "special_price", "promotion_observed"]]
        else:
            store_prices = purchase_prices
            promotions = pd.DataFrame(columns=[
                "product_id", "store", "source_week", "period", "display",
                "advertised", "special_price", "promotion_observed"])
        store_prices = store_prices[
            store_prices.product_id.isin(kept_products)
            & store_prices.store.isin(stores)].copy()
        store_prices["store_id"] = store_prices.store.map(store_map).astype(np.int32)
        promotions = promotions[
            promotions.product_id.isin(kept_products)
            & promotions.store.isin(stores)].copy()
        if len(promotions):
            promotions["store_id"] = promotions.store.map(store_map).astype(np.int32)

        tx_columns = [
            "basket_id", "customer_id", "store_id", "period", "source_week", "day",
            "dow", "trip", "product_id", "item_id", "category", "quantity",
            "pre_coupon_value", "unit_price", "display", "advertised",
            "special_price", "price_imputed", "split", "market", "household", "store",
        ]
        transactions = grouped[tx_columns].sort_values(
            ["basket_id", "item_id"]).reset_index(drop=True)
        opportunities = opportunities[
            opportunities.household.isin(kept_customers)
            & opportunities.store.isin(stores)].copy()
        opportunities["customer_id"] = opportunities.household.map(customer_map).astype(np.int32)
        opportunities["store_id"] = opportunities.store.map(store_map).astype(np.int32)

        basket_summary = transactions.groupby("basket_id").agg(
            products=("product_id", "nunique"), categories=("category", "nunique"))
        split_weeks = {
            "train": [int(weeks[0]), int(weeks[train_end - 1])],
            "validation": [int(weeks[train_end]), int(weeks[validation_end - 1])],
            "test": [int(weeks[validation_end]), int(weeks[-1])],
        }
        audit = {
            "categories": raw_audit,
            "category_count": len(self.categories),
            "raw_purchase_rows": int(len(purchase_all)),
            "common_shopping_support_cells": int(len(common_support)),
            "supported_purchase_rows_before_aggregation": int(len(supported)),
            "common_support_purchase_fraction": float(len(supported) / len(purchase_all)),
            "cohort_policy": {
                "minimum_product_training_lines": self.policy.minimum_product_training_lines,
                "minimum_customer_training_baskets": self.policy.minimum_customer_training_baskets,
                "maximum_customer_training_baskets": self.policy.maximum_customer_training_baskets,
            },
            "rows_before_cohort": int(before_cohort_rows),
            "nonpositive_value_rows_with_reference_price": int(
                grouped.price_imputed.sum()),
            "transaction_rows": int(len(transactions)),
            "baskets": int(transactions.basket_id.nunique()),
            "customers": int(transactions.customer_id.nunique()),
            "stores": int(transactions.store_id.nunique()),
            "products": int(len(products)),
            "periods": int(len(weeks)),
            "source_week_range": [int(weeks[0]), int(weeks[-1])],
            "split_source_weeks": split_weeks,
            "basket_product_count": {
                "mean": float(basket_summary.products.mean()),
                "median": float(basket_summary.products.median()),
                "maximum": int(basket_summary.products.max()),
            },
            "basket_category_count": {
                "mean": float(basket_summary.categories.mean()),
                "multi_category_fraction": float(basket_summary.categories.gt(1).mean()),
                "three_or_more_fraction": float(basket_summary.categories.ge(3).mean()),
                "maximum": int(basket_summary.categories.max()),
            },
            "price_sources": store_prices.price_source.value_counts().to_dict(),
            "invalid_purchase_aggregate_price_cells_excluded": (
                invalid_purchase_price_cells),
            "promotion_observation": (
                "retail aggregate only; frozen dinner has no archive retail file"),
            "outcome_scope": (
                "nonempty sub-baskets over eight tracked ERIM categories, conditional "
                "on a common-panel shopping week and at least one retained-category purchase"),
        }
        dataset = CanonicalBasketDataset(
            transactions, products, opportunities, store_prices, promotions,
            audit, tuple(source_files))
        CanonicalBasketValidator.validate(dataset)
        return dataset

    def _split_boundaries(self, periods: int) -> tuple[int, int]:
        if periods < 5:
            raise ValueError("at least five common periods are required")
        train_end = max(1, int(np.floor(self.policy.train_fraction * periods)))
        validation_end = max(
            train_end + 1,
            int(np.floor((self.policy.train_fraction + self.policy.validation_fraction)
                         * periods)))
        validation_end = min(validation_end, periods - 1)
        return train_end, validation_end

    @staticmethod
    def _any_yes(values: pd.Series) -> bool:
        return bool(values.astype(str).str.upper().eq("Y").any())
