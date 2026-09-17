"""Convert a validated canonical basket dataset to the energy model's input contract."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from canonical_contract import validate_canonical_tables

DEFAULT_METADATA = {"MANUFACTURER": "UNKNOWN", "DEPARTMENT": "UNKNOWN"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class ModelInputBuildResult:
    root: Path
    basket_input: Path
    data: Path
    manifest: dict[str, Any]


class CanonicalBasketModelInputBuilder:
    """Dataset-neutral bridge; all source-specific work happens before this class."""

    PRICE_SOURCES = ("retail_aggregate", "purchase_aggregate")
    AVAILABILITY_RULES = ("disabled", "retail_first_sale")
    AVAILABILITY_FLOOR_BOUNDS = (1e-4, 1.0)

    def __init__(self, canonical_directory: Path, output_root: Path,
                 *, price_basis: str, promotion_feature_name: str = "advertised",
                 model_price_sources: tuple[str, ...] = ("retail_aggregate",),
                 availability: str = "disabled",
                 availability_left_censor_periods: int = 13,
                 dataset_name: str = "canonical_external_basket",
                 metadata_defaults: dict[str, str] | None = None,
                 promotion_coverage_note: str = "declared by the canonical adapter",
                 product_metadata: Path | None = None):
        self.canonical_directory = canonical_directory.resolve()
        self.output_root = output_root.resolve()
        self.price_basis = str(price_basis)
        self.promotion_feature_name = promotion_feature_name
        self.model_price_sources = tuple(dict.fromkeys(model_price_sources))
        if not self.model_price_sources or not set(self.model_price_sources).issubset(
                self.PRICE_SOURCES):
            raise ValueError(
                f"model price sources must be a nonempty subset of {self.PRICE_SOURCES}")
        if not self.price_basis:
            raise ValueError("price_basis must explicitly describe the modeled price")
        if promotion_feature_name not in {"advertised", "special_price", "disabled"}:
            raise ValueError(
                "promotion feature must be advertised, special_price, or disabled")
        if availability not in self.AVAILABILITY_RULES:
            raise ValueError(f"availability must be one of {self.AVAILABILITY_RULES}")
        if availability_left_censor_periods < 0:
            raise ValueError("availability left-censor window must be nonnegative")
        self.availability = availability
        self.availability_left_censor_periods = int(availability_left_censor_periods)
        self.dataset_name = str(dataset_name)
        unknown = set(metadata_defaults or {}).difference(DEFAULT_METADATA)
        if unknown:
            raise ValueError(f"unknown metadata defaults: {sorted(unknown)}")
        self.metadata_defaults = {**DEFAULT_METADATA, **(metadata_defaults or {})}
        self.promotion_coverage_note = str(promotion_coverage_note)
        self.product_metadata = None if product_metadata is None else Path(product_metadata).resolve()

    def build(self) -> ModelInputBuildResult:
        canonical = self.canonical_directory
        required = {
            name: canonical / f"{name}.parquet"
            for name in ("transactions", "products", "store_week_prices", "promotions")
        }
        if (canonical / "shopping_opportunities.parquet").is_file():
            required["shopping_opportunities"] = canonical / "shopping_opportunities.parquet"
        required["build_audit"] = canonical / "build_audit.json"
        missing = [str(path) for path in required.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError("canonical basket bundle is incomplete: " + ", ".join(missing))
        tx = pd.read_parquet(required["transactions"])
        products = pd.read_parquet(required["products"]).sort_values("item_id")
        metadata_audit = None
        if self.product_metadata is not None:
            products, metadata_audit = self._apply_product_metadata(products, self.product_metadata)
        prices = pd.read_parquet(required["store_week_prices"])
        promotions = pd.read_parquet(required["promotions"])
        source_audit = json.loads(required["build_audit"].read_text())
        self.contract_summary = validate_canonical_tables({
            "transactions": tx, "products": products, "store_week_prices": prices,
            "promotions": promotions, "build_audit": source_audit,
            "shopping_opportunities": (pd.read_parquet(required["shopping_opportunities"])
                                       if "shopping_opportunities" in required else None)})
        self._validate_ids(tx, products)

        basket_input = self.output_root / "basket_input"
        data_directory = self.output_root / "data"
        basket_input.mkdir(parents=True, exist_ok=True)
        data_directory.mkdir(parents=True, exist_ok=True)
        items = self._items(products, tx, self.metadata_defaults)
        baskets = self._baskets(tx, items)
        n_items = len(items)
        n_stores = int(baskets.store_id.max()) + 1
        n_users = int(baskets.user_id.max()) + 1
        n_periods = int(baskets.WEEK_NO.max())
        if n_periods >= 128:
            raise ValueError("model feature keys support at most 127 periods")
        if not (baskets.WEEK_NO == baskets.DAY // 7 + 1).all():
            raise ValueError("canonical day and period disagree: period must be day // 7 + 1")
        n_days = n_periods * 7
        train_period_max = int(baskets.loc[baskets.split == "train", "WEEK_NO"].max())

        items_path = basket_input / "items.parquet"
        baskets_path = basket_input / "baskets.parquet"
        items.to_parquet(items_path, index=False)
        baskets.to_parquet(baskets_path, index=False)
        price_audit = self._write_prices(
            prices, items, n_stores, n_periods, n_days, train_period_max, basket_input)
        promotion_audit = self._write_promotions(
            promotions, items, n_stores, n_periods, basket_input)
        state_audit = self._write_state(
            baskets, items, n_users, n_days, basket_input)
        availability_audit = self._write_availability(
            prices, baskets, items, n_stores, n_periods, train_period_max, basket_input)

        val_from = int(baskets.loc[baskets.split == "validation", "WEEK_NO"].min())
        test_from = int(baskets.loc[baskets.split == "test", "WEEK_NO"].min())
        meta = {
            "schema_version": 1,
            "dataset": "canonical_external_basket",
            "dataset_name": self.dataset_name,
            **({"product_metadata": metadata_audit} if metadata_audit is not None else {}),
            "n_users": n_users,
            "n_items": n_items,
            "n_subs": int(items.sub_id.max()) + 1,
            "n_days": n_days,
            "n_baskets": int(baskets.BASKET_ID.nunique()),
            "n_rows": int(len(baskets)),
            "n_commodities": int(items.cat_id.max()) + 1,
            "n_stores": n_stores,
            "val_from_week": val_from,
            "test_from_week": test_from,
            "analysis_first_week": 1,
            "analysis_last_week": n_periods,
            "promotion_coverage_required": [1, n_periods],
            "day_stride": 1024,
            "split_rows": baskets.split.value_counts().to_dict(),
            "price_basis": self.price_basis,
            "model_price_sources": list(self.model_price_sources),
            "max_units": int(baskets.units.max()),
            "share_rows_units_gt1": float((baskets.units > 1).mean()),
            "share_units_in_multi_rows": float(
                baskets.loc[baskets.units > 1, "units"].sum() / baskets.units.sum()),
            "store_price_cells": price_audit["store_price_cells"],
            "price_obs_share": price_audit["observed_item_period_fraction"],
            "median_repurchase_gap_days": state_audit["median_gap_days"],
            "promotion_contract": promotion_audit,
            "availability_contract": availability_audit,
            "canonical_build_audit_sha256": sha256(required["build_audit"]),
        }
        meta_path = basket_input / "meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
        build_meta = {
            "schema_version": 1,
            "dataset": "canonical_external_basket",
            "price_basis": self.price_basis,
            "model_price_sources": list(self.model_price_sources),
            "canonical_directory": str(canonical),
            "canonical_build_audit_sha256": sha256(required["build_audit"]),
        }
        build_meta_path = data_directory / "build_meta.json"
        build_meta_path.write_text(json.dumps(build_meta, indent=2, sort_keys=True) + "\n")

        # Hash only the files produced above.  Iterating an existing output directory
        # would pull stale downstream artifacts (or the prior manifest itself) into the
        # preprocessing identity and create a non-verifiable self-reference.
        derived_paths = {
            name: basket_input / name for name in (
                "items.parquet", "baskets.parquet", "log_price.npy",
                "log_price_dev.npy", "store_price.npz", "promo.npz",
                "promo_observed.npz", "state.npz", "meta.json")
        }
        if availability_audit["enabled"]:
            derived_paths["availability.npz"] = basket_input / "availability.npz"
        derived_paths["data/price_week.parquet"] = (
            data_directory / "price_week.parquet")
        derived = {name: sha256(path) for name, path in derived_paths.items()}
        preprocessing = {
            "schema_version": 1,
            "dataset": "canonical_external_basket",
            "status": "passed",
            "price_basis": self.price_basis,
            "cohort": {
                "n_items": n_items,
                "n_users": n_users,
                "n_basket_product_rows": int(len(baskets)),
                "n_baskets": int(baskets.BASKET_ID.nunique()),
                "weeks": [1, n_periods],
                "validation_from": val_from,
                "test_from": test_from,
                "minimum_selected_item_training_lines": int(
                    source_audit["audit"]["cohort_policy"][
                        "minimum_product_training_lines"]),
            },
            "raw_sha256": source_audit["source_sha256"],
            "canonical_sha256": {
                name: sha256(path) for name, path in required.items()
            },
            "derived_sha256": derived,
        }
        preprocessing_path = basket_input / "preprocessing_manifest.json"
        preprocessing_path.write_text(
            json.dumps(preprocessing, indent=2, sort_keys=True) + "\n")
        return ModelInputBuildResult(
            self.output_root, basket_input, data_directory,
            {"meta": meta, "preprocessing": preprocessing,
             "price": price_audit, "promotion": promotion_audit, "state": state_audit,
             "availability": availability_audit})

    @staticmethod
    def _validate_ids(tx: pd.DataFrame, products: pd.DataFrame) -> None:
        if not np.array_equal(products.item_id.to_numpy(), np.arange(len(products))):
            raise ValueError("canonical products need contiguous item_id values")
        if not set(tx.item_id).issubset(set(products.item_id)):
            raise ValueError("canonical transactions contain unknown item IDs")
        for column in ("customer_id", "store_id"):
            values = np.sort(tx[column].unique())
            if not np.array_equal(values, np.arange(len(values))):
                raise ValueError(f"{column} values must be contiguous")

    METADATA_COLUMNS = ("subcategory", "brand", "manufacturer", "department")

    @classmethod
    def _apply_product_metadata(cls, products: pd.DataFrame, path: Path):
        """Add declared catalogue columns from a separate file keyed by product_id.

        Catalogue attributes often come from a source other than the transaction adapter.
        The file supplies optional product columns of the canonical contract; it may not
        change identifiers or categories, and it must cover every product it names exactly
        once. Products it omits keep their canonical values (or the contract defaults).
        """
        import hashlib
        if not path.is_file():
            raise FileNotFoundError(f"product metadata file is missing: {path}")
        extra = pd.read_parquet(path)
        if "product_id" not in extra:
            raise ValueError("product metadata requires a product_id column")
        columns = [c for c in extra.columns if c != "product_id"]
        unknown = set(columns).difference(cls.METADATA_COLUMNS)
        if not columns or unknown:
            raise ValueError(f"product metadata columns must be a nonempty subset of "
                             f"{cls.METADATA_COLUMNS}; got {sorted(columns)}")
        if extra.product_id.duplicated().any():
            raise ValueError("product metadata lists a product_id more than once")
        missing = set(extra.product_id).difference(products.product_id)
        if missing:
            raise ValueError(f"product metadata names {len(missing)} unknown products")
        if extra[columns].isna().any().any():
            raise ValueError("product metadata values must be non-null")
        products = products.copy()
        lookup = extra.set_index("product_id")
        for column in columns:
            fallback = (products[column].astype(str) if column in products
                        else products.category.astype(str) if column == "subcategory" else None)
            mapped = products.product_id.map(lookup[column].astype(str))
            if fallback is None:
                if mapped.isna().any():
                    raise ValueError(f"product metadata must cover every product for {column}")
                products[column] = mapped
            else:
                products[column] = mapped.fillna(fallback)
        audit = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                 "columns": columns, "products_covered": int(len(extra))}
        return products, audit

    @staticmethod
    def _items(products: pd.DataFrame, tx: pd.DataFrame,
               metadata_defaults: dict[str, str] | None = None) -> pd.DataFrame:
        defaults = {**DEFAULT_METADATA, **(metadata_defaults or {})}
        category_values = sorted(products.category.unique())
        category_map = {value: index for index, value in enumerate(category_values)}
        counts = tx.groupby("item_id").agg(
            n_lines=("quantity", "size"), n_households=("customer_id", "nunique"))
        training = tx[tx.split == "train"].groupby("item_id").agg(
            n_train_lines=("quantity", "size"),
            n_train_households=("customer_id", "nunique"))
        items = products.merge(counts, on="item_id").merge(training, on="item_id")
        items["cat_id"] = items.category.map(category_map).astype(np.int32)
        if "subcategory" in items:
            sub_label = items.subcategory.astype(str)
            sub_values = sorted(sub_label.unique())
            items["sub_id"] = sub_label.map(
                {value: index for index, value in enumerate(sub_values)}).astype(np.int32)
        else:
            sub_label = items.category
            items["sub_id"] = items.cat_id
        items["PRODUCT_ID"] = items.item_id.astype(np.int64)
        items["COMMODITY_DESC"] = items.category
        items["SUB_COMMODITY_DESC"] = sub_label
        items["BRAND"] = items.brand.astype(str) if "brand" in items else items.label
        items["MANUFACTURER"] = (items.manufacturer.astype(str) if "manufacturer" in items
                                 else defaults["MANUFACTURER"])
        items["DEPARTMENT"] = (items.department.astype(str) if "department" in items
                               else defaults["DEPARTMENT"])
        return items.sort_values("item_id").reset_index(drop=True)

    @staticmethod
    def _baskets(tx: pd.DataFrame, items: pd.DataFrame) -> pd.DataFrame:
        item_sub = items.set_index("item_id").sub_id
        baskets = tx.rename(columns={
            "basket_id": "BASKET_ID", "customer_id": "user_id",
            "day": "DAY", "period": "WEEK_NO", "quantity": "units",
            "unit_price": "price",
        }).copy()
        baskets["sub_id"] = baskets.item_id.map(item_sub).astype(np.int32)
        raw_units = baskets.units.to_numpy(dtype=np.float64)
        rounded_units = np.rint(raw_units)
        if (not np.isfinite(raw_units).all()
                or bool((raw_units <= 0).any())
                or bool((np.abs(raw_units - rounded_units) > 1e-9).any())):
            raise ValueError(
                "canonical transaction quantities must be finite, positive integers")
        if bool((rounded_units > np.iinfo(np.int16).max).any()):
            raise ValueError("canonical transaction quantity exceeds int16 capacity")
        baskets["units"] = rounded_units.astype(np.int16)
        columns = [
            "BASKET_ID", "user_id", "DAY", "WEEK_NO", "item_id", "sub_id",
            "units", "price", "store_id", "split"]
        baskets = baskets[columns].sort_values(
            ["user_id", "DAY", "BASKET_ID", "item_id"]).reset_index(drop=True)
        if baskets.duplicated(["BASKET_ID", "item_id"]).any():
            raise ValueError("model basket input contains duplicate basket-item rows")
        return baskets

    def _write_prices(self, prices: pd.DataFrame, items: pd.DataFrame, n_stores: int,
                      n_periods: int, n_days: int, train_period_max: int,
                      output: Path) -> dict[str, Any]:
        """Write the chain product/day price panel from the declared price sources.

        A purchase-aggregate cell exists only because a panel household bought the
        product, so using it as a covariate conditions the price on the modeled outcome.
        By default only retail aggregates price the model; products with no such price get
        a constant training-period reference price (zero deviation on every day).
        """
        item_map = items.set_index("product_id").item_id
        prices = prices[
            prices.product_id.isin(item_map.index)
            & prices.store_id.between(0, n_stores - 1)].copy()
        prices["item_id"] = prices.product_id.map(item_map).astype(np.int32)
        reliable = prices[prices.price_source.eq("retail_aggregate")].copy()
        reliable = (reliable.groupby(["item_id", "period"], as_index=False)
                    .agg(units=("units", "sum"), revenue=("revenue", "sum")))
        reliable["price"] = reliable.revenue / reliable.units
        reliable["base_price"] = reliable.price
        reliable["PRODUCT_ID"] = reliable.item_id.astype(np.int64)
        reliable[["PRODUCT_ID", "period", "price", "base_price"]].rename(
            columns={"period": "WEEK_NO"}).to_parquet(
                output.parent / "data" / "price_week.parquet", index=False)
        sources_by_cell = prices.groupby(["item_id", "period"]).price_source.agg(
            lambda values: "retail_aggregate" in set(values))
        purchase_only_cells = sources_by_cell[~sources_by_cell].reset_index()
        excluded = prices[~prices.price_source.isin(self.model_price_sources)]
        modeled = prices[prices.price_source.isin(self.model_price_sources)]

        def store_week(frame):
            answer = (frame.groupby(["item_id", "store_id", "period"], as_index=False)
                      .agg(units=("units", "sum"), revenue=("revenue", "sum")))
            answer["price"] = answer.revenue / answer.units
            if bool((answer.price <= 0).any() or ~np.isfinite(answer.price).all()):
                raise ValueError("canonical store price panel contains invalid prices")
            return answer

        store_prices = store_week(modeled)
        chain = store_prices.groupby(["item_id", "period"], as_index=False).agg(
            units=("units", "sum"), revenue=("revenue", "sum"))
        chain["price"] = chain.revenue / chain.units
        grid = np.full((len(items), n_periods + 1), np.nan, dtype=np.float64)
        grid[chain.item_id.to_numpy(), chain.period.to_numpy()] = chain.price.to_numpy()
        observed = np.isfinite(grid[:, 1:])
        first_observed = np.where(observed.any(axis=1), observed.argmax(axis=1) + 1, 0)
        backfilled_heldout = (first_observed > train_period_max) & observed.any(axis=1)
        if bool(backfilled_heldout.any()):
            raise ValueError(
                "a training-period price would be backfilled from a held-out period for "
                f"items {np.flatnonzero(backfilled_heldout)[:10].tolist()}")
        filled = pd.DataFrame(grid[:, 1:]).ffill(axis=1).bfill(axis=1).to_numpy()
        unsupported = np.flatnonzero(~observed.any(axis=1))
        if len(unsupported):
            all_sources = store_week(prices[prices.period <= train_period_max])
            reference = all_sources.groupby("item_id").price.median()
            lacking = [int(item) for item in unsupported if item not in reference.index]
            if lacking:
                raise ValueError(
                    f"items have no training-period price from any source: {lacking[:10]}")
            filled[unsupported] = reference.loc[unsupported].to_numpy()[:, None]
        day_period = np.arange(n_days) // 7
        daily = filled[:, day_period]
        log_price = np.log(daily).astype(np.float32)
        training_days = day_period < train_period_max
        log_price_dev = (
            log_price - log_price[:, training_days].mean(axis=1, keepdims=True)
        ).astype(np.float32)
        # A constant reference price has exactly zero deviation; avoid float32 mean noise.
        log_price_dev[unsupported] = 0.0
        np.save(output / "log_price.npy", log_price)
        np.save(output / "log_price_dev.npy", log_price_dev)
        chain_lookup = chain.set_index(["item_id", "period"]).price
        store_prices["chain_price"] = [
            chain_lookup.loc[(item, period)]
            for item, period in zip(store_prices.item_id, store_prices.period)]
        store_prices["dev"] = np.log(store_prices.price) - np.log(store_prices.chain_price)
        deviations = store_prices[store_prices.dev.abs() > 1e-8]
        carried = np.zeros((len(items), n_stores), dtype=bool)
        training_prices = store_prices[store_prices.period <= train_period_max]
        carried[
            training_prices.item_id.to_numpy(),
            training_prices.store_id.to_numpy(),
        ] = True
        np.savez_compressed(
            output / "store_price.npz",
            item=deviations.item_id.to_numpy(np.int32),
            store=deviations.store_id.to_numpy(np.int32),
            week=deviations.period.to_numpy(np.int16),
            dev=deviations.dev.to_numpy(np.float32),
            carried=carried,
            n_stores=np.int32(n_stores),
        )
        return {
            "model_price_sources": list(self.model_price_sources),
            "purchase_derived_only_item_period_cells": int(len(purchase_only_cells)),
            "purchase_derived_only_items": int(purchase_only_cells.item_id.nunique()),
            "excluded_source_rows": int(len(excluded)),
            "excluded_source_item_period_cells": int(
                excluded[["item_id", "period"]].drop_duplicates().shape[0]),
            "constant_reference_price_items": [int(item) for item in unsupported],
            "backfilled_leading_item_periods": int(
                sum(max(int(first) - 1, 0) for first in first_observed[observed.any(axis=1)])),
            "store_price_cells": int(len(store_prices)),
            "store_deviation_cells": int(len(deviations)),
            "observed_item_period_fraction": float(
                len(chain) / (len(items) * n_periods)),
        }

    def _write_promotions(self, promotions: pd.DataFrame, items: pd.DataFrame,
                          n_stores: int, n_periods: int, output: Path) -> dict[str, Any]:
        item_map = items.set_index("product_id").item_id
        frame = promotions[
            promotions.product_id.isin(item_map.index)
            & promotions.store_id.between(0, n_stores - 1)].copy()
        frame["item_id"] = frame.product_id.map(item_map).astype(np.int32)
        frame = frame.groupby(["item_id", "store_id", "period"], as_index=False).agg(
            display=("display", "max"), advertised=("advertised", "max"),
            special_price=("special_price", "max"))
        observed_active = frame[
            frame.display | frame.advertised | frame.special_price].copy()
        observed_keys = ((observed_active.item_id.to_numpy(np.int64) * n_stores
                          + observed_active.store_id.to_numpy(np.int64)) * 128
                         + observed_active.period.to_numpy(np.int64))
        np.savez_compressed(
            output / "promo_observed.npz", keys=observed_keys,
            disp=observed_active.display.to_numpy(np.int8),
            mail=(observed_active.advertised | observed_active.special_price
                  ).to_numpy(np.int8),
            coverage_min_week=np.int16(1), coverage_max_week=np.int16(n_periods))
        if self.promotion_feature_name == "disabled":
            active = frame.iloc[0:0].copy()
            active["modeled_promotion"] = np.int8(0)
            mail = active.modeled_promotion.to_numpy(np.int8)
            feature_mapping = {
                "display": "disabled because source coverage is incomplete",
                "mail": "disabled because source coverage is incomplete",
            }
        else:
            active = frame[frame.display | frame[self.promotion_feature_name]].copy()
            mail = active[self.promotion_feature_name].to_numpy(np.int8)
            feature_mapping = {
                "display": "any ERIM display flag",
                "mail": f"ERIM {self.promotion_feature_name} flag",
            }
        keys = ((active.item_id.to_numpy(np.int64) * n_stores
                 + active.store_id.to_numpy(np.int64)) * 128
                + active.period.to_numpy(np.int64))
        np.savez_compressed(
            output / "promo.npz", keys=keys,
            disp=active.display.to_numpy(np.int8),
            mail=mail,
            coverage_min_week=np.int16(1), coverage_max_week=np.int16(n_periods))
        return {
            "enabled": self.promotion_feature_name != "disabled",
            "feature_mapping": feature_mapping,
            "observed_cells": int(len(frame)),
            "active_cells": int(len(active)),
            "source_coverage": self.promotion_coverage_note,
            "model_treatment": (
                "all promotion features set to zero"
                if self.promotion_feature_name == "disabled"
                else "partial source panel used as modeled features"),
        }

    def _write_availability(self, prices: pd.DataFrame, baskets: pd.DataFrame,
                            items: pd.DataFrame, n_stores: int, n_periods: int,
                            train_period_max: int, output: Path) -> dict[str, Any]:
        """Write when each product became purchasable at each store.

        Transaction logs record purchases, not stock, so availability comes from an
        independent store sales feed: a product is confirmed at a store from the first
        period in which that store's retail aggregate records a sale by shoppers other than
        the modeled cohort.  Subtracting the cohort's own units prevents a modeled purchase
        from confirming its own availability.  A first sale within the store's first
        ``availability_left_censor_periods`` feed periods is treated as available from the
        start, because a slow seller need not sell in the opening weeks of the feed.

        A missing sale is not proof of absence, so an unconfirmed product is not removed.
        Its utility is shifted by log(epsilon), where epsilon is the training-period purchase
        rate of unconfirmed (product, store, period) cells relative to confirmed cells.
        Products or stores without any retail feed are treated as always available, which
        is the declared-catalogue support used when no feed exists.

        ``first_period[item, store]`` holds the first available period (0 means always
        available; ``n_periods + 1`` means never confirmed in the window).
        """
        if self.availability == "disabled":
            return {"enabled": False, "rule": "disabled",
                    "reason": "no store availability feed requested; declared catalogue support"}
        item_map = items.set_index("product_id").item_id
        retail = prices[
            prices.price_source.eq("retail_aggregate")
            & prices.product_id.isin(item_map.index)
            & prices.store_id.between(0, n_stores - 1)].copy()
        n_items = len(items)
        first_period = np.zeros((n_items, n_stores), dtype=np.int16)
        never = n_periods + 1
        if retail.empty:
            np.savez_compressed(output / "availability.npz", first_period=first_period,
                                log_floor=np.float64(0.0), n_periods=np.int16(n_periods))
            return {"enabled": True, "rule": self.availability, "tracked_items": 0,
                    "tracked_stores": 0, "floor": 1.0, "log_floor": 0.0,
                    "note": "no retail aggregate rows; every product is always available"}
        retail["item_id"] = retail.product_id.map(item_map).astype(np.int64)
        retail_units = retail.groupby(["item_id", "store_id", "period"]).units.sum()
        cohort_units = baskets.groupby(["item_id", "store_id", "WEEK_NO"]).units.sum()
        cohort_units.index = cohort_units.index.set_names(["item_id", "store_id", "period"])
        net = retail_units - cohort_units.reindex(retail_units.index).fillna(0.0)
        sold = net[net > 0].reset_index()
        store_start = retail.groupby("store_id").period.min()
        tracked_items = np.sort(retail.item_id.unique())
        tracked_stores = np.sort(store_start.index.to_numpy())
        first_period[np.ix_(tracked_items, tracked_stores)] = never
        first_sale = sold.groupby(["item_id", "store_id"]).period.min().reset_index()
        censored = first_sale.period <= (
            first_sale.store_id.map(store_start) + self.availability_left_censor_periods)
        launch = np.where(censored, 0, first_sale.period).astype(np.int16)
        first_period[first_sale.item_id.to_numpy(), first_sale.store_id.to_numpy()] = launch

        lines = baskets[["item_id", "store_id", "WEEK_NO", "split"]]
        confirmed = lines.WEEK_NO.to_numpy() >= first_period[
            lines.item_id.to_numpy(), lines.store_id.to_numpy()]
        unconfirmed_by_split = lines[~confirmed].split.value_counts().to_dict()

        # epsilon: purchase rate per exposure in unconfirmed vs confirmed training cells.
        training = baskets[baskets.split.eq("train")]
        trips = training.groupby(["store_id", "WEEK_NO"]).BASKET_ID.nunique()
        sorted_first = np.sort(first_period, axis=0)
        exposure_unconfirmed = exposure_confirmed = 0.0
        for (store, period), count in trips.items():
            unconfirmed = n_items - int(np.searchsorted(
                sorted_first[:, store], period, side="right"))
            exposure_unconfirmed += float(count) * unconfirmed
            exposure_confirmed += float(count) * (n_items - unconfirmed)
        train_confirmed = training.WEEK_NO.to_numpy() >= first_period[
            training.item_id.to_numpy(), training.store_id.to_numpy()]
        lines_confirmed = int(train_confirmed.sum())
        lines_unconfirmed = int((~train_confirmed).sum())
        if exposure_unconfirmed > 0:
            epsilon = (((lines_unconfirmed + 0.5) / (exposure_unconfirmed + 1.0))
                       / ((lines_confirmed + 0.5) / (exposure_confirmed + 1.0)))
        else:
            epsilon = 1.0
        floor = float(np.clip(epsilon, *self.AVAILABILITY_FLOOR_BOUNDS))
        np.savez_compressed(output / "availability.npz", first_period=first_period,
                            log_floor=np.float64(np.log(floor)),
                            n_periods=np.int16(n_periods))
        launched_after_training = int((
            (first_period > train_period_max) & (first_period <= n_periods)).sum())
        return {
            "enabled": True,
            "rule": self.availability,
            "source": "retail_aggregate store-period units net of the modeled cohort's units",
            "left_censor_periods": self.availability_left_censor_periods,
            "tracked_items": int(len(tracked_items)),
            "tracked_stores": int(len(tracked_stores)),
            "untracked_items_always_available": int(n_items - len(tracked_items)),
            "confirmed_item_store_pairs": int((first_period[np.ix_(
                tracked_items, tracked_stores)] <= n_periods).sum()),
            "never_confirmed_item_store_pairs": int((first_period == never).sum()),
            "pairs_launched_after_training": launched_after_training,
            "estimated_epsilon": float(epsilon),
            "floor": floor,
            "log_floor": float(np.log(floor)),
            "training_lines_confirmed": lines_confirmed,
            "training_lines_unconfirmed": lines_unconfirmed,
            "training_exposure_unconfirmed_share": float(
                exposure_unconfirmed / max(exposure_unconfirmed + exposure_confirmed, 1.0)),
            "unconfirmed_lines_by_split": {str(k): int(v) for k, v in unconfirmed_by_split.items()},
        }

    @staticmethod
    def _write_state(baskets: pd.DataFrame, items: pd.DataFrame, n_users: int,
                     n_days: int, output: Path) -> dict[str, Any]:
        del n_days
        item_sub = items.sort_values("item_id").sub_id.to_numpy(np.int32)
        n_sub = int(item_sub.max()) + 1
        events = baskets[["user_id", "sub_id", "DAY"]].drop_duplicates()
        events["group"] = events.user_id.astype(np.int64) * n_sub + events.sub_id
        events["key"] = events.group * 1024 + events.DAY
        keys = np.sort(events.key.to_numpy(np.int64))
        groups = keys // 1024
        starts = np.r_[True, groups[1:] != groups[:-1]]
        gstart_keys = groups[starts]
        gstart_vals = np.flatnonzero(starts).astype(np.int64)
        gap_values: dict[int, list[int]] = {sub: [] for sub in range(n_sub)}
        for (_user, sub), group in events.groupby(["user_id", "sub_id"]):
            days = np.sort(group.DAY.unique())
            if len(days) > 1:
                gap_values[int(sub)].extend(np.diff(days).tolist())
        sub_gap = np.asarray([
            np.median(gap_values[sub]) if gap_values[sub] else 28.0
            for sub in range(n_sub)], dtype=np.float32)
        np.savez_compressed(
            output / "state.npz", keys=keys, gstart_keys=gstart_keys,
            gstart_vals=gstart_vals, sub_gap=sub_gap, item_sub=item_sub,
            cat_keys=keys, cat_gap=sub_gap.copy(), item_cat=item_sub.copy())
        return {
            "events": int(len(keys)),
            "groups": int(len(gstart_keys)),
            "median_gap_days": float(np.median(sub_gap)),
            "n_users": n_users,
        }
