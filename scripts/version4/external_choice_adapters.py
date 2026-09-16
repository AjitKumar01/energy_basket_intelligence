"""Source adapters for the canonical external choice experiment contract."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Mapping

import numpy as np
import pandas as pd

from external_choice import ChoicePanel, validate_panel
from external_choice_data import build_erim_choice_data, read_bayesm_csv
from external_choice_pipeline import (
    ChoiceDatasetAdapter,
    DatasetCapabilities,
    GroupedRandomSplit,
    PreparedChoiceDataset,
    SplitStrategy,
    TemporalSplit,
)


def _existing(path: Path) -> Path:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@dataclass(frozen=True)
class BayesmRdaAdapter(ChoiceDatasetAdapter):
    rda_path: Path
    dataset_id: str = "bayesm_margarine"
    seed_offset: int = 0
    minimum_group_size: int = 5

    def load(self) -> PreparedChoiceDataset:
        rda = _existing(self.rda_path)
        with tempfile.TemporaryDirectory(prefix="bayesm_choice_") as temporary:
            temporary_path = Path(temporary)
            choice = temporary_path / "choice.csv"
            demographics = temporary_path / "demographics.csv"
            expression = (
                "a <- commandArgs(trailingOnly=TRUE); load(a[1]); "
                "write.csv(margarine$choicePrice,a[2],row.names=FALSE); "
                "write.csv(margarine$demos,a[3],row.names=FALSE)"
            )
            subprocess.run(
                ["Rscript", "-e", expression, str(rda), str(choice), str(demographics)],
                check=True,
            )
            panel, labels = read_bayesm_csv(choice, demographics)
        return PreparedChoiceDataset(
            dataset_id=self.dataset_id,
            panel=panel,
            product_labels=tuple(labels),
            split_strategy=GroupedRandomSplit(
                minimum_group_size=self.minimum_group_size),
            capabilities=DatasetCapabilities(
                conditional_choice=True,
                full_baskets=False,
                temporal_order=False,
                persistent_customers=True,
                outside_option=False,
                quantities=False,
                promotions=False,
                known_assortment=True,
                randomized_prices=False,
            ),
            source_files=(rda,),
            construction_audit={
                "source_format": "R data object",
                "occasion_definition": "one observed margarine-category choice",
                "alternatives": "all ten products have a price on every occasion",
            },
            seed_offset=self.seed_offset,
        )


@dataclass(frozen=True)
class ErimCategoryAdapter(ChoiceDatasetAdapter):
    purchase_path: Path
    shopping_path: Path
    retail_path: Path
    product_path: Path | None = None
    dataset_id: str = "erim_category"
    seed_offset: int = 0
    time_block_name: str = "week"

    def load(self) -> PreparedChoiceDataset:
        purchase = _existing(self.purchase_path)
        shopping = _existing(self.shopping_path)
        retail = _existing(self.retail_path)
        products = _existing(self.product_path) if self.product_path is not None else None
        built = build_erim_choice_data(purchase, shopping, retail, products)
        sources = (purchase, shopping, retail) + (() if products is None else (products,))
        return PreparedChoiceDataset(
            dataset_id=self.dataset_id,
            panel=built.panel,
            product_labels=tuple(built.product_labels),
            split_strategy=TemporalSplit(block_name=self.time_block_name),
            capabilities=DatasetCapabilities(
                conditional_choice=True,
                full_baskets=False,
                temporal_order=True,
                persistent_customers=True,
                outside_option=False,
                quantities=True,
                promotions=True,
                known_assortment=False,
                randomized_prices=False,
            ),
            source_files=sources,
            periods=built.week,
            construction_audit=built.audit,
            seed_offset=self.seed_offset,
        )


@dataclass(frozen=True)
class LongChoiceCsvAdapter(ChoiceDatasetAdapter):
    """Generic adapter for one-row-per-offered-product choice data.

    The source must contain one and only one chosen alternative per occasion. Column
    names, covariates, labels, and split behavior are configuration rather than code.
    """

    csv_path: Path
    dataset_id: str
    occasion_column: str = "occasion_id"
    group_column: str = "customer_id"
    product_column: str = "product_id"
    price_column: str = "price"
    chosen_column: str = "chosen"
    label_column: str | None = None
    period_column: str | None = None
    covariate_columns: tuple[str, ...] = ()
    split_strategy: SplitStrategy = GroupedRandomSplit()
    capabilities: DatasetCapabilities = DatasetCapabilities()
    seed_offset: int = 0

    def load(self) -> PreparedChoiceDataset:
        path = _existing(self.csv_path)
        frame = pd.read_csv(path)
        required = {
            self.occasion_column, self.group_column, self.product_column,
            self.price_column, self.chosen_column, *self.covariate_columns,
        }
        if self.label_column is not None:
            required.add(self.label_column)
        if self.period_column is not None:
            required.add(self.period_column)
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{path.name} is missing configured columns: {missing}")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{path.name} contains nulls in required choice fields")
        price = pd.to_numeric(frame[self.price_column], errors="coerce")
        chosen = pd.to_numeric(frame[self.chosen_column], errors="coerce")
        if price.isna().any() or bool((price <= 0).any()):
            raise ValueError("offered prices must be finite and positive")
        if chosen.isna().any() or not set(chosen.unique()).issubset({0, 1}):
            raise ValueError("chosen indicator must contain only zero and one")
        frame = frame.copy()
        frame["__price"] = price.astype(np.float64)
        frame["__chosen"] = chosen.astype(np.int8)
        frame["__product_key"] = frame[self.product_column].astype(str)
        if frame.duplicated([self.occasion_column, "__product_key"]).any():
            raise ValueError("an occasion contains a duplicate offered product")
        chosen_counts = frame.groupby(self.occasion_column, sort=False)["__chosen"].sum()
        if not bool(chosen_counts.eq(1).all()):
            raise ValueError("every occasion must contain exactly one chosen alternative")
        sizes = frame.groupby(self.occasion_column, sort=False).size()
        if bool((sizes < 2).any()):
            raise ValueError("every occasion must contain at least two alternatives")

        product_keys = sorted(frame["__product_key"].unique())
        product_index = {key: index for index, key in enumerate(product_keys)}
        label_map = {key: key for key in product_keys}
        if self.label_column is not None:
            counts = frame.groupby("__product_key")[self.label_column].nunique()
            if bool((counts != 1).any()):
                raise ValueError("a product maps to more than one configured label")
            label_map.update(
                frame.drop_duplicates("__product_key").set_index("__product_key")[
                    self.label_column].astype(str).to_dict())

        width = int(sizes.max())
        occasions = list(frame.groupby(self.occasion_column, sort=False))
        n = len(occasions)
        product = np.zeros((n, width), dtype=np.int64)
        log_price = np.zeros((n, width), dtype=np.float64)
        mask = np.zeros((n, width), dtype=bool)
        chosen_slot = np.empty(n, dtype=np.int64)
        groups = np.empty(n, dtype=object)
        covariates = np.empty((n, len(self.covariate_columns)), dtype=np.float64)
        periods = np.empty(n, dtype=object) if self.period_column is not None else None
        for row, (_, alternatives) in enumerate(occasions):
            self._require_constant(alternatives, self.group_column)
            if self.period_column is not None:
                self._require_constant(alternatives, self.period_column)
            for column in self.covariate_columns:
                self._require_constant(alternatives, column)
            length = len(alternatives)
            product[row, :length] = [
                product_index[key] for key in alternatives["__product_key"]]
            log_price[row, :length] = np.log(alternatives["__price"].to_numpy())
            mask[row, :length] = True
            chosen_slot[row] = int(np.flatnonzero(alternatives["__chosen"].to_numpy())[0])
            groups[row] = alternatives[self.group_column].iloc[0]
            if self.covariate_columns:
                covariates[row] = pd.to_numeric(
                    alternatives.iloc[0][list(self.covariate_columns)], errors="raise")
            if periods is not None:
                periods[row] = alternatives[self.period_column].iloc[0]
        panel = ChoicePanel(product, log_price, mask, chosen_slot, groups, covariates)
        validate_panel(panel, n_products=len(product_keys))
        return PreparedChoiceDataset(
            dataset_id=self.dataset_id,
            panel=panel,
            product_labels=tuple(label_map[key] for key in product_keys),
            split_strategy=self.split_strategy,
            capabilities=self.capabilities,
            source_files=(path,),
            periods=periods,
            construction_audit={
                "source_format": "canonical long choice CSV",
                "rows": int(len(frame)),
                "occasions": n,
                "products": len(product_keys),
                "minimum_choice_set": int(sizes.min()),
                "maximum_choice_set": width,
            },
            seed_offset=self.seed_offset,
        )

    @staticmethod
    def _require_constant(frame: pd.DataFrame, column: str) -> None:
        if frame[column].nunique(dropna=False) != 1:
            raise ValueError(f"occasion-level field {column!r} varies within an occasion")


def _resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _split_from_spec(spec: Mapping[str, Any]) -> SplitStrategy:
    kind = spec.get("type", "grouped_random")
    if kind == "grouped_random":
        return GroupedRandomSplit(
            minimum_group_size=int(spec.get("minimum_group_size", 5)),
            train_fraction=float(spec.get("train_fraction", .6)),
            validation_fraction=float(spec.get("validation_fraction", .2)),
        )
    if kind == "temporal":
        return TemporalSplit(
            train_fraction=float(spec.get("train_fraction", .6)),
            validation_fraction=float(spec.get("validation_fraction", .2)),
            block_name=str(spec.get("block_name", "period")),
        )
    raise ValueError(f"unknown split strategy: {kind!r}")


def adapter_from_config(spec: Mapping[str, Any], *, base_directory: Path) -> ChoiceDatasetAdapter:
    """Construct a registered adapter from declarative configuration."""
    kind = spec.get("type")
    dataset_id = str(spec.get("id", ""))
    sources = spec.get("sources", {})
    seed_offset = int(spec.get("seed_offset", 0))
    if not dataset_id:
        raise ValueError("every dataset configuration needs a nonempty id")
    if kind == "bayesm_rda":
        return BayesmRdaAdapter(
            _resolve(base_directory, sources["rda"]), dataset_id, seed_offset,
            int(spec.get("minimum_group_size", 5)))
    if kind == "erim_category":
        product_value = sources.get("products")
        return ErimCategoryAdapter(
            purchase_path=_resolve(base_directory, sources["purchase"]),
            shopping_path=_resolve(base_directory, sources["shopping"]),
            retail_path=_resolve(base_directory, sources["retail"]),
            product_path=(None if product_value is None
                          else _resolve(base_directory, product_value)),
            dataset_id=dataset_id,
            seed_offset=seed_offset,
            time_block_name=str(spec.get("time_block_name", "week")),
        )
    if kind == "long_choice_csv":
        columns = spec.get("columns", {})
        capability_values = spec.get("capabilities", {})
        capabilities = DatasetCapabilities(**capability_values)
        return LongChoiceCsvAdapter(
            csv_path=_resolve(base_directory, sources["choices"]),
            dataset_id=dataset_id,
            occasion_column=str(columns.get("occasion", "occasion_id")),
            group_column=str(columns.get("group", "customer_id")),
            product_column=str(columns.get("product", "product_id")),
            price_column=str(columns.get("price", "price")),
            chosen_column=str(columns.get("chosen", "chosen")),
            label_column=columns.get("label"),
            period_column=columns.get("period"),
            covariate_columns=tuple(columns.get("covariates", ())),
            split_strategy=_split_from_spec(spec.get("split", {})),
            capabilities=capabilities,
            seed_offset=seed_offset,
        )
    raise ValueError(
        f"unknown dataset adapter type {kind!r}; expected bayesm_rda, "
        "erim_category, or long_choice_csv")
