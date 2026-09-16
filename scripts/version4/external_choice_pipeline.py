"""Dataset-independent orchestration for held-out retail choice validation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
import hashlib
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np

from external_choice import (
    ChoicePanel,
    choice_metrics,
    clustered_loss_difference,
    counterfactual_price_increase_audit,
    fit_choice_model,
    merge_panels,
    price_alignment_placebo,
    select_regularization,
    standardize_covariates,
    temporal_split,
    validate_panel,
    within_group_split,
)


@dataclass(frozen=True)
class DatasetCapabilities:
    """Facts about what a prepared dataset can scientifically support."""

    conditional_choice: bool = True
    full_baskets: bool = False
    temporal_order: bool = False
    persistent_customers: bool = True
    outside_option: bool = False
    quantities: bool = False
    promotions: bool = False
    known_assortment: bool = False
    randomized_prices: bool = False

    def full_basket_pipeline_blockers(self) -> list[str]:
        requirements = {
            "full_baskets": self.full_baskets,
            "temporal_order": self.temporal_order,
            "persistent_customers": self.persistent_customers,
        }
        return [name for name, available in requirements.items() if not available]


@dataclass(frozen=True)
class SplitResult:
    train: np.ndarray
    validation: np.ndarray
    test: np.ndarray
    metadata: Mapping[str, Any]
    test_blocks: Mapping[str, np.ndarray] = field(default_factory=dict)


class SplitStrategy(ABC):
    """Replaceable split policy; the experiment engine never inspects source format."""

    @abstractmethod
    def split(self, dataset: "PreparedChoiceDataset", seed: int) -> SplitResult:
        raise NotImplementedError


@dataclass(frozen=True)
class GroupedRandomSplit(SplitStrategy):
    minimum_group_size: int = 5
    train_fraction: float = .6
    validation_fraction: float = .2

    def split(self, dataset: "PreparedChoiceDataset", seed: int) -> SplitResult:
        train, validation, test = within_group_split(
            dataset.panel, seed, minimum_group_size=self.minimum_group_size,
            train_fraction=self.train_fraction,
            validation_fraction=self.validation_fraction)
        retained = np.concatenate([train, validation, test])
        return SplitResult(
            train, validation, test,
            {
                "type": "within-group randomized",
                "reason": "source has no usable temporal ordering",
                "seed": seed,
                "minimum_group_choices": self.minimum_group_size,
                "groups": int(len(np.unique(dataset.panel.group[retained]))),
            },
        )


@dataclass(frozen=True)
class TemporalSplit(SplitStrategy):
    train_fraction: float = .6
    validation_fraction: float = .2
    block_name: str = "period"

    def split(self, dataset: "PreparedChoiceDataset", seed: int) -> SplitResult:
        del seed
        if dataset.periods is None:
            raise ValueError(f"{dataset.dataset_id} requested a temporal split without periods")
        train, validation, test, boundaries = temporal_split(
            dataset.periods, train_fraction=self.train_fraction,
            validation_fraction=self.validation_fraction)
        return SplitResult(
            train, validation, test,
            {
                "type": "whole-period temporal",
                "block_name": self.block_name,
                "last_training_period": boundaries[0],
                "last_validation_period": boundaries[1],
            },
            {self.block_name: np.asarray(dataset.periods)[test]},
        )


@dataclass(frozen=True)
class PreparedChoiceDataset:
    dataset_id: str
    panel: ChoicePanel
    product_labels: Sequence[str]
    split_strategy: SplitStrategy
    capabilities: DatasetCapabilities
    source_files: Sequence[Path]
    periods: np.ndarray | None = None
    construction_audit: Mapping[str, Any] = field(default_factory=dict)
    seed_offset: int = 0

    def __post_init__(self) -> None:
        if not self.dataset_id or any(character.isspace() for character in self.dataset_id):
            raise ValueError("dataset_id must be a nonempty token without whitespace")
        validate_panel(self.panel, n_products=len(self.product_labels))
        if len(self.product_labels) != self.panel.n_products:
            raise ValueError("product_labels must contain exactly one label per product")
        if self.periods is not None and np.asarray(self.periods).shape != (
                len(self.panel.chosen_slot),):
            raise ValueError("periods must have one value per choice occasion")
        if not self.capabilities.conditional_choice:
            raise ValueError("the conditional-choice engine requires conditional_choice capability")


class ChoiceDatasetAdapter(ABC):
    """Source-specific boundary that must emit one canonical choice representation."""

    @abstractmethod
    def load(self) -> PreparedChoiceDataset:
        raise NotImplementedError


@dataclass(frozen=True)
class ValidationSettings:
    seed: int = 24051991
    regularization_grid: tuple[float, ...] = (0.0, 1e-4, 1e-3, 1e-2, 1e-1)
    bootstrap_replicates: int = 2000
    placebo_replicates: int = 200
    relative_price_increase: float = .10

    def __post_init__(self) -> None:
        if not self.regularization_grid or any(value < 0 for value in self.regularization_grid):
            raise ValueError("regularization grid must contain nonnegative values")
        if self.bootstrap_replicates < 20 or self.placebo_replicates < 20:
            raise ValueError("audit replicate counts must each be at least 20")
        if self.relative_price_increase <= 0:
            raise ValueError("relative price increase must be positive")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class ExternalChoiceExperiment:
    """Reusable fit/select/evaluate workflow over canonical choice datasets."""

    def __init__(self, settings: ValidationSettings):
        self.settings = settings

    def evaluate(self, dataset: PreparedChoiceDataset) -> dict[str, Any]:
        seed = self.settings.seed + dataset.seed_offset
        split = dataset.split_strategy.split(dataset, seed)
        standardized, scaling = standardize_covariates(dataset.panel, split.train)
        train = standardized.subset(split.train)
        validation = standardized.subset(split.validation)
        test = standardized.subset(split.test)
        n_products = len(dataset.product_labels)
        print(
            f"[{dataset.dataset_id}] train={len(split.train)} "
            f"validation={len(split.validation)} test={len(split.test)}",
            flush=True,
        )

        baseline_selection, baseline_grid = select_regularization(
            train, validation, n_products=n_products, include_price=False,
            grid=self.settings.regularization_grid)
        price_selection, price_grid = select_regularization(
            train, validation, n_products=n_products, include_price=True,
            grid=self.settings.regularization_grid)
        training_and_validation = merge_panels(train, validation)
        baseline = fit_choice_model(
            training_and_validation, n_products=n_products, include_price=False,
            regularization=baseline_selection.regularization)
        price = fit_choice_model(
            training_and_validation, n_products=n_products, include_price=True,
            regularization=price_selection.regularization)

        primary_cluster = (
            "customer" if dataset.capabilities.persistent_customers else "occasion")
        comparisons = {
            primary_cluster: clustered_loss_difference(
                test, price, baseline, seed=seed + 901,
                replicates=self.settings.bootstrap_replicates,
                cluster_name=primary_cluster)
        }
        for offset, (block_name, values) in enumerate(split.test_blocks.items(), start=1):
            comparisons[f"{block_name}_block"] = clustered_loss_difference(
                test, price, baseline, seed=seed + 901 + 1000 * offset,
                replicates=self.settings.bootstrap_replicates,
                clusters=values, cluster_name=block_name)

        counterfactual, case = counterfactual_price_increase_audit(
            test, price, training_and_validation,
            relative_increase=self.settings.relative_price_increase)
        placebo = price_alignment_placebo(
            test, price, baseline, seed=seed + 2901,
            replicates=self.settings.placebo_replicates)
        if case is not None:
            for key in ("changed_product", "largest_recipient_product"):
                case[key + "_label"] = dataset.product_labels[int(case[key])]
            case["interpretation"] = "model-implied predictive scenario; not a causal effect"

        predictive_gate = all(
            comparison["estimate"] < 0
            and comparison["95_cluster_bootstrap_interval"][1] < 0
            for comparison in comparisons.values()) and placebo["passed"]
        blockers = dataset.capabilities.full_basket_pipeline_blockers()
        split_metadata = dict(split.metadata)
        split_metadata["covariate_training_standardization"] = scaling
        result = {
            "dataset": dataset.dataset_id,
            "claim_level": (
                "heldout predictive price choice; causal only when randomized_prices "
                "capability is independently verified"),
            "capabilities": asdict(dataset.capabilities),
            "full_basket_pipeline": {
                "eligible": not blockers,
                "blockers": blockers,
            },
            "construction_audit": dict(dataset.construction_audit),
            "split": split_metadata,
            "partitions": {
                "train": len(train.chosen_slot),
                "validation": len(validation.chosen_slot),
                "test": len(test.chosen_slot),
            },
            "products": n_products,
            "regularization_search": {
                "no_price": baseline_grid,
                "with_price": price_grid,
                "selected_no_price": baseline.regularization,
                "selected_with_price": price.regularization,
            },
            "test_metrics": {
                "no_price": choice_metrics(test, baseline),
                "with_price": choice_metrics(test, price),
            },
            "test_nll_comparisons": comparisons,
            "heldout_price_alignment_placebo": placebo,
            "price_coefficient": price.price_coefficient,
            "price_evidence_gate": "passed" if predictive_gate else "failed",
            "counterfactual_numerics": counterfactual,
            "worked_supported_counterfactual": case,
            "final_model_optimization": {
                "no_price": self._optimization_record(baseline),
                "with_price": self._optimization_record(price),
            },
            "source_files": {
                str(Path(path).resolve()): file_sha256(Path(path).resolve())
                for path in dataset.source_files
            },
        }
        print(
            f"[{dataset.dataset_id}] price_gate={result['price_evidence_gate']} "
            f"beta={price.price_coefficient:.6f}",
            flush=True,
        )
        return result

    def run(self, adapters: Sequence[ChoiceDatasetAdapter]) -> dict[str, Any]:
        if not adapters:
            raise ValueError("at least one dataset adapter is required")
        started = time.monotonic()
        results: dict[str, dict[str, Any]] = {}
        for adapter in adapters:
            dataset = adapter.load()
            if dataset.dataset_id in results:
                raise ValueError(f"duplicate dataset_id: {dataset.dataset_id}")
            results[dataset.dataset_id] = self.evaluate(dataset)
        return {
            "status": "completed",
            "engine": "ExternalChoiceExperiment",
            "settings": asdict(self.settings),
            "datasets": results,
            "decisions": {
                key: {
                    "predictive_price_counterfactual": (
                        value["price_evidence_gate"] == "passed"),
                    "causal_price_effect": bool(
                        value["price_evidence_gate"] == "passed"
                        and value["capabilities"]["randomized_prices"]
                        and value["capabilities"]["outside_option"]
                        and value["capabilities"]["known_assortment"]),
                    "full_basket_pipeline": bool(
                        value["full_basket_pipeline"]["eligible"]),
                }
                for key, value in results.items()
            },
            "runtime_seconds": time.monotonic() - started,
        }

    @staticmethod
    def _optimization_record(fit) -> dict[str, Any]:
        return {
            "iterations": fit.iterations,
            "gradient_infinity_norm": fit.gradient_infinity_norm,
            "termination_message": fit.termination_message,
        }
