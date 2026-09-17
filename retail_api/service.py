"""Thread-safe inference service over the fitted, audited joint basket law."""
from __future__ import annotations

import json
import math
import os
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .errors import RetailAPIError
from .schemas import (
    BasketCompletionRequest,
    BasketCompletionResponse,
    HistoricalContext,
    NumericalCertificate,
    ProductSearchResult,
    Recommendation,
    RetailContext,
    SegmentResponse,
    SizeProbability,
)


ROOT = Path(__file__).resolve().parents[1]
VERSION4 = ROOT / "scripts" / "version4"
if str(VERSION4) not in sys.path:
    sys.path.insert(0, str(VERSION4))
os.environ.setdefault("V3_AFFINITY", "1")
# The default checkpoint is the ERIM category-partition refit.  Entry points select its
# model-data bundle with retail_api.runtime.apply_default_data_root() before importing
# this module; load_checkpoint rejects a checkpoint whose fingerprint does not match.

from checkpoint_io import load_checkpoint  # noqa: E402
from conditional_basket import conditional_completion_quadrature  # noqa: E402
from data import build  # noqa: E402
from features import Features  # noqa: E402
from fit import Batcher  # noqa: E402
from provenance import file_sha256, model_data_root  # noqa: E402
from ragged import RaggedIndex, smolyak_grid  # noqa: E402


torch.set_default_dtype(torch.float64)

DEFAULT_CHECKPOINT = (
    ROOT / "artifacts/erim_category_refit/full/artifacts/candidate_rank1.pt")
DEFAULT_COMPLETION_AUDIT = (
    ROOT / "artifacts/erim_category_refit/retail_application/"
    "real_basket_completion_corrected.json")
DEFAULT_SEGMENT_REPORT = (
    ROOT / "artifacts/erim_category_refit/full/reports/customer_segments.json")


def configured_path(environment: str, default: Path) -> Path:
    return Path(os.environ.get(environment, str(default))).expanduser().resolve()


def validate_context_range(day: int, week: int, *, n_days: int,
                           week_min: int, week_max: int) -> None:
    """Reject contexts the fitted price and promotion panels cannot index."""
    if not 0 <= int(day) < int(n_days):
        raise RetailAPIError(
            f"day must lie in 0..{int(n_days) - 1} for the fitted price panel",
            code="context_out_of_range")
    if not int(week_min) <= int(week) <= int(week_max):
        raise RetailAPIError(
            f"week must lie in {int(week_min)}..{int(week_max)} for the fitted "
            "promotion coverage", code="context_out_of_range")


def certified_levels(completion_audit: dict, active_rank: int) -> list[int]:
    """The Smolyak levels the completion audit certified for this checkpoint.

    Audits written before the offset was recorded used active_rank + 2, 3 and 4.
    """
    certified = completion_audit.get("numerical_certification", {})
    offset = int(certified.get("level_offset", 2))
    levels = [active_rank + offset + step for step in (0, 1, 2)]
    if certified.get("levels") is not None and list(certified["levels"]) != levels:
        raise RetailAPIError(
            "completion audit levels do not match the checkpoint's active rank",
            status_code=503, code="artifact_lineage_mismatch")
    return levels


def padded_rule(model, active_rank: int, level: int):
    active, weights = smolyak_grid(active_rank, level)
    nodes = torch.zeros(len(weights), model.Kz, dtype=model.phi.dtype)
    nodes[:, :active_rank] = active
    return nodes, weights


class RetailModelService:
    """Loaded model plus evidence and ID contracts used by every endpoint."""

    STOP_TOLERANCE = 1e-4
    EXPECTED_SIZE_TOLERANCE = .02
    SIZE_PROBABILITY_TOLERANCE = 1e-4
    ITEM_INCIDENCE_TOLERANCE = 1e-4

    def __init__(self, checkpoint: Path | None = None,
                 completion_audit: Path | None = None,
                 segment_report: Path | None = None,
                 threads: int | None = None):
        self.checkpoint = (checkpoint or configured_path(
            "RETAIL_API_CHECKPOINT", DEFAULT_CHECKPOINT)).resolve()
        self.completion_audit_path = (completion_audit or configured_path(
            "RETAIL_API_COMPLETION_AUDIT", DEFAULT_COMPLETION_AUDIT)).resolve()
        self.segment_report_path = (segment_report or configured_path(
            "RETAIL_API_SEGMENT_REPORT", DEFAULT_SEGMENT_REPORT)).resolve()
        for path in (self.checkpoint, self.completion_audit_path,
                     self.segment_report_path):
            if not path.is_file():
                raise RetailAPIError(
                    f"required artifact is missing: {path}", status_code=503,
                    code="artifact_missing")
        torch.set_num_threads(threads or int(os.environ.get("RETAIL_API_THREADS", "4")))
        self.data = build()
        self.model, self.checkpoint_blob, self.meta = load_checkpoint(
            self.checkpoint, self.data,
            required_capabilities=("conditional_nonempty_incidence", "gram_interactions"))
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.checkpoint_sha256 = file_sha256(self.checkpoint)
        self.features = Features(
            int(self.data["n_item"]), int(self.data["n_store"]), include_recency=False)
        self.batcher = Batcher(
            self.data, self.features, int(self.meta["nmax"]), include_recency=False)
        self._lock = threading.RLock()
        self._load_evidence()
        self._load_products()
        self._load_segments()
        singular = torch.linalg.svdvals(self.model.phi)
        self.active_rank = int((singular > singular[0] * 1e-10).sum())
        self.levels = certified_levels(self.completion_audit, self.active_rank)
        self.rules = [padded_rule(self.model, self.active_rank, level)
                      for level in self.levels]

    def _load_evidence(self):
        self.completion_audit = json.loads(self.completion_audit_path.read_text())
        audit = self.completion_audit
        if audit.get("status") != "passed":
            raise RetailAPIError(
                "basket-completion audit did not pass", status_code=503,
                code="evidence_gate_failed")
        if audit.get("checkpoint_sha256") != self.checkpoint_sha256:
            raise RetailAPIError(
                "basket-completion audit belongs to another checkpoint",
                status_code=503, code="artifact_lineage_mismatch")
        if audit.get("data_fingerprint_sha256") != self.checkpoint_blob.get(
                "data_fingerprint_sha256"):
            raise RetailAPIError(
                "basket-completion audit belongs to another dataset",
                status_code=503, code="artifact_lineage_mismatch")
        gates = audit.get("numerical_certification", {}).get("gates", {})
        if not gates or not all(gates.values()):
            raise RetailAPIError(
                "basket-completion numerical gates are incomplete or failed",
                status_code=503, code="evidence_gate_failed")

    def _load_products(self):
        products = pd.read_parquet(model_data_root(ROOT) / "basket_input" / "items.parquet")
        products = products.sort_values("item_id", kind="stable").reset_index(drop=True)
        if not np.array_equal(products.item_id.to_numpy(),
                              np.arange(int(self.data["n_item"]))):
            raise RetailAPIError(
                "product metadata does not match model item indices",
                status_code=503, code="artifact_lineage_mismatch")
        self.products = products
        self.product_search_text = (
            products.PRODUCT_ID.astype(str) + " "
            + products.DEPARTMENT.fillna("").astype(str) + " "
            + products.COMMODITY_DESC.fillna("").astype(str) + " "
            + products.SUB_COMMODITY_DESC.fillna("").astype(str) + " "
            + products.BRAND.fillna("").astype(str)).str.casefold()
        self.external_to_internal = {
            int(row.PRODUCT_ID): int(row.item_id) for row in products.itertuples()}
        self.internal_to_external = products.PRODUCT_ID.to_numpy(np.int64)

    def _load_segments(self):
        report = json.loads(self.segment_report_path.read_text())
        if report.get("checkpoint_sha256") != self.checkpoint_sha256:
            raise RetailAPIError(
                "segment report belongs to another checkpoint", status_code=503,
                code="artifact_lineage_mismatch")
        configured = Path(report["assignments"])
        candidates = [
            configured,
            self.checkpoint.parent / configured.name,
            DEFAULT_CHECKPOINT.parent / configured.name,
        ]
        assignment_path = next((path for path in candidates if path.is_file()), None)
        if assignment_path is None:
            raise RetailAPIError(
                "segment assignments are missing", status_code=503,
                code="artifact_missing")
        if file_sha256(assignment_path) != report.get("assignments_sha256"):
            raise RetailAPIError(
                "segment assignment digest mismatch", status_code=503,
                code="artifact_lineage_mismatch")
        values = np.load(assignment_path)
        household = values["household"].astype(np.int64)
        if not np.array_equal(household, np.arange(int(self.data["n_user"]))):
            raise RetailAPIError(
                "segment household indices do not match the fitted cohort",
                status_code=503, code="artifact_lineage_mismatch")
        self.segment = values["segment"].astype(np.int64)
        self.segment_rows = {int(row["segment"]): row for row in report["segments"]}
        chosen = str(report["chosen_segments"])
        selection = report.get("candidate_selection", {}).get(chosen, {})
        if selection.get("accepted_minimum_5pct") is not True:
            raise RetailAPIError(
                "selected segmentation did not pass its minimum-size gate",
                status_code=503, code="evidence_gate_failed")
        self.segment_report = report
        self.segment_assignment_path = assignment_path

    def _product(self, internal: int):
        row = self.products.iloc[int(internal)]
        return {
            "product_id": int(row.PRODUCT_ID),
            "internal_item_index": int(internal),
            "department": str(row.DEPARTMENT),
            "commodity": str(row.COMMODITY_DESC),
            "sub_commodity": str(row.SUB_COMMODITY_DESC),
            "brand": str(row.BRAND),
        }

    def _resolve_context(self, context: HistoricalContext | RetailContext):
        if isinstance(context, HistoricalContext):
            trip = context.trip_index
            if trip >= len(self.data["trip_user"]):
                raise RetailAPIError("trip_index is outside the fitted dataset")
            ix, ctx, _line_ctx, house, *_ = self.batcher.make(
                np.asarray([trip], dtype=np.int64))
            return ix, ctx, house

        if context.household_index >= int(self.data["n_user"]):
            raise RetailAPIError("household_index is outside the fitted cohort")
        if context.store_index >= int(self.data["n_store"]):
            raise RetailAPIError("store_index is outside the fitted store set")
        validate_context_range(
            context.day, context.week, n_days=int(self.features.dev.shape[1]),
            week_min=self.features.promo_week_min, week_max=self.features.promo_week_max)
        C = int(self.data["n_cat"])
        ptr, store_items = self.data["store_cat_ptr"], self.data["store_items"]
        item_blocks, row_of, row_trip, row_cat = [], [], [], []
        row = 0
        for category in range(C):
            key = context.store_index * C + category
            lo, hi = int(ptr[key]), int(ptr[key + 1])
            if hi == lo:
                continue
            item_blocks.append(store_items[lo:hi])
            row_of.append(np.full(hi-lo, row, dtype=np.int64))
            row_trip.append(0); row_cat.append(category); row += 1
        if not item_blocks:
            raise RetailAPIError("store has no declared assortment")
        ix = RaggedIndex(
            np.concatenate(item_blocks), np.concatenate(row_of),
            np.asarray(row_trip, dtype=np.int64),
            np.asarray(row_cat, dtype=np.int64), 1)
        store = torch.full_like(ix.item, context.store_index)
        day = torch.full_like(ix.item, context.day)
        week = torch.full_like(ix.item, context.week)
        dlp, disp, mail = self.features.gather(ix.item, store, day, week)
        dbar = dlp.double().mean().reshape(1)
        ctx = {
            "dlp_bar": dbar,
            "dlp": dlp.double(),
            "disp": disp.double(),
            "mail": mail.double(),
            "week": (week - 1) % 52,
            "store": store,
        }
        if self.features.availability_enabled:
            ctx["log_avail"] = self.features.log_availability(ix.item, store, week)
        house = torch.as_tensor([context.household_index], dtype=torch.long)
        return ix, ctx, house

    def _size_log_weight(self, protocol: str, revealed_count: int):
        if protocol == "literal_cart":
            return None
        values = []
        for additional in range(self.model.nmax + 1):
            total = revealed_count + additional
            values.append(
                -math.log(math.comb(total, revealed_count))
                if total <= self.model.nmax else -float("inf"))
        return torch.as_tensor(values, dtype=self.model.phi.dtype)

    @staticmethod
    def _result_gaps(left, right):
        return {
            "stop": float((left.stop_probability-right.stop_probability).abs().max()),
            "expected": float((left.expected_additional_items
                               - right.expected_additional_items).abs().max()),
            "size": float((left.completion_size_probability
                           - right.completion_size_probability).abs().max()),
            "incidence": float((left.item_incidence-right.item_incidence).abs().max()),
        }

    def _within_tolerance(self, gap):
        return (gap["stop"] <= self.STOP_TOLERANCE
                and gap["expected"] <= self.EXPECTED_SIZE_TOLERANCE
                and gap["size"] <= self.SIZE_PROBABILITY_TOLERANCE
                and gap["incidence"] <= self.ITEM_INCIDENCE_TOLERANCE)

    def complete(self, request: BasketCompletionRequest) -> BasketCompletionResponse:
        internal = []
        missing = []
        for product_id in request.revealed_product_ids:
            item = self.external_to_internal.get(int(product_id))
            if item is None:
                missing.append(int(product_id))
            else:
                internal.append(item)
        if missing:
            raise RetailAPIError(
                f"products are outside the fitted catalogue: {missing}",
                code="unknown_product")
        if len(internal) > self.model.nmax:
            raise RetailAPIError("revealed cart exceeds model size support")
        if request.protocol == "uniform_random_subset" and len(internal) != 2:
            raise RetailAPIError(
                "the real-data uniform-random-subset audit supports exactly two "
                "revealed products", code="unsupported_anchor_size")
        size_weight = self._size_log_weight(request.protocol, len(internal))
        with self._lock:
            ix, ctx, house = self._resolve_context(request.context)
            self.model.house, self.model.ctx = house, ctx
            slot_b = self.model.b_flat(ix).detach()
            try:
                low = conditional_completion_quadrature(
                    self.model, ix, slot_b, [internal], *self.rules[0],
                    completion_size_log_weight=size_weight)
                high = conditional_completion_quadrature(
                    self.model, ix, slot_b, [internal], *self.rules[1],
                    completion_size_log_weight=size_weight)
            except ValueError as exc:
                raise RetailAPIError(str(exc)) from exc
            gap = self._result_gaps(low, high)
            selected, selected_rule = high, 1
            used_followup = not self._within_tolerance(gap)
            if used_followup:
                follow = conditional_completion_quadrature(
                    self.model, ix, slot_b, [internal], *self.rules[2],
                    completion_size_log_weight=size_weight)
                gap = self._result_gaps(high, follow)
                if not self._within_tolerance(gap):
                    raise RetailAPIError(
                        f"adjacent quadrature precision gate failed: {gap}",
                        status_code=503, code="numerical_gate_failed")
                selected, selected_rule = follow, 2

            size = selected.completion_size_probability[0].detach().numpy()
            incidence = selected.item_incidence[0].detach().numpy()
            expected = float(selected.expected_additional_items[0])
            identity_gap = abs(float(incidence.sum()) - expected)
            if identity_gap > 1e-8 or incidence.min() < -1e-8 \
                    or incidence.max() > 1 + 1e-8:
                raise RetailAPIError(
                    "conditional item probabilities failed algebraic bounds",
                    status_code=503, code="numerical_gate_failed")
            incidence = np.clip(incidence, 0.0, 1.0)
            available = ix.item.detach().numpy()
            candidates = available[~np.isin(available, internal)].astype(np.int64)
            score = incidence[candidates]
            external = self.internal_to_external[candidates]
            order = np.lexsort((external, -score))[:request.top_k]
            recommendations = [Recommendation(
                **self._product(int(candidates[position])),
                probability_in_completion=float(score[position]))
                for position in order]

        maximum_additional = self.model.nmax - len(internal)
        distribution = [SizeProbability(
            additional_items=index, probability=float(size[index]))
            for index in range(maximum_additional + 1)]
        if request.protocol == "literal_cart":
            estimand = "P(T | revealed cart A is a subset of the eventual basket, x)"
            evidence_status = "synthetically_verified_real_pair_incidence_size_weak"
            limitation = (
                "Set-conditional scenario; current data do not validate chronological "
                "scan-order completion, and the pair-incidence size audit was weak.")
        else:
            estimand = (
                "P(T | A was uniformly selected from equal-size subsets of the final "
                "basket, x)")
            evidence_status = "passed_256_context_retrospective_masking_audit"
            limitation = (
                "Validated retrospective masking protocol; do not interpret it as a "
                "chronological next-scan or causal recommendation effect.")
        return BasketCompletionResponse(
            estimand=estimand,
            protocol=request.protocol,
            evidence_status=evidence_status,
            checkpoint_sha256=self.checkpoint_sha256,
            revealed_product_ids=request.revealed_product_ids,
            stop_probability=float(size[0]),
            expected_additional_items=expected,
            size_distribution=distribution,
            recommendations=recommendations,
            numerical_certificate=NumericalCertificate(
                selected_level=self.levels[selected_rule],
                selected_nodes=len(self.rules[selected_rule][1]),
                used_followup=used_followup,
                adjacent_stop_probability_gap=gap["stop"],
                adjacent_expected_size_gap=gap["expected"],
                adjacent_size_probability_gap=gap["size"],
                adjacent_item_incidence_gap=gap["incidence"],
                item_incidence_sum_identity_gap=identity_gap,
            ),
            limitation=limitation,
        )

    def household_segment(self, household_index: int) -> SegmentResponse:
        if household_index < 0 or household_index >= len(self.segment):
            raise RetailAPIError("household_index is outside the fitted cohort")
        segment = int(self.segment[household_index])
        row = self.segment_rows[segment]
        return SegmentResponse(
            household_index=household_index,
            segment=segment,
            label=str(row["label"]),
            segment_households=int(row["households"]),
            mean_price_coefficient=float(row["mean_price_coefficient"]),
            evidence_status="descriptive_stratification_only",
            limitation=(
                "Segment labels describe fitted taste and price surfaces; they do not "
                "identify treatment response or justify differential offers."),
        )

    def search_products(self, query: str, limit: int) -> list[ProductSearchResult]:
        query = query.strip().casefold()
        if not query:
            raise RetailAPIError("q must contain a search term")
        rows = self.products[
            self.product_search_text.str.contains(query, regex=False)].head(limit)
        return [ProductSearchResult(**self._product(int(row.item_id)))
                for row in rows.itertuples()]

    def model_data_contract(self):
        """The served bundle's identity, substitution groups and availability contract."""
        basket_input = model_data_root(ROOT) / "basket_input"
        meta = json.loads((basket_input / "meta.json").read_text())
        contract = meta.get("availability_contract") or {"enabled": False}
        groups_path = basket_input / "affinity_manifest.json"
        groups = json.loads(groups_path.read_text()) if groups_path.is_file() else {}
        return {
            "dataset": meta.get("dataset", "dunnhumby"),
            "substitution_groups": {
                "partition": groups.get("partition", "co_purchase_affinity"),
                "groups": groups.get("n_groups"),
            },
            "availability": {
                "enabled": bool(contract.get("enabled")),
                "rule": contract.get("rule", "declared_catalogue"),
                "unconfirmed_product_weight": contract.get("floor", 1.0),
            },
        }

    def capabilities(self):
        masked = self.completion_audit["shopper_uniform_random_pair_mask_target"]
        return {
            "api_version": "1.0.0",
            "checkpoint_sha256": self.checkpoint_sha256,
            "data_fingerprint_sha256": self.checkpoint_blob["data_fingerprint_sha256"],
            "model_data": self.model_data_contract(),
            "available": {
                "basket_completion": {
                    "status": "validated_for_uniform_random_subset_masking",
                    "contexts": self.completion_audit["contexts"],
                    "mae_items": masked["model"]["mae_items"],
                    "baseline_mae_items": masked[
                        "training_only_empirical_size_baseline"]["mae_items"],
                },
                "cross_sell": {
                    "status": "promising_offline",
                    "mrr": masked["cross_sell_retrieval"]["model"]["mrr"],
                    "popularity_mrr": masked[
                        "cross_sell_retrieval"]["training_popularity"]["mrr"],
                },
                "customer_segmentation": {
                    "status": "descriptive_only",
                    "segments": self.segment_report["chosen_segments"],
                },
            },
            "unavailable": {
                "causal_price_optimization": "causal effects and profit were not validated",
                "stockout_substitution": (
                    "store availability is inferred from sales and lost-demand labels are absent"),
                "promotion_policy": "incremental lift, cost, and margin are not identified",
                "assortment_optimization": "historical availability interventions are absent",
                "total_demand_forecasting": "visit incidence is outside the fitted law",
                "chronological_stopping": "within-trip scan order is absent",
                "personalized_bundle_policy": "only a small matched-negative proxy passed",
            },
        }
