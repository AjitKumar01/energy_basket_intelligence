"""Validated public request and response contracts."""
from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class HistoricalContext(StrictModel):
    kind: Literal["historical_trip"] = "historical_trip"
    trip_index: Annotated[int, Field(ge=0)]


class RetailContext(StrictModel):
    kind: Literal["retail_context"] = "retail_context"
    household_index: Annotated[int, Field(ge=0)]
    store_index: Annotated[int, Field(ge=0)]
    day: Annotated[int, Field(ge=0, le=711)]
    week: Annotated[int, Field(ge=9, le=101)]


Context = Annotated[HistoricalContext | RetailContext, Field(discriminator="kind")]


class BasketCompletionRequest(StrictModel):
    context: Context
    revealed_product_ids: Annotated[list[int], Field(min_length=1)]
    top_k: Annotated[int, Field(ge=1, le=100)] = 20
    protocol: Literal["literal_cart", "uniform_random_subset"] = "literal_cart"

    @model_validator(mode="after")
    def unique_revealed_products(self):
        if len(self.revealed_product_ids) != len(set(self.revealed_product_ids)):
            raise ValueError("revealed_product_ids must be unique")
        return self


class Recommendation(StrictModel):
    product_id: int
    internal_item_index: int
    probability_in_completion: float
    department: str
    commodity: str
    sub_commodity: str
    brand: str


class SizeProbability(StrictModel):
    additional_items: int
    probability: float


class NumericalCertificate(StrictModel):
    selected_level: int
    selected_nodes: int
    used_followup: bool
    adjacent_stop_probability_gap: float
    adjacent_expected_size_gap: float
    adjacent_size_probability_gap: float
    adjacent_item_incidence_gap: float
    item_incidence_sum_identity_gap: float


class BasketCompletionResponse(StrictModel):
    estimand: str
    protocol: str
    evidence_status: str
    checkpoint_sha256: str
    revealed_product_ids: list[int]
    stop_probability: float
    expected_additional_items: float
    size_distribution: list[SizeProbability]
    recommendations: list[Recommendation]
    numerical_certificate: NumericalCertificate
    limitation: str


class SegmentResponse(StrictModel):
    household_index: int
    segment: int
    label: str
    segment_households: int
    mean_price_coefficient: float
    evidence_status: str
    limitation: str


class ProductSearchResult(StrictModel):
    product_id: int
    internal_item_index: int
    department: str
    commodity: str
    sub_commodity: str
    brand: str
