"""FastAPI application exposing only evidence-gated retail capabilities."""
from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from fastapi import Depends, FastAPI, Query, Request
from fastapi.responses import JSONResponse

from . import __version__
from .errors import RetailAPIError
from .schemas import (
    BasketCompletionRequest,
    BasketCompletionResponse,
    PriceScenarioRequest,
    PriceScenarioResponse,
    ProductSearchResult,
    SegmentResponse,
)
from .service import RetailModelService


@lru_cache(maxsize=1)
def get_service() -> RetailModelService:
    return RetailModelService()


app = FastAPI(
    title="Energy Basket Intelligence API",
    version=__version__,
    description=(
        "Evidence-gated basket completion, cross-sell ranking, and descriptive "
        "segmentation from the fitted joint basket law."),
)


@app.exception_handler(RetailAPIError)
async def retail_error(_request: Request, exc: RetailAPIError):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "message": exc.message}},
    )


@app.get("/live", tags=["operations"])
def live():
    return {"status": "alive", "api_version": __version__}


@app.get("/ready", tags=["operations"])
def ready(service: Annotated[RetailModelService, Depends(get_service)]):
    return {
        "status": "ready",
        "checkpoint_sha256": service.checkpoint_sha256,
        "data_fingerprint_sha256": service.checkpoint_blob[
            "data_fingerprint_sha256"],
    }


@app.get("/v1/capabilities", tags=["evidence"])
def capabilities(service: Annotated[RetailModelService, Depends(get_service)]):
    return service.capabilities()


@app.get("/v1/products/search", response_model=list[ProductSearchResult],
         tags=["catalogue"])
def product_search(
        q: Annotated[str, Query(min_length=1)],
        limit: Annotated[int, Query(ge=1, le=100)] = 20,
        service: Annotated[RetailModelService, Depends(get_service)] = None):
    return service.search_products(q, limit)


@app.post("/v1/baskets/complete", response_model=BasketCompletionResponse,
          tags=["basket"])
def complete_basket(
        body: BasketCompletionRequest,
        service: Annotated[RetailModelService, Depends(get_service)]):
    return service.complete(body)


@app.post("/v1/baskets/price_scenario", response_model=PriceScenarioResponse,
          tags=["basket"])
def price_scenario(
        body: PriceScenarioRequest,
        service: Annotated[RetailModelService, Depends(get_service)]):
    """Purchase probabilities with and without a declared price change.

    Served only where the deployment's capability verdict claims causal price optimization;
    otherwise the request is refused with HTTP 403.
    """
    return service.price_scenario(body)


@app.get("/v1/households/{household_index}/segment",
         response_model=SegmentResponse, tags=["segmentation"])
def household_segment(
        household_index: int,
        service: Annotated[RetailModelService, Depends(get_service)]):
    return service.household_segment(household_index)
