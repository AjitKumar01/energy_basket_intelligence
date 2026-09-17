#!/usr/bin/env python3
"""Exercise every available API capability against the real audited artifacts."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retail_api.runtime import apply_default_data_root  # noqa: E402

apply_default_data_root()

from retail_api.app import app, get_service  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=Path("artifacts/retail_api_smoke.json"))
    args = parser.parse_args()
    service = get_service()
    audited = service.completion_audit["per_context"][0]
    trip = int(audited["trip"])
    revealed = [int(service.internal_to_external[item])
                for item in audited["revealed_items"]]
    with TestClient(app) as client:
        live = client.get("/live")
        ready = client.get("/ready")
        capabilities = client.get("/v1/capabilities")
        products = client.get(
            "/v1/products/search", params={"q": str(revealed[0]), "limit": 3})
        segment = client.get(
            f"/v1/households/{audited['household']}/segment")
        completion = {}
        for protocol in ("uniform_random_subset", "literal_cart"):
            response = client.post("/v1/baskets/complete", json={
                "context": {"kind": "historical_trip", "trip_index": trip},
                "revealed_product_ids": revealed,
                "top_k": 5,
                "protocol": protocol,
            })
            if response.status_code != 200:
                raise RuntimeError(
                    f"{protocol} failed: {response.status_code} {response.text}")
            value = response.json()
            if abs(sum(row["probability"] for row in value["size_distribution"])-1) > 1e-9:
                raise RuntimeError(f"{protocol} size distribution is not normalized")
            completion[protocol] = {
                "stop_probability": value["stop_probability"],
                "expected_additional_items": value["expected_additional_items"],
                "recommendations": value["recommendations"],
                "numerical_certificate": value["numerical_certificate"],
                "limitation": value["limitation"],
            }
        # The deployable explicit context path must reproduce the historical-context
        # adapter exactly when given the same household/store/day/week fields.
        explicit_response = client.post("/v1/baskets/complete", json={
            "context": {
                "kind": "retail_context",
                "household_index": int(service.data["trip_user"][trip]),
                "store_index": int(service.data["trip_store"][trip]),
                "day": int(service.data["trip_day"][trip]),
                "week": int(service.data["trip_week"][trip]),
            },
            "revealed_product_ids": revealed,
            "top_k": 5,
            "protocol": "uniform_random_subset",
        })
        if explicit_response.status_code != 200:
            raise RuntimeError(
                f"explicit retail context failed: {explicit_response.status_code} "
                f"{explicit_response.text}")
        explicit = explicit_response.json()
        historical = completion["uniform_random_subset"]
        context_gap = max(
            abs(explicit["stop_probability"]-historical["stop_probability"]),
            abs(explicit["expected_additional_items"]
                - historical["expected_additional_items"]),
            max(abs(left["probability_in_completion"]
                    - right["probability_in_completion"])
                for left, right in zip(
                    explicit["recommendations"], historical["recommendations"])),
        )
        if context_gap > 1e-12:
            raise RuntimeError(
                f"explicit and historical context paths differ by {context_gap}")
        for name, response in {
            "live": live, "ready": ready, "capabilities": capabilities,
            "products": products, "segment": segment,
        }.items():
            if response.status_code != 200:
                raise RuntimeError(f"{name} failed: {response.status_code} {response.text}")
        result = {
            "status": "passed",
            "checkpoint_sha256": service.checkpoint_sha256,
            "historical_trip_index": trip,
            "revealed_product_ids": revealed,
            "ready": ready.json(),
            "capabilities": capabilities.json(),
            "product_search": products.json(),
            "segment": segment.json(),
            "completion": completion,
            "explicit_vs_historical_context_maximum_gap": context_gap,
        }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
