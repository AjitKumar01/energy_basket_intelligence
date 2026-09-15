#!/usr/bin/env python3
"""Measure cold and warm HTTP latency for every retail API endpoint."""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx
import numpy as np


def summary(samples, sizes, statuses):
    values = np.asarray(samples, dtype=np.float64)
    return {
        "requests": int(len(values)),
        "status_codes": sorted(set(statuses)),
        "latency_ms": {
            "minimum": float(values.min()),
            "mean": float(values.mean()),
            "median_p50": float(np.median(values)),
            "p95": float(np.quantile(values, .95)),
            "maximum": float(values.max()),
            "standard_deviation": float(values.std()),
        },
        "response_bytes": {
            "median": float(statistics.median(sizes)),
            "maximum": int(max(sizes)),
        },
    }


def measure(client, method, path, *, repeats, expected=200, **kwargs):
    samples, sizes, statuses = [], [], []
    for _ in range(repeats):
        started = time.perf_counter_ns()
        response = client.request(method, path, **kwargs)
        elapsed = (time.perf_counter_ns() - started) / 1e6
        samples.append(elapsed); sizes.append(len(response.content))
        statuses.append(response.status_code)
        if response.status_code != expected:
            raise RuntimeError(
                f"{method} {path}: expected {expected}, got "
                f"{response.status_code}: {response.text}")
    return summary(samples, sizes, statuses)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8011")
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/retail_api_latency.json"))
    parser.add_argument("--cheap-repeats", type=int, default=30)
    parser.add_argument("--inference-repeats", type=int, default=10)
    parser.add_argument("--smoke-artifact", type=Path,
                        default=Path("artifacts/retail_api_smoke.json"))
    args = parser.parse_args()
    smoke = json.loads(args.smoke_artifact.read_text())
    trip = smoke["historical_trip_index"]
    household = smoke["segment"]["household_index"]
    revealed = smoke["revealed_product_ids"]
    request = {
        "context": {"kind": "historical_trip", "trip_index": trip},
        "revealed_product_ids": revealed,
        "top_k": 20,
    }
    result = {
        "status": "passed",
        "method": "sequential HTTP/1.1 requests to one local uvicorn worker",
        "base_url": args.base_url,
        "cheap_repeats": args.cheap_repeats,
        "inference_repeats": args.inference_repeats,
        "historical_trip_index": trip,
        "revealed_product_ids": revealed,
        "measurements": {},
    }
    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        result["measurements"]["live"] = measure(
            client, "GET", "/live", repeats=args.cheap_repeats)
        # This is intentionally the first model-dependent request in a fresh server.
        result["measurements"]["ready_cold"] = measure(
            client, "GET", "/ready", repeats=1)
        result["measurements"]["ready_warm"] = measure(
            client, "GET", "/ready", repeats=args.cheap_repeats)
        result["measurements"]["capabilities"] = measure(
            client, "GET", "/v1/capabilities", repeats=args.cheap_repeats)
        result["measurements"]["openapi_schema"] = measure(
            client, "GET", "/openapi.json", repeats=args.cheap_repeats)
        result["measurements"]["swagger_docs"] = measure(
            client, "GET", "/docs", repeats=args.cheap_repeats)
        result["measurements"]["redoc_docs"] = measure(
            client, "GET", "/redoc", repeats=args.cheap_repeats)
        result["measurements"]["product_search"] = measure(
            client, "GET", "/v1/products/search", repeats=args.cheap_repeats,
            params={"q": str(revealed[0]), "limit": 20})
        result["measurements"]["household_segment"] = measure(
            client, "GET", f"/v1/households/{household}/segment",
            repeats=args.cheap_repeats)
        for protocol in ("uniform_random_subset", "literal_cart"):
            payload = {**request, "protocol": protocol}
            result["measurements"][f"basket_complete_{protocol}_first"] = measure(
                client, "POST", "/v1/baskets/complete", repeats=1, json=payload)
            result["measurements"][f"basket_complete_{protocol}_warm"] = measure(
                client, "POST", "/v1/baskets/complete",
                repeats=args.inference_repeats, json=payload)
        invalid = {**request, "revealed_product_ids": revealed[:1],
                   "protocol": "uniform_random_subset"}
        result["measurements"]["unsupported_mask_size_rejection"] = measure(
            client, "POST", "/v1/baskets/complete", repeats=args.cheap_repeats,
            expected=400, json=invalid)
    output = args.output.resolve(); output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
