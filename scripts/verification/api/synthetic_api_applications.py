#!/usr/bin/env python3
"""Exercise every retail API capability on the known-truth synthetic retailer.

The synthetic world (scripts/synthetic/generate_canonical_world.py) has a known generating
law, so each API answer can be scored against the truth as well as against what shoppers
actually did. Six applications, one per capability the API exposes:

  A. operations      is the service up and serving the checkpoint it claims? (/live, /ready)
  B. governance      which decisions is it allowed to support? (/v1/capabilities)
  C. catalogue       find a product to put in a request (/v1/products/search)
  D. checkout cross-sell   what else will this shopper buy? (/v1/baskets/complete, literal cart)
  E. "one more item?"      will the shopper add anything, and how much? (masked-pair protocol)
  F. segmentation    which reporting segment is this household in? (/v1/households/{id}/segment)
  G. refusals        requests the service must reject rather than guess

Scored against truth: the oracle add-one score b_j + sum_{k in cart} (phi_j.phi_k -
rho_c 1[same category]) ranks the true next product; truth segments come from truth.npz.
Scored against outcomes: held-out purchases from the test split.

Run from the repository root:
  python scripts/verification/api/synthetic_api_applications.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = {
    "RETAIL_API_DATA_ROOT": ROOT / "data/synthetic_capability_world/model_input_category",
    "RETAIL_API_CHECKPOINT": ROOT / "artifacts/synthetic_capability_test_category/full/artifacts/candidate_rank1.pt",
    "RETAIL_API_COMPLETION_AUDIT": ROOT / "artifacts/synthetic_capability_test_category/retail_application/real_basket_completion_corrected.json",
    "RETAIL_API_SEGMENT_REPORT": ROOT / "artifacts/synthetic_capability_test_category/full/reports/customer_segments.json",
}
for key, value in DEFAULTS.items():
    os.environ.setdefault(key, str(value))

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts/version4"))
sys.path.insert(0, str(ROOT / "scripts/synthetic"))

from retail_api.runtime import apply_default_data_root  # noqa: E402

BUNDLE = apply_default_data_root()

from fastapi.testclient import TestClient  # noqa: E402
from retail_api.app import app, get_service  # noqa: E402


def truth_tables(world: Path):
    t = np.load(world / "truth.npz")
    pair = t["phi"] @ t["phi"].T - np.where(t["category"][:, None] == t["category"][None],
                                            t["rho_c"][t["category"]][:, None], 0.0)
    np.fill_diagonal(pair, 0.0)
    return t, pair


def oracle_utilities(t, basket_ids: np.ndarray):
    """Truth utilities b [B, J] for canonical basket ids, -inf where unstocked."""
    from basket_world import World, utilities
    world = World(lam=t["lam"], alpha=t["alpha"], phi=t["phi"], category=t["category"],
                  rho_c=t["rho_c"], rho0=t["rho0"], price_sensitivity=t["price_sensitivity"])
    hh, store, period = (t["trip_household"][basket_ids], t["trip_store"][basket_ids],
                         t["trip_period"][basket_ids])
    available = t["first_stocked"][:, store].T <= period[:, None]
    return utilities(world, t["taste"][hh], t["size_shift"][hh], t["dlp"][:, period - 1].T, available)


def rank_of(scores: np.ndarray, target: int) -> float:
    return float((scores > scores[target]).sum() + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", type=Path, default=ROOT / "data/synthetic_capability_world")
    parser.add_argument("--cases", type=int, default=60)
    parser.add_argument("--households", type=int, default=300)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/synthetic_api_demo/report.json")
    args = parser.parse_args()
    rng = np.random.default_rng(11)
    service = get_service()
    report: dict = {"bundle": str(BUNDLE), "checkpoint": os.environ["RETAIL_API_CHECKPOINT"]}

    baskets = pd.read_parquet(BUNDLE / "basket_input" / "baskets.parquet",
                              columns=["BASKET_ID", "user_id", "WEEK_NO", "DAY", "store_id", "item_id", "split"])
    items = pd.read_parquet(BUNDLE / "basket_input" / "items.parquet").sort_values("item_id")
    external = items.PRODUCT_ID.to_numpy()
    t, true_pair = truth_tables(args.world)
    train_counts = np.bincount(baskets.item_id[baskets.split == "train"].to_numpy(),
                               minlength=len(items))

    with TestClient(app) as client:
        # ---- A. operations ------------------------------------------------------------
        live, ready = client.get("/live"), client.get("/ready")
        fingerprint = json.loads((BUNDLE / "basket_input" / "model_data_fingerprint.json").read_text())
        report["A_operations"] = {
            "question": "is the service up, and is it serving the data bundle it claims?",
            "live": live.json(), "ready_status": ready.status_code,
            "checkpoint_sha256": ready.json()["checkpoint_sha256"],
            "data_fingerprint_matches_bundle":
                ready.json()["data_fingerprint_sha256"] == fingerprint["fingerprint_sha256"],
        }

        # ---- B. governance ------------------------------------------------------------
        capabilities = client.get("/v1/capabilities").json()
        report["B_governance"] = {
            "question": "which decisions may this model support, and which are out of scope?",
            "model_data": capabilities["model_data"],
            "available": {k: v.get("status") for k, v in capabilities["available"].items()},
            "available_decisions": {k: {"status": v["status"], "evidence": v["evidence"]}
                                    for k, v in capabilities["available_decisions"].items()},
            "unavailable": capabilities["unavailable"],
            "capability_verdicts": capabilities["capability_verdicts"],
            "offline_cross_sell_mrr": capabilities["available"]["cross_sell"]["mrr"],
        }

        # ---- C. catalogue -------------------------------------------------------------
        found = client.get("/v1/products/search", params={"q": "bakery", "limit": 5}).json()
        report["C_catalogue"] = {
            "question": "which product ids does the shopper's cart map to?",
            "results": found[:3], "returned": len(found),
            "all_in_queried_category": all(r["commodity"] == "bakery" for r in found),
        }

        # ---- D. checkout cross-sell (literal cart) ------------------------------------
        test_trips = np.flatnonzero(np.asarray(service.data["trip_split"]) == 2)
        sizes = np.asarray(service.data["trip_nlines"]).astype(int)
        usable = test_trips[sizes[test_trips] >= 3]
        chosen = rng.choice(usable, size=min(args.cases, len(usable)), replace=False)
        basket_of_trip = np.sort(baskets.BASKET_ID.unique())
        truth_b = oracle_utilities(t, basket_of_trip[chosen])
        rows, latency = [], []
        for position, trip in enumerate(chosen):
            bought = np.flatnonzero(t["memberships"][basket_of_trip[trip]])
            hidden = int(rng.choice(bought))
            cart = [int(x) for x in bought if x != hidden]
            start = time.perf_counter()
            response = client.post("/v1/baskets/complete", json={
                "context": {"kind": "historical_trip", "trip_index": int(trip)},
                "revealed_product_ids": [int(external[j]) for j in cart],
                "top_k": 5, "protocol": "literal_cart"})
            latency.append(time.perf_counter() - start)
            body = response.json()
            top = [r["internal_item_index"] for r in body["recommendations"]]
            score = truth_b[position] + true_pair[:, cart].sum(1)
            score[cart] = -np.inf
            oracle_top = list(np.argsort(-score)[:5])
            popularity = [int(j) for j in np.argsort(-train_counts) if j not in cart][:5]
            rows.append({
                "hit": hidden in top, "rank": (top.index(hidden) + 1) if hidden in top else None,
                "oracle_hit": hidden in oracle_top, "popularity_hit": hidden in popularity,
                "top5_overlap_with_oracle": len(set(top) & set(oracle_top)),
                "true_rank_of_api_top1": rank_of(score, top[0]),
                "stop_probability": body["stop_probability"],
                "expected_additional_items": body["expected_additional_items"],
            })
        hits = np.array([r["hit"] for r in rows])
        oracle_hits = np.mean([r["oracle_hit"] for r in rows])
        popularity_hits = np.mean([r["popularity_hit"] for r in rows])
        report["D_cross_sell"] = {
            "question": "a shopper's cart is on the belt: which product should the till suggest?",
            "cases": len(rows), "cart_size_mean": float(np.mean([sizes[t_] - 1 for t_ in chosen])),
            "hit_rate_at_5": float(hits.mean()),
            "mrr_at_5": float(np.mean([1 / r["rank"] if r["rank"] else 0.0 for r in rows])),
            "oracle_hit_rate_at_5": float(np.mean([r["oracle_hit"] for r in rows])),
            "popularity_hit_rate_at_5": float(np.mean([r["popularity_hit"] for r in rows])),
            "mean_top5_overlap_with_oracle": float(np.mean([r["top5_overlap_with_oracle"] for r in rows])),
            "share_of_attainable_lift_over_popularity": None,
            "median_true_rank_of_api_top_suggestion": float(np.median([r["true_rank_of_api_top1"] for r in rows])),
            "latency_ms": {"median": float(np.median(latency) * 1e3), "p95": float(np.quantile(latency, 0.95) * 1e3)},
        }
        report["D_cross_sell"]["share_of_attainable_lift_over_popularity"] = (
            float((hits.mean() - popularity_hits) / (oracle_hits - popularity_hits))
            if oracle_hits > popularity_hits else None)

        # ---- E. "will they add more?" (validated masked-pair protocol) ----------------
        pair_usable = test_trips[sizes[test_trips] >= 2]
        pair_trips = rng.choice(pair_usable, size=min(args.cases, len(pair_usable)), replace=False)
        masked = []
        for trip in pair_trips:
            bought = np.flatnonzero(t["memberships"][basket_of_trip[trip]])
            pair = [int(x) for x in rng.choice(bought, size=2, replace=False)]
            remaining = [int(x) for x in bought if x not in pair]
            body = client.post("/v1/baskets/complete", json={
                "context": {"kind": "historical_trip", "trip_index": int(trip)},
                "revealed_product_ids": [int(external[j]) for j in pair],
                "top_k": 5, "protocol": "uniform_random_subset"}).json()
            top = [r["internal_item_index"] for r in body["recommendations"]]
            masked.append({
                "observed_additional": len(remaining),
                "expected_additional": body["expected_additional_items"],
                "stop_probability": body["stop_probability"],
                "observed_stop": len(remaining) == 0,
                "any_remaining_in_top5": bool(set(remaining) & set(top)),
                "certificate": body["numerical_certificate"]["selected_level"],
            })
        observed = np.array([m["observed_additional"] for m in masked], dtype=float)
        predicted = np.array([m["expected_additional"] for m in masked])
        stop_p = np.array([m["stop_probability"] for m in masked])
        stopped = np.array([m["observed_stop"] for m in masked], dtype=float)
        report["E_basket_size"] = {
            "question": "with two items scanned, will this shopper add more, and how many?",
            "cases": len(masked),
            "trips_with_exactly_two_items": int(sum(1 for m in masked if m["observed_additional"] == 0)),
            "observed_mean_additional": float(observed.mean()),
            "predicted_mean_additional": float(predicted.mean()),
            "mean_absolute_error_items": float(np.abs(observed - predicted).mean()),
            "stop_rate_observed": float(stopped.mean()), "stop_rate_predicted": float(stop_p.mean()),
            "stop_brier_score": float(np.mean((stop_p - stopped) ** 2)),
            "stop_brier_score_constant_rate_baseline": float(np.mean((stopped.mean() - stopped) ** 2)),
            "mean_absolute_error_constant_baseline": float(np.abs(observed - observed.mean()).mean()),
            "at_least_one_remaining_item_in_top5": float(np.mean([m["any_remaining_in_top5"] for m in masked])),
            "quadrature_levels_used": sorted({m["certificate"] for m in masked}),
        }

        # ---- F. segmentation ----------------------------------------------------------
        households = rng.choice(int(service.data["n_user"]), size=args.households, replace=False)
        assigned, labels = [], {}
        for household in households:
            body = client.get(f"/v1/households/{int(household)}/segment").json()
            assigned.append(body["segment"])
            labels[body["segment"]] = body["label"]
        from sklearn.metrics import adjusted_rand_score
        report["F_segmentation"] = {
            "question": "which reporting segment does this household belong to?",
            "households": len(households), "segments": sorted(labels),
            "labels": labels,
            "adjusted_rand_index_vs_truth": float(adjusted_rand_score(t["segment"][households], assigned)),
            "example": body,
        }

        # ---- G. refusals --------------------------------------------------------------
        refusals = {}
        trip = int(chosen[0])
        bought = [int(external[j]) for j in np.flatnonzero(t["memberships"][basket_of_trip[trip]])]
        checks = {
            "unknown_product": {"context": {"kind": "historical_trip", "trip_index": trip},
                                "revealed_product_ids": [999999]},
            "duplicate_products": {"context": {"kind": "historical_trip", "trip_index": trip},
                                   "revealed_product_ids": [bought[0], bought[0]]},
            "empty_cart": {"context": {"kind": "historical_trip", "trip_index": trip},
                           "revealed_product_ids": []},
            "wrong_anchor_size_for_audited_protocol": {
                "context": {"kind": "historical_trip", "trip_index": trip},
                "revealed_product_ids": bought[:1], "protocol": "uniform_random_subset"},
            "context_out_of_range": {"context": {"kind": "retail_context", "household_index": 0,
                                                 "store_index": 0, "day": 5, "week": 999},
                                     "revealed_product_ids": bought[:1]},
            "unknown_field": {"context": {"kind": "historical_trip", "trip_index": trip},
                              "revealed_product_ids": bought[:1], "discount": 0.2},
        }
        def rejection(response):
            body = response.json()
            if "error" in body:                       # service-level refusal
                return {"status": response.status_code, "code": body["error"]["code"],
                        "message": body["error"]["message"]}
            detail = body.get("detail")               # schema validation
            message = (detail[0].get("msg") if isinstance(detail, list) and detail
                       else str(detail))
            return {"status": response.status_code, "code": "schema_validation", "message": message}

        for name, payload in checks.items():
            refusals[name] = rejection(client.post("/v1/baskets/complete", json=payload))
        for name, path in (("unknown_household", "/v1/households/999999/segment"),
                           ("unknown_trip", None)):
            if path:
                refusals[name] = rejection(client.get(path))
        response = client.post("/v1/baskets/complete", json={
            "context": {"kind": "historical_trip", "trip_index": 10 ** 9},
            "revealed_product_ids": bought[:1]})
        refusals["unknown_trip"] = rejection(response)
        report["G_refusals"] = {
            "question": "does the service reject what it cannot answer instead of guessing?",
            "checks": refusals,
            "all_rejected": all(v["status"] >= 400 for v in refusals.values()),
        }

        # ---- H. live retail context (no historical trip) -------------------------------
        row = baskets[baskets.BASKET_ID == basket_of_trip[trip]].iloc[0]
        live_body = client.post("/v1/baskets/complete", json={
            "context": {"kind": "retail_context", "household_index": int(row.user_id),
                        "store_index": int(row.store_id), "day": int(row.DAY), "week": int(row.WEEK_NO)},
            "revealed_product_ids": bought[:2], "top_k": 5, "protocol": "literal_cart"}).json()
        historical_body = client.post("/v1/baskets/complete", json={
            "context": {"kind": "historical_trip", "trip_index": trip},
            "revealed_product_ids": bought[:2], "top_k": 5, "protocol": "literal_cart"}).json()
        report["H_live_context"] = {
            "question": "can the till ask about a basket that is not a recorded trip?",
            "same_top5": [r["product_id"] for r in live_body["recommendations"]]
                         == [r["product_id"] for r in historical_body["recommendations"]],
            "expected_additional_gap": abs(live_body["expected_additional_items"]
                                           - historical_body["expected_additional_items"]),
            "example_recommendations": live_body["recommendations"][:3],
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k.startswith(("A", "B", "C", "D", "E", "F", "G", "H"))},
                     indent=2)[:6000])


if __name__ == "__main__":
    main()
