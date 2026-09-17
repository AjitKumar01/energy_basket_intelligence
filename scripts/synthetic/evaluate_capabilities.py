#!/usr/bin/env python3
"""Compare a pipeline run on the synthetic world with the known truth, capability by capability.

Every oracle quantity is computed from the generating law (scripts/synthetic/basket_world.py)
on exactly the trips, products and actions the pipeline's own reports used:

  1. held-out likelihood          model log p(S|x) vs oracle log p(S|x) per test/validation trip
  2. structure recovery           effective pairwise interaction, product appeal, household
                                  taste and size propensity, price sensitivity
  3. availability                 bundle availability panel vs the true stock calendar
  4. customer segments            fitted segments vs true latent segments (adjusted Rand)
  5. recommendation               model add-one ranks vs oracle add-one ranks vs popularity
  6. price scenarios              own-product incidence and basket-size response
  7. generation / size            model and oracle expected basket size vs observed
  8. promotion policy             model vs oracle incremental post-discount sales per action,
                                  and oracle value of the policy the pipeline selected

Oracle expectations use d/dt log Z_+(b + t v) = E[sum_{j in S} v_j] with a central difference.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "synthetic"))
sys.path.insert(0, str(ROOT / "scripts" / "version4"))

from basket_world import World, energy, log_total, quadrature, utilities  # noqa: E402

STEP = 1e-4


def paired(delta: np.ndarray, cluster: np.ndarray) -> dict:
    from uncertainty import paired_score_summary
    s = paired_score_summary(np.asarray(delta, dtype=np.float64), np.asarray(cluster))
    return {"mean": s["mean"], "95_interval": s["95_interval"]}


class Oracle:
    def __init__(self, truth_path: Path, nodes: int):
        t = np.load(truth_path)
        self.t = t
        self.world = World(lam=t["lam"], alpha=t["alpha"], phi=t["phi"], category=t["category"],
                           rho_c=t["rho_c"], rho0=t["rho0"],
                           price_sensitivity=t["price_sensitivity"])
        self.grid, self.log_weight = quadrature(self.world.phi.shape[1], nodes)
        self.first_stocked = t["first_stocked"]
        self.dlp = t["dlp"]
        self.price = t["price"]

    def b(self, baskets: np.ndarray, price_change: np.ndarray | None = None) -> np.ndarray:
        """Truth utilities for canonical basket ids; price_change [B, J] in log price."""
        t = self.t
        hh, store, period = (t["trip_household"][baskets], t["trip_store"][baskets],
                             t["trip_period"][baskets])
        available = self.first_stocked[:, store].T <= period[:, None]
        dlp = self.dlp[:, period - 1].T
        if price_change is not None:
            dlp = dlp + price_change
        return utilities(self.world, t["taste"][hh], t["size_shift"][hh], dlp, available)

    def log_z(self, b: np.ndarray, chunk: int = 2048) -> np.ndarray:
        out = np.empty(len(b))
        for start in range(0, len(b), chunk):
            rows = b[start:start + chunk]
            values = np.empty((len(rows), len(self.grid)))
            for q, point in enumerate(self.grid):
                values[:, q] = log_total(self.world, rows, np.broadcast_to(point, (len(rows), len(point))))
            out[start:start + chunk] = np.logaddexp.reduce(values + self.log_weight[None, :], axis=1)
        return out

    def expectation(self, b: np.ndarray, v: np.ndarray) -> np.ndarray:
        """E[sum_{j in S} v_j] for each row (v [B, J]; unstocked products contribute 0)."""
        v = np.where(np.isfinite(b), v, 0.0)
        both = self.log_z(np.concatenate([b + STEP * v, b - STEP * v]))
        return (both[:len(b)] - both[len(b):]) / (2 * STEP)


def trip_to_basket(bundle: Path) -> np.ndarray:
    baskets = pd.read_parquet(bundle / "basket_input" / "baskets.parquet", columns=["BASKET_ID"])
    return np.sort(baskets.BASKET_ID.unique())


def midrank(scores: np.ndarray, position: int) -> float:
    target = scores[position]
    return float((scores > target).sum() + 0.5 * ((scores == target).sum() - 1) + 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--world", type=Path, default=ROOT / "data/synthetic_capability_world")
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/synthetic_capability_test/full")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/synthetic_capability_test/capability_report.json")
    parser.add_argument("--nodes", type=int, default=16)
    parser.add_argument("--bundle", type=Path,
                        help="model-data bundle the run was fitted on (default: <world>/model_input)")
    args = parser.parse_args()
    bundle = (args.bundle or args.world / "model_input").resolve()
    os.environ.setdefault("ENERGY_MODEL_DATA_ROOT", str(bundle))
    os.environ.setdefault("V3_AFFINITY", "1")
    import torch
    from checkpoint_io import load_checkpoint
    from data import build
    torch.set_default_dtype(torch.float64)

    started = time.time()
    oracle = Oracle(args.world / "truth.npz", args.nodes)
    truth = oracle.t
    world = oracle.world
    J = len(world.lam)
    basket_of_trip = trip_to_basket(bundle)
    data = build()
    model, blob, meta = load_checkpoint(args.run / "artifacts" / "candidate_rank1.pt", data)
    report: dict = {"run": str(args.run), "world": str(args.world), "quadrature_nodes_per_axis": args.nodes}

    # quadrature accuracy check against a finer rule on a few trips
    sample = basket_of_trip[np.flatnonzero(data["trip_split"] == 2)[:64]]
    fine = Oracle(args.world / "truth.npz", 28).log_z(oracle.b(sample))
    report["quadrature_max_abs_error_vs_28_nodes"] = float(np.abs(oracle.log_z(oracle.b(sample)) - fine).max())

    # ---- 1. held-out likelihood ----
    likelihood = {}
    for split in ("validation", "test"):
        per = np.load(args.run / "reports" / f"likelihood_{split}_per_trip.npz")
        trips = per["trips"].astype(np.int64)
        baskets = basket_of_trip[trips]
        membership = truth["memberships"][baskets]
        # the bundle's purchased lines must be the truth's baskets
        lines = [set(data["line_item"][data["line_ptr"][t]:data["line_ptr"][t + 1]].tolist()) for t in trips[:200]]
        assert all(set(np.flatnonzero(membership[i]).tolist()) == lines[i] for i in range(len(lines)))
        b = oracle.b(baskets)
        oracle_ll = energy(world, b, membership) - oracle.log_z(b)
        household = per["household"]
        likelihood[split] = {
            "trips": int(len(trips)),
            "oracle_mean": float(oracle_ll.mean()),
            "additive_parent_mean": float(per["exact_parent"].mean()),
            "final_model_mean": float(per["target_child"].mean()),
            "oracle_minus_final": paired(oracle_ll - per["target_child"], household),
            "oracle_minus_parent": paired(oracle_ll - per["exact_parent"], household),
            "interaction_gain": paired(per["target_child"] - per["exact_parent"], household),
        }
        gap_parent = oracle_ll.mean() - per["exact_parent"].mean()
        likelihood[split]["share_of_parent_gap_closed_by_interactions"] = float(
            (per["target_child"].mean() - per["exact_parent"].mean()) / gap_parent) if gap_parent > 0 else None
    report["likelihood"] = likelihood

    # ---- 2. structure recovery ----
    with torch.no_grad():
        phi = model.phi.numpy()
        group = model.cat_of.numpy().astype(int)
        rho_c = model.rho_c.numpy()
        theta = model.theta_c().numpy()
        alpha = model.alpha.numpy()
        lam = model.lam.numpy()
    fitted_pair = phi @ phi.T - rho_c[group][:, None] * (group[:, None] == group[None, :])
    true_pair = world.phi @ world.phi.T - world.rho_c[world.category][:, None] * (
        world.category[:, None] == world.category[None, :])
    upper = np.triu_indices(J, 1)
    same_category = (world.category[:, None] == world.category[None, :])[upper]
    slope = np.polyfit(true_pair[upper], fitted_pair[upper], 1)[0]
    if model.household_size_rank1:
        fitted_taste = theta[:, :-1] @ (alpha[:, :-1] - alpha[:, :-1].mean(0)).T
        fitted_size = theta[:, -1]
    else:
        fitted_taste, fitted_size = theta @ alpha.T, None
    true_taste = truth["taste"] @ world.alpha.T

    def double_centre(matrix):
        return matrix - matrix.mean(0, keepdims=True) - matrix.mean(1, keepdims=True) + matrix.mean()

    coefficients = json.loads((args.run / "artifacts/supported_price_response/coefficients.json").read_text())
    fitted_sensitivity = np.zeros(J)
    for row in coefficients["product_sensitivity"]:
        fitted_sensitivity[row["item_id"]] = row["sensitivity"]
    true_sensitivity = world.price_sensitivity[world.category]
    report["structure"] = {
        "interaction_rank_selected": int(blob["active_rank"]),
        "true_interaction_rank": int(world.phi.shape[1]),
        "affinity_groups": int(group.max() + 1),
        "true_categories": int(world.n_categories),
        "effective_pair_correlation": float(np.corrcoef(true_pair[upper], fitted_pair[upper])[0, 1]),
        "effective_pair_slope_fitted_on_true": float(slope),
        "within_category_pair_mean": {"true": float(true_pair[upper][same_category].mean()),
                                      "fitted": float(fitted_pair[upper][same_category].mean())},
        "cross_category_pair_mean": {"true": float(true_pair[upper][~same_category].mean()),
                                     "fitted": float(fitted_pair[upper][~same_category].mean())},
        "strongest_true_complements_recovered": float(np.mean(
            fitted_pair[upper][np.argsort(true_pair[upper])[-200:]]
            > np.quantile(fitted_pair[upper], 0.9))),
        "product_appeal_correlation": float(np.corrcoef(world.lam, lam)[0, 1]),
        "household_taste_correlation": float(np.corrcoef(
            double_centre(true_taste).ravel(), double_centre(fitted_taste).ravel())[0, 1]),
        "household_size_propensity_correlation": (
            float(np.corrcoef(truth["size_shift"], fitted_size)[0, 1]) if fitted_size is not None else None),
        "price_sensitivity": {
            "level_selected": coefficients["level"],
            "correlation_with_truth": float(np.corrcoef(true_sensitivity, fitted_sensitivity)[0, 1]),
            "mean_true": float(true_sensitivity.mean()), "mean_fitted": float(fitted_sensitivity.mean()),
            "mean_absolute_error": float(np.abs(true_sensitivity - fitted_sensitivity).mean()),
            "by_category": {str(c): {"true": float(world.price_sensitivity[c]),
                                     "fitted_mean": float(fitted_sensitivity[world.category == c].mean())}
                            for c in range(world.n_categories)},
        },
    }

    # ---- 3. availability ----
    with np.load(bundle / "basket_input" / "availability.npz") as panel:
        first_period = panel["first_period"].astype(int)
        floor = float(np.exp(panel["log_floor"]))
    weeks = np.arange(1, first_period.shape[0] and truth["price"].shape[1] + 1)
    confirmed = first_period[:, :, None] <= weeks[None, None, :]
    stocked = oracle.first_stocked[:, :, None].astype(int) <= weeks[None, None, :]
    report["availability"] = {
        "cells": int(stocked.size),
        "truly_stocked_share": float(stocked.mean()),
        "confirmed_share": float(confirmed.mean()),
        "precision_confirmed_is_stocked": float((confirmed & stocked).sum() / confirmed.sum()),
        "recall_stocked_is_confirmed": float((confirmed & stocked).sum() / stocked.sum()),
        "never_stocked_pairs_true": int((oracle.first_stocked > len(weeks)).sum()),
        "never_confirmed_pairs": int((first_period > len(weeks)).sum()),
        "launch_week_absolute_error_mean": float(np.abs(
            np.clip(first_period, 1, len(weeks) + 1) - oracle.first_stocked)[
                (oracle.first_stocked > 1) & (oracle.first_stocked <= len(weeks))].mean()),
        "estimated_unconfirmed_weight": floor, "true_unstocked_weight": 0.0,
    }

    # ---- 4. segments ----
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    segments = np.load(args.run / "artifacts" / "customer_segments.npz")
    report["segments"] = {
        "chosen": int(len(np.unique(segments["segment"]))), "true": 3,
        "adjusted_rand_index": float(adjusted_rand_score(truth["segment"], segments["segment"])),
        "normalized_mutual_information": float(normalized_mutual_info_score(
            truth["segment"], segments["segment"])),
        "contingency": pd.crosstab(truth["segment"], segments["segment"]).values.tolist(),
    }

    # ---- 5. recommendation ----
    rec = json.loads((args.run / "reports" / "recommendation.json").read_text())["recommendation"]["per_case"]
    oracle_ranks = []
    for trip, hidden in zip(rec["trips"], rec["hidden_items"]):
        basket = basket_of_trip[trip]
        bought = np.flatnonzero(truth["memberships"][basket])
        revealed = bought[bought != hidden]
        b = oracle.b(np.array([basket]))[0]
        score = b + true_pair[:, revealed].sum(1)
        candidates = np.setdiff1d(np.arange(J), revealed)     # the pipeline ranks the whole catalogue
        scores = score[candidates]
        oracle_ranks.append(midrank(np.where(np.isfinite(scores), scores, -1e300),
                                    int(np.flatnonzero(candidates == hidden)[0])))
    rec_mrr = {name: float(np.mean(1.0 / np.asarray(ranks, dtype=float)))
               for name, ranks in rec["ranks"].items()}
    rec_mrr["oracle_add_one"] = float(np.mean(1.0 / np.asarray(oracle_ranks)))
    report["recommendation"] = {"cases": len(oracle_ranks), "mrr": rec_mrr,
                                "model_share_of_oracle_mrr_above_popularity": float(
                                    (rec_mrr["full_interaction"] - rec_mrr["popularity"])
                                    / (rec_mrr["oracle_add_one"] - rec_mrr["popularity"]))}

    # ---- 6. price scenarios and 7. generation/size ----
    generation = json.loads((args.run / "reports" / "generation_counterfactual.json").read_text())
    trips = np.asarray(generation["trips"], dtype=np.int64)
    baskets = basket_of_trip[trips]
    rng = np.random.default_rng(2561900 + 2)                 # audit default seed + 2
    chosen = []
    for trip in trips:
        bought = np.unique(data["line_item"][data["line_ptr"][trip]:data["line_ptr"][trip + 1]])
        chosen.append(int(bought[rng.integers(len(bought))]))
    chosen = np.asarray(chosen)
    one_hot = np.zeros((len(trips), J))
    one_hot[np.arange(len(trips)), chosen] = 1.0
    ones = np.ones((len(trips), J))
    base_b = oracle.b(baskets)
    factual_own = oracle.expectation(base_b, one_hot)
    factual_size = oracle.expectation(base_b, ones)
    scenarios = []
    for row in generation["counterfactuals"]:
        action = row["log_price_change"]
        own_b = oracle.b(baskets, price_change=one_hot * action)
        uniform_b = oracle.b(baskets, price_change=ones * action)
        own = oracle.expectation(own_b, one_hot)
        size = oracle.expectation(uniform_b, ones)
        scenarios.append({
            "price_multiplier": row["price_multiplier"],
            "own_incidence_retained": {"model": row["own_incidence_retained"],
                                       "oracle": float((own / factual_own).mean())},
            "uniform_size_change": {"model": row["uniform_size_change"],
                                    "oracle": float((size - factual_size).mean())},
        })
    report["price_scenarios"] = scenarios
    report["size_and_generation"] = {
        "trips": int(len(trips)),
        "observed_mean_size": generation["observed_size_mean"],
        "model_expected_size": generation["factual_expected_size"],
        "oracle_expected_size": float(factual_size.mean()),
        "model_generated_size_mean": generation["generation"]["generated_size_mean"],
        "category_total_variation_generated_vs_observed": generation["generation"]["category_total_variation"],
        "population_size_gates": json.loads((args.run / "reports" / "population_size.json").read_text())["gates"],
    }

    # ---- 8. promotion policy ----
    mdp = json.loads((args.run / "reports" / "segment_promotion_mdp.json").read_text())
    log_price = np.log(oracle.price)
    policy_rows = []
    oracle_daily = {}
    for segment in mdp["segments"]:
        seg_trips = np.asarray(segment["trips"], dtype=np.int64)
        seg_baskets = basket_of_trip[seg_trips]
        week = truth["trip_period"][seg_baskets] - 1
        price = np.exp(log_price[:, week].T)                       # factual chain price [B, J]
        base = oracle.b(seg_baskets)
        base_value = oracle.expectation(base, price)
        for action in segment["actions"]:
            products = np.asarray(segment["bundles"][action["bundle"]]["products"]
                                  if isinstance(segment["bundles"][action["bundle"]], dict)
                                  else segment["bundles"][action["bundle"]], dtype=int)
            promoted = np.zeros(J)
            promoted[products] = 1.0
            change = promoted[None, :] * math.log1p(-action["discount"])
            changed = oracle.b(seg_baskets, price_change=np.repeat(change, len(seg_baskets), 0))
            value = oracle.expectation(changed, price)
            markdown = oracle.expectation(changed, price * promoted[None, :] * action["discount"])
            incremental = value - markdown - base_value
            name = f"segment_{segment['segment']}_bundle_{action['bundle']}_discount_{int(round(action['discount'] * 100))}"
            oracle_daily[name] = (float(incremental.mean()), float(np.mean(markdown)))
            policy_rows.append({
                "action": name,
                "model_incremental_post_discount_sales": action["incremental_post_discount_sales"],
                "model_lcb95": action["incremental_post_discount_sales_lcb95"],
                "oracle_incremental_post_discount_sales": float(incremental.mean()),
                "model_markdown": action["markdown_spend"],
                "oracle_markdown": float(np.mean(markdown)),
            })
    model_values = np.array([r["model_incremental_post_discount_sales"] for r in policy_rows])
    oracle_values = np.array([r["oracle_incremental_post_discount_sales"] for r in policy_rows])
    scenarios_out = []
    trips_per_day = {s["segment"]: s["expected_trips_per_day"] for s in mdp["segments"]}
    for scenario in mdp["budget_scenarios"]:
        realized = recomputed = 0.0
        model_daily = {r["action"]: r["model_incremental_post_discount_sales"] for r in policy_rows}
        for name, days in scenario["action_day_counts"].items():
            if name == "no_promotion":
                continue
            segment_index = int(name.split("_")[1])
            realized += days * trips_per_day[segment_index] * oracle_daily[name][0]
            recomputed += days * trips_per_day[segment_index] * model_daily[name]
        scenarios_out.append({
            "budget_fraction": scenario["budget_fraction_of_maximum_action_spend"],
            "action_day_counts": scenario["action_day_counts"],
            "model_total_incremental_post_discount_sales": scenario["total_incremental_post_discount_sales"],
            "model_total_lcb95": scenario["total_robust_reward_lcb95"],
            "model_total_recomputed_from_actions": recomputed,
            "oracle_total_incremental_post_discount_sales": realized,
        })
    report["promotion_policy"] = {
        "actions": policy_rows,
        "action_value_correlation": float(np.corrcoef(model_values, oracle_values)[0, 1]),
        "sign_agreement": float(np.mean(np.sign(model_values) == np.sign(oracle_values))),
        "oracle_best_action": policy_rows[int(np.argmax(oracle_values))]["action"],
        "model_best_action": policy_rows[int(np.argmax(model_values))]["action"],
        "budget_scenarios": scenarios_out,
    }
    report["runtime_seconds"] = round(time.time() - started, 1)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
