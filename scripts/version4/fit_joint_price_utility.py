#!/usr/bin/env python3
"""Estimate a SHOPPER-style current-price term inside the joint basket likelihood.

This is deliberately a low-dimensional diagnostic, not another bespoke choice model.
Starting from a fitted additive basket checkpoint with price fixed to zero, it estimates

    U(S | x) = U_0(S | x) - sum_{j in S} tau_category(j) * dlog_price_jt

with nonnegative ``tau`` and the *same exact normalizer* used by the additive basket
model.  A global coefficient is fitted first.  Category coefficients are then fitted under
a validation-selected ridge toward their common mean.  Only training baskets estimate
coefficients; validation selects the hierarchy; test is scored once after selection.

For ERIM, only products in the retail-aggregate chain/week panel receive a price
coefficient.  Store deviations are disabled because those cells are observed only after a
sale.  Products without reliable price exposure are fixed to zero.
"""
from __future__ import annotations

import argparse
import contextlib
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

from checkpoint_io import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from interaction_particles import differentiable_log_size_beta0
from pipeline_support import supported_trips
from provenance import file_sha256, model_data_root, strict_json_dumps
from uncertainty import paired_score_summary


torch.set_default_dtype(torch.float64)
DATA_ROOT = model_data_root(ROOT)


class Tee:
    def __init__(self, stream, path: Path):
        self.stream = stream
        self.file = path.open("w", buffering=1)

    def write(self, value):
        self.stream.write(value)
        self.file.write(value)
        return len(value)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="price-zero additive checkpoint")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--maximum-iterations", type=int, default=40)
    parser.add_argument("--coefficient-upper-bound", type=float, default=5.0)
    parser.add_argument("--category-penalties", type=float, nargs="+",
                        default=(0.01, 0.1, 1.0, 10.0))
    parser.add_argument("--minimum-validation-gain", type=float, default=0.0)
    return parser.parse_args()


def reliable_price_support(products: int) -> tuple[np.ndarray, np.ndarray, list[str], dict]:
    items = pd.read_parquet(
        DATA_ROOT / "basket_input/items.parquet",
        columns=["PRODUCT_ID", "item_id", "category"])
    if len(items) != products or items.item_id.nunique() != products:
        raise RuntimeError("item table does not contain the complete model catalogue")
    category_names = sorted(items.category.astype(str).unique().tolist())
    category_index = {name: index for index, name in enumerate(category_names)}
    item_category = np.full(products, -1, dtype=np.int64)
    item_category[items.item_id.to_numpy(np.int64)] = [
        category_index[str(value)] for value in items.category]
    weekly = pd.read_parquet(
        DATA_ROOT / "data/price_week.parquet", columns=["PRODUCT_ID", "price"])
    if (not np.isfinite(weekly.price).all()) or bool((weekly.price <= 0).any()):
        raise RuntimeError("reliable price panel contains invalid prices")
    reliable_product_ids = set(weekly.PRODUCT_ID.astype(np.int64).unique().tolist())
    product_to_item = items.set_index("PRODUCT_ID").item_id
    unknown = reliable_product_ids.difference(product_to_item.index.astype(int))
    if unknown:
        raise RuntimeError(f"price panel contains unknown products: {sorted(unknown)[:10]}")
    reliable = np.zeros(products, dtype=bool)
    reliable[product_to_item.loc[sorted(reliable_product_ids)].to_numpy(np.int64)] = True
    categories_supported = sorted(set(item_category[reliable].tolist()))
    return reliable, item_category, category_names, {
        "reliable_products": int(reliable.sum()),
        "unsupported_products_fixed_to_zero": int((~reliable).sum()),
        "supported_categories": [category_names[index] for index in categories_supported],
        "unsupported_categories": [name for index, name in enumerate(category_names)
                                   if index not in categories_supported],
        "price_panel": str(DATA_ROOT / "data/price_week.parquet"),
        "price_panel_sha256": file_sha256(DATA_ROOT / "data/price_week.parquet"),
        "store_price_deviation_used": False,
    }


def coefficient_vector(parameters: torch.Tensor, level: str,
                       reliable: torch.Tensor, item_category: torch.Tensor) -> torch.Tensor:
    if level == "zero":
        value = torch.zeros_like(item_category, dtype=parameters.dtype)
    elif level == "global":
        value = parameters[0].expand(item_category.shape[0])
    elif level == "category":
        value = parameters[item_category]
    else:
        raise ValueError(f"unknown price level {level!r}")
    return torch.where(reliable, value, torch.zeros_like(value))


def batch_loglik(model, batcher, trips: np.ndarray, sensitivity: torch.Tensor):
    ix, ctx, line_ctx, house, line_item, line_trip, line_cat, _ = batcher.make(trips)
    model.house, model.ctx = house, ctx
    # The checkpoint's price block is exactly zero.  Add the learned current-price term
    # explicitly to both the observed energy and every alternative in the exact DP.
    slot_b = model.b_flat(ix) - sensitivity[ix.item] * ctx["dlp"]
    line_price = sensitivity[line_item] * line_ctx["dlp"]
    energy = model.energy(line_item, line_trip, line_cat, ix.B, line_ctx)
    energy = energy - torch.zeros(ix.B, dtype=energy.dtype).index_add_(
        0, line_trip, line_price)
    logz = torch.logsumexp(differentiable_log_size_beta0(model, ix, slot_b), dim=-1)
    return energy - logz, ctx, ix


def objective_and_gradient(values: np.ndarray, *, model, batcher, trips,
                           level, reliable, item_category, penalty, chunk):
    parameters = torch.tensor(values, dtype=torch.float64, requires_grad=True)
    total = 0.0
    for start in range(0, len(trips), chunk):
        # Rebuild this cheap indexing graph for every chunk.  Backward frees each chunk's
        # graph, while gradients continue to accumulate on the common parameter leaf.
        sensitivity = coefficient_vector(
            parameters, level, reliable, item_category)
        loglik, _, _ = batch_loglik(
            model, batcher, trips[start:start + chunk], sensitivity)
        loss = -loglik.sum() / len(trips)
        loss.backward()
        total += float(loss.detach())
    if level == "category" and penalty > 0:
        active = torch.unique(item_category[reliable])
        active_values = parameters[active]
        ridge = float(penalty) * (active_values - active_values.mean()).square().mean()
        ridge.backward()
        total += float(ridge.detach())
    gradient = parameters.grad.detach().numpy().copy()
    if not np.isfinite(total) or not np.isfinite(gradient).all():
        raise FloatingPointError("non-finite joint-price objective")
    return total, gradient


def fit_model(initial, *, label, model, batcher, trips, level, reliable,
              item_category, penalty, chunk, upper, maximum_iterations):
    calls = 0
    started = time.perf_counter()

    def evaluate(values):
        nonlocal calls
        calls += 1
        result = objective_and_gradient(
            values, model=model, batcher=batcher, trips=trips, level=level,
            reliable=reliable, item_category=item_category, penalty=penalty,
            chunk=chunk)
        print(f"[joint-price] {label} call={calls} objective={result[0]:.8f} "
              f"gradient_max={np.abs(result[1]).max():.3e} "
              f"coefficient={np.array2string(values, precision=4)}", flush=True)
        return result

    answer = minimize(
        evaluate, np.asarray(initial, dtype=np.float64), method="L-BFGS-B", jac=True,
        bounds=[(0.0, upper)] * len(initial),
        options={"maxiter": maximum_iterations, "ftol": 1e-11, "gtol": 1e-7,
                 "maxls": 20})
    return answer.x, {
        "label": label, "success": bool(answer.success), "message": str(answer.message),
        "iterations": int(answer.nit), "objective_calls": int(answer.nfev),
        "training_objective": float(answer.fun),
        "projected_gradient_max": float(np.abs(answer.jac).max()),
        "elapsed_seconds": time.perf_counter() - started,
    }


@torch.no_grad()
def score_model(model, batcher, trips, parameters, level, reliable, item_category,
                chunk):
    parameters = torch.as_tensor(parameters, dtype=torch.float64)
    sensitivity = coefficient_vector(
        parameters, level, reliable, item_category)
    scores, exposure = [], []
    reliable_float = reliable.to(torch.float64)
    denominator = reliable_float.sum().clamp_min(1.0)
    for start in range(0, len(trips), chunk):
        loglik, ctx, ix = batch_loglik(
            model, batcher, trips[start:start + chunk], sensitivity)
        scores.append(loglik.cpu().numpy())
        squared = ctx["dlp"].square() * reliable_float[ix.item]
        per_trip = torch.zeros(ix.B, dtype=torch.float64).index_add_(
            0, ix.item_trip, squared)
        exposure.append(torch.sqrt(per_trip / denominator).cpu().numpy())
    return np.concatenate(scores), np.concatenate(exposure)


def comparison(candidate, baseline, trips, data, exposure):
    gain = candidate - baseline
    result = {
        "all": paired_score_summary(gain, data["trip_user"][trips]),
        "candidate_mean_log_likelihood": float(candidate.mean()),
        "baseline_mean_log_likelihood": float(baseline.mean()),
        "price_exposure_rms_quantiles": {
            str(q): float(np.quantile(exposure, q)) for q in (0, .5, .9, .95, .975, 1)
        },
        "most_price_extreme_subsets": {},
    }
    for fraction in (.20, .10, .05, .025):
        count = max(2, int(np.ceil(len(trips) * fraction)))
        chosen = np.argsort(exposure, kind="stable")[-count:]
        result["most_price_extreme_subsets"][str(fraction)] = paired_score_summary(
            gain[chosen], data["trip_user"][trips[chosen]])
    return result


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    output = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "joint_price_utility.log"
    with contextlib.redirect_stdout(Tee(sys.stdout, log_path)), \
            contextlib.redirect_stderr(Tee(sys.stderr, log_path)):
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
        data = build()
        model, blob, meta = load_checkpoint(
            checkpoint, data, required_capabilities=("conditional_nonempty_incidence",))
        if bool((model.phi.detach() != 0).any()):
            raise ValueError("joint price fitting requires the exact additive checkpoint")
        with torch.no_grad():
            maximum_household_factor = torch.nn.functional.softplus(
                model.gamma).amax(0)
            maximum_price = (
                maximum_household_factor
                * torch.nn.functional.softplus(model.beta)).sum(-1).max()
        if float(maximum_price) > 1e-12:
            raise ValueError("parent checkpoint must have price fixed to zero")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        features = Features(
            int(data["n_item"]), int(data["n_store"]),
            include_recency=False, include_store_price=False)
        batcher = Batcher(data, features, model.nmax, include_recency=False)
        reliable_np, item_category_np, names, price_audit = reliable_price_support(model.J)
        reliable = torch.as_tensor(reliable_np, dtype=torch.bool)
        item_category = torch.as_tensor(item_category_np, dtype=torch.long)
        populations = {
            "train": supported_trips(data, 0, model.nmax),
            "validation": supported_trips(data, 1, model.nmax),
            "test": supported_trips(data, 2, model.nmax),
        }
        print(f"[joint-price] exact additive joint likelihood; train/validation/test="
              f"{len(populations['train'])}/{len(populations['validation'])}/"
              f"{len(populations['test'])}", flush=True)
        print(f"[joint-price] reliable chain-price support: {reliable_np.sum()}/{model.J} "
              f"products; unsupported={price_audit['unsupported_categories']}", flush=True)

        global_value, global_fit = fit_model(
            [0.5], label="global", model=model, batcher=batcher,
            trips=populations["train"], level="global", reliable=reliable,
            item_category=item_category, penalty=0.0, chunk=args.chunk,
            upper=args.coefficient_upper_bound,
            maximum_iterations=args.maximum_iterations)
        baseline_validation, validation_exposure = score_model(
            model, batcher, populations["validation"], [0.0], "global",
            reliable, item_category, args.chunk)
        global_validation, _ = score_model(
            model, batcher, populations["validation"], global_value, "global",
            reliable, item_category, args.chunk)
        candidates = [{
            "level": "global", "penalty": 0.0, "parameters": global_value,
            "fit": global_fit,
            "validation_gain": float((global_validation - baseline_validation).mean()),
            "optimizer_eligible": bool(
                global_fit["success"] and global_fit["projected_gradient_max"] <= 1e-5),
        }]
        initial_category = np.full(len(names), global_value[0])
        for penalty in args.category_penalties:
            values, fit = fit_model(
                initial_category, label=f"category-ridge-{penalty:g}", model=model,
                batcher=batcher, trips=populations["train"], level="category",
                reliable=reliable, item_category=item_category, penalty=penalty,
                chunk=args.chunk, upper=args.coefficient_upper_bound,
                maximum_iterations=args.maximum_iterations)
            validation, _ = score_model(
                model, batcher, populations["validation"], values, "category",
                reliable, item_category, args.chunk)
            candidates.append({
                "level": "category", "penalty": float(penalty), "parameters": values,
                "fit": fit,
                "validation_gain": float((validation - baseline_validation).mean()),
                "optimizer_eligible": bool(
                    fit["success"] and fit["projected_gradient_max"] <= 1e-5),
            })
        eligible_candidates = [row for row in candidates if row["optimizer_eligible"]]
        if not eligible_candidates:
            raise RuntimeError("no price hierarchy reached the optimizer convergence gate")
        selected = max(eligible_candidates, key=lambda row: row["validation_gain"])
        selected_level = selected["level"]
        selected_parameters = selected["parameters"]
        validation_selected, _ = score_model(
            model, batcher, populations["validation"], selected_parameters,
            selected_level, reliable, item_category, args.chunk)
        # The test split is not used above.  It is opened only after the hierarchy and
        # shrinkage penalty have been fixed by validation.
        baseline_test, test_exposure = score_model(
            model, batcher, populations["test"], [0.0], "global",
            reliable, item_category, args.chunk)
        selected_test, _ = score_model(
            model, batcher, populations["test"], selected_parameters,
            selected_level, reliable, item_category, args.chunk)
        validation_comparison = comparison(
            validation_selected, baseline_validation, populations["validation"],
            data, validation_exposure)
        test_comparison = comparison(
            selected_test, baseline_test, populations["test"], data, test_exposure)
        validation_pass = (
            validation_comparison["all"]["95_interval"][0]
            > args.minimum_validation_gain)
        test_pass = test_comparison["all"]["95_interval"][0] > 0.0
        active_parameter = (selected_parameters if selected_level == "global" else
                            selected_parameters[np.unique(item_category_np[reliable_np])])
        boundary = bool(np.any(active_parameter <= 1e-8)
                        or np.any(active_parameter >= args.coefficient_upper_bound - 1e-8))
        optimizer_pass = bool(selected["optimizer_eligible"])
        certified = bool(optimizer_pass and validation_pass and test_pass and not boundary)
        sensitivity = np.zeros(model.J, dtype=np.float64)
        if selected_level == "global":
            sensitivity[reliable_np] = selected_parameters[0]
        else:
            sensitivity[reliable_np] = selected_parameters[item_category_np[reliable_np]]
        report = {
            "status": "completed",
            "claim_level": ("heldout_predictive_price_component" if certified
                            else "not_certified"),
            "method": "SHOPPER-style normalized contemporaneous price in joint basket utility",
            "energy_term": "-sensitivity[item] * (log price[item,week] - training item mean)",
            "normalizer": "exact additive category/cardinality dynamic program",
            "nonprice_parameters": "fixed at the validated price-zero additive checkpoint",
            "causal_interpretation": False,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": file_sha256(checkpoint),
            "data_fingerprint_sha256": blob["data_fingerprint_sha256"],
            "price_support": price_audit,
            "splits": {key: int(len(value)) for key, value in populations.items()},
            "selection_rule": (
                "fit coefficients on train; select level/ridge by validation mean joint "
                "log likelihood; open test after selection"),
            "candidates": [{
                **{key: value for key, value in row.items() if key != "parameters"},
                "parameters": row["parameters"].tolist(),
            } for row in candidates],
            "selected": {
                "level": selected_level, "penalty": selected["penalty"],
                "category_names": names,
                "parameter_interpretation": (
                    "optimizer coordinates; categories without reliable price support are "
                    "unused and have an effective coefficient of zero"),
                "parameters": selected_parameters.tolist(),
                "effective_category_parameters": {
                    name: (float(selected_parameters[index])
                           if index in set(item_category_np[reliable_np].tolist()) else None)
                    for index, name in enumerate(names)},
                "product_sensitivity_minimum": float(sensitivity.min()),
                "product_sensitivity_median_reliable": float(np.median(sensitivity[reliable_np])),
                "product_sensitivity_maximum": float(sensitivity.max()),
                "coefficient_at_bound": boundary,
            },
            "validation": validation_comparison,
            "test": test_comparison,
            "gates": {
                "selected_optimizer_converged": optimizer_pass,
                "validation_95_interval_above_minimum": validation_pass,
                "test_95_interval_above_zero": test_pass,
                "no_selected_coefficient_at_bound": not boundary,
                "eligible_for_observational_price_scenario_api": certified,
            },
            "limitations": [
                "The source price is a sales-weighted retail aggregate, not a randomized shelf-price intervention.",
                "The non-price basket parameters are held fixed, so this tests incremental price information before a full joint refit.",
                "The chronological test split has been inspected in earlier experiments and is not a pristine confirmatory holdout.",
                "Household- and SKU-specific price heterogeneity is intentionally not estimated.",
            ],
            "artifacts": {
                "log": str(log_path),
                "per_trip": str(output / "joint_price_per_trip.npz"),
                "coefficients": str(output / "joint_price_coefficients.json"),
            },
        }
        np.savez_compressed(
            output / "joint_price_per_trip.npz",
            validation_trips=populations["validation"],
            validation_household=data["trip_user"][populations["validation"]],
            validation_baseline=baseline_validation,
            validation_selected=validation_selected,
            validation_price_exposure=validation_exposure,
            test_trips=populations["test"],
            test_household=data["trip_user"][populations["test"]],
            test_baseline=baseline_test,
            test_selected=selected_test,
            test_price_exposure=test_exposure)
        coefficients = {
            "status": ("certified_observational_predictor" if certified else "not_certified"),
            "price_response_estimator": "joint_basket_current_price_category_pooling_v1",
            "price_feature_contract": "chain_product_week_only",
            "level": "global" if selected_level == "global" else "product",
            "global_sensitivity": float(np.median(sensitivity[reliable_np])),
            "unsupported_catalogue_products_use_global_sensitivity": False,
            "supported_item_ids": np.flatnonzero(reliable_np).tolist(),
            "product_sensitivity": [
                {"item_id": int(item), "sensitivity": float(sensitivity[item])}
                for item in np.flatnonzero(reliable_np)],
            "individual_coefficient_interpretation_supported": False,
            "source_report": str(output / "report.json"),
        }
        (output / "joint_price_coefficients.json").write_text(
            strict_json_dumps(coefficients))
        (output / "report.json").write_text(strict_json_dumps(report))
        print(f"[joint-price] selected={selected_level} penalty={selected['penalty']:g} "
              f"validation_gain={validation_comparison['all']['mean']:+.8f} "
              f"test_gain={test_comparison['all']['mean']:+.8f}", flush=True)
        print(f"[joint-price] certified={certified}; report={output / 'report.json'}",
              flush=True)


if __name__ == "__main__":
    main()
