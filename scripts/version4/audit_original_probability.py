#!/usr/bin/env python3
"""Non-mutating probability/price audit of the saved original-data checkpoints.

Re-evaluates additive responses with an independent first-adjoint covariance, checks
the actual common/relative price map, and reanalyses locked paired scores by household.
Historical particle diagnostics are assessed against the new absolute ESS requirement.
This is a checkpoint audit, not a refit or a causal identification experiment.
"""
from __future__ import annotations

import argparse
import contextlib
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
from torch.nn.functional import softplus

from checkpoint_io import ROOT, load_checkpoint
from features import Features
from fit import Batcher
from interaction_particles import differentiable_log_size_beta0
from pipeline_support import smolyak_rule
from price_response import additive_uniform_price_response, changed_price_context
from provenance import file_sha256, strict_json_dumps
from uncertainty import paired_score_summary


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, value):
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def exact_covariance_response(model, ix):
    """-Cov(N, sum(g_j Y_j)) via one native adjoint; no finite differences."""
    with torch.enable_grad():
        utility = model.b_flat(ix).detach().requires_grad_(True)
        probability = differentiable_log_size_beta0(model, ix, utility).softmax(-1)
        grid = torch.arange(1, probability.shape[-1] + 1, dtype=utility.dtype)
        mean = probability @ grid
        covariance = torch.autograd.grad(mean.sum(), utility)[0]
    coefficient = model.price_coefficients(ix.item, ix.item_trip).detach()
    slope = -torch.zeros_like(mean).index_add(0, ix.item_trip, coefficient * covariance)
    variance = probability.detach() @ grid.square() - mean.detach().square()
    return mean.detach(), variance, slope.detach(), coefficient


def additive_audit(model, batcher, trips, chunk):
    rows = {name: [] for name in ("mean", "variance", "covariance_slope", "fourth_order_slope")}
    coefficient_sum, slot_count = 0.0, 0
    worst = {"covariance_vs_fourth_order": 0.0, "step_halving": 0.0,
             "second_vs_fourth_order": 0.0, "uniform_price_map": 0.0,
             "single_price_map": 0.0}
    steps = []
    for start in range(0, len(trips), chunk):
        ix, ctx, _, house, *_ = batcher.make(trips[start:start + chunk])
        model.house, model.ctx = house, ctx
        mean, variance, slope, coefficient = exact_covariance_response(model, ix)
        with torch.no_grad():
            response = additive_uniform_price_response(model, ix)
            half = additive_uniform_price_response(model, ix, step=response.step / 2)
            worst["covariance_vs_fourth_order"] = max(worst["covariance_vs_fourth_order"],
                float((slope - response.size_slope).abs().max()))
            worst["step_halving"] = max(worst["step_halving"],
                float((half.size_slope - response.size_slope).abs().max()))
            worst["second_vs_fourth_order"] = max(worst["second_vs_fourth_order"],
                float(response.second_fourth_discrepancy.max()))
            base = model.b_flat(ix)
            counts = torch.bincount(ix.item_trip, minlength=ix.B)
            kappa = softplus(model.price_kappa)
            for name, change in (("uniform_price_map", torch.full_like(base, .03)),
                                 ("single_price_map", .03 * (ix.item == ix.item[0]))):
                action = changed_price_context(ctx, ix.item_trip, change)
                average = torch.zeros(ix.B).index_add(0, ix.item_trip, change) / counts
                expected = -coefficient * (kappa * change + (1 - kappa) * average[ix.item_trip])
                actual = model.b_at(ix.item, ix.item_trip, action) - base
                worst[name] = max(worst[name], float((actual - expected).abs().max()))
        for name, value in zip(rows, (mean, variance, slope, response.size_slope)):
            rows[name].append(value.numpy())
        coefficient_sum += float(coefficient.sum())
        slot_count += coefficient.numel()
        steps.append(response.step)
        print(f"additive contexts {min(start + chunk, len(trips))}/{len(trips)}", flush=True)
    rows = {key: np.concatenate(value) for key, value in rows.items()}
    exact = float(rows["covariance_slope"].mean() / rows["mean"].mean())
    legacy = float(-coefficient_sum / slot_count * rows["variance"].mean() / rows["mean"].mean())
    passed = (worst["covariance_vs_fourth_order"] < 1e-6 and worst["step_halving"] < 1e-6
              and worst["uniform_price_map"] < 1e-10 and worst["single_price_map"] < 1e-10)
    return {"contexts": len(trips), "selection": "prefix of previously locked validation manifest",
            "mean_expected_size": float(rows["mean"].mean()), "exact_covariance_elasticity": exact,
            "fourth_order_elasticity": float(rows["fourth_order_slope"].mean() / rows["mean"].mean()),
            "legacy_proxy_elasticity": legacy, "legacy_relative_magnitude_error": abs(legacy / exact - 1),
            "kappa": float(softplus(model.price_kappa)), "worst_absolute_errors": worst,
            "difference_step_range": [min(steps), max(steps)], "passed": passed}, rows


@torch.no_grad()
def child_replay(model, batcher, trips, saved, level):
    """Replay locked validation scores; no new model or quadrature selection."""
    singular = torch.linalg.svdvals(model.phi)
    rank = int((singular > singular[0] * 1e-10).sum())
    model.quad = smolyak_rule(model, rank, level)
    values = []
    for trip in trips:
        ix, ctx, line_ctx, house, li, lt, lc, _ = batcher.make(np.array([trip]))
        model.house, model.ctx = house, ctx
        values.append(float(model.energy(li, lt, lc, ix.B, line_ctx)
                            - model.log_Z(ix, drop_empty=True)))
        print(f"child replay {len(values)}/{len(trips)}", flush=True)
    error = float(np.max(np.abs(np.asarray(values) - saved)))
    return {"contexts": len(trips), "active_rank": rank, "quadrature_level": level,
            "maximum_saved_score_error": error, "passed": error < 1e-8}


def analyse_saved_scores(data, split, parent_digest, child_digest, reports_dir=ROOT / "reports"):
    reports_dir = Path(reports_dir)
    report_path = reports_dir / f"likelihood_{split}.json"
    score_path = reports_dir / f"likelihood_{split}_per_trip.npz"
    report = json.loads(report_path.read_text())
    if (report["parent_sha256"] != parent_digest or report["checkpoint_sha256"] != child_digest
            or report["per_trip_sha256"] != file_sha256(score_path)):
        raise ValueError(f"{split} saved-score provenance mismatch")
    with np.load(score_path) as z:
        trips, audit_trips = z["trips"], z["audit_trips"]
        if not np.array_equal(trips[:len(audit_trips)], audit_trips):
            raise ValueError("audit scores must refer to the locked prefix")
        gain = paired_score_summary(z["target_child"] - z["exact_parent"], data["trip_user"][trips])
        audit = paired_score_summary(z["target_child"][:len(audit_trips)] - z["audit_child"],
                                     data["trip_user"][audit_trips])
    allowance = abs(audit["mean"]) + 1.96 * audit["standard_error"]
    return {"gain": gain, "adjacent_rule_difference": audit,
            "empirical_adjacent_rule_allowance": allowance,
            "gain_lower_after_empirical_allowance": gain["95_interval"][0] - allowance,
            "interpretation": "fixed-model household-cluster SE; adjacent-rule allowance is not an exact error bound"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="isolated completed pipeline directory to audit")
    parser.add_argument("--parent", type=Path,
                        help="additive parent; defaults to RUN_DIR/out/v3_pipeline_additive_best.pt")
    parser.add_argument("--checkpoint", type=Path,
                        help="final checkpoint; defaults to RUN_DIR/artifacts/candidate_rank1.pt")
    parser.add_argument("--data", type=Path,
                        default=ROOT / "basket_input/v3_index_affinity.npz")
    parser.add_argument("--contexts", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--replay-contexts", type=int, default=8)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if min(args.contexts, args.chunk, args.replay_contexts, args.threads) < 1:
        parser.error("all budgets must be positive")
    run_dir = args.run_dir.resolve()
    parent_path = (args.parent or run_dir / "out/v3_pipeline_additive_best.pt").resolve()
    child_path = (args.checkpoint or run_dir / "artifacts/candidate_rank1.pt").resolve()
    data_path = args.data.resolve()
    reports_dir = run_dir / "reports"
    output = args.output_dir or ROOT / "artifacts" / (
        "original_probability_audit_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"))
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    with (output / "pipeline.log").open("w") as log, contextlib.redirect_stdout(Tee(sys.stdout, log)), \
            contextlib.redirect_stderr(Tee(sys.stderr, log)):
        report = {"status": "running", "output": str(output), "refitted": False,
                  "command": [sys.executable, *sys.argv]}
        (output / "report.json").write_text(strict_json_dumps(report))
        try:
            torch.set_default_dtype(torch.float64)
            torch.set_num_threads(args.threads)
            paths = [parent_path, child_path, data_path, run_dir / "artifacts/candidate.json",
                     reports_dir / "generation_counterfactual.json"]
            paths += [reports_dir / f"likelihood_{s}{suffix}" for s in ("validation", "test")
                      for suffix in (".json", "_per_trip.npz")]
            report["input_sha256"] = {str(p): file_sha256(p) for p in paths}
            report["source_sha256"] = {str(p.relative_to(ROOT)): file_sha256(p)
                                       for p in sorted((ROOT / "scripts").rglob("*.py"))}
            with np.load(paths[2]) as z:
                data = {key: z[key] for key in z.files}
            parent, blob, meta = load_checkpoint(paths[0], data,
                required_capabilities=("conditional_nonempty_incidence",))
            child, child_blob, _ = load_checkpoint(paths[1], data,
                required_capabilities=("conditional_nonempty_incidence", "gram_interactions"))
            for model in (parent, child):
                for parameter in model.parameters():
                    parameter.requires_grad_(False)
            report["data_fingerprint_sha256"] = blob["data_fingerprint_sha256"]
            report["products"] = int(data["n_item"])
            report["support"] = f"nonempty incidence baskets of size 1..{parent.nmax}"
            report["trained_capabilities"] = child_blob["trained_capabilities"]
            report["parent_price_response_estimator"] = blob.get("price_response_estimator", "legacy_unlabelled")
            batcher = Batcher(data, Features(int(data["n_item"]), int(data["n_store"]),
                                             include_recency=False), parent.nmax, include_recency=False)
            with np.load(reports_dir / "likelihood_validation_per_trip.npz") as z:
                trips = z["trips"][:args.contexts].copy()
                replay_trips = z["trips"][:args.replay_contexts].copy()
                replay_scores = z["target_child"][:args.replay_contexts].copy()
            if not bool(np.all(data["trip_split"][trips] == 1)):
                raise ValueError("price audit requires locked validation contexts")
            report["additive_response"], rows = additive_audit(parent, batcher, trips, args.chunk)
            np.savez_compressed(output / "per_context.npz", trips=trips,
                                household=data["trip_user"][trips], **rows)
            original = json.loads((reports_dir / "likelihood_validation.json").read_text())
            report["child_score_replay"] = child_replay(child, batcher, replay_trips, replay_scores,
                                                       original["levels"]["target"])
            report["paired_likelihood"] = {split: analyse_saved_scores(data, split,
                report["input_sha256"][str(paths[0])], report["input_sha256"][str(paths[1])],
                reports_dir)
                for split in ("validation", "test")}
            old = json.loads(paths[3].read_text())
            bands = [{"size_band": old["size_bands"][b["band"]], "draws": b["draws"],
                      "minimum_absolute_ess": b["draws"] * b["ess_fraction_min"],
                      "minimum_fraction": b["ess_fraction_min"]} for b in old["full_fit"]["bands"]]
            report["historical_fit_ess"] = {"bands": bands, "new_absolute_minimum": 2.0,
                "new_fraction_minimum": .2, "passes_new_gate": all(
                    b["minimum_absolute_ess"] >= 2 and b["minimum_fraction"] >= .2 for b in bands),
                "interpretation": "reconstructed from saved full-fit band minima; not a newly sampled bank"}
            generation = json.loads(paths[4].read_text())
            if generation["checkpoint_sha256"] != report["input_sha256"][str(paths[1])]:
                raise ValueError("generation checkpoint mismatch")
            report["historical_generation"] = {"contexts": len(generation["trips"]),
                "factual_expected_size": generation["factual_expected_size"],
                **{k: v for k, v in generation["generation"].items() if k != "examples"},
                "interpretation": "existing small-context predictive check, not rerun or population certification"}
            for path, digest in report["input_sha256"].items():
                if file_sha256(path) != digest:
                    raise RuntimeError(f"input changed during audit: {path}")
            report["implementation_checks_passed"] = bool(report["additive_response"]["passed"]
                                                          and report["child_score_replay"]["passed"])
            report["status"] = "completed" if report["implementation_checks_passed"] else "failed"
            report["interpretation"] = ("Successful numerical checks do not certify the historical fit under new ESS "
                "gates. No retraining, optimizer continuation, causal identification, arrival or quantity fit was performed.")
        except Exception:
            report["status"] = "failed"
            report["exception"] = traceback.format_exc()
            traceback.print_exc()
            raise
        finally:
            report["runtime_seconds"] = time.monotonic() - start
            (output / "report.json").write_text(strict_json_dumps(report))
            print(strict_json_dumps(report))
        if not report["implementation_checks_passed"]:
            raise SystemExit(2)


if __name__ == "__main__":
    main()
