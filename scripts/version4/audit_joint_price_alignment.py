#!/usr/bin/env python3
"""Falsification checks for the fitted direct current-price basket component."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch

from checkpoint_io import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from fit_joint_price_utility import (Tee, reliable_price_support, score_model)
from pipeline_support import supported_trips
from provenance import file_sha256, strict_json_dumps
from uncertainty import paired_score_summary


torch.set_default_dtype(torch.float64)


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fit-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--placebo-replicates", type=int, default=19)
    parser.add_argument("--seed", type=int, default=84731)
    return parser.parse_args()


def clone_features(source, dev):
    # Features is a read-only collection after construction.  A shallow clone avoids
    # re-reading panels and changes only the explicitly supplied chain-price matrix.
    answer = object.__new__(Features)
    answer.__dict__ = source.__dict__.copy()
    answer.dev = torch.as_tensor(dev, dtype=source.dev.dtype)
    return answer


def two_way_summary(values, household, week):
    values = np.asarray(values, dtype=np.float64)
    hh = paired_score_summary(values, household, "household")
    wk = paired_score_summary(values, week, "week")
    iid = float(values.std(ddof=1) / np.sqrt(len(values)))
    variance = max(0.0, hh["standard_error"] ** 2 + wk["standard_error"] ** 2 - iid ** 2)
    se = float(np.sqrt(variance))
    mean = float(values.mean())
    return {
        "mean": mean, "standard_error": se,
        "95_interval": [mean - 1.96 * se, mean + 1.96 * se],
        "method": "two-way household and week cluster robust",
        "household_cluster_standard_error": hh["standard_error"],
        "week_cluster_standard_error": wk["standard_error"],
        "trip_iid_standard_error": iid,
        "households": int(len(np.unique(household))),
        "weeks": int(len(np.unique(week))),
    }


def score_with_panel(model, data, source_features, dev, trips, parameters, level,
                     reliable, item_category, chunk):
    batcher = Batcher(
        data, clone_features(source_features, dev), model.nmax, include_recency=False)
    return score_model(
        model, batcher, trips, parameters, level, reliable, item_category, chunk)[0]


def main():
    args = arguments()
    torch.set_num_threads(args.threads)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    log_path = output.with_suffix(".log")
    with contextlib.redirect_stdout(Tee(sys.stdout, log_path)), \
            contextlib.redirect_stderr(Tee(sys.stderr, log_path)):
        checkpoint = args.checkpoint if args.checkpoint.is_absolute() else ROOT / args.checkpoint
        fit_path = args.fit_report if args.fit_report.is_absolute() else ROOT / args.fit_report
        fitted = json.loads(fit_path.read_text())
        data = build()
        model, blob, _ = load_checkpoint(
            checkpoint, data, required_capabilities=("conditional_nonempty_incidence",))
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        reliable_np, item_category_np, category_names, support = reliable_price_support(model.J)
        if category_names != fitted["selected"]["category_names"]:
            raise RuntimeError("fit report category order differs from current data")
        reliable = torch.as_tensor(reliable_np, dtype=torch.bool)
        item_category = torch.as_tensor(item_category_np, dtype=torch.long)
        parameters = np.asarray(fitted["selected"]["parameters"], dtype=np.float64)
        level = str(fitted["selected"]["level"])
        source = Features(
            int(data["n_item"]), int(data["n_store"]),
            include_recency=False, include_store_price=False)
        original = source.dev.numpy().copy()
        rng = np.random.default_rng(args.seed)
        report = {
            "status": "running", "fit_report": str(fit_path),
            "fit_report_sha256": file_sha256(fit_path),
            "checkpoint_sha256": file_sha256(checkpoint),
            "placebo_replicates": args.placebo_replicates,
            "seed": args.seed, "price_support": support, "splits": {},
        }
        for split, code in (("validation", 1), ("test", 2)):
            trips = supported_trips(data, code, model.nmax)
            baseline = np.load(
                Path(fitted["artifacts"]["per_trip"]))[f"{split}_baseline"]
            actual = score_with_panel(
                model, data, source, original, trips, parameters, level,
                reliable, item_category, args.chunk)
            if not np.allclose(
                    actual, np.load(Path(fitted["artifacts"]["per_trip"]))[
                        f"{split}_selected"], atol=1e-10, rtol=0):
                raise RuntimeError("could not exactly replay fitted price scores")
            gain = actual - baseline
            week = data["trip_week"][trips]
            household = data["trip_user"][trips]

            # Remove product-specific price variation but preserve each category/week's
            # common price level.  If this performs as well as actual prices, the fitted
            # term is largely a time/category proxy rather than an own-product signal.
            common = original.copy()
            for category in np.unique(item_category_np[reliable_np]):
                rows = np.flatnonzero(reliable_np & (item_category_np == category))
                common[rows] = original[rows].mean(0, keepdims=True)
            common_score = score_with_panel(
                model, data, source, common, trips, parameters, level,
                reliable, item_category, args.chunk)

            product_placebos = []
            time_placebos = []
            for replicate in range(args.placebo_replicates):
                permuted = original.copy()
                for category in np.unique(item_category_np[reliable_np]):
                    rows = np.flatnonzero(reliable_np & (item_category_np == category))
                    permuted[rows] = original[rng.permutation(rows)]
                product_score = score_with_panel(
                    model, data, source, permuted, trips, parameters, level,
                    reliable, item_category, args.chunk)
                product_placebos.append(float((product_score - baseline).mean()))

                weeks = original.reshape(model.J, -1, 7)
                order = rng.permutation(weeks.shape[1])
                time_panel = weeks[:, order, :].reshape(original.shape)
                time_score = score_with_panel(
                    model, data, source, time_panel, trips, parameters, level,
                    reliable, item_category, args.chunk)
                time_placebos.append(float((time_score - baseline).mean()))
                print(f"[price-alignment] {split} placebo {replicate + 1}/"
                      f"{args.placebo_replicates}", flush=True)
            actual_gain = float(gain.mean())
            report["splits"][split] = {
                "contexts": int(len(trips)),
                "actual_gain_over_zero_price": actual_gain,
                "two_way_clustered_gain": two_way_summary(gain, household, week),
                "week_means": {
                    str(int(value)): float(gain[week == value].mean())
                    for value in np.unique(week)},
                "category_week_common_price_gain": float((common_score - baseline).mean()),
                "actual_minus_category_week_common": paired_score_summary(
                    actual - common_score, household),
                "product_alignment_placebo": {
                    "mean_gains": product_placebos,
                    "maximum_gain": float(max(product_placebos)),
                    "actual_exceeds_every_placebo": bool(actual_gain > max(product_placebos)),
                    "one_sided_randomization_p_upper_bound": float(
                        (1 + np.sum(np.asarray(product_placebos) >= actual_gain))
                        / (1 + args.placebo_replicates)),
                },
                "time_alignment_placebo": {
                    "mean_gains": time_placebos,
                    "maximum_gain": float(max(time_placebos)),
                    "actual_exceeds_every_placebo": bool(actual_gain > max(time_placebos)),
                    "one_sided_randomization_p_upper_bound": float(
                        (1 + np.sum(np.asarray(time_placebos) >= actual_gain))
                        / (1 + args.placebo_replicates)),
                },
            }
        report["status"] = "completed"
        output.write_text(strict_json_dumps(report))
        print(f"[price-alignment] report={output}", flush=True)


if __name__ == "__main__":
    main()
