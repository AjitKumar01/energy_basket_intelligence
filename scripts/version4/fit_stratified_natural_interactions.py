#!/usr/bin/env python3
"""Fit the Version-4 interaction/full-size block with stratified MC-MLE.

The exact additive parent supplies ``P_0(N=n|x)`` and exact conditional basket draws.
Every declared size band receives draws, eliminating the ordinary proposal bank's blind
spot for rare large baskets.  Fixed draws make the sampled natural-parameter objective
deterministic and concave.  Independent Smolyak likelihood and population-tail stages
remain mandatory before a resulting checkpoint can be certified.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch
from scipy import sparse

from checkpoint_io import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from fit_convex_natural_interactions import atomic_save, pair_statistic, spectral_basis
from pipeline_support import supported_trips
from provenance import file_sha256, strict_json_dumps
from stratified_natural import (
    StratifiedNaturalBank,
    alternating_solve,
    evaluation_summary,
    linear_size_basis,
    split_parameters,
)
from tempered_block_gibbs import conditional_slots_stratified, default_size_bands


torch.set_default_dtype(torch.float64)


def category_entries(category: torch.Tensor, n_category: int
                     ) -> tuple[np.ndarray, np.ndarray]:
    count = torch.bincount(category, minlength=n_category).cpu().numpy()
    index = np.flatnonzero(count >= 2)
    value = -count[index] * (count[index] - 1) / 2.0
    return index.astype(np.int32), value.astype(np.float64)


def sparse_rows(indices: list[np.ndarray], values: list[np.ndarray], columns: int
                ) -> sparse.csr_matrix:
    indptr = np.zeros(len(indices) + 1, dtype=np.int64)
    indptr[1:] = np.cumsum([len(row) for row in indices])
    return sparse.csr_matrix(
        (np.concatenate(values) if values else np.empty(0, dtype=np.float64),
         np.concatenate(indices) if indices else np.empty(0, dtype=np.int32),
         indptr), shape=(len(indices), columns))


def save_bank(path: Path, bank: StratifiedNaturalBank, trips: np.ndarray,
              half: np.ndarray, band_of_draw: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, observed_pair=bank.observed_pair, draw_pair=bank.draw_pair,
        observed_size=bank.observed_size, draw_size=bank.draw_size,
        observed_category_data=bank.observed_category.data,
        observed_category_indices=bank.observed_category.indices,
        observed_category_indptr=bank.observed_category.indptr,
        observed_category_shape=np.asarray(bank.observed_category.shape),
        draw_category_data=bank.draw_category.data,
        draw_category_indices=bank.draw_category.indices,
        draw_category_indptr=bank.draw_category.indptr,
        draw_category_shape=np.asarray(bank.draw_category.shape),
        log_draw_weight=bank.log_draw_weight, n_size=np.asarray(bank.n_size),
        size_basis=bank.size_basis, trips=trips, half=half,
        band_of_draw=band_of_draw,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)))


def load_bank(path: Path, expected: dict, trips: np.ndarray, half: np.ndarray
              ) -> tuple[StratifiedNaturalBank, np.ndarray]:
    value = np.load(path, allow_pickle=False)
    metadata = json.loads(str(value["metadata"].item()))
    if json.dumps(metadata, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise RuntimeError("stratified bank metadata does not match this fit request")
    if not np.array_equal(value["trips"], trips) or not np.array_equal(value["half"], half):
        raise RuntimeError("stratified bank context manifest or cross-fit split differs")
    observed_category = sparse.csr_matrix(
        (value["observed_category_data"], value["observed_category_indices"],
         value["observed_category_indptr"]),
        shape=tuple(value["observed_category_shape"].tolist()))
    draw_category = sparse.csr_matrix(
        (value["draw_category_data"], value["draw_category_indices"],
         value["draw_category_indptr"]),
        shape=tuple(value["draw_category_shape"].tolist()))
    bank = StratifiedNaturalBank(
        value["observed_pair"], value["draw_pair"], value["observed_size"],
        value["draw_size"], observed_category, draw_category,
        value["log_draw_weight"], int(value["n_size"]), value["size_basis"])
    return bank, np.asarray(value["band_of_draw"], dtype=np.int64)


@torch.no_grad()
def build_bank(model, batcher, trips: np.ndarray, basis: torch.Tensor,
               bands: list[tuple[int, int]], allocation: list[int], batch: int,
               generator: torch.Generator,
               size_basis: np.ndarray, *, fit_categories: bool = False
               ) -> tuple[StratifiedNaturalBank, np.ndarray]:
    contexts = len(trips)
    draws = int(sum(allocation))
    rank = basis.shape[1]
    pair_width = rank * rank
    observed_pair = np.empty((contexts, pair_width), dtype=np.float64)
    draw_pair = np.empty((contexts, draws, pair_width), dtype=np.float64)
    observed_size = np.empty(contexts, dtype=np.int64)
    draw_size = np.empty((contexts, draws), dtype=np.int64)
    log_weight = np.empty((contexts, draws), dtype=np.float64)
    observed_category_index: list[np.ndarray] = []
    observed_category_value: list[np.ndarray] = []
    draw_category_index: list[np.ndarray] = []
    draw_category_value: list[np.ndarray] = []
    band_of_draw = None
    for start in range(0, contexts, batch):
        sub = trips[start:start + batch]
        ix, ctx, _line_ctx, house, li, lt, lc, _lq = batcher.make(sub)
        model.house, model.ctx = house, ctx
        z = torch.zeros(ix.B, model.Kz, dtype=model.phi.dtype)
        states, batch_log_weight, batch_band = conditional_slots_stratified(
            model, ix, z, 0.0, bands, allocation, generator)
        if band_of_draw is None:
            band_of_draw = batch_band
        elif not np.array_equal(band_of_draw, batch_band):
            raise RuntimeError("stratified draw ordering changed between batches")
        log_weight[start:start + ix.B] = batch_log_weight.cpu().numpy()
        for local in range(ix.B):
            output_row = start + local
            items = li[lt == local]
            observed_pair[output_row] = pair_statistic(items, basis).reshape(-1)
            observed_size[output_row] = int(torch.unique(items).numel()) - 1
            if fit_categories:
                ci, cv = category_entries(lc[lt == local], model.C)
                observed_category_index.append(ci)
                observed_category_value.append(cv)
            else:
                observed_category_index.append(np.empty(0, dtype=np.int32))
                observed_category_value.append(np.empty(0, dtype=np.float64))
            for draw_index, state in enumerate(states):
                slots = state[local]
                draw_items = ix.item[slots]
                draw_pair[output_row, draw_index] = pair_statistic(
                    draw_items, basis).reshape(-1)
                draw_size[output_row, draw_index] = int(slots.numel()) - 1
                if fit_categories:
                    rows = ix.row_of[slots]
                    ci, cv = category_entries(ix.row_cat[rows], model.C)
                    draw_category_index.append(ci)
                    draw_category_value.append(cv)
                else:
                    draw_category_index.append(np.empty(0, dtype=np.int32))
                    draw_category_value.append(np.empty(0, dtype=np.float64))
        if ((start // batch + 1) % 10 == 0 or start + batch >= contexts):
            print(f"[stratified-natural] sampled {min(start + batch, contexts)}/"
                  f"{contexts} contexts", flush=True)
    if band_of_draw is None:
        raise ValueError("cannot build an empty stratified bank")
    bank = StratifiedNaturalBank(
        observed_pair, draw_pair, observed_size, draw_size,
        sparse_rows(observed_category_index, observed_category_value,
                    model.C if fit_categories else 0),
        sparse_rows(draw_category_index, draw_category_value,
                    model.C if fit_categories else 0),
        log_weight, model.nmax, size_basis)
    return bank, band_of_draw


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--spectral", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=12000)
    parser.add_argument("--band-draws", type=int, nargs="+",
                        default=[16, 16, 12, 8, 5, 4, 3])
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--score-mass", type=float, default=1.0)
    parser.add_argument("--spectral-max", type=float, default=1.0)
    parser.add_argument("--ridges", type=float, nargs="+",
                        default=[3e-4, 1e-3, 3e-3])
    parser.add_argument("--category-ridge", type=float, default=1e-3)
    parser.add_argument("--size-ridge", type=float, default=1e-3)
    parser.add_argument("--size-smoothness", type=float, default=1e-1)
    parser.add_argument("--size-knots", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5, 7, 10, 15, 25, 40, 70, 120])
    parser.add_argument(
        "--category-bound", type=float, default=0.0,
        help=("0 freezes the already fitted category penalty and omits its 300 "
              "coordinates from the draw bank; positive values enable the "
              "experimental joint category update"))
    parser.add_argument("--size-bound", type=float, default=12.0)
    parser.add_argument("--max-iterations", type=int, default=300,
                        help="maximum L-BFGS iterations per nuisance-block solve")
    parser.add_argument("--max-outer-iterations", type=int, default=20)
    parser.add_argument("--pair-steps", type=int, default=100)
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--minimum-crossfit-gain", type=float, default=0.005)
    parser.add_argument("--minimum-half-gain", type=float, default=0.0)
    parser.add_argument("--minimum-within-band-ess-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=40701)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/candidate_stratified.pt"))
    parser.add_argument("--bank-cache", type=Path,
                        help="fixed draw-bank cache; defaults beside --output")
    parser.add_argument(
        "--reuse-bank", action="store_true",
        help="require an existing compatible bank; fail instead of rebuilding")
    parser.add_argument(
        "--rebuild-bank", action="store_true",
        help="ignore any compatible cache and resample the fixed bank")
    args = parser.parse_args()
    if args.contexts < 4:
        raise ValueError("at least four contexts are required")
    if args.category_bound < 0:
        raise ValueError("--category-bound must be nonnegative")
    if args.reuse_bank and args.rebuild_bank:
        raise ValueError("--reuse-bank and --rebuild-bank are mutually exclusive")
    torch.set_num_threads(args.threads)
    data = build()
    parent = args.parent if args.parent.is_absolute() else ROOT / args.parent
    spectral_path = args.spectral if args.spectral.is_absolute() else ROOT / args.spectral
    model, parent_blob, _meta = load_checkpoint(
        parent, data, required_capabilities=("conditional_nonempty_incidence",))
    if float(model.phi.detach().abs().max()) != 0.0:
        raise RuntimeError("stratified proposal must be the exact Phi=0 additive parent")
    basis_np, keep_count = spectral_basis(spectral_path, args.rank, args.score_mass)
    spectral_report = json.loads(spectral_path.with_suffix(".json").read_text())
    if spectral_report.get("basis_sha256") != file_sha256(spectral_path):
        raise RuntimeError("spectral basis failed its report content digest")
    if spectral_report.get("parent_sha256") != file_sha256(parent):
        raise RuntimeError("spectral basis and additive parent do not share lineage")
    basis = torch.as_tensor(basis_np, dtype=model.phi.dtype)
    bands = default_size_bands(model.nmax)
    if args.size_knots[-1] != model.nmax:
        raise ValueError("the last --size-knots value must equal model nmax")
    size_basis = linear_size_basis(model.nmax, args.size_knots)
    if len(args.band_draws) != len(bands):
        raise ValueError(f"--band-draws needs {len(bands)} values for bands {bands}")
    population = supported_trips(data, 0, model.nmax)
    if args.contexts > len(population):
        raise ValueError("requested more contexts than the training population")
    rng = np.random.default_rng(args.seed)
    trips = population[rng.permutation(len(population))[:args.contexts]]
    half = rng.random(args.contexts) < 0.5
    if half.all() or (~half).all():
        raise RuntimeError("degenerate cross-fit split")
    output = args.output if args.output.is_absolute() else ROOT / args.output
    cache = (args.bank_cache if args.bank_cache is not None
             else output.with_suffix(".bank.npz"))
    cache = cache if cache.is_absolute() else ROOT / cache
    bank_metadata = {
        "parent_sha256": file_sha256(parent),
        "spectral_sha256": file_sha256(spectral_path),
        "contexts": args.contexts, "rank": args.rank,
        "bands": bands, "draws_per_band": args.band_draws,
        "size_knots": args.size_knots, "seed": args.seed,
        "fit_categories": bool(args.category_bound > 0),
    }
    bank = band_of_draw = None
    if cache.exists() and not args.rebuild_bank:
        try:
            bank, band_of_draw = load_bank(cache, bank_metadata, trips, half)
            print(f"[stratified-natural] reused fixed bank: {cache}", flush=True)
        except RuntimeError as error:
            if args.reuse_bank:
                raise
            print(f"[stratified-natural] incompatible cache will be rebuilt: "
                  f"{error}", flush=True)
    elif args.reuse_bank:
        raise FileNotFoundError(f"required stratified bank does not exist: {cache}")
    if bank is None:
        batcher = Batcher(
            data, Features(int(data["n_item"]), int(data["n_store"]), 712,
                           include_recency=False), model.nmax, include_recency=False)
        bank, band_of_draw = build_bank(
            model, batcher, trips, basis, bands, args.band_draws, args.batch,
            torch.Generator().manual_seed(args.seed + 1), size_basis,
            fit_categories=args.category_bound > 0)
        save_bank(cache, bank, trips, half, band_of_draw, bank_metadata)
        print(f"[stratified-natural] cached fixed bank: {cache}", flush=True)
    bank_a = bank.subset(np.flatnonzero(half))
    bank_b = bank.subset(np.flatnonzero(~half))
    rows, candidates = [], {}
    for ridge in args.ridges:
        fit_a, solve_a = alternating_solve(
            bank_a, args.rank, spectral_max=args.spectral_max,
            category_bound=args.category_bound, size_bound=args.size_bound,
            interaction_ridge=ridge, category_ridge=args.category_ridge,
            size_ridge=args.size_ridge, size_smoothness=args.size_smoothness,
            nuisance_iterations=args.max_iterations,
            max_outer_iterations=args.max_outer_iterations,
            pair_steps=args.pair_steps, tolerance=args.tolerance,
            label=f"stratified-ridge-{ridge:g}-a")
        fit_b, solve_b = alternating_solve(
            bank_b, args.rank, spectral_max=args.spectral_max,
            category_bound=args.category_bound, size_bound=args.size_bound,
            interaction_ridge=ridge, category_ridge=args.category_ridge,
            size_ridge=args.size_ridge, size_smoothness=args.size_smoothness,
            nuisance_iterations=args.max_iterations,
            max_outer_iterations=args.max_outer_iterations,
            pair_steps=args.pair_steps, tolerance=args.tolerance,
            label=f"stratified-ridge-{ridge:g}-b")
        a_on_b = evaluation_summary(fit_a, bank_b, band_of_draw)
        b_on_a = evaluation_summary(fit_b, bank_a, band_of_draw)
        row = {
            "ridge": float(ridge), "a_fit_b": a_on_b, "b_fit_a": b_on_a,
            "mean_crossfit_gain": 0.5 * (a_on_b["gain"] + b_on_a["gain"]),
            "minimum_crossfit_gain": min(a_on_b["gain"], b_on_a["gain"]),
            "minimum_within_band_ess_fraction": min(
                a_on_b["minimum_within_band_ess_fraction"],
                b_on_a["minimum_within_band_ess_fraction"]),
            "solve_a": solve_a, "solve_b": solve_b,
        }
        rows.append(row)
        candidates[float(ridge)] = (fit_a, fit_b)
    eligible = [row for row in rows if row["minimum_crossfit_gain"] >
                args.minimum_half_gain]
    selected = max(eligible or rows, key=lambda row: row["mean_crossfit_gain"])
    vector, solve = alternating_solve(
        bank, args.rank, spectral_max=args.spectral_max,
        category_bound=args.category_bound, size_bound=args.size_bound,
        interaction_ridge=selected["ridge"], category_ridge=args.category_ridge,
        size_ridge=args.size_ridge, size_smoothness=args.size_smoothness,
        nuisance_iterations=args.max_iterations,
        max_outer_iterations=args.max_outer_iterations,
        pair_steps=args.pair_steps, tolerance=args.tolerance,
        label="stratified-full")
    full = evaluation_summary(vector, bank, band_of_draw)
    accepted = bool(
        selected["minimum_crossfit_gain"] > args.minimum_half_gain
        and selected["mean_crossfit_gain"] >= args.minimum_crossfit_gain
        and selected["minimum_within_band_ess_fraction"] >=
        args.minimum_within_band_ess_fraction
        and solve["converged"] and solve["accepted_steps_monotone"])
    pair, fitted_category_delta, size_coefficient = split_parameters(vector, bank)
    category_delta = (fitted_category_delta if bank.categories
                      else np.zeros(model.C, dtype=np.float64))
    size_delta = size_basis @ size_coefficient
    c_matrix = pair.reshape(args.rank, args.rank)
    eigenvalue, eigenvector = np.linalg.eigh(c_matrix)
    positive = eigenvalue > max(float(eigenvalue.max()) * 1e-10, 1e-12)
    factor = eigenvector[:, positive] * np.sqrt(eigenvalue[positive])[None, :]
    phi = basis_np @ factor
    report = {
        "method": "size_stratified_sparse_natural_mcle",
        "parent": str(parent), "parent_sha256": file_sha256(parent),
        "parent_iteration": int(parent_blob["iter"]),
        "data_fingerprint_sha256": parent_blob["data_fingerprint_sha256"],
        "spectral": str(spectral_path), "spectral_sha256": file_sha256(spectral_path),
        "rank": args.rank, "active_rank": int(positive.sum()),
        "interaction_products": keep_count, "contexts": args.contexts,
        "size_bands": bands, "draws_per_band": args.band_draws,
        "size_knots": args.size_knots,
        "draws_per_context": int(sum(args.band_draws)),
        "natural_parameters": bank.width,
        "category_update": ("experimental_joint_update" if bank.categories
                            else "frozen_at_additive_parent"),
        "storage": {
            "pair_bytes": int(bank.observed_pair.nbytes + bank.draw_pair.nbytes),
            "category_nonzeros": int(bank.observed_category.nnz + bank.draw_category.nnz),
            "size_representation": "integer lookup indices",
            "bank_cache": str(cache), "bank_cache_sha256": file_sha256(cache),
        },
        "ridge_audit": rows, "selected_ridge": selected["ridge"],
        "selected_crossfit_gain": selected["mean_crossfit_gain"],
        "selected_minimum_half_gain": selected["minimum_crossfit_gain"],
        "selected_minimum_within_band_ess_fraction": selected[
            "minimum_within_band_ess_fraction"],
        "full_solve": solve, "full_fit": full,
        "candidate_c_eigenvalues": eigenvalue[::-1].tolist(),
        "category_delta_quantiles": np.quantile(
            category_delta, [0, .01, .1, .5, .9, .99, 1]).tolist(),
        "size_coefficients_at_knots": size_coefficient.tolist(),
        "size_delta": size_delta.tolist(),
        "penalty": {
            "category_ridge": args.category_ridge,
            "size_ridge": args.size_ridge,
            "size_second_difference": args.size_smoothness,
        },
        "accepted_for_independent_smolyak_audit": accepted,
        "required_next_gates": [
            "independent Smolyak likelihood value and score audit",
            "complete-population localized and aggregate tail audit",
            "generation calibration audit",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(strict_json_dumps(report))
    print(strict_json_dumps(report), end="")
    if not accepted:
        raise RuntimeError("stratified natural candidate failed cross-fit or ESS gates")
    with torch.no_grad():
        model.phi.zero_()
        model.phi[:, :phi.shape[1]].copy_(torch.as_tensor(phi, dtype=model.phi.dtype))
        model.rho_c.add_(torch.as_tensor(category_delta, dtype=model.rho_c.dtype))
        model.rho_0_free.add_(torch.as_tensor(size_delta, dtype=model.rho_0_free.dtype))
    payload = {
        "format": 3,
        "estimator": "size_stratified_sparse_natural_mcle_version4",
        "iter": 0, "model": model.state_dict(),
        "fresh_artifact_digest": parent_blob["fresh_artifact_digest"],
        "data_fingerprint_sha256": parent_blob["data_fingerprint_sha256"],
        "config": parent_blob["config"], "parent": str(parent),
        "parent_iteration": int(parent_blob["iter"]),
        "parent_sha256": file_sha256(parent),
        "spectral_sha256": file_sha256(spectral_path),
        "active_rank": int(phi.shape[1]), "interaction_products": keep_count,
        "best_validation": None, "best_iteration": 0,
        "evaluations": [], "records": [],
        "trained_capabilities": {
            "conditional_nonempty_incidence": True, "gram_interactions": True,
            "recency": False, "quantities": False,
            "arrival_or_null_basket": False,
        },
        "stratified_natural_report": str(output.with_suffix(".json")),
    }
    atomic_save(output, payload)
    print(f"[stratified-natural] accepted checkpoint: {output}", flush=True)


if __name__ == "__main__":
    main()
