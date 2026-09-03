#!/usr/bin/env python3
"""Fit the version-4 Gram residual as a 28-parameter projected Fisher problem.

For a fixed, cross-fit spectral basis U and C >= 0,

    K = Phi Phi' = U C U'

and the interaction energy is linear in the upper-triangular entries of C.  Exact draws
from the fitted Phi=0 law estimate the score and conditional Fisher covariance.  Ridge is
selected by swapping two context halves; no Smolyak gradient or long SGD trajectory is
used here.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch

from audit_particle_counterfactual_generation import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from fit_interaction_particles import supported_trips
from tempered_block_gibbs import conditional_slots_repeated


torch.set_default_dtype(torch.float64)


def upper_indices(rank):
    return [(i, j) for i in range(rank) for j in range(i, rank)]


def basket_feature(items, basis, upper):
    items = torch.unique(items)
    rows = basis[items]
    total = rows.sum(0)
    projected_pair = torch.outer(total, total) - rows.T @ rows
    # 1/2 tr(Q C): diagonal feature is Qkk/2 and off-diagonal feature is Qkl.
    return torch.stack([
        0.5 * projected_pair[i, j] if i == j else projected_pair[i, j]
        for i, j in upper])


def unpack(vector, rank, upper):
    matrix = np.zeros((rank, rank), dtype=np.float64)
    for value, (i, j) in zip(vector, upper):
        matrix[i, j] = matrix[j, i] = value
    return matrix


def pack(matrix, upper):
    return np.asarray([matrix[i, j] for i, j in upper], dtype=np.float64)


def psd_candidate(score, fisher, ridge, rank, upper, spectral_max):
    regularized = fisher + ridge * np.eye(len(score))
    raw = np.linalg.solve(regularized, score)
    matrix = unpack(raw, rank, upper)
    value, vector = np.linalg.eigh((matrix + matrix.T) * 0.5)
    value = np.clip(value, 0.0, None)
    if value.max(initial=0.0) > spectral_max ** 2:
        value *= spectral_max ** 2 / value.max()
    matrix = (vector * value[None, :]) @ vector.T
    return pack(matrix, upper), matrix, value


def predicted(score, fisher, candidate):
    return float(score @ candidate - 0.5 * candidate @ fisher @ candidate)


def accumulator(dimension):
    return {
        "contexts": 0,
        "score_sum": np.zeros(dimension, dtype=np.float64),
        "fisher_sum": np.zeros((dimension, dimension), dtype=np.float64),
    }


def finalize(value):
    count = float(value["contexts"])
    return value["score_sum"] / count, value["fisher_sum"] / count


def atomic_save(path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--spectral", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=10000,
                        help="training contexts; 0 uses the complete supported population")
    parser.add_argument("--draws", type=int, default=8)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--score-mass", type=float, default=0.99)
    parser.add_argument("--spectral-max", type=float, default=1.0)
    parser.add_argument("--ridges", type=float, nargs="+",
                        default=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2,
                                 3e-2, 1e-1, 3e-1, 1.0])
    parser.add_argument("--minimum-crossfit-gain", type=float, default=0.005)
    parser.add_argument("--minimum-half-gain", type=float, default=0.0,
                        help="minimum gain required in each swapped-half direction")
    parser.add_argument("--seed", type=int, default=28401)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("out/v3_projected_fisher_rank7.pt"))
    args = parser.parse_args()
    if args.draws < 2:
        raise ValueError("Fisher covariance requires at least two draws per context")
    torch.set_num_threads(args.threads)
    data = build()
    parent = args.parent if args.parent.is_absolute() else ROOT / args.parent
    spectral_path = (args.spectral if args.spectral.is_absolute()
                     else ROOT / args.spectral)
    model, parent_blob, meta = load_checkpoint(parent, data)
    if float(model.phi.detach().abs().max()) != 0.0:
        raise RuntimeError("projected Fisher fit requires a Phi=0 parent")
    spectral = np.load(spectral_path)
    raw_basis = np.asarray(spectral["eigenvectors"][:, :args.rank], dtype=np.float64)
    values = np.asarray(spectral["eigenvalues"][:args.rank], dtype=np.float64)
    row_mass = (np.square(raw_basis) * np.clip(values, 0.0, None)[None, :]).sum(1)
    order = np.argsort(row_mass)[::-1]
    cumulative = np.cumsum(row_mass[order]) / row_mass.sum()
    # Roundoff can leave cumulative[-1] a few ulps below one.  A requested mass of
    # exactly 1.0 must mean the complete catalogue, not the impossible J + 1 rows.
    keep_count = min(
        len(row_mass), int(np.searchsorted(cumulative, args.score_mass) + 1))
    keep = np.zeros(len(row_mass), dtype=bool)
    keep[order[:keep_count]] = True
    raw_basis[~keep] = 0.0
    # Masking perturbs orthogonality. QR supplies an orthonormal basis for exactly the
    # retained subspace, so eigenvalues of C equal squared singular values of Phi.
    basis_np, _ = np.linalg.qr(raw_basis)
    basis = torch.as_tensor(basis_np, dtype=model.phi.dtype)
    upper = upper_indices(args.rank)
    dimension = len(upper)

    train_population = supported_trips(data, 0, int(meta["nmax"]))
    context_count = len(train_population) if args.contexts == 0 else args.contexts
    if context_count < 1:
        raise ValueError("contexts must be positive or zero for the full population")
    if context_count > len(train_population):
        raise ValueError("requested more contexts than the training population")
    rng = np.random.default_rng(args.seed)
    trips = train_population[
        rng.permutation(len(train_population))[:context_count]]
    half = rng.random(len(trips)) < 0.5
    if half.all() or (~half).all():
        raise RuntimeError("degenerate cross-fit split")
    accumulators = [accumulator(dimension), accumulator(dimension)]
    batcher = Batcher(
        data, Features(int(data["n_item"]), int(data["n_store"]), 712),
        int(meta["nmax"]))
    generator = torch.Generator().manual_seed(args.seed + 1)
    for start in range(0, len(trips), args.batch):
        sub = trips[start:start + args.batch]
        ix, ctx, _line_ctx, house, li, lt, _lc, _lq = batcher.make(sub)
        model.house, model.ctx = house, ctx
        z = torch.zeros(ix.B, model.Kz, dtype=model.phi.dtype)
        states = conditional_slots_repeated(
            model, ix, z, 0.0, args.draws, generator)
        for local in range(ix.B):
            observed = basket_feature(li[lt == local], basis, upper).numpy()
            generated = np.stack([
                basket_feature(ix.item[draw[local]], basis, upper).numpy()
                for draw in states], axis=0)
            expected = generated.mean(0)
            centred = generated - expected
            covariance = centred.T @ centred / float(args.draws - 1)
            group = int(not half[start + local])
            accumulators[group]["contexts"] += 1
            accumulators[group]["score_sum"] += observed - expected
            accumulators[group]["fisher_sum"] += covariance
        if (start // args.batch + 1) % 10 == 0 or start + args.batch >= len(trips):
            print(f"[projected-fisher] {min(start+args.batch,len(trips))}/"
                  f"{len(trips)} contexts", flush=True)

    score_a, fisher_a = finalize(accumulators[0])
    score_b, fisher_b = finalize(accumulators[1])
    score_full = (
        accumulators[0]["score_sum"] + accumulators[1]["score_sum"]) / len(trips)
    fisher_full = (
        accumulators[0]["fisher_sum"] + accumulators[1]["fisher_sum"]) / len(trips)
    ridge_rows = []
    candidates = {}
    for ridge in args.ridges:
        candidate_a, _, _ = psd_candidate(
            score_a, fisher_a, ridge, args.rank, upper, args.spectral_max)
        candidate_b, _, _ = psd_candidate(
            score_b, fisher_b, ridge, args.rank, upper, args.spectral_max)
        gain_ab = predicted(score_b, fisher_b, candidate_a)
        gain_ba = predicted(score_a, fisher_a, candidate_b)
        candidate, matrix, eigenvalues = psd_candidate(
            score_full, fisher_full, ridge, args.rank, upper, args.spectral_max)
        row = {
            "ridge": ridge,
            "a_fit_b_gain": gain_ab,
            "b_fit_a_gain": gain_ba,
            "mean_crossfit_gain": 0.5 * (gain_ab + gain_ba),
            "minimum_crossfit_gain": min(gain_ab, gain_ba),
            "full_predicted_gain": predicted(score_full, fisher_full, candidate),
            "c_eigenvalues": eigenvalues[::-1].tolist(),
        }
        ridge_rows.append(row)
        candidates[ridge] = (candidate, matrix, eigenvalues)
    eligible = [
        row for row in ridge_rows
        if row["minimum_crossfit_gain"] > args.minimum_half_gain
    ]
    if not eligible:
        selected = max(ridge_rows, key=lambda row: row["mean_crossfit_gain"])
    else:
        selected = max(eligible, key=lambda row: row["mean_crossfit_gain"])
    selected_ridge = selected["ridge"]
    candidate, matrix, eigenvalues = candidates[selected_ridge]
    accepted = bool(
        selected["minimum_crossfit_gain"] > args.minimum_half_gain
        and selected["mean_crossfit_gain"] >= args.minimum_crossfit_gain)

    result = {
        "parent": str(parent),
        "parent_iteration": int(parent_blob["iter"]),
        "spectral": str(spectral_path),
        "contexts": int(len(trips)),
        "draws_per_context": args.draws,
        "rank": args.rank,
        "projected_parameters": dimension,
        "interaction_products": keep_count,
        "score_mass": args.score_mass,
        "half_contexts": [accumulators[0]["contexts"],
                          accumulators[1]["contexts"]],
        "ridge_audit": ridge_rows,
        "selected_ridge": selected_ridge,
        "selected_crossfit_gain": selected["mean_crossfit_gain"],
        "selected_minimum_half_gain": selected["minimum_crossfit_gain"],
        "minimum_required_crossfit_gain": args.minimum_crossfit_gain,
        "minimum_required_half_gain": args.minimum_half_gain,
        "accepted_for_smolyak_audit": accepted,
        "candidate_c": matrix.tolist(),
        "candidate_c_eigenvalues": eigenvalues[::-1].tolist(),
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report_path = output.with_suffix(".json")
    report_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not accepted:
        raise RuntimeError(
            "projected Fisher candidate failed the cross-fit gain gate")

    eigval, eigvec = np.linalg.eigh(matrix)
    positive = eigval > max(float(eigval.max()) * 1e-10, 1e-12)
    factor = eigvec[:, positive] * np.sqrt(eigval[positive])[None, :]
    phi = basis_np @ factor
    with torch.no_grad():
        model.phi.zero_()
        model.phi[:, :phi.shape[1]].copy_(torch.as_tensor(
            phi, dtype=model.phi.dtype))
    payload = {
        "format": 3,
        "estimator": "projected_fisher_version4_interaction_initialization",
        "iter": 0,
        "model": model.state_dict(),
        "config": {**parent_blob["config"],
                   "artifact": parent_blob["config"]["artifact"]},
        "parent": str(parent),
        "parent_iteration": int(parent_blob["iter"]),
        "active_rank": int(phi.shape[1]),
        "interaction_products": keep_count,
        "spectral_scales_only": False,
        "recalibrate_basic": False,
        "recalibrate_lam": False,
        "recalibrate_rho0": False,
        "best_validation": None,
        "best_iteration": 0,
        "evaluations": [],
        "records": [],
        "projected_fisher_report": str(report_path),
    }
    atomic_save(output, payload)
    print(f"[projected-fisher] checkpoint: {output}", flush=True)


if __name__ == "__main__":
    main()
