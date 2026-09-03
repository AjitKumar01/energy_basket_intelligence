#!/usr/bin/env python3
"""Build a rank-r Gram initialization from the additive model's pair-score matrix.

At Phi=0 the ordinary gradient with respect to Phi is identically zero.  The local
likelihood change is instead

    ell(Phi) - ell(0) = 1/2 tr(Phi' R Phi) + O(||Phi||^4),

where R is observed minus additive-model expected off-diagonal co-incidence.  Therefore
the leading positive eigenvectors of R are the locally optimal Gram directions.  Expected
co-incidence is estimated with exact draws from the tractable additive law; this is a
one-time score calculation, not a log-normalizer estimator used during training.
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
from fit import Batcher, build_observed_phi_operator
from fit_interaction_particles import supported_trips
from tempered_block_gibbs import conditional_slots_repeated


torch.set_default_dtype(torch.float64)


def generated_operator(row_chunks, col_chunks, n_baskets, n_item):
    from scipy import sparse
    row = np.concatenate(row_chunks) if row_chunks else np.empty(0, dtype=np.int32)
    col = np.concatenate(col_chunks) if col_chunks else np.empty(0, dtype=np.int32)
    value = sparse.coo_matrix(
        (np.ones(len(row), dtype=np.float64),
         (row, col)),
        shape=(n_item, n_item)).tocsr()
    value.sum_duplicates()
    value /= float(n_baskets)
    value.setdiag(0.0)
    value.eliminate_zeros()
    return value


def leading(matrix, count, seed):
    from scipy.sparse.linalg import eigsh
    values, vectors = eigsh(matrix, k=count, which="LA",
                            v0=np.random.default_rng(seed).normal(size=matrix.shape[0]),
                            tol=1e-7, maxiter=5000)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def mass_counts(vectors, values):
    positive = np.clip(values, 0.0, None)
    row_mass = (np.square(vectors) * positive[None, :]).sum(1)
    total = row_mass.sum()
    order = np.argsort(row_mass)[::-1]
    cumulative = np.cumsum(row_mass[order]) / max(total, np.finfo(float).tiny)
    return {str(level): int(np.searchsorted(cumulative, level) + 1)
            for level in (0.90, 0.95, 0.99, 0.999)}, row_mass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--trips", type=int, default=20000)
    parser.add_argument("--draws", type=int, default=2)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--minimum-stability", type=float, default=0.5,
                        help="split-half mean squared overlap required for acceptance")
    parser.add_argument("--seed", type=int, default=26601)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("out/v3_spectral_phi_initialization.npz"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = build()
    parent = args.parent if args.parent.is_absolute() else ROOT / args.parent
    model, blob, meta = load_checkpoint(parent, data)
    if float(model.phi.detach().abs().max()) != 0.0:
        raise RuntimeError("spectral score initialization requires an exact Phi=0 parent")
    train = supported_trips(data, 0, int(meta["nmax"]))
    if args.trips > len(train):
        raise ValueError("requested more score contexts than the training population")
    rng = np.random.default_rng(args.seed)
    trips = train[rng.permutation(len(train))[:args.trips]]
    half_a = rng.random(args.trips) < 0.5
    if half_a.all() or (~half_a).all():
        raise RuntimeError("degenerate split-half assignment")
    batcher = Batcher(data, Features(int(data["n_item"]), int(data["n_store"]), 712),
                      int(meta["nmax"]))
    generator = torch.Generator().manual_seed(args.seed + 1)
    rows = [[], [], []]
    cols = [[], [], []]
    basket_counts = [args.trips * args.draws,
                     int(half_a.sum()) * args.draws,
                     int((~half_a).sum()) * args.draws]
    for start in range(0, len(trips), args.batch):
        sub = trips[start:start + args.batch]
        ix, ctx, line_ctx, house, *_ = batcher.make(sub)
        model.house, model.ctx = house, ctx
        z = torch.zeros(ix.B, model.Kz, dtype=model.phi.dtype)
        states = conditional_slots_repeated(model, ix, z, 0.0, args.draws, generator)
        batch_rows = [[], [], []]
        batch_cols = [[], [], []]
        for draw in states:
            for local, slots in enumerate(draw):
                item = torch.unique(ix.item[slots]).cpu().numpy()
                if len(item) < 2:
                    continue
                left, right = np.triu_indices(len(item), 1)
                a, b = item[left].astype(np.int32), item[right].astype(np.int32)
                pair_row = np.concatenate((a, b))
                pair_col = np.concatenate((b, a))
                groups = (0, 1 if half_a[start + local] else 2)
                for group in groups:
                    batch_rows[group].append(pair_row)
                    batch_cols[group].append(pair_col)
        for group in range(3):
            if batch_rows[group]:
                rows[group].append(np.concatenate(batch_rows[group]))
                cols[group].append(np.concatenate(batch_cols[group]))
        if (start // args.batch + 1) % 25 == 0 or start + args.batch >= len(trips):
            print(f"[spectral-score] generated {min(start + args.batch, len(trips))}/"
                  f"{len(trips)} contexts", flush=True)

    n_item = int(data["n_item"])
    observed = [
        build_observed_phi_operator(data, trips, n_item),
        build_observed_phi_operator(data, trips[half_a], n_item),
        build_observed_phi_operator(data, trips[~half_a], n_item),
    ]
    expected = [generated_operator(rows[i], cols[i], basket_counts[i], n_item)
                for i in range(3)]
    residual = []
    eig = []
    for i in range(3):
        score = (observed[i] - expected[i]).tocsr()
        score = ((score + score.T) * 0.5).tocsr()
        score.setdiag(0.0); score.eliminate_zeros()
        residual.append(score)
        eig.append(leading(score, max(args.rank + 4, 12), args.seed + 10 + i))
    values, vectors = eig[0]
    positive = values > 0
    if int(positive.sum()) < args.rank:
        raise RuntimeError(f"only {int(positive.sum())} positive pair-score directions")
    selected_values = values[:args.rank]
    selected_vectors = vectors[:, :args.rank]
    counts, row_mass = mass_counts(selected_vectors, selected_values)
    rank_stability = {}
    for candidate_rank in range(4, args.rank + 1):
        half_rank = min(candidate_rank, int((eig[1][0] > 0).sum()),
                        int((eig[2][0] > 0).sum()))
        candidate_overlap = np.linalg.svd(
            eig[1][1][:, :half_rank].T @ eig[2][1][:, :half_rank],
            compute_uv=False)
        candidate_score = (float(np.square(candidate_overlap).mean())
                           if len(candidate_overlap) else 0.0)
        rank_stability[str(candidate_rank)] = {
            "split_half_subspace_cosines": candidate_overlap.tolist(),
            "split_half_mean_squared_subspace_overlap": candidate_score,
            "enough_positive_half_directions": bool(
                half_rank == candidate_rank),
            "accepted": bool(half_rank == candidate_rank
                             and candidate_score >= args.minimum_stability),
        }
    selected_profile = rank_stability[str(args.rank)]
    overlap = np.asarray(selected_profile["split_half_subspace_cosines"])
    overlap_score = selected_profile[
        "split_half_mean_squared_subspace_overlap"]
    accepted = selected_profile["accepted"]
    output = args.output if args.output.is_absolute() else ROOT / args.output
    np.savez_compressed(
        output, eigenvalues=selected_values, eigenvectors=selected_vectors,
        row_mass=row_mass, trips=trips, half_a=half_a,
        parent=np.asarray(str(parent)), parent_iteration=np.asarray(int(blob["iter"])))
    report = {
        "parent": str(parent),
        "parent_iteration": int(blob["iter"]),
        "contexts": int(args.trips),
        "model_draws_per_context": int(args.draws),
        "leading_full_score_eigenvalues": values.tolist(),
        "selected_rank": int(args.rank),
        "products_for_cumulative_score_mass": counts,
        "split_half_subspace_cosines": overlap.tolist(),
        "split_half_mean_squared_subspace_overlap": overlap_score,
        "predeclared_stability_threshold": args.minimum_stability,
        "rank_stability": rank_stability,
        "stable_for_scale_profile": accepted,
        "observed_pair_nnz": int(observed[0].nnz),
        "expected_pair_nnz": int(expected[0].nnz),
        "residual_pair_nnz": int(residual[0].nnz),
        "interpretation": (
            "positive eigenvalues are locally supported PSD Gram directions; split-half "
            "cosines diagnose whether their span is stable enough to train"),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not accepted:
        print("[spectral-score] rank rejected by the predeclared split-half "
              "stability gate", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
