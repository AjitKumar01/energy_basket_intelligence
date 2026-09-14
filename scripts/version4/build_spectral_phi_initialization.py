#!/usr/bin/env python3
"""Build a rank-r Gram initialization from the additive model's pair-score matrix.

At Phi=0 the ordinary gradient with respect to Phi is identically zero.  The local
likelihood change is instead

    ell(Phi) - ell(0) = 1/2 tr(Phi' R Phi) + O(||Phi||^4),

where R is observed minus additive-model expected off-diagonal co-incidence.  Therefore
the leading positive eigenvectors of R are the locally optimal Gram directions.  Expected
co-incidence is estimated with exact draws from the tractable additive law; this is a
one-time score calculation, not a log-normalizer estimator used during training.
Binary basket incidence supplies matrix-vector products without storing product pairs.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch

from checkpoint_io import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from basket_incidence import incidence, off_diagonal_operator
from pipeline_support import supported_trips
from provenance import file_sha256, strict_json_dumps
from tempered_block_gibbs import conditional_slots_repeated


torch.set_default_dtype(torch.float64)


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
    if total == 0:
        return {str(level): 0 for level in (0.90, 0.95, 0.99, 0.999)}, row_mass
    order = np.argsort(row_mass)[::-1]
    cumulative = np.cumsum(row_mass[order]) / max(total, np.finfo(float).tiny)
    return {str(level): min(len(row_mass), int(np.searchsorted(cumulative, level) + 1))
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
    parser.add_argument("--draw-seed", type=int,
                        help="independent proposal RNG, with unchanged contexts/halves")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("out/v3_spectral_phi_initialization.npz"))
    args = parser.parse_args()
    if args.rank < 1 or args.trips < 4 or args.draws < 1 or args.batch < 1:
        parser.error("positive rank/draws/batch and at least four trips are required")
    torch.set_num_threads(args.threads)
    data = build()
    parent = args.parent if args.parent.is_absolute() else ROOT / args.parent
    model, blob, meta = load_checkpoint(
        parent, data, required_capabilities=("conditional_nonempty_incidence",))
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
    features = Features(int(data["n_item"]), int(data["n_store"]), 712,
                        include_recency=False)
    batcher = Batcher(data, features, int(meta["nmax"]), include_recency=False)
    draw_seed = args.seed + 1 if args.draw_seed is None else args.draw_seed
    generator = torch.Generator().manual_seed(draw_seed)
    generated, generated_half = [], []
    for start in range(0, len(trips), args.batch):
        sub = trips[start:start + args.batch]
        ix, ctx, line_ctx, house, *_ = batcher.make(sub)
        model.house, model.ctx = house, ctx
        z = torch.zeros(ix.B, model.Kz, dtype=model.phi.dtype)
        states = conditional_slots_repeated(model, ix, z, 0.0, args.draws, generator)
        for draw in states:
            for local, slots in enumerate(draw):
                item = torch.unique(ix.item[slots]).cpu().numpy()
                generated.append(item.astype(np.int32))
                generated_half.append(half_a[start + local])
        if (start // args.batch + 1) % 25 == 0 or start + args.batch >= len(trips):
            print(f"[spectral-score] generated {min(start + args.batch, len(trips))}/"
                  f"{len(trips)} contexts", flush=True)

    n_item = int(data["n_item"])
    ptr = data["line_ptr"]
    observed_x = incidence([data["line_item"][ptr[t]:ptr[t + 1]] for t in trips], n_item)
    expected_x = incidence(generated, n_item)
    generated_half = np.asarray(generated_half, dtype=bool)
    del generated
    observed = [observed_x, observed_x[half_a], observed_x[~half_a]]
    expected = [expected_x, expected_x[generated_half], expected_x[~generated_half]]
    eig = []
    for i in range(3):
        score = off_diagonal_operator(observed[i]) - off_diagonal_operator(expected[i])
        eig.append(leading(score, min(n_item - 1, max(args.rank + 4, 12)),
                           args.seed + 10 + i))
    values, vectors = eig[0]
    positive = values > 0
    full_positive = int(positive.sum())
    stored_rank = min(args.rank, full_positive)
    selected_values = values[:stored_rank]
    selected_vectors = vectors[:, :stored_rank]
    counts, row_mass = mass_counts(selected_vectors, selected_values)
    rank_stability = {}
    for candidate_rank in range(1, args.rank + 1):
        if candidate_rank > full_positive:
            rank_stability[str(candidate_rank)] = {
                "split_half_subspace_cosines": [],
                "split_half_mean_squared_subspace_overlap": 0.0,
                "enough_positive_full_directions": False,
                "enough_positive_half_directions": False,
                "accepted": False,
                "rejection_reason": "insufficient positive full-score directions",
            }
            continue
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
            "enough_positive_full_directions": True,
            "enough_positive_half_directions": bool(
                half_rank == candidate_rank),
            "accepted": bool(half_rank == candidate_rank
                             and candidate_score >= args.minimum_stability),
        }
    largest_stable_rank = max(
        (rank for rank in range(1, args.rank + 1)
         if rank_stability[str(rank)]["accepted"]), default=None)
    selected_profile = rank_stability[
        str(largest_stable_rank if largest_stable_rank is not None else 1)]
    overlap = np.asarray(selected_profile["split_half_subspace_cosines"])
    overlap_score = selected_profile[
        "split_half_mean_squared_subspace_overlap"]
    accepted = largest_stable_rank is not None
    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output, eigenvalues=selected_values, eigenvectors=selected_vectors,
        row_mass=row_mass, trips=trips, half_a=half_a,
        parent=np.asarray(str(parent)), parent_iteration=np.asarray(int(blob["iter"])),
        parent_sha256=np.asarray(file_sha256(parent)),
        data_fingerprint_sha256=np.asarray(blob["data_fingerprint_sha256"]))
    basis_digest = file_sha256(output)
    report = {
        "parent": str(parent),
        "parent_iteration": int(blob["iter"]),
        "parent_sha256": file_sha256(parent),
        "data_fingerprint_sha256": blob["data_fingerprint_sha256"],
        "basis_sha256": basis_digest,
        "contexts": int(args.trips),
        "model_draws_per_context": int(args.draws),
        "context_seed": args.seed, "draw_seed": draw_seed,
        "minimum_candidate_rank": 1,
        "leading_full_score_eigenvalues": values.tolist(),
        "requested_maximum_rank": int(args.rank),
        "stored_basis_rank": stored_rank,
        "selected_rank": largest_stable_rank,
        "products_for_cumulative_score_mass": counts,
        "split_half_subspace_cosines": overlap.tolist(),
        "split_half_mean_squared_subspace_overlap": overlap_score,
        "predeclared_stability_threshold": args.minimum_stability,
        "rank_stability": rank_stability,
        "largest_stable_rank": largest_stable_rank,
        "stable_for_scale_profile": accepted,
        "operator": "matrix_free_binary_basket_incidence",
        "observed_incidence_nnz": int(observed_x.nnz),
        "expected_incidence_nnz": int(expected_x.nnz),
        "interpretation": (
            "positive eigenvalues are locally supported PSD Gram directions; split-half "
            "cosines diagnose whether their span is stable enough to train"),
    }
    output.with_suffix(".json").write_text(strict_json_dumps(report))
    print(strict_json_dumps(report), end="")
    if report["largest_stable_rank"] is None:
        print("[spectral-score] no candidate rank passed the predeclared split-half "
              "stability gate", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
