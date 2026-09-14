#!/usr/bin/env python3
"""Bounded real-data comparison: convex fixed basis, then anchored free Phi.

No changes to the additive/price/category parameters. Both arms share the fitted
size curve. Validation is never optimized; two independent proposal streams
quantify Monte Carlo sensitivity. Outputs are experimental, never auto-promoted.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch
from scipy import sparse

from basket_incidence import incidence
from checkpoint_io import ROOT, load_checkpoint
from data import build
from embedding_refinement import IncidenceBank, evaluation, fit_embeddings, compact_factor
from features import Features
from fit import Batcher
from fit_convex_natural_interactions import atomic_save, spectral_basis
from pipeline_support import supported_trips
from provenance import file_sha256, strict_json_dumps
from stratified_natural import StratifiedNaturalBank, alternating_solve, linear_size_basis
from tempered_block_gibbs import conditional_slots_stratified, default_size_bands
from uncertainty import paired_score_summary

torch.set_default_dtype(torch.float64)


@torch.no_grad()
def sample_bank(model, batcher, trips, allocation, size_basis, seed, batch=64):
    observed, generated, weights = [], [], []
    generator = torch.Generator().manual_seed(seed)
    bands = default_size_bands(model.nmax)
    band = None
    for start in range(0, len(trips), batch):
        ix, ctx, _, house, li, lt, _, _ = batcher.make(trips[start:start + batch])
        model.house, model.ctx = house, ctx
        states, log_weight, current_band = conditional_slots_stratified(
            model, ix, torch.zeros(ix.B, model.Kz, dtype=model.phi.dtype),
            0.0, bands, allocation, generator)
        if band is not None and not np.array_equal(band, current_band):
            raise RuntimeError("draw band ordering changed")
        band = current_band
        weights.append(log_weight.numpy())
        for local in range(ix.B):
            observed.append(li[lt == local].numpy())
            generated.extend(ix.item[state[local]].numpy() for state in states)
        if start % (batch * 8) == 0 or start + batch >= len(trips):
            print(f"[incidence-bank seed={seed}] {min(start + batch, len(trips))}/{len(trips)}", flush=True)
    products = model.phi.shape[0]
    return IncidenceBank(incidence(observed, products), incidence(generated, products),
                         np.concatenate(weights), band, size_basis)


def projected_bank(bank, basis):
    rank = basis.shape[1]
    diagonal = (basis[:, :, None] * basis[:, None, :]).reshape(len(basis), -1)

    def statistic(x):
        totals = x @ basis
        return 0.5 * ((totals[:, :, None] * totals[:, None, :]).reshape(x.shape[0], -1)
                      - x @ diagonal)

    m, d = bank.log_weight.shape
    return StratifiedNaturalBank(
        statistic(bank.observed), statistic(bank.generated).reshape(m, d, rank * rank),
        bank.observed_size, bank.generated_size.reshape(m, d),
        sparse.csr_matrix((m, 0)), sparse.csr_matrix((m * d, 0)),
        bank.log_weight, len(bank.size_basis), bank.size_basis)


def save_incidence_bank(path, bank, trips, parent_digest, seed):
    values = {"trips": trips, "log_weight": bank.log_weight, "band": bank.band,
              "size_basis": bank.size_basis, "parent_sha256": np.asarray(parent_digest),
              "seed": np.asarray(seed)}
    for name in ("observed", "generated"):
        x = getattr(bank, name)
        values.update({name + "_indices": x.indices, name + "_indptr": x.indptr,
                       name + "_shape": np.asarray(x.shape)})
    np.savez_compressed(path, **values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--spectral", type=Path, required=True)
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--contexts", type=int, default=2048)
    parser.add_argument("--validation-contexts", type=int, default=512)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--anchor-ridge", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=91301)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    if args.contexts < 4 or args.validation_contexts < 4:
        parser.error("at least four training and validation contexts are required")
    data = build()
    parent, spectral = args.parent.resolve(), args.spectral.resolve()
    model, parent_blob, meta = load_checkpoint(parent, data,
        required_capabilities=("conditional_nonempty_incidence",))
    if float(model.phi.detach().abs().max()) != 0:
        raise ValueError("requires the exact additive proposal")
    import json
    sr = json.loads(spectral.with_suffix(".json").read_text())
    if sr["parent_sha256"] != file_sha256(parent) or sr["basis_sha256"] != file_sha256(spectral):
        raise ValueError("spectral/additive lineage mismatch")
    if not sr["rank_stability"].get(str(args.rank), {}).get("accepted"):
        raise ValueError("requested rank failed spectral eligibility")
    if not 1 <= args.rank <= model.Kz:
        raise ValueError("rank outside model support")
    basis, _ = spectral_basis(spectral, args.rank, 1.0)
    size_basis = linear_size_basis(model.nmax, [1, 2, 3, 4, 5, 7, 10, 15, 25, 40, 70, model.nmax])
    rng = np.random.default_rng(args.seed)
    train = supported_trips(data, 0, model.nmax)
    validation = supported_trips(data, 1, model.nmax)
    if args.contexts > len(train) or args.validation_contexts > len(validation):
        raise ValueError("requested context count exceeds the supported population")
    train = rng.permutation(train)[:args.contexts]
    validation = rng.permutation(validation)[:args.validation_contexts]
    np.savez_compressed(output / "context_manifest.npz", train=train, validation=validation,
                         validation_household=data["trip_user"][validation])
    batcher = Batcher(data, Features(int(data["n_item"]), int(data["n_store"]), 712,
                                     include_recency=False), model.nmax, include_recency=False)
    training = sample_bank(model, batcher, train, [16, 16, 12, 8, 16, 16, 16],
                           size_basis, args.seed + 1)
    save_incidence_bank(output / "training_bank.npz", training, train, file_sha256(parent), args.seed + 1)
    natural = projected_bank(training, basis)
    vector, fixed_solve = alternating_solve(
        natural, args.rank, spectral_max=1.0, category_bound=0.0, size_bound=12.0,
        interaction_ridge=0.001, category_ridge=0.001, size_ridge=0.001,
        size_smoothness=0.1, nuisance_iterations=300, max_outer_iterations=20,
        pair_steps=100, tolerance=0.001, label="pilot-convex-fixed")
    value, direction = np.linalg.eigh(vector[:args.rank**2].reshape(args.rank, args.rank))
    phi_fixed = basis @ (direction * np.sqrt(np.maximum(value, 0)))
    size = vector[args.rank**2:]
    zero_fixed = np.linalg.norm(phi_fixed) < 1e-10
    # At zero the Phi gradient vanishes. A predeclared small spectral seed allows
    # the experimental free arm to be tested even if the constrained optimum is zero.
    initial_phi = 0.05 * basis if zero_fixed else phi_fixed
    del natural
    phi_free, _, free_solve = fit_embeddings(
        training, initial_phi, initial_size=size, anchor=phi_fixed, fit_size=False,
        steps=args.steps, tolerance=1e-5, anchor_ridge=args.anchor_ridge,
        gram_ridge=0.001, label="pilot-free-phi")
    free_solve["outside_initial_span_norm"] = float(np.linalg.norm(phi_free - basis @ (basis.T @ phi_free)))
    training_evaluation = {name: evaluation(training, phi, size, data["trip_user"][train])
                           for name, phi in (("fixed", phi_fixed), ("free", phi_free))}
    del training
    reports, paired_gains = [], []
    for replicate in range(2):
        draw_seed = args.seed + 101 + replicate
        bank = sample_bank(model, batcher, validation, [32] * 7, size_basis, draw_seed)
        save_incidence_bank(output / f"validation_bank_{replicate}.npz", bank, validation,
                             file_sha256(parent), draw_seed)
        fixed = evaluation(bank, phi_fixed, size, data["trip_user"][validation])
        free = evaluation(bank, phi_free, size, data["trip_user"][validation])
        delta = bank.gain(phi_free, size) - bank.gain(phi_fixed, size)
        paired_gains.append(delta)
        reports.append({"draw_seed": draw_seed, "fixed": fixed, "free": free,
                        "free_minus_fixed": paired_score_summary(delta, data["trip_user"][validation])})
        del bank
    mc = paired_score_summary(paired_gains[0] - paired_gains[1])
    mc_bound = abs(mc["mean"]) + 1.96 * mc["standard_error"]
    screening = bool(
        fixed_solve["converged"] and free_solve["converged"] and free_solve["monotone"]
        and all(r["ess_passed"] for r in training_evaluation.values())
        and all(r["free"]["ess_passed"] and r["fixed"]["ess_passed"]
                and r["free_minus_fixed"]["95_interval"][0] > 0
                and r["free"]["gain"]["mean"] >= 0.005 for r in reports)
        and mc_bound <= 0.002)
    report = {
        "experimental_only": True, "certified": False,
        "parent": str(parent), "parent_sha256": file_sha256(parent),
        "spectral": str(spectral), "spectral_sha256": file_sha256(spectral),
        "data_fingerprint_sha256": parent_blob["data_fingerprint_sha256"],
        "config": vars(args), "fixed_solve": fixed_solve, "free_solve": free_solve,
        "free_initialization": "0.05_spectral_seed_zero_fixed_optimum" if zero_fixed else "convex_fixed_optimum",
        "training": training_evaluation, "validation": reports,
        "paired_gain_mc_difference": mc, "paired_gain_mc_error_bound": mc_bound,
        "predeclared_gates": {"ess_absolute": 2, "ess_fraction": 0.2,
                              "gain_vs_parent": 0.005, "free_vs_fixed_lower95": 0,
                              "mc_bound": 0.002, "solver_convergence": True},
        "passed_screening_for_further_audit": screening,
        "frozen_blocks": ["additive utilities", "prices", "categories", "size curve during free refinement"],
        "limitations": ["nonconvex finite-bank estimate, not global MLE",
                        "replicates share validation contexts; independent proposal RNG only",
                        "validation was used upstream for additive early stopping; untouched test audit still required",
                        "validation households may also occur in training; trip means use cluster SE",
                        "independent Smolyak, recommendation and population calibration still required"],
    }
    for name, phi in (("fixed", phi_fixed), ("free", phi_free)):
        phi = compact_factor(phi)
        with torch.no_grad():
            model.phi.zero_()
            model.phi[:, :phi.shape[1]].copy_(torch.as_tensor(phi, dtype=model.phi.dtype))
            model.rho_0_free.copy_(parent_blob["model"]["rho_0_free"] + torch.as_tensor(size_basis @ size))
        payload = dict(parent_blob)
        payload.update({"model": model.state_dict(), "iter": 0,
                        "estimator": "experimental_anchored_embedding_refinement_" + name,
                        "parent": str(parent), "parent_sha256": file_sha256(parent),
                        "active_rank": int(phi.shape[1]),
                        "experimental_only": True, "certified": False,
                        "best_validation": None, "best_iteration": 0,
                        "evaluations": [], "records": [],
                        "trained_capabilities": {**parent_blob["trained_capabilities"], "gram_interactions": True},
                        "refinement_report": str(output / "report.json")})
        atomic_save(output / f"{name}.pt", payload)
    report["artifacts_sha256"] = {p.name: file_sha256(p) for p in output.iterdir() if p.is_file()}
    (output / "report.json").write_text(strict_json_dumps(report))
    print(strict_json_dumps(report), end="", flush=True)


if __name__ == "__main__":
    main()
