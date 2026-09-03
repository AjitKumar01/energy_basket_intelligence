#!/usr/bin/env python3
"""Recalibrate the existing version-4 size potential after a fixed interaction fit.

The interaction matrix is not changed.  Exact draws from the Phi=0 parent are reused as
an importance proposal for the child, and only

    Delta rho_0(n) = a n + c n^2,  c >= 0

is fitted.  The two coefficients match the first two size moments without changing basket
composition conditional on size.  Two context halves select/certify the correction before
a checkpoint is written.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import logsumexp

from audit_particle_counterfactual_generation import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from fit_interaction_particles import supported_trips
from tempered_block_gibbs import conditional_slots_repeated


torch.set_default_dtype(torch.float64)


def pair_energy(items, phi):
    rows = phi[torch.unique(items)]
    total = rows.sum(0)
    return 0.5 * ((total * total).sum() - (rows * rows).sum())


def features(size):
    z = np.asarray(size, dtype=np.float64) / 10.0
    return np.stack((z, z * z), axis=-1)


def gain_and_derivatives(theta, observed, draw_size, draw_pair):
    obs_f = features(observed)
    draw_f = features(draw_size)
    logits = draw_pair - np.einsum("mdk,k->md", draw_f, theta)
    base = logsumexp(draw_pair, axis=1) - np.log(draw_pair.shape[1])
    tilted = logsumexp(logits, axis=1) - np.log(draw_pair.shape[1])
    gain = -obs_f @ theta - tilted + base
    weight = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    mean_f = np.einsum("md,mdk->mk", weight, draw_f)
    gradient = (-obs_f + mean_f).mean(0)
    second = np.einsum("md,mdi,mdj->mij", weight, draw_f, draw_f)
    covariance = second - np.einsum("mi,mj->mij", mean_f, mean_f)
    hessian = -covariance.mean(0)
    return gain, gradient, hessian


def fit_theta(observed, draw_size, draw_pair):
    def objective(theta):
        gain, gradient, _ = gain_and_derivatives(
            theta, observed, draw_size, draw_pair)
        return -float(gain.mean()), -gradient

    result = minimize(
        objective, np.zeros(2, dtype=np.float64), method="L-BFGS-B",
        jac=True, bounds=[(None, None), (0.0, None)],
        options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 100})
    if not result.success:
        raise RuntimeError(f"size solve failed: {result.message}")
    return np.asarray(result.x, dtype=np.float64)


def evaluate(theta, observed, draw_size, draw_pair):
    gain, gradient, hessian = gain_and_derivatives(
        theta, observed, draw_size, draw_pair)
    logits = draw_pair - np.einsum(
        "mdk,k->md", features(draw_size), theta)
    weight = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
    model_mean = float((weight * draw_size).sum(1).mean())
    model_second = float((weight * np.square(draw_size)).sum(1).mean())
    return {
        "gain": float(gain.mean()),
        "gain_se": float(gain.std(ddof=1) / np.sqrt(len(gain))),
        "observed_mean": float(observed.mean()),
        "observed_variance": float(observed.var()),
        "model_mean": model_mean,
        "model_variance": model_second - model_mean * model_mean,
        "score": gradient.tolist(),
        "negative_hessian": (-hessian).tolist(),
    }


def atomic_save(path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--child", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=6000)
    parser.add_argument("--draws", type=int, default=32)
    parser.add_argument("--batch", type=int, default=96)
    parser.add_argument("--minimum-crossfit-gain", type=float, default=0.002)
    parser.add_argument("--minimum-half-gain", type=float, default=0.0,
                        help="minimum gain required in each swapped-half direction")
    parser.add_argument("--seed", type=int, default=28501)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("out/v3_projected_fisher_size_calibrated.pt"))
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = build()
    parent_path = args.parent if args.parent.is_absolute() else ROOT / args.parent
    child_path = args.child if args.child.is_absolute() else ROOT / args.child
    parent, parent_blob, meta = load_checkpoint(parent_path, data)
    child, child_blob, child_meta = load_checkpoint(child_path, data)
    if float(parent.phi.abs().max()) != 0.0:
        raise RuntimeError("importance proposal must be the Phi=0 parent")
    for name in ("lam", "alpha", "theta", "rho_c", "rho_0_free"):
        if not torch.equal(getattr(parent, name), getattr(child, name)):
            raise RuntimeError(f"parent/child mismatch outside interaction block: {name}")

    population = supported_trips(data, 0, int(meta["nmax"]))
    if args.contexts > len(population):
        raise ValueError("requested more contexts than the training population")
    rng = np.random.default_rng(args.seed)
    trips = population[rng.permutation(len(population))[:args.contexts]]
    half = rng.random(len(trips)) < 0.5
    if half.all() or (~half).all():
        raise RuntimeError("degenerate cross-fit split")
    observed = np.empty(len(trips), dtype=np.float64)
    draw_size = np.empty((len(trips), args.draws), dtype=np.float64)
    draw_pair = np.empty_like(draw_size)
    batcher = Batcher(
        data, Features(int(data["n_item"]), int(data["n_store"]), 712),
        int(meta["nmax"]))
    generator = torch.Generator().manual_seed(args.seed + 1)

    for start in range(0, len(trips), args.batch):
        sub = trips[start:start + args.batch]
        ix, ctx, _line_ctx, house, li, lt, _lc, _lq = batcher.make(sub)
        parent.house, parent.ctx = house, ctx
        z = torch.zeros(ix.B, parent.Kz, dtype=parent.phi.dtype)
        states = conditional_slots_repeated(
            parent, ix, z, 0.0, args.draws, generator)
        observed[start:start + ix.B] = torch.bincount(
            lt, minlength=ix.B).cpu().numpy()
        for local in range(ix.B):
            for draw, state in enumerate(states):
                items = ix.item[state[local]]
                draw_size[start + local, draw] = len(torch.unique(items))
                draw_pair[start + local, draw] = float(
                    pair_energy(items, child.phi))
        if (start // args.batch + 1) % 10 == 0 or start + args.batch >= len(trips):
            print(f"[size-fisher] {min(start+args.batch,len(trips))}/"
                  f"{len(trips)} contexts", flush=True)

    theta_a = fit_theta(observed[half], draw_size[half], draw_pair[half])
    theta_b = fit_theta(observed[~half], draw_size[~half], draw_pair[~half])
    theta = fit_theta(observed, draw_size, draw_pair)
    a_on_b = evaluate(theta_a, observed[~half], draw_size[~half], draw_pair[~half])
    b_on_a = evaluate(theta_b, observed[half], draw_size[half], draw_pair[half])
    full = evaluate(theta, observed, draw_size, draw_pair)
    mean_crossfit = 0.5 * (a_on_b["gain"] + b_on_a["gain"])
    accepted = (
        a_on_b["gain"] > args.minimum_half_gain
        and b_on_a["gain"] > args.minimum_half_gain
        and mean_crossfit >= args.minimum_crossfit_gain)
    result = {
        "parent": str(parent_path),
        "child": str(child_path),
        "contexts": len(trips),
        "draws_per_context": args.draws,
        "half_contexts": [int(half.sum()), int((~half).sum())],
        "theta_scaled": theta.tolist(),
        "rho0_linear_a": float(theta[0] / 10.0),
        "rho0_quadratic_c": float(theta[1] / 100.0),
        "a_fit_b": a_on_b,
        "b_fit_a": b_on_a,
        "mean_crossfit_gain": mean_crossfit,
        "minimum_required_crossfit_gain": args.minimum_crossfit_gain,
        "minimum_required_half_gain": args.minimum_half_gain,
        "full_fit": full,
        "accepted_for_nonlinear_audit": accepted,
    }
    output = args.output if args.output.is_absolute() else ROOT / args.output
    report = output.with_suffix(".json")
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not accepted:
        raise RuntimeError("size correction failed the cross-fit gain gate")

    n = torch.arange(
        1, child.rho_0_free.numel() + 1, dtype=child.rho_0_free.dtype)
    child.rho_0_free.add_(theta[0] * n / 10.0 + theta[1] * n.square() / 100.0)
    payload = dict(child_blob)
    payload["model"] = child.state_dict()
    payload["estimator"] = "projected_fisher_version4_interaction_plus_size_moment_solve"
    payload["parent_interaction_checkpoint"] = str(child_path)
    payload["size_calibration_report"] = str(report)
    payload["best_validation"] = None
    payload["evaluations"] = []
    payload["records"] = []
    atomic_save(output, payload)
    print(f"[size-fisher] checkpoint: {output}", flush=True)


if __name__ == "__main__":
    main()
