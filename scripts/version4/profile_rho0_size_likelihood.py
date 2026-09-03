#!/usr/bin/env python3
"""Profile the original version-4 total-size potential at fixed remaining parameters.

For fixed utilities and interactions, rho_0 is a concave likelihood block.  A single
quadrature pass recovers each context's current size law.  Every candidate rho_0 tilt can
then be evaluated exactly (at that quadrature level) without another normalizer call:

    gain_t(delta) = -delta[n_t] - log sum_n p_old(n|x_t) exp(-delta[n]).

The profiler uses training cross-fitting, a disjoint validation acceptance panel, and a
q10 audit of the selected q9 validation gain.  It changes no part of the version-4 law.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch
from scipy.special import logsumexp

from audit_particle_counterfactual_generation import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from fit_interaction_particles import supported_trips
from fit_multifidelity_rank8 import (JOINT_OPTIMIZER_PARAMETER_NAMES, atomic_save,
                                     install, rule)


torch.set_default_dtype(torch.float64)


def normalized_log_probability(log_probability):
    value = np.asarray(log_probability, dtype=np.float64)
    return value - logsumexp(value, axis=1, keepdims=True)


def size_gain(delta, observed_size, old_log_probability):
    """Per-context likelihood gain for a rho_0 increment indexed by size 1..nmax."""
    delta = np.asarray(delta, dtype=np.float64)
    observed_index = np.asarray(observed_size, dtype=np.int64) - 1
    log_ratio = logsumexp(old_log_probability - delta[None, :], axis=1)
    return -delta[observed_index] - log_ratio


def tilted_probability(delta, old_log_probability):
    logits = old_log_probability - np.asarray(delta, dtype=np.float64)[None, :]
    logits -= logsumexp(logits, axis=1, keepdims=True)
    return np.exp(logits)


def profile_delta(observed_size, old_log_probability, prior_mass=1.0,
                  bound=15.0, score_tolerance=1e-7, max_iterations=100):
    """Fisher/Newton solve of the concave rho_0 block with size-one gauge.

    The negative Hessian is the mean conditional covariance of the size one-hot vector.
    Solving that 119-dimensional system is both cheaper and better conditioned than asking
    a generic optimizer to discover the exponential-family geometry from line searches.
    """
    old_log_probability = normalized_log_probability(old_log_probability)
    nmax = old_log_probability.shape[1]
    count = np.bincount(
        np.asarray(observed_size, dtype=np.int64), minlength=nmax + 1)[1:nmax + 1]
    target = (count + prior_mass / nmax) / (count.sum() + prior_mass)

    def value_score_information(delta):
        probability = tilted_probability(delta, old_log_probability)
        gain = -target @ delta - np.mean(logsumexp(
            old_log_probability - delta[None, :], axis=1))
        score = probability.mean(0) - target
        information = (np.diag(probability.mean(0))
                       - probability.T @ probability / len(probability))
        return float(gain), score, information

    delta = np.zeros(nmax, dtype=np.float64)
    evaluations = 0
    for iteration in range(1, max_iterations + 1):
        value, score, information = value_score_information(delta)
        evaluations += 1
        free_score = score[1:]
        if float(np.abs(free_score).max()) <= score_tolerance:
            break
        free_information = information[1:, 1:]
        # Only vanishingly rare tail coordinates need the ridge; it tends to zero with
        # their information and does not move the likelihood fixed point.
        ridge = max(1e-12, 1e-8 * float(np.diag(free_information).max()))
        try:
            direction = np.linalg.solve(
                free_information + ridge * np.eye(nmax - 1), free_score)
        except np.linalg.LinAlgError as error:
            raise RuntimeError("singular rho_0 Fisher system") from error
        accepted = False
        scale = 1.0
        for _ in range(40):
            candidate = delta.copy()
            candidate[1:] = np.clip(
                delta[1:] + scale * direction, -bound, bound)
            candidate_value, _, _ = value_score_information(candidate)
            evaluations += 1
            if candidate_value >= value + 1e-4 * scale * float(
                    free_score @ direction):
                delta = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            raise RuntimeError("rho_0 Newton line search failed")
    else:
        raise RuntimeError("rho_0 Newton solve reached its iteration ceiling")
    value, score, _ = value_score_information(delta)
    evaluations += 1
    return delta, {
        "iterations": int(iteration), "function_evaluations": int(evaluations),
        "objective": value, "maximum_absolute_free_score": float(
            np.abs(score[1:]).max()), "prior_mass": float(prior_mass),
        "bound": float(bound), "score_tolerance": float(score_tolerance),
    }


def moment_summary(probability, observed_size):
    probability = np.asarray(probability, dtype=np.float64)
    axis = np.arange(1, probability.shape[1] + 1, dtype=np.float64)
    conditional_mean = probability @ axis
    conditional_second = probability @ np.square(axis)
    within = np.mean(conditional_second - np.square(conditional_mean))
    between = np.var(conditional_mean)
    observed = np.asarray(observed_size, dtype=np.float64)
    return {
        "model_mean": float(conditional_mean.mean()),
        "model_within_context_variance": float(within),
        "model_between_context_variance": float(between),
        "model_total_variance": float(within + between),
        "observed_mean": float(observed.mean()),
        "observed_variance": float(observed.var()),
    }


def evaluate_delta(delta, observed_size, old_log_probability):
    gain = size_gain(delta, observed_size, old_log_probability)
    before = np.exp(normalized_log_probability(old_log_probability))
    after = tilted_probability(delta, old_log_probability)
    return {
        "mean_loglik_gain": float(gain.mean()),
        "loglik_gain_standard_error": float(
            gain.std(ddof=1) / math.sqrt(len(gain))),
        "minimum_trip_gain": float(gain.min()),
        "maximum_trip_gain": float(gain.max()),
        "before": moment_summary(before, observed_size),
        "after": moment_summary(after, observed_size),
    }


@torch.no_grad()
def collect_size_law(model, batcher, trips, quadrature, chunk, label):
    install(model, quadrature)
    probability, observed = [], []
    for start in range(0, len(trips), chunk):
        sub = trips[start:start + chunk]
        ix, ctx, _line_ctx, house, _li, lt, _lc, _lq = batcher.make(sub)
        model.house, model.ctx = house, ctx
        _logz, size_probability = model.log_Z(
            ix, drop_empty=True, return_size=True)
        probability.append(size_probability.cpu().numpy())
        observed.append(torch.bincount(lt, minlength=ix.B).cpu().numpy())
        if ((start // chunk + 1) % 20 == 0
                or start + chunk >= len(trips)):
            print(f"[rho0-profile] {label} "
                  f"{min(start + chunk, len(trips))}/{len(trips)}", flush=True)
    probability = np.concatenate(probability)
    return np.concatenate(observed), normalized_log_probability(
        np.log(np.clip(probability, 1e-300, None)))


def clear_rho0_optimizer_moments(payload):
    """A profiled block invalidates only rho_0's stored Adam moments."""
    optimizer = payload.get("optimizer")
    if optimizer is None:
        return False
    identifiers = [identifier for group in optimizer.get("param_groups", [])
                   for identifier in group.get("params", [])]
    if len(identifiers) != len(JOINT_OPTIMIZER_PARAMETER_NAMES):
        return False
    index = JOINT_OPTIMIZER_PARAMETER_NAMES.index("rho_0_free")
    state = optimizer.get("state", {}).get(identifiers[index])
    if state is None:
        return False
    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
        if key in state:
            state[key].zero_()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--target-level", type=int, default=9)
    parser.add_argument("--audit-level", type=int, default=10)
    parser.add_argument("--train-contexts", type=int, default=10000)
    parser.add_argument("--validation-contexts", type=int, default=768)
    parser.add_argument("--audit-contexts", type=int, default=64)
    parser.add_argument("--chunk", type=int, default=24)
    parser.add_argument("--prior-mass", type=float, default=1.0)
    parser.add_argument("--bound", type=float, default=15.0)
    parser.add_argument("--minimum-crossfit-gain", type=float, default=0.0)
    parser.add_argument("--minimum-validation-gain", type=float, default=0.0)
    parser.add_argument("--maximum-fidelity-gap", type=float, default=0.002)
    parser.add_argument("--candidate-scales", type=float, nargs="+",
                        default=[0.25, 0.5, 0.75, 1.0])
    parser.add_argument("--seed", type=int, default=31071)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("out/v3_rho0_profiled.pt"))
    parser.add_argument("--cache", type=Path,
                        help="quadrature size-law cache; defaults beside --output")
    parser.add_argument("--reuse-cache", action="store_true")
    args = parser.parse_args()
    if args.audit_contexts > args.validation_contexts:
        raise ValueError("audit contexts cannot exceed validation contexts")
    if args.target_level >= args.audit_level:
        raise ValueError("audit level must exceed target level")
    torch.set_num_threads(args.threads)
    data = build()
    checkpoint = (args.checkpoint if args.checkpoint.is_absolute()
                  else ROOT / args.checkpoint)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    cache = (args.cache if args.cache is not None else
             output.with_suffix(".profile_cache.npz"))
    cache = cache if cache.is_absolute() else ROOT / cache
    model, blob, meta = load_checkpoint(checkpoint, data)
    if int((torch.linalg.svdvals(model.phi) > 1e-10).sum()) != args.rank:
        raise RuntimeError("checkpoint active rank does not match --rank")
    batcher = Batcher(
        data, Features(int(data["n_item"]), int(data["n_store"]), 712),
        int(meta["nmax"]))
    training = supported_trips(data, 0, int(meta["nmax"]))
    validation = supported_trips(data, 1, int(meta["nmax"]))
    if args.train_contexts > len(training) or args.validation_contexts > len(validation):
        raise ValueError("requested more contexts than the corresponding population")
    rng = np.random.default_rng(args.seed)
    train_trips = training[rng.permutation(len(training))[:args.train_contexts]]
    validation_trips = validation[
        rng.permutation(len(validation))[:args.validation_contexts]]
    target_rule = rule(model, args.rank, args.target_level)
    audit_rule = rule(model, args.rank, args.audit_level)
    print(f"[rho0-profile] q{args.target_level} nodes={len(target_rule[1])}; "
          f"q{args.audit_level} nodes={len(audit_rule[1])}", flush=True)

    if args.reuse_cache:
        cached = np.load(cache)
        if (str(cached["checkpoint"].item()) != str(checkpoint)
                or int(cached["target_level"]) != args.target_level
                or int(cached["audit_level"]) != args.audit_level
                or not np.array_equal(cached["train_trips"], train_trips)
                or not np.array_equal(cached["validation_trips"], validation_trips)):
            raise RuntimeError("quadrature cache does not match this profiling request")
        train_size, train_logp = cached["train_size"], cached["train_logp"]
        valid_size, valid_logp = cached["valid_size"], cached["valid_logp"]
        audit_size, audit_logp = cached["audit_size"], cached["audit_logp"]
        print(f"[rho0-profile] reused quadrature cache: {cache}", flush=True)
    else:
        train_size, train_logp = collect_size_law(
            model, batcher, train_trips, target_rule, args.chunk, "training-q9")
        valid_size, valid_logp = collect_size_law(
            model, batcher, validation_trips, target_rule, args.chunk,
            "validation-q9")
        audit_size, audit_logp = collect_size_law(
            model, batcher, validation_trips[:args.audit_contexts], audit_rule,
            args.chunk, "validation-q10")
        np.savez_compressed(
            cache, checkpoint=np.asarray(str(checkpoint)),
            target_level=np.asarray(args.target_level),
            audit_level=np.asarray(args.audit_level), train_trips=train_trips,
            validation_trips=validation_trips, train_size=train_size,
            train_logp=train_logp, valid_size=valid_size, valid_logp=valid_logp,
            audit_size=audit_size, audit_logp=audit_logp)
        print(f"[rho0-profile] saved quadrature cache: {cache}", flush=True)

    half = rng.random(len(train_size)) < 0.5
    if half.all() or (~half).all():
        raise RuntimeError("degenerate cross-fit split")
    delta_a, solve_a = profile_delta(
        train_size[half], train_logp[half], args.prior_mass, args.bound)
    delta_b, solve_b = profile_delta(
        train_size[~half], train_logp[~half], args.prior_mass, args.bound)
    cross_a_on_b = evaluate_delta(delta_a, train_size[~half], train_logp[~half])
    cross_b_on_a = evaluate_delta(delta_b, train_size[half], train_logp[half])
    crossfit_gain = 0.5 * (cross_a_on_b["mean_loglik_gain"]
                           + cross_b_on_a["mean_loglik_gain"])
    delta, solve_full = profile_delta(
        train_size, train_logp, args.prior_mass, args.bound)
    training_full = evaluate_delta(delta, train_size, train_logp)

    trials = []
    for scale in sorted(set(args.candidate_scales)):
        candidate = scale * delta
        target = evaluate_delta(candidate, valid_size, valid_logp)
        audit = evaluate_delta(candidate, audit_size, audit_logp)
        fidelity_gap = (audit["mean_loglik_gain"]
                        - evaluate_delta(candidate, audit_size,
                                         valid_logp[:args.audit_contexts])[
                                             "mean_loglik_gain"])
        trials.append({
            "scale": float(scale), "validation_q9": target,
            "validation_q10": audit,
            "q10_minus_q9_gain": float(fidelity_gap),
        })
    admissible = [row for row in trials
                  if row["validation_q9"]["mean_loglik_gain"]
                  >= args.minimum_validation_gain
                  and row["validation_q10"]["mean_loglik_gain"]
                  >= args.minimum_validation_gain
                  and abs(row["q10_minus_q9_gain"])
                  <= args.maximum_fidelity_gap]
    selected = (max(admissible, key=lambda row:
                    row["validation_q9"]["mean_loglik_gain"])
                if admissible else None)
    accepted = bool(
        cross_a_on_b["mean_loglik_gain"] >= args.minimum_crossfit_gain
        and cross_b_on_a["mean_loglik_gain"] >= args.minimum_crossfit_gain
        and selected is not None)
    result = {
        "checkpoint": str(checkpoint), "checkpoint_iteration": int(blob["iter"]),
        "probability_law": "original version-4; rho_0 block only",
        "target_level": args.target_level, "target_nodes": len(target_rule[1]),
        "audit_level": args.audit_level, "audit_nodes": len(audit_rule[1]),
        "train_contexts": len(train_size),
        "validation_contexts": len(valid_size),
        "audit_contexts": len(audit_size),
        "crossfit": {
            "half_contexts": [int(half.sum()), int((~half).sum())],
            "a_solve": solve_a, "b_solve": solve_b,
            "a_fit_b": cross_a_on_b, "b_fit_a": cross_b_on_a,
            "mean_gain": float(crossfit_gain),
        },
        "full_solve": solve_full, "training_full": training_full,
        "candidate_trials": trials, "selected": selected,
        "accepted": accepted,
    }
    report = output.with_suffix(".json")
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not accepted:
        raise RuntimeError("rho_0 profile failed cross-fit/validation/fidelity gates")

    with torch.no_grad():
        model.rho_0_free.add_(torch.as_tensor(
            selected["scale"] * delta, dtype=model.rho_0_free.dtype))
    payload = dict(blob)
    payload["model"] = model.state_dict()
    payload["estimator"] = str(blob.get("estimator", "")) + "+profiled_rho0_q9"
    payload["rho0_profile_parent"] = str(checkpoint)
    payload["rho0_profile_report"] = str(report)
    payload["rho0_profile_scale"] = selected["scale"]
    payload["rho0_optimizer_moments_cleared"] = clear_rho0_optimizer_moments(payload)
    payload["best_validation"] = None
    payload["best_iteration"] = int(blob["iter"])
    payload["evaluations"] = []
    payload["records"] = []
    atomic_save(output, payload)
    print(f"[rho0-profile] accepted checkpoint: {output}", flush=True)


if __name__ == "__main__":
    main()
