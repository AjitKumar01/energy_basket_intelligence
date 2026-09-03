#!/usr/bin/env python3
"""Exact joint-MLE stage for the version-4 model with the Gram residual held at zero.

The category/cardinality dynamic program normalizes all 5,455 offered products and every
size 1..nmax exactly.  This stage learns propensity, household/context, price, promotion,
season/store, category and total-size parameters without quadrature or sampling.  Rank-8
interactions are added as a separate residual stage only after this exact block converges.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch
from torch.nn.functional import softplus

from data import build
from category_safety import category_capacities, project_category_reward_
from features import Features
from fit import Batcher
from fit_interaction_particles import supported_trips
from interaction_particles import (differentiable_log_size_beta0,
                                   differentiable_logz_beta0)
from ragged import RaggedModel
from sparse_artifact import load_sparse_initialization_artifact


torch.set_default_dtype(torch.float64)
ROOT = Path(__file__).resolve().parents[2]


class Tee:
    def __init__(self, stream, path, append=False):
        self.stream = stream
        self.file = Path(path).open("a" if append else "w", buffering=1)

    def write(self, value):
        self.stream.write(value)
        self.file.write(value)
        return len(value)

    def flush(self):
        self.stream.flush()
        self.file.flush()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path,
                        default=Path("out/v3_version4_sparse_init.pt"))
    parser.add_argument("--label", default="run259_exact_additive")
    parser.add_argument("--resume", type=Path,
                        help="resume an exact Phi=0 checkpoint; --iters is the final update")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch", type=int, default=128)
    parser.add_argument("--lr", type=float, default=.001)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--validation-trips", type=int, default=1024)
    parser.add_argument("--validation-chunk", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--milestone-every", type=int, default=500)
    parser.add_argument("--lr-patience", type=int, default=4,
                        help="validation plateaus before halving the learning rate")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=6.25e-5)
    parser.add_argument("--convergence-patience", type=int, default=8,
                        help="evaluations without a new best at minimum LR")
    parser.add_argument("--convergence-min-updates", type=int, default=4000)
    parser.add_argument("--require-convergence", action="store_true",
                        help="exit nonzero if --iters is reached before convergence")
    parser.add_argument("--validation-min-delta", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=25901)
    parser.add_argument("--threads", type=int, default=8)
    # These are estimation/identification controls used by the version-4 experiment.
    # They regularize fitting; they do not add or remove a term from the basket energy.
    parser.add_argument("--size-kl", type=float, default=0.0)
    parser.add_argument("--rkl-w", type=float, default=0.0)
    parser.add_argument("--rkl-eps", type=float, default=1e-4)
    parser.add_argument("--elast-w", type=float, default=0.0)
    parser.add_argument("--elast-target", type=float, default=-0.121)
    parser.add_argument("--pool-prod", type=float, default=0.0)
    parser.add_argument("--lam-centre", type=int, default=0)
    parser.add_argument("--lam-sd-max", type=float, default=0.0)
    parser.add_argument(
        "--rho-c-max-category-reward", type=float, default=0.0,
        help=("if positive, constrain each existing category term to contribute at "
              "most this many attractive nats anywhere on support 1..nmax"))
    return parser.parse_args()


def atomic_save(path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_model(artifact, data):
    raw = torch.load(artifact, map_location="cpu", weights_only=False)
    meta = raw["metadata"]
    model = RaggedModel(
        int(data["n_item"]), int(data["n_user"]), int(data["n_cat"]),
        K=int(meta["K"]), Kz=int(meta["Kz"]), nmax=int(meta["nmax"]),
        R=int(meta["R"]), seed=int(meta["seed"]), S=int(data["n_store"]),
        Kp=int(meta["Kp"]), phi_init=0.0,
        household_size_rank1=bool(meta.get("household_size_rank1", False)))
    restored = load_sparse_initialization_artifact(artifact, model)
    with torch.no_grad():
        model.phi.zero_()
    model._poly_degree_native = True
    model._esp_native = True
    model._esp_log_blocked = True
    model.double().train()
    return model, meta, restored


def fitted_parameters(model):
    # psi is excluded because the fresh experiment contract is no_rec=True.  Quantity-only
    # blocks are absent because this file fits incidence baskets, not line-unit counts.
    names = ("lam", "alpha", "theta", "rho_c", "rho_0_free", "price_kappa",
             "gamma", "beta", "w_dsp", "w_mlr", "mu", "delta", "zeta", "xi")
    return [getattr(model, name) for name in names]


def exact_loglik(model, ix, line_item, line_trip, line_cat, line_ctx):
    energy = model.energy(line_item, line_trip, line_cat, ix.B, line_ctx)
    logz = differentiable_logz_beta0(model, ix)
    return energy - logz


def exact_loglik_and_size(model, ix, line_item, line_trip, line_cat, line_ctx):
    """One exact DP for both normalized likelihood and the size-law penalties."""
    log_size = differentiable_log_size_beta0(model, ix)
    logz = torch.logsumexp(log_size, dim=-1)
    energy = model.energy(line_item, line_trip, line_cat, ix.B, line_ctx)
    return energy - logz, torch.softmax(log_size, dim=-1)


@torch.no_grad()
def validate(model, batcher, trips, chunk):
    model.eval()
    values, sizes, size_laws = [], [], []
    for start in range(0, len(trips), chunk):
        ix, ctx, line_ctx, house, li, lt, lc, _lq = batcher.make(
            trips[start:start + chunk])
        model.house, model.ctx = house, ctx
        score, size_probability = exact_loglik_and_size(
            model, ix, li, lt, lc, line_ctx)
        values.append(score.cpu())
        size_laws.append(size_probability.cpu())
        sizes.append(torch.bincount(lt, minlength=ix.B).double().cpu())
    model.train()
    value = torch.cat(values).numpy()
    observed_size = torch.cat(sizes).numpy()
    probability = torch.cat(size_laws).numpy()
    grid = np.arange(1, probability.shape[1] + 1, dtype=np.float64)
    expected_size = probability @ grid
    aggregate_second = float((probability @ grid ** 2).mean())
    aggregate_mean = float(expected_size.mean())
    return {
        "basket_loglik": float(value.mean()),
        "standard_error": float(value.std(ddof=1) / math.sqrt(len(value))),
        "observed_size_mean": float(observed_size.mean()),
        "observed_size_variance": float(observed_size.var(ddof=1)),
        "expected_size_mean": aggregate_mean,
        "expected_size_median": float(np.median(expected_size)),
        "expected_size_p95": float(np.quantile(expected_size, .95)),
        "expected_size_max": float(expected_size.max()),
        "model_population_size_variance": aggregate_second - aggregate_mean ** 2,
        "trips": int(len(value)),
    }


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    artifact = args.artifact if args.artifact.is_absolute() else ROOT / args.artifact
    output = ROOT / "out"
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / f"v3_{args.label}.log"
    checkpoint_path = output / f"v3_{args.label}.pt"
    best_path = output / f"v3_{args.label}_best.pt"
    history_path = output / f"v3_{args.label}_history.json"
    sys.stdout = Tee(sys.stdout, log_path, append=args.resume is not None)

    data = build()
    model, meta, restored = load_model(artifact, data)
    if int(meta["active_rank"]) != 8:
        raise RuntimeError("expected the certified rank-8 parent artifact")
    batcher = Batcher(data, Features(int(data["n_item"]), int(data["n_store"]), 712),
                      int(meta["nmax"]))
    train = supported_trips(data, 0, int(meta["nmax"]))
    empirical_count = np.bincount(
        np.clip(data["trip_nlines"][train], 1, int(meta["nmax"])),
        minlength=int(meta["nmax"]) + 1)[1:].astype(np.float64)
    empirical_size = torch.as_tensor(
        empirical_count / empirical_count.sum(), dtype=model.lam.dtype)
    valid_population = supported_trips(data, 1, int(meta["nmax"]))
    valid_rng = np.random.default_rng(args.seed + 1)
    valid = valid_population[valid_rng.permutation(len(valid_population))[
        :args.validation_trips]]
    rng = np.random.default_rng(args.seed + 2)
    parameters = fitted_parameters(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.lr,
                                  weight_decay=args.weight_decay)
    rho_c_capacities = category_capacities(
        data, int(data["n_cat"]), int(meta["nmax"]))

    start_iteration = 0
    resumed = None
    if args.resume is not None:
        resume_path = args.resume if args.resume.is_absolute() else ROOT / args.resume
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        if resumed.get("estimator") != "exact_version4_no_gram_dynamic_program":
            raise RuntimeError("resume checkpoint is not from the exact Phi=0 stage")
        prior = resumed["config"]
        for key in ("batch", "seed"):
            if int(prior[key]) != int(getattr(args, key)):
                raise RuntimeError(f"--{key} must remain {prior[key]} when resuming")
        prior_reward = float(prior.get("rho_c_max_category_reward", 0.0))
        if prior_reward != float(args.rho_c_max_category_reward):
            raise RuntimeError(
                "--rho-c-max-category-reward must remain "
                f"{prior_reward:g} when resuming")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        start_iteration = int(resumed["iter"])
        if args.iters <= start_iteration:
            raise RuntimeError(
                f"--iters={args.iters} must exceed resumed iteration {start_iteration}")
        # The only stochastic training operation is NumPy's uniform minibatch draw.
        # Replaying those inexpensive draws reproduces the exact next minibatch without
        # storing a fragile pickled RNG implementation in historical checkpoints.
        for _ in range(start_iteration):
            rng.choice(len(train), size=args.batch, replace=False)
        if float(model.phi.detach().abs().max()) != 0.0:
            raise RuntimeError("resume checkpoint has a nonzero Gram residual")

    print("[exact-additive] unchanged version-4 utilities/category/size law; Gram residual "
          "held at zero for this exact MLE stage", flush=True)
    print(f"[exact-additive] J={model.J}, full support 1..{model.nmax}, batch={args.batch}, "
          f"lr={args.lr:g}", flush=True)
    print("[exact-additive] exact native category/cardinality DP; no quadrature, particles, "
          "ESS, retry, skip, or fallback", flush=True)
    if model.household_size_rank1:
        print("[exact-additive] taste rank split: K-1 catalogue-centred composition "
              "directions plus one household-common size direction", flush=True)
    if args.rho_c_max_category_reward > 0:
        initial_category_safety = project_category_reward_(
            model, rho_c_capacities, args.rho_c_max_category_reward,
            optimizer=optimizer)
        print("[exact-additive] complete-support category constraint: "
              f"max attractive reward={args.rho_c_max_category_reward:g} nats; "
              f"initial max={initial_category_safety['maximum_reward_after']:.6f}",
              flush=True)
    if resumed is not None:
        print(f"[exact-additive] resumed iteration {start_iteration} from {resume_path}; "
              f"optimizer and minibatch stream restored", flush=True)
    print(f"[exact-additive] log: {log_path}", flush=True)

    started = time.perf_counter()
    resumed_value = validate(model, batcher, valid, args.validation_chunk)
    if resumed is None:
        initial = resumed_value
        evaluations = [{"iter": 0, **initial}]
        records = []
        best_score = initial["basket_loglik"]
        best_iteration = 0
        plateau_anchor = best_score
        plateau_evaluations = 0
        evaluations_since_best = 0
        print(f"[exact-additive] initial validation LL={best_score:.6f}", flush=True)
    else:
        evaluations = list(resumed["evaluations"])
        records = list(resumed["records"])
        initial = {key: value for key, value in evaluations[0].items() if key != "iter"}
        best_score = float(resumed["best_validation"])
        best_iteration = int(resumed["best_iteration"])
        scheduler = resumed.get("scheduler", {})
        plateau_anchor = float(scheduler.get("plateau_anchor", best_score))
        plateau_evaluations = int(scheduler.get("plateau_evaluations", 0))
        evaluations_since_best = int(scheduler.get("evaluations_since_best", 0))
        saved = next(x for x in reversed(evaluations)
                     if int(x["iter"]) == start_iteration)
        tolerance = 5e-10
        if abs(resumed_value["basket_loglik"] - saved["basket_loglik"]) > tolerance:
            raise RuntimeError("resume validation panel does not reproduce the checkpoint")
        print(f"[exact-additive] reproduced validation {start_iteration}: "
              f"LL={resumed_value['basket_loglik']:.6f}", flush=True)

    def payload(iteration):
        return {
            "format": 2,
            "estimator": "exact_version4_no_gram_dynamic_program",
            "fresh_artifact_digest": restored["model_state_sha256"],
            "iter": iteration,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": vars(args),
            "objective": "exact normalized joint likelihood with Phi=0",
            "parent_active_rank": 8,
            "best_validation": best_score,
            "best_iteration": best_iteration,
            "scheduler": {
                "plateau_anchor": plateau_anchor,
                "plateau_evaluations": plateau_evaluations,
                "evaluations_since_best": evaluations_since_best,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            },
            "evaluations": evaluations,
            "records": records,
        }

    atomic_save(best_path, payload(start_iteration))
    converged = False
    for iteration in range(start_iteration + 1, args.iters + 1):
        tick = time.perf_counter()
        trips = train[rng.choice(len(train), size=args.batch, replace=False)]
        ix, ctx, line_ctx, house, li, lt, lc, _lq = batcher.make(trips)
        model.house, model.ctx = house, ctx
        optimizer.zero_grad(set_to_none=True)
        if args.size_kl > 0 or args.elast_w > 0:
            score_values, size_probability = exact_loglik_and_size(
                model, ix, li, lt, lc, line_ctx)
        else:
            score_values = exact_loglik(model, ix, li, lt, lc, line_ctx)
            size_probability = None
        score = score_values.mean()
        loss = -score
        size_penalty = torch.zeros((), dtype=score.dtype)
        elasticity_penalty = torch.zeros((), dtype=score.dtype)
        elasticity = torch.full((), float("nan"), dtype=score.dtype)
        if args.size_kl > 0:
            pbar = size_probability.mean(0).clamp_min(1e-12)
            pbar = pbar / pbar.sum()
            target = empirical_size[:pbar.numel()]
            size_penalty = -(target * pbar.log()).sum()
            if args.rkl_w > 0:
                smooth = target + args.rkl_eps
                smooth = smooth / smooth.sum()
                size_penalty = size_penalty + args.rkl_w * (
                    pbar * (pbar.log() - smooth.log())).sum()
            loss = loss + args.size_kl * size_penalty
        if args.elast_w > 0:
            grid = torch.arange(1, size_probability.shape[1] + 1,
                                dtype=score.dtype)
            mean_size = (size_probability * grid).sum(1)
            var_size = ((size_probability * grid.square()).sum(1)
                        - mean_size.square())
            price_coefficient = (
                softplus(model.gamma[house][ix.item_trip])
                * softplus(model.beta[ix.item])).sum(-1).mean()
            elasticity = -(price_coefficient * var_size.mean()
                           / mean_size.mean().clamp_min(1e-6))
            elasticity_penalty = args.elast_w * (
                elasticity - args.elast_target).square()
            loss = loss + elasticity_penalty
        if args.pool_prod > 0:
            for product, context in (
                    (model.mu, model.delta_c()),
                    (model.zeta, model.xi_c()),
                    (model.alpha, model.theta_c())):
                loss = loss + args.pool_prod * torch.trace(
                    (product.T @ product) @ (context.T @ context)) / (
                        product.shape[0] * context.shape[0])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite exact loss at {iteration}")
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.clip))
        optimizer.step()
        model.project_context_gauges()
        if args.lam_centre:
            # Exact gauge transformation: subtracting mu from every item utility and
            # subtracting n*mu from rho_0 leaves every basket energy unchanged.
            with torch.no_grad():
                mean_lam = model.lam.mean().clone()
                model.lam.sub_(mean_lam)
                sizes = torch.arange(1, model.rho_0_free.numel() + 1,
                                     dtype=model.lam.dtype)
                model.rho_0_free.sub_(mean_lam * sizes)
        if args.lam_sd_max > 0:
            with torch.no_grad():
                spread = model.lam.std()
                if float(spread) > args.lam_sd_max:
                    model.lam.mul_(args.lam_sd_max / spread.clamp_min(1e-12))
        model.project_rho_c(-1.5)
        category_safety = None
        if args.rho_c_max_category_reward > 0:
            category_safety = project_category_reward_(
                model, rho_c_capacities, args.rho_c_max_category_reward,
                optimizer=optimizer)
        if float(model.phi.detach().abs().max()) != 0.0:
            raise RuntimeError("exact additive stage changed the frozen Gram residual")
        row = {
            "iter": iteration,
            "train_loglik": float(score.detach()),
            "grad_norm": grad_norm,
            "size_penalty": float(size_penalty.detach()),
            "elasticity": float(elasticity.detach()),
            "elasticity_penalty": float(elasticity_penalty.detach()),
            "lam_sd": float(model.lam.std().detach()),
            "house_size_sd": (
                float(model.theta_c()[:, -1].std().detach())
                if model.household_size_rank1 else float("nan")),
            "house_size_abs_max": (
                float(model.theta_c()[:, -1].abs().max().detach())
                if model.household_size_rank1 else float("nan")),
            "maximum_category_reward": (
                category_safety["maximum_reward_after"]
                if category_safety is not None else float("nan")),
            "projected_category_coefficients": (
                category_safety["projected_coefficients"]
                if category_safety is not None else 0),
            "seconds": time.perf_counter() - tick,
        }
        records.append(row)
        if iteration % args.log_every == 0 or iteration == args.iters:
            window = records[-min(args.log_every, len(records)):]
            print(f"[exact-additive] step {iteration:4d} LL="
                  f"{np.mean([x['train_loglik'] for x in window]):.5f} "
                  f"grad={np.mean([x['grad_norm'] for x in window]):.3f} "
                  f"lam.sd={row['lam_sd']:.3f} "
                  + (f"kappa.sd={row['house_size_sd']:.3f} "
                     f"kappa.max={row['house_size_abs_max']:.3f} "
                     if model.household_size_rank1 else "")
                  + (f"cat.max={row['maximum_category_reward']:.3f} "
                     if category_safety is not None else "")
                  + (f"elast={row['elasticity']:+.3f} "
                     if math.isfinite(row['elasticity']) else "")
                  + f"{np.mean([x['seconds'] for x in window]):.3f}s/it", flush=True)
        if iteration % args.eval_every == 0 or iteration == args.iters:
            current = validate(model, batcher, valid, args.validation_chunk)
            evaluations.append({"iter": iteration, **current})
            change = current["basket_loglik"] - initial["basket_loglik"]
            print(f"[exact-additive] validation {iteration}: LL="
                  f"{current['basket_loglik']:.6f}, change={change:+.6f}, "
                  f"E[n]={current['expected_size_mean']:.2f}, "
                  f"p95={current['expected_size_p95']:.2f}, "
                  f"Var(n)={current['model_population_size_variance']:.2f}", flush=True)
            numerical_improved = current["basket_loglik"] > best_score
            material_improved = (
                current["basket_loglik"]
                > plateau_anchor + args.validation_min_delta)
            if numerical_improved:
                best_score = current["basket_loglik"]
                best_iteration = iteration
                evaluations_since_best = 0
                atomic_save(best_path, payload(iteration))
                print(f"[exact-additive] new best checkpoint at {iteration}", flush=True)
            else:
                evaluations_since_best += 1
            if material_improved:
                plateau_anchor = current["basket_loglik"]
                plateau_evaluations = 0
            else:
                plateau_evaluations += 1
            if (args.lr_patience > 0
                    and plateau_evaluations >= args.lr_patience):
                old_lr = float(optimizer.param_groups[0]["lr"])
                new_lr = max(args.min_lr, old_lr * args.lr_factor)
                if new_lr < old_lr:
                    for group in optimizer.param_groups:
                        group["lr"] = new_lr
                    print(f"[exact-additive] validation plateau: lr "
                          f"{old_lr:.3g} -> {new_lr:.3g}", flush=True)
                plateau_evaluations = 0
            atomic_save(checkpoint_path, payload(iteration))
            if args.milestone_every and iteration % args.milestone_every == 0:
                milestone = output / f"v3_{args.label}_iter{iteration}.pt"
                atomic_save(milestone, payload(iteration))
                print(f"[exact-additive] milestone: {milestone}", flush=True)
            history_path.write_text(json.dumps({
                "initial_validation": initial,
                "latest_validation": current,
                "best_validation": best_score,
                "best_iteration": best_iteration,
                "evaluations": evaluations,
                "records": records,
                "wall_seconds": time.perf_counter() - started,
            }, indent=2) + "\n")
            if (iteration >= args.convergence_min_updates
                    and float(optimizer.param_groups[0]["lr"]) <= args.min_lr
                    and evaluations_since_best >= args.convergence_patience):
                converged = True
                print(f"[exact-additive] convergence declared at {iteration}: "
                      f"{evaluations_since_best} evaluations since best at minimum LR",
                      flush=True)
                break
    outcome = "converged" if converged else "reached safety ceiling"
    print(f"[exact-additive] {outcome} in {time.perf_counter()-started:.1f}s; "
          f"best LL={best_score:.6f} at {best_iteration}", flush=True)
    if args.require_convergence and not converged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
