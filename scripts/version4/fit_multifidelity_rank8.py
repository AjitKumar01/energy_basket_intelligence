#!/usr/bin/env python3
"""Fit a fixed-rank Gram residual with an unbiased multilevel Smolyak score.

For rank r, the level-(r+1) rule supplies a large-context control variate.  On a
uniform sub-batch the level-(r+2) minus level-(r+1) normalizer gradient is added
back.  Consequently the expected gradient is the level-(r+2) joint-likelihood
gradient.  Optionally, the centre rule and two telescoping corrections provide the same
level-(r+2) target with less node-context work.  A still finer level-(r+3) rule is used
only as a validation-time fidelity guard; it never supplies a training gradient.
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

from audit_particle_counterfactual_generation import ROOT, load_checkpoint
from category_safety import category_capacities, project_category_reward_
from data import build
from features import Features
from fit import (Batcher, build_observed_basic_scores,
                 build_observed_phi_operator, full_observed_phi_score,
                 observed_basic_scores, observed_phi_score)
from fit_interaction_particles import supported_trips
from ragged import smolyak_grid


torch.set_default_dtype(torch.float64)


class Tee:
    def __init__(self, stream, path, append=False):
        self.stream = stream
        self.file = Path(path).open("a" if append else "w", buffering=1)

    def write(self, value):
        self.stream.write(value); self.file.write(value); return len(value)

    def flush(self):
        self.stream.flush(); self.file.flush()


def rule(model, rank, level):
    active, weights = smolyak_grid(rank, level)
    nodes = torch.zeros(len(weights), model.Kz, dtype=model.phi.dtype)
    nodes[:, :rank] = active
    return nodes, weights


def install(model, quadrature):
    model.quad = quadrature; model.quad_a = None


def spectral_scale_gradient(phi_gradient, basis, rank):
    """Chain rule for Phi[:, k] = basis[:, k] * scale[k]."""
    return (phi_gradient[:, :rank] * basis[:, :rank]).sum(0)


def spectral_transform_gradient(phi_gradient, basis, rank):
    """Chain rule for ``Phi[:, :r] = basis[:, :r] @ transform``."""
    return basis[:, :rank].T @ phi_gradient[:, :rank]


@torch.no_grad()
def install_spectral_transform(phi, basis, transform, rank):
    """Reconstruct the catalogue interaction factor from r-by-r coordinates."""
    phi.zero_()
    phi[:, :rank].copy_(basis[:, :rank] @ transform)


@torch.no_grad()
def project_active_spectral_ball(phi, rank, radius):
    """Euclidean projection of active Phi onto ||Phi||_2 <= radius.

    Projection onto an operator-norm ball clips singular values individually.  Uniformly
    rescaling Phi is not this projection and destroys already-feasible directions.
    """
    active = phi[:, :rank]
    left, singular, right = torch.linalg.svd(active, full_matrices=False)
    clipped = singular.clamp(max=radius)
    changed = bool((clipped != singular).any())
    if changed:
        active.copy_((left * clipped.unsqueeze(0)) @ right)
    return float(singular[0]), float(clipped[0]), changed


def decay_learning_rates(optimizer, factor, minimum):
    """Decay each parameter group independently without erasing LR separation."""
    old = [float(group["lr"]) for group in optimizer.param_groups]
    new = [max(float(minimum), value * float(factor)) for value in old]
    for group, value in zip(optimizer.param_groups, new):
        group["lr"] = value
    return old, new


def joint_values(model, ix, li, lt, lc, line_ctx):
    return (model.energy(li, lt, lc, ix.B, line_ctx)
            - model.log_Z(ix, drop_empty=True))


INCIDENCE_PARAMETER_NAMES = (
    "lam", "alpha", "theta", "phi", "rho_c", "rho_0_free",
    "price_kappa", "gamma", "beta", "w_dsp", "w_mlr", "mu", "delta",
    "zeta", "xi",
)

# Exact-additive trainer order.  Keeping this explicit makes optimizer-state transfer
# auditable instead of relying on opaque integer parameter identifiers in torch.save.
ADDITIVE_PARAMETER_NAMES = tuple(
    name for name in INCIDENCE_PARAMETER_NAMES if name != "phi")
JOINT_OPTIMIZER_PARAMETER_NAMES = ("phi",) + ADDITIVE_PARAMETER_NAMES


def category_pair_reference(data, trips, nmax, categories, prior=100.0):
    """Training-only E[choose(N_c,2)|N=n] with a smooth pair-rate prior.

    The result is used only as a coordinate system for the existing rho_c/rho_0 blocks.
    For any fixed reference a_c(n), subtracting a_c(n) from the category statistic and
    compensating rho_0 is an exact reparameterization of the version-4 energy.
    """
    total = np.zeros((nmax, categories), dtype=np.float64)
    count = np.zeros(nmax, dtype=np.float64)
    pair_denominator = 0.0
    pair_total = np.zeros(categories, dtype=np.float64)
    pointer, line_category = data["line_ptr"], data["line_cat"]
    for trip in np.asarray(trips, dtype=np.int64):
        lo, hi = int(pointer[trip]), int(pointer[trip + 1])
        n = hi - lo
        if not 1 <= n <= nmax:
            continue
        category_count = np.bincount(
            line_category[lo:hi], minlength=categories).astype(np.float64)
        statistic = category_count * (category_count - 1.0) / 2.0
        total[n - 1] += statistic
        count[n - 1] += 1.0
        pair_total += statistic
        pair_denominator += n * (n - 1.0) / 2.0
    rate = pair_total / max(pair_denominator, 1.0)
    size = np.arange(1, nmax + 1, dtype=np.float64)
    baseline = (size * (size - 1.0) / 2.0)[:, None] * rate[None, :]
    reference = ((total + prior * baseline)
                 / (count[:, None] + prior))
    return torch.as_tensor(reference.T)


@torch.no_grad()
def project_rho_c_trust_region(model, anchor, radius, optimizer=None):
    """Project rho_c into an L2 ball and remove Adam's redundant outward momentum."""
    displacement = model.rho_c - anchor
    norm = float(displacement.norm())
    if norm <= radius:
        return norm, False
    model.rho_c.copy_(anchor + displacement * (radius / norm))
    if optimizer is not None:
        state = optimizer.state.get(model.rho_c, {})
        moment = state.get("exp_avg")
        if moment is not None:
            radial = model.rho_c - anchor
            coefficient = float((moment * radial).sum() / radial.square().sum())
            # Adam updates in the negative-moment direction. A negative coefficient would
            # immediately push outward again and waste the next update on projection.
            if coefficient < 0:
                moment.sub_(coefficient * radial)
    return radius, True


def restore_named_adam_state(optimizer, model, checkpoint_path, names):
    """Restore mature Adam moments by parameter name from an additive checkpoint."""
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = blob.get("optimizer")
    if saved is None or len(saved.get("param_groups", [])) != 1:
        raise RuntimeError("optimizer parent has no single-group Adam state")
    identifiers = saved["param_groups"][0]["params"]
    if len(identifiers) != len(names):
        raise RuntimeError("optimizer parent parameter count does not match additive order")
    restored = []
    for name, identifier in zip(names, identifiers):
        parameter = getattr(model, name)
        source = saved["state"].get(identifier)
        if source is None or tuple(source["exp_avg"].shape) != tuple(parameter.shape):
            raise RuntimeError(f"optimizer parent state mismatch for {name}")
        optimizer.state[parameter] = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in source.items()
        }
        restored.append(name)
    return int(blob.get("iter", -1)), restored


def restore_joint_adam_state(optimizer, model, checkpoint_path):
    """Restore both Phi and mature-block moments from a prior free-Phi joint fit."""
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if bool(blob.get("spectral_scales_only", False)):
        raise RuntimeError("joint optimizer parent used spectral amplitudes, not free Phi")
    saved = blob.get("optimizer")
    if saved is None:
        raise RuntimeError("joint optimizer parent has no optimizer state")
    identifiers = [
        identifier
        for group in saved.get("param_groups", [])
        for identifier in group["params"]
    ]
    if len(identifiers) != len(JOINT_OPTIMIZER_PARAMETER_NAMES):
        raise RuntimeError("joint optimizer parent parameter count/order is incompatible")
    restored = []
    for name, identifier in zip(JOINT_OPTIMIZER_PARAMETER_NAMES, identifiers):
        parameter = getattr(model, name)
        source = saved["state"].get(identifier)
        if source is None or tuple(source["exp_avg"].shape) != tuple(parameter.shape):
            raise RuntimeError(f"joint optimizer parent state mismatch for {name}")
        optimizer.state[parameter] = {
            key: value.detach().clone() if torch.is_tensor(value) else value
            for key, value in source.items()
        }
        restored.append(name)
    return int(blob.get("iter", -1)), restored


def add_normalizer_rule_correction(model, ix, parameters, high, low):
    """Add d(log Z_high-log Z_low) to an existing negative-LL gradient.

    If every trip in the low-rule minibatch is used here, the resulting gradient is
    *exactly* the gradient of the high-rule minibatch likelihood.  A uniform sub-batch is
    therefore an unbiased estimator of that same gradient.  The positive phase cancels
    from the rule difference, so this operation cannot change the version-4 energy.
    """
    install(model, high)
    logz_high = model.log_Z(ix, drop_empty=True).mean()
    grad_high = torch.autograd.grad(logz_high, parameters)
    install(model, low)
    logz_low = model.log_Z(ix, drop_empty=True).mean()
    grad_low = torch.autograd.grad(logz_low, parameters)
    corrections = [high_grad - low_grad
                   for high_grad, low_grad in zip(grad_high, grad_low)]
    for parameter, correction in zip(parameters, corrections):
        parameter.grad.add_(correction)
    return logz_high, logz_low, corrections


@torch.no_grad()
def validation(model, batcher, trips, qlow, qhigh, qaudit, chunk,
               correction_trips, fidelity_audit_trips):
    low, high = [], []
    audit = []
    cancellation = {"low": 0.0, "high": 0.0, "audit": 0.0}
    for start in range(0, len(trips), chunk):
        sub = trips[start:start + chunk]
        ix, ctx, line_ctx, house, li, lt, lc, _ = batcher.make(sub)
        model.house, model.ctx = house, ctx
        install(model, qlow)
        low.append(joint_values(model, ix, li, lt, lc, line_ctx).cpu())
        cancellation["low"] = max(
            cancellation["low"], float(model._last_quad_log_condition.max()))
        if start < correction_trips:
            use = min(len(sub), correction_trips - start)
            if use != len(sub):
                ix, ctx, line_ctx, house, li, lt, lc, _ = batcher.make(sub[:use])
                model.house, model.ctx = house, ctx
            install(model, qhigh)
            high.append(joint_values(model, ix, li, lt, lc, line_ctx).cpu())
            cancellation["high"] = max(
                cancellation["high"], float(model._last_quad_log_condition.max()))
            if start < fidelity_audit_trips:
                audit_use = min(use, fidelity_audit_trips - start)
                if audit_use != use:
                    ix, ctx, line_ctx, house, li, lt, lc, _ = batcher.make(
                        sub[:audit_use])
                    model.house, model.ctx = house, ctx
                install(model, qaudit)
                audit.append(joint_values(model, ix, li, lt, lc, line_ctx).cpu())
                cancellation["audit"] = max(
                    cancellation["audit"],
                    float(model._last_quad_log_condition.max()))
    low = torch.cat(low).numpy()
    high = torch.cat(high).numpy()
    audit = torch.cat(audit).numpy()
    correction = high - low[:len(high)]
    estimate = float(low.mean() + correction.mean())
    correction_se = float(correction.std(ddof=1) / math.sqrt(len(correction)))
    fidelity = audit - high[:len(audit)]
    return {
        "controlled_high_loglik": estimate,
        "low_loglik": float(low.mean()),
        "low_standard_error": float(low.std(ddof=1) / math.sqrt(len(low))),
        "high_minus_low_mean": float(correction.mean()),
        "high_minus_low_standard_error": correction_se,
        "high_minus_low_max_abs": float(np.abs(correction).max()),
        "audit_minus_high_mean": float(fidelity.mean()),
        "audit_minus_high_standard_error": float(
            fidelity.std(ddof=1) / math.sqrt(len(fidelity))),
        "audit_minus_high_max_abs": float(np.abs(fidelity).max()),
        "low_max_log_cancellation": cancellation["low"],
        "high_max_log_cancellation": cancellation["high"],
        "audit_max_log_cancellation": cancellation["audit"],
        "trips": int(len(low)),
        "correction_trips": int(len(high)),
        "fidelity_audit_trips": int(len(audit)),
    }


def atomic_save(path, payload):
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary); os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", type=Path, required=True)
    parser.add_argument("--resume", type=Path,
                        help="resume this trainer; --iters is the final update")
    parser.add_argument("--label", default="run277_multifidelity_rank7")
    parser.add_argument("--rank", type=int, default=7)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--batch", type=int, default=24)
    parser.add_argument("--correction-batch", type=int, default=4)
    parser.add_argument(
        "--middle-correction-batch", type=int, default=0,
        help=("if positive, use q_r on the full batch, this many contexts for "
              "q_(r+1)-q_r, and --correction-batch for q_(r+2)-q_(r+1)"))
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--interaction-lr", type=float, default=0.0,
                        help="Phi/spectral LR; <=0 uses --lr")
    parser.add_argument("--mature-lr", type=float, default=0.0,
                        help="other jointly fitted blocks' LR; <=0 uses --lr")
    parser.add_argument("--resume-lr", type=float, default=0.0,
                        help="override restored optimizer LR; <=0 preserves it")
    parser.add_argument("--lr-patience", type=int, default=0,
                        help="validation evaluations without improvement before LR decay")
    parser.add_argument("--lr-factor", type=float, default=0.5)
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--convergence-patience", type=int, default=12)
    parser.add_argument("--convergence-min-updates", type=int, default=500)
    parser.add_argument("--validation-min-delta", type=float, default=1e-5)
    parser.add_argument(
        "--require-convergence", action="store_true",
        help="exit nonzero if the joint optimizer reaches a stop without convergence")
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--clip", type=float, default=10.0)
    parser.add_argument("--validation-trips", type=int, default=384)
    parser.add_argument("--audit-trips", type=int, default=48)
    parser.add_argument("--fidelity-audit-trips", type=int, default=8)
    parser.add_argument("--fidelity-max-abs", type=float, default=0.01)
    parser.add_argument("--fidelity-mean-abs", type=float, default=0.0,
                        help="if >0, gate the population-mean objective error; max stays logged")
    parser.add_argument("--spectral-max", type=float, default=0.0,
                        help="certified operator-norm radius; <=0 uses initial norm")
    parser.add_argument("--spectral-scales-only", action="store_true",
                        help="hold audited spectral directions fixed; fit r amplitudes")
    parser.add_argument(
        "--spectral-subspace-matrix", action="store_true",
        help=("fit an r-by-r transform inside the audited product subspace; "
              "all catalogue rows remain active"))
    parser.add_argument("--recalibrate-basic", action="store_true",
                        help="jointly fit lam and rho_0 after inserting interactions")
    parser.add_argument("--recalibrate-lam", action="store_true")
    parser.add_argument("--recalibrate-rho0", action="store_true")
    parser.add_argument("--joint-all-incidence", action="store_true",
                        help="jointly optimize every original version-4 incidence block")
    parser.add_argument("--category-size-orthogonal", action="store_true",
                        help=("use the exact size-centred rho_c coordinate and compensate "
                              "rho_0 after every update; probability law is unchanged"))
    parser.add_argument("--category-reference-prior", type=float, default=100.0)
    parser.add_argument("--rho-c-anchor", type=Path,
                        help="additive checkpoint defining the rho_c trust-region centre")
    parser.add_argument("--rho-c-trust-radius", type=float, default=0.0,
                        help="L2 radius around --rho-c-anchor; <=0 disables projection")
    parser.add_argument(
        "--rho-c-max-category-reward", type=float, default=0.0,
        help=("cap (-rho_c)+ choose(m_c,2) over complete support; "
              "<=0 disables the support-aware projection"))
    parser.add_argument("--optimizer-parent", type=Path,
                        help="restore mature additive Adam moments by parameter name")
    parser.add_argument("--joint-optimizer-parent", type=Path,
                        help="restore Phi and all incidence Adam moments from a joint fit")
    parser.add_argument("--regression-patience", type=int, default=0,
                        help="abort after this many consecutive materially worse evals")
    parser.add_argument("--regression-delta", type=float, default=1e-5)
    parser.add_argument("--validation-chunk", type=int, default=24)
    parser.add_argument("--eval-every", type=int, default=50)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=26701)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    fit_lam = bool(args.joint_all_incidence or args.recalibrate_basic
                   or args.recalibrate_lam)
    fit_rho0 = bool(args.joint_all_incidence or args.recalibrate_basic
                    or args.recalibrate_rho0)
    if not 0 <= args.correction_batch <= args.batch:
        raise ValueError("correction batch must lie in [0,batch]")
    if not 0 <= args.middle_correction_batch <= args.batch:
        raise ValueError("middle correction batch must lie in [0,batch]")
    if args.middle_correction_batch > 0 and args.correction_batch == 0:
        raise ValueError("three-level training requires a final correction batch")
    if args.category_size_orthogonal and not args.joint_all_incidence:
        raise ValueError("--category-size-orthogonal requires --joint-all-incidence")
    if args.category_reference_prior < 0:
        raise ValueError("--category-reference-prior must be nonnegative")
    if args.spectral_scales_only and args.spectral_subspace_matrix:
        raise ValueError("choose spectral scales or a subspace matrix, not both")
    if (args.rho_c_anchor is None) != (args.rho_c_trust_radius <= 0):
        raise ValueError("provide both --rho-c-anchor and a positive --rho-c-trust-radius")
    if args.rho_c_anchor is not None and not args.joint_all_incidence:
        raise ValueError("rho_c trust projection requires --joint-all-incidence")
    if args.rho_c_max_category_reward < 0:
        raise ValueError("--rho-c-max-category-reward must be nonnegative")
    if args.rho_c_max_category_reward > 0 and not args.joint_all_incidence:
        raise ValueError(
            "category reward projection requires --joint-all-incidence")
    if args.lr_patience < 0 or not 0 < args.lr_factor < 1 or args.min_lr <= 0:
        raise ValueError("invalid convergence scheduler")
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    data = build()
    initial_path = args.initial if args.initial.is_absolute() else ROOT / args.initial
    model, initial_blob, meta = load_checkpoint(initial_path, data)
    if args.fidelity_audit_trips > args.audit_trips:
        raise ValueError("fidelity-audit-trips cannot exceed audit-trips")
    singular = torch.linalg.svdvals(model.phi)
    active_rank = int((singular > singular[0] * 1e-10).sum())
    if active_rank != args.rank or float(model.phi[:, args.rank:].abs().max()) != 0.0:
        raise RuntimeError(f"initial checkpoint must have exactly {args.rank} "
                           "active columns/rank")
    active_rows = model.phi[:, :args.rank].norm(dim=1) > 0
    if not bool(active_rows.any()):
        raise RuntimeError("initial checkpoint has no interaction products")
    # A zero row is an initialization fact, not a structural exclusion in version 4.
    # Once another row is nonzero, its joint score need not vanish; freezing it would fit
    # a different restricted interaction model.  The full joint mode therefore lets all
    # catalogue rows move while keeping only the certified rank columns active.
    if args.joint_all_incidence:
        active_rows = torch.ones_like(active_rows)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.phi.requires_grad_(True)
    if args.joint_all_incidence:
        for name in INCIDENCE_PARAMETER_NAMES:
            getattr(model, name).requires_grad_(True)
    else:
        if fit_lam:
            model.lam.requires_grad_(True)
        if fit_rho0:
            model.rho_0_free.requires_grad_(True)
    spectral_basis = model.phi.detach().clone()
    spectral_scale = None
    spectral_transform = None
    if args.spectral_scales_only:
        # The eigenspace audit validates these columns, not arbitrary movement of every
        # product row.  Phi = U diag(s) retains the accepted directions and reduces the
        # residual fit from interaction_products * rank parameters to exactly rank.
        spectral_scale = torch.nn.Parameter(torch.ones(
            args.rank, dtype=model.phi.dtype, device=model.phi.device))
    elif args.spectral_subspace_matrix:
        # Phi0 has full column rank. Phi=Phi0 A spans every rank-r PSD kernel in the
        # split-half-certified product subspace while avoiding J*r row-wise Adam states.
        spectral_transform = torch.nn.Parameter(torch.eye(
            args.rank, dtype=model.phi.dtype, device=model.phi.device))
    three_level = args.middle_correction_batch > 0
    low_level = args.rank if three_level else args.rank + 1
    middle_level = args.rank + 1 if three_level else None
    high_level, audit_level = args.rank + 2, args.rank + 3
    qlow = rule(model, args.rank, low_level)
    qmiddle = rule(model, args.rank, middle_level) if three_level else None
    qhigh = rule(model, args.rank, high_level)
    qaudit = rule(model, args.rank, audit_level)
    initial_spectral_norm = float(singular[0])
    spectral_max = (args.spectral_max if args.spectral_max > 0
                    else initial_spectral_norm)
    batcher = Batcher(data, Features(int(data["n_item"]), int(data["n_store"]), 712),
                      int(meta["nmax"]))
    train = supported_trips(data, 0, int(meta["nmax"]))
    valid_population = supported_trips(data, 1, int(meta["nmax"]))
    valid = valid_population[np.random.default_rng(args.seed + 1).permutation(
        len(valid_population))[:args.validation_trips]]
    rng = np.random.default_rng(args.seed + 2)
    category_reference = (category_pair_reference(
        data, train, int(meta["nmax"]), int(data["n_cat"]),
        args.category_reference_prior).to(
            dtype=model.rho_0_free.dtype, device=model.rho_0_free.device)
                          if args.category_size_orthogonal else None)
    rho_c_capacities = category_capacities(
        data, int(data["n_cat"]), int(meta["nmax"]))
    rho_c_anchor = None
    if args.rho_c_anchor is not None:
        anchor_path = (args.rho_c_anchor if args.rho_c_anchor.is_absolute()
                       else ROOT / args.rho_c_anchor)
        anchor_blob = torch.load(anchor_path, map_location="cpu", weights_only=False)
        rho_c_anchor = anchor_blob["model"]["rho_c"].to(
            dtype=model.rho_c.dtype, device=model.rho_c.device)
        if rho_c_anchor.shape != model.rho_c.shape:
            raise RuntimeError("rho_c anchor shape does not match the model")
    observed_operator = build_observed_phi_operator(data, train, int(data["n_item"]))
    observed_basic = (build_observed_basic_scores(
        data, train, int(data["n_item"]), int(meta["nmax"]))
                      if (fit_lam or fit_rho0) else None)
    optimized = ([spectral_scale] if args.spectral_scales_only else
                 [spectral_transform] if args.spectral_subspace_matrix else
                 [model.phi])
    if args.joint_all_incidence:
        optimized.extend(getattr(model, name) for name in INCIDENCE_PARAMETER_NAMES
                         if name != "phi")
    else:
        if fit_lam:
            optimized.append(model.lam)
        if fit_rho0:
            optimized.append(model.rho_0_free)
    interaction_lr = args.interaction_lr if args.interaction_lr > 0 else args.lr
    mature_lr = args.mature_lr if args.mature_lr > 0 else args.lr
    if args.joint_all_incidence:
        optimizer = torch.optim.AdamW([
            {"params": optimized[:1], "lr": interaction_lr},
            {"params": optimized[1:], "lr": mature_lr},
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.AdamW(optimized, lr=args.lr,
                                      weight_decay=args.weight_decay)
    optimizer_parent_iteration = None
    restored_optimizer_names = []
    if args.optimizer_parent is not None and args.joint_optimizer_parent is not None:
        raise ValueError("choose additive or joint optimizer parent, not both")
    if ((args.optimizer_parent is not None or args.joint_optimizer_parent is not None)
            and args.resume is not None):
        raise ValueError("optimizer parents apply only to a new joint continuation")
    if args.optimizer_parent is not None:
        if not args.joint_all_incidence:
            raise ValueError("--optimizer-parent requires --joint-all-incidence")
        optimizer_parent_path = (args.optimizer_parent if args.optimizer_parent.is_absolute()
                                 else ROOT / args.optimizer_parent)
        optimizer_parent_iteration, restored_optimizer_names = restore_named_adam_state(
            optimizer, model, optimizer_parent_path, ADDITIVE_PARAMETER_NAMES)
    if args.joint_optimizer_parent is not None:
        if (not args.joint_all_incidence or args.spectral_scales_only
                or args.spectral_subspace_matrix):
            raise ValueError("--joint-optimizer-parent requires free-Phi joint fitting")
        joint_parent_path = (
            args.joint_optimizer_parent if args.joint_optimizer_parent.is_absolute()
            else ROOT / args.joint_optimizer_parent)
        optimizer_parent_iteration, restored_optimizer_names = restore_joint_adam_state(
            optimizer, model, joint_parent_path)
    output = ROOT / "out"
    log_path = output / f"v3_{args.label}.log"
    checkpoint_path = output / f"v3_{args.label}.pt"
    best_path = output / f"v3_{args.label}_best.pt"
    history_path = output / f"v3_{args.label}_history.json"
    start_iteration = 0
    resumed = None
    if args.resume is not None:
        resume_path = args.resume if args.resume.is_absolute() else ROOT / args.resume
        resumed = torch.load(resume_path, map_location="cpu", weights_only=False)
        if int(resumed.get("active_rank", -1)) != args.rank:
            raise RuntimeError("resume checkpoint has a different active rank")
        if bool(resumed.get("spectral_scales_only", False)) != bool(
                args.spectral_scales_only):
            raise RuntimeError("resume checkpoint has a different interaction parameterization")
        if bool(resumed.get("spectral_subspace_matrix", False)) != bool(
                args.spectral_subspace_matrix):
            raise RuntimeError("resume checkpoint has a different spectral subspace")
        resumed_lam = bool(resumed.get(
            "recalibrate_lam", resumed.get("recalibrate_basic", False)))
        resumed_rho0 = bool(resumed.get(
            "recalibrate_rho0", resumed.get("recalibrate_basic", False)))
        if resumed_lam != fit_lam or resumed_rho0 != fit_rho0:
            raise RuntimeError("resume checkpoint has a different basic parameterization")
        if bool(resumed.get("joint_all_incidence", False)) != bool(
                args.joint_all_incidence):
            raise RuntimeError("resume checkpoint has a different joint parameterization")
        prior = resumed["config"]
        for key in ("batch", "correction_batch", "middle_correction_batch", "seed"):
            if int(prior[key]) != int(getattr(args, key)):
                raise RuntimeError(f"--{key} must remain {prior[key]} when resuming")
        prior_reward = float(prior.get("rho_c_max_category_reward", 0.0))
        if prior_reward != float(args.rho_c_max_category_reward):
            raise RuntimeError(
                "--rho-c-max-category-reward must remain "
                f"{prior_reward:g} when resuming")
        model.load_state_dict(resumed["model"])
        optimizer.load_state_dict(resumed["optimizer"])
        optimizer_parent_iteration = resumed.get("optimizer_parent_iteration")
        restored_optimizer_names = list(resumed.get("restored_optimizer_names", []))
        if args.resume_lr > 0:
            for group in optimizer.param_groups:
                group["lr"] = args.resume_lr
        if spectral_scale is not None:
            spectral_scale.data.copy_(resumed["spectral_scale"])
        if spectral_transform is not None:
            spectral_transform.data.copy_(resumed["spectral_transform"])
        start_iteration = int(resumed["iter"])
        if args.iters <= start_iteration:
            raise RuntimeError("--iters must exceed the resumed iteration")
        # Each completed update makes these two NumPy draws and no other training RNG draw.
        for _ in range(start_iteration):
            rng.choice(len(train), size=args.batch, replace=False)
            if args.middle_correction_batch:
                rng.choice(args.batch, size=args.middle_correction_batch,
                           replace=False)
            if args.correction_batch:
                rng.choice(args.batch, size=args.correction_batch, replace=False)
    sys.stdout = Tee(sys.stdout, log_path, append=args.resume is not None)
    prefix = f"[multifidelity-r{args.rank}]"
    print(f"{prefix} initial={initial_path}", flush=True)
    print(f"{prefix} {int(active_rows.sum())}/{model.J} interaction products; "
          f"rank={args.rank}; version-4 probability law unchanged", flush=True)
    if three_level:
        node_work = (args.batch * len(qlow[1])
                     + args.middle_correction_batch
                     * (len(qmiddle[1]) + len(qlow[1]))
                     + args.correction_batch
                     * (len(qhigh[1]) + len(qmiddle[1])))
        print(f"{prefix} unbiased telescoping target q{high_level}: q{low_level} "
              f"nodes={len(qlow[1])} on B={args.batch}; "
              f"q{middle_level}-q{low_level} nodes="
              f"{len(qmiddle[1])}+{len(qlow[1])} on m="
              f"{args.middle_correction_batch}; q{high_level}-q{middle_level} "
              f"nodes={len(qhigh[1])}+{len(qmiddle[1])} on m="
              f"{args.correction_batch}; node-context work={node_work}", flush=True)
    elif args.correction_batch:
        print(f"{prefix} q{low_level} nodes={len(qlow[1])} on B={args.batch}; unbiased "
              f"q{high_level}-q{low_level} correction nodes="
              f"{len(qhigh[1])}+{len(qlow[1])} on m="
              f"{args.correction_batch}", flush=True)
    else:
        print(f"{prefix} certified q{low_level} optimization rule: "
              f"{len(qlow[1])} nodes on B={args.batch}; q{high_level} is the validation "
              "target and is not evaluated on every update", flush=True)
    fidelity_contract = (f"|mean delta|={args.fidelity_mean_abs:g}"
                         if args.fidelity_mean_abs > 0 else
                         f"max |delta|={args.fidelity_max_abs:g}")
    print(f"{prefix} q{audit_level} fidelity guard on "
          f"{args.fidelity_audit_trips} trips; {fidelity_contract}; "
          f"spectral radius={spectral_max:.6f}", flush=True)
    print(f"{prefix} interaction optimization="
          f"{'fixed spectral directions, '+str(args.rank)+' amplitudes' if args.spectral_scales_only else 'audited product subspace, '+str(args.rank*args.rank)+' transform coordinates' if args.spectral_subspace_matrix else 'free active rows'}",
          flush=True)
    print(f"{prefix} joint basic recalibration="
          f"lam={'on' if fit_lam else 'off'}, rho_0={'on' if fit_rho0 else 'off'}; "
          f"{'exact observed sufficient statistics' if (fit_lam or fit_rho0) else 'off'}",
          flush=True)
    print(f"{prefix} original incidence blocks jointly optimized="
          f"{'yes' if args.joint_all_incidence else 'no'}", flush=True)
    if category_reference is not None:
        print(f"{prefix} category score uses exact size-centred coordinates; "
              f"training-only reference prior={args.category_reference_prior:g}",
              flush=True)
    if rho_c_anchor is not None:
        print(f"{prefix} rho_c L2 trust radius={args.rho_c_trust_radius:.6f} "
              f"around {anchor_path}", flush=True)
    if args.rho_c_max_category_reward > 0:
        initial_category_safety = project_category_reward_(
            model, rho_c_capacities, args.rho_c_max_category_reward,
            optimizer=optimizer)
        print(f"{prefix} complete-support category constraint: max attractive "
              f"reward={args.rho_c_max_category_reward:g} nats; initial max="
              f"{initial_category_safety['maximum_reward_after']:.6f}",
              flush=True)
    if args.joint_all_incidence:
        print(f"{prefix} learning rates: interaction={interaction_lr:g}, "
              f"mature incidence={mature_lr:g}", flush=True)
    if restored_optimizer_names:
        phi_state = "restored" if "phi" in restored_optimizer_names else "starts fresh"
        print(f"{prefix} restored Adam moments for {len(restored_optimizer_names)} "
              f"named incidence blocks from parent iteration "
              f"{optimizer_parent_iteration}; Phi {phi_state}", flush=True)
    print(f"{prefix} log: {log_path}", flush=True)
    resumed_value = validation(model, batcher, valid, qlow, qhigh, qaudit,
                               args.validation_chunk, args.audit_trips,
                               args.fidelity_audit_trips)
    if resumed is None:
        initial = resumed_value
        evaluations = [{"iter": 0, **initial}]
        records = []
        best_score = initial["controlled_high_loglik"]
        best_iteration = 0
        plateau_anchor_score = best_score
        plateau_evaluations = 0
        evaluations_since_best = 0
        consecutive_regressions = 0
    else:
        evaluations = list(resumed["evaluations"])
        records = list(resumed["records"])
        initial = {key: value for key, value in evaluations[0].items()
                   if key != "iter"}
        best_score = float(resumed["best_validation"])
        best_iteration = int(resumed["best_iteration"])
        scheduler_state = resumed.get("scheduler", {})
        plateau_anchor_score = float(
            scheduler_state.get("plateau_anchor_score", best_score))
        plateau_evaluations = int(scheduler_state.get("plateau_evaluations", 0))
        evaluations_since_best = int(
            scheduler_state.get("evaluations_since_best", 0))
        consecutive_regressions = int(
            scheduler_state.get("consecutive_regressions", 0))
        saved = next(x for x in reversed(evaluations)
                     if int(x["iter"]) == start_iteration)
        if abs(resumed_value["controlled_high_loglik"]
               - saved["controlled_high_loglik"]) > 5e-10:
            raise RuntimeError("resume validation panel does not reproduce checkpoint")
        print(f"{prefix} resumed iteration {start_iteration} from {resume_path}; "
              "optimizer and minibatch stream restored", flush=True)
    started = time.perf_counter()
    display_value = resumed_value if resumed is not None else initial
    print(f"{prefix} current controlled validation="
          f"{display_value['controlled_high_loglik']:.6f}; "
          f"q{high_level}-q{low_level}="
          f"{display_value['high_minus_low_mean']:+.6f}; "
          f"q{audit_level}-q{high_level} max="
          f"{display_value['audit_minus_high_max_abs']:.6f}", flush=True)
    initial_fidelity_failed = (
        abs(resumed_value["audit_minus_high_mean"]) > args.fidelity_mean_abs
        if args.fidelity_mean_abs > 0 else
        resumed_value["audit_minus_high_max_abs"] > args.fidelity_max_abs)
    if initial_fidelity_failed:
        raise RuntimeError("initial checkpoint lies outside the certified "
                           "quadrature-fidelity envelope")
    if (resumed is not None and resumed_value["controlled_high_loglik"]
            > best_score):
        best_score = resumed_value["controlled_high_loglik"]
        best_iteration = start_iteration
    if (resumed is not None and resumed_value["controlled_high_loglik"]
            > plateau_anchor_score + args.validation_min_delta):
        plateau_anchor_score = resumed_value["controlled_high_loglik"]
        plateau_evaluations = 0
        evaluations_since_best = 0
        consecutive_regressions = 0
        print(f"{prefix} resumed checkpoint is a new material plateau anchor at "
              f"{start_iteration}", flush=True)

    def payload(iteration):
        return {
            "format": 3,
            "estimator": ((f"rank{args.rank}_smolyak_q{high_level}_"
                           f"telescoping_q{low_level}_q{middle_level}_score")
                          if three_level else
                          ((f"rank{args.rank}_smolyak_q{high_level}_"
                           "multifidelity_score") if args.correction_batch else
                          (f"rank{args.rank}_smolyak_q{low_level}_"
                           f"certified_against_q{high_level}"))),
            "iter": iteration, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {**vars(args),
                       "artifact": initial_blob["config"]["artifact"]},
            "initial": str(initial_path), "active_rank": args.rank,
            "interaction_products": int(active_rows.sum()),
            "spectral_scales_only": bool(args.spectral_scales_only),
            "spectral_subspace_matrix": bool(args.spectral_subspace_matrix),
            "recalibrate_basic": bool(args.recalibrate_basic),
            "recalibrate_lam": fit_lam,
            "recalibrate_rho0": fit_rho0,
            "joint_all_incidence": bool(args.joint_all_incidence),
            "optimizer_parent_iteration": optimizer_parent_iteration,
            "restored_optimizer_names": restored_optimizer_names,
            "spectral_scale": (spectral_scale.detach().clone()
                               if spectral_scale is not None else None),
            "spectral_transform": (spectral_transform.detach().clone()
                                   if spectral_transform is not None else None),
            "best_validation": best_score, "best_iteration": best_iteration,
            "scheduler": {
                "plateau_evaluations": plateau_evaluations,
                "evaluations_since_best": evaluations_since_best,
                "consecutive_regressions": consecutive_regressions,
                "plateau_anchor_score": plateau_anchor_score,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "learning_rates": [float(group["lr"])
                                   for group in optimizer.param_groups],
            },
            "evaluations": evaluations, "records": records,
        }

    # On a same-label resume, the current checkpoint need not be the numerical best.
    # Do not overwrite an existing best artifact with a worse resume point.
    if resumed is None or start_iteration == best_iteration or not best_path.exists():
        atomic_save(best_path, payload(start_iteration))
    converged = False
    regression_aborted = False
    fidelity_stopped = False
    for iteration in range(start_iteration + 1, args.iters + 1):
        tick = time.perf_counter()
        if args.spectral_scales_only:
            with torch.no_grad():
                model.phi.copy_(spectral_basis)
                model.phi[:, :args.rank].mul_(spectral_scale.unsqueeze(0))
            model.phi.grad = None
        elif args.spectral_subspace_matrix:
            install_spectral_transform(
                model.phi, spectral_basis, spectral_transform, args.rank)
            model.phi.grad = None
        trips = train[rng.choice(len(train), size=args.batch, replace=False)]
        ix, ctx, line_ctx, house, li, lt, lc, _ = batcher.make(trips)
        model.house, model.ctx = house, ctx
        optimizer.zero_grad(set_to_none=True)
        rho_c_before = (model.rho_c.detach().clone()
                        if category_reference is not None else None)
        install(model, qlow)
        low_score = joint_values(model, ix, li, lt, lc, line_ctx).mean()
        (-low_score).backward()
        batch_positive = observed_phi_score(model, li, lt, ix.B)
        full_positive = full_observed_phi_score(observed_operator, model.phi)
        model.phi.grad.add_(batch_positive - full_positive)
        if fit_lam or fit_rho0:
            batch_basic = observed_basic_scores(model, li, lt, ix.B)
        if fit_lam:
            model.lam.grad.add_(batch_basic["lam"] - observed_basic["lam"])
        if fit_rho0:
            model.rho_0_free.grad.add_(
                batch_basic["rho_0_free"] - observed_basic["rho_0_free"])

        middle_correction = torch.zeros_like(model.phi)
        if three_level:
            middle_select = rng.choice(
                args.batch, size=args.middle_correction_batch, replace=False)
            middle_trips = trips[middle_select]
            mix, mctx, mline_ctx, mhouse, *_ = batcher.make(middle_trips)
            model.house, model.ctx = mhouse, mctx
            correction_parameters = [model.phi]
            if args.joint_all_incidence:
                correction_parameters.extend(
                    getattr(model, name) for name in INCIDENCE_PARAMETER_NAMES
                    if name != "phi")
            else:
                if fit_lam:
                    correction_parameters.append(model.lam)
                if fit_rho0:
                    correction_parameters.append(model.rho_0_free)
            _middle_logz, _base_logz, middle_corrections = (
                add_normalizer_rule_correction(
                    model, mix, correction_parameters, qmiddle, qlow))
            middle_correction = middle_corrections[0]

        if args.correction_batch:
            select = rng.choice(args.batch, size=args.correction_batch, replace=False)
            correction_trips = trips[select]
            cix, cctx, cline_ctx, chouse, *_ = batcher.make(correction_trips)
            model.house, model.ctx = chouse, cctx
            correction_parameters = [model.phi]
            if args.joint_all_incidence:
                correction_parameters.extend(
                    getattr(model, name) for name in INCIDENCE_PARAMETER_NAMES
                    if name != "phi")
            else:
                if fit_lam:
                    correction_parameters.append(model.lam)
                if fit_rho0:
                    correction_parameters.append(model.rho_0_free)
            correction_low_rule = qmiddle if three_level else qlow
            logz_high, logz_low, corrections = add_normalizer_rule_correction(
                model, cix, correction_parameters, qhigh, correction_low_rule)
            correction = corrections[0]
        else:
            logz_high = torch.zeros((), dtype=low_score.dtype)
            logz_low = torch.zeros_like(logz_high)
            correction = torch.zeros_like(model.phi)
        model.phi.grad[~active_rows] = 0.0
        model.phi.grad[:, args.rank:] = 0.0
        if args.spectral_scales_only:
            # Chain rule for Phi[:,k] = U[:,k] s[k].  Keeping Adam in this
            # r-dimensional coordinate system is essential: elementwise Adam on Phi
            # would immediately rotate the supposedly fixed directions.
            spectral_scale.grad = spectral_scale_gradient(
                model.phi.grad, spectral_basis, args.rank)
            clip_parameters = [spectral_scale] + [
                parameter for parameter in optimized if parameter is not spectral_scale]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                clip_parameters, args.clip))
        elif args.spectral_subspace_matrix:
            spectral_transform.grad = spectral_transform_gradient(
                model.phi.grad, spectral_basis, args.rank)
            clip_parameters = [spectral_transform] + [
                parameter for parameter in optimized
                if parameter is not spectral_transform]
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                clip_parameters, args.clip))
        else:
            clip_parameters = optimized
            grad_norm = float(torch.nn.utils.clip_grad_norm_(
                clip_parameters, args.clip))
        optimizer.step()
        with torch.no_grad():
            if args.spectral_scales_only:
                # A whole-column sign flip leaves Phi Phi' unchanged.  A nonnegative
                # representative avoids an optimizer crossing that redundant gauge.
                spectral_scale.clamp_(min=0.0)
                model.phi.copy_(spectral_basis)
                model.phi[:, :args.rank].mul_(spectral_scale.unsqueeze(0))
            elif args.spectral_subspace_matrix:
                install_spectral_transform(
                    model.phi, spectral_basis, spectral_transform, args.rank)
            model.phi[~active_rows] = 0.0
            model.phi[:, args.rank:] = 0.0
            current_norm = float(torch.linalg.svdvals(
                model.phi[:, :args.rank])[0])
            if current_norm > spectral_max:
                if spectral_scale is None:
                    _before, current_norm, _changed = project_active_spectral_ball(
                        model.phi, args.rank, spectral_max)
                    if spectral_transform is not None:
                        # The projection retains the current column space, so it has
                        # exact r-by-r coordinates in the audited basis.
                        basis_active = spectral_basis[:, :args.rank]
                        gram = basis_active.T @ basis_active
                        spectral_transform.copy_(torch.linalg.solve(
                            gram, basis_active.T @ model.phi[:, :args.rank]))
                else:
                    ratio = spectral_max / current_norm
                    model.phi[:, :args.rank].mul_(ratio)
                    spectral_scale.mul_(ratio)
            if fit_lam and fit_rho0:
                # Exact lam--rho_0 gauge transformation: every basket energy is unchanged.
                mean_lam = model.lam.mean().clone()
                model.lam.sub_(mean_lam)
                sizes = torch.arange(
                    1, model.rho_0_free.numel() + 1, dtype=model.lam.dtype,
                    device=model.lam.device)
                model.rho_0_free.sub_(mean_lam * sizes)
            if args.joint_all_incidence:
                model.project_context_gauges()
                model.project_rho_c(-1.5)
                rho_c_trust_norm, rho_c_projected = (0.0, False)
                if rho_c_anchor is not None:
                    rho_c_trust_norm, rho_c_projected = project_rho_c_trust_region(
                        model, rho_c_anchor, args.rho_c_trust_radius, optimizer)
                category_safety = None
                if args.rho_c_max_category_reward > 0:
                    category_safety = project_category_reward_(
                        model, rho_c_capacities,
                        args.rho_c_max_category_reward, optimizer=optimizer)
                if category_reference is not None:
                    # Holding the centred size coordinate fixed means
                    # delta rho_0(n) = -sum_c delta rho_c a_c(n). This makes the update
                    # -sum_c delta rho_c[T_c-a_c(n)] while leaving the energy family and
                    # all full-support likelihood identities exactly unchanged.
                    model.rho_0_free.sub_(
                        (model.rho_c - rho_c_before) @ category_reference)
            else:
                rho_c_trust_norm, rho_c_projected = 0.0, False
                category_safety = None
        if not all(bool(torch.isfinite(parameter).all()) for parameter in optimized):
            raise FloatingPointError("non-finite jointly optimized parameter")
        row = {
            "iter": iteration, "low_train_loglik": float(low_score.detach()),
            "high_minus_low_logz": float((logz_high - logz_low).detach()),
            "middle_correction_grad_norm": float(middle_correction.norm()),
            "correction_grad_norm": float(correction.norm()),
            "grad_norm": grad_norm,
            "spectral_scale": (spectral_scale.detach().cpu().tolist()
                               if spectral_scale is not None else None),
            "spectral_transform_norm": (
                float(spectral_transform.detach().norm())
                if spectral_transform is not None else float("nan")),
            "lam_sd": float(model.lam.std().detach()),
            "spectral_norm": float(torch.linalg.svdvals(
                model.phi[:, :args.rank])[0]),
            "rho_c_trust_norm": rho_c_trust_norm,
            "rho_c_projected": rho_c_projected,
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
            print(f"{prefix} step {iteration:4d} q{low_level}LL="
                  f"{np.mean([x['low_train_loglik'] for x in window]):.5f} "
                  f"mid|g|={np.mean([x['middle_correction_grad_norm'] for x in window]):.4f} "
                  f"corr|g|={np.mean([x['correction_grad_norm'] for x in window]):.4f} "
                  f"op={row['spectral_norm']:.4f} "
                  f"cat.max={row['maximum_category_reward']:.3f} "
                  f"{np.mean([x['seconds'] for x in window]):.3f}s/it", flush=True)
        if iteration % args.eval_every == 0 or iteration == args.iters:
            current = validation(model, batcher, valid, qlow, qhigh, qaudit,
                                 args.validation_chunk, args.audit_trips,
                                 args.fidelity_audit_trips)
            evaluations.append({"iter": iteration, **current})
            change = (current["controlled_high_loglik"]
                      - initial["controlled_high_loglik"])
            print(f"{prefix} validation {iteration}: controlled "
                  f"q{high_level}={current['controlled_high_loglik']:.6f}, "
                  f"change={change:+.6f}, q{high_level}-q{low_level}="
                  f"{current['high_minus_low_mean']:+.6f}, "
                  f"q{audit_level}-q{high_level} max="
                  f"{current['audit_minus_high_max_abs']:.6f}, mean="
                  f"{current['audit_minus_high_mean']:+.6f}", flush=True)
            fidelity_failed = (
                abs(current["audit_minus_high_mean"]) > args.fidelity_mean_abs
                if args.fidelity_mean_abs > 0 else
                current["audit_minus_high_max_abs"] > args.fidelity_max_abs)
            if fidelity_failed:
                atomic_save(checkpoint_path, payload(iteration))
                fidelity_stopped = True
                print(f"{prefix} fidelity stop at {iteration}: {fidelity_contract}; "
                      "last certified best preserved", flush=True)
                break
            numerical_improved = current["controlled_high_loglik"] > best_score
            material_improved = (current["controlled_high_loglik"]
                                 > plateau_anchor_score
                                 + args.validation_min_delta)
            if numerical_improved:
                best_score = current["controlled_high_loglik"]
                best_iteration = iteration
            if material_improved:
                plateau_anchor_score = current["controlled_high_loglik"]
                plateau_evaluations = 0
                evaluations_since_best = 0
                consecutive_regressions = 0
            else:
                plateau_evaluations += 1
                evaluations_since_best += 1
                if (current["controlled_high_loglik"]
                        < best_score - args.regression_delta):
                    consecutive_regressions += 1
                else:
                    consecutive_regressions = 0
            if numerical_improved:
                atomic_save(best_path, payload(iteration))
                print(f"{prefix} new numerical best at {iteration}", flush=True)
            if (args.lr_patience > 0
                    and plateau_evaluations >= args.lr_patience):
                old_lrs, new_lrs = decay_learning_rates(
                    optimizer, args.lr_factor, args.min_lr)
                if new_lrs != old_lrs:
                    transitions = ", ".join(
                        f"{old:.3g}->{new:.3g}"
                        for old, new in zip(old_lrs, new_lrs))
                    print(f"{prefix} validation plateau: group LRs "
                          f"{transitions}", flush=True)
                plateau_evaluations = 0
            atomic_save(checkpoint_path, payload(iteration))
            history_path.write_text(json.dumps({
                "initial_validation": initial, "latest_validation": current,
                "best_validation": best_score, "best_iteration": best_iteration,
                "evaluations": evaluations, "records": records,
                "wall_seconds": time.perf_counter() - started}, indent=2) + "\n")
            if (args.lr_patience > 0
                    and iteration >= args.convergence_min_updates
                    and max(float(group["lr"])
                            for group in optimizer.param_groups) <= args.min_lr
                    and evaluations_since_best >= args.convergence_patience):
                converged = True
                print(f"{prefix} convergence declared at {iteration}: max lr="
                      f"{max(float(group['lr']) for group in optimizer.param_groups):.3g}, "
                      f"{evaluations_since_best} evaluations since best", flush=True)
                break
            if (args.regression_patience > 0
                    and consecutive_regressions >= args.regression_patience):
                regression_aborted = True
                print(f"{prefix} regression abort at {iteration}: "
                      f"{consecutive_regressions} consecutive validations below "
                      f"best by >{args.regression_delta:g}", flush=True)
                break
    outcome = ("converged" if converged else
               "regression-aborted" if regression_aborted else
               "fidelity-stopped" if fidelity_stopped else
               "reached safety ceiling")
    print(f"{prefix} {outcome} in "
          f"{time.perf_counter()-started:.1f}s; best={best_score:.6f} "
          f"at {best_iteration}", flush=True)
    if args.require_convergence and not converged:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
