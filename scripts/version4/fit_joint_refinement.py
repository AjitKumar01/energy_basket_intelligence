#!/usr/bin/env python3
"""Joint refinement of a staged Version-4 fit: every block moves together, from the staged optimum.

The staged pipeline is a two-step estimator: the additive block is fitted with Phi = 0 and then
frozen while Phi is solved.  Utilities fitted without interactions absorb part of them, and the
Phi stage cannot hand that back.  Following the one-step / refinement principle (a good starting
estimate followed by steps on the full likelihood), this stage starts from the staged final
checkpoint and takes gradient steps on the exact Version-4 likelihood with respect to all
parameters jointly:

    utilities, household taste and size, promotion, seasonality/store blocks, rho_c, rho_0, Phi,

with the certified price component kept frozen (it was fixed by held-out evidence; a free joint
price fit is not identified as well by the basket likelihood).  The energy, the Smolyak normalizer
at the pipeline's target level (rank + 2), the rho_c clamps and Phi's contract (rank fixed,
singular values <= 1, the staged 0 <= C <= I) are Model A's own.

Every `--eval-every` steps the validation likelihood is computed at the training rule and at a rule
one level finer; a growing gap means the optimizer is exploiting integration error.  The refined
checkpoint is written (pipeline checkpoint format) only when it beats the staged starting point on
the finer rule, and it is accepted only if the paired household-clustered 95% lower bound of the
validation gain is positive; otherwise the staged fit stands.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch

from checkpoint_io import ROOT, load_checkpoint
from data import build
from features import Features
from fit import Batcher
from fit_exact_additive import category_capacities, fitted_parameters, project_category_reward_
from pipeline_support import install_quadrature, smolyak_rule, supported_trips
from uncertainty import paired_score_summary

torch.set_default_dtype(torch.float64)


def log_likelihood(model, batcher, trips):
    ix, ctx, line_ctx, house, li, lt, lc, _lq = batcher.make(trips)
    model.house, model.ctx = house, ctx
    return model.energy(li, lt, lc, ix.B, line_ctx) - model.log_Z(ix, drop_empty=True)


def per_trip(model, batcher, trips, rule, chunk=128):
    install_quadrature(model, rule)
    with torch.no_grad():
        return torch.cat([log_likelihood(model, batcher, trips[s:s + chunk])
                          for s in range(0, len(trips), chunk)]).numpy()


def project_phi(model, rank):
    with torch.no_grad():
        model.phi[:, rank:] = 0.0
        U, S, Vh = torch.linalg.svd(model.phi[:, :rank], full_matrices=False)
        model.phi[:, :rank] = (U * S.clamp(max=1.0)) @ Vh


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True, help="staged final checkpoint")
    p.add_argument("--output", type=Path, required=True, help="refined checkpoint (pipeline format)")
    p.add_argument("--steps", type=int, default=600)
    p.add_argument("--batch", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--pool-prod", type=float, default=1.45)
    p.add_argument("--clip", type=float, default=10.0)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--patience", type=int, default=4)
    p.add_argument("--validation-trips", type=int, default=1024)
    p.add_argument("--level-offset", type=int, default=2, help="training rule level = rank + offset")
    p.add_argument("--seed", type=int, default=31001)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    started = time.time()

    data = build()
    model, blob, meta = load_checkpoint(args.checkpoint, data)
    rank = int(blob["active_rank"])
    nmax = int(meta["nmax"])
    train_rule = smolyak_rule(model, rank, rank + args.level_offset)
    check_rule = smolyak_rule(model, rank, rank + args.level_offset + 1)
    features = Features(int(data["n_item"]), int(data["n_store"]), include_recency=False)
    batcher = Batcher(data, features, nmax, include_recency=False)
    train = supported_trips(data, 0, nmax)
    valid_all = supported_trips(data, 1, nmax)
    rng = np.random.default_rng(args.seed)
    valid = np.sort(valid_all[rng.permutation(len(valid_all))[:args.validation_trips]])
    households = np.asarray(data["trip_user"])[valid]

    model.train()
    for name in ("gamma", "beta", "price_kappa"):            # certified price stays frozen
        getattr(model, name).requires_grad_(False)
    parameters = [q for q in fitted_parameters(model) if q.requires_grad] + [model.phi]
    model.phi.requires_grad_(True)
    optimizer = torch.optim.Adam(parameters, lr=args.lr)
    capacities = category_capacities(data, int(data["n_cat"]), nmax)

    start_valid = per_trip(model, batcher, valid, check_rule)
    start_gap = float(per_trip(model, batcher, valid, train_rule).mean() - start_valid.mean())
    best_valid, best_step, stale = start_valid.copy(), 0, 0
    history = [{"step": 0, "validation_check_rule": float(start_valid.mean()), "rule_gap": start_gap,
                "seconds": time.time() - started}]
    print(f"[joint-refine] staged start: validation {start_valid.mean():.5f} (level {rank + args.level_offset + 1}), "
          f"rule gap {start_gap:+.5f}, rank {rank}, {len(parameters)} tensors, price frozen", flush=True)

    def save(step, validation):
        summary = paired_score_summary(validation - start_valid, households)
        payload = dict(blob)
        payload["model"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
        payload["joint_refinement"] = {
            "parent_checkpoint": str(args.checkpoint), "steps": step, "lr": args.lr,
            "training_level": rank + args.level_offset, "check_level": rank + args.level_offset + 1,
            "price_component": "frozen certified", "phi_contract": "rank fixed, singular values <= 1",
            "validation_trips": int(len(valid)),
            "validation_gain": {"mean": summary["mean"], "95_interval": summary["95_interval"]},
            "accepted": bool(summary["95_interval"][0] > 0), "history": history}
        tmp = args.output.with_suffix(".tmp")
        torch.save(payload, tmp)
        os.replace(tmp, args.output)
        return summary

    install_quadrature(model, train_rule)
    for step in range(1, args.steps + 1):
        trips = train[rng.choice(len(train), size=args.batch, replace=False)]
        optimizer.zero_grad(set_to_none=True)
        loss = -log_likelihood(model, batcher, trips).mean()
        for product, context in ((model.mu, model.delta_c()), (model.zeta, model.xi_c()),
                                 (model.alpha, model.theta_c())):
            loss = loss + args.pool_prod * torch.trace((product.T @ product) @ (context.T @ context)) / (
                product.shape[0] * context.shape[0])
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, args.clip)
        optimizer.step()
        model.project_context_gauges()
        model.project_rho_c(-1.5)
        project_category_reward_(model, capacities, 1.5, optimizer=optimizer)
        project_phi(model, rank)
        if step % args.eval_every == 0:
            model.eval()
            v_train = per_trip(model, batcher, valid, train_rule)
            v_check = per_trip(model, batcher, valid, check_rule)
            install_quadrature(model, train_rule)
            model.train()
            gap = float(v_train.mean() - v_check.mean())
            history.append({"step": step, "validation_check_rule": float(v_check.mean()), "rule_gap": gap,
                            "seconds": time.time() - started})
            line = (f"[joint-refine] step {step}: validation {v_check.mean():.5f} "
                    f"(gain vs staged {v_check.mean() - start_valid.mean():+.5f}), rule gap {gap:+.5f}, "
                    f"{time.time() - started:.0f}s elapsed")
            if v_check.mean() > best_valid.mean() + 1e-4:
                best_valid, best_step, stale = v_check, step, 0
                s = save(step, v_check)
                line += f"; saved (paired gain {s['mean']:+.5f}, 95% {s['95_interval'][0]:+.5f}..{s['95_interval'][1]:+.5f})"
            else:
                stale += 1
            print(line, flush=True)
            if stale >= args.patience:
                break
    if best_step == 0:
        print("[joint-refine] no improvement over the staged fit; nothing written", flush=True)
    else:
        record = torch.load(args.output, weights_only=False)["joint_refinement"]
        print(f"[joint-refine] best step {best_step}; validation gain {record['validation_gain']}; "
              f"accepted={record['accepted']}", flush=True)
    print(f"[joint-refine] finished in {time.time() - started:.0f}s", flush=True)


if __name__ == "__main__":
    main()
