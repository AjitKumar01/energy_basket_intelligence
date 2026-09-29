#!/usr/bin/env python3
"""Joint refinement stage: damped iterated Monte Carlo likelihood from the staged Version-4 fit.

Starting from the staged final checkpoint (additive stage, then Phi in a fixed spectral basis),
each round draws a bank of baskets from the current law with Model A's blocked Gibbs sampler,
solves the fixed-bank likelihood for the linear blocks -- interaction C in the fixed basis, item
intercepts, rho_c, the size potential, household taste (then product taste) -- under the smallest
trust region that keeps the bank's effective sample size above target, and makes the solution the
next parent.  See joint_refinement.py for the estimator.

Frozen: the certified price component, promotion, season and store blocks, and Phi's basis.
Kept from Model A: its energy, its Phi contract (0 <= C <= I), its rho_c clamps and its pooling
term.  Accepted only if the paired validation log likelihood (Model A's quadrature at the
pipeline's evaluation level) improves with a positive 95% lower bound AND the refined model passes
the one-level-finer numerical audit; the output checkpoint records the decision.

  --mixing-check   only draw four independent banks for a sample of contexts and compare them
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

os.environ.setdefault("V3_AFFINITY", "1")

import numpy as np
import torch

from category_safety import category_capacities, project_category_reward_
from checkpoint_io import load_checkpoint
from data import build
from features import Features
from fit import Batcher
from joint_refinement import (SlotBaskets, baskets_from_items, damped_round, draw_bank_fast,
                              orthonormal_basis, set_phi_from, slot_table, taste_utility)
from pipeline_support import install_quadrature, smolyak_rule, supported_trips
from uncertainty import paired_score_summary

torch.set_default_dtype(torch.float64)
REFINED = ("lam", "theta", "alpha", "rho_c", "rho_0_free")


def choose_contexts(data, nmax, per_household, maximum, rng):
    train = supported_trips(data, 0, nmax)
    household = np.asarray(data["trip_user"])[train]
    chosen = []
    for h in np.unique(household):
        trips = train[household == h]
        if len(trips) > per_household:
            trips = rng.choice(trips, size=per_household, replace=False)
        chosen.extend(trips.tolist())
    chosen = np.sort(np.asarray(chosen))
    if maximum and len(chosen) > maximum:
        chosen = np.sort(rng.choice(chosen, size=maximum, replace=False))
    return chosen


def observed_items_of(data, trips):
    ptr = data["line_ptr"]
    return [np.unique(data["line_item"][ptr[t]:ptr[t + 1]]) for t in trips]


def draw_all(model, batcher, trips, table, chains, draws_per_chain, burn, batch, generator, full,
             observed_items=None, baskets_per_z=1):
    """Bank for every context, drawn in batches of contexts (each repeated once per chain).

    Chains start at z ~ N(sum of phi over the context's observed basket, I) when observed_items is
    given (a warm start), and each recorded z yields `baskets_per_z` exact baskets.
    """
    slots, basket, context = [], [], []
    offset = 0
    for c0 in range(0, len(trips), batch):
        c1 = min(c0 + batch, len(trips))
        ix_rep, ctx_rep, _lc, house_rep, *_ = batcher.make(np.repeat(trips[c0:c1], chains))
        model.house, model.ctx = house_rep, ctx_rep
        base = np.repeat(np.arange(c0, c1), chains)
        warm = None if observed_items is None else [observed_items[c] for c in base]
        part = draw_bank_fast(model, ix_rep, base, table, draws_per_chain, burn, generator,
                              init_items=warm, baskets_per_z=baskets_per_z)
        slots.append(part.slots); basket.append(part.basket + offset); context.append(part.context)
        offset += part.n
    model.house, model.ctx = full
    return SlotBaskets(torch.cat(slots), torch.cat(basket), torch.cat(context))


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True, help="staged final checkpoint")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--contexts-per-household", type=int, default=10)
    p.add_argument("--max-contexts", type=int, default=0, help="0 = no cap beyond per-household")
    p.add_argument("--chains", type=int, default=16)
    p.add_argument("--draws-per-chain", type=int, default=2)
    p.add_argument("--burn", type=int, default=3)
    p.add_argument("--baskets-per-z", type=int, default=2)
    p.add_argument("--batch-contexts", type=int, default=128)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--round-tolerance", type=float, default=1e-4)
    p.add_argument("--trust-ladder", type=float, nargs="+", default=[1, 10, 100, 1e3, 1e4, 1e5])
    p.add_argument("--ess-q05", type=float, default=0.2, help="5th percentile of per-context ESS")
    p.add_argument("--ess-median", type=float, default=0.5)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--pool-prod", type=float, default=1.45)
    p.add_argument("--cap", type=float, default=1.0, help="Phi contract: C <= cap * I")
    p.add_argument("--validation-trips", type=int, default=1024)
    p.add_argument("--level-offset", type=int, default=2, help="evaluation level = rank + offset")
    p.add_argument("--mixing-check", action="store_true")
    p.add_argument("--seed", type=int, default=41001)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    started = time.time()
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    data = build()
    model, blob, meta = load_checkpoint(args.checkpoint, data)
    rank, nmax, J = int(blob["active_rank"]), int(meta["nmax"]), int(data["n_item"])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in REFINED)
    features = Features(J, int(data["n_store"]), include_recency=False)
    batcher = Batcher(data, features, nmax, include_recency=False)
    trips = choose_contexts(data, nmax, args.contexts_per_household, args.max_contexts, rng)
    ix, ctx, _lc, house, *_ = batcher.make(trips)
    full = (house, ctx)
    model.house, model.ctx = full
    table = slot_table(ix, J)
    observed_items = observed_items_of(data, trips)
    observed = baskets_from_items(table, list(range(len(trips))), observed_items)
    draws = args.chains * args.draws_per_chain * args.baskets_per_z
    households = np.asarray(data["trip_user"])[trips]
    print(f"[joint-refinement] {len(trips)} contexts from {len(np.unique(households))} households "
          f"(<= {args.contexts_per_household} each), {draws} draws per context, rank {rank}", flush=True)

    # the frozen/refined utility split must reproduce b_flat exactly under a parameter move
    with torch.no_grad():
        frozen = model.b_flat(ix) - taste_utility(model, ix)
        saved = model.lam.clone(); model.lam.add_(0.01 * torch.randn(J, generator=generator))
        split_error = float((model.b_flat(ix) - frozen - taste_utility(model, ix)).abs().max())
        model.lam.copy_(saved)
    if split_error > 1e-10:
        raise RuntimeError(f"utility split does not reproduce b_flat (max error {split_error:.2e})")

    if args.mixing_check:
        sample = np.sort(rng.choice(len(trips), size=min(200, len(trips)), replace=False))
        t0 = time.time()
        banks = [draw_all(model, batcher, trips[sample], slot_table(batcher.make(trips[sample])[0], J),
                          args.chains, args.draws_per_chain, args.burn, args.batch_contexts,
                          torch.Generator().manual_seed(args.seed + k), full,
                          [observed_items[i] for i in sample], args.baskets_per_z) for k in range(4)]
        bank_seconds = (time.time() - t0) / 4
        sub_ix = batcher.make(trips[sample])[0]
        sizes = [torch.bincount(b.basket, minlength=b.n).double().mean().item() for b in banks]
        incidence = []
        for b in banks:
            inc = torch.zeros(J).index_add(0, sub_ix.item[b.slots], torch.ones(b.slots.numel(), dtype=torch.float64))
            incidence.append(inc / b.n)
        inc = torch.stack(incidence)
        top = torch.argsort(inc.mean(0), descending=True)[:20]
        spread = (inc[:, top].max(0).values - inc[:, top].min(0).values)
        se = (inc[:, top].mean(0) * (1 - inc[:, top].mean(0)) / banks[0].n).sqrt()
        report = {"contexts": int(len(sample)), "draws_per_context": draws,
                  "mean_basket_size_per_bank": sizes,
                  "top20_incidence_spread_over_monte_carlo_se": {
                      "max": float((spread / se).max()), "median": float((spread / se).median())},
                  "observed_mean_size": float(np.mean(np.diff(data["line_ptr"])[trips[sample]])),
                  "seconds_per_bank": bank_seconds,
                  "seconds_per_context_per_bank": bank_seconds / len(sample),
                  "projected_seconds_per_round_bank": bank_seconds / len(sample) * len(trips)}
        print(json.dumps(report, indent=2), flush=True)
        return

    valid_all = supported_trips(data, 1, nmax)
    valid = np.sort(valid_all[rng.permutation(len(valid_all))[:args.validation_trips]])
    valid_households = np.asarray(data["trip_user"])[valid]

    def validation(level):
        install_quadrature(model, smolyak_rule(model, rank, level))
        out = []
        with torch.no_grad():
            for s in range(0, len(valid), 128):
                vix, vctx, vline, vhouse, li, lt, lc, _lq = batcher.make(valid[s:s + 128])
                model.house, model.ctx = vhouse, vctx
                out.append(model.energy(li, lt, lc, vix.B, vline) - model.log_Z(vix, drop_empty=True))
        model.house, model.ctx = full
        model.quad = None
        return torch.cat(out).numpy()

    level = rank + args.level_offset
    start_valid = validation(level)
    best_valid, best_round, best_state = start_valid, 0, None
    U, C = orthonormal_basis(model.phi.detach(), rank)
    capacities = category_capacities(data, int(data["n_cat"]), nmax)

    def ess_rule(ess):
        q05, med = float(ess.quantile(0.05)), float(ess.median())
        return q05 >= args.ess_q05 and med >= args.ess_median, {"q05": round(q05, 3), "median": round(med, 3)}

    rounds = []
    print(f"[joint-refinement] staged start: validation {start_valid.mean():.5f} at level {level} "
          f"({time.time() - started:.0f}s)", flush=True)
    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        bank = draw_all(model, batcher, trips, table, args.chains, args.draws_per_chain, args.burn,
                        args.batch_contexts, generator, full, observed_items, args.baskets_per_z)
        draw_seconds = time.time() - t0
        ok, record = damped_round(model, ix, observed, bank, draws, U, C, rank, args.trust_ladder, ess_rule,
                                  args.cycles, args.pool_prod, args.cap,
                                  log=lambda m: print(m, flush=True))
        if not ok:
            rounds.append({"round": rnd, **record})
            print(f"[joint-refinement] round {rnd}: {record['status']}; stopping", flush=True)
            break
        C = set_phi_from(model, U, record["C"], rank, args.cap)
        model.project_rho_c(-1.5)
        project_category_reward_(model, capacities, 1.5)
        model.project_context_gauges()
        v = validation(level)
        entry = {"round": rnd, "trust": record["trust"], "bank_gain": record["bank_gain"], "ess": record["ess"],
                 "validation": float(v.mean()), "draw_seconds": draw_seconds,
                 "round_seconds": time.time() - t0}
        rounds.append(entry)
        print(f"[joint-refinement] round {rnd}: validation {v.mean():.5f} "
              f"(gain vs staged {v.mean() - start_valid.mean():+.5f}), bank gain {record['bank_gain']:+.5f}, "
              f"trust {record['trust']:g}, ESS {record['ess']}, {entry['round_seconds']:.0f}s", flush=True)
        if v.mean() > best_valid.mean() + 1e-5:
            best_valid, best_round = v, rnd
            best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            payload = dict(blob)
            payload["model"] = best_state
            payload["joint_refinement"] = {"round": rnd, "rounds": rounds, "accepted": None}
            torch.save(payload, args.output.with_suffix(".tmp")); os.replace(args.output.with_suffix(".tmp"), args.output)
        if record["bank_gain"] < args.round_tolerance:
            break

    decision = {"accepted": False, "reason": "no round improved validation"}
    if best_state is not None:
        model.load_state_dict(best_state)
        gain = paired_score_summary(best_valid - start_valid, valid_households)
        audit = validation(level + 1)
        audit_gap = float(abs(audit.mean() - best_valid.mean()))
        accepted = bool(gain["95_interval"][0] > 0 and audit_gap <= 0.01)
        decision = {"accepted": accepted, "best_round": best_round,
                    "validation_gain": {"mean": gain["mean"], "95_interval": gain["95_interval"]},
                    "audit_level": level + 1, "audit_gap": audit_gap,
                    "reason": "positive paired gain and numerical audit passed" if accepted else
                    "gain interval or numerical audit failed"}
        payload = dict(blob)
        payload["model"] = best_state
        payload["joint_refinement"] = {"decision": decision, "rounds": rounds, "contexts": int(len(trips)),
                                       "draws_per_context": draws, "evaluation_level": level,
                                       "frozen": "price, promotion, season, store blocks; Phi basis"}
        torch.save(payload, args.output.with_suffix(".tmp")); os.replace(args.output.with_suffix(".tmp"), args.output)
    print(f"[joint-refinement] decision {json.dumps(decision)} ({time.time() - started:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
