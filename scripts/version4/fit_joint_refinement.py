#!/usr/bin/env python3
"""Joint refinement stage: iterated Monte Carlo maximum likelihood from the staged Version-4 fit.

Starting from the staged final checkpoint, each round draws a bank of baskets per training
context from the current law (blocked Gibbs with an exact S | z sampler), solves the fixed-bank
likelihood for the linear blocks -- interaction C in the fixed basis U, item intercepts, rho_c,
the size potential, household taste and product taste -- for each candidate trust weight, and
keeps the candidate with the best exact log likelihood on held-out selection trips.  Rounds stop
when the selection score stops improving.  See joint_refinement.py and
docs/JOINT_REFINEMENT_STAGE.md.

Frozen: the certified price component, promotion, season and store blocks, and Phi's basis.
Kept from the staged model: its energy, its Phi contract (0 <= C <= cap), its rho_c clamps and its
pooling term.  Rounds are tried in order of held-out validation; the first with a positive
paired 95% gain interval that passes the one-level-finer numerical audit and (with --size-gate)
the certification population-size audit is written to --output.  Otherwise no output is written
and the staged model stays.  --report always records the decision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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
from joint_refinement import (SlotBaskets, bank_design, baskets_from_items, concat_designs, damped_round,
                              draw_bank, orthonormal_basis, set_phi_from, slot_table, taste_utility)
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
             observed_items, baskets_per_z):
    """Bank for every context, drawn in batches of contexts (each repeated once per chain).

    Chains start at z ~ N(sum of phi over the context's observed basket, I) (a warm start), and
    each recorded z yields `baskets_per_z` exact baskets.
    """
    slots, basket, context = [], [], []
    offset = 0
    for c0 in range(0, len(trips), batch):
        c1 = min(c0 + batch, len(trips))
        ix_rep, ctx_rep, _lc, house_rep, *_ = batcher.make(np.repeat(trips[c0:c1], chains))
        model.house, model.ctx = house_rep, ctx_rep
        base = np.repeat(np.arange(c0, c1), chains)
        part = draw_bank(model, ix_rep, base, table, draws_per_chain, burn, generator,
                         init_items=[observed_items[c] for c in base], baskets_per_z=baskets_per_z)
        slots.append(part.slots); basket.append(part.basket + offset); context.append(part.context)
        offset += part.n
    model.house, model.ctx = full
    return SlotBaskets(torch.cat(slots), torch.cat(basket), torch.cat(context))


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--checkpoint", type=Path, required=True, help="staged final checkpoint")
    p.add_argument("--output", type=Path, required=True, help="refined checkpoint (written only if accepted)")
    p.add_argument("--report", type=Path, help="JSON decision report (always written)")
    p.add_argument("--contexts-per-household", type=int, default=10)
    p.add_argument("--max-contexts", type=int, default=0, help="0 = no cap beyond per-household")
    p.add_argument("--chains", type=int, default=4)
    p.add_argument("--draws-per-chain", type=int, default=4)
    p.add_argument("--burn", type=int, default=3)
    p.add_argument("--baskets-per-z", type=int, default=2)
    p.add_argument("--batch-contexts", type=int, default=128)
    p.add_argument("--chunk-contexts", type=int, default=1024,
                   help="contexts whose assortments are held in memory at once (multiple of --batch-contexts)")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--round-tolerance", type=float, default=1e-4,
                   help="minimum selection-score improvement that continues the rounds")
    p.add_argument("--select-trust", type=float, nargs="+", default=[300, 1000],
                   help="candidate trust weights, chosen each round on held-out selection trips")
    p.add_argument("--select-fallback", type=float, nargs="*", default=[1e4, 1e5],
                   help="smaller steps tried in order only when no candidate passes the ESS rule")
    p.add_argument("--ess-q05", type=float, default=0.2, help="5th percentile of per-context ESS")
    p.add_argument("--ess-median", type=float, default=0.5)
    p.add_argument("--cycles", type=int, default=2)
    p.add_argument("--pool-prod", type=float, default=1.45)
    p.add_argument("--cap", type=float, default=1.0, help="Phi contract: C <= cap * I")
    p.add_argument("--validation-trips", type=int, default=1024)
    p.add_argument("--selection-trips", type=int, default=1024)
    p.add_argument("--level-offset", type=int, default=2, help="evaluation level = rank + offset")
    p.add_argument("--node-item-trips", type=int, default=16_000_000,
                   help="quadrature nodes x products x trips per validation batch (memory-bound above this)")
    p.add_argument("--size-gate", action="store_true",
                   help="accept a round only if it also passes audit_population_size.py (the certification gate)")
    p.add_argument("--size-gate-contexts", type=int, default=0, help="0 = every training context")
    p.add_argument("--size-gate-confirm-contexts", type=int, default=2048)
    p.add_argument("--size-gate-calibration-contexts", type=int, default=2048)
    p.add_argument("--size-gate-chunk", type=int, default=48)
    p.add_argument("--seed", type=int, default=41001)
    p.add_argument("--threads", type=int, default=8)
    args = p.parse_args()
    if args.chunk_contexts % args.batch_contexts:
        p.error("--chunk-contexts must be a multiple of --batch-contexts")
    torch.set_num_threads(args.threads)
    started = time.time()
    for stale in (args.output, args.output.with_suffix(".tmp")):
        stale.unlink(missing_ok=True)          # this run owns its output; never report an old file
    rng = np.random.default_rng(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    data = build()
    model, blob, meta = load_checkpoint(args.checkpoint, data)
    rank, nmax, J = int(blob["active_rank"]), int(meta["nmax"]), int(data["n_item"])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name in REFINED)
    batcher = Batcher(data, Features(J, int(data["n_store"]), include_recency=False), nmax,
                      include_recency=False)
    trips = choose_contexts(data, nmax, args.contexts_per_household, args.max_contexts, rng)
    observed_items = observed_items_of(data, trips)
    chunks = [(a, min(a + args.chunk_contexts, len(trips))) for a in range(0, len(trips), args.chunk_contexts)]

    def chunk_index(a, b):
        """Assortment index of contexts a..b; installs their household/context arrays."""
        ixc, ctxc, _lc, housec, *_ = batcher.make(trips[a:b])
        model.house, model.ctx = housec, ctxc
        return ixc, slot_table(ixc, J), (housec, ctxc)

    draws = args.chains * args.draws_per_chain * args.baskets_per_z
    households = np.asarray(data["trip_user"])[trips]
    print(f"[joint-refinement] {len(trips)} contexts from {len(np.unique(households))} households "
          f"(<= {args.contexts_per_household} each), {draws} draws per context, rank {rank}", flush=True)

    # the frozen/refined utility split must reproduce b_flat exactly under a parameter move
    ix, table, _ = chunk_index(*chunks[0])
    with torch.no_grad():
        frozen = model.b_flat(ix) - taste_utility(model, ix)
        saved = model.lam.clone(); model.lam.add_(0.01 * torch.randn(J, generator=generator))
        split_error = float((model.b_flat(ix) - frozen - taste_utility(model, ix)).abs().max())
        model.lam.copy_(saved)
    del ix, table, frozen
    if split_error > 1e-10:
        raise RuntimeError(f"utility split does not reproduce b_flat (max error {split_error:.2e})")

    valid_all = supported_trips(data, 1, nmax)
    valid = np.sort(valid_all[rng.permutation(len(valid_all))[:args.validation_trips]])
    valid_households = np.asarray(data["trip_user"])[valid]
    rest = np.setdiff1d(valid_all, valid)
    select = np.sort(rest[rng.permutation(len(rest))[:args.selection_trips]])

    def validation(level, subset):
        rule = smolyak_rule(model, rank, level)
        install_quadrature(model, rule)
        batch = max(1, min(128, args.node_item_trips // (len(rule[1]) * J)))   # bound nodes x items x trips
        out = []
        with torch.no_grad():
            for s in range(0, len(subset), batch):
                vix, vctx, vline, vhouse, li, lt, lc, _lq = batcher.make(subset[s:s + batch])
                model.house, model.ctx = vhouse, vctx
                out.append(model.energy(li, lt, lc, vix.B, vline) - model.log_Z(vix, drop_empty=True))
        model.quad = None
        return torch.cat(out).numpy()

    level = rank + args.level_offset
    start_valid = validation(level, valid)
    select_score = float(validation(level, select).mean())
    round_states = []                     # (round, validation per trip, state) for every completed round
    U, C = orthonormal_basis(model.phi.detach(), rank)
    t0 = time.time()
    parts = []
    for a, b in chunks:
        ixc, tablec, _ = chunk_index(a, b)
        with torch.no_grad():
            frozen = model.b_flat(ixc) - taste_utility(model, ixc)
            observed_c = baskets_from_items(tablec, list(range(b - a)), observed_items[a:b])
            parts.append(bank_design(model, ixc, observed_c, U, frozen))
    obs_design = concat_designs(parts)
    del parts, ixc, tablec, frozen
    print(f"[joint-refinement] observed design for {len(trips)} contexts in {len(chunks)} chunks "
          f"({time.time() - t0:.0f}s)", flush=True)

    def bank_designs_for_round():
        """Draw the bank chunk by chunk and keep only its per-basket statistics."""
        parts = []
        for a, b in chunks:
            ixc, tablec, chunk_full = chunk_index(a, b)
            bank_c = draw_all(model, batcher, trips[a:b], tablec, args.chains, args.draws_per_chain, args.burn,
                              args.batch_contexts, generator, chunk_full, observed_items[a:b], args.baskets_per_z)
            with torch.no_grad():
                model.house, model.ctx = chunk_full
                frozen = model.b_flat(ixc) - taste_utility(model, ixc)
                parts.append(bank_design(model, ixc, bank_c, U, frozen))
        return concat_designs(parts)

    capacities = category_capacities(data, int(data["n_cat"]), nmax)

    def ess_rule(ess):
        q05, med = float(ess.quantile(0.05)), float(ess.median())
        return q05 >= args.ess_q05 and med >= args.ess_median, {"q05": round(q05, 3), "median": round(med, 3)}

    def step(bank_design_, trust):
        """Solve one candidate step from the current parent, then apply the staged model's projections."""
        ok, record = damped_round(model, obs_design, bank_design_, draws, U, C, rank, [trust], ess_rule,
                                  args.cycles, args.pool_prod, args.cap, log=lambda m: print(m, flush=True))
        if not ok:
            return ok, record, C
        C_new = set_phi_from(model, U, record["C"], rank, args.cap)
        model.project_rho_c(-1.5)
        project_category_reward_(model, capacities, 1.5)
        model.project_context_gauges()
        return ok, record, C_new

    rounds = []
    print(f"[joint-refinement] staged start: validation {start_valid.mean():.5f} at level {level} "
          f"({time.time() - started:.0f}s)", flush=True)
    for rnd in range(1, args.rounds + 1):
        t0 = time.time()
        bank_design_ = bank_designs_for_round()
        draw_seconds = time.time() - t0
        t1 = time.time()
        parent_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
        best = None
        # normal candidates first; fallback levels (smaller steps) are tried in order only while
        # no candidate has passed the ESS rule, so a staged fit far from the optimum still moves
        levels = list(args.select_trust) + list(args.select_fallback)
        for i, trust in enumerate(levels):
            if i >= len(args.select_trust) and best is not None:
                break
            model.load_state_dict(parent_state)
            ok_c, rec_c, C_c = step(bank_design_, trust)
            if not ok_c:
                print(f"[joint-refinement]   candidate trust {trust:g}: ESS rule failed", flush=True)
                continue
            score = float(validation(level, select).mean())
            print(f"[joint-refinement]   candidate trust {trust:g}: selection {score:.5f} "
                  f"({score - select_score:+.5f} vs parent)", flush=True)
            if best is None or score > best[0]:
                best = (score, {k: t.detach().clone() for k, t in model.state_dict().items()}, C_c,
                        {**rec_c, "selection": score})
        model.load_state_dict(parent_state)
        if best is None or best[0] <= select_score + args.round_tolerance:
            rounds.append({"round": rnd, "status": "no candidate improved the selection trips"})
            print(f"[joint-refinement] round {rnd}: no candidate improved the selection trips; stopping", flush=True)
            break
        select_score = best[0]
        model.load_state_dict(best[1])
        C, record = best[2], best[3]
        solve_seconds = time.time() - t1
        t2 = time.time()
        v = validation(level, valid)
        entry = {"round": rnd, "trust": record["trust"], "selection": record["selection"],
                 "bank_gain": record["bank_gain"], "ess": record["ess"], "validation": float(v.mean()),
                 "draw_seconds": draw_seconds, "solve_seconds": solve_seconds,
                 "validation_seconds": time.time() - t2, "round_seconds": time.time() - t0}
        rounds.append(entry)
        print(f"[joint-refinement] round {rnd}: validation {v.mean():.5f} "
              f"(gain vs staged {v.mean() - start_valid.mean():+.5f}), bank gain {record['bank_gain']:+.5f}, "
              f"trust {record['trust']:g}, ESS {record['ess']}, bank {draw_seconds:.0f}s"
              f", solve {solve_seconds:.0f}s, validation {entry['validation_seconds']:.0f}s, "
              f"{entry['round_seconds']:.0f}s", flush=True)
        round_states.append((rnd, v, {k: t.detach().clone() for k, t in model.state_dict().items()}))

    # Acceptance: rounds are tried in order of held-out validation; the first that has a positive
    # paired gain interval, passes the numerical audit one level finer and (with --size-gate) the
    # pipeline's population-size safety audit is the output.  None passing keeps the staged model.
    def save(state, path, decision):
        payload = dict(blob)
        payload["model"] = state
        payload["joint_refinement"] = {"decision": decision, "rounds": rounds, "contexts": int(len(trips)),
                                       "draws_per_context": draws, "evaluation_level": level,
                                       "frozen": "price, promotion, season, store blocks; Phi basis"}
        torch.save(payload, path.with_suffix(".tmp")); os.replace(path.with_suffix(".tmp"), path)

    def size_gate(state, rnd, attempt):
        path = args.output.with_name(f"{args.output.stem}_round{rnd}.pt")
        save(state, path, attempt)
        report_path = (args.report or args.output).with_name(
            f"{args.output.stem}_round{rnd}_population_size.json")
        command = [sys.executable, "-u", str(Path(__file__).with_name("audit_population_size.py")),
                   "--checkpoint", str(path), "--rank", str(rank), "--screen-level", str(level - 1),
                   "--confirm-level", str(level), "--contexts", str(args.size_gate_contexts),
                   "--confirm-contexts", str(args.size_gate_confirm_contexts),
                   "--calibration-contexts", str(args.size_gate_calibration_contexts),
                   "--chunk", str(args.size_gate_chunk), "--threads", str(args.threads),
                   "--output", str(report_path)]
        code = subprocess.run(command, check=False).returncode
        try:
            audit_report = json.loads(report_path.read_text())
            passed, gates = bool(audit_report.get("passed")), audit_report.get("gates")
        except Exception:
            passed, gates = False, None
        path.unlink(missing_ok=True)
        return passed, {"exit_code": code, "passed": passed, "gates": gates, "report": str(report_path)}

    decision = {"accepted": False, "reason": "no round improved validation", "attempts": []}
    ordered = sorted((r for r in round_states if r[1].mean() > start_valid.mean() + 1e-5),
                     key=lambda r: -float(r[1].mean()))
    for rnd, v, state in ordered:
        model.load_state_dict(state)
        gain = paired_score_summary(v - start_valid, valid_households)
        attempt = {"round": rnd, "validation_gain": {"mean": gain["mean"], "95_interval": gain["95_interval"]}}
        if not gain["95_interval"][0] > 0:
            decision["attempts"].append({**attempt, "result": "gain interval includes zero"})
            continue
        audit_gap = float(abs(validation(level + 1, valid).mean() - v.mean()))
        attempt.update(audit_level=level + 1, audit_gap=audit_gap)
        if audit_gap > 0.01:
            decision["attempts"].append({**attempt, "result": "numerical audit failed"})
            continue
        if args.size_gate:
            passed, size = size_gate(state, rnd, attempt)
            attempt["population_size"] = size
            print(f"[joint-refinement] round {rnd}: population-size gate {'passed' if passed else 'FAILED'} "
                  f"{size['gates']}", flush=True)
            if not passed:
                decision["attempts"].append({**attempt, "result": "population-size gate failed"})
                continue
        decision = {**attempt, "accepted": True, "best_round": rnd,
                    "attempts": decision["attempts"] + [{**attempt, "result": "accepted"}],
                    "reason": "positive paired gain, numerical audit"
                              + (" and population-size gate" if args.size_gate else "") + " passed"}
        save(state, args.output, decision)
        break
    else:
        if ordered:
            decision["reason"] = "no improving round passed every acceptance gate"
    print(f"[joint-refinement] decision {json.dumps(decision)} ({time.time() - started:.0f}s)", flush=True)
    if args.report is not None:
        report = {"decision": decision, "rounds": rounds, "contexts": int(len(trips)),
                  "draws_per_context": draws, "evaluation_level": level,
                  "parent": str(args.checkpoint), "parent_sha256": file_digest(args.checkpoint),
                  "output": str(args.output) if args.output.is_file() else None,
                  "output_sha256": file_digest(args.output) if args.output.is_file() else None,
                  "seconds": time.time() - started}
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2, default=float))


if __name__ == "__main__":
    main()
