#!/usr/bin/env python3
"""Compare fitted conditional-basket price responses with observational event contrasts."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch

os.environ.setdefault("V3_AFFINITY", "1")

from checkpoint_io import load_checkpoint
from data import build
from features import Features
from fit import Batcher
from interaction_particles import (differentiable_logz_beta0, direct_interaction_particles,
                                   rao_blackwell_selected_incidence,
                                   reweight_additive_counterfactual)
from price_response import changed_price_context
from provenance import file_sha256, strict_json_dumps
from uncertainty import paired_score_summary


torch.set_default_dtype(torch.float64)


def basket_contains(data, trips, item):
    return np.asarray([
        item in data["line_item"][int(data["line_ptr"][trip]):int(data["line_ptr"][trip+1])]
        for trip in trips], dtype=float)


def exact_parent_incidence(model, ix, item):
    with torch.enable_grad():
        utility = model.b_flat(ix).detach().requires_grad_(True)
        logz = differentiable_logz_beta0(model, ix, utility)
        # Contexts are independent conditional laws.  Summing their log normalizers
        # gives each context's own incidence gradient; logsumexp would instead create
        # an artificial softmax weighting across unrelated trips.
        incidence = torch.autograd.grad(logz.sum(), utility)[0]
    slots = [torch.nonzero((ix.item_trip == b) & (ix.item == item), as_tuple=True)[0][0]
             for b in range(ix.B)]
    return incidence[torch.as_tensor(slots)].detach()


@torch.no_grad()
def event_prediction(parent, child, batcher, data, trips, item, log_change, particles, seed):
    started = time.monotonic()
    ix, ctx, _lc, house, *_ = batcher.make(trips)
    parent.house, parent.ctx = house, ctx
    old_parent = exact_parent_incidence(parent, ix, item)
    parent_factual_seconds = time.monotonic() - started
    change = torch.zeros_like(ix.item, dtype=torch.float64)
    chosen_slots = []
    for b in range(ix.B):
        slot = torch.nonzero((ix.item_trip == b) & (ix.item == item), as_tuple=True)[0]
        if not len(slot): raise RuntimeError("event item absent from declared assortment")
        change[int(slot[0])] = log_change; chosen_slots.append(int(slot[0]))
    parent.ctx = changed_price_context(ctx, ix.item_trip, change)
    new_parent = exact_parent_incidence(parent, ix, item)
    parent_counterfactual_seconds = time.monotonic() - started - parent_factual_seconds
    parent.ctx = ctx

    child.house, child.ctx = house, ctx
    factual_b = child.b_flat(ix).clone()
    bank = direct_interaction_particles(
        child, ix, particles, torch.Generator().manual_seed(seed))
    child_bank_seconds = (time.monotonic() - started - parent_factual_seconds
                          - parent_counterfactual_seconds)
    selected_slots_tensor=torch.as_tensor(chosen_slots,dtype=torch.long)
    factual = rao_blackwell_selected_incidence(
        child,ix,bank.states,selected_slots_tensor,bank.log_weights)
    child.ctx = changed_price_context(ctx, ix.item_trip, change)
    target_b = child.b_flat(ix).clone()
    target = reweight_additive_counterfactual(child, ix, bank, factual_b, target_b)
    counterfactual = rao_blackwell_selected_incidence(
        child,ix,bank.states,selected_slots_tensor,target.log_weights)
    child_statistics_seconds = (time.monotonic() - started - parent_factual_seconds
                                - parent_counterfactual_seconds - child_bank_seconds)
    child.ctx = ctx
    return {
        "parent": float((new_parent-old_parent).mean()),
        "child": float((counterfactual-factual).mean()),
        "minimum_ess_fraction": float(target.ess_fraction.min()),
        "minimum_absolute_ess": float(target.ess_fraction.min()) * particles,
        "runtime_seconds": {
            "parent_factual": parent_factual_seconds,
            "parent_counterfactual": parent_counterfactual_seconds,
            "child_particle_bank": child_bank_seconds,
            "child_statistics_and_reweighting": child_statistics_seconds,
            "total": time.monotonic() - started,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-event-output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args(); started=time.monotonic(); torch.set_num_threads(args.threads)
    protocol_path=args.protocol.resolve(); protocol=json.loads(protocol_path.read_text())
    spec=protocol["observational_price"]; run_dir=args.run_dir.resolve()
    if int(spec["particles"]) < 16:
        raise ValueError("observational particles must be at least the absolute ESS gate of 16")
    event=pd.read_parquet(args.events.resolve())
    data=build(); parent, pblob, meta=load_checkpoint(
        run_dir/"out/v3_pipeline_additive_best.pt", data,
        required_capabilities=("conditional_nonempty_incidence",))
    child, cblob, _=load_checkpoint(
        run_dir/"artifacts/candidate_rank1.pt", data,
        required_capabilities=("conditional_nonempty_incidence","gram_interactions"))
    for model in (parent,child):
        for parameter in model.parameters(): parameter.requires_grad_(False)
    clean_rule=(~event.concurrent_promotion_change)&(event.weeks_since_previous_observation==1)
    training_support=event[(event.event_split=="train")&clean_rule]
    if training_support.empty:
        raise RuntimeError("no clean training price events define action support")
    support_low=float(training_support.log_price_change.min())
    support_high=float(training_support.log_price_change.max())
    product_support=(training_support.groupby("item_id").log_price_change
                     .agg(product_support_low="min", product_support_high="max"))
    candidate=event[(event.event_split==spec["split"])
        & (~event.concurrent_promotion_change)
        & (event.weeks_since_previous_observation==1)].copy()
    candidate=candidate.join(product_support,on="item_id")
    supported_depth=(candidate.product_support_low.notna()
                     &(candidate.log_price_change>=candidate.product_support_low)
                     &(candidate.log_price_change<=candidate.product_support_high))
    unsupported_depth=int((~supported_depth).sum())
    eligible=candidate[supported_depth].copy()
    items=pd.read_parquet("basket_input/items.parquet").set_index("item_id")
    eligible=eligible[eligible.item_id.map(items.n_train_lines)>=spec["minimum_training_lines_per_product"]]
    rng=np.random.default_rng(int(spec["seed"])); eligible=eligible.iloc[rng.permutation(len(eligible))]
    batcher=Batcher(data,Features(int(data["n_item"]),int(data["n_store"]),
                                  include_recency=False),int(meta["nmax"]),include_recency=False)
    rows=[]; excluded={"unsupported_training_price_depth":unsupported_depth,
                       "insufficient_contexts":0,"low_ess":0}
    for candidate_number, record in enumerate(eligible.itertuples(index=False), start=1):
        item=int(record.item_id); prior_week=int(record.previous_week); current_week=int(record.WEEK_NO)
        supported=(data["trip_nlines"]>=1)&(data["trip_nlines"]<=int(meta["nmax"]))
        available=data["item_slot"][data["trip_store"],item]>=0
        before=np.flatnonzero(supported&available&(data["trip_week"]==prior_week))
        after=np.flatnonzero(supported&available&(data["trip_week"]==current_week))
        if min(len(before),len(after))<int(spec["minimum_contexts_per_side"]):
            excluded["insufficient_contexts"]+=1; continue
        model_panel=before[rng.permutation(len(before))[:min(int(spec["contexts_per_event"]),len(before))]]
        prediction=event_prediction(parent,child,batcher,data,model_panel,item,
            float(record.log_price_change),int(spec["particles"]),
            int(spec["seed"])+100003*int(record.event_id))
        if candidate_number % 100 == 0:
            print(f"[observational-price] candidates={candidate_number} accepted={len(rows)} "
                  f"last_model_seconds={prediction['runtime_seconds']['total']:.1f} "
                  f"last_ess={prediction['minimum_absolute_ess']:.1f}", flush=True)
        if prediction["minimum_ess_fraction"]<.2 or prediction["minimum_absolute_ess"]<16:
            excluded["low_ess"]+=1; continue
        before_y=basket_contains(data,before,item); after_y=basket_contains(data,after,item)
        rows.append({"event_id":int(record.event_id),"item_id":item,"PRODUCT_ID":int(record.PRODUCT_ID),
            "previous_week":prior_week,"week":current_week,"log_price_change":float(record.log_price_change),
            "before_contexts":len(before),"after_contexts":len(after),
            "observed_before_incidence":float(before_y.mean()),"observed_after_incidence":float(after_y.mean()),
            "observed_association":float(after_y.mean()-before_y.mean()),
            "parent_predicted_response":prediction["parent"],
            "child_predicted_response":prediction["child"],
            "minimum_ess_fraction":prediction["minimum_ess_fraction"],
            "minimum_absolute_ess":prediction["minimum_absolute_ess"],
            "model_runtime_seconds":prediction["runtime_seconds"]["total"]})
        print(f"[observational-price] events={len(rows)}/{spec['events']}",flush=True)
        if len(rows)>=int(spec["events"]): break
    frame=pd.DataFrame(rows); per=args.per_event_output.resolve(); per.parent.mkdir(parents=True,exist_ok=True)
    frame.to_parquet(per,index=False)
    if len(frame)>=2:
        child_error=paired_score_summary(
            (frame.child_predicted_response-frame.observed_association).to_numpy(),
            frame.item_id.to_numpy(),cluster_label="product")
        parent_error=paired_score_summary(
            (frame.parent_predicted_response-frame.observed_association).to_numpy(),
            frame.item_id.to_numpy(),cluster_label="product")
        metrics={"child_minus_observed":child_error,"parent_minus_observed":parent_error,
            "child_mae":float(np.mean(abs(frame.child_predicted_response-frame.observed_association))),
            "parent_mae":float(np.mean(abs(frame.parent_predicted_response-frame.observed_association))),
            "child_sign_agreement":float(np.mean(np.sign(frame.child_predicted_response)==np.sign(frame.observed_association)))}
        status="completed" if len(frame)==int(spec["events"]) else "inconclusive"
    else:
        metrics={}; status="inconclusive"
    output={"status":status,"claim_level":"observational_conditional_predictive_assessment",
        "observed_price_response_evaluated":(
            "passed" if status == "completed" else "inconclusive"),
        "model_accuracy_status":"not_assessed",
        "causal_price_identification":"not_identifiable",
        "reason_causal_not_identifiable":("transaction-derived prices, no documented assignment, "
            "no opportunity denominator, and potential time-varying confounding"),
        "checkpoint_sha256":file_sha256(run_dir/"artifacts/candidate_rank1.pt"),
        "parent_sha256":file_sha256(run_dir/"out/v3_pipeline_additive_best.pt"),
        "data_fingerprint_sha256":cblob["data_fingerprint_sha256"],
        "protocol_sha256":file_sha256(protocol_path),"requested_events":int(spec["events"]),
        "events_sha256":file_sha256(args.events.resolve()),
        "evaluated_events":len(frame),
        "training_price_change_support":[support_low,support_high],
        "training_price_change_support_rule":"within the same product's clean training-event range",
        "exclusions":excluded,"metrics":metrics,
        "per_event_output":str(per),"per_event_sha256":file_sha256(per),
        "limitations":["before/after associations are not counterfactual ground truth",
            "the raw before/after association changes the context population and is not the fixed-context price-only estimand",
            "outcomes condition on an observed nonempty basket","shared event shocks and sparse events limit inference",
            "the fitted model does not represent visit probability or quantities"],
        "runtime_seconds":time.monotonic()-started}
    out=args.output.resolve(); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(strict_json_dumps(output)); print(strict_json_dumps(output),end="")


if __name__=="__main__": main()
