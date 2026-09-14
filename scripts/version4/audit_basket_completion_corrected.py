#!/usr/bin/env python3
"""Corrected set-only evaluation of two-item basket completion.

Two targets are reported because uniform random masking is informative about final basket
size.  The literal model law P(T | A subset S,x) is evaluated with pair-incidence weights
C(|S|,2).  The shopper-uniform masking experiment is evaluated after dividing the model
law by C(|A|+|T|,2).  Neither target is a chronological next-scan experiment.
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("V3_AFFINITY", "1")

from audit_cart_conditionals_quadrature import padded_rule
from checkpoint_io import load_checkpoint
from conditional_basket import conditional_completion_quadrature
from data import build
from evaluate_retail_applications import basket, midrank
from features import Features
from fit import Batcher, popularity_logits
from provenance import file_sha256, strict_json_dumps


torch.set_default_dtype(torch.float64)


def natural_validation_panel(data, population, count, seed):
    rng = np.random.default_rng(seed)
    shuffled = np.asarray(population)[rng.permutation(len(population))]
    if len(shuffled) < count:
        raise RuntimeError("not enough eligible validation baskets")
    # A simple random sample preserves the eligible validation-trip distribution.
    # Household de-duplication would silently change the target toward household-uniform
    # weighting and is therefore deliberately not used here.
    return shuffled[:count].astype(np.int64, copy=False)


def normalized(values):
    values = np.asarray(values, dtype=np.float64)
    return values / values.sum()


def weighted_mean(values, weights):
    return float(np.sum(np.asarray(values) * normalized(weights)))


def weighted_auc(label, probability, weights):
    label = np.asarray(label, dtype=bool); probability = np.asarray(probability)
    weights = np.asarray(weights, dtype=np.float64)
    pos, neg = np.flatnonzero(label), np.flatnonzero(~label)
    if not len(pos) or not len(neg):
        return None
    comparison = ((probability[pos, None] > probability[None, neg]).astype(float)
                  + .5 * (probability[pos, None] == probability[None, neg]))
    pair_weight = weights[pos, None] * weights[None, neg]
    return float(np.sum(pair_weight * comparison) / pair_weight.sum())


def score_metrics(actual, size_probability, weights):
    actual = np.asarray(actual, dtype=np.int64)
    probability = np.asarray(size_probability, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    axis = np.arange(probability.shape[1], dtype=np.float64)
    predicted = probability @ axis
    error = predicted - actual
    selected = np.clip(probability[np.arange(len(actual)), actual], 1e-300, 1.0)
    stop_label = actual == 0
    stop_probability = probability[:, 0]
    clipped_stop = np.clip(stop_probability, 1e-12, 1-1e-12)
    stop_logloss = -(stop_label * np.log(clipped_stop)
                     + (~stop_label) * np.log(1-clipped_stop))
    return {
        "cases": int(len(actual)),
        "weight_effective_sample_size": float(weights.sum()**2 / np.square(weights).sum()),
        "maximum_normalized_weight": float(normalized(weights).max()),
        "observed_additional_items_mean": weighted_mean(actual, weights),
        "predicted_additional_items_mean": weighted_mean(predicted, weights),
        "mean_error_items": weighted_mean(error, weights),
        "mae_items": weighted_mean(np.abs(error), weights),
        "rmse_items": math.sqrt(weighted_mean(np.square(error), weights)),
        "size_log_loss": weighted_mean(-np.log(selected), weights),
        "stop_prevalence": weighted_mean(stop_label, weights),
        "mean_stop_probability": weighted_mean(stop_probability, weights),
        "stop_brier_score": weighted_mean(np.square(stop_probability-stop_label), weights),
        "stop_log_loss": weighted_mean(stop_logloss, weights),
        "stop_roc_auc": weighted_auc(stop_label, stop_probability, weights),
    }


def weighted_retrieval_metrics(ranks, recalls, weights, cutoffs=(5, 10, 20, 100)):
    ranks = np.asarray(ranks, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    result = {
        "cases": int(len(ranks)),
        "weight_effective_sample_size": (
            float(weights.sum()**2 / np.square(weights).sum()) if len(weights) else None),
        "mrr": weighted_mean(1.0 / ranks, weights) if len(ranks) else None,
        "mean_rank": weighted_mean(ranks, weights) if len(ranks) else None,
    }
    for cutoff in cutoffs:
        result[f"hit_rate_at_{cutoff}"] = weighted_mean(ranks <= cutoff, weights)
        result[f"mean_hidden_set_recall_at_{cutoff}"] = weighted_mean(
            recalls[cutoff], weights)
    return result


def paired_cluster_bootstrap(actual, model_probability, baseline_probability,
                             weights, households, repetitions, seed):
    """Household-cluster percentile intervals for model and paired improvements."""
    actual = np.asarray(actual, dtype=np.int64)
    model_probability = np.asarray(model_probability, dtype=np.float64)
    baseline_probability = np.asarray(baseline_probability, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    households = np.asarray(households)
    clusters = np.unique(households)
    members = {cluster: np.flatnonzero(households == cluster) for cluster in clusters}
    rng = np.random.default_rng(seed)

    def values(index):
        w = normalized(weights[index])
        y = actual[index]
        axis = np.arange(model_probability.shape[1], dtype=np.float64)
        model_mean = model_probability[index] @ axis
        base_mean = baseline_probability[index] @ axis
        model_selected = np.clip(
            model_probability[index, y], 1e-300, 1.0)
        base_selected = np.clip(
            baseline_probability[index, y], 1e-300, 1.0)
        label = y == 0
        mp0 = np.clip(model_probability[index, 0], 1e-12, 1-1e-12)
        bp0 = np.clip(baseline_probability[index, 0], 1e-12, 1-1e-12)
        model_stop_log = -(label*np.log(mp0) + (~label)*np.log(1-mp0))
        base_stop_log = -(label*np.log(bp0) + (~label)*np.log(1-bp0))
        return np.asarray([
            np.sum(w * (model_mean-y)),
            np.sum(w * np.abs(model_mean-y)) - np.sum(w * np.abs(base_mean-y)),
            np.sum(w * -np.log(model_selected)) - np.sum(w * -np.log(base_selected)),
            np.sum(w * np.square(mp0-label)) - np.sum(w * np.square(bp0-label)),
            np.sum(w * model_stop_log) - np.sum(w * base_stop_log),
        ])

    estimates = values(np.arange(len(actual)))
    draws = np.empty((repetitions, len(estimates)), dtype=np.float64)
    for repeat in range(repetitions):
        sampled = rng.choice(clusters, len(clusters), replace=True)
        index = np.concatenate([members[cluster] for cluster in sampled])
        draws[repeat] = values(index)
    names = [
        "model_mean_error_items",
        "model_minus_baseline_mae_items",
        "model_minus_baseline_size_log_loss",
        "model_minus_baseline_stop_brier_score",
        "model_minus_baseline_stop_log_loss",
    ]
    return {
        "method": "household-cluster percentile bootstrap",
        "repetitions": int(repetitions),
        "clusters": int(len(clusters)),
        "negative_paired_difference_favors_model": True,
        **{name: {
            "estimate": float(estimates[i]),
            "percentile_95_interval": [
                float(np.quantile(draws[:, i], .025)),
                float(np.quantile(draws[:, i], .975)),
            ]}
           for i, name in enumerate(names)},
    }


def empirical_size_baselines(data, nmax):
    count = np.zeros(nmax - 1, dtype=np.float64)       # remainder 0..nmax-2
    pair_count = np.zeros_like(count)
    training = np.flatnonzero(data["trip_split"] == 0)
    used = 0
    for trip in training:
        size = len(basket(data, int(trip)))
        if 2 <= size <= nmax:
            count[size-2] += 1.0
            pair_count[size-2] += size * (size-1) / 2.0
            used += 1
    return normalized(count), normalized(pair_count), used


@torch.no_grad()
def evaluate(args):
    started = time.monotonic(); torch.set_num_threads(args.threads)
    data = build(); checkpoint = args.checkpoint.resolve()
    model, blob, meta = load_checkpoint(
        checkpoint, data,
        required_capabilities=("conditional_nonempty_incidence", "gram_interactions"))
    for parameter in model.parameters(): parameter.requires_grad_(False)
    nmax = int(meta["nmax"])
    population = []
    for trip in np.flatnonzero(data["trip_split"] == 1):
        size = len(basket(data, int(trip)))
        if 2 <= size <= nmax:
            population.append(int(trip))
    trips = natural_validation_panel(data, population, args.contexts, args.seed)
    rng = np.random.default_rng(args.seed + 1)
    anchors, hidden, actual = [], [], []
    for trip in trips:
        observed = basket(data, int(trip))
        anchor = rng.choice(observed, 2, replace=False)
        anchors.append(list(map(int, anchor)))
        hidden.append([int(item) for item in observed if item not in set(anchor)])
        actual.append(int(len(observed)-2))
    actual = np.asarray(actual, dtype=np.int64)
    pair_weight = (actual + 2) * (actual + 1) / 2.0

    batcher = Batcher(
        data, Features(int(data["n_item"]), int(data["n_store"]), 712,
                       include_recency=False), nmax, include_recency=False)
    singular = torch.linalg.svdvals(model.phi)
    active_rank = int((singular > singular[0] * 1e-10).sum())
    low_level, high_level, follow_level = (
        active_rank + 2, active_rank + 3, active_rank + 4)
    low_rule = padded_rule(model, active_rank, low_level)
    high_rule = padded_rule(model, active_rank, high_level)
    follow_rule = padded_rule(model, active_rank, follow_level)
    completion_axis = np.arange(nmax + 1, dtype=np.float64)
    selection_denominator = (completion_axis + 2) * (completion_axis + 1) / 2.0
    selection_log_weight = -torch.log(torch.as_tensor(
        selection_denominator, dtype=model.phi.dtype, device=model.phi.device))
    low_stop, low_expected, low_size = [], [], []
    high_stop, high_expected, high_size = [], [], []
    low_incidence, high_incidence, high_logz = [], [], []
    for start in range(0, len(trips), args.chunk):
        sub = trips[start:start+args.chunk]
        ix, ctx, _lc, house, *_ = batcher.make(sub)
        model.house, model.ctx = house, ctx
        slot_b = model.b_flat(ix).detach()
        lo = conditional_completion_quadrature(
            model, ix, slot_b, anchors[start:start+len(sub)], *low_rule,
            completion_size_log_weight=selection_log_weight)
        hi = conditional_completion_quadrature(
            model, ix, slot_b, anchors[start:start+len(sub)], *high_rule,
            completion_size_log_weight=selection_log_weight)
        low_stop.append(lo.stop_probability.numpy())
        low_expected.append(lo.expected_additional_items.numpy())
        low_size.append(lo.completion_size_probability.numpy())
        low_incidence.append(lo.item_incidence.numpy())
        high_stop.append(hi.stop_probability.numpy())
        high_expected.append(hi.expected_additional_items.numpy())
        high_size.append(hi.completion_size_probability.numpy())
        high_incidence.append(hi.item_incidence.numpy())
        high_logz.append(hi.log_normalizer.numpy())
        print(f"[completion-corrected] contexts={start+len(sub)}/{len(trips)}", flush=True)
    low_stop=np.concatenate(low_stop); low_expected=np.concatenate(low_expected)
    low_size=np.concatenate(low_size)
    low_incidence=np.concatenate(low_incidence)
    high_stop=np.concatenate(high_stop); high_expected=np.concatenate(high_expected)
    high_size=np.concatenate(high_size); high_logz=np.concatenate(high_logz)
    high_incidence=np.concatenate(high_incidence)
    size_gap=np.max(np.abs(high_size-low_size),axis=1)
    incidence_gap=np.max(np.abs(high_incidence-low_incidence),axis=1)
    follow_indices=np.flatnonzero(
        (np.abs(high_stop-low_stop)>args.maximum_stop_gap)
        | (np.abs(high_expected-low_expected)>args.maximum_size_gap)
        | (size_gap>args.maximum_size_probability_gap)
        | (incidence_gap>args.maximum_incidence_gap))
    reference_size=high_size.copy(); reference_logz=high_logz.copy()
    reference_incidence=high_incidence.copy()
    follow_stop_gap=[]; follow_expected_gap=[]; follow_size_gap=[]; follow_incidence_gap=[]
    for start in range(0,len(follow_indices),args.followup_chunk):
        selected=follow_indices[start:start+args.followup_chunk]
        sub=trips[selected]
        ix,ctx,_lc,house,*_=batcher.make(sub); model.house,model.ctx=house,ctx
        slot_b=model.b_flat(ix).detach()
        follow=conditional_completion_quadrature(
            model,ix,slot_b,[anchors[int(i)] for i in selected],*follow_rule,
            completion_size_log_weight=selection_log_weight)
        fs=follow.completion_size_probability.numpy()
        follow_stop_gap.extend(np.abs(fs[:,0]-high_stop[selected]).tolist())
        follow_expected_gap.extend(np.abs(
            follow.expected_additional_items.numpy()-high_expected[selected]).tolist())
        follow_size_gap.extend(np.max(np.abs(fs-high_size[selected]),axis=1).tolist())
        fi=follow.item_incidence.numpy()
        follow_incidence_gap.extend(
            np.max(np.abs(fi-high_incidence[selected]),axis=1).tolist())
        reference_size[selected]=fs; reference_logz[selected]=follow.log_normalizer.numpy()
        reference_incidence[selected]=fi
        print(f"[completion-corrected-followup] contexts="
              f"{min(start+args.followup_chunk,len(follow_indices))}/"
              f"{len(follow_indices)}",flush=True)
    gates={
        "stop_probability": max(follow_stop_gap,default=0)<=args.maximum_stop_gap,
        "expected_additional_items": max(follow_expected_gap,default=0)<=args.maximum_size_gap,
        "size_probability": max(follow_size_gap,default=0)<=args.maximum_size_probability_gap,
        "item_incidence": max(follow_incidence_gap,default=0)<=args.maximum_incidence_gap,
    }

    # The quadrature result is already adjusted for uniform pair masking. Recover the
    # literal conditional size law by undoing that known observation weight.
    mask_adjusted=reference_size
    literal_size=mask_adjusted*selection_denominator[None,:]
    literal_size/=literal_size.sum(1,keepdims=True)
    incidence_count_gap=np.abs(reference_incidence.sum(1)
                               - mask_adjusted@completion_axis)
    gates["item_incidence_sum_identity"] = bool(incidence_count_gap.max() <= 1e-8)
    natural_baseline,pair_baseline,training_cases=empirical_size_baselines(data,nmax)
    baseline_natural=np.broadcast_to(
        np.pad(natural_baseline,(0,reference_size.shape[1]-len(natural_baseline))),
        reference_size.shape)
    baseline_pair=np.broadcast_to(
        np.pad(pair_baseline,(0,reference_size.shape[1]-len(pair_baseline))),
        reference_size.shape)

    # Recommendation metrics must use the same masking-aware joint law as the size and
    # stopping metrics.  Ranking the raw literal incidence would evaluate a different
    # estimand because final basket size and product identity are dependent.
    popularity=popularity_logits(
        data,np.flatnonzero(data["trip_split"]==0)).numpy()
    model_ranks,popularity_ranks,ranking_weights=[],[],[]
    model_recalls={5:[],10:[],20:[],100:[]}
    popularity_recalls={5:[],10:[],20:[],100:[]}
    for start in range(0,len(trips),args.chunk):
        sub=trips[start:start+args.chunk]
        ix,*_=batcher.make(sub)
        for local,_trip in enumerate(sub):
            absolute=start+local
            if not hidden[absolute]:
                continue
            candidates=ix.item[ix.item_trip==local].numpy()
            candidates=np.asarray([
                int(item) for item in candidates
                if int(item) not in set(anchors[absolute])],dtype=np.int64)
            positions=np.flatnonzero(np.isin(candidates,hidden[absolute]))
            if not len(positions):
                continue
            model_score=reference_incidence[absolute,candidates]
            popularity_score=popularity[candidates]
            mranks=np.asarray([midrank(model_score,int(position))
                               for position in positions])
            pranks=np.asarray([midrank(popularity_score,int(position))
                               for position in positions])
            model_ranks.append(float(mranks.min()))
            popularity_ranks.append(float(pranks.min()))
            ranking_weights.append(1.0)
            for cutoff in model_recalls:
                model_recalls[cutoff].append(float(np.mean(mranks<=cutoff)))
                popularity_recalls[cutoff].append(float(np.mean(pranks<=cutoff)))

    households=data["trip_user"][trips]
    natural_uncertainty=paired_cluster_bootstrap(
        actual,mask_adjusted,baseline_natural,np.ones(len(actual)),households,
        args.bootstrap_repetitions,args.seed+11)
    literal_uncertainty=paired_cluster_bootstrap(
        actual,literal_size,baseline_pair,pair_weight,households,
        args.bootstrap_repetitions,args.seed+12)
    per_context=[]
    for i,trip in enumerate(trips):
        per_context.append({
            "trip":int(trip),"household":int(data["trip_user"][trip]),
            "observed_size":int(actual[i]+2),"revealed_items":anchors[i],
            "actual_additional_items":int(actual[i]),"pair_incidence_weight":float(pair_weight[i]),
            "literal_expected_additional_items":float(literal_size[i]@completion_axis),
            "mask_adjusted_expected_additional_items":float(mask_adjusted[i]@completion_axis),
            "literal_stop_probability":float(literal_size[i,0]),
            "mask_adjusted_stop_probability":float(mask_adjusted[i,0]),
            "literal_conditional_log_normalizer":float(-np.log(literal_size[i,0])),
            "mask_adjusted_log_normalizer":float(reference_logz[i]),
            "literal_actual_size_probability":float(literal_size[i,actual[i]]),
            "mask_adjusted_actual_size_probability":float(mask_adjusted[i,actual[i]]),
            "training_baseline_actual_size_probability":float(
                baseline_natural[i,actual[i]]),
        })
    return {
        "status":"passed" if all(gates.values()) else "failed",
        "claim":"set-only masked completion evaluation; not chronological cart validation",
        "checkpoint":str(checkpoint),"checkpoint_sha256":file_sha256(checkpoint),
        "data_fingerprint_sha256":blob["data_fingerprint_sha256"],
        "split":"validation","contexts":len(trips),"panel_seed":args.seed,
        "distinct_households":int(np.unique(data["trip_user"][trips]).size),
        "observed_size_distribution":{
            "minimum":int(actual.min()+2),"mean":float((actual+2).mean()),
            "maximum":int(actual.max()+2),"size_two_fraction":float(np.mean(actual==0)),
        },
        "numerical_certification":{
            "levels":[low_level,high_level,follow_level],
            "nodes":[len(low_rule[1]),len(high_rule[1]),len(follow_rule[1])],
            "followup_contexts":int(len(follow_indices)),
            "maximum_initial_stop_gap":float(np.max(np.abs(high_stop-low_stop))),
            "maximum_initial_expected_size_gap":float(np.max(np.abs(high_expected-low_expected))),
            "maximum_initial_size_probability_gap":float(size_gap.max()),
            "maximum_initial_item_incidence_gap":float(incidence_gap.max()),
            "maximum_followup_stop_gap":float(max(follow_stop_gap,default=0)),
            "maximum_followup_expected_size_gap":float(max(follow_expected_gap,default=0)),
            "maximum_followup_size_probability_gap":float(max(follow_size_gap,default=0)),
            "maximum_followup_item_incidence_gap":float(
                max(follow_incidence_gap,default=0)),
            "maximum_item_incidence_sum_vs_expected_size_gap":float(
                incidence_count_gap.max()),
            "gates":gates,
        },
        "literal_cart_conditional_pair_incidence_target":{
            "selection":"one random pair per natural trip, weighted by C(observed_size,2)",
            "model":score_metrics(actual,literal_size,pair_weight),
            "training_only_empirical_size_baseline":score_metrics(
                actual,baseline_pair,pair_weight),
            "uncertainty":literal_uncertainty,
        },
        "shopper_uniform_random_pair_mask_target":{
            "selection":"one natural validation trip, then one uniformly random observed pair",
            "model":score_metrics(actual,mask_adjusted,np.ones(len(actual))),
            "training_only_empirical_size_baseline":score_metrics(
                actual,baseline_natural,np.ones(len(actual))),
            "uncertainty":natural_uncertainty,
            "cross_sell_retrieval":{
                "model":weighted_retrieval_metrics(
                    model_ranks,model_recalls,ranking_weights),
                "training_popularity":weighted_retrieval_metrics(
                    popularity_ranks,popularity_recalls,ranking_weights),
                "eligible_cases_exclude_true_stop_baskets":True,
            },
        },
        "training_baseline_baskets":training_cases,
        "limitations":[
            "no chronological scan or cart-event order is available",
            "random-pair masking is a set-only proxy and its selection mechanism must be modeled",
            "one sampled anchor pair per trip introduces finite masking variance",
        ],
        "per_context":per_context,"runtime_seconds":time.monotonic()-started,
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--contexts",type=int,default=256)
    parser.add_argument("--chunk",type=int,default=8)
    parser.add_argument("--followup-chunk",type=int,default=2)
    parser.add_argument("--threads",type=int,default=4)
    parser.add_argument("--seed",type=int,default=91531)
    parser.add_argument("--maximum-stop-gap",type=float,default=1e-4)
    parser.add_argument("--maximum-size-gap",type=float,default=.02)
    parser.add_argument("--maximum-size-probability-gap",type=float,default=1e-4)
    parser.add_argument("--maximum-incidence-gap",type=float,default=1e-4)
    parser.add_argument("--bootstrap-repetitions",type=int,default=1000)
    args=parser.parse_args(); result=evaluate(args)
    output=args.output.resolve();output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(strict_json_dumps(result))
    print(strict_json_dumps({k:v for k,v in result.items() if k!="per_context"}),end="")


if __name__=="__main__":main()
