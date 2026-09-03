#!/usr/bin/env python3
"""Customer segments, conditional generation, and price-response audit.

Segments are fitted only from rotation-invariant household taste and price surfaces in a
trained version-4 checkpoint.  The test split is used only after clustering.  For each
segment, interaction-tempered SMC generates baskets in real held-out store/household/date
contexts, and the generated law is compared with observed baskets using smoothed KL, JS,
TV, and moment diagnostics at size, commodity, item, and commodity-pair levels.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from torch.nn.functional import softplus

from audit_particle_counterfactual_generation import (ROOT, copied_context,
                                                       load_checkpoint,
                                                       named_basket,
                                                       particle_delta)
from data import build
from features import Features
from fit import Batcher
from interaction_particles import (blocked_rejuvenation,
                                   rao_blackwell_particle_statistics)
from tempered_ais import annealed_smc_logz


torch.set_default_dtype(torch.float64)


def induced_representation(left, right):
    """Coordinates whose distances preserve the identified bilinear surface."""
    gram = right.T @ right
    value, vector = np.linalg.eigh(0.5 * (gram + gram.T))
    keep = value > max(float(value.max()), 1.0) * 1e-12
    return (left @ vector[:, keep]) * np.sqrt(value[keep])[None, :]


def household_representation(model):
    taste = induced_representation(
        model.theta_c().detach().numpy(), model.alpha.detach().numpy())
    gamma = softplus(model.gamma.detach()).numpy()
    beta = softplus(model.beta.detach()).numpy()
    price = induced_representation(gamma, beta)
    # Standardize the two identified surfaces separately, then give each block equal
    # aggregate Euclidean weight rather than letting its raw dimension choose the result.
    taste = StandardScaler().fit_transform(taste) / math.sqrt(taste.shape[1])
    price = StandardScaler().fit_transform(price) / math.sqrt(price.shape[1])
    return np.concatenate([taste, price], axis=1), taste, price


def choose_segments(representation, candidates, seed):
    fits = {}
    for count in candidates:
        labels, silhouettes = [], []
        for repeat in range(3):
            fit = KMeans(n_clusters=count, n_init=20, random_state=seed + repeat)
            label = fit.fit_predict(representation)
            labels.append(label)
            silhouettes.append(float(silhouette_score(
                representation, label, sample_size=min(2000, len(label)),
                random_state=seed + 100 + repeat)))
        stability = float(np.mean([
            adjusted_rand_score(labels[i], labels[j])
            for i in range(3) for j in range(i + 1, 3)]))
        fractions = np.bincount(labels[0], minlength=count) / len(labels[0])
        accepted = bool(fractions.min() >= 0.05)
        score = float(np.mean(silhouettes)) + 0.05 * stability
        fits[count] = {
            "silhouette": float(np.mean(silhouettes)),
            "stability_ari": stability,
            "minimum_fraction": float(fractions.min()),
            "accepted_minimum_5pct": accepted,
            "selection_score": score if accepted else None,
        }
    accepted = [count for count in candidates
                if fits[count]["selection_score"] is not None]
    if not accepted:
        raise RuntimeError("no candidate segmentation has at least 5% in every segment")
    chosen = max(accepted, key=lambda count: fits[count]["selection_score"])
    final = KMeans(n_clusters=chosen, n_init=50, random_state=seed).fit(representation)
    return final.labels_.astype(np.int64), chosen, fits, final.cluster_centers_


def trip_basket(data, trip):
    lo, hi = int(data["line_ptr"][trip]), int(data["line_ptr"][trip + 1])
    return np.unique(data["line_item"][lo:hi]).astype(np.int64, copy=False)


def segment_trips(data, labels, split, nmax):
    result = {}
    candidates = np.flatnonzero(
        (data["trip_split"] == split) & (data["trip_nlines"] <= nmax))
    for segment in np.unique(labels):
        result[int(segment)] = candidates[
            labels[data["trip_user"][candidates]] == segment]
    return result


def basket_counts(baskets, item_category, nmax, n_item, n_category):
    size = np.zeros(nmax, dtype=np.float64)
    category = np.zeros(n_category, dtype=np.float64)
    item = np.zeros(n_item, dtype=np.float64)
    pair = np.zeros(n_category * n_category, dtype=np.float64)
    for basket in baskets:
        basket = np.unique(np.asarray(basket, dtype=np.int64))
        if not len(basket):
            continue
        size[min(len(basket), nmax) - 1] += 1
        np.add.at(item, basket, 1)
        cats = item_category[basket]
        np.add.at(category, cats, 1)
        for first, second in combinations(cats.tolist(), 2):
            lo, hi = sorted((int(first), int(second)))
            pair[lo * n_category + hi] += 1
    return {"size": size, "category": category, "item": item,
            "category_pair": pair}


def probability(count, prior_mass=1.0):
    count = np.asarray(count, dtype=np.float64)
    return (count + prior_mass / len(count)) / (count.sum() + prior_mass)


def divergence(observed, generated):
    p, q = probability(observed), probability(generated)
    middle = 0.5 * (p + q)
    return {
        "kl_observed_to_generated": float(np.sum(p * np.log(p / q))),
        "kl_generated_to_observed": float(np.sum(q * np.log(q / p))),
        "jensen_shannon": float(
            0.5 * np.sum(p * np.log(p / middle))
            + 0.5 * np.sum(q * np.log(q / middle))),
        "total_variation": float(0.5 * np.abs(p - q).sum()),
        "observed_events": float(np.asarray(observed).sum()),
        "generated_events": float(np.asarray(generated).sum()),
        "symmetric_dirichlet_prior_mass": 1.0,
    }


def distribution_audit(observed_baskets, generated_baskets, item_category,
                       nmax, n_item, n_category):
    observed = basket_counts(
        observed_baskets, item_category, nmax, n_item, n_category)
    generated = basket_counts(
        generated_baskets, item_category, nmax, n_item, n_category)
    answer = {name: divergence(observed[name], generated[name])
              for name in observed}
    observed_size = np.asarray([len(np.unique(x)) for x in observed_baskets], float)
    generated_size = np.asarray([len(np.unique(x)) for x in generated_baskets], float)
    answer["moments"] = {
        "observed_size_mean": float(observed_size.mean()),
        "generated_size_mean": float(generated_size.mean()),
        "observed_size_variance": float(observed_size.var()),
        "generated_size_variance": float(generated_size.var()),
    }
    # Sampling noise floor: disagreement between two observed context halves.
    first = observed_baskets[::2]
    second = observed_baskets[1::2]
    if first and second:
        left = basket_counts(first, item_category, nmax, n_item, n_category)
        right = basket_counts(second, item_category, nmax, n_item, n_category)
        answer["observed_split_half_reference"] = {
            name: divergence(left[name], right[name]) for name in left}
    generated_first = generated_baskets[::2]
    generated_second = generated_baskets[1::2]
    if generated_first and generated_second:
        left = basket_counts(
            generated_first, item_category, nmax, n_item, n_category)
        right = basket_counts(
            generated_second, item_category, nmax, n_item, n_category)
        answer["generated_split_half_reference"] = {
            name: divergence(left[name], right[name]) for name in left}
    return answer


def category_names(metadata, n_category):
    names = []
    for category in range(n_category):
        rows = metadata[metadata.cat_id == category]
        names.append(str(rows.COMMODITY_DESC.iloc[0]) if len(rows) else str(category))
    return names


def top_overindex(data, trips, item_category, global_count, names, limit=6):
    count = np.zeros_like(global_count, dtype=np.float64)
    for trip in trips:
        np.add.at(count, item_category[trip_basket(data, int(trip))], 1)
    local = probability(count)
    overall = probability(global_count)
    log_overindex = np.log(local / overall)
    # Pure lift promotes categories represented by only a handful of lines.  Weight lift
    # by sqrt(evidence) and require at least 0.1% of the segment's lines (minimum 20).
    minimum = max(20, int(math.ceil(0.001 * count.sum())))
    eligible = count >= minimum
    score = np.maximum(log_overindex, 0.0) * np.sqrt(count)
    order = np.argsort(np.where(eligible, score, -np.inf))[-limit:][::-1]
    return [{"category": int(index), "name": names[index],
             "log_overindex": float(log_overindex[index]),
             "evidence_weighted_score": float(score[index]),
             "minimum_lines_for_label": minimum, "lines": int(count[index])}
            for index in order if np.isfinite(score[index]) and eligible[index]]


@torch.no_grad()
def simulate_segment(model, batcher, data, trips, item_category, metadata, args,
                     segment, reference_trips):
    schedule_axis = torch.linspace(0.0, 1.0, args.levels)
    schedule = 1.0 - (1.0 - schedule_axis).pow(args.power)
    observed_baskets, generated_baskets, examples = [], [], []
    factual_sizes, invalid, duplicates = [], 0, 0
    counterfactual = {action: {"uniform_size_change": [], "own_retained": [],
                               "own_ess": [], "uniform_ess": []}
                      for action in args.actions}
    smc_seconds = 0.0
    for start in range(0, len(trips), args.context_chunk):
        sub = trips[start:start + args.context_chunk]
        ix, ctx, _line_ctx, house, line_item, line_trip, _line_cat, _ = batcher.make(sub)
        model.house, model.ctx = house, ctx
        generator = torch.Generator().manual_seed(
            args.seed + 100003 * segment + start)
        tick = time.perf_counter()
        smc = annealed_smc_logz(
            model, ix, schedule, particles=args.particles,
            mutation_steps=1, generator=generator)
        smc_seconds += time.perf_counter() - tick
        factual = rao_blackwell_particle_statistics(model, ix, smc.states)
        size_axis = torch.arange(1, model.nmax + 1, dtype=model.phi.dtype)
        factual_size = (factual.size_probability * size_axis).sum(1)
        factual_sizes.extend(factual_size.tolist())
        factual_b = model.b_flat(ix).clone()
        chosen_slots, chosen_items = [], []
        rng = np.random.default_rng(args.seed + 700001 * segment + start)
        for b in range(ix.B):
            observed = torch.unique(line_item[line_trip == b]).numpy()
            observed_baskets.append(observed.tolist())
            chosen = int(observed[rng.integers(len(observed))])
            slot = torch.nonzero(
                (ix.item_trip == b) & (ix.item == chosen), as_tuple=True)[0]
            chosen_slots.append(int(slot[0])); chosen_items.append(chosen)
        chosen_slots = torch.as_tensor(chosen_slots, dtype=torch.long)
        chosen_items_tensor = torch.as_tensor(chosen_items, dtype=torch.long)
        factual_own = factual.item_incidence[
            torch.arange(ix.B), chosen_items_tensor].clamp_min(1e-12)
        assortment_size = torch.bincount(ix.item_trip, minlength=ix.B).double()
        for action in args.actions:
            own = copied_context(ctx)
            own["dlp"][chosen_slots] += action
            own["dlp_bar"] += action / assortment_size
            model.ctx = own
            own_delta = particle_delta(smc.states, model.b_flat(ix) - factual_b, ix.B)
            own_log_weight = torch.log_softmax(own_delta, dim=0)
            own_ess = torch.exp(-torch.logsumexp(
                2.0 * own_log_weight, dim=0)) / args.particles
            own_stats = rao_blackwell_particle_statistics(
                model, ix, smc.states, own_log_weight)
            changed_own = own_stats.item_incidence[
                torch.arange(ix.B), chosen_items_tensor]
            counterfactual[action]["own_retained"].extend(
                (changed_own / factual_own).tolist())
            counterfactual[action]["own_ess"].extend(own_ess.tolist())

            uniform = copied_context(ctx)
            uniform["dlp"] += action
            uniform["dlp_bar"] += action
            model.ctx = uniform
            uniform_delta = particle_delta(
                smc.states, model.b_flat(ix) - factual_b, ix.B)
            uniform_log_weight = torch.log_softmax(uniform_delta, dim=0)
            uniform_ess = torch.exp(-torch.logsumexp(
                2.0 * uniform_log_weight, dim=0)) / args.particles
            uniform_stats = rao_blackwell_particle_statistics(
                model, ix, smc.states, uniform_log_weight)
            uniform_size = (uniform_stats.size_probability * size_axis).sum(1)
            counterfactual[action]["uniform_size_change"].extend(
                (uniform_size - factual_size).tolist())
            counterfactual[action]["uniform_ess"].extend(uniform_ess.tolist())
        model.ctx = ctx
        states = blocked_rejuvenation(
            model, ix, smc.states, beta=1.0, steps=args.rejuvenation,
            generator=torch.Generator().manual_seed(
                args.seed + 900001 * segment + start))
        for b in range(ix.B):
            allowed = set(ix.item[ix.item_trip == b].numpy().tolist())
            for particle in states:
                basket = ix.item[particle[b]].numpy().tolist()
                generated_baskets.append(basket)
                invalid += int(any(item not in allowed for item in basket))
                duplicates += int(len(basket) != len(set(basket)))
            if len(examples) < 3:
                examples.append({
                    "trip": int(sub[b]),
                    "observed": named_basket(observed_baskets[-ix.B + b], metadata),
                    "generated": named_basket(
                        ix.item[states[0][b]].numpy().tolist(), metadata),
                })
        print(
            f"[segments] segment={segment} simulated_contexts="
            f"{min(start + args.context_chunk, len(trips))}/{len(trips)}",
            flush=True,
        )
    action_rows = []
    for action in args.actions:
        row = counterfactual[action]
        action_rows.append({
            "price_multiplier": math.exp(action),
            "log_price_change": action,
            "own_incidence_retained_mean": float(np.mean(row["own_retained"])),
            "uniform_expected_size_change_mean": float(
                np.mean(row["uniform_size_change"])),
            "own_reweight_ess_min": float(np.min(row["own_ess"])),
            "uniform_reweight_ess_min": float(np.min(row["uniform_ess"])),
        })
    reference_baskets = [trip_basket(data, int(trip)) for trip in reference_trips]
    return {
        "contexts": int(len(trips)), "particles_per_context": args.particles,
        "smc_seconds": smc_seconds,
        "factual_expected_size_mean": float(np.mean(factual_sizes)),
        "invalid_assortment_baskets": invalid,
        "duplicate_item_baskets": duplicates,
        "counterfactuals": action_rows,
        "distribution": distribution_audit(
            reference_baskets, generated_baskets, item_category, model.nmax,
            model.J, model.C),
        # The historical audit compared generated baskets from a small context sample to
        # every basket in the segment.  That confounds conditional model error with a
        # nonrepresentative context draw (in run310, selected segment-0 and segment-2 means
        # missed their populations by about two items).  This predictive check gives every
        # selected context one observed basket and the same number of generated particles,
        # so both sides use the identical household/store/week mixture.
        "same_context_distribution": distribution_audit(
            observed_baskets, generated_baskets, item_category, model.nmax,
            model.J, model.C),
        "examples": examples,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--candidate-segments", type=int, nargs="+",
                        default=[3, 4, 5, 6, 7])
    parser.add_argument("--contexts-per-segment", type=int, default=48)
    parser.add_argument("--particles", type=int, default=32)
    parser.add_argument("--levels", type=int, default=17)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--rejuvenation", type=int, default=1)
    parser.add_argument("--context-chunk", type=int, default=8)
    parser.add_argument("--actions", type=float, nargs="+",
                        default=[math.log(.8), math.log(.9), 0.0,
                                 math.log(1.1), math.log(1.2)])
    parser.add_argument("--seed", type=int, default=30741)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("out/v3_customer_segment_audit.json"))
    parser.add_argument("--assignments", type=Path,
                        default=Path("out/v3_customer_segments.npz"))
    parser.add_argument("--skip-simulation", action="store_true",
                        help="fit/profile segments without running SMC")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    data = build()
    ckpt = args.ckpt if args.ckpt.is_absolute() else ROOT / args.ckpt
    model, blob, meta = load_checkpoint(ckpt, data)
    representation, taste, price = household_representation(model)
    labels, count, selection, centers = choose_segments(
        representation, args.candidate_segments, args.seed)
    selected = selection[count]
    print(
        f"[segments] selected_k={count} "
        f"silhouette={selected['silhouette']:.6f} "
        f"stability_ari={selected['stability_ari']:.6f}",
        flush=True,
    )
    metadata = pd.read_parquet(ROOT / "basket_input" / "items.parquet").sort_values(
        "item_id")
    item_category = metadata.cat_id.to_numpy(dtype=np.int64)
    names = category_names(metadata, model.C)
    test_by_segment = segment_trips(data, labels, 2, model.nmax)
    validation_by_segment = segment_trips(data, labels, 1, model.nmax)
    all_test = np.concatenate(list(test_by_segment.values()))
    global_category = np.zeros(model.C, dtype=np.float64)
    for trip in all_test:
        np.add.at(global_category, item_category[trip_basket(data, int(trip))], 1)
    gamma = softplus(model.gamma.detach()).numpy()
    beta = softplus(model.beta.detach()).numpy()
    line_trip = np.repeat(np.arange(len(data["trip_nlines"])),
                          data["trip_nlines"])
    training_line_trip = line_trip[data["trip_split"][line_trip] == 0]
    train_lines = np.bincount(
        data["trip_user"][training_line_trip],
        minlength=int(data["n_user"]))
    mean_beta = beta.mean(0)
    price_coefficient = gamma @ mean_beta * float(softplus(model.price_kappa.detach()))
    batcher = Batcher(
        data, Features(int(data["n_item"]), int(data["n_store"]), 712),
        model.nmax)
    rng = np.random.default_rng(args.seed + 1)
    segments = []
    for segment in range(count):
        households = np.flatnonzero(labels == segment)
        trips = test_by_segment[segment]
        selected = trips[rng.permutation(len(trips))[:args.contexts_per_segment]]
        overindex = top_overindex(
            data, trips, item_category, global_category, names)
        sensitivity = float(np.mean(price_coefficient[households]))
        global_quantile = float(np.mean(price_coefficient <= sensitivity))
        label = (" / ".join(row["name"] for row in overindex[:2])
                 + ("; high price sensitivity" if global_quantile >= .67 else
                    "; low price sensitivity" if global_quantile <= .33 else
                    "; medium price sensitivity"))
        print(
            f"[segments] starting segment={segment} "
            f"households={len(households)} real_test_baskets={len(trips)} "
            f"simulation_contexts={len(selected)}",
            flush=True,
        )
        simulation = (None if args.skip_simulation else simulate_segment(
            model, batcher, data, selected, item_category, metadata, args, segment,
            trips))
        segments.append({
            "segment": segment, "label": label,
            "households": int(len(households)),
            "validation_trips": int(len(validation_by_segment[segment])),
            "test_trips": int(len(trips)),
            "mean_training_lines_per_household": float(train_lines[households].mean()),
            "mean_price_coefficient": sensitivity,
            "price_coefficient_population_quantile": global_quantile,
            "top_test_category_overindex": overindex,
            "simulation": simulation,
        })
    assignments = (args.assignments if args.assignments.is_absolute()
                   else ROOT / args.assignments)
    np.savez_compressed(assignments, household=np.arange(len(labels)), segment=labels,
                        representation=representation, taste=taste, price=price,
                        centers=centers)
    output = {
        "checkpoint": str(ckpt), "checkpoint_iteration": int(blob["iter"]),
        "method": ("KMeans on separately standardized, equal-block-weighted, "
                   "rotation-invariant household taste and price surfaces"),
        "test_leakage": False,
        "chosen_segments": count, "candidate_selection": selection,
        "assignments": str(assignments),
        "distribution_metrics": (
            "symmetric Dirichlet total prior mass 1; KL in nats; JS and TV reported; "
            "all eligible segment test baskets form the real reference; observed and "
            "generated split-half divergences quantify finite-sample noise"),
        "segments": segments,
    }
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    output_path.write_text(json.dumps(output, indent=2) + "\n")
    summary = {
        "checkpoint": output["checkpoint"],
        "chosen_segments": output["chosen_segments"],
        "assignments": output["assignments"],
        "segments": [{
            key: segment[key] for key in (
                "segment", "label", "households", "validation_trips",
                "test_trips", "mean_price_coefficient")
        } for segment in output["segments"]],
        "full_report": str(output_path),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
