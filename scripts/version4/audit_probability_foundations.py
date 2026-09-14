#!/usr/bin/env python3
"""Exact-enumeration oracle audits of the actual Version-4 model and samplers.

All tolerances, seeds and sampling budgets are fixed before generating outcomes. The
audit covers identities, price-penalty gradients, null/interacting generators, randomized
price opportunities, and pathological ESS. Finite experiments validate implementation;
they do not prove universal theorems or establish effects in observational retail data.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import time
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.stats import norm
import torch
from torch.nn.functional import softplus

from interaction_particles import differentiable_log_size_beta0
from pipeline_support import smolyak_rule
from price_response import (additive_uniform_price_response, changed_price_context,
                            price_jacobian, single_utility_incidence)
from provenance import strict_json_dumps
from ragged import RaggedIndex, RaggedModel
from stratified_natural import (StratifiedNaturalBank, evaluation_summary,
                               within_band_ess_passes)
from tempered_ais import annealed_smc_logz
from tempered_block_gibbs import conditional_slots_repeated
from uncertainty import household_cluster_se

torch.set_default_dtype(torch.float64)


def make_world(seed=90701, products=10, contexts=4, nmax=5, strength=0.5, kappa=2.4):
    """Small retailer: heterogeneous households/prices, promotions, and basket missions."""
    generator = torch.Generator().manual_seed(seed)
    categories, rank = 3, 2
    model = RaggedModel(products, contexts, categories, K=3, Kz=rank,
                        nmax=nmax, R=nmax, Kp=1, S=2, seed=seed,
                        household_size_rank1=True).double()
    cat = torch.arange(products) % categories
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.lam.copy_(-1.15 + .35 * torch.randn(products, generator=generator))
        model.alpha[:, :-1].copy_(.25 * torch.randn(products, 2, generator=generator))
        model.alpha[:, -1].fill_(1)
        model.theta.copy_(.2 * torch.randn(contexts, 3, generator=generator))
        model.gamma.copy_(torch.log(torch.expm1(torch.linspace(.45, 1.2, contexts)))[:, None])
        model.beta.copy_(torch.log(torch.expm1(torch.linspace(.35, 1.65, products)))[:, None])
        model.price_kappa.fill_(math.log(math.expm1(kappa)))
        model.w_dsp.copy_(.08 + .04 * torch.rand(products, generator=generator))
        model.w_mlr.fill_(.03)
        model.rho_c.copy_(torch.tensor([.16, -.03, .10]))
        size = torch.arange(1, nmax + 1, dtype=torch.float64)
        model.rho_0_free.copy_(.06 * (size - 2.5).square())
        embedding = torch.randn(products, rank, generator=generator)
        embedding -= embedding.mean(0)
        model.phi.copy_(strength * .22 * embedding)
        model.cat_of.copy_(cat)
        model.project_context_gauges()
    items, rows, row_trip, row_cat = [], [], [], []
    for trip in range(contexts):
        for category in range(categories):
            offered = torch.nonzero(cat == category).flatten().tolist()
            items.extend(offered)
            rows.extend([len(row_trip)] * len(offered))
            row_trip.append(trip)
            row_cat.append(category)
    ix = RaggedIndex(np.asarray(items), np.asarray(rows), np.asarray(row_trip),
                     np.asarray(row_cat), contexts)
    dlp = .15 * torch.randn(len(items), generator=generator)
    ctx = {"dlp": dlp, "dlp_bar": torch.zeros(contexts).index_add(0, ix.item_trip, dlp) / products,
           "disp": (torch.rand(len(items), generator=generator) < .2).double(),
           "mail": (torch.rand(len(items), generator=generator) < .15).double(),
           "week": ix.item_trip % 53, "store": ix.item_trip % 2}
    model.ctx, model.house = ctx, torch.arange(contexts)
    model._poly_degree_native = model._esp_native = model._esp_log_blocked = True
    baskets = [s for n in range(1, nmax + 1)
               for s in itertools.combinations(range(products), n)]
    membership = torch.zeros(len(baskets), products)
    for row, basket in enumerate(baskets):
        membership[row, list(basket)] = 1
    return model, ix, membership, baskets


def exact_energy(model, ix, membership, context=None):
    """Independent dense off-diagonal energy, retaining actual b_at utility features."""
    values = model.b_at(ix.item, ix.item_trip, model.ctx if context is None else context)
    utilities = torch.zeros(ix.B, model.J, dtype=values.dtype).index_put(
        (ix.item_trip, ix.item), values)
    kernel = model.phi @ model.phi.T
    kernel = kernel - torch.diag(torch.diag(kernel))
    cat = model.cat_of
    category_pairs = (cat[:, None] == cat[None, :]).to(values.dtype)
    category_pairs.fill_diagonal_(0)
    kernel = kernel - category_pairs * model.rho_c[cat][:, None]
    pair = .5 * ((membership @ kernel) * membership).sum(1)
    sizes = membership.sum(1).long()
    return utilities @ membership.T + pair[None, :] - model.rho_0()[sizes][None, :]


def exact_size_probability(energy, membership, nmax):
    probability = energy.softmax(-1)
    sizes = membership.sum(1).long() - 1
    return torch.zeros(energy.shape[0], nmax).scatter_add(
        1, sizes.expand(energy.shape[0], -1), probability)


def deterministic_audit(seed, products=10, contexts=4, nmax=5):
    results = []
    for strength in (0.0, 0.8):
        for kappa in (1.0, 2.4):
            model, ix, y, baskets = make_world(seed, products, contexts, nmax, strength, kappa)
            energy = exact_energy(model, ix, y)
            p = energy.softmax(-1)
            pi = p @ y
            pairs = torch.einsum("bs,si,sj->bij", p, y, y)
            covariance = pairs - pi[:, :, None] * pi[:, None, :]
            metrics = {}
            # Compare energy() for every supported basket, including n=1 and n=nmax.
            old_house = model.house
            line_item = torch.tensor([j for s in baskets for j in s], dtype=torch.long)
            line_trip = torch.repeat_interleave(torch.arange(len(baskets)), y.sum(1).long())
            native_energy = []
            for b in range(ix.B):
                slot = torch.empty(model.J, dtype=torch.long)
                selected = ix.item_trip == b
                slot[ix.item[selected]] = torch.nonzero(selected).flatten()
                line_context = {key: value[b].expand(len(baskets)) if key == "dlp_bar"
                                else value[slot[line_item]] for key, value in model.ctx.items()}
                model.house = old_house[b].expand(len(baskets))
                native_energy.append(model.energy(line_item, line_trip, model.cat_of[line_item],
                                                  len(baskets), line_context))
            model.house = old_house
            metrics["energy_max_error"] = float((torch.stack(native_energy) - energy).detach().abs().max())
            # Actual price-context transformation and analytic Jacobian, all products.
            analytic, finite = [], []
            eps = 1e-5
            for k in range(model.J):
                delta = (ix.item == k).double() * eps
                plus = exact_energy(model, ix, y, changed_price_context(model.ctx, ix.item_trip, delta))
                minus = exact_energy(model, ix, y, changed_price_context(model.ctx, ix.item_trip, -delta))
                finite.append(((plus.softmax(-1) - minus.softmax(-1)) @ y) / (2 * eps))
            for b in range(ix.B):
                g = model.price_coefficients(torch.arange(model.J), torch.full((model.J,), b))
                analytic.append(covariance[b] @ price_jacobian(g, kappa))
            metrics["price_jacobian_max_error"] = float(
                (torch.stack(analytic) - torch.stack(finite, -1)).detach().abs().max())
            # Finite price changes, including a bundle, use full delta utilities.
            action = torch.where(ix.item % 3 == 0, math.log(.8), 0.0)
            changed = exact_energy(model, ix, y, changed_price_context(model.ctx, ix.item_trip, action))
            reweighted = (p.log() + changed - energy).softmax(-1)
            metrics["action_reweight_max_error"] = float((reweighted - changed.softmax(-1)).detach().abs().max())
            tilt_errors = []
            for d in (-.4, 0.0, .3):
                updated, log_ratio = single_utility_incidence(pi, pairs[..., 1], 1, d)
                changed = energy + d * y[:, 1]
                tilt_errors.extend([float((updated - changed.softmax(-1) @ y).detach().abs().max()),
                                    float((log_ratio - changed.logsumexp(-1) + energy.logsumexp(-1)).detach().abs().max())])
            metrics["single_utility_max_error"] = max(tilt_errors)
            if strength == 0:
                log_size = differentiable_log_size_beta0(model, ix)
                metrics["dp_logz_max_error"] = float((log_size.logsumexp(-1) - energy.logsumexp(-1)).detach().abs().max())
                metrics["dp_size_max_error"] = float((log_size.softmax(-1) - exact_size_probability(energy, y, nmax)).detach().abs().max())
                response = additive_uniform_price_response(model, ix, log_size.softmax(-1))
                size = y.sum(1)
                exact_slopes = []
                for b in range(ix.B):
                    g = model.price_coefficients(torch.arange(model.J), torch.full((model.J,), b))
                    G = y @ g
                    exact_slopes.append(-((p[b] * size * G).sum() - (p[b] * size).sum() * (p[b] * G).sum()))
                exact_elasticity = torch.stack(exact_slopes).mean() / (p @ size).mean()
                metrics["elasticity_error"] = float((response.elasticity - exact_elasticity).detach().abs())
                # Validate actual penalty gradients through the native first adjoints.
                parameters = [model.lam, model.beta, model.gamma, model.rho_c, model.rho_0_free]
                numerical_gradient = torch.autograd.grad((response.elasticity + .2).square(), parameters)
                oracle_gradient = torch.autograd.grad((exact_elasticity + .2).square(), parameters)
                metrics["elasticity_penalty_gradient_max_error"] = max(
                    float((a - b).abs().max()) for a, b in zip(numerical_gradient, oracle_gradient))
                metrics["difference_step"] = response.step
                metrics["difference_discrepancy"] = float(response.second_fourth_discrepancy.detach().max())
            else:
                # Independent scalar and score audit of the H-S/native/Smolyak path.
                model.quad = smolyak_rule(model, model.Kz, model.Kz + 5)
                model.quad_a = None
                logz = model.log_Z(ix, drop_empty=True)
                metrics["hs_logz_max_error"] = float((logz - energy.logsumexp(-1)).detach().abs().max())
                parameters = [model.lam, model.phi, model.rho_c, model.rho_0_free]
                actual = torch.autograd.grad(logz.sum(), parameters)
                oracle = torch.autograd.grad(energy.logsumexp(-1).sum(), parameters)
                metrics["hs_score_max_error"] = max(float((a - b).abs().max()) for a, b in zip(actual, oracle))
            thresholds = {key: (2e-5 if key.startswith("hs_") else 2e-7)
                          for key in metrics if "error" in key}
            gates = {key: math.isfinite(metrics[key]) and metrics[key] <= tolerance
                     for key, tolerance in thresholds.items()}
            results.append({"strength": strength, "kappa": kappa, "support": len(baskets),
                            "metrics": metrics, "thresholds": thresholds, "gates": gates,
                            "passed": all(gates.values())})
            print(f"[foundations] strength={strength} kappa={kappa} identities_passed={all(gates.values())}", flush=True)
    return results


def ess_adversary():
    """A one-draw-dominated tail band used to pass the fractional gate by definition."""
    rows = []
    for draws in (3, 4, 5, 8):
        feature = np.zeros((2, draws, 1)); feature[:, 0, 0] = 1000
        bank = StratifiedNaturalBank(np.zeros((2, 1)), feature, np.zeros(2, dtype=int),
                                    np.zeros((2, draws), dtype=int), sparse.csr_matrix((2, 0)),
                                    sparse.csr_matrix((2 * draws, 0)),
                                    np.full((2, draws), -math.log(draws)), 2)
        vector = np.r_[1.0, np.zeros(bank.size_width)]
        report = evaluation_summary(vector, bank, np.zeros(draws, dtype=int))
        rows.append({"draws": draws, **report,
                     "rejected": not within_band_ess_passes(report, .2, 2.0)})
    return {"cases": rows, "passed": all(row["rejected"] for row in rows)}


@torch.no_grad()
def tail_ess_recovery(seed):
    """Increase band coverage on an enumerated interacting law, then independently audit.

    The pilot chooses the budget, never the retained production draws. The fixed
    independent bank must meet the actual gates as well; larger D alone is no proof.
    """
    model, ix, membership, _ = make_world(seed, contexts=16, nmax=8, strength=2.5)
    energy = exact_energy(model, ix, membership).numpy()
    model.phi.zero_()
    base = exact_energy(model, ix, membership).numpy()
    increment = energy - base
    sizes = membership.sum(1).numpy().astype(int)
    bands = [(1, 2), (3, 4), (5, 6), (7, 8)]
    rng = np.random.default_rng(seed + 271)

    def bank(draws, generator):
        features, weights, draw_sizes, labels = [], [], [], []
        for band, ((lo, hi), count) in enumerate(zip(bands, draws)):
            eligible = np.flatnonzero((sizes >= lo) & (sizes <= hi))
            logits = torch.as_tensor(base[:, eligible])
            conditional = logits.softmax(-1).numpy()
            mass = (logits.logsumexp(-1) - torch.as_tensor(base).logsumexp(-1)).numpy()
            chosen = np.stack([generator.choice(eligible, count, p=p) for p in conditional])
            features.append(np.take_along_axis(increment, chosen, axis=1))
            weights.append(np.broadcast_to(mass[:, None] - math.log(count), chosen.shape))
            draw_sizes.append(sizes[chosen] - 1)
            labels.extend([band] * count)
        features = np.concatenate(features, axis=1)
        size = np.concatenate(draw_sizes, axis=1)
        value = StratifiedNaturalBank(np.zeros((ix.B, 1)), features[..., None],
            np.zeros(ix.B, dtype=int), size, sparse.csr_matrix((ix.B, 0)),
            sparse.csr_matrix((features.size, 0)), np.concatenate(weights, axis=1), 8)
        vector = np.r_[1., np.zeros(8)]
        summary = evaluation_summary(vector, value, np.asarray(labels))
        log_ratio = torch.logsumexp(torch.as_tensor(features + value.log_draw_weight), -1).numpy()
        oracle = (torch.as_tensor(energy).logsumexp(-1) - torch.as_tensor(base).logsumexp(-1)).numpy()
        return summary, log_ratio - oracle

    legacy, _ = bank([16, 8, 4, 3], rng)
    draws = [16, 16, 16, 16]
    history = []
    for _ in range(7):
        pilot, _ = bank(draws, rng)
        history.append({"draws": list(draws), "minimum_ess": pilot["minimum_within_band_ess"]})
        if within_band_ess_passes(pilot, .2, 8.):
            break
        draws = [2 * n for n in draws]
    # Auditing uses a larger fixed precision budget, independent of oracle error.
    final_draws = [max(n, 2048) for n in draws]
    final, error = bank(final_draws, np.random.default_rng(seed + 272))
    passed = within_band_ess_passes(final, .2, 8.) and float(abs(error).max()) < .08
    return {"legacy_minimum_ess": legacy["minimum_within_band_ess"],
            "legacy_passes_new_gate": within_band_ess_passes(legacy, .2, 2.),
            "pilot": history, "independent_draws_per_band": final_draws,
            "independent_minimum_ess": final["minimum_within_band_ess"],
            "maximum_log_normalizer_ratio_error": float(abs(error).max()),
            "passed": bool(passed),
            "scope": "enumerated small-world proposal/ESS recovery; original-data banks still require their own gates"}


@torch.no_grad()
def sampling_audit(seed, draws=4000, particles=128, replicates=24):
    rows = []
    for strength in (0.0, 0.8):
        model, ix, y, baskets = make_world(seed, contexts=2, strength=strength)
        exact = exact_energy(model, ix, y)
        probability = exact.softmax(-1)
        statistics = torch.cat((y, y.sum(1, keepdim=True),
                                (y.sum(1, keepdim=True) >= 4).double()), dim=1)
        expected = probability @ statistics
        if strength == 0:
            states = conditional_slots_repeated(model, ix, torch.zeros(ix.B, model.Kz),
                                               0.0, draws, torch.Generator().manual_seed(seed + 1))
            samples = states_to_statistics(model, ix, states)
            # Simultaneous Hoeffding intervals for independent exact additive draws.
            scales = torch.ones(statistics.shape[1]); scales[-2] = model.nmax - 1
            tolerance = scales * math.sqrt(math.log(2 * expected.numel() / .001) / (2 * draws))
            errors = (samples.mean(0) - expected).abs()
            rows.append({"strength": strength, "method": "exact_additive_reverse_sampling",
                         "draws": draws, "maximum_scaled_error": float((errors / scales).max()),
                         "simultaneous_failure_probability": .001,
                         "passed": bool((errors <= tolerance).all())})
        else:
            # SMC particles are dependent. Use independent runs as uncertainty units.
            means, ratios = [], []
            for replicate in range(replicates):
                smc = annealed_smc_logz(model, ix, torch.linspace(0, 1, 13), particles,
                                       generator=torch.Generator().manual_seed(seed + 100 + replicate))
                means.append(states_to_statistics(model, ix, smc.states).mean(0))
                ratios.append((smc.log_z - exact.logsumexp(-1)).exp())
                if (replicate + 1) % 8 == 0:
                    print(f"[foundations] SMC independent runs {replicate + 1}/{replicates}", flush=True)
            values, ratios = torch.stack(means), torch.stack(ratios)
            errors = (values.mean(0) - expected).abs()
            se = values.std(0) / math.sqrt(replicates)
            # Finite-particle bias is explicitly allowed and reported, not called exactness.
            tolerance = 5 * se + .03
            ratio_error = (ratios.mean(0) - 1).abs()
            ratio_tolerance = 5 * ratios.std(0) / math.sqrt(replicates) + .01
            rows.append({"strength": strength, "method": "independently_replicated_smc",
                         "particles": particles, "replicates": replicates,
                         "maximum_moment_error": float(errors.max()),
                         "maximum_logz_ratio_error": float(ratio_error.max()),
                         "normalizer_ratio_mean": ratios.mean(0).tolist(),
                         "normalizer_ratio_se": (ratios.std(0) / math.sqrt(replicates)).tolist(),
                         "passed": bool((errors <= tolerance).all() and (ratio_error <= ratio_tolerance).all())})
    return rows


def states_to_statistics(model, ix, states):
    samples = torch.zeros(len(states), ix.B, model.J + 2)
    for r, population in enumerate(states):
        for b, slots in enumerate(population):
            items = ix.item[slots]
            if not 1 <= len(items) <= model.nmax or len(torch.unique(items)) != len(items):
                raise AssertionError("sampler produced an unsupported basket")
            samples[r, b, items] = 1
            samples[r, b, -2] = len(items)
            samples[r, b, -1] = len(items) >= 4
    return samples


@torch.no_grad()
def randomized_opportunity_audit(seed, households=400, days=120):
    """Known randomized prices, customer heterogeneity and explicit no-purchase days."""
    model, ix, y, _ = make_world(seed, contexts=3, strength=.8)
    truth = []
    price = torch.linspace(1.5, 6, model.J)
    for action in (0.0, math.log(.8)):
        delta = (ix.item % 3 == 0).double() * action
        e = exact_energy(model, ix, y, changed_price_context(model.ctx, ix.item_trip, delta))
        truth.append(e.softmax(-1).numpy())
    rng = np.random.default_rng(seed + 9)
    segment = rng.integers(0, ix.B, households)
    frailty = rng.normal(0, .5, households)
    h = np.repeat(np.arange(households), days)
    action = rng.integers(0, 2, len(h))
    arrival = 1 / (1 + np.exp(-(-1.4 + .25 * segment[h] + frailty[h] + .3 * action)))
    purchase = rng.random(len(h)) < arrival
    outcome = np.zeros((len(h), 4))
    outcome[:, 0] = purchase
    oracle = np.zeros((households, 2, 4))
    sizes = y.sum(1).numpy()
    product = y[:, 0].numpy()
    for a in range(2):
        actual_price = price.numpy().copy(); actual_price[np.arange(model.J) % 3 == 0] *= (1 if a == 0 else .8)
        revenue = y.numpy() @ actual_price
        conditional = np.stack([np.ones(len(y)), product, sizes, revenue], axis=1)
        for s in range(ix.B):
            selected = np.flatnonzero(purchase & (action == a) & (segment[h] == s))
            basket = rng.choice(len(y), len(selected), p=truth[a][s])
            outcome[selected] = conditional[basket]
            hh = np.flatnonzero(segment == s)
            v = 1 / (1 + np.exp(-(-1.4 + .25 * s + frailty[hh] + .3 * a)))
            oracle[hh, a] = v[:, None] * (truth[a][s] @ conditional)[None, :]
    effect = (oracle[:, 1] - oracle[:, 0]).mean(0)
    # Horvitz-Thompson contrast exploits the known 1/2 assignment probability.
    contrast = 2 * (2 * action - 1)[:, None] * outcome
    rows = []
    critical = float(norm.ppf(1 - .001 / (2 * outcome.shape[1])))
    for j, name in enumerate(("purchase", "item_0", "distinct_products", "distinct_product_revenue")):
        se = household_cluster_se(contrast[:, j], h)
        estimate = float(contrast[:, j].mean())
        rows.append({"outcome": name, "oracle_effect": float(effect[j]),
                     "randomized_estimate": estimate, "household_cluster_se": se,
                     "interval": [estimate-critical*se, estimate+critical*se],
                     "covered": bool(abs(estimate-effect[j]) <= critical*se)})
    return {"households": households, "opportunities": len(h),
            "purchases": int(purchase.sum()), "no_purchases": int((~purchase).sum()),
            "simultaneous_nominal_confidence": .999, "results": rows,
            "passed": all(row["covered"] for row in rows)}


def price_training_audit(seed, steps=100, trips_per_context=1000):
    """Fit actual native-DP utilities/price loadings to randomized synthetic checkouts.

    An oracle aggregate elasticity is supplied as an explicitly known calibration
    moment to isolate old versus corrected penalty behavior. It is not estimated from
    the locked validation baskets. Model selection uses validation likelihood only.
    """
    truth, ix, y, _ = make_world(seed, contexts=18, strength=0, kappa=2.4)
    truth.house = torch.arange(ix.B) % 3
    generator = torch.Generator().manual_seed(seed + 400)
    # Wider independent price variation supplies identification within each household.
    truth.ctx["dlp"] = .35 * torch.randn(ix.item.shape, generator=generator)
    truth.ctx["dlp_bar"] = torch.zeros(ix.B).index_add(0, ix.item_trip, truth.ctx["dlp"]) / truth.J
    with torch.no_grad():
        probability = exact_energy(truth, ix, y).softmax(-1)
        size = y.sum(1)
        true_slope = []
        for b in range(ix.B):
            g = truth.price_coefficients(torch.arange(truth.J), torch.full((truth.J,), b))
            G = y @ g
            true_slope.append(-((probability[b]*size*G).sum() - (probability[b]*size).sum()*(probability[b]*G).sum()))
        target = torch.stack(true_slope).mean() / (probability @ size).mean()
        train_draws = torch.multinomial(probability, trips_per_context, replacement=True, generator=generator)
        validation_draws = torch.multinomial(probability, trips_per_context, replacement=True, generator=generator)
        train_count = torch.zeros_like(probability).scatter_add(1, train_draws, torch.ones_like(train_draws, dtype=torch.float64))
        valid_count = torch.zeros_like(probability).scatter_add(1, validation_draws, torch.ones_like(validation_draws, dtype=torch.float64))
    initial = copy.deepcopy(truth)
    with torch.no_grad():
        initial.lam.add_(.3 * torch.randn(truth.J, generator=generator))
        initial.beta.add_(.65)
    rows = []
    for method in ("likelihood_only", "legacy_proxy", "corrected_response"):
        model = copy.deepcopy(initial)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.lam.requires_grad_(True); model.beta.requires_grad_(True)
        optimizer = torch.optim.Adam([model.lam, model.beta], lr=.04)
        best, best_iteration, best_state = -float("inf"), 0, None
        with torch.no_grad():
            initial_logp = exact_energy(model, ix, y).log_softmax(-1)
            initial_kl = float((probability * (probability.log()-initial_logp)).sum(-1).mean())
        for iteration in range(steps + 1):
            log_size = differentiable_log_size_beta0(model, ix)
            energy = exact_energy(model, ix, y)
            logp = energy - log_size.logsumexp(-1)[:, None]
            if iteration % 10 == 0 or iteration == steps:
                validation = float(((valid_count * logp).sum()/valid_count.sum()).detach())
                if validation > best:
                    best, best_iteration = validation, iteration
                    best_state = copy.deepcopy(model.state_dict())
            if iteration == steps:
                break
            loss = -(train_count * logp).sum()/train_count.sum()
            if method != "likelihood_only":
                size_p = log_size.softmax(-1)
                if method == "corrected_response":
                    elasticity = additive_uniform_price_response(model, ix, size_p).elasticity
                else:
                    grid = torch.arange(1, model.nmax+1, dtype=torch.float64)
                    mean = size_p @ grid
                    variance = size_p @ grid.square() - mean.square()
                    g = model.price_coefficients(ix.item, ix.item_trip).mean()
                    elasticity = -g * variance.mean()/mean.mean()
                loss = loss + 20 * (elasticity-target).square()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if not all(bool(torch.isfinite(parameter.grad).all()) for parameter in (model.lam, model.beta)):
                raise FloatingPointError("nonfinite native-DP price-training gradient")
            optimizer.step()
        model.load_state_dict(best_state)
        with torch.no_grad():
            fitted = exact_energy(model, ix, y).log_softmax(-1)
            kl = float((probability * (probability.log()-fitted)).sum(-1).mean())
            actual_response = additive_uniform_price_response(model, ix).elasticity
            rows.append({"method": method, "selected_iteration": best_iteration,
                         "initial_oracle_kl": initial_kl, "final_oracle_kl": kl,
                         "oracle_calibration_target": float(target),
                         "actual_elasticity": float(actual_response),
                         "absolute_calibration_error": float((actual_response-target).abs()),
                         "price_coefficient_rmse": float((model.price_coefficients(ix.item,ix.item_trip)
                                                         -truth.price_coefficients(ix.item,ix.item_trip)).square().mean().sqrt()),
                         "validation_loglik": best})
        print(f"[foundations] price fit {method}: KL={kl:.6g}, actual elasticity={float(actual_response):.6g}", flush=True)
    corrected = rows[-1]
    # Selection/estimation is stochastic: gate improvement over the common initial fit;
    # compare corrected-versus-proxy performance descriptively, without cherry-picking.
    return {"training_trips": ix.B*trips_per_context, "validation_trips": ix.B*trips_per_context,
            "steps": steps, "oracle_target_supplied": True, "fits": rows,
            "passed": bool(np.isfinite(corrected["final_oracle_kl"])
                           and corrected["final_oracle_kl"] < corrected["initial_oracle_kl"])}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("smoke", "full"), default="full")
    parser.add_argument("--seed", type=int, default=90701)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    full = args.profile == "full"
    seeds = [args.seed + offset for offset in ((0, 1000, 2000) if full else (0,))]
    result = {"schema": 1, "profile": args.profile, "seeds": seeds,
              "status": "running", "deterministic": [], "sampling": [], "randomized": [],
              "price_training": [], "tail_ess_recovery": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        print(f"[foundations] exact identities and penalty gradients seed={seed}", flush=True)
        result["deterministic"].extend(deterministic_audit(seed))
        args.output.write_text(strict_json_dumps(result))
        print(f"[foundations] actual reverse and SMC samplers seed={seed}", flush=True)
        result["sampling"].extend(sampling_audit(seed, draws=12000 if full else 1500,
                                                particles=256 if full else 64,
                                                replicates=48 if full else 8))
        print(f"[foundations] randomized price opportunities seed={seed}", flush=True)
        result["randomized"].append(randomized_opportunity_audit(
            seed, households=1200 if full else 200, days=180 if full else 60))
        print(f"[foundations] native-DP price calibration experiment seed={seed}", flush=True)
        result["price_training"].append(price_training_audit(
            seed, steps=250 if full else 40, trips_per_context=3000 if full else 300))
        result["tail_ess_recovery"].append(tail_ess_recovery(seed))
        args.output.write_text(strict_json_dumps(result))
    result["ess_adversary"] = ess_adversary()
    result["passed"] = (all(row["passed"] for key in ("deterministic", "sampling", "randomized", "price_training", "tail_ess_recovery")
                            for row in result[key]) and result["ess_adversary"]["passed"])
    result["status"] = "passed" if result["passed"] else "failed"
    result["runtime_seconds"] = time.perf_counter() - started
    result["interpretation"] = "Finite implementation audits against known synthetic truth, not a proof of universal theory or real-world causality."
    args.output.write_text(strict_json_dumps(result))
    print(f"[foundations] {result['status']}: {args.output}", flush=True)
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
