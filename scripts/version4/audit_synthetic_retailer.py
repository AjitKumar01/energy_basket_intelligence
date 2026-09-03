#!/usr/bin/env python3
"""End-to-end synthetic retailer audit with known ground truth.

The experiment is intentionally smaller than the dunnhumby fit so every nonempty
basket through ``nmax`` can be enumerated.  This makes the basket likelihood,
generation probabilities, counterfactuals and policy oracle exact.  The synthetic
retailer additionally contains the pieces absent from the real transaction-only fit:

* customer-day purchase/no-purchase opportunities;
* store choice;
* shifted-negative-binomial quantities;
* randomized, logged promotion actions;
* product costs and a finite promotion budget.

Synthetic evidence is an implementation/recovery audit, not evidence that the same
causal effects hold in dunnhumby or at another retailer.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score


torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class Config:
    customers: int = 240
    products: int = 20
    categories: int = 5
    segments: int = 3
    stores: int = 3
    rank: int = 3
    nmax: int = 6
    days: int = 180
    train_days: int = 120
    validation_days: int = 30
    additive_steps: int = 350
    interaction_steps: int = 500
    quantity_steps: int = 500
    eval_every: int = 10
    patience: int = 12
    seed: int = 73021
    threads: int = 8
    world: str = "well_specified"


def enumerate_support(products: int, nmax: int, categories: int) -> dict:
    baskets = [combination for size in range(1, nmax + 1)
               for combination in itertools.combinations(range(products), size)]
    membership = torch.zeros((len(baskets), products))
    for row, basket in enumerate(baskets):
        membership[row, list(basket)] = 1.0
    sizes = membership.sum(1).long()
    item_category = torch.arange(products) % categories
    category_indicator = torch.nn.functional.one_hot(
        item_category, categories).double()
    category_count = membership @ category_indicator
    category_pairs = category_count * (category_count - 1.0) * 0.5
    return {
        "baskets": baskets,
        "membership": membership,
        "sizes": sizes,
        "item_category": item_category,
        "category_count": category_count,
        "category_pairs": category_pairs,
    }


def gram_energy(membership: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    total = membership @ phi
    diagonal = membership @ phi.square().sum(1)
    return 0.5 * (total.square().sum(1) - diagonal)


def paired_summary(first, second) -> dict:
    delta = np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)
    standard_error = float(delta.std(ddof=1) / math.sqrt(len(delta)))
    mean = float(delta.mean())
    return {
        "mean": mean,
        "standard_deviation": float(delta.std(ddof=1)),
        "standard_error": standard_error,
        "interval_95": [mean - 1.96 * standard_error,
                        mean + 1.96 * standard_error],
    }


def summarize(values) -> dict:
    value = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(value.mean()),
        "standard_deviation": float(value.std(ddof=1)),
        "standard_error": float(value.std(ddof=1) / math.sqrt(len(value))),
    }


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def make_truth(config: Config, support: dict) -> dict:
    rng = np.random.default_rng(config.seed)
    products = config.products
    segments = config.segments
    rank = config.rank

    mission = np.arange(products) % rank
    directions = np.eye(rank) - np.ones((rank, rank)) / rank
    raw_phi = directions[mission] + 0.08 * rng.normal(size=(products, rank))
    raw_phi -= raw_phi.mean(0, keepdims=True)
    phi = 0.72 * raw_phi

    base = -1.48 + 0.30 * rng.normal(size=products)
    segment = np.empty((segments, products), dtype=np.float64)
    for group in range(segments):
        segment[group] = np.where(mission == group % rank, 0.48, -0.16)
        segment[group] += 0.08 * rng.normal(size=products)
    segment -= segment.mean(0, keepdims=True)
    elasticity = np.clip(1.15 + 0.20 * rng.normal(size=products), 0.65, 1.65)

    rho_size = np.asarray([0.0, -0.12, -0.08, 0.12, 0.52, 1.05])[:config.nmax]
    rho_category = np.asarray([0.20, 0.16, 0.22, 0.18, 0.15])[:config.categories]
    base_price = np.exp(rng.uniform(math.log(1.5), math.log(8.0), size=products))
    unit_cost = base_price * rng.uniform(0.48, 0.68, size=products)

    # No promotion plus 10% and 20% discounts for each cross-category mission bundle.
    actions = [{"name": "none", "mission": None, "discount": 0.0,
                "products": []}]
    for group in range(rank):
        bundle = np.flatnonzero(mission == group).astype(int).tolist()
        for discount in (0.10, 0.20):
            actions.append({
                "name": f"mission_{group}_{int(100 * discount)}pct",
                "mission": group,
                "discount": discount,
                "products": bundle,
            })
    action_discount = np.zeros((len(actions), products), dtype=np.float64)
    for action_index, action in enumerate(actions):
        action_discount[action_index, action["products"]] = action["discount"]
    log_price_ratio = np.log1p(-action_discount)

    context_segment = np.repeat(np.arange(segments), len(actions))
    context_action = np.tile(np.arange(len(actions)), segments)
    b = (base[None, :] + segment[context_segment]
         - elasticity[None, :] * log_price_ratio[context_action])
    membership = support["membership"]
    sizes = support["sizes"]
    category_pairs = support["category_pairs"]
    energy = torch.as_tensor(b) @ membership.T
    energy -= torch.as_tensor(rho_size)[sizes - 1][None, :]
    energy -= (category_pairs @ torch.as_tensor(rho_category))[None, :]
    energy += gram_energy(membership, torch.as_tensor(phi))[None, :]
    if config.world == "misspecified":
        # Local pair and triple terms cannot all be represented by the fitted rank-three
        # PSD Gram kernel plus category/size potentials.  This is a deliberate robustness
        # world, not a second data-generating copy of the fitted model.
        # Select the triple relative to catalog size so the same stress test is valid for
        # both the full and smoke profiles.
        triple = (0, config.products // 3, 2 * config.products // 3)
        local = (0.45 * membership[:, 0] * membership[:, 1]
                 - 0.35 * membership[:, 2] * membership[:, 3]
                 + 0.40 * membership[:, triple[0]]
                 * membership[:, triple[1]] * membership[:, triple[2]])
        energy += local[None, :]
    logp = energy - torch.logsumexp(energy, dim=1, keepdim=True)

    arrival_segment = np.asarray([-2.75, -2.45, -2.20])[:segments]
    action_lift = np.zeros((segments, len(actions)), dtype=np.float64)
    for group in range(segments):
        for action_index, action in enumerate(actions):
            if action_index == 0:
                continue
            alignment = 1.0 if action["mission"] == group % rank else 0.55
            action_lift[group, action_index] = alignment * (
                1.15 * action["discount"] + 0.12)
    household_frailty = rng.normal(0.0, 0.42, size=config.customers)
    store_probability = np.asarray([
        [0.68, 0.22, 0.10],
        [0.20, 0.62, 0.18],
        [0.14, 0.24, 0.62],
    ])[:segments, :config.stores]
    store_probability /= store_probability.sum(1, keepdims=True)

    quantity_item = -1.15 + 0.22 * rng.normal(size=products)
    quantity_segment = np.asarray([-0.10, 0.00, 0.14])[:segments]
    quantity_discount = 2.15
    quantity_dispersion = 2.6
    return {
        "mission": mission,
        "phi": phi,
        "base": base,
        "segment": segment,
        "elasticity": elasticity,
        "rho_size": rho_size,
        "rho_category": rho_category,
        "base_price": base_price,
        "unit_cost": unit_cost,
        "actions": actions,
        "action_discount": action_discount,
        "log_price_ratio": log_price_ratio,
        "context_segment": context_segment,
        "context_action": context_action,
        "logp": logp.detach().numpy(),
        "arrival_segment": arrival_segment,
        "action_lift": action_lift,
        "household_frailty": household_frailty,
        "store_probability": store_probability,
        "quantity_item": quantity_item,
        "quantity_segment": quantity_segment,
        "quantity_discount": quantity_discount,
        "quantity_discount_quadratic": 2.5 if config.world == "misspecified" else 0.0,
        "quantity_dispersion": quantity_dispersion,
        "arrival_nonlinear": 0.22 if config.world == "misspecified" else 0.0,
    }


def simulate_retailer(config: Config, support: dict, truth: dict) -> dict:
    rng = np.random.default_rng(config.seed + 1)
    customer_segment = np.repeat(
        np.arange(config.segments), math.ceil(config.customers / config.segments)
    )[:config.customers]
    rng.shuffle(customer_segment)
    promotion_actions = len(truth["actions"]) - 1
    action_probability = np.asarray(
        [0.40] + [0.60 / promotion_actions] * promotion_actions)
    # Randomize logged offers at the household-day level.  Segment-day assignment would
    # create only ``days`` independent treatment clusters per segment while pretending
    # that all household rows were independent, which is inadequate for action ranking.
    action_by_household_day = rng.choice(
        len(truth["actions"]), size=(config.customers, config.days),
        p=action_probability)

    household, day, segment, action, recency, sin_day, cos_day = [], [], [], [], [], [], []
    purchase, oracle_purchase_probability, store = [], [], []
    last_purchase = np.full(config.customers, -14, dtype=np.int64)
    for current_day in range(config.days):
        seasonal_sin = math.sin(2.0 * math.pi * current_day / 30.0)
        seasonal_cos = math.cos(2.0 * math.pi * current_day / 30.0)
        for h in range(config.customers):
            group = int(customer_segment[h])
            chosen_action = int(action_by_household_day[h, current_day])
            gap = min(current_day - int(last_purchase[h]), 30)
            logit = (truth["arrival_segment"][group]
                     + truth["household_frailty"][h]
                     + 0.32 * seasonal_sin - 0.10 * seasonal_cos
                     + 0.030 * gap
                     + truth["action_lift"][group, chosen_action])
            logit += truth["arrival_nonlinear"] * (
                seasonal_sin * gap / 30.0 + (gap / 30.0) ** 2)
            probability = 1.0 / (1.0 + math.exp(-logit))
            bought = int(rng.random() < probability)
            household.append(h); day.append(current_day); segment.append(group)
            action.append(chosen_action); recency.append(gap)
            sin_day.append(seasonal_sin); cos_day.append(seasonal_cos)
            purchase.append(bought); oracle_purchase_probability.append(probability)
            if bought:
                last_purchase[h] = current_day
                store.append(int(rng.choice(
                    config.stores, p=truth["store_probability"][group])))
            else:
                store.append(-1)

    opportunity = {
        "household": np.asarray(household, dtype=np.int64),
        "day": np.asarray(day, dtype=np.int64),
        "segment": np.asarray(segment, dtype=np.int64),
        "action": np.asarray(action, dtype=np.int64),
        "recency": np.asarray(recency, dtype=np.float64),
        "sin_day": np.asarray(sin_day, dtype=np.float64),
        "cos_day": np.asarray(cos_day, dtype=np.float64),
        "purchase": np.asarray(purchase, dtype=np.int8),
        "oracle_purchase_probability": np.asarray(
            oracle_purchase_probability, dtype=np.float64),
        "store": np.asarray(store, dtype=np.int64),
    }
    positive = np.flatnonzero(opportunity["purchase"] == 1)
    context = (opportunity["segment"][positive] * len(truth["actions"])
               + opportunity["action"][positive])
    subset = np.empty(len(positive), dtype=np.int64)
    for context_value in np.unique(context):
        selected = np.flatnonzero(context == context_value)
        subset[selected] = rng.choice(
            len(support["baskets"]), size=len(selected),
            p=np.exp(truth["logp"][context_value]))

    line_trip, line_item, line_quantity = [], [], []
    for trip, (opportunity_row, subset_value) in enumerate(zip(positive, subset)):
        group = int(opportunity["segment"][opportunity_row])
        chosen_action = int(opportunity["action"][opportunity_row])
        for item in support["baskets"][int(subset_value)]:
            discount = truth["action_discount"][chosen_action, item]
            extra_mean = math.exp(
                truth["quantity_item"][item]
                + truth["quantity_segment"][group]
                + truth["quantity_discount"] * discount
                + truth["quantity_discount_quadratic"] * discount ** 2)
            dispersion = truth["quantity_dispersion"]
            probability = dispersion / (dispersion + extra_mean)
            extra = int(rng.negative_binomial(dispersion, probability))
            line_trip.append(trip); line_item.append(item)
            line_quantity.append(1 + extra)
    return {
        "opportunity": opportunity,
        "trip_opportunity": positive,
        "trip_context": context,
        "trip_subset": subset,
        "line_trip": np.asarray(line_trip, dtype=np.int64),
        "line_item": np.asarray(line_item, dtype=np.int64),
        "line_quantity": np.asarray(line_quantity, dtype=np.int64),
        "customer_segment": customer_segment,
        "action_by_household_day": action_by_household_day,
    }


def split_mask(day: np.ndarray, config: Config, split: str) -> np.ndarray:
    validation_start = config.train_days
    test_start = config.train_days + config.validation_days
    if split == "train":
        return day < validation_start
    if split == "validation":
        return (day >= validation_start) & (day < test_start)
    if split == "test":
        return day >= test_start
    raise ValueError(split)


def arrival_features(opportunity: dict, config: Config,
                     action_override: int | None = None) -> sparse.csr_matrix:
    rows = len(opportunity["household"])
    household = sparse.csr_matrix((
        np.ones(rows), (np.arange(rows), opportunity["household"])),
        shape=(rows, config.customers))
    segment = sparse.csr_matrix((
        np.ones(rows), (np.arange(rows), opportunity["segment"])),
        shape=(rows, config.segments))
    chosen = (opportunity["action"] if action_override is None else
              np.full(rows, action_override, dtype=np.int64))
    # The synthetic truth deliberately makes promotion response segment-specific.
    # A global action dummy is therefore misspecified even though its aggregate Brier
    # score can look acceptable.  Encode the identified segment-action cell explicitly;
    # actions are randomized within every segment in the data-generating design.
    segment_action = opportunity["segment"] * (1 + 2 * config.rank) + chosen
    action = sparse.csr_matrix((
        np.ones(rows), (np.arange(rows), segment_action)),
        shape=(rows, config.segments * (1 + 2 * config.rank)))
    continuous = sparse.csr_matrix(np.column_stack([
        opportunity["recency"] / 30.0,
        opportunity["sin_day"], opportunity["cos_day"],
    ]))
    return sparse.hstack([household, segment, action, continuous], format="csr")


def calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.minimum(np.searchsorted(edges, probability, side="right") - 1, bins - 1)
    answer = 0.0
    for block in range(bins):
        selected = index == block
        if selected.any():
            answer += selected.mean() * abs(y[selected].mean() - probability[selected].mean())
    return float(answer)


def fit_arrival(config: Config, simulated: dict) -> tuple[LogisticRegression, dict]:
    opportunity = simulated["opportunity"]
    train = split_mask(opportunity["day"], config, "train")
    validation = split_mask(opportunity["day"], config, "validation")
    test = split_mask(opportunity["day"], config, "test")
    x = arrival_features(opportunity, config)
    candidates = (0.03, 0.10, 0.30, 1.0, 3.0)
    validation_scores = []
    for regularization in candidates:
        candidate = LogisticRegression(
            C=regularization, solver="lbfgs", max_iter=500, tol=1e-9,
            random_state=config.seed, n_jobs=1)
        candidate.fit(x[train], opportunity["purchase"][train])
        probability = candidate.predict_proba(x[validation])[:, 1]
        validation_scores.append(float(log_loss(
            opportunity["purchase"][validation], probability)))
    selected_c = candidates[int(np.argmin(validation_scores))]
    model = LogisticRegression(
        C=selected_c, solver="lbfgs", max_iter=500, tol=1e-9,
        random_state=config.seed, n_jobs=1)
    development = train | validation
    model.fit(x[development], opportunity["purchase"][development])
    probability = model.predict_proba(x[test])[:, 1]
    oracle = opportunity["oracle_purchase_probability"][test]
    y = opportunity["purchase"][test]
    return model, {
        "test_opportunities": int(test.sum()),
        "test_positive_rate": float(y.mean()),
        "log_loss": float(log_loss(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "auc": float(roc_auc_score(y, probability)),
        "expected_calibration_error": calibration_error(y, probability),
        "probability_mae_to_oracle": float(np.mean(np.abs(probability - oracle))),
        "probability_correlation_to_oracle": float(np.corrcoef(probability, oracle)[0, 1]),
        "selected_regularization_C": selected_c,
        "validation_log_loss_by_C": {
            str(value): score for value, score in zip(candidates, validation_scores)},
    }


def fit_store_choice(config: Config, simulated: dict) -> tuple[np.ndarray, dict]:
    opportunity = simulated["opportunity"]
    positive = opportunity["purchase"] == 1
    train = positive & split_mask(opportunity["day"], config, "train")
    test = positive & split_mask(opportunity["day"], config, "test")
    count = np.ones((config.segments, config.stores), dtype=np.float64)
    np.add.at(count, (opportunity["segment"][train], opportunity["store"][train]), 1.0)
    probability = count / count.sum(1, keepdims=True)
    test_probability = probability[opportunity["segment"][test]]
    observed = opportunity["store"][test]
    return probability, {
        "test_trips": int(test.sum()),
        "accuracy": float(np.mean(test_probability.argmax(1) == observed)),
        "log_loss": float(-np.log(test_probability[np.arange(len(observed)), observed]).mean()),
    }


class BasketLaw(torch.nn.Module):
    def __init__(self, config: Config, truth: dict, interaction: bool, seed: int):
        super().__init__()
        self.config = config
        self.register_buffer("log_price_ratio", torch.as_tensor(truth["log_price_ratio"]))
        self.base = torch.nn.Parameter(torch.full((config.products,), -1.2))
        self.segment_free = torch.nn.Parameter(torch.zeros(config.segments - 1,
                                                            config.products))
        self.raw_elasticity = torch.nn.Parameter(torch.full(
            (config.products,), inverse_softplus(1.0)))
        self.rho_size_tail = torch.nn.Parameter(torch.zeros(config.nmax - 1))
        self.rho_category = torch.nn.Parameter(torch.full((config.categories,), 0.1))
        if interaction:
            generator = torch.Generator().manual_seed(seed)
            self.phi = torch.nn.Parameter(
                0.06 * torch.randn(config.products, config.rank, generator=generator))
        else:
            self.register_parameter("phi", None)

    def segment_effect(self) -> torch.Tensor:
        return torch.cat([self.segment_free,
                          -self.segment_free.sum(0, keepdim=True)], dim=0)

    def elasticity(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_elasticity)

    def rho_size(self) -> torch.Tensor:
        return torch.cat([torch.zeros(1), self.rho_size_tail])

    def log_probability(self, support: dict, context_segment: torch.Tensor,
                        context_action: torch.Tensor) -> torch.Tensor:
        b = (self.base[None, :] + self.segment_effect()[context_segment]
             - self.elasticity()[None, :] * self.log_price_ratio[context_action])
        energy = b @ support["membership"].T
        energy -= self.rho_size()[support["sizes"] - 1][None, :]
        energy -= (support["category_pairs"] @ self.rho_category)[None, :]
        if self.phi is not None:
            energy += gram_energy(support["membership"], self.phi)[None, :]
        return energy - torch.logsumexp(energy, dim=1, keepdim=True)


def aggregate_basket_counts(config: Config, simulated: dict, support: dict,
                            split: str) -> torch.Tensor:
    opportunity = simulated["opportunity"]
    trip_day = opportunity["day"][simulated["trip_opportunity"]]
    selected = split_mask(trip_day, config, split)
    flat = (simulated["trip_context"][selected] * len(support["baskets"])
            + simulated["trip_subset"][selected])
    contexts = config.segments * (1 + 2 * config.rank)
    count = np.bincount(flat, minlength=contexts * len(support["baskets"]))
    return torch.as_tensor(count.reshape(contexts, len(support["baskets"])),
                           dtype=torch.float64)


def initialize_basket_model(model: BasketLaw, train_count: torch.Tensor,
                            support: dict, base_model: BasketLaw | None = None) -> None:
    with torch.no_grad():
        if base_model is not None:
            own = model.state_dict()
            for name, value in base_model.state_dict().items():
                if name in own and own[name].shape == value.shape:
                    own[name].copy_(value)
            return
        item_count = train_count.sum(0) @ support["membership"]
        trips = train_count.sum().clamp_min(1.0)
        probability = ((item_count + 0.5) / (trips + 1.0)).clamp(0.01, 0.7)
        model.base.copy_(torch.logit(probability))
        sizes = support["sizes"]
        size_count = torch.zeros(model.config.nmax)
        size_count.index_add_(0, sizes - 1, train_count.sum(0))
        size_probability = (size_count + 0.5) / (size_count.sum() + 0.5 * len(size_count))
        # A mild empirical curvature start avoids placing all mass at nmax.
        axis = torch.arange(1, model.config.nmax + 1, dtype=torch.float64)
        mode = float(axis[size_probability.argmax()])
        model.rho_size_tail.copy_(0.08 * (axis[1:] - mode).square())


def fit_basket_law(config: Config, truth: dict, support: dict,
                   train_count: torch.Tensor, validation_count: torch.Tensor,
                   *, interaction: bool, base_model: BasketLaw | None = None) -> tuple[BasketLaw, dict]:
    model = BasketLaw(config, truth, interaction, config.seed + (97 if interaction else 53))
    initialize_basket_model(model, train_count, support, base_model)
    context_segment = torch.as_tensor(truth["context_segment"], dtype=torch.long)
    context_action = torch.as_tensor(truth["context_action"], dtype=torch.long)
    steps = config.interaction_steps if interaction else config.additive_steps
    learning_rate = 0.018 if interaction else 0.030
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    train_total = train_count.sum()
    validation_total = validation_count.sum()
    best = -float("inf"); best_step = 0; stale = 0; best_state = None; history = []
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logp = model.log_probability(support, context_segment, context_action)
        nll = -(train_count * logp).sum() / train_total
        penalty = (2e-5 * model.base.square().mean()
                   + 3e-5 * model.segment_free.square().mean()
                   + 2e-5 * model.elasticity().square().mean()
                   + 2e-5 * model.rho_size_tail.square().mean()
                   + 2e-5 * model.rho_category.square().mean())
        if model.phi is not None:
            penalty = penalty + 2e-5 * model.phi.square().mean()
        (nll + penalty).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        if step % config.eval_every == 0 or step == steps:
            with torch.no_grad():
                valid_logp = model.log_probability(support, context_segment, context_action)
                score = float((validation_count * valid_logp).sum() / validation_total)
            history.append({"step": step, "train_nll": float(nll.detach()),
                            "validation_log_likelihood": score})
            print(f"[synthetic-retailer] basket={'interaction' if interaction else 'additive'} "
                  f"step={step} train_nll={float(nll.detach()):.5f} "
                  f"valid={score:.5f}", flush=True)
            if score > best + 1e-5:
                best = score; best_step = step; stale = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("basket optimization did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model.eval(), {
        "best_step": best_step,
        "terminal_step": step,
        "best_validation_log_likelihood": best,
        "history": history,
    }


class QuantityLaw(torch.nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.item = torch.nn.Parameter(torch.full((config.products,), -1.0))
        self.segment_free = torch.nn.Parameter(torch.zeros(config.segments - 1))
        self.raw_discount = torch.nn.Parameter(torch.tensor(inverse_softplus(1.0)))
        self.raw_dispersion = torch.nn.Parameter(torch.tensor(inverse_softplus(2.0)))

    def segment_effect(self) -> torch.Tensor:
        return torch.cat([self.segment_free, -self.segment_free.sum().reshape(1)])

    def discount_coefficient(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_discount)

    def dispersion(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_dispersion) + 1e-6

    def mean_extra(self, item: torch.Tensor, segment: torch.Tensor,
                   discount: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.item[item] + self.segment_effect()[segment]
                         + self.discount_coefficient() * discount)

    def log_probability(self, extra: torch.Tensor, item: torch.Tensor,
                        segment: torch.Tensor, discount: torch.Tensor) -> torch.Tensor:
        mean = self.mean_extra(item, segment, discount).clamp_min(1e-10)
        dispersion = self.dispersion()
        return (torch.lgamma(extra + dispersion) - torch.lgamma(dispersion)
                - torch.lgamma(extra + 1.0)
                + dispersion * (torch.log(dispersion) - torch.log(dispersion + mean))
                + extra * (torch.log(mean) - torch.log(dispersion + mean)))


def quantity_rows(config: Config, simulated: dict, truth: dict) -> dict:
    opportunity = simulated["opportunity"]
    trip_opportunity = simulated["trip_opportunity"]
    line_trip = simulated["line_trip"]
    trip_segment = opportunity["segment"][trip_opportunity]
    trip_action = opportunity["action"][trip_opportunity]
    item = simulated["line_item"]
    return {
        "day": opportunity["day"][trip_opportunity][line_trip],
        "item": item,
        "segment": trip_segment[line_trip],
        "discount": truth["action_discount"][trip_action[line_trip], item],
        "extra": simulated["line_quantity"].astype(np.float64) - 1.0,
    }


def fit_quantity(config: Config, simulated: dict, truth: dict) -> tuple[QuantityLaw, dict]:
    rows = quantity_rows(config, simulated, truth)
    train = split_mask(rows["day"], config, "train")
    validation = split_mask(rows["day"], config, "validation")
    test = split_mask(rows["day"], config, "test")
    tensors = {name: torch.as_tensor(value) for name, value in rows.items() if name != "day"}
    tensors["item"] = tensors["item"].long(); tensors["segment"] = tensors["segment"].long()
    train_tensor = torch.as_tensor(train); validation_tensor = torch.as_tensor(validation)
    model = QuantityLaw(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.035)
    best = -float("inf"); best_step = 0; stale = 0; best_state = None; history = []
    for step in range(1, config.quantity_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logp = model.log_probability(tensors["extra"][train_tensor],
                                     tensors["item"][train_tensor],
                                     tensors["segment"][train_tensor],
                                     tensors["discount"][train_tensor])
        loss = -logp.mean() + 2e-5 * model.item.square().mean()
        loss.backward(); optimizer.step()
        if step % config.eval_every == 0 or step == config.quantity_steps:
            with torch.no_grad():
                valid = model.log_probability(tensors["extra"][validation_tensor],
                                              tensors["item"][validation_tensor],
                                              tensors["segment"][validation_tensor],
                                              tensors["discount"][validation_tensor]).mean()
                score = float(valid)
            history.append({"step": step, "train_nll": float(loss.detach()),
                            "validation_log_likelihood": score})
            if score > best + 1e-5:
                best = score; best_step = step; stale = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("quantity optimization did not produce a checkpoint")
    model.load_state_dict(best_state); model.eval()
    test_tensor = torch.as_tensor(test)
    with torch.no_grad():
        test_logp = model.log_probability(tensors["extra"][test_tensor],
                                          tensors["item"][test_tensor],
                                          tensors["segment"][test_tensor],
                                          tensors["discount"][test_tensor])
        predicted_quantity = 1.0 + model.mean_extra(
            tensors["item"][test_tensor], tensors["segment"][test_tensor],
            tensors["discount"][test_tensor])
    observed_quantity = tensors["extra"][test_tensor] + 1.0
    return model, {
        "test_lines": int(test.sum()),
        "test_log_likelihood": float(test_logp.mean()),
        "quantity_mean_observed": float(observed_quantity.mean()),
        "quantity_mean_predicted": float(predicted_quantity.mean()),
        "mean_absolute_error": float((predicted_quantity - observed_quantity).abs().mean()),
        "true_discount_coefficient": float(truth["quantity_discount"]),
        "fitted_discount_coefficient": float(model.discount_coefficient().detach()),
        "true_dispersion": float(truth["quantity_dispersion"]),
        "fitted_dispersion": float(model.dispersion().detach()),
        "training": {"best_step": best_step, "terminal_step": step,
                     "best_validation_log_likelihood": best, "history": history},
    }


def basket_log_probabilities(model: BasketLaw, truth: dict, support: dict) -> np.ndarray:
    with torch.no_grad():
        return model.log_probability(
            support, torch.as_tensor(truth["context_segment"], dtype=torch.long),
            torch.as_tensor(truth["context_action"], dtype=torch.long)).numpy()


def kernel_correlation(true_phi: np.ndarray, fitted_phi: torch.Tensor) -> float:
    true = true_phi @ true_phi.T
    fitted = fitted_phi.detach().numpy() @ fitted_phi.detach().numpy().T
    upper = np.triu_indices_from(true, k=1)
    return float(np.corrcoef(true[upper], fitted[upper])[0, 1])


def recommendation(model: BasketLaw, simulated: dict, truth: dict, support: dict,
                   config: Config, seed: int) -> dict:
    opportunity = simulated["opportunity"]
    trip_day = opportunity["day"][simulated["trip_opportunity"]]
    selected = np.flatnonzero(split_mask(trip_day, config, "test"))
    rng = np.random.default_rng(seed)
    reciprocal, ranks = [], []
    segment_effect = model.segment_effect().detach().numpy()
    base = model.base.detach().numpy(); elasticity = model.elasticity().detach().numpy()
    rho_category = model.rho_category.detach().numpy()
    phi = None if model.phi is None else model.phi.detach().numpy()
    item_category = support["item_category"].numpy()
    for trip in selected:
        basket = list(support["baskets"][int(simulated["trip_subset"][trip])])
        hidden = int(basket[rng.integers(len(basket))])
        observed = [j for j in basket if j != hidden]
        context = int(simulated["trip_context"][trip])
        group = int(truth["context_segment"][context]); action = int(truth["context_action"][context])
        candidates = np.asarray([j for j in range(config.products) if j not in observed])
        score = (base[candidates] + segment_effect[group, candidates]
                 - elasticity[candidates] * truth["log_price_ratio"][action, candidates])
        category_count = np.bincount(item_category[observed], minlength=config.categories)
        score -= rho_category[item_category[candidates]] * category_count[item_category[candidates]]
        if phi is not None and observed:
            score += phi[candidates] @ phi[observed].sum(0)
        hidden_score = score[np.flatnonzero(candidates == hidden)[0]]
        rank = 1.0 + np.sum(score > hidden_score) + 0.5 * (np.sum(score == hidden_score) - 1)
        ranks.append(float(rank)); reciprocal.append(1.0 / rank)
    ranks = np.asarray(ranks); reciprocal = np.asarray(reciprocal)
    return {
        "cases": len(ranks), "mrr": float(reciprocal.mean()),
        "mrr_standard_error": float(reciprocal.std(ddof=1) / math.sqrt(len(reciprocal))),
        "recall_at_5": float(np.mean(ranks <= 5)),
        "recall_at_10": float(np.mean(ranks <= 10)),
        "median_rank": float(np.median(ranks)),
    }


def discrete_js(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.float64); second = np.asarray(second, dtype=np.float64)
    first /= first.sum(); second /= second.sum(); middle = 0.5 * (first + second)
    def kl(p, q):
        selected = p > 0
        return float(np.sum(p[selected] * np.log(p[selected] / q[selected])))
    return 0.5 * kl(first, middle) + 0.5 * kl(second, middle)


def generation_audit(model: BasketLaw, simulated: dict, truth: dict, support: dict,
                     config: Config, seed: int) -> dict:
    logp = basket_log_probabilities(model, truth, support)
    opportunity = simulated["opportunity"]
    trip_day = opportunity["day"][simulated["trip_opportunity"]]
    selected = np.flatnonzero(split_mask(trip_day, config, "test"))
    contexts = simulated["trip_context"][selected]
    observed_subset = simulated["trip_subset"][selected]
    rng = np.random.default_rng(seed)
    generated_subset = np.empty(len(selected), dtype=np.int64)
    for context in np.unique(contexts):
        rows = np.flatnonzero(contexts == context)
        generated_subset[rows] = rng.choice(
            len(support["baskets"]), size=len(rows), p=np.exp(logp[context]))
    sizes = support["sizes"].numpy()
    observed_size = sizes[observed_subset]; generated_size = sizes[generated_subset]
    observed_hist = np.bincount(observed_size, minlength=config.nmax + 1)[1:]
    generated_hist = np.bincount(generated_size, minlength=config.nmax + 1)[1:]
    membership = support["membership"].numpy()
    observed_incidence = membership[observed_subset].mean(0)
    generated_incidence = membership[generated_subset].mean(0)
    return {
        "test_baskets": len(selected),
        "observed_mean_size": float(observed_size.mean()),
        "generated_mean_size": float(generated_size.mean()),
        "observed_size_variance": float(observed_size.var(ddof=1)),
        "generated_size_variance": float(generated_size.var(ddof=1)),
        "size_total_variation": float(0.5 * np.abs(
            observed_hist / observed_hist.sum() - generated_hist / generated_hist.sum()).sum()),
        "size_jensen_shannon": discrete_js(observed_hist, generated_hist),
        "item_incidence_rmse": float(np.sqrt(np.mean(
            (observed_incidence - generated_incidence) ** 2))),
    }


def evaluate_baskets(config: Config, simulated: dict, truth: dict, support: dict,
                     additive: BasketLaw, interaction: BasketLaw) -> dict:
    opportunity = simulated["opportunity"]
    trip_day = opportunity["day"][simulated["trip_opportunity"]]
    selected = split_mask(trip_day, config, "test")
    context = simulated["trip_context"][selected]
    subset = simulated["trip_subset"][selected]
    oracle = truth["logp"][context, subset]
    additive_logp = basket_log_probabilities(additive, truth, support)[context, subset]
    interaction_logp = basket_log_probabilities(interaction, truth, support)[context, subset]
    true_kernel = truth["phi"]
    return {
        "test_trips": int(selected.sum()),
        "test_log_likelihood": {
            "oracle": summarize(oracle), "additive": summarize(additive_logp),
            "interaction": summarize(interaction_logp),
        },
        "paired_gains": {
            "interaction_minus_additive": paired_summary(interaction_logp, additive_logp),
            "oracle_minus_interaction": paired_summary(oracle, interaction_logp),
        },
        "kernel_correlation": kernel_correlation(true_kernel, interaction.phi),
        "price_elasticity_correlation": float(np.corrcoef(
            truth["elasticity"], interaction.elasticity().detach().numpy())[0, 1]),
        "recommendation": {
            "additive": recommendation(additive, simulated, truth, support, config,
                                       config.seed + 701),
            "interaction": recommendation(interaction, simulated, truth, support, config,
                                          config.seed + 701),
        },
        "generation": generation_audit(interaction, simulated, truth, support, config,
                                       config.seed + 809),
    }


def counterfactual_audit(model: BasketLaw, arrival: LogisticRegression,
                         config: Config, simulated: dict, truth: dict,
                         support: dict) -> dict:
    fitted_logp = basket_log_probabilities(model, truth, support)
    membership = support["membership"].numpy(); sizes = support["sizes"].numpy()
    true_probability = np.exp(truth["logp"]); fitted_probability = np.exp(fitted_logp)
    true_size = true_probability @ sizes; fitted_size = fitted_probability @ sizes
    true_incidence = true_probability @ membership
    fitted_incidence = fitted_probability @ membership

    opportunity = simulated["opportunity"]
    test = split_mask(opportunity["day"], config, "test")
    test_opportunity = {name: value[test] for name, value in opportunity.items()}
    arrival_errors = []
    for action in range(len(truth["actions"])):
        fitted = arrival.predict_proba(
            arrival_features(test_opportunity, config, action_override=action))[:, 1]
        logit = (truth["arrival_segment"][test_opportunity["segment"]]
                 + truth["household_frailty"][test_opportunity["household"]]
                 + 0.32 * test_opportunity["sin_day"]
                 - 0.10 * test_opportunity["cos_day"]
                 + 0.030 * test_opportunity["recency"]
                 + truth["action_lift"][test_opportunity["segment"], action])
        logit += truth["arrival_nonlinear"] * (
            test_opportunity["sin_day"] * test_opportunity["recency"] / 30.0
            + (test_opportunity["recency"] / 30.0) ** 2)
        oracle = 1.0 / (1.0 + np.exp(-logit))
        arrival_errors.extend((fitted - oracle).tolist())
    return {
        "contexts": int(len(true_size)),
        "basket_size_mae": float(np.mean(np.abs(fitted_size - true_size))),
        "item_incidence_mae": float(np.mean(np.abs(fitted_incidence - true_incidence))),
        "arrival_probability_mae": float(np.mean(np.abs(arrival_errors))),
        "maximum_basket_size_error": float(np.max(np.abs(fitted_size - true_size))),
    }


def expected_quantity(model: QuantityLaw | None, truth: dict, segment: int,
                      action: int) -> np.ndarray:
    discount = truth["action_discount"][action]
    if model is None:
        return 1.0 + np.exp(truth["quantity_item"]
                            + truth["quantity_segment"][segment]
                            + truth["quantity_discount"] * discount
                            + truth["quantity_discount_quadratic"] * discount ** 2)
    with torch.no_grad():
        return (1.0 + model.mean_extra(
            torch.arange(len(discount)),
            torch.full((len(discount),), segment, dtype=torch.long),
            torch.as_tensor(discount))).numpy()


def solve_budget_policy(actions: list[dict], horizon: int, budget: float,
                        bins: int = 240, minimum_utilization: float = 0.85) -> dict:
    if budget <= 0 or horizon <= 0:
        raise ValueError("budget and horizon must be positive")
    width = budget / bins
    costs = [0 if row["cost"] == 0 else max(1, int(math.ceil(row["cost"] / width)))
             for row in actions]
    maximum_leftover = int(math.floor((1.0 - minimum_utilization) * bins - horizon))
    if maximum_leftover < 0:
        raise ValueError("budget grid too coarse for utilization guarantee")
    value = np.full((horizon + 1, bins + 1), -np.inf)
    value[0, :maximum_leftover + 1] = 0.0
    policy = np.full((horizon + 1, bins + 1), -1, dtype=np.int64)
    for remaining_days in range(1, horizon + 1):
        for remaining_budget in range(bins + 1):
            for action_index, (row, cost) in enumerate(zip(actions, costs)):
                if cost <= remaining_budget and np.isfinite(
                        value[remaining_days - 1, remaining_budget - cost]):
                    candidate = row["reward"] + value[
                        remaining_days - 1, remaining_budget - cost]
                    if candidate > value[remaining_days, remaining_budget]:
                        value[remaining_days, remaining_budget] = candidate
                        policy[remaining_days, remaining_budget] = action_index
    if not np.isfinite(value[horizon, bins]):
        return {"feasible": False}
    remaining = bins; chosen = []
    for remaining_days in range(horizon, 0, -1):
        action_index = int(policy[remaining_days, remaining])
        chosen.append(action_index); remaining -= costs[action_index]
    return {
        "feasible": True, "chosen": chosen,
        "predicted_reward": float(sum(actions[i]["reward"] for i in chosen)),
        "predicted_cost": float(sum(actions[i]["cost"] for i in chosen)),
        "budget": budget,
    }


def policy_audit(config: Config, simulated: dict, truth: dict, support: dict,
                 arrival: LogisticRegression, basket: BasketLaw,
                 quantity: QuantityLaw) -> dict:
    true_logp = truth["logp"]; fitted_logp = basket_log_probabilities(basket, truth, support)
    membership = support["membership"].numpy()
    true_incidence = np.exp(true_logp) @ membership
    fitted_incidence = np.exp(fitted_logp) @ membership
    base_price = truth["base_price"]; cost = truth["unit_cost"]
    opportunity = simulated["opportunity"]
    test = split_mask(opportunity["day"], config, "test")
    test_opportunity = {name: value[test] for name, value in opportunity.items()}

    true_metrics = {}; fitted_metrics = {}
    for segment in range(config.segments):
        segment_rows = test_opportunity["segment"] == segment
        local = {name: value[segment_rows] for name, value in test_opportunity.items()}
        households_per_day = int(np.sum(simulated["customer_segment"] == segment))
        for action in range(len(truth["actions"])):
            logit = (truth["arrival_segment"][segment]
                     + truth["household_frailty"][local["household"]]
                     + 0.32 * local["sin_day"] - 0.10 * local["cos_day"]
                     + 0.030 * local["recency"] + truth["action_lift"][segment, action])
            logit += truth["arrival_nonlinear"] * (
                local["sin_day"] * local["recency"] / 30.0
                + (local["recency"] / 30.0) ** 2)
            true_arrival = float((1.0 / (1.0 + np.exp(-logit))).mean())
            fitted_arrival = float(arrival.predict_proba(
                arrival_features(local, config, action_override=action))[:, 1].mean())
            context = segment * len(truth["actions"]) + action
            price = base_price * (1.0 - truth["action_discount"][action])
            markdown = base_price - price
            true_q = expected_quantity(None, truth, segment, action)
            fitted_q = expected_quantity(quantity, truth, segment, action)
            true_profit = true_arrival * np.sum(
                true_incidence[context] * true_q * (price - cost))
            fitted_profit = fitted_arrival * np.sum(
                fitted_incidence[context] * fitted_q * (price - cost))
            true_spend = true_arrival * np.sum(
                true_incidence[context] * true_q * markdown)
            fitted_spend = fitted_arrival * np.sum(
                fitted_incidence[context] * fitted_q * markdown)
            true_metrics[(segment, action)] = (households_per_day * true_profit,
                                               households_per_day * true_spend)
            fitted_metrics[(segment, action)] = (households_per_day * fitted_profit,
                                                 households_per_day * fitted_spend)

    predicted_actions = [{"name": "none", "segment": None, "action": 0,
                          "reward": 0.0, "cost": 0.0}]
    oracle_actions = [dict(predicted_actions[0])]
    action_comparison = []
    for segment in range(config.segments):
        for action in range(1, len(truth["actions"])):
            predicted_actions.append({
                "name": f"segment_{segment}:{truth['actions'][action]['name']}",
                "segment": segment, "action": action,
                "reward": fitted_metrics[(segment, action)][0]
                          - fitted_metrics[(segment, 0)][0],
                "cost": fitted_metrics[(segment, action)][1],
            })
            oracle_actions.append({
                "name": f"segment_{segment}:{truth['actions'][action]['name']}",
                "segment": segment, "action": action,
                "reward": true_metrics[(segment, action)][0]
                          - true_metrics[(segment, 0)][0],
                "cost": true_metrics[(segment, action)][1],
            })
            action_comparison.append({
                "name": predicted_actions[-1]["name"],
                "predicted_incremental_profit": predicted_actions[-1]["reward"],
                "oracle_incremental_profit": oracle_actions[-1]["reward"],
                "predicted_markdown_spend": predicted_actions[-1]["cost"],
                "oracle_markdown_spend": oracle_actions[-1]["cost"],
            })
    horizon = 28
    positive_cost = [row["cost"] for row in oracle_actions if row["cost"] > 0]
    budget = horizon * float(np.median(positive_cost)) * 0.55
    predicted = solve_budget_policy(predicted_actions, horizon, budget)
    oracle = solve_budget_policy(oracle_actions, horizon, budget)
    if not predicted["feasible"] or not oracle["feasible"]:
        raise RuntimeError("synthetic policy problem is unexpectedly infeasible")
    realized_reward = float(sum(oracle_actions[i]["reward"] for i in predicted["chosen"]))
    realized_cost = float(sum(oracle_actions[i]["cost"] for i in predicted["chosen"]))
    oracle_reward = float(sum(oracle_actions[i]["reward"] for i in oracle["chosen"]))
    names = [predicted_actions[i]["name"] for i in predicted["chosen"]]
    counts = {name: names.count(name) for name in sorted(set(names))}
    oracle_names = [oracle_actions[i]["name"] for i in oracle["chosen"]]
    oracle_counts = {name: oracle_names.count(name) for name in sorted(set(oracle_names))}
    predicted_reward_vector = np.asarray(
        [row["predicted_incremental_profit"] for row in action_comparison])
    oracle_reward_vector = np.asarray(
        [row["oracle_incremental_profit"] for row in action_comparison])
    return {
        "horizon_days": horizon, "budget": budget,
        "predicted_policy_actions": counts,
        "oracle_policy_actions": oracle_counts,
        "predicted_incremental_profit": predicted["predicted_reward"],
        "oracle_realized_profit_of_predicted_policy": realized_reward,
        "oracle_realized_spend_of_predicted_policy": realized_cost,
        "oracle_optimal_incremental_profit": oracle_reward,
        "policy_regret": oracle_reward - realized_reward,
        "budget_violation": max(0.0, realized_cost - budget),
        "action_value_mae": float(np.mean(np.abs(
            predicted_reward_vector - oracle_reward_vector))),
        "action_value_correlation": float(np.corrcoef(
            predicted_reward_vector, oracle_reward_vector)[0, 1]),
        "action_comparison": action_comparison,
    }


def markdown_report(result: dict) -> str:
    basket = result["basket"]
    gain = basket["paired_gains"]["interaction_minus_additive"]
    rec = basket["recommendation"]
    gen = basket["generation"]
    arrival = result["arrival"]
    quantity = result["quantity"]
    policy = result["policy"]
    lines = [
        "# Synthetic end-to-end retailer experiment", "",
        "> Synthetic recovery evidence validates implementation under known truth. It is not",
        "> evidence that synthetic causal effects hold in the real retailer data.", "",
        "## Configuration", "",
        f"- {result['config']['customers']} customers, {result['config']['products']} products, "
        f"{result['config']['segments']} segments and {result['config']['stores']} stores.",
        f"- {result['config']['days']} customer-day opportunities with chronological train/validation/test splits.",
        f"- Exact basket support: {result['support_baskets']:,} nonempty baskets through size {result['config']['nmax']}.",
        f"- Randomized logged promotions: {result['action_count']} actions.", "",
        "## Measured results", "",
        "| Component | Result |", "|---|---:|",
        f"| Arrival test log loss | {arrival['log_loss']:.5f} |",
        f"| Arrival Brier score | {arrival['brier']:.5f} |",
        f"| Arrival probability MAE to oracle | {arrival['probability_mae_to_oracle']:.5f} |",
        f"| Interaction - additive test likelihood | {gain['mean']:+.5f} +/- {gain['standard_error']:.5f} nats/trip |",
        f"| Gram-kernel correlation | {basket['kernel_correlation']:.4f} |",
        f"| Additive / interaction MRR | {rec['additive']['mrr']:.4f} / {rec['interaction']['mrr']:.4f} |",
        f"| Generated / observed mean basket size | {gen['generated_mean_size']:.4f} / {gen['observed_mean_size']:.4f} |",
        f"| Size total variation | {gen['size_total_variation']:.5f} |",
        f"| Quantity true / fitted discount coefficient | {quantity['true_discount_coefficient']:.4f} / {quantity['fitted_discount_coefficient']:.4f} |",
        f"| Counterfactual basket-size MAE | {result['counterfactual']['basket_size_mae']:.5f} products |",
        f"| Oracle policy regret | {policy['policy_regret']:.5f} |",
        f"| Oracle budget violation | {policy['budget_violation']:.5f} |", "",
        "## Policy", "",
        f"The fitted policy used actions `{policy['predicted_policy_actions']}`. Under the known",
        f"oracle it earned {policy['oracle_realized_profit_of_predicted_policy']:.4f} incremental",
        f"profit versus the oracle optimum {policy['oracle_optimal_incremental_profit']:.4f},",
        f"for regret {policy['policy_regret']:.4f}.", "",
    ]
    return "\n".join(lines)


def run(config: Config) -> dict:
    if config.products < config.rank or config.categories > config.products:
        raise ValueError("invalid synthetic dimensions")
    if config.train_days + config.validation_days >= config.days:
        raise ValueError("test period must be nonempty")
    torch.set_num_threads(config.threads)
    tick = time.perf_counter()
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    simulated = simulate_retailer(config, support, truth)
    opportunities = simulated["opportunity"]
    print(f"[synthetic-retailer] opportunities={len(opportunities['day']):,} "
          f"trips={len(simulated['trip_subset']):,} support={len(support['baskets']):,}",
          flush=True)
    arrival_model, arrival_result = fit_arrival(config, simulated)
    _store_probability, store_result = fit_store_choice(config, simulated)
    train_count = aggregate_basket_counts(config, simulated, support, "train")
    validation_count = aggregate_basket_counts(config, simulated, support, "validation")
    additive, additive_training = fit_basket_law(
        config, truth, support, train_count, validation_count, interaction=False)
    interaction, interaction_training = fit_basket_law(
        config, truth, support, train_count, validation_count, interaction=True,
        base_model=additive)
    quantity_model, quantity_result = fit_quantity(config, simulated, truth)
    basket_result = evaluate_baskets(
        config, simulated, truth, support, additive, interaction)
    counterfactual = counterfactual_audit(
        interaction, arrival_model, config, simulated, truth, support)
    policy = policy_audit(config, simulated, truth, support, arrival_model,
                          interaction, quantity_model)
    return {
        "schema": 1,
        "experiment": f"exact-support complete synthetic retailer: {config.world}",
        "config": asdict(config),
        "support_baskets": len(support["baskets"]),
        "action_count": len(truth["actions"]),
        "opportunities": len(opportunities["day"]),
        "trips": len(simulated["trip_subset"]),
        "purchase_rate": float(opportunities["purchase"].mean()),
        "arrival": arrival_result,
        "store": store_result,
        "basket": basket_result,
        "quantity": quantity_result,
        "counterfactual": counterfactual,
        "policy": policy,
        "training": {"additive": additive_training,
                     "interaction": interaction_training},
        "runtime_seconds": time.perf_counter() - tick,
        "scope_warning": (
            "Synthetic recovery validates code under a declared data-generating law; "
            "it does not establish real-data causal identification or production readiness."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=73021)
    parser.add_argument("--world", choices=("well_specified", "misspecified"),
                        default="well_specified")
    parser.add_argument("--output", type=Path,
                        default=Path("reports/synthetic_retailer_experiment.json"))
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    config = Config(seed=args.seed, threads=args.threads, world=args.world)
    if args.profile == "smoke":
        config = Config(
            customers=60, products=10, categories=5, segments=3, stores=3,
            rank=2, nmax=4, days=60, train_days=36, validation_days=12,
            additive_steps=30, interaction_steps=40, quantity_steps=40,
            eval_every=5, patience=5, seed=args.seed, threads=args.threads,
            world=args.world)
    result = run(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    args.output.with_suffix(".md").write_text(markdown_report(result))
    print(f"[synthetic-retailer] report={args.output}", flush=True)
    print(f"[synthetic-retailer] runtime={result['runtime_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
