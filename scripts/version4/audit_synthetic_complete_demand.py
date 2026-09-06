#!/usr/bin/env python3
"""Exact synthetic audit of a joint visit--store--purchase--basket law.

The small recovery world enumerates every supported basket so the joint likelihood,
marginals, counterfactuals, and gradients have an exact oracle.  A separate dynamic-
program benchmark measures catalogue scaling without enumeration.

This is an implementation and identifiability experiment.  It is not evidence that
synthetic behavioral effects hold for a real retailer.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np
import torch


torch.set_default_dtype(torch.float64)


@dataclass(frozen=True)
class Config:
    customers: int = 180
    products: int = 12
    categories: int = 3
    segments: int = 3
    stores: int = 3
    actions: int = 4
    rank: int = 2
    nmax: int = 4
    days: int = 150
    train_days: int = 95
    validation_days: int = 25
    additive_steps: int = 500
    interaction_steps: int = 650
    independent_steps: int = 500
    eval_every: int = 10
    patience: int = 18
    learning_rate: float = 0.035
    seed: int = 94017
    threads: int = 8
    world: str = "linked_interaction"


def inverse_softplus(value: float) -> float:
    return math.log(math.expm1(value))


def enumerate_support(products: int, nmax: int, categories: int) -> dict:
    """Enumerate the small audit support; never used by the scale benchmark."""
    baskets = [b for n in range(1, nmax + 1)
               for b in itertools.combinations(range(products), n)]
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
        "category_pairs": category_pairs,
    }


def gram_energy(membership: torch.Tensor, phi: torch.Tensor) -> torch.Tensor:
    total = membership @ phi
    diagonal = membership @ phi.square().sum(1)
    return 0.5 * (total.square().sum(1) - diagonal)


def centered_last(free: torch.Tensor) -> torch.Tensor:
    return torch.cat([free, -free.sum(0, keepdim=True)], dim=0)


def make_truth(config: Config, support: dict) -> dict:
    """Construct a fully identified, moderate-signal joint data-generating law."""
    rng = np.random.default_rng(config.seed)
    j, g, s, r = (config.products, config.segments,
                  config.stores, config.rank)
    mission = np.arange(j) % r
    raw_basis = np.eye(r)[mission] + 0.10 * rng.normal(size=(j, r))
    raw_basis -= raw_basis.mean(0, keepdims=True)
    basis, _ = np.linalg.qr(raw_basis)
    # The primary recovery world is deliberately above the finite-sample detection
    # boundary.  A separate weak world measures that boundary and the null world checks
    # false discovery.  This is preferable to declaring recovery from an underpowered
    # realization after inspecting its fitted result.
    interaction_scale = np.linspace(0.78, 0.44, r)
    if config.world == "linked_interaction":
        interaction_scale *= 3.0
    if config.world == "null_interaction":
        interaction_scale[:] = 0.0

    base = -1.80 + 0.22 * rng.normal(size=j)
    segment = 0.20 * rng.normal(size=(g, j))
    for group in range(g):
        segment[group, mission == group % r] += 0.30
    segment -= segment.mean(0, keepdims=True)
    store_item = 0.10 * rng.normal(size=(s, j))
    store_item -= store_item.mean(0, keepdims=True)
    elasticity = np.clip(1.10 + 0.16 * rng.normal(size=j), 0.65, 1.55)
    rho_size = np.asarray([0.0, 0.05, 0.38, 0.95])[:config.nmax]
    if config.nmax > len(rho_size):
        tail = 0.95 + 0.65 * np.arange(1, config.nmax - len(rho_size) + 1)
        rho_size = np.concatenate([rho_size, tail])
    rho_category = 0.12 + 0.04 * rng.random(config.categories)

    # Actions are store-specific price interventions.  Action zero is no discount;
    # each other action discounts one mission slice.
    discount = np.zeros((config.actions, j), dtype=np.float64)
    for action in range(1, config.actions):
        discount[action, mission == (action - 1) % r] = 0.08 + 0.04 * action
    log_price_ratio = np.log1p(-discount)

    store_direct = np.linspace(-2.45, -2.20, s)
    store_direct -= store_direct.mean()
    store_direct -= 2.35
    segment_store = 0.22 * rng.normal(size=(g, s))
    for group in range(g):
        segment_store[group, group % s] += 0.40
    segment_store -= segment_store.mean(1, keepdims=True)
    distance_coefficient = 1.15
    purchase_segment = np.linspace(-1.05, -0.55, g)
    purchase_sin = 0.22
    purchase_cos = -0.10

    return {
        "mission": mission,
        "basis": basis,
        "interaction_scale": interaction_scale,
        "base": base,
        "segment": segment,
        "store_item": store_item,
        "elasticity": elasticity,
        "rho_size": rho_size,
        "rho_category": rho_category,
        "discount": discount,
        "log_price_ratio": log_price_ratio,
        "store_direct": store_direct,
        "segment_store": segment_store,
        "distance_coefficient": distance_coefficient,
        "purchase_segment": purchase_segment,
        "purchase_sin": purchase_sin,
        "purchase_cos": purchase_cos,
    }


class CompleteDemandLaw(torch.nn.Module):
    """Joint outside/store/empty/nonempty law with an exact small basket support."""

    def __init__(self, config: Config, truth: dict, *, interaction: bool,
                 linked: bool = True):
        super().__init__()
        self.config = config
        self.interaction = interaction
        self.linked = linked
        self.register_buffer("basis", torch.as_tensor(truth["basis"]))
        self.register_buffer("log_price_ratio",
                             torch.as_tensor(truth["log_price_ratio"]))
        self.base = torch.nn.Parameter(torch.full((config.products,), -1.65))
        self.segment_free = torch.nn.Parameter(torch.zeros(
            config.segments - 1, config.products))
        self.store_item_free = torch.nn.Parameter(torch.zeros(
            config.stores - 1, config.products))
        self.raw_elasticity = torch.nn.Parameter(torch.full(
            (config.products,), inverse_softplus(0.95)))
        self.rho_size_tail = torch.nn.Parameter(torch.linspace(
            0.10, 0.70, max(config.nmax - 1, 1))[:config.nmax - 1])
        self.rho_category = torch.nn.Parameter(torch.full(
            (config.categories,), 0.10))
        if interaction:
            self.raw_interaction_scale = torch.nn.Parameter(torch.full(
                (config.rank,), inverse_softplus(0.12)))
        else:
            self.register_buffer("fixed_interaction_scale",
                                 torch.zeros(config.rank))
        self.store_direct_free = torch.nn.Parameter(torch.zeros(config.stores - 1))
        self.store_direct_level = torch.nn.Parameter(torch.tensor(-2.15))
        self.segment_store_free = torch.nn.Parameter(torch.zeros(
            config.segments, config.stores - 1))
        self.raw_distance = torch.nn.Parameter(torch.tensor(inverse_softplus(0.9)))
        self.purchase_segment = torch.nn.Parameter(torch.full(
            (config.segments,), -0.75))
        self.purchase_sin = torch.nn.Parameter(torch.tensor(0.0))
        self.purchase_cos = torch.nn.Parameter(torch.tensor(0.0))

    def segment_effect(self) -> torch.Tensor:
        return centered_last(self.segment_free)

    def store_item_effect(self) -> torch.Tensor:
        return centered_last(self.store_item_free)

    def store_direct(self) -> torch.Tensor:
        centered = torch.cat([self.store_direct_free,
                              -self.store_direct_free.sum().reshape(1)])
        return self.store_direct_level + centered

    def segment_store_effect(self) -> torch.Tensor:
        return torch.cat([self.segment_store_free,
                          -self.segment_store_free.sum(1, keepdim=True)], dim=1)

    def elasticity(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_elasticity)

    def distance_coefficient(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_distance)

    def interaction_scale(self) -> torch.Tensor:
        if self.interaction:
            return torch.nn.functional.softplus(self.raw_interaction_scale)
        return self.fixed_interaction_scale

    def phi(self) -> torch.Tensor:
        return self.basis * torch.sqrt(self.interaction_scale()).unsqueeze(0)

    def rho_size(self) -> torch.Tensor:
        return torch.cat([torch.zeros(1), self.rho_size_tail])

    def basket_energy(self, support: dict) -> torch.Tensor:
        """Return [segment, store, action, basket] Version-4 energies."""
        b = (self.base[None, None, None, :]
             + self.segment_effect()[:, None, None, :]
             + self.store_item_effect()[None, :, None, :]
             - self.elasticity()[None, None, None, :]
             * self.log_price_ratio[None, None, :, :])
        energy = torch.einsum("gsaj,bj->gsab", b, support["membership"])
        energy -= self.rho_size()[support["sizes"] - 1][None, None, None, :]
        category = support["category_pairs"] @ self.rho_category
        energy -= category[None, None, None, :]
        energy += gram_energy(support["membership"], self.phi())[None, None, None, :]
        return energy

    def context_terms(self, data: dict, support: dict) -> dict:
        energy = self.basket_energy(support)
        logz = torch.logsumexp(energy, dim=-1)
        segment = data["segment"]
        actions = data["actions"]
        stores = torch.arange(self.config.stores)[None, :]
        context_logz = logz[segment[:, None], stores, actions]
        direct = (self.store_direct()[None, :]
                  + self.segment_store_effect()[segment]
                  - self.distance_coefficient() * data["distance"])
        activation = (self.purchase_segment[segment, None]
                      + self.purchase_sin * data["sin_day"][:, None]
                      + self.purchase_cos * data["cos_day"][:, None])
        if self.linked:
            log_inner = torch.logaddexp(torch.zeros_like(context_logz),
                                        activation + context_logz)
            store_log_weight = direct + log_inner
            logd = torch.logsumexp(torch.cat([
                torch.zeros((len(segment), 1)), store_log_weight], dim=1), dim=1)
            log_no = -logd
            log_empty_store = direct - logd[:, None]
            log_purchase_store = direct + activation + context_logz - logd[:, None]
        else:
            # Independent hurdle baseline: visit and purchase heads do not receive log Z.
            logd_store = torch.logsumexp(torch.cat([
                torch.zeros((len(segment), 1)), direct], dim=1), dim=1)
            log_no = -logd_store
            log_visit_store = direct - logd_store[:, None]
            log_buy_given_visit = torch.nn.functional.logsigmoid(activation)
            log_empty_given_visit = torch.nn.functional.logsigmoid(-activation)
            log_empty_store = log_visit_store + log_empty_given_visit
            log_purchase_store = (log_visit_store + log_buy_given_visit
                                  + torch.zeros_like(context_logz))
        return {
            "energy": energy,
            "logz": logz,
            "context_logz": context_logz,
            "direct": direct,
            "activation": activation,
            "log_no": log_no,
            "log_empty_store": log_empty_store,
            "log_purchase_store": log_purchase_store,
        }

    def observed_log_probability(self, data: dict, support: dict) -> torch.Tensor:
        terms = self.context_terms(data, support)
        outcome = data["outcome"]
        store = data["store"].clamp_min(0)
        row = torch.arange(len(outcome))
        answer = terms["log_no"].clone()
        empty = outcome == 1
        purchase = outcome == 2
        answer[empty] = terms["log_empty_store"][row[empty], store[empty]]
        if purchase.any():
            chosen_energy = terms["energy"][
                data["segment"][purchase], store[purchase],
                data["actions"][purchase, store[purchase]],
                data["basket"][purchase]]
            if self.linked:
                conditional = chosen_energy - terms["context_logz"][
                    purchase, store[purchase]]
                answer[purchase] = (
                    terms["log_purchase_store"][purchase, store[purchase]]
                    + conditional)
            else:
                conditional = chosen_energy - terms["context_logz"][
                    purchase, store[purchase]]
                answer[purchase] = (
                    terms["log_purchase_store"][purchase, store[purchase]]
                    + conditional)
        return answer

    def marginal_probabilities(self, data: dict, support: dict) -> dict:
        terms = self.context_terms(data, support)
        no = terms["log_no"].exp()
        empty_store = terms["log_empty_store"].exp()
        if self.linked:
            purchase_store = terms["log_purchase_store"].exp()
        else:
            purchase_store = terms["log_purchase_store"].exp()
        basket_probability = torch.softmax(terms["energy"], dim=-1)
        incidence = torch.einsum(
            "gsab,bj->gsaj", basket_probability, support["membership"])
        stores = torch.arange(self.config.stores)[None, :]
        conditional_incidence = incidence[
            data["segment"][:, None], stores, data["actions"]]
        item = torch.einsum("ns,nsj->nj", purchase_store,
                            conditional_incidence)
        return {
            "no_visit": no,
            "empty_store": empty_store,
            "purchase_store": purchase_store,
            "visit_store": empty_store + purchase_store,
            "purchase": purchase_store.sum(1),
            "visit": 1.0 - no,
            "item": item,
            "normalization": no + empty_store.sum(1) + purchase_store.sum(1),
        }

    def load_truth(self, truth: dict) -> None:
        with torch.no_grad():
            self.base.copy_(torch.as_tensor(truth["base"]))
            self.segment_free.copy_(torch.as_tensor(truth["segment"][:-1]))
            self.store_item_free.copy_(torch.as_tensor(truth["store_item"][:-1]))
            self.raw_elasticity.copy_(torch.as_tensor([
                inverse_softplus(float(x)) for x in truth["elasticity"]]))
            self.rho_size_tail.copy_(torch.as_tensor(truth["rho_size"][1:]))
            self.rho_category.copy_(torch.as_tensor(truth["rho_category"]))
            if self.interaction:
                self.raw_interaction_scale.copy_(torch.as_tensor([
                    inverse_softplus(float(x)) if x > 0.0 else -50.0
                    for x in truth["interaction_scale"]]))
            centered = truth["store_direct"] - np.mean(truth["store_direct"])
            self.store_direct_free.copy_(torch.as_tensor(centered[:-1]))
            self.store_direct_level.copy_(torch.tensor(np.mean(truth["store_direct"])))
            self.segment_store_free.copy_(torch.as_tensor(
                truth["segment_store"][:, :-1]))
            self.raw_distance.copy_(torch.tensor(inverse_softplus(
                float(truth["distance_coefficient"]))))
            self.purchase_segment.copy_(torch.as_tensor(truth["purchase_segment"]))
            self.purchase_sin.copy_(torch.tensor(truth["purchase_sin"]))
            self.purchase_cos.copy_(torch.tensor(truth["purchase_cos"]))


def tensor_data(data: dict, mask: np.ndarray | None = None) -> dict:
    if mask is None:
        mask = np.ones(len(data["day"]), dtype=bool)
    result = {}
    for key in ("day", "segment", "actions", "distance", "sin_day", "cos_day",
                "outcome", "store", "basket"):
        value = data[key][mask]
        tensor = torch.as_tensor(value)
        if key in {"day", "segment", "actions", "outcome", "store", "basket"}:
            tensor = tensor.long()
        result[key] = tensor
    return result


def split_mask(day: np.ndarray, config: Config, split: str) -> np.ndarray:
    if split == "train":
        return day < config.train_days
    if split == "validation":
        return ((day >= config.train_days)
                & (day < config.train_days + config.validation_days))
    if split == "test":
        return day >= config.train_days + config.validation_days
    raise ValueError(split)


def simulate(config: Config, support: dict, truth: dict) -> tuple[dict, CompleteDemandLaw]:
    rng = np.random.default_rng(config.seed + 1)
    customer_segment = np.arange(config.customers) % config.segments
    rng.shuffle(customer_segment)
    distance_by_customer = rng.uniform(0.08, 1.0,
                                       size=(config.customers, config.stores))
    for h in range(config.customers):
        distance_by_customer[h, customer_segment[h] % config.stores] *= 0.55

    household = np.tile(np.arange(config.customers), config.days)
    day = np.repeat(np.arange(config.days), config.customers)
    segment = customer_segment[household]
    actions = rng.integers(0, config.actions,
                           size=(len(day), config.stores), endpoint=False)
    distance = distance_by_customer[household]
    sin_day = np.sin(2 * np.pi * day / 28.0)
    cos_day = np.cos(2 * np.pi * day / 28.0)
    blank = {
        "household": household, "day": day, "segment": segment,
        "actions": actions, "distance": distance,
        "sin_day": sin_day, "cos_day": cos_day,
        "outcome": np.zeros(len(day), dtype=np.int64),
        "store": np.full(len(day), -1, dtype=np.int64),
        "basket": np.full(len(day), -1, dtype=np.int64),
    }
    oracle = CompleteDemandLaw(config, truth, interaction=True, linked=True)
    oracle.load_truth(truth)
    oracle.eval()
    td = tensor_data(blank)
    with torch.no_grad():
        terms = oracle.context_terms(td, support)
        no = terms["log_no"].exp().numpy()
        empty = terms["log_empty_store"].exp().numpy()
        purchase = terms["log_purchase_store"].exp().numpy()
        basket_logp = torch.log_softmax(terms["energy"], dim=-1).numpy()
    joint = np.column_stack([no, empty, purchase])
    joint /= joint.sum(1, keepdims=True)
    draw = np.asarray([rng.choice(joint.shape[1], p=row) for row in joint])
    empty_mask = (draw >= 1) & (draw <= config.stores)
    purchase_mask = draw > config.stores
    blank["outcome"][empty_mask] = 1
    blank["store"][empty_mask] = draw[empty_mask] - 1
    blank["outcome"][purchase_mask] = 2
    blank["store"][purchase_mask] = draw[purchase_mask] - 1 - config.stores
    for row in np.flatnonzero(purchase_mask):
        group = blank["segment"][row]
        store = blank["store"][row]
        action = blank["actions"][row, store]
        blank["basket"][row] = rng.choice(
            len(support["baskets"]), p=np.exp(basket_logp[group, store, action]))
    return blank, oracle


def initialize_from(model: CompleteDemandLaw, parent: CompleteDemandLaw) -> None:
    own = model.state_dict()
    with torch.no_grad():
        for name, value in parent.state_dict().items():
            if name in own and own[name].shape == value.shape:
                own[name].copy_(value)


def fit_model(config: Config, support: dict, data: dict, *, interaction: bool,
              linked: bool, parent: CompleteDemandLaw | None = None) -> tuple[CompleteDemandLaw, dict]:
    model = CompleteDemandLaw(config, make_truth(config, support),
                              interaction=interaction, linked=linked)
    if parent is not None:
        initialize_from(model, parent)
    train = tensor_data(data, split_mask(data["day"], config, "train"))
    validation = tensor_data(data, split_mask(data["day"], config, "validation"))
    steps = (config.interaction_steps if interaction else
             config.independent_steps if not linked else config.additive_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best = -float("inf")
    best_step = 0
    best_state = None
    stale = 0
    history = []
    tick = time.perf_counter()
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logp = model.observed_log_probability(train, support)
        penalty = 1e-5 * sum(p.square().mean() for p in model.parameters())
        loss = -logp.mean() + penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
        optimizer.step()
        if step % config.eval_every == 0 or step == steps:
            with torch.no_grad():
                validation_score = float(model.observed_log_probability(
                    validation, support).mean())
            history.append({"step": step, "train_nll": float(loss.detach()),
                            "validation_joint_log_likelihood": validation_score})
            if validation_score > best + 1e-6:
                best = validation_score
                best_step = step
                best_state = copy.deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("joint optimization produced no checkpoint")
    model.load_state_dict(best_state)
    acceptance = None
    if interaction and parent is not None:
        with torch.no_grad():
            candidate_logp = model.observed_log_probability(validation, support)
            parent_logp = parent.observed_log_probability(validation, support)
        delta = (candidate_logp - parent_logp).numpy()
        mean = float(delta.mean())
        standard_error = float(delta.std(ddof=1) / math.sqrt(len(delta)))
        lower = mean - 1.96 * standard_error
        accepted = lower > 0.0
        acceptance = {
            "paired_validation_gain": mean,
            "paired_validation_standard_error": standard_error,
            "paired_validation_lower_95": lower,
            "accepted": accepted,
            "fallback": None if accepted else "linked_additive_parent",
        }
        if not accepted:
            initialize_from(model, parent)
            with torch.no_grad():
                # Softplus(-50) is below floating-point resolution in the resulting
                # basket probabilities while retaining one
                # common model class for evaluation.
                model.raw_interaction_scale.fill_(-50.0)
    model.eval()
    return model, {
        "best_step": best_step,
        "terminal_step": step,
        "best_validation_joint_log_likelihood": best,
        "runtime_seconds": time.perf_counter() - tick,
        "interaction_acceptance": acceptance,
        "history": history,
    }


def calibration_error(observed: np.ndarray, probability: np.ndarray,
                      bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    index = np.clip(np.searchsorted(edges, probability, side="right") - 1,
                    0, bins - 1)
    answer = 0.0
    for block in range(bins):
        selected = index == block
        if selected.any():
            answer += selected.mean() * abs(
                observed[selected].mean() - probability[selected].mean())
    return float(answer)


def paired_summary(first: np.ndarray, second: np.ndarray) -> dict:
    delta = np.asarray(first) - np.asarray(second)
    se = float(delta.std(ddof=1) / math.sqrt(len(delta)))
    mean = float(delta.mean())
    return {
        "mean": mean,
        "standard_error": se,
        "interval_95": [mean - 1.96 * se, mean + 1.96 * se],
    }


def predictions(model: CompleteDemandLaw, data: dict, support: dict,
                mask: np.ndarray) -> tuple[np.ndarray, dict]:
    td = tensor_data(data, mask)
    with torch.no_grad():
        logp = model.observed_log_probability(td, support).numpy()
        marginal = {key: value.numpy() for key, value in
                    model.marginal_probabilities(td, support).items()}
    return logp, marginal


def evaluate(config: Config, support: dict, data: dict, oracle: CompleteDemandLaw,
             additive: CompleteDemandLaw, full: CompleteDemandLaw,
             independent: CompleteDemandLaw) -> dict:
    mask = split_mask(data["day"], config, "test")
    selected = {key: value[mask] for key, value in data.items()}
    models = {"oracle": oracle, "linked_additive": additive,
              "linked_interaction": full, "independent_hurdle": independent}
    logp = {}
    marginal = {}
    for name, model in models.items():
        logp[name], marginal[name] = predictions(model, data, support, mask)

    observed_visit = (selected["outcome"] != 0).astype(float)
    observed_purchase = (selected["outcome"] == 2).astype(float)
    calibration = {}
    for name in models:
        calibration[name] = {
            "visit_brier": float(np.mean(
                (observed_visit - marginal[name]["visit"]) ** 2)),
            "purchase_brier": float(np.mean(
                (observed_purchase - marginal[name]["purchase"]) ** 2)),
            "visit_ece": calibration_error(observed_visit, marginal[name]["visit"]),
            "purchase_ece": calibration_error(
                observed_purchase, marginal[name]["purchase"]),
            "visit_probability_mae_to_oracle": float(np.mean(np.abs(
                marginal[name]["visit"] - marginal["oracle"]["visit"]))),
            "purchase_probability_mae_to_oracle": float(np.mean(np.abs(
                marginal[name]["purchase"] - marginal["oracle"]["purchase"]))),
        }

    oracle_item = marginal["oracle"]["item"].mean(0)
    demand = {}
    for name in models:
        item = marginal[name]["item"].mean(0)
        demand[name] = {
            "item_probability_mae_to_oracle": float(np.mean(np.abs(item-oracle_item))),
            "item_probability_correlation_to_oracle": float(np.corrcoef(
                item, oracle_item)[0, 1]),
            "total_expected_items_per_opportunity": float(item.sum()),
        }

    true_kernel = oracle.phi().detach().numpy() @ oracle.phi().detach().numpy().T
    fitted_kernel = full.phi().detach().numpy() @ full.phi().detach().numpy().T
    upper = np.triu_indices(config.products, k=1)
    return {
        "test_opportunities": int(mask.sum()),
        "outcome_rates": {
            "no_visit": float((selected["outcome"] == 0).mean()),
            "empty_visit": float((selected["outcome"] == 1).mean()),
            "purchase": float((selected["outcome"] == 2).mean()),
        },
        "joint_log_likelihood": {name: float(values.mean())
                                 for name, values in logp.items()},
        "paired_gains": {
            "interaction_minus_additive": paired_summary(
                logp["linked_interaction"], logp["linked_additive"]),
            "linked_minus_independent": paired_summary(
                logp["linked_interaction"], logp["independent_hurdle"]),
            "oracle_minus_fitted": paired_summary(
                logp["oracle"], logp["linked_interaction"]),
        },
        "calibration": calibration,
        "unconditional_demand": demand,
        "maximum_probability_normalization_error": {
            name: float(np.max(np.abs(value["normalization"] - 1.0)))
            for name, value in marginal.items()},
        "parameter_recovery": {
            "interaction_kernel_correlation": float(np.corrcoef(
                true_kernel[upper], fitted_kernel[upper])[0, 1]),
            "true_interaction_scale": oracle.interaction_scale().detach().tolist(),
            "fitted_interaction_scale": full.interaction_scale().detach().tolist(),
            "elasticity_correlation": float(np.corrcoef(
                oracle.elasticity().detach().numpy(),
                full.elasticity().detach().numpy())[0, 1]),
            "distance_true": float(oracle.distance_coefficient().detach()),
            "distance_fitted": float(full.distance_coefficient().detach()),
        },
    }


def counterfactual_audit(config: Config, support: dict, data: dict,
                         oracle: CompleteDemandLaw,
                         full: CompleteDemandLaw,
                         independent: CompleteDemandLaw) -> dict:
    mask = split_mask(data["day"], config, "test")
    base = {key: value[mask].copy() for key, value in data.items()}
    models = {"oracle": oracle, "linked_interaction": full,
              "independent_hurdle": independent}
    curves = {name: [] for name in models}
    for action in range(config.actions):
        changed = dict(base)
        changed["actions"] = np.full_like(base["actions"], action)
        td = tensor_data(changed)
        for name, model in models.items():
            with torch.no_grad():
                marginal = model.marginal_probabilities(td, support)
            curves[name].append({
                "action": action,
                "visit_probability": float(marginal["visit"].mean()),
                "purchase_probability": float(marginal["purchase"].mean()),
                "expected_distinct_items": float(marginal["item"].sum(1).mean()),
            })
    def mae(name: str, field: str) -> float:
        return float(np.mean([abs(curves[name][a][field] - curves["oracle"][a][field])
                              for a in range(config.actions)]))
    return {
        "curves": curves,
        "linked_mae": {field: mae("linked_interaction", field) for field in
                       ("visit_probability", "purchase_probability",
                        "expected_distinct_items")},
        "independent_mae": {field: mae("independent_hurdle", field) for field in
                            ("visit_probability", "purchase_probability",
                             "expected_distinct_items")},
        "oracle_extensive_margin_change": {
            field: curves["oracle"][-1][field] - curves["oracle"][0][field]
            for field in ("visit_probability", "purchase_probability")},
        "linked_extensive_margin_change": {
            field: curves["linked_interaction"][-1][field]
                   - curves["linked_interaction"][0][field]
            for field in ("visit_probability", "purchase_probability")},
        "independent_extensive_margin_change": {
            field: curves["independent_hurdle"][-1][field]
                   - curves["independent_hurdle"][0][field]
            for field in ("visit_probability", "purchase_probability")},
    }


def log_esp_partition(log_weight: np.ndarray, rho_size: np.ndarray,
                      nmax: int) -> float:
    """Stable O(J*nmax), O(nmax)-memory additive partition for scaling tests."""
    coefficient = np.full(nmax + 1, -np.inf, dtype=np.float64)
    coefficient[0] = 0.0
    highest = 0
    for value in np.asarray(log_weight, dtype=np.float64):
        new_highest = min(highest + 1, nmax)
        degree = np.arange(1, new_highest + 1)
        coefficient[degree] = np.logaddexp(
            coefficient[degree], coefficient[degree - 1] + value)
        highest = new_highest
    return float(np.logaddexp.reduce(coefficient[1:] - rho_size[:nmax]))


def log_convolve(first: np.ndarray, second: np.ndarray, nmax: int) -> np.ndarray:
    degree = min(nmax, len(first) + len(second) - 2)
    answer = np.full(degree + 1, -np.inf, dtype=np.float64)
    for total in range(degree + 1):
        lo = max(0, total - (len(second) - 1))
        hi = min(total, len(first) - 1)
        left = first[lo:hi + 1]
        right = second[total - np.arange(lo, hi + 1)]
        answer[total] = np.logaddexp.reduce(left + right)
    return answer


def log_category_partition(log_weight: np.ndarray, category: np.ndarray,
                           rho_category: np.ndarray, rho_size: np.ndarray,
                           nmax: int) -> float:
    """Exact log-domain category/size DP used to benchmark the full recurrence."""
    combined = np.asarray([0.0], dtype=np.float64)
    for group in range(len(rho_category)):
        selected = np.asarray(log_weight)[np.asarray(category) == group]
        if not len(selected):
            continue
        degree = min(nmax, len(selected))
        coefficient = np.full(degree + 1, -np.inf, dtype=np.float64)
        coefficient[0] = 0.0
        highest = 0
        for value in selected:
            new_highest = min(highest + 1, degree)
            axis = np.arange(1, new_highest + 1)
            coefficient[axis] = np.logaddexp(
                coefficient[axis], coefficient[axis - 1] + value)
            highest = new_highest
        axis = np.arange(degree + 1)
        coefficient -= rho_category[group] * axis * (axis - 1) / 2.0
        combined = log_convolve(combined, coefficient, nmax)
    return float(np.logaddexp.reduce(combined[1:] - rho_size[:len(combined)-1]))


def exhaustive_additive_partition(log_weight: np.ndarray,
                                  rho_size: np.ndarray, nmax: int) -> float:
    values = []
    for size in range(1, nmax + 1):
        for basket in itertools.combinations(range(len(log_weight)), size):
            values.append(sum(log_weight[list(basket)]) - rho_size[size - 1])
    return float(torch.logsumexp(torch.as_tensor(values), dim=0))


def scalability_benchmark(seed: int, quick: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    sizes = [50, 200, 1000, 5455] if quick else [50, 200, 1000, 5455, 10000]
    nmax = 40 if quick else 120
    rows = []
    category_rows = []
    for products in sizes:
        active_nmax = min(products, nmax)
        log_weight = -2.0 + 0.35 * rng.normal(size=products)
        rho = 0.015 * np.arange(1, active_nmax + 1) ** 2
        repetitions = 2 if quick else 4
        elapsed = []
        result = None
        for _ in range(repetitions):
            tick = time.perf_counter()
            result = log_esp_partition(log_weight, rho, active_nmax)
            elapsed.append(time.perf_counter() - tick)
        rows.append({
            "products": products,
            "nmax": active_nmax,
            "median_seconds": float(np.median(elapsed)),
            "log_partition": result,
            "coefficient_memory_bytes": int((active_nmax + 1) * 8),
        })
        groups = min(300, max(2, products // 5))
        category = np.arange(products) % groups
        rho_category = 0.08 + 0.04 * rng.random(groups)
        category_elapsed = []
        category_result = None
        category_repetitions = 1 if quick else 2
        for _ in range(category_repetitions):
            tick = time.perf_counter()
            category_result = log_category_partition(
                log_weight, category, rho_category, rho, active_nmax)
            category_elapsed.append(time.perf_counter() - tick)
        category_rows.append({
            "products": products,
            "categories": groups,
            "nmax": active_nmax,
            "median_seconds": float(np.median(category_elapsed)),
            "log_partition": category_result,
            # The category loop is streaming.  At most a category coefficient,
            # accumulated coefficient, convolution output and temporary reduction
            # vector are live; this workspace is independent of J and C.
            "coefficient_working_memory_bytes": int(
                4 * (active_nmax + 1) * 8),
        })
    x = np.log([row["products"] for row in rows[-3:]])
    y = np.log([max(row["median_seconds"], 1e-9) for row in rows[-3:]])
    slope = float(np.polyfit(x, y, 1)[0])
    small_weight = -1.5 + 0.2 * rng.normal(size=10)
    small_rho = 0.04 * np.arange(1, 5) ** 2
    dp = log_esp_partition(small_weight, small_rho, 4)
    exact = exhaustive_additive_partition(small_weight, small_rho, 4)
    try:
        from ragged import smolyak_grid
        smolyak = {
            str(rank): {
                "screen": len(smolyak_grid(rank, rank + 1)[1]),
                "reported": len(smolyak_grid(rank, rank + 2)[1]),
                "audit": len(smolyak_grid(rank, rank + 3)[1]),
            }
            for rank in range(4, 9)
        }
    except ImportError:
        smolyak = {}
    return {
        "catalogue_benchmark": rows,
        "category_factorized_benchmark": category_rows,
        "empirical_log_log_slope_last_three": slope,
        "small_exact_log_partition": exact,
        "small_dp_log_partition": dp,
        "small_dp_absolute_error": abs(dp - exact),
        "smolyak_node_multipliers": smolyak,
        "interpretation": (
            "The small-world recovery enumerates baskets only for oracle validation. "
            "The production recurrence is linear in products for fixed nmax; interaction "
            "quadrature and SMC multiply this recurrence and are separate costs."),
    }


def run(config: Config, *, quick_scale: bool = False) -> dict:
    if config.train_days + config.validation_days >= config.days:
        raise ValueError("test period must be nonempty")
    if config.rank > config.products or config.categories > config.products:
        raise ValueError("invalid synthetic dimensions")
    torch.set_num_threads(config.threads)
    tick = time.perf_counter()
    support = enumerate_support(config.products, config.nmax, config.categories)
    truth = make_truth(config, support)
    data, oracle = simulate(config, support, truth)
    print(f"[complete-demand] world={config.world} opportunities={len(data['day']):,} "
          f"support={len(support['baskets']):,}", flush=True)
    additive, additive_fit = fit_model(
        config, support, data, interaction=False, linked=True)
    full, full_fit = fit_model(
        config, support, data, interaction=True, linked=True, parent=additive)
    independent, independent_fit = fit_model(
        config, support, data, interaction=True, linked=False)
    evaluation = evaluate(config, support, data, oracle, additive, full, independent)
    counterfactual = counterfactual_audit(
        config, support, data, oracle, full, independent)
    scale = scalability_benchmark(config.seed + 99, quick=quick_scale)
    return {
        "schema": 1,
        "experiment": "joint inclusive-value complete-demand synthetic audit",
        "config": asdict(config),
        "support_baskets": len(support["baskets"]),
        "opportunities": len(data["day"]),
        "evaluation": evaluation,
        "counterfactual": counterfactual,
        "scalability": scale,
        "training": {
            "linked_additive": additive_fit,
            "linked_interaction": full_fit,
            "independent_hurdle": independent_fit,
        },
        "runtime_seconds": time.perf_counter() - tick,
        "scope_warning": (
            "Synthetic recovery validates the joint likelihood and implementation under "
            "known truth; it does not establish real-data causal identification."),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--world", choices=("linked_interaction", "weak_interaction",
                                             "null_interaction"),
                        default="linked_interaction")
    parser.add_argument("--seed", type=int, default=94017)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=Path("reports/synthetic_complete_demand.json"))
    args = parser.parse_args()
    config = Config(seed=args.seed, threads=args.threads, world=args.world)
    if args.profile == "smoke":
        config = replace(
            config, customers=36, products=8, categories=2, segments=2,
            stores=2, actions=3, rank=2, nmax=3, days=48,
            train_days=30, validation_days=9, additive_steps=80,
            interaction_steps=100, independent_steps=80, eval_every=5,
            patience=8, learning_rate=0.045)
    result = run(config, quick_scale=args.profile == "smoke")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[complete-demand] report={args.output} "
          f"runtime={result['runtime_seconds']:.2f}s", flush=True)


if __name__ == "__main__":
    main()
