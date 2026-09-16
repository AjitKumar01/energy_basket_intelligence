"""Auditable conditional-choice tools for external price datasets.

The model in this module is the size-one reduction of the basket energy law:

    P(Y=j | offered products, prices, context) proportional to
        exp(product utility + context-product utility - beta * log(price_j)).

It is intentionally small.  Its purpose is to establish that an external dataset
contains held-out, learnable price information before adapting the full basket model.
The nonnegative ``beta`` constraint makes own-price monotonicity structural rather
than a result selected after looking at the test set.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp


@dataclass(frozen=True)
class ChoicePanel:
    """Padded choice occasions.

    ``product`` and ``log_price`` have shape ``(occasion, slot)``.  Padding slots
    have ``mask=False``; their product value is ignored.  ``chosen_slot`` indexes an
    offered slot, not a global product.  Covariates are known before the choice.
    """

    product: np.ndarray
    log_price: np.ndarray
    mask: np.ndarray
    chosen_slot: np.ndarray
    group: np.ndarray
    covariates: np.ndarray

    def subset(self, index) -> "ChoicePanel":
        index = np.asarray(index)
        return ChoicePanel(
            self.product[index], self.log_price[index], self.mask[index],
            self.chosen_slot[index], self.group[index], self.covariates[index])

    @property
    def n_products(self) -> int:
        return int(self.product[self.mask].max()) + 1


@dataclass(frozen=True)
class ChoiceFit:
    product_utility: np.ndarray
    context_utility: np.ndarray
    price_coefficient: float
    regularization: float
    objective: float
    iterations: int
    converged: bool
    gradient_infinity_norm: float
    termination_message: str


def merge_panels(*panels: ChoicePanel) -> ChoicePanel:
    """Concatenate compatible panels without introducing dataset assumptions."""
    if not panels:
        raise ValueError("at least one choice panel is required")
    widths = {panel.product.shape[1] for panel in panels}
    covariate_widths = {panel.covariates.shape[1] for panel in panels}
    if len(widths) != 1 or len(covariate_widths) != 1:
        raise ValueError("choice panels must have matching padded and covariate widths")
    merged = ChoicePanel(
        product=np.concatenate([panel.product for panel in panels]),
        log_price=np.concatenate([panel.log_price for panel in panels]),
        mask=np.concatenate([panel.mask for panel in panels]),
        chosen_slot=np.concatenate([panel.chosen_slot for panel in panels]),
        group=np.concatenate([panel.group for panel in panels]),
        covariates=np.concatenate([panel.covariates for panel in panels]),
    )
    validate_panel(merged, n_products=max(panel.n_products for panel in panels))
    return merged


def validate_panel(panel: ChoicePanel, *, n_products: int | None = None) -> None:
    shape = panel.product.shape
    if len(shape) != 2 or panel.log_price.shape != shape or panel.mask.shape != shape:
        raise ValueError("product, log-price, and mask arrays must have one padded shape")
    n = shape[0]
    if (panel.chosen_slot.shape != (n,) or panel.group.shape != (n,)
            or panel.covariates.ndim != 2 or panel.covariates.shape[0] != n):
        raise ValueError("occasion-level arrays have inconsistent shapes")
    if n == 0 or bool((panel.mask.sum(axis=1) < 2).any()):
        raise ValueError("every choice occasion needs at least two alternatives")
    rows = np.arange(n)
    if (bool((panel.chosen_slot < 0).any())
            or bool((panel.chosen_slot >= shape[1]).any())
            or not bool(panel.mask[rows, panel.chosen_slot].all())):
        raise ValueError("chosen_slot must identify an offered alternative")
    if not bool(np.isfinite(panel.log_price[panel.mask]).all()):
        raise ValueError("offered log-prices must be finite")
    if not bool(np.isfinite(panel.covariates).all()):
        raise ValueError("choice covariates must be finite")
    offered = panel.product[panel.mask]
    limit = panel.n_products if n_products is None else int(n_products)
    if bool((offered < 0).any()) or bool((offered >= limit).any()):
        raise ValueError("offered product identifiers are outside the declared universe")


def within_group_split(panel: ChoicePanel, seed: int, *, minimum_group_size: int = 5,
                       train_fraction: float = .6, validation_fraction: float = .2
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically split within persistent entities.

    Groups too small to put at least one record in every partition are excluded.  The
    caller gets integer indices so that preprocessing can be fitted on training only.
    """
    if minimum_group_size < 3 or not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("invalid grouped split specification")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("the test fraction must be positive")
    rng = np.random.default_rng(seed)
    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for value in np.unique(panel.group):
        index = np.flatnonzero(panel.group == value)
        if len(index) < minimum_group_size:
            continue
        index = index[rng.permutation(len(index))]
        n_train = max(1, int(math.floor(train_fraction * len(index))))
        n_validation = max(1, int(math.floor(validation_fraction * len(index))))
        if n_train + n_validation >= len(index):
            n_train = len(index) - 2
            n_validation = 1
        train.extend(index[:n_train])
        validation.extend(index[n_train:n_train + n_validation])
        test.extend(index[n_train + n_validation:])
    if not train or not validation or not test:
        raise ValueError("the grouped split produced an empty partition")
    return tuple(np.asarray(x, dtype=np.int64) for x in (train, validation, test))


def temporal_split(values: np.ndarray, *, train_fraction: float = .6,
                   validation_fraction: float = .2
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[float, float]]:
    """Split whole ordered periods, never individual observations within a period."""
    values = np.asarray(values)
    periods = np.unique(values)
    if len(periods) < 5 or train_fraction + validation_fraction >= 1:
        raise ValueError("at least five periods and a positive test fraction are required")
    a = max(1, int(math.floor(train_fraction * len(periods))))
    b = max(a + 1, int(math.floor((train_fraction + validation_fraction) * len(periods))))
    b = min(b, len(periods) - 1)
    train = np.flatnonzero(values <= periods[a - 1])
    validation = np.flatnonzero((values > periods[a - 1]) & (values <= periods[b - 1]))
    test = np.flatnonzero(values > periods[b - 1])
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("the temporal split produced an empty partition")
    return train, validation, test, (float(periods[a - 1]), float(periods[b - 1]))


def standardize_covariates(panel: ChoicePanel, train_index: np.ndarray
                           ) -> tuple[ChoicePanel, dict[str, list[float]]]:
    x = panel.covariates.astype(np.float64, copy=True)
    if x.shape[1] == 0:
        return panel, {"mean": [], "scale": []}
    mean = x[train_index].mean(axis=0)
    scale = x[train_index].std(axis=0)
    scale[scale < 1e-12] = 1.0
    x = (x - mean) / scale
    result = ChoicePanel(panel.product, panel.log_price, panel.mask,
                         panel.chosen_slot, panel.group, x)
    return result, {"mean": mean.tolist(), "scale": scale.tolist()}


def _unpack(parameters: np.ndarray, n_products: int, n_covariates: int,
            include_price: bool) -> tuple[np.ndarray, np.ndarray, float]:
    p = n_products - 1
    alpha = np.zeros(n_products, dtype=np.float64)
    alpha[1:] = parameters[:p]
    cursor = p
    gamma = np.zeros((n_covariates, n_products), dtype=np.float64)
    if n_covariates:
        gamma[:, 1:] = parameters[cursor:cursor + n_covariates * p].reshape(n_covariates, p)
        cursor += n_covariates * p
    beta = float(parameters[cursor]) if include_price else 0.0
    return alpha, gamma, beta


def _probabilities(panel: ChoicePanel, alpha: np.ndarray, gamma: np.ndarray,
                   beta: float) -> tuple[np.ndarray, np.ndarray]:
    utility = alpha[panel.product].astype(np.float64, copy=True)
    for column in range(panel.covariates.shape[1]):
        utility += panel.covariates[:, column, None] * gamma[column, panel.product]
    utility -= beta * panel.log_price
    utility[~panel.mask] = -np.inf
    normalizer = logsumexp(utility, axis=1)
    probability = np.exp(utility - normalizer[:, None])
    probability[~panel.mask] = 0.0
    return probability, normalizer


def fit_choice_model(panel: ChoicePanel, *, n_products: int | None = None,
                     include_price: bool = True, regularization: float = .01,
                     maximum_iterations: int = 1000) -> ChoiceFit:
    validate_panel(panel, n_products=n_products)
    n_products = panel.n_products if n_products is None else int(n_products)
    d = panel.covariates.shape[1]
    p = n_products - 1
    size = p + d * p + int(include_price)
    initial = np.zeros(size, dtype=np.float64)
    if include_price:
        initial[-1] = 1.0
    rows = np.arange(len(panel.chosen_slot))

    def objective(parameters):
        alpha, gamma, beta = _unpack(parameters, n_products, d, include_price)
        probability, normalizer = _probabilities(panel, alpha, gamma, beta)
        chosen_product = panel.product[rows, panel.chosen_slot]
        chosen_utility = alpha[chosen_product].copy()
        for column in range(d):
            chosen_utility += panel.covariates[:, column] * gamma[column, chosen_product]
        chosen_utility -= beta * panel.log_price[rows, panel.chosen_slot]
        loss = float(np.mean(normalizer - chosen_utility))
        free = parameters[:-1] if include_price else parameters
        loss += .5 * regularization * float(free @ free)

        error = probability
        error[rows, panel.chosen_slot] -= 1.0
        error /= len(rows)
        alpha_gradient = np.zeros(n_products, dtype=np.float64)
        np.add.at(alpha_gradient, panel.product[panel.mask], error[panel.mask])
        pieces = [alpha_gradient[1:]]
        if d:
            gamma_gradient = np.zeros((d, n_products), dtype=np.float64)
            for column in range(d):
                np.add.at(gamma_gradient[column], panel.product[panel.mask],
                          (error * panel.covariates[:, column, None])[panel.mask])
            pieces.append(gamma_gradient[:, 1:].ravel())
        if include_price:
            beta_gradient = -float(np.sum(error[panel.mask] * panel.log_price[panel.mask]))
            gradient = np.concatenate((*pieces, np.asarray([beta_gradient])))
            gradient[:-1] += regularization * free
        else:
            gradient = np.concatenate(pieces)
            gradient += regularization * free
        return loss, gradient

    bounds = [(None, None)] * size
    if include_price:
        bounds[-1] = (0.0, 100.0)
    result = minimize(objective, initial, method="L-BFGS-B", jac=True, bounds=bounds,
                      options={"maxiter": maximum_iterations, "ftol": 1e-12,
                               "gtol": 1e-7, "maxls": 50})
    alpha, gamma, beta = _unpack(result.x, n_products, d, include_price)
    gradient_norm = float(np.max(np.abs(result.jac)))
    converged = bool(result.success or gradient_norm <= 2e-5)
    if not converged or not np.isfinite(result.fun):
        raise RuntimeError(f"choice optimization failed: {result.message}; gradient={gradient_norm}")
    return ChoiceFit(alpha, gamma, beta, regularization, float(result.fun),
                     int(result.nit), converged, gradient_norm, str(result.message))


def predict_choice(panel: ChoicePanel, fit: ChoiceFit) -> np.ndarray:
    validate_panel(panel, n_products=len(fit.product_utility))
    probability, _ = _probabilities(
        panel, fit.product_utility, fit.context_utility, fit.price_coefficient)
    if not np.allclose(probability.sum(axis=1), 1.0, atol=1e-11, rtol=0):
        raise FloatingPointError("choice probabilities do not sum to one")
    return probability


def choice_metrics(panel: ChoicePanel, fit: ChoiceFit) -> dict[str, float | int]:
    probability = predict_choice(panel, fit)
    rows = np.arange(len(panel.chosen_slot))
    chosen = np.maximum(probability[rows, panel.chosen_slot], np.finfo(float).tiny)
    target = np.zeros_like(probability)
    target[rows, panel.chosen_slot] = 1.0
    return {
        "occasions": int(len(rows)),
        "negative_log_likelihood": float(-np.log(chosen).mean()),
        "top1_accuracy": float((probability.argmax(axis=1) == panel.chosen_slot).mean()),
        "multiclass_brier": float(np.square(probability - target).sum(axis=1).mean()),
    }


def price_alignment_placebo(panel: ChoicePanel, price_fit: ChoiceFit,
                            baseline_fit: ChoiceFit, *, seed: int = 991,
                            replicates: int = 200) -> dict:
    """Break product-specific price/occasion alignment without changing its marginal values.

    Prices are shuffled only among appearances of the same product.  Thus the test
    does not obtain an easy win from mixing cheap and expensive product tiers.  It
    asks whether the actual pairing of price with occasion predicts choices better
    than a pairing that is deliberately wrong.
    """
    rows = np.arange(len(panel.chosen_slot))
    base_probability = predict_choice(panel, baseline_fit)[rows, panel.chosen_slot]
    baseline_nll = float(-np.log(np.maximum(
        base_probability, np.finfo(float).tiny)).mean())
    actual_probability = predict_choice(panel, price_fit)[rows, panel.chosen_slot]
    actual_difference = float(-np.log(np.maximum(
        actual_probability, np.finfo(float).tiny)).mean() - baseline_nll)
    locations = [np.where(panel.mask & (panel.product == product))
                 for product in range(len(price_fit.product_utility))]
    rng = np.random.default_rng(seed)
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        shuffled_price = panel.log_price.copy()
        for location in locations:
            if len(location[0]) > 1:
                values = panel.log_price[location]
                shuffled_price[location] = values[rng.permutation(len(values))]
        shuffled = ChoicePanel(
            panel.product, shuffled_price, panel.mask, panel.chosen_slot,
            panel.group, panel.covariates)
        probability = predict_choice(shuffled, price_fit)[rows, panel.chosen_slot]
        draws[replicate] = float(-np.log(np.maximum(
            probability, np.finfo(float).tiny)).mean() - baseline_nll)
    return {
        "estimand": "price_model_nll_minus_no_price_model_nll",
        "actual_price_alignment_estimate": actual_difference,
        "product_specific_shuffled_price_mean": float(draws.mean()),
        "95_product_specific_shuffle_interval": [
            float(np.quantile(draws, .025)), float(np.quantile(draws, .975))],
        "fraction_shuffles_as_good_or_better_than_actual": float(
            (1 + np.sum(draws <= actual_difference)) / (replicates + 1)),
        "replicates": int(replicates),
        "passed": bool(actual_difference < float(np.quantile(draws, .025))),
    }


def select_regularization(train: ChoicePanel, validation: ChoicePanel, *,
                          n_products: int, include_price: bool,
                          grid=(0.0, 1e-4, 1e-3, 1e-2, 1e-1)) -> tuple[ChoiceFit, list[dict]]:
    rows = []
    fits = []
    for value in grid:
        fit = fit_choice_model(train, n_products=n_products, include_price=include_price,
                               regularization=float(value))
        metric = choice_metrics(validation, fit)
        rows.append({"regularization": float(value), **metric,
                     "iterations": fit.iterations,
                     "converged": fit.converged,
                     "gradient_infinity_norm": fit.gradient_infinity_norm,
                     "termination_message": fit.termination_message,
                     "price_coefficient": fit.price_coefficient})
        fits.append(fit)
    best = min(range(len(rows)), key=lambda i: rows[i]["negative_log_likelihood"])
    return fits[best], rows


def clustered_loss_difference(panel: ChoicePanel, price_fit: ChoiceFit,
                              baseline_fit: ChoiceFit, *, seed: int = 731,
                              replicates: int = 2000,
                              clusters: np.ndarray | None = None,
                              cluster_name: str = "household") -> dict:
    """Cluster bootstrap for price-model NLL minus no-price-model NLL.

    The default resamples persistent entities.  Callers may instead supply blocks,
    such as whole weeks, to preserve shared shocks within those blocks.
    """
    rows = np.arange(len(panel.chosen_slot))
    price_probability = predict_choice(panel, price_fit)[rows, panel.chosen_slot]
    base_probability = predict_choice(panel, baseline_fit)[rows, panel.chosen_slot]
    difference = -np.log(np.maximum(price_probability, np.finfo(float).tiny))
    difference += np.log(np.maximum(base_probability, np.finfo(float).tiny))
    clusters = panel.group if clusters is None else np.asarray(clusters)
    if clusters.shape != (len(panel.chosen_slot),):
        raise ValueError("bootstrap clusters must have one value per choice occasion")
    groups = np.unique(clusters)
    if len(groups) < 2:
        raise ValueError("cluster bootstrap requires at least two clusters")
    grouped = [difference[clusters == group] for group in groups]
    rng = np.random.default_rng(seed)
    draw = np.empty(replicates, dtype=np.float64)
    for r in range(replicates):
        selected = rng.integers(0, len(grouped), size=len(grouped))
        numerator = sum(float(grouped[i].sum()) for i in selected)
        denominator = sum(len(grouped[i]) for i in selected)
        draw[r] = numerator / denominator
    estimate = float(difference.mean())
    return {
        "estimand": "price_model_nll_minus_no_price_model_nll",
        "estimate": estimate,
        "95_cluster_bootstrap_interval": [float(np.quantile(draw, .025)),
                                            float(np.quantile(draw, .975))],
        "cluster_unit": cluster_name,
        "clusters": int(len(groups)),
        "replicates": int(replicates),
        "price_improves_if_negative": True,
    }


def training_price_support(panel: ChoicePanel, n_products: int) -> tuple[np.ndarray, np.ndarray]:
    low = np.full(n_products, np.inf)
    high = np.full(n_products, -np.inf)
    np.minimum.at(low, panel.product[panel.mask], panel.log_price[panel.mask])
    np.maximum.at(high, panel.product[panel.mask], panel.log_price[panel.mask])
    return low, high


def counterfactual_price_increase_audit(panel: ChoicePanel, fit: ChoiceFit,
                                        training_panel: ChoicePanel, *,
                                        relative_increase: float = .10) -> tuple[dict, dict | None]:
    """Audit in-support one-product price increases and return one worked case."""
    if relative_increase <= 0:
        raise ValueError("the counterfactual price increase must be positive")
    log_change = math.log1p(relative_increase)
    low, high = training_price_support(training_panel, len(fit.product_utility))
    base = predict_choice(panel, fit)
    considered_actions = int(panel.mask.sum())
    factual_in_support = 0
    own_changes: list[float] = []
    mass_errors: list[float] = []
    records: list[dict] = []
    for row in range(len(panel.chosen_slot)):
        for slot in np.flatnonzero(panel.mask[row]):
            product = int(panel.product[row, slot])
            old_log_price = float(panel.log_price[row, slot])
            new_log_price = float(panel.log_price[row, slot] + log_change)
            if (not np.isfinite(low[product]) or old_log_price < low[product] - 1e-12
                    or old_log_price > high[product] + 1e-12):
                continue
            factual_in_support += 1
            if new_log_price > high[product] + 1e-12:
                continue
            changed_price = panel.log_price[row:row + 1].copy()
            changed_price[0, slot] = new_log_price
            changed = ChoicePanel(
                panel.product[row:row + 1], changed_price, panel.mask[row:row + 1],
                panel.chosen_slot[row:row + 1], panel.group[row:row + 1],
                panel.covariates[row:row + 1])
            probability = predict_choice(changed, fit)[0]
            own_change = float(probability[slot] - base[row, slot])
            own_changes.append(own_change)
            mass_errors.append(abs(float(probability.sum()) - 1.0))
            if slot == int(panel.chosen_slot[row]):
                recipients = probability - base[row]
                recipients[slot] = -np.inf
                recipient_slot = int(np.argmax(recipients))
                records.append({
                    "row": row,
                    "changed_slot": int(slot),
                    "changed_product": product,
                    "old_price": float(np.exp(panel.log_price[row, slot])),
                    "new_price": float(np.exp(new_log_price)),
                    "old_probability": float(base[row, slot]),
                    "new_probability": float(probability[slot]),
                    "largest_recipient_product": int(panel.product[row, recipient_slot]),
                    "largest_recipient_old_probability": float(base[row, recipient_slot]),
                    "largest_recipient_new_probability": float(probability[recipient_slot]),
                    "offered_products": [int(x) for x in panel.product[row, panel.mask[row]]],
                })
    if not own_changes:
        return {
            "status": "unsupported",
            "considered_actions": considered_actions,
            "factual_price_in_support_actions": factual_in_support,
            "supported_actions": 0,
            "supported_action_fraction": 0.0,
        }, None
    own = np.asarray(own_changes)
    tolerance = 2e-12
    audit = {
        "status": "passed" if float(own.max()) <= tolerance else "failed",
        "relative_price_increase": relative_increase,
        "considered_actions": considered_actions,
        "factual_price_in_support_actions": factual_in_support,
        "supported_actions": int(len(own)),
        "supported_action_fraction": float(len(own) / considered_actions),
        "mean_own_probability_change": float(own.mean()),
        "maximum_own_probability_change": float(own.max()),
        "monotonicity_violations": int((own > tolerance).sum()),
        "maximum_probability_mass_error": float(max(mass_errors)),
    }
    case = max(records, key=lambda record: record["old_probability"], default=None)
    return audit, case
