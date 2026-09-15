import numpy as np
import pandas as pd
import torch
from scipy.special import expit

from fit_supported_price_response import (
    base_eligible_events,
    conditional_loss,
    event_sensitivity,
    fit_model,
    initial_parameters,
    objective,
    restrict_to_training_price_support,
    unconstrained_global_evidence,
)
from fit_exact_additive import (
    configure_global_price,
    configure_supported_price,
    fitted_parameters,
    load_supported_price_sensitivity,
)
from ragged import RaggedModel


def _frame(seed=817, events=4000, sensitivity=0.8):
    generator = np.random.default_rng(seed)
    price_change = generator.uniform(-0.45, 0.45, events)
    offset = generator.normal(0.0, 0.15, events)
    totals = generator.integers(10, 35, events)
    purchases = generator.binomial(
        totals, expit(offset - sensitivity * price_change))
    product_index = generator.integers(0, 4, events)
    category_index = np.asarray([0, 0, 1, 1])[product_index]
    return pd.DataFrame({
        "item_id": product_index,
        "store_id": generator.integers(0, 20, events),
        "WEEK_NO": generator.integers(1, 82, events),
        "log_price_change": price_change,
        "offset": offset,
        "purchases": purchases,
        "total_purchases": totals,
        "product_index": product_index,
        "category_index": category_index,
        "observed_before_incidence": np.full(events, 0.1),
        "observed_response": np.zeros(events),
    })


def _finite_difference(function, parameters, step=1e-6):
    result = np.empty_like(parameters)
    for index in range(len(parameters)):
        left = parameters.copy()
        right = parameters.copy()
        left[index] -= step
        right[index] += step
        result[index] = (function(right) - function(left)) / (2.0 * step)
    return result


def test_supported_price_objective_gradients_match_finite_differences():
    frame = _frame(events=80)
    product_category = np.asarray([0, 0, 1, 1])
    cases = (
        ("global", np.asarray([0.7]), 0.0, 0.0),
        ("category", np.asarray([0.7, 0.4, 1.1]), 0.3, 0.0),
        ("product", np.asarray([0.7, 0.4, 1.1, 0.2, 0.6, 0.9, 1.4]), 0.3, 0.6),
    )
    for level, parameters, category_penalty, product_penalty in cases:
        def loss(value):
            return objective(
                value, frame, level, 2, 4, product_category,
                category_penalty, product_penalty)[0]

        analytic = objective(
            parameters, frame, level, 2, 4, product_category,
            category_penalty, product_penalty)[1]
        numeric = _finite_difference(loss, parameters)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-5, atol=2e-7)


def test_global_fit_recovers_known_conditional_price_sensitivity():
    frame = _frame(events=12000, sensitivity=0.8)
    product_category = np.asarray([0, 0, 1, 1])
    model = fit_model(frame, "global", 2, 4, product_category)
    fitted = float(model["parameters"][0])
    assert abs(fitted - 0.8) < 0.06
    fitted_loss = conditional_loss(model, frame, 2, 4)[0]
    no_price_loss = conditional_loss(None, frame, 2, 4)[0]
    assert fitted_loss < no_price_loss
    signed = unconstrained_global_evidence(frame)
    assert abs(signed["sensitivity"] - 0.8) < 0.06
    assert signed["product_clustered_95_interval"][0] < 0.8
    assert signed["product_clustered_95_interval"][1] > 0.8


def test_hierarchical_fit_is_nonnegative_and_shrinks_to_parent():
    frame = _frame(events=1200)
    product_category = np.asarray([0, 0, 1, 1])
    initial = initial_parameters(
        "product", 2, 4, product_category, global_value=0.5)
    model = fit_model(
        frame, "product", 2, 4, product_category,
        category_penalty=100.0, product_penalty=100.0, initial=initial)
    sensitivities = event_sensitivity(model["parameters"], frame, "product", 2, 4)
    assert np.all(sensitivities >= 0.0)
    global_value = model["parameters"][0]
    categories = model["parameters"][1:3]
    products = model["parameters"][3:]
    assert np.max(np.abs(categories - global_value)) < 0.02
    assert np.max(np.abs(products - categories[product_category])) < 0.02


def test_supported_global_price_factor_is_exact_frozen_and_kappa_one():
    model = RaggedModel(J=5, N=3, C=2, K=3, Kz=2, nmax=5, R=3,
                        S=2, Kp=4, Kt=2, Ks=2)
    audit = configure_global_price(model, 0.32024650666336923)
    assert audit["relative_price_kappa"] == 1.0
    assert audit["maximum_product_coefficient_error"] < 1e-12
    assert not model.gamma.requires_grad
    assert not model.beta.requires_grad
    assert not model.price_kappa.requires_grad
    trainable_ids = {id(parameter) for parameter in fitted_parameters(model)}
    assert id(model.gamma) not in trainable_ids
    assert id(model.beta) not in trainable_ids
    assert id(model.price_kappa) not in trainable_ids

    model.house = torch.tensor([0, 1], dtype=torch.long)
    item = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    trip = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    common = {
        "dlp_bar": torch.tensor([0.08, -0.03]),
        "disp": torch.zeros(4),
        "mail": torch.zeros(4),
        "week": torch.zeros(4, dtype=torch.long),
        "store": torch.zeros(4, dtype=torch.long),
    }
    base = {**common, "dlp": torch.tensor([0.02, 0.12, -0.04, 0.01])}
    changed = {**common, "dlp": base["dlp"] + torch.tensor([0.10, -0.05, 0.03, 0.20])}
    difference = model.b_at(item, trip, changed) - model.b_at(item, trip, base)
    np.testing.assert_allclose(
        difference.detach().numpy(),
        -0.32024650666336923 * (changed["dlp"] - base["dlp"]).numpy(),
        rtol=0.0, atol=2e-15)


def test_supported_product_price_factor_accepts_zero_slopes():
    model = RaggedModel(J=4, N=3, C=2, K=3, Kz=2, nmax=5, R=3,
                        S=2, Kp=4, Kt=2, Ks=2)
    target = np.asarray([0.0, 0.2, 0.0, 0.7])
    audit = configure_supported_price(model, target, "product")
    with torch.no_grad():
        actual = (torch.nn.functional.softplus(model.gamma).mean(0)
                  * torch.nn.functional.softplus(model.beta)).sum(-1)
    np.testing.assert_allclose(actual.numpy(), target, rtol=0.0, atol=1e-12)
    assert audit["sensitivity_minimum"] == 0.0


def test_supported_product_artifact_defaults_unestimated_products_to_global(tmp_path):
    path = tmp_path / "coefficients.json"
    path.write_text("""{
      "status": "certified_observational_predictor",
      "level": "product",
      "global_sensitivity": 0.0,
      "product_sensitivity": [
        {"item_id": 1, "sensitivity": 0.2},
        {"item_id": 4, "sensitivity": 0.0}
      ],
      "individual_coefficient_interpretation_supported": false
    }""")
    sensitivity, level, audit = load_supported_price_sensitivity(path, 6)
    np.testing.assert_allclose(sensitivity, [0.0, 0.2, 0.0, 0.0, 0.0, 0.0])
    assert level == "product"
    assert audit["estimated_product_deviations"] == 2
    assert audit["unestimated_products_using_global_sensitivity"] == 4
    assert not audit["individual_coefficient_interpretation_supported"]


def test_training_price_support_is_learned_after_other_filters():
    frame = pd.DataFrame({
        "item_id": [0, 0, 0, 0, 0],
        "split": ["train", "train", "validation", "test", "test"],
        "log_price_change": [0.1, 0.5, 0.1, 0.3, -0.1],
        "promotion_depth_change": [0.0, 0.5, 0.0, 0.0, 0.0],
        "display_changed": [False] * 5,
        "mailer_changed": [False] * 5,
        "n_train_lines": [1000] * 5,
        "trips": [20] * 5,
        "previous_trips": [20] * 5,
    })
    assert base_eligible_events(frame).tolist() == [True, False, True, True, True]
    supported, audit = restrict_to_training_price_support(frame)
    assert supported.split.tolist() == ["train", "validation"]
    assert audit["eligible_before_training_price_support"]["test"] == 2
    assert audit["after_training_price_support"].get("test", 0) == 0
