
import numpy as np
import pandas as pd

from external_choice import (ChoicePanel, choice_metrics,
                             clustered_loss_difference,
                             counterfactual_price_increase_audit,
                             fit_choice_model, price_alignment_placebo, temporal_split,
                             within_group_split)
from external_choice_data import (ERIM_PURCHASE_COLUMNS, ERIM_PURCHASE_WIDTHS,
                                  _read_fixed, read_erim_products)
from external_choice_adapters import LongChoiceCsvAdapter, adapter_from_config
from external_choice_pipeline import (DatasetCapabilities, ExternalChoiceExperiment,
                                      TemporalSplit, ValidationSettings)


def synthetic_panel(seed=17, n=2500):
    rng = np.random.default_rng(seed)
    products = np.broadcast_to(np.arange(3), (n, 3)).copy()
    log_price = rng.normal(0, .35, size=(n, 3))
    true_utility = np.array([.4, 0.0, -.3]) - 2.0 * log_price
    probability = np.exp(true_utility - np.logaddexp.reduce(true_utility, axis=1)[:, None])
    chosen = np.asarray([rng.choice(3, p=p) for p in probability])
    return ChoicePanel(products, log_price, np.ones_like(products, dtype=bool), chosen,
                       np.arange(n) // 5, np.empty((n, 0)))


def test_price_choice_recovers_signal_and_counterfactual_monotonicity():
    panel = synthetic_panel()
    fit = fit_choice_model(panel, include_price=True, regularization=0.0)
    baseline = fit_choice_model(panel, include_price=False, regularization=0.0)
    assert 1.6 < fit.price_coefficient < 2.4
    assert choice_metrics(panel, fit)["negative_log_likelihood"] < choice_metrics(
        panel, baseline)["negative_log_likelihood"]
    audit, case = counterfactual_price_increase_audit(
        panel, fit, panel, relative_increase=.10)
    assert audit["status"] == "passed"
    assert audit["monotonicity_violations"] == 0
    assert audit["mean_own_probability_change"] < 0
    assert audit["considered_actions"] == panel.mask.sum()
    assert 0 < audit["supported_action_fraction"] <= 1
    assert case is not None


def test_cluster_bootstrap_accepts_explicit_time_blocks():
    panel = synthetic_panel(n=300)
    fit = fit_choice_model(panel, include_price=True, regularization=0.001)
    baseline = fit_choice_model(panel, include_price=False, regularization=0.001)
    result = clustered_loss_difference(
        panel, fit, baseline, clusters=np.arange(300) // 30,
        cluster_name="synthetic_period", replicates=100)
    assert result["cluster_unit"] == "synthetic_period"
    assert result["clusters"] == 10
    assert result["estimate"] < 0


def test_price_alignment_placebo_detects_known_signal():
    panel = synthetic_panel(n=1200)
    fit = fit_choice_model(panel, include_price=True, regularization=0.001)
    baseline = fit_choice_model(panel, include_price=False, regularization=0.001)
    result = price_alignment_placebo(panel, fit, baseline, replicates=99)
    assert result["passed"]
    assert result["fraction_shuffles_as_good_or_better_than_actual"] <= .02


def test_splits_preserve_groups_and_whole_periods():
    panel = synthetic_panel(n=150)
    train, validation, test = within_group_split(panel, 12)
    assert set(train).isdisjoint(validation)
    assert set(train).isdisjoint(test)
    assert set(validation).isdisjoint(test)
    periods = np.repeat(np.arange(10), 3)
    train, validation, test, boundaries = temporal_split(periods)
    assert periods[train].max() == boundaries[0]
    assert periods[validation].max() == boundaries[1]
    assert periods[test].min() > boundaries[1]


def test_erim_fixed_width_reader_rejects_bad_record_length(tmp_path):
    good = "".join("0" * width for width in ERIM_PURCHASE_WIDTHS)
    path = tmp_path / "purchase.dat"
    path.write_text(good + "\n")
    numeric = ERIM_PURCHASE_COLUMNS[:19]
    frame = _read_fixed(path, ERIM_PURCHASE_WIDTHS, ERIM_PURCHASE_COLUMNS, numeric)
    assert len(frame) == 1
    path.write_text(good[:-1] + "\n")
    try:
        _read_fixed(path, ERIM_PURCHASE_WIDTHS, ERIM_PURCHASE_COLUMNS, numeric)
    except ValueError as error:
        assert "record length" in str(error)
    else:
        raise AssertionError("invalid fixed-width record was accepted")


def test_erim_product_reader_uses_upc_description_and_size_unit(tmp_path):
    path = tmp_path / "products.dat"
    path.write_text(
        "0000000000123" + "TEST KETCHUP".ljust(30) + "GR" + "OUNCES" +
        "0032000" + "001" + "00000010000" + "0001\n")
    assert read_erim_products(path) == {123: "TEST KETCHUP (OUNCES)"}


def write_generic_long_choices(path, *, seed=39):
    rng = np.random.default_rng(seed)
    rows = []
    for customer in range(30):
        income = 20 + customer
        for occasion_number in range(5):
            occasion = customer * 5 + occasion_number
            prices = rng.uniform(.5, 2.0, 3)
            utilities = np.array([.3, 0, -.2]) - 1.8 * np.log(prices)
            probabilities = np.exp(utilities - np.logaddexp.reduce(utilities))
            selected = rng.choice(3, p=probabilities)
            for product in range(3):
                rows.append({
                    "visit": occasion,
                    "shopper": f"h{customer}",
                    "sku": f"p{product}",
                    "amount": prices[product],
                    "selected": int(product == selected),
                    "period": occasion_number,
                    "income": income,
                })
    pd.DataFrame(rows).to_csv(path, index=False)


def test_generic_long_adapter_maps_columns_without_dataset_code(tmp_path):
    path = tmp_path / "choices.csv"
    write_generic_long_choices(path)
    adapter = LongChoiceCsvAdapter(
        path, "generic_test", occasion_column="visit", group_column="shopper",
        product_column="sku", price_column="amount", chosen_column="selected",
        period_column="period", covariate_columns=("income",),
        split_strategy=TemporalSplit(block_name="period"),
        capabilities=DatasetCapabilities(
            temporal_order=True, known_assortment=True),
    )
    dataset = adapter.load()
    assert dataset.panel.product.shape == (150, 3)
    assert dataset.product_labels == ("p0", "p1", "p2")
    assert dataset.periods.shape == (150,)
    split = dataset.split_strategy.split(dataset, 1)
    assert set(dataset.periods[split.train]) == {0, 1, 2}
    assert set(dataset.periods[split.validation]) == {3}
    assert set(dataset.periods[split.test]) == {4}


def test_adapter_factory_and_generic_engine_are_dataset_independent(tmp_path):
    path = tmp_path / "choices.csv"
    write_generic_long_choices(path)
    specification = {
        "type": "long_choice_csv",
        "id": "configured_retailer",
        "sources": {"choices": "choices.csv"},
        "columns": {
            "occasion": "visit", "group": "shopper", "product": "sku",
            "price": "amount", "chosen": "selected", "covariates": ["income"],
        },
        "split": {"type": "grouped_random", "minimum_group_size": 5},
        "capabilities": {"known_assortment": True},
    }
    adapter = adapter_from_config(specification, base_directory=tmp_path)
    assert isinstance(adapter, LongChoiceCsvAdapter)
    result = ExternalChoiceExperiment(ValidationSettings(
        seed=7, regularization_grid=(.001,), bootstrap_replicates=20,
        placebo_replicates=20)).run([adapter])
    dataset_result = result["datasets"]["configured_retailer"]
    assert dataset_result["partitions"] == {"train": 90, "validation": 30, "test": 30}
    assert not dataset_result["full_basket_pipeline"]["eligible"]
    assert "full_baskets" in dataset_result["full_basket_pipeline"]["blockers"]


def test_generic_adapter_rejects_multiple_choices_per_occasion(tmp_path):
    path = tmp_path / "choices.csv"
    write_generic_long_choices(path)
    frame = pd.read_csv(path)
    frame.loc[frame.visit == 0, "selected"] = 1
    frame.to_csv(path, index=False)
    adapter = LongChoiceCsvAdapter(
        path, "invalid", occasion_column="visit", group_column="shopper",
        product_column="sku", price_column="amount", chosen_column="selected")
    try:
        adapter.load()
    except ValueError as error:
        assert "exactly one chosen" in str(error)
    else:
        raise AssertionError("multiple chosen alternatives were accepted")
