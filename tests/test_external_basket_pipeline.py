import json
import zipfile

import numpy as np
import pandas as pd
import pytest

from canonical_basket_input import CanonicalBasketModelInputBuilder
from scripts.data.fetch_erim_categories import safe_extract
from features import Features
from fit_exact_additive import (load_supported_price_sensitivity,
                                material_validation_patience)
from fit_stratified_natural_interactions import default_size_knots
from audit_customer_segments import standardized_block
from diagnose_size_phase import interval_summary


def model_items():
    return pd.DataFrame({
        "item_id": [0], "product_id": ["category:1"], "sub_id": [0],
    })


def model_transactions(quantity):
    return pd.DataFrame({
        "basket_id": [0], "customer_id": [0], "day": [0], "period": [1],
        "quantity": [quantity], "unit_price": [1.0], "store_id": [0],
        "split": ["train"], "item_id": [0],
    })


def test_model_bridge_rejects_fractional_quantity_instead_of_rounding():
    with pytest.raises(ValueError, match="positive integers"):
        CanonicalBasketModelInputBuilder._baskets(
            model_transactions(1.25), model_items())
    basket = CanonicalBasketModelInputBuilder._baskets(
        model_transactions(2.0), model_items())
    assert basket.units.tolist() == [2]


def test_disabled_promotions_keep_observed_audit_panel(tmp_path):
    builder = CanonicalBasketModelInputBuilder(
        tmp_path, tmp_path / "output", price_basis="paid_price",
        promotion_feature_name="disabled")
    output = tmp_path / "basket_input"
    output.mkdir()
    promotions = pd.DataFrame({
        "product_id": ["category:1"], "store_id": [0], "period": [1],
        "display": [True], "advertised": [False], "special_price": [True],
    })
    result = builder._write_promotions(promotions, model_items(), 1, 2, output)
    with np.load(output / "promo.npz") as modeled:
        assert len(modeled["keys"]) == 0
    with np.load(output / "promo_observed.npz") as observed:
        assert len(observed["keys"]) == 1
        assert observed["disp"].tolist() == [1]
        assert observed["mail"].tolist() == [1]
    assert not result["enabled"]


def test_empty_sparse_feature_lookup_is_exact_zero():
    query = __import__("torch").tensor([1, 17], dtype=__import__("torch").long)
    values = __import__("torch").empty(0)
    keys = __import__("torch").empty(0, dtype=__import__("torch").long)
    assert Features._lookup(keys, values, query).tolist() == [0.0, 0.0]


def test_unsupported_price_products_are_fixed_to_zero(tmp_path):
    artifact = tmp_path / "coefficients.json"
    artifact.write_text(json.dumps({
        "status": "certified_observational_predictor",
        "level": "product",
        "global_sensitivity": 1.5,
        "supported_item_ids": [1, 3],
        "product_sensitivity": [{"item_id": 1, "sensitivity": 2.0}],
        "unsupported_catalogue_products_use_global_sensitivity": False,
    }))
    sensitivity, level, audit = load_supported_price_sensitivity(artifact, 5)
    assert level == "product"
    assert sensitivity.tolist() == [0.0, 2.0, 0.0, 1.5, 0.0]
    assert audit["unsupported_products_fixed_to_zero"] == 3


def test_safe_extract_rejects_zip_slip(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as stream:
        stream.writestr("../outside.dat", "bad")
    with pytest.raises(ValueError, match="unsafe archive member"):
        safe_extract(archive, tmp_path / "target")
    assert not (tmp_path / "outside.dat").exists()


def test_convergence_patience_ignores_subthreshold_numerical_drift():
    evaluations = [
        {"basket_loglik": -8.0},
        {"basket_loglik": -7.9996},
        {"basket_loglik": -7.9994},
        {"basket_loglik": -7.9988},
        {"basket_loglik": -7.9987},
        {"basket_loglik": -7.9986},
    ]
    # The third update is the only improvement exceeding 0.001 from the
    # current material anchor; the two later microscopic gains count as patience.
    assert material_validation_patience(evaluations, .001) == 2


def test_interaction_size_knots_follow_checkpoint_support():
    assert default_size_knots(16) == [1, 2, 3, 4, 5, 7, 10, 15, 16]
    assert default_size_knots(120)[-1] == 120


def test_segment_representation_accepts_disabled_feature_block():
    values = np.empty((7, 0))
    result = standardized_block(values)
    assert result.shape == (7, 0)


def test_size_phase_intervals_clip_to_small_checkpoint_support():
    result = interval_summary(np.zeros((3, 15)))
    assert list(result) == ["1:5", "5:11", "11:16"]
    assert all(value["maximum"] == 0.0 for value in result.values())


def _price_fixture():
    items = pd.DataFrame({
        "item_id": [0, 1], "product_id": ["retail:1", "fallback:2"], "sub_id": [0, 0]})
    rows = []
    # item 0: retail prices in weeks 1-2, a purchase-derived price in week 3
    rows += [("retail:1", 0, 1, 10, 20.0, "retail_aggregate"),
             ("retail:1", 0, 2, 10, 30.0, "retail_aggregate"),
             ("retail:1", 0, 3, 1, 1.0, "purchase_aggregate")]
    # item 1: purchase-derived prices only (training weeks 1-2 and test week 4)
    rows += [("fallback:2", 0, 1, 2, 4.0, "purchase_aggregate"),
             ("fallback:2", 0, 2, 2, 6.0, "purchase_aggregate"),
             ("fallback:2", 0, 4, 1, 9.0, "purchase_aggregate")]
    prices = pd.DataFrame(rows, columns=[
        "product_id", "store_id", "period", "units", "revenue", "price_source"])
    return items, prices


def test_model_prices_exclude_purchase_derived_cells_by_default(tmp_path):
    items, prices = _price_fixture()
    for name in ("basket_input", "data"):
        (tmp_path / name).mkdir()
    builder = CanonicalBasketModelInputBuilder(tmp_path, tmp_path, price_basis="retail")
    audit = builder._write_prices(prices, items, 1, 4, 28, 2, tmp_path / "basket_input")
    log_price = np.load(tmp_path / "basket_input" / "log_price.npy")
    dev = np.load(tmp_path / "basket_input" / "log_price_dev.npy")
    # week 3's purchase-derived 1.00 is ignored; retail week 2 (3.00) is carried forward
    assert np.allclose(np.exp(log_price[0, 14:28]), 3.0)
    # the purchase-only item gets its constant training reference price and zero deviation
    assert np.allclose(np.exp(log_price[1]), 2.5) and np.all(dev[1] == 0.0)
    assert audit["constant_reference_price_items"] == [1]
    assert audit["excluded_source_item_period_cells"] == 4


def test_including_purchase_sources_reproduces_the_legacy_price_panel(tmp_path):
    items, prices = _price_fixture()
    for name in ("basket_input", "data"):
        (tmp_path / name).mkdir()
    builder = CanonicalBasketModelInputBuilder(
        tmp_path, tmp_path, price_basis="legacy",
        model_price_sources=("retail_aggregate", "purchase_aggregate"))
    audit = builder._write_prices(prices, items, 1, 4, 28, 2, tmp_path / "basket_input")
    log_price = np.load(tmp_path / "basket_input" / "log_price.npy")
    assert np.allclose(np.exp(log_price[0, 14:21]), 1.0)
    assert audit["constant_reference_price_items"] == []
