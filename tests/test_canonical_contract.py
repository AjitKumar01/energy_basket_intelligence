"""Tests for the dataset-neutral canonical input contract and bundle preparation."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from canonical_basket_input import CanonicalBasketModelInputBuilder
from canonical_contract import CanonicalContractError, validate_canonical_tables

ROOT = Path(__file__).resolve().parents[1]


def tiny_tables():
    products = pd.DataFrame({"item_id": [0, 1, 2], "product_id": ["p0", "p1", "p2"],
                             "category": ["a", "a", "b"], "label": ["x", "y", "z"]})
    rows = []
    basket = 0
    for split, periods in (("train", (1, 2)), ("validation", (3,)), ("test", (4,))):
        for period in periods:
            for customer in (0, 1):
                items = [0, 2] if customer == 0 else [1]
                for item in items:
                    rows.append(dict(basket_id=basket, customer_id=customer, store_id=customer,
                                     period=period, day=(period - 1) * 7 + 2,
                                     product_id=f"p{item}", item_id=item,
                                     category=products.category[item], quantity=1,
                                     unit_price=1.5, split=split))
                basket += 1
    tx = pd.DataFrame(rows)
    prices = pd.DataFrame({"product_id": ["p0", "p1"], "store_id": [0, 1], "period": [1, 2],
                           "units": [3.0, 4.0], "revenue": [4.5, 6.0],
                           "price_source": ["retail_aggregate", "retail_aggregate"]})
    promotions = pd.DataFrame(columns=["product_id", "store_id", "period", "display",
                                       "advertised", "special_price"])
    audit = {"source_sha256": {}, "audit": {"cohort_policy": {"minimum_product_training_lines": 1}}}
    return {"transactions": tx, "products": products, "store_week_prices": prices,
            "promotions": promotions, "build_audit": audit, "shopping_opportunities": None}


def test_valid_tables_pass_and_summarize():
    summary = validate_canonical_tables(tiny_tables())
    assert summary["n_items"] == 3 and summary["n_customers"] == 2 and summary["n_stores"] == 2
    assert summary["split_periods"]["test"] == [4, 4]


@pytest.mark.parametrize("mutate, message", [
    (lambda t: t["transactions"].drop(columns="unit_price", inplace=True), "missing required"),
    (lambda t: t["products"].loc.__setitem__((2, "item_id"), 5), "contiguous"),
    (lambda t: t["transactions"].loc.__setitem__((0, "day"), 30), "day // 7"),
    (lambda t: t["transactions"].loc.__setitem__((0, "quantity"), 1.5), "positive integers"),
    (lambda t: t["transactions"].loc.__setitem__(
        (t["transactions"].split == "test", "split"), "validation"), "exactly"),
    (lambda t: t["store_week_prices"].loc.__setitem__((0, "price_source"), "guess"), "price_source"),
    (lambda t: t["build_audit"]["audit"].pop("cohort_policy"), "cohort_policy"),
])
def test_contract_violations_are_rejected(mutate, message):
    tables = tiny_tables()
    mutate(tables)
    with pytest.raises(CanonicalContractError, match=message):
        validate_canonical_tables(tables)


def test_products_without_training_purchases_are_rejected():
    tables = tiny_tables()
    tx = tables["transactions"]
    tables["transactions"] = tx[~((tx.item_id == 2) & (tx.split == "train"))]
    with pytest.raises(CanonicalContractError, match="no training purchases"):
        validate_canonical_tables(tables)


def test_catalogue_metadata_comes_from_optional_columns_or_defaults():
    tables = tiny_tables()
    products = tables["products"].assign(subcategory=["s1", "s2", "s1"], brand=["B", "C", "D"])
    items = CanonicalBasketModelInputBuilder._items(
        products, tables["transactions"], {"DEPARTMENT": "GROCERY"})
    assert items.SUB_COMMODITY_DESC.tolist() == ["s1", "s2", "s1"]
    assert items.sub_id.tolist() == [0, 1, 0]
    assert items.BRAND.tolist() == ["B", "C", "D"]
    assert set(items.DEPARTMENT) == {"GROCERY"} and set(items.MANUFACTURER) == {"UNKNOWN"}
    plain = CanonicalBasketModelInputBuilder._items(tables["products"], tables["transactions"])
    assert plain.sub_id.tolist() == plain.cat_id.tolist()


def test_builder_rejects_unknown_metadata_defaults(tmp_path):
    with pytest.raises(ValueError, match="metadata defaults"):
        CanonicalBasketModelInputBuilder(tmp_path, tmp_path, price_basis="p",
                                         metadata_defaults={"COLOUR": "red"})


def test_dataset_config_rejects_unknown_and_missing_keys(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prepare_bundle", ROOT / "scripts" / "prepare_model_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = tmp_path / "config.json"
    path.write_text(json.dumps({"schema_version": 1, "dataset_name": "d", "canonical_dir": "c",
                                "model_data_root": "m", "price_basis": "p", "colour": 1}))
    with pytest.raises(SystemExit, match="unknown"):
        module.load_config(path)
    path.write_text(json.dumps({"schema_version": 1, "dataset_name": "d"}))
    with pytest.raises(SystemExit, match="canonical_dir"):
        module.load_config(path)


def test_shipped_dataset_configs_are_valid():
    spec = importlib.util.spec_from_file_location(
        "prepare_bundle_configs", ROOT / "scripts" / "prepare_model_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    configs = sorted((ROOT / "configs" / "datasets").glob("*.json"))
    assert configs
    for path in configs:
        module.load_config(path)


def test_category_partition_writes_a_manifest_the_initializer_accepts(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "prepare_bundle_partition", ROOT / "scripts" / "prepare_model_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    pd.DataFrame({"item_id": [2, 0, 1], "cat_id": [1, 0, 1]}).to_parquet(tmp_path / "items.parquet")
    module.write_category_partition(tmp_path)
    partition = pd.read_parquet(tmp_path / "items_affinity.parquet")
    manifest = json.loads((tmp_path / "affinity_manifest.json").read_text())
    assert partition.item_id.tolist() == [0, 1, 2] and partition.cat_id.tolist() == [0, 1, 1]
    assert manifest["training_only"] and manifest["n_groups"] == 2 and manifest["n_items"] == 3
    import hashlib
    assert manifest["partition_sha256"] == hashlib.sha256(
        (tmp_path / "items_affinity.parquet").read_bytes()).hexdigest()


def load_prepare_module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "prepare_model_bundle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hierarchy_items():
    # category a: subcategories x (3 products, enough lines), y (3 products, too few lines),
    # one undeclared product; category b: two products, no declared subcategory
    return pd.DataFrame({
        "item_id": list(range(9)),
        "cat_id": [0] * 7 + [1] * 2,
        "COMMODITY_DESC": ["a"] * 7 + ["b"] * 2,
        "SUB_COMMODITY_DESC": ["x", "x", "x", "y", "y", "y", "a", "b", "b"],
        "n_train_lines": [200, 200, 200, 10, 10, 10, 50, 40, 40],
    })


def test_catalogue_hierarchy_splits_only_qualifying_subcategories():
    module = load_prepare_module("prepare_bundle_hierarchy")
    groups, decisions = module.catalogue_hierarchy_groups(hierarchy_items(), 3, 300)
    # remainder of a (y products + undeclared) = 0, subcategory x = 1, category b = 2
    assert groups.tolist() == [1, 1, 1, 0, 0, 0, 0, 2, 2]
    assert decisions["a"]["groups"] == ["(category remainder)", "x"]
    assert decisions["a"]["declared_subcategories"]["y"]["own_group"] is False
    assert decisions["b"]["groups"] == ["(category remainder)"]


def test_catalogue_hierarchy_without_subcategories_equals_category_partition():
    module = load_prepare_module("prepare_bundle_hierarchy_noop")
    items = hierarchy_items().assign(SUB_COMMODITY_DESC=lambda f: f.COMMODITY_DESC)
    groups, _ = module.catalogue_hierarchy_groups(items, 3, 300)
    assert groups.tolist() == items.cat_id.tolist()


def test_catalogue_hierarchy_floors_are_declared_not_inferred(tmp_path):
    module = load_prepare_module("prepare_bundle_hierarchy_floors")
    loose, _ = module.catalogue_hierarchy_groups(hierarchy_items(), 2, 20)
    assert len(set(loose.tolist())) == 4          # y now qualifies as well
    config = json.loads((ROOT / "configs" / "datasets" / "erim_availability_catalogue.json").read_text())
    config["affinity"]["minimum_pair_count"] = 8   # an affinity-partition key
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    with pytest.raises(SystemExit, match="do not apply"):
        module.load_config(path)


def test_product_metadata_adds_declared_columns_and_validates(tmp_path):
    products = tiny_tables()["products"]
    path = tmp_path / "meta.parquet"
    pd.DataFrame({"product_id": ["p0"], "subcategory": ["fine"]}).to_parquet(path)
    merged, audit = CanonicalBasketModelInputBuilder._apply_product_metadata(products, path)
    assert merged.subcategory.tolist() == ["fine", "a", "b"]      # others fall back to category
    assert audit["columns"] == ["subcategory"] and audit["products_covered"] == 1
    for bad, message in (
            (pd.DataFrame({"product_id": ["p9"], "subcategory": ["s"]}), "unknown products"),
            (pd.DataFrame({"product_id": ["p0", "p0"], "subcategory": ["s", "t"]}), "more than once"),
            (pd.DataFrame({"product_id": ["p0"], "category": ["z"]}), "subset"),
            (pd.DataFrame({"product_id": ["p0"], "brand": ["B"]}), "cover every product")):
        bad.to_parquet(path)
        with pytest.raises(ValueError, match=message):
            CanonicalBasketModelInputBuilder._apply_product_metadata(products, path)
