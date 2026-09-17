"""Tests for model-free catalogue partitions and their partition checks."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from catalogue_partition import (PartitionError, catalogue_hierarchy_groups, finest_level_groups,
                                 validate_partition, write_partition)

ROOT = Path(__file__).resolve().parents[1]


def catalogue(sub=None, category=None):
    category = category or ["a"] * 7 + ["b"] * 2
    return pd.DataFrame({
        "item_id": list(range(len(category))),
        "cat_id": [sorted(set(category)).index(c) for c in category],
        "COMMODITY_DESC": category,
        "SUB_COMMODITY_DESC": sub or ["x", "x", "x", "y", "y", "y", "a", "b", "b"],
        "n_train_lines": [200, 200, 200, 10, 10, 10, 50, 40, 40],
        "DEPARTMENT": ["d"] * len(category),
    })


def test_finest_level_uses_every_declared_subcategory_and_a_remainder():
    groups, decisions = finest_level_groups(catalogue())
    # a: remainder (undeclared product 6) = 0, x = 1, y = 2; b has no subcategory = 3
    assert groups.tolist() == [1, 1, 1, 2, 2, 2, 0, 3, 3]
    assert decisions["a"]["groups"] == ["(category remainder)", "x", "y"]
    assert validate_partition(catalogue(), groups)["n_groups"] == 4


def test_finest_level_equals_hierarchy_without_floors_and_category_without_subcategories():
    assert np.array_equal(finest_level_groups(catalogue())[0],
                          catalogue_hierarchy_groups(catalogue(), 1, 0)[0])
    flat = catalogue(sub=["a"] * 7 + ["b"] * 2)
    assert finest_level_groups(flat)[0].tolist() == flat.cat_id.tolist()


def test_reused_subcategory_labels_stay_in_separate_groups():
    items = catalogue(sub=["x", "x", "x", "y", "y", "y", "a", "x", "x"])   # "x" also under b
    groups, _ = finest_level_groups(items)
    assert len(set(groups[[0, 1, 2]])) == 1 and len(set(groups[[7, 8]])) == 1
    assert groups[0] != groups[7]
    diagnostics = validate_partition(items, groups)
    assert diagnostics["subcategory_labels_reused_across_categories"] == 1
    assert diagnostics["nests_in_category"]


def test_blank_subcategories_count_as_undeclared():
    items = catalogue(sub=["x", "x", "x", "", None, "", "a", "b", "b"])
    groups, _ = finest_level_groups(items)
    assert groups[3] == groups[4] == groups[5] == groups[6]


@pytest.mark.parametrize("groups, message", [
    ([0, 0, 0, 0, 0, 0, 0, 0], "covers"),
    ([0, 0, 0, 0, 0, 0, 0, 2, 2], "contiguous"),
    ([0, 0, 0, 0, 0, 0, 0, -1, -1], "nonnegative"),
    ([0, 0, 0, 0, 0, 0, 1, 1, 1], "span more than one category"),
])
def test_invalid_partitions_are_rejected(groups, message):
    with pytest.raises(PartitionError, match=message):
        validate_partition(catalogue(), np.array(groups))


def test_written_partition_matches_manifest(tmp_path):
    import hashlib
    items = catalogue()
    groups, decisions = finest_level_groups(items)
    manifest = write_partition(tmp_path, items, groups, {"schema_version": 1, "training_only": True,
                                                          "partition": "finest_catalogue_level",
                                                          "categories": decisions})
    written = pd.read_parquet(tmp_path / "items_affinity.parquet")
    assert written.cat_id.tolist() == groups.tolist()
    assert manifest["n_groups"] == 4 and list(manifest)[-2:] == ["categories", "partition_sha256"]
    assert manifest["partition_sha256"] == hashlib.sha256(
        (tmp_path / "items_affinity.parquet").read_bytes()).hexdigest()


def test_cli_refuses_to_replace_an_existing_partition(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    catalogue().to_parquet(source / "items.parquet")
    command = [sys.executable, str(ROOT / "scripts/version4/build_catalogue_partition.py"),
               "--basket-input", str(source), "--rule", "finest_catalogue_level",
               "--output-dir", str(tmp_path / "out")]
    assert subprocess.run(command, capture_output=True, cwd=ROOT / "scripts/version4").returncode == 0
    again = subprocess.run(command, capture_output=True, text=True, cwd=ROOT / "scripts/version4")
    assert again.returncode != 0 and "--force" in again.stderr
    manifest = json.loads((tmp_path / "out" / "affinity_manifest.json").read_text())
    assert manifest["partition"] == "finest_catalogue_level" and manifest["validation"]["is_partition"]
