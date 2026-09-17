"""Model-free product partitions from a declared catalogue hierarchy.

A dataset declares its catalogue at the levels it has: ERIM stops at category, Dunnhumby goes
down to sub-commodity. In a model-data bundle these levels are the ``items.parquet`` columns
``COMMODITY_DESC`` (category) and ``SUB_COMMODITY_DESC`` (subcategory; equal to the category
when a dataset does not declare one).

Groups are keyed by the **full path** (category, subcategory), never by a subcategory label
alone. Real catalogues reuse labels under different parents (Dunnhumby reuses 39
sub-commodity labels across commodities); keying by path keeps those products apart and makes
every group nest inside exactly one category.

Rules:
  category                one group per category
  finest_catalogue_level  one group per declared (category, subcategory); products without a
                          declared subcategory form their category's remainder group
  catalogue_hierarchy     as finest_catalogue_level, but a subcategory forms its own group only
                          if it meets fixed product and training-line floors; the rest are
                          pooled into the category remainder group

``validate_partition`` checks that a grouping is a partition consistent with the hierarchy.
The department level is not part of the path; departments that disagree with categories are
reported, not used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SEP = "\x1f"
HIERARCHY_RULES = ("category", "finest_catalogue_level", "catalogue_hierarchy")
HIERARCHY_DEFAULTS = {"minimum_group_products": 3, "minimum_group_training_lines": 300}
REQUIRED_COLUMNS = ("item_id", "cat_id", "COMMODITY_DESC", "SUB_COMMODITY_DESC", "n_train_lines")


class PartitionError(ValueError):
    """A grouping that is not a valid partition of the catalogue."""


def _labels(items: pd.DataFrame):
    missing = [c for c in REQUIRED_COLUMNS if c not in items]
    if missing:
        raise PartitionError(f"items table lacks catalogue columns {missing}")
    items = items.sort_values("item_id").reset_index(drop=True)
    if not np.array_equal(items.item_id.to_numpy(), np.arange(len(items))):
        raise PartitionError("items.item_id must be contiguous 0..J-1")
    category = items.COMMODITY_DESC
    if category.isna().any() or (category.astype(str).str.strip() == "").any():
        raise PartitionError("every product needs a category label")
    category = category.astype(str)
    raw_sub = items.SUB_COMMODITY_DESC
    blank = raw_sub.isna() | (raw_sub.astype(str).str.strip() == "")
    sub = raw_sub.astype(str).where(~blank, category)
    declared = sub != category
    return items, category, sub, declared


def catalogue_hierarchy_groups(items: pd.DataFrame, minimum_products: int, minimum_training_lines: int):
    """Group ids [J] and per-category decisions for the floored hierarchy rule.

    Groups are numbered by (category id, subcategory label), with the remainder first.
    """
    items, category, sub, declared = _labels(items)
    key = category + SEP + sub
    stats = items[declared].groupby(key[declared]).agg(
        products=("item_id", "size"), training_lines=("n_train_lines", "sum"))
    qualifying = set(stats[(stats.products >= minimum_products)
                           & (stats.training_lines >= minimum_training_lines)].index)
    group_label = np.where(key.isin(qualifying), key, category + SEP)   # "" = remainder
    first_cat = pd.Series(items.cat_id.to_numpy()).groupby(group_label).first()
    order = sorted(first_cat.index, key=lambda g: (int(first_cat[g]), g.split(SEP, 1)[1]))
    index = {g: i for i, g in enumerate(order)}
    group_id = np.array([index[g] for g in group_label], dtype=np.int32)
    decisions = {}
    for cat in sorted(category.unique()):
        cat_stats = stats[stats.index.str.startswith(cat + SEP)]
        decisions[cat] = {
            "groups": [g.split(SEP, 1)[1] or "(category remainder)"
                       for g in order if g.startswith(cat + SEP)],
            "declared_subcategories": {
                name.split(SEP, 1)[1]: {"products": int(row.products),
                                        "training_lines": int(row.training_lines),
                                        "own_group": name in qualifying}
                for name, row in cat_stats.iterrows()},
        }
    return group_id, decisions


def finest_level_groups(items: pd.DataFrame):
    """Group ids [J] at each product's finest declared level (no floors)."""
    return catalogue_hierarchy_groups(items, 1, 0)


def validate_partition(items: pd.DataFrame, group_id, nest_in_category: bool = True) -> dict:
    """Check that group_id is a partition that (by default) nests in the category level.

    Raises PartitionError on: wrong length, missing or negative ids, non-contiguous ids, or a
    group spanning more than one category. Returns diagnostics about the declared hierarchy.
    """
    items, category, sub, declared = _labels(items)
    group_id = np.asarray(group_id)
    if group_id.shape != (len(items),):
        raise PartitionError(f"partition covers {group_id.shape} products, catalogue has {len(items)}")
    if not np.issubdtype(group_id.dtype, np.integer) or (group_id < 0).any():
        raise PartitionError("group ids must be nonnegative integers, one per product")
    groups = np.unique(group_id)
    if not np.array_equal(groups, np.arange(len(groups))):
        raise PartitionError("group ids must be contiguous 0..G-1")
    spanning = pd.Series(category.to_numpy()).groupby(group_id).nunique()
    if nest_in_category and (spanning > 1).any():
        raise PartitionError(f"{int((spanning > 1).sum())} groups span more than one category")
    sizes = np.bincount(group_id)
    reused = (pd.DataFrame({"sub": sub[declared], "category": category[declared]})
              .groupby("sub").category.nunique())
    diagnostics = {
        "is_partition": True, "nests_in_category": bool((spanning <= 1).all()),
        "n_categories": int(category.nunique()), "n_groups": int(len(sizes)),
        "products_with_declared_subcategory": int(declared.sum()),
        "subcategory_labels_reused_across_categories": int((reused > 1).sum()),
        "group_size": {"singletons": int((sizes == 1).sum()), "median": float(np.median(sizes)),
                       "maximum": int(sizes.max())},
    }
    if "DEPARTMENT" in items:
        per_category = items.groupby(category).DEPARTMENT.nunique()
        diagnostics["categories_in_more_than_one_department"] = int((per_category > 1).sum())
    return diagnostics


def write_partition(output_dir: Path, items: pd.DataFrame, group_id, manifest: dict) -> dict:
    """Write items_affinity.parquet and affinity_manifest.json in the format the model reads."""
    items = items.sort_values("item_id")
    output = Path(output_dir) / "items_affinity.parquet"
    pd.DataFrame({"item_id": items.item_id.to_numpy(), "cat_id": np.asarray(group_id, dtype=np.int32)}
                 ).to_parquet(output, index=False)
    sizes = np.bincount(np.asarray(group_id))
    manifest = {**manifest, "n_items": int(len(items)), "n_groups": int(len(sizes)),
                "maximum_group_size_observed": int(sizes.max())}
    categories = manifest.pop("categories", None)
    if categories is not None:
        manifest["categories"] = categories
    manifest["partition_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
    (Path(output_dir) / "affinity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
