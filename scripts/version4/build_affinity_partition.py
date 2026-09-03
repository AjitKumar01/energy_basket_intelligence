#!/usr/bin/env python3
"""Build the training-only co-purchase partition used by version4.html.

The partition is a deterministic preprocessing choice.  It is learned only from training
baskets and caps non-residual groups at 128 products, keeping the exact category
convolution tractable.  Products that never join a supported pair share one residual
group; singleton groups would add computation while their pair statistic is identically
zero.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data import BI


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-pair-count", type=int, default=8)
    parser.add_argument("--maximum-group-size", type=int, default=128)
    parser.add_argument("--output", type=Path,
                        default=Path(BI) / "items_affinity.parquet")
    args = parser.parse_args()

    baskets = pd.read_parquet(
        Path(BI) / "baskets.parquet", columns=["BASKET_ID", "item_id", "split"])
    items = pd.read_parquet(Path(BI) / "items.parquet", columns=["item_id"])
    products = len(items)
    if not np.array_equal(np.sort(items.item_id.unique()), np.arange(products)):
        raise RuntimeError("item_id must be complete and contiguous")
    training = baskets[baskets.split == "train"]
    incidence = np.zeros(products, dtype=np.int64)
    pairs: dict[tuple[int, int], int] = {}
    for _basket, group in training.groupby("BASKET_ID", sort=False):
        trip_items = sorted(set(map(int, group.item_id.to_numpy())))
        incidence[trip_items] += 1
        for left in range(len(trip_items)):
            for right in range(left + 1, len(trip_items)):
                key = (trip_items[left], trip_items[right])
                pairs[key] = pairs.get(key, 0) + 1

    n = training.BASKET_ID.nunique()
    edges = []
    for (left, right), count in pairs.items():
        if count < args.minimum_pair_count:
            continue
        lift = (count * n) / max(int(incidence[left]) * int(incidence[right]), 1)
        if lift > 1.0:
            # Count, rather than lift, maximizes retained observed co-purchase mass.
            edges.append((count, left, right))
    edges.sort(reverse=True)

    parent = np.arange(products)
    size = np.ones(products, dtype=np.int64)

    def find(item: int) -> int:
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = int(parent[item])
        return item

    for _count, left, right in edges:
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            continue
        if size[root_left] + size[root_right] > args.maximum_group_size:
            continue
        if size[root_left] < size[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        size[root_left] += size[root_right]

    root = np.asarray([find(item) for item in range(products)])
    root_size = np.bincount(root, minlength=products)
    root[root_size[root] == 1] = -1
    _unique, category = np.unique(root, return_inverse=True)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"item_id": np.arange(products),
                  "cat_id": category}).to_parquet(output, index=False)
    groups = int(category.max()) + 1
    sizes = np.bincount(category, minlength=groups)
    if len(category) != products or set(np.unique(category)) != set(range(groups)):
        raise RuntimeError("affinity partition is incomplete or non-contiguous")
    # Category zero pools isolated products; every connected group remains bounded so
    # the exact within-category convolution has the declared complexity.
    if groups > 1 and int(sizes[1:].max()) > args.maximum_group_size:
        raise RuntimeError("a non-residual affinity group exceeds the complexity cap")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1,
        "training_only": True,
        "n_items": products,
        "n_groups": groups,
        "residual_group": 0,
        "residual_group_size": int(sizes[0]),
        "maximum_non_residual_group_size": int(sizes[1:].max()) if groups > 1 else 0,
        "minimum_pair_count": args.minimum_pair_count,
        "maximum_group_size": args.maximum_group_size,
        "partition_sha256": digest,
    }
    output.with_name("affinity_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n")
    print(f"[affinity] wrote {output}: {products} products, "
          f"{groups} groups; residual {int(sizes[0])}, maximum connected "
          f"{int(sizes[1:].max()) if groups > 1 else 0}")


if __name__ == "__main__":
    main()
