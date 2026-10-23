#!/usr/bin/env python3
"""Stage 1 (alternative): the model's categories from the product hierarchy.

One group per commodity (``--column COMMODITY_DESC``, the default) or per sub-commodity, so
products that compete for the same need sit in the same group.  That is what the
within-category terms (rho_c, f_c, the competition shift) need: they can only make rivals
compete when the rivals share a group.  The co-purchase partition
(build_affinity_partition.py) groups products bought *together*, which splits rivals apart
(soft drinks over 41 groups) and pools 1,724 products in one leftover group.

The exact dynamic program costs about (products offered) x R per trip whatever the group
sizes, so no size cap is needed.  Uses product descriptions only (no purchases), so it is
training-only by construction.

Writes basket_input/items_affinity.parquet (item_id, cat_id) and affinity_manifest.json, the
files the model reads.

    build_commodity_partition.py --data-root BUNDLE [--column COMMODITY_DESC]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from v4.bundle import data_root  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, help="model-data bundle (default $ENERGY_MODEL_DATA_ROOT)")
    p.add_argument("--column", default="COMMODITY_DESC", choices=["COMMODITY_DESC", "SUB_COMMODITY_DESC"])
    p.add_argument("--output", type=Path, help="default: the bundle's basket_input/items_affinity.parquet")
    args = p.parse_args()
    folder = data_root(args.data_root) / "basket_input"
    output = (args.output or folder / "items_affinity.parquet").resolve()

    items = pd.read_parquet(folder / "items.parquet").sort_values("item_id").reset_index(drop=True)
    products = len(items)
    if not np.array_equal(items.item_id.to_numpy(), np.arange(products)):
        raise RuntimeError("item_id must be complete and contiguous")
    keys = items.COMMODITY_DESC.astype(str)
    if args.column == "SUB_COMMODITY_DESC":                # sub-commodity names repeat across commodities
        keys = keys + " / " + items.SUB_COMMODITY_DESC.astype(str)
    names, category = np.unique(keys.to_numpy(), return_inverse=True)

    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"item_id": np.arange(products), "cat_id": category}).to_parquet(output, index=False)
    sizes = np.bincount(category)
    manifest = {
        "schema_version": 1,
        "training_only": True,
        "source": f"product hierarchy: {args.column}",
        "n_items": products,
        "n_groups": int(len(names)),
        "residual_group": None,
        "maximum_group_size": int(sizes.max()),
        "groups": {str(n): int(s) for n, s in zip(names, sizes)},
        "partition_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    output.with_name("affinity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[partition] wrote {output}: {products} products, {len(names)} groups by {args.column}; "
          f"largest {int(sizes.max())}, median {int(np.median(sizes))}, single-product {int((sizes == 1).sum())}")


if __name__ == "__main__":
    main()
