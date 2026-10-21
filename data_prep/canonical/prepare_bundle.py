#!/usr/bin/env python3
"""Prepare a model-data bundle from a canonical basket directory and a dataset config.

Any dataset enters the pipeline as a *canonical directory* (contract.py: transactions,
products, store-week prices, promotions, optional shopping opportunities, build audit).
This script validates it and writes the bundle the model reads:

  1. basket_input/ and data/ from the canonical tables (build_model_input.py): baskets,
     items, the chain price panel, store deviations, promotions, purchase-history state and,
     optionally, availability;
  2. the category partition (items_affinity.parquet + affinity_manifest.json):
       "category"  the declared merchandise categories, or
       "affinity"  groups grown from training co-purchases (scripts/build_affinity_partition.py);
  3. the ragged assortment index (v4/bundle.py);
  4. the data fingerprint that every checkpoint records.

The config (JSON) holds every dataset-specific choice, for example
data_prep/configs/stress_world.json:

    {"schema_version": 1, "dataset_name": "...",
     "price_basis": "...", "promotion_feature": "advertised | special_price | disabled",
     "model_price_sources": ["retail_aggregate"],
     "availability": {"rule": "disabled | retail_first_sale", "left_censor_periods": 13},
     "affinity": {"partition": "category | affinity", "minimum_pair_count": 8, "maximum_group_size": 128},
     "metadata_defaults": {"MANUFACTURER": "UNKNOWN", "DEPARTMENT": "UNKNOWN"},
     "promotion_coverage_note": "..."}

    prepare_bundle.py --config CONFIG --canonical CANONICAL_DIR --out BUNDLE_ROOT [--force]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
from build_model_input import CanonicalBasketModelInputBuilder  # noqa: E402
from v4.bundle import load_bundle, write_fingerprint  # noqa: E402

CONFIG_KEYS = {"schema_version", "dataset_name", "price_basis",
               "promotion_feature", "model_price_sources", "availability", "affinity",
               "metadata_defaults", "promotion_coverage_note", "description"}
PARTITION_KEYS = {"category": {"partition"},
                  "affinity": {"partition", "minimum_pair_count", "maximum_group_size"}}


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise SystemExit("the dataset config needs schema_version 1")
    unknown = set(config).difference(CONFIG_KEYS)
    if unknown:
        raise SystemExit(f"unknown dataset config keys: {sorted(unknown)}")
    for key in ("dataset_name", "price_basis"):
        if not config.get(key):
            raise SystemExit(f"the dataset config needs {key}")
    affinity = config.get("affinity", {})
    partition = affinity.get("partition", "affinity")
    if partition not in PARTITION_KEYS:
        raise SystemExit(f"affinity.partition must be one of {sorted(PARTITION_KEYS)}")
    if set(affinity).difference(PARTITION_KEYS[partition]):
        raise SystemExit(f"affinity keys {sorted(set(affinity) - PARTITION_KEYS[partition])} "
                         f"do not apply to partition {partition!r}")
    return config


def write_category_partition(basket_input: Path) -> dict:
    """The declared merchandise categories as the model's categories.

    Categories are catalogue metadata, not outcomes, so the partition is training-only.
    (Co-purchase groups put complements together; merchandise categories put substitutes
    together, so rho_c can express substitution inside a category.)"""
    items = pd.read_parquet(basket_input / "items.parquet").sort_values("item_id")
    group = items.cat_id.to_numpy().astype(np.int32)
    if not np.array_equal(np.unique(group), np.arange(group.max() + 1)):
        raise SystemExit("category ids are not contiguous")
    output = basket_input / "items_affinity.parquet"
    pd.DataFrame({"item_id": items.item_id.to_numpy(), "cat_id": group}).to_parquet(output, index=False)
    sizes = np.bincount(group)
    manifest = {"schema_version": 1, "training_only": True, "partition": "merchandise_category",
                "n_items": int(len(items)), "n_groups": int(len(sizes)),
                "maximum_group_size_observed": int(sizes.max()),
                "partition_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (basket_input / "affinity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[prepare] category partition: {manifest['n_groups']} groups, largest "
          f"{manifest['maximum_group_size_observed']}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--canonical", type=Path, required=True, help="the canonical directory")
    parser.add_argument("--out", type=Path, required=True, help="bundle root to write")
    parser.add_argument("--product-metadata", type=Path, help="optional product metadata parquet")
    parser.add_argument("--force", action="store_true", help="rebuild into an existing bundle folder")
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_config(config_path)
    canonical, root = args.canonical.resolve(), args.out.resolve()
    if (root / "basket_input").exists() and not args.force:
        raise SystemExit(f"{root} already holds a bundle; pass --force to rebuild it")

    availability = config.get("availability", {"rule": "disabled"})
    builder = CanonicalBasketModelInputBuilder(
        canonical, root, price_basis=config["price_basis"],
        promotion_feature_name=config.get("promotion_feature", "disabled"),
        model_price_sources=tuple(config.get("model_price_sources", ["retail_aggregate"])),
        availability=availability.get("rule", "disabled"),
        availability_left_censor_periods=int(availability.get("left_censor_periods", 13)),
        dataset_name=config["dataset_name"], metadata_defaults=config.get("metadata_defaults"),
        promotion_coverage_note=config.get("promotion_coverage_note", "declared by the canonical adapter"),
        product_metadata=args.product_metadata)
    result = builder.build()
    print(f"[prepare] canonical contract passed: {json.dumps(builder.contract_summary)}", flush=True)

    affinity = config.get("affinity", {})
    if affinity.get("partition", "affinity") == "category":
        write_category_partition(root / "basket_input")
    else:
        subprocess.run([sys.executable, "-u", str(REPO / "scripts/build_affinity_partition.py"),
                        "--data-root", str(root),
                        "--minimum-pair-count", str(int(affinity.get("minimum_pair_count", 8))),
                        "--maximum-group-size", str(int(affinity.get("maximum_group_size", 128)))], check=True)
    index = root / "basket_input/v3_index_affinity.npz"
    if index.exists():
        index.unlink()                       # rebuilt from the new partition
    load_bundle(root)
    digest = write_fingerprint(root)
    record = {"schema_version": 1, "status": "passed", "dataset_name": config["dataset_name"],
              "dataset_config": str(config_path),
              "dataset_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
              "canonical_dir": str(canonical), "model_data_root": str(root),
              "contract_summary": builder.contract_summary, "availability": result.manifest["availability"],
              "data_fingerprint_sha256": digest}
    (root / "bundle_preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(f"[prepare] bundle ready: {root} (fingerprint {digest[:12]})", flush=True)


if __name__ == "__main__":
    main()
