#!/usr/bin/env python3
"""Prepare a model-data bundle from any canonical basket directory and a dataset config.

One command replaces the per-dataset sequence of bundle build, affinity partition,
ragged index and data fingerprint. Everything dataset-specific lives in the JSON config:

    {
      "schema_version": 1,
      "dataset_name": "my_retailer",
      "canonical_dir": "data/my_retailer/canonical",
      "model_data_root": "data/my_retailer/model_input",
      "price_basis": "chain week retail unit price ...",
      "promotion_feature": "disabled | advertised | special_price",
      "model_price_sources": ["retail_aggregate"],
      "availability": {"rule": "disabled | retail_first_sale", "left_censor_periods": 13},
      "affinity": {"partition": "affinity | category | catalogue_hierarchy",
                   "minimum_pair_count": 8, "maximum_group_size": 128,
                   "minimum_group_products": 3, "minimum_group_training_lines": 300},
      "product_metadata": "optional parquet: product_id + subcategory/brand/manufacturer/department",
      "metadata_defaults": {"MANUFACTURER": "UNKNOWN", "DEPARTMENT": "UNKNOWN"},
      "promotion_coverage_note": "..."
    }

Relative paths are resolved against the repository root. Run the pipeline afterwards with
``scripts/run_pipeline.py --model-data-root <model_data_root>``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
V4 = ROOT / "scripts" / "version4"
sys.path.insert(0, str(V4))

from canonical_basket_input import CanonicalBasketModelInputBuilder  # noqa: E402

CONFIG_KEYS = {
    "schema_version", "dataset_name", "canonical_dir", "model_data_root", "price_basis",
    "promotion_feature", "model_price_sources", "availability", "affinity",
    "metadata_defaults", "promotion_coverage_note", "description", "product_metadata",
}
AFFINITY_KEYS = {
    "affinity": {"partition", "minimum_pair_count", "maximum_group_size"},
    "category": {"partition"},
    "catalogue_hierarchy": {"partition", "minimum_group_products", "minimum_group_training_lines"},
}
HIERARCHY_DEFAULTS = {"minimum_group_products": 3, "minimum_group_training_lines": 300}


def resolve(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config.get("schema_version") != 1:
        raise SystemExit("dataset config requires schema_version = 1")
    unknown = set(config).difference(CONFIG_KEYS)
    if unknown:
        raise SystemExit(f"unknown dataset config keys: {sorted(unknown)}")
    for key in ("dataset_name", "canonical_dir", "model_data_root", "price_basis"):
        if not config.get(key):
            raise SystemExit(f"dataset config requires {key}")
    affinity = config.get("affinity", {})
    partition = affinity.get("partition", "affinity")
    if partition not in AFFINITY_KEYS:
        raise SystemExit(f"affinity.partition must be one of {sorted(AFFINITY_KEYS)}")
    unknown_affinity = set(affinity).difference(AFFINITY_KEYS[partition])
    if unknown_affinity:
        raise SystemExit(f"affinity keys {sorted(unknown_affinity)} do not apply to partition {partition!r}")
    return config


def write_category_partition(basket_input: Path) -> None:
    """Use the declared merchandise categories as the exact within-group partition.

    Co-purchase affinity groups place complements together, so substitutes (rarely bought
    together) fall in different groups and rho_c cannot express within-category
    substitution. Categories are catalogue metadata, not outcomes, so this partition is
    also training-only in the sense the initializer requires.
    """
    import numpy as np
    import pandas as pd
    items = pd.read_parquet(basket_input / "items.parquet", columns=["item_id", "cat_id"])
    items = items.sort_values("item_id")
    output = basket_input / "items_affinity.parquet"
    items[["item_id", "cat_id"]].to_parquet(output, index=False)
    sizes = np.bincount(items.cat_id.to_numpy())
    manifest = {
        "schema_version": 1, "training_only": True, "partition": "merchandise_category",
        "n_items": int(len(items)), "n_groups": int(len(sizes)),
        "maximum_group_size_observed": int(sizes.max()),
        "partition_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (basket_input / "affinity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[prepare] category partition: {len(sizes)} groups, largest {int(sizes.max())}",
          flush=True)


def catalogue_hierarchy_groups(items, minimum_products: int, minimum_training_lines: int):
    """Finest declared catalogue level that meets fixed evidence floors (model-free).

    Within each category, every declared subcategory with at least ``minimum_products``
    products and ``minimum_training_lines`` training purchase lines becomes its own group.
    Products without a declared subcategory (subcategory equal to the category) and those in
    subcategories below either floor stay together in one category remainder group. A
    category without any qualifying subcategory therefore keeps exactly its category group.
    Groups are numbered by (category, subcategory label), with the remainder first.
    """
    import numpy as np
    items = items.sort_values("item_id").reset_index(drop=True)
    category = items.COMMODITY_DESC.astype(str)
    sub = items.SUB_COMMODITY_DESC.astype(str)
    declared = sub != category
    key = category + "\x1f" + sub
    stats = items[declared].groupby(key[declared]).agg(
        products=("item_id", "size"), training_lines=("n_train_lines", "sum"))
    qualifying = set(stats[(stats.products >= minimum_products)
                           & (stats.training_lines >= minimum_training_lines)].index)
    own = key.isin(qualifying)
    group_label = np.where(own, key, category + "\x1f")      # "" subcategory = remainder
    order = sorted(set(group_label), key=lambda g: (items.cat_id[group_label == g].iloc[0],
                                                     g.split("\x1f", 1)[1]))
    group_id = pd_map(group_label, {g: i for i, g in enumerate(order)})
    decisions = {}
    for cat in sorted(category.unique()):
        cat_stats = stats[stats.index.str.startswith(cat + "\x1f")]
        decisions[cat] = {
            "groups": [g.split("\x1f", 1)[1] or "(category remainder)"
                       for g in order if g.startswith(cat + "\x1f")],
            "declared_subcategories": {
                name.split("\x1f", 1)[1]: {"products": int(row.products),
                                            "training_lines": int(row.training_lines),
                                            "own_group": name in qualifying}
                for name, row in cat_stats.iterrows()},
        }
    return group_id, decisions


def pd_map(values, mapping):
    import numpy as np
    return np.array([mapping[v] for v in values], dtype=np.int32)


def write_catalogue_hierarchy_partition(basket_input: Path, settings: dict) -> None:
    import numpy as np
    import pandas as pd
    floors = {**HIERARCHY_DEFAULTS, **{k: v for k, v in settings.items() if k in HIERARCHY_DEFAULTS}}
    items = pd.read_parquet(basket_input / "items.parquet").sort_values("item_id")
    group_id, decisions = catalogue_hierarchy_groups(
        items, int(floors["minimum_group_products"]), int(floors["minimum_group_training_lines"]))
    output = basket_input / "items_affinity.parquet"
    pd.DataFrame({"item_id": items.item_id.to_numpy(), "cat_id": group_id}).to_parquet(output, index=False)
    sizes = np.bincount(group_id)
    manifest = {
        "schema_version": 1, "training_only": True, "partition": "catalogue_hierarchy",
        "rule": "finest declared catalogue level meeting fixed floors; others pooled per category",
        "floors": {k: int(v) for k, v in floors.items()},
        "n_items": int(len(items)), "n_groups": int(len(sizes)),
        "maximum_group_size_observed": int(sizes.max()),
        "categories": decisions,
        "partition_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }
    (basket_input / "affinity_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[prepare] catalogue-hierarchy partition: {len(sizes)} groups, largest {int(sizes.max())}",
          flush=True)


def run(command: list[str], root: Path) -> None:
    env = {**os.environ, "ENERGY_MODEL_DATA_ROOT": str(root), "V3_AFFINITY": "1"}
    print("[prepare] " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, env=env, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--force", action="store_true",
                        help="rebuild into an existing model_data_root")
    args = parser.parse_args()
    config_path = args.dataset_config.resolve()
    config = load_config(config_path)
    canonical = resolve(config["canonical_dir"])
    root = resolve(config["model_data_root"])
    if (root / "basket_input").exists() and not args.force:
        raise SystemExit(f"{root} already holds a bundle; pass --force to rebuild it")
    availability = config.get("availability", {"rule": "disabled"})
    affinity = config.get("affinity", {})
    partition = affinity.get("partition", "affinity")
    builder = CanonicalBasketModelInputBuilder(
        canonical, root, price_basis=config["price_basis"],
        promotion_feature_name=config.get("promotion_feature", "disabled"),
        model_price_sources=tuple(config.get("model_price_sources", ["retail_aggregate"])),
        availability=availability.get("rule", "disabled"),
        availability_left_censor_periods=int(availability.get("left_censor_periods", 13)),
        dataset_name=config["dataset_name"],
        metadata_defaults=config.get("metadata_defaults"),
        promotion_coverage_note=config.get(
            "promotion_coverage_note", "declared by the canonical adapter"),
        product_metadata=(resolve(config["product_metadata"]) if config.get("product_metadata") else None))
    result = builder.build()
    print(f"[prepare] canonical contract passed: {json.dumps(builder.contract_summary)}",
          flush=True)
    if partition == "affinity":
        run([sys.executable, "-u", V4 / "build_affinity_partition.py",
             "--minimum-pair-count", int(affinity.get("minimum_pair_count", 8)),
             "--maximum-group-size", int(affinity.get("maximum_group_size", 128)),
             "--output", root / "basket_input" / "items_affinity.parquet"], root)
    elif partition == "category":
        write_category_partition(root / "basket_input")
    else:
        write_catalogue_hierarchy_partition(root / "basket_input", affinity)
    run([sys.executable, "-u", V4 / "data.py", "--force"], root)
    run([sys.executable, "-u", V4 / "provenance.py"], root)
    fingerprint = json.loads((root / "basket_input" / "model_data_fingerprint.json").read_text())
    record = {
        "schema_version": 1,
        "status": "passed",
        "dataset_name": config["dataset_name"],
        "dataset_config": str(config_path),
        "dataset_config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "canonical_dir": str(canonical),
        "model_data_root": str(root),
        "contract_summary": builder.contract_summary,
        "availability": result.manifest["availability"],
        "data_fingerprint_sha256": fingerprint["fingerprint_sha256"],
        "next_command": (f"python -u scripts/run_pipeline.py --model-data-root {root} "
                         "--run-dir <new run directory> --profile full"),
    }
    (root / "bundle_preparation.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
