#!/usr/bin/env python3
"""Write a model-free catalogue partition for any model-data bundle.

Reads ``<basket-input>/items.parquet`` and writes ``items_affinity.parquet`` and
``affinity_manifest.json`` to ``--output-dir``. Use this for bundles not built by
``scripts/prepare_model_bundle.py`` (for example the Dunnhumby route). It refuses to replace
an existing partition unless ``--force`` is given, because checkpoints trained on one
partition do not apply to another.

  python scripts/version4/build_catalogue_partition.py --basket-input basket_input \\
      --rule finest_catalogue_level --output-dir <new bundle>/basket_input
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from catalogue_partition import (HIERARCHY_RULES, catalogue_hierarchy_groups, finest_level_groups,
                                 validate_partition, write_partition)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--basket-input", type=Path, required=True)
    parser.add_argument("--rule", choices=HIERARCHY_RULES, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-group-products", type=int, default=3)
    parser.add_argument("--minimum-group-training-lines", type=int, default=300)
    parser.add_argument("--validate-only", action="store_true",
                        help="print the partition diagnostics without writing files")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    items = pd.read_parquet(args.basket_input / "items.parquet").sort_values("item_id")
    if args.rule == "category":
        group_id, decisions = items.cat_id.to_numpy(), None
        manifest = {"schema_version": 1, "training_only": True, "partition": "merchandise_category"}
    elif args.rule == "finest_catalogue_level":
        group_id, decisions = finest_level_groups(items)
        manifest = {"schema_version": 1, "training_only": True, "partition": "finest_catalogue_level",
                    "rule": "one group per declared (category, subcategory) path; undeclared "
                            "products form their category's remainder group"}
    else:
        floors = {"minimum_group_products": args.minimum_group_products,
                  "minimum_group_training_lines": args.minimum_group_training_lines}
        group_id, decisions = catalogue_hierarchy_groups(items, *floors.values())
        manifest = {"schema_version": 1, "training_only": True, "partition": "catalogue_hierarchy",
                    "rule": "finest declared catalogue level meeting fixed floors; others pooled per category",
                    "floors": floors}
    diagnostics = validate_partition(items, group_id)
    print(json.dumps(diagnostics, indent=2))
    if args.validate_only:
        return
    target = args.output_dir / "items_affinity.parquet"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists; pass --force to replace the partition")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.rule == "finest_catalogue_level":
        manifest["validation"] = diagnostics
    if decisions is not None:
        manifest["categories"] = decisions
    written = write_partition(args.output_dir, items, group_id, manifest)
    print(f"wrote {written['n_groups']} groups to {args.output_dir}")


if __name__ == "__main__":
    main()
