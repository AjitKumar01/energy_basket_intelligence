#!/usr/bin/env python3
"""Write a model-free product partition for any model-data bundle.

Reads ``<basket-input>/items.parquet`` (and, for ``substitution_evidence``, the training
baskets and optional availability panel) and writes ``items_affinity.parquet`` and
``affinity_manifest.json`` to ``--output-dir``. Use this for bundles not built by
``scripts/prepare_model_bundle.py`` (for example the Dunnhumby route). It refuses to replace
an existing partition unless ``--force`` is given, because checkpoints trained on one
partition do not apply to another.

  python scripts/version4/build_catalogue_partition.py --basket-input basket_input \\
      --rule finest_catalogue_level --output-dir <new bundle>/basket_input
  python scripts/version4/build_catalogue_partition.py --basket-input basket_input \\
      --rule substitution_evidence --settings '{"category_boundary": false}' --validate-only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from catalogue_partition import (HIERARCHY_DEFAULTS, HIERARCHY_RULES, catalogue_hierarchy_groups,
                                 finest_level_groups, validate_partition, write_partition)
from evidence_partition import build_evidence_partition, settings_from

RULES = (*HIERARCHY_RULES, "substitution_evidence")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--basket-input", type=Path, required=True)
    parser.add_argument("--rule", choices=RULES, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--minimum-group-products", type=int,
                        default=HIERARCHY_DEFAULTS["minimum_group_products"])
    parser.add_argument("--minimum-group-training-lines", type=int,
                        default=HIERARCHY_DEFAULTS["minimum_group_training_lines"])
    parser.add_argument("--settings", default="{}", help="substitution_evidence settings as JSON")
    parser.add_argument("--validate-only", action="store_true",
                        help="print the partition diagnostics without writing files")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.validate_only and args.output_dir is None:
        parser.error("--output-dir is required unless --validate-only is given")
    items = pd.read_parquet(args.basket_input / "items.parquet").sort_values("item_id")
    nest = True
    manifest = {"schema_version": 1, "training_only": True}
    if args.rule == "category":
        group_id, decisions = items.cat_id.to_numpy(), None
        manifest["partition"] = "merchandise_category"
    elif args.rule == "finest_catalogue_level":
        group_id, decisions = finest_level_groups(items)
        manifest.update(partition="finest_catalogue_level",
                        rule="one group per declared (category, subcategory) path; undeclared "
                             "products form their category's remainder group")
    elif args.rule == "catalogue_hierarchy":
        floors = {"minimum_group_products": args.minimum_group_products,
                  "minimum_group_training_lines": args.minimum_group_training_lines}
        group_id, decisions = catalogue_hierarchy_groups(items, *floors.values())
        manifest.update(partition="catalogue_hierarchy", floors=floors,
                        rule="finest declared catalogue level meeting fixed floors; others pooled per category")
    else:
        settings = settings_from(json.loads(args.settings))
        group_id, summary = build_evidence_partition(args.basket_input, settings)
        decisions, nest = None, bool(settings["category_boundary"])
        manifest.update(partition="substitution_evidence", evidence=summary,
                        rule="household-level co-purchase shortfall on training trips; constrained "
                             "average-linkage merging and single-product moves; declared thresholds")
    diagnostics = validate_partition(items, group_id, nest_in_category=nest)
    print(json.dumps(diagnostics, indent=2))
    if args.validate_only:
        return
    target = args.output_dir / "items_affinity.parquet"
    if target.exists() and not args.force:
        raise SystemExit(f"{target} exists; pass --force to replace the partition")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.rule in ("finest_catalogue_level", "substitution_evidence"):
        manifest["validation"] = diagnostics
    if decisions is not None:
        manifest["categories"] = decisions
    written = write_partition(args.output_dir, items, group_id, manifest)
    print(f"wrote {written['n_groups']} groups to {args.output_dir}")


if __name__ == "__main__":
    main()
