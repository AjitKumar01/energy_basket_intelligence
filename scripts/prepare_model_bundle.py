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
      "affinity": {"minimum_pair_count": 8, "maximum_group_size": 128},
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
    "metadata_defaults", "promotion_coverage_note", "description",
}


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
    return config


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
    builder = CanonicalBasketModelInputBuilder(
        canonical, root, price_basis=config["price_basis"],
        promotion_feature_name=config.get("promotion_feature", "disabled"),
        model_price_sources=tuple(config.get("model_price_sources", ["retail_aggregate"])),
        availability=availability.get("rule", "disabled"),
        availability_left_censor_periods=int(availability.get("left_censor_periods", 13)),
        dataset_name=config["dataset_name"],
        metadata_defaults=config.get("metadata_defaults"),
        promotion_coverage_note=config.get(
            "promotion_coverage_note", "declared by the canonical adapter"))
    result = builder.build()
    print(f"[prepare] canonical contract passed: {json.dumps(builder.contract_summary)}",
          flush=True)
    run([sys.executable, "-u", V4 / "build_affinity_partition.py",
         "--minimum-pair-count", int(affinity.get("minimum_pair_count", 8)),
         "--maximum-group-size", int(affinity.get("maximum_group_size", 128)),
         "--output", root / "basket_input" / "items_affinity.parquet"], root)
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
