#!/usr/bin/env python3
"""Build audited canonical multi-category baskets from official ERIM archives."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/version4"))

from external_basket import (
    BasketBuildPolicy,
    BasketCategorySource,
    ERIMMultiCategoryBasketAdapter,
    file_sha256,
)


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def category_from_config(specification: dict, base: Path) -> BasketCategorySource:
    sources = specification["sources"]
    shopping_values = sources["shopping"]
    if isinstance(shopping_values, str):
        shopping_values = [shopping_values]
    return BasketCategorySource(
        name=str(specification["name"]),
        purchase=resolve(base, sources["purchase"]),
        shopping=tuple(resolve(base, value) for value in shopping_values),
        products=resolve(base, sources["products"]),
        retail=(None if sources.get("retail") is None
                else resolve(base, sources["retail"])),
    )


def verify_download_manifest(categories, manifest_path: Path) -> dict:
    manifest = json.loads(manifest_path.read_text())
    recorded = manifest.get("categories", {})
    checked = {}
    for category in categories:
        if category.name not in recorded:
            raise ValueError(f"download manifest has no category {category.name!r}")
        expected = recorded[category.name].get("files", {})
        checked[category.name] = {}
        for path in category.files:
            digest = file_sha256(path)
            if path.name not in expected or expected[path.name].get("sha256") != digest:
                raise ValueError(f"source hash mismatch for {path}")
            checked[category.name][path.name] = digest
    return checked


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=ROOT / "configs/erim_basket.json")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    configuration = json.loads(config_path.read_text())
    if configuration.get("schema_version") != 1:
        raise ValueError("ERIM basket configuration requires schema_version=1")
    categories = tuple(
        category_from_config(specification, config_path.parent)
        for specification in configuration["categories"])
    source_manifest = resolve(config_path.parent, configuration["source_manifest"])
    source_hashes = verify_download_manifest(categories, source_manifest)
    policy = BasketBuildPolicy(**configuration.get("policy", {}))
    dataset = ERIMMultiCategoryBasketAdapter(categories, policy).load()
    output = (args.output_dir.resolve() if args.output_dir else
              resolve(config_path.parent, configuration["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    outputs = {
        "transactions": output / "transactions.parquet",
        "products": output / "products.parquet",
        "shopping_opportunities": output / "shopping_opportunities.parquet",
        "store_week_prices": output / "store_week_prices.parquet",
        "promotions": output / "promotions.parquet",
    }
    dataset.transactions.to_parquet(outputs["transactions"], index=False)
    dataset.products.to_parquet(outputs["products"], index=False)
    dataset.opportunities.to_parquet(outputs["shopping_opportunities"], index=False)
    dataset.store_week_prices.to_parquet(outputs["store_week_prices"], index=False)
    dataset.promotions.to_parquet(outputs["promotions"], index=False)
    result = {
        "schema_version": 1,
        "status": "passed",
        "configuration": str(config_path),
        "configuration_sha256": file_sha256(config_path),
        "download_manifest": str(source_manifest),
        "download_manifest_sha256": file_sha256(source_manifest),
        "source_sha256": source_hashes,
        "audit": dataset.audit,
        "outputs": {
            name: {
                "path": str(path), "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
            for name, path in outputs.items()
        },
    }
    report = output / "build_audit.json"
    report.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
