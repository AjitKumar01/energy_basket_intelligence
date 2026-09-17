#!/usr/bin/env python3
"""Write ERIM's declared product types as a product-metadata file for bundle preparation.

Output: a parquet with product_id and subcategory for every product a rule types, plus a
JSON summary. Reference it from a dataset config as "product_metadata".
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/version4"))

from erim_catalogue import TYPE_RULES, product_type  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--canonical-dir", type=Path, default=ROOT / "data/erim_basket/canonical")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "data/erim_basket/catalogue/product_types.parquet")
    args = parser.parse_args()
    products = pd.read_parquet(args.canonical_dir / "products.parquet")
    types = [product_type(c, l) for c, l in zip(products.category, products.label)]
    table = pd.DataFrame({"product_id": products.product_id, "subcategory": types}).dropna()
    table = table.sort_values("product_id").reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(args.output, index=False)
    products = products.assign(subcategory=types)
    summary = {
        "source_products_sha256": hashlib.sha256((args.canonical_dir / "products.parquet").read_bytes()).hexdigest(),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "typed_categories": sorted(TYPE_RULES),
        "products": int(len(products)), "typed_products": int(len(table)),
        "by_category": {
            category: {"products": int(len(frame)),
                       "types": {k: int(v) for k, v in frame.subcategory.value_counts().items()},
                       "untyped": int(frame.subcategory.isna().sum())}
            for category, frame in products.groupby("category")},
    }
    args.output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
