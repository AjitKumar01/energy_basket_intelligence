#!/usr/bin/env python3
"""Build isolated energy-model inputs from a canonical basket bundle."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/version4"))

from canonical_basket_input import CanonicalBasketModelInputBuilder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--price-basis", required=True)
    parser.add_argument("--promotion-feature",
                        choices=("advertised", "special_price", "disabled"),
                        default="advertised")
    parser.add_argument(
        "--model-price-sources", nargs="+", default=["retail_aggregate"],
        choices=CanonicalBasketModelInputBuilder.PRICE_SOURCES,
        help=("canonical price sources that may price model features; purchase aggregates "
              "exist only when a panel household bought the product"))
    args = parser.parse_args()
    result = CanonicalBasketModelInputBuilder(
        args.canonical_dir, args.output_root, price_basis=args.price_basis,
        promotion_feature_name=args.promotion_feature,
        model_price_sources=tuple(args.model_price_sources)).build()
    report = result.root / "model_input_build.json"
    report.write_text(json.dumps(result.manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "status": "passed", "root": str(result.root),
        "basket_input": str(result.basket_input), "report": str(report),
        **result.manifest,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
