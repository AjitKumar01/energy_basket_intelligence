#!/usr/bin/env bash
# Build the Dunnhumby model-data bundle from the raw "The Complete Journey" CSVs.
#   RAW   folder with transaction_data.csv, product.csv, causal_data.csv (required)
#   OUT   bundle root to write: OUT/data and OUT/basket_input (required)
# About 15-20 minutes.  The audit stops the run if any invariant fails.
set -euo pipefail
cd "$(dirname "$0")/.."
RAW=${RAW:?set RAW to the raw CSV folder}
OUT=${OUT:?set OUT to the bundle root to write}
mkdir -p "$OUT"
P=data_prep/dunnhumby
python3 -u $P/build_base.py --raw "$RAW" --root "$OUT"              # cleaned lines, weekly price panels
python3 -u $P/build_bundle.py --raw "$RAW" --root "$OUT"            # cohort, baskets, prices, state
python3 -u $P/build_promotions.py --raw "$RAW" --root "$OUT"        # display and mailer
python3 -u $P/audit.py --raw "$RAW" --root "$OUT"                   # invariants, preprocessing manifest
python3 -u scripts/build_affinity_partition.py --data-root "$OUT"   # categories from co-purchases
python3 -u data_prep/fingerprint.py --data-root "$OUT" --core-only  # assortment index and fingerprint
