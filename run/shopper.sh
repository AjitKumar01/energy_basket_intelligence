#!/usr/bin/env bash
# SHOPPER baseline under the locked protocol (docs/SHOPPER_BASELINE_PROTOCOL.md):
# taste size K in {8, 16, 32}, chosen on validation; chain-only prices.
#   nohup run/shopper.sh > runs/shopper/pipeline.out 2>&1 &
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT=${DATA:-/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence}
OUT=${OUT:-runs/shopper}
THREADS=${THREADS:-4}
export OMP_NUM_THREADS=$THREADS
mkdir -p "$OUT"
for K in 8 16 32; do
  python3 -u shopper/fit_shopper.py --bundle --price-contract chain_product_week_only --K "$K" \
    --threads "$THREADS" --out "$OUT/dunnhumby_k$K.pt" > "$OUT/dunnhumby_k$K.log" 2>&1
done
