#!/usr/bin/env bash
# SHOPPER baseline under the locked protocol (docs/SHOPPER_BASELINE_PROTOCOL.md).
#   DATA       the model-data bundle (required)
#   CONTRACT   price panel: chain_product_week_only for Dunnhumby, chain_and_store for the
#              stress world (as the protocol's stress fits)
#   KS         taste sizes to fit (default "8 16 32"); choose the best validation log-likelihood
#   OUT        where everything is written (default runs/shopper)
# Then score the chosen fit once:
#   CHOSEN=<K> REFERENCE=<reference trips .npz> [WORLD=<stress world dir>] run/shopper.sh
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT=${DATA:?set DATA to the model-data bundle}
CONTRACT=${CONTRACT:?set CONTRACT to chain_product_week_only or chain_and_store}
OUT=${OUT:-runs/shopper}
THREADS=${THREADS:-4}
export OMP_NUM_THREADS=$THREADS
mkdir -p "$OUT"
if [[ -z ${CHOSEN:-} ]]; then
  for K in ${KS:-8 16 32}; do
    python3 -u shopper/fit_shopper.py --bundle --price-contract "$CONTRACT" --K "$K" --P 10 --D 10 --lr 0.01 \
      --batch 256 --iters 20000 --eval-every 500 --val-trips 2048 --val-perms 32 --threads "$THREADS" \
      --out "$OUT/k$K.pt" > "$OUT/k$K.log" 2>&1
  done
  python3 -c "
import sys, torch
for k in sys.argv[1:]:
    print('K', k, 'best validation log P(S):', torch.load('$OUT/k' + k + '.pt', weights_only=False)['best_val_basket_ll'])" ${KS:-8 16 32}
else
  python3 -u shopper/evaluate_shopper.py --checkpoint "$OUT/k$CHOSEN.pt" --reference "${REFERENCE:?set REFERENCE}" \
    ${WORLD:+--world "$WORLD"} --perms 8192 --threads "$THREADS" --output "$OUT/k$CHOSEN.eval.json" \
    > "$OUT/k$CHOSEN.eval.log" 2>&1
fi
