#!/usr/bin/env bash
# Refit on the commodity partition (run/commodity_bundle.sh).  Run detached, e.g.
#   nohup caffeinate -i run/commodity.sh > runs/commodity/pipeline.out 2>&1 &
# Set FROM=<stage> to start later.
#   DATA      the commodity bundle (default runs/bundle_commodity)
#   OUT       where everything is written (default runs/commodity)
#   PANEL     the discount panel (product x day; the partition does not enter it)
#   WARM      an additive fit on the co-purchase partition to start the base model from
#   EVENTS    the validation discount events (checks/discount_promotion.py events)
#   V4FROM    a Version-4 fit on the co-purchase partition; stage 4 starts its product-level
#             blocks from it (fit_interactions.py --warm-start), the per-category ones afresh
#   MODE      history centring for stage 4: mean (default) or inclusive
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=$(cd "${DATA:-runs/bundle_commodity}" && pwd -P)
OUT=${OUT:-runs/commodity}
PANEL=${PANEL:?set PANEL to the discount panel}
WARM=${WARM:?set WARM to an additive fit}
EVENTS=${EVENTS:-runs/dunnhumby/discount_events}
FROM=${FROM:-2}
export ENERGY_MODEL_DATA_ROOT=$DATA OMP_NUM_THREADS=4
mkdir -p "$OUT"
stage() { [[ $1 -ge $FROM ]]; }
INIT=$OUT/initialization.pt
ADDITIVE=$OUT/additive/v3_pipeline_additive_best.pt

# 2b. Starting values (seconds).
stage 2 && [[ ! -e $INIT ]] && python3 scripts/initialize.py --output "$INIT" --threads 4 > "$OUT/2b_initialize.log" 2>&1

# 3. Exact additive base model, same objective as run/dunnhumby.sh, warm-started from WARM.
stage 3 && python3 -u scripts/fit_additive.py --artifact "$INIT" --output-dir "$OUT/additive" --warm-start "$WARM" \
  --discount-panel "$PANEL" --disc-kappa --disc-common --require-convergence --threads 4 > "$OUT/3_additive.log" 2>&1

# 3c. Discount check and the real-data halo of the base model.
stage 3 && python3 checks/discount_promotion.py model --events "$EVENTS" --checkpoint "$ADDITIVE" \
  --out "$OUT/additive/discount_check.json" --threads 4 > "$OUT/3c_discount_check.log" 2>&1
stage 3 && python3 checks/halo_real.py --checkpoint "$ADDITIVE" --out "$OUT/halo_real_base.json" \
  > "$OUT/3c_halo_real.log" 2>&1

# 4. Version 4 with recency and history centring, warm-started across partitions, one
#    pseudo-likelihood epoch first so phi settles on the new categories; then its checks.
MODE=${MODE:-mean}
V4=$OUT/v4_hist_$MODE
stage 4 && python3 -u scripts/fit_interactions.py --initialization "$INIT" --additive "$ADDITIVE" \
  --out "$V4" --recency --centre-history $MODE --warm-start "${V4FROM:?set V4FROM}" \
  --pl-epochs 1 --sml-epochs 4 --min-sml-epochs 2 --val-every 1 --cal-epochs 1 \
  --val-trips 1024 --val-draws 128 --max-per-category 40 --threads 4 > "$OUT/4_v4_hist_$MODE.out" 2>&1
stage 4 && python3 checks/discount_promotion.py model --events "$EVENTS" --checkpoint "$V4.pt" \
  --out "${V4}_discount_check.json" --threads 4 > "${V4}_discount.log" 2>&1
stage 4 && python3 checks/size_by_period.py --checkpoint "$V4.pt" --out "${V4}_size_by_period.json" \
  > "${V4}_size_by_period.log" 2>&1
stage 4 && python3 checks/halo_real.py --checkpoint "$V4.pt" --model-trips 12 --out "${V4}_halo_real.json" \
  > "${V4}_halo_real.log" 2>&1
