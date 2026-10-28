#!/usr/bin/env bash
# From a model-data bundle to the fitted mixture model, stage by stage.  Run detached, e.g.
#   nohup run/fit.sh > runs/commodity/fit.out 2>&1 &
# Each stage writes its own log under $OUT.  Set FROM=<stage number> to start later.
#
#   DATA     the model-data bundle (run/prepare_dunnhumby.sh, then run/commodity_bundle.sh) (required)
#   OUT      where everything is written (default runs/commodity)
#   PANEL    the discount panel (default: built by stage 2 into $OUT/log_discount.npy)
#   WARM     an additive fit to start the base model from (optional; same products, any partition)
#   NAME     name of the mixture fit (default mixture)
#   EPOCHS   passes over the training trips for the mixture fit (default 2)
#   THREADS  default 4
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=$(cd "${DATA:?set DATA to the model-data bundle}" && pwd -P)
OUT=${OUT:-runs/commodity}
NAME=${NAME:-mixture}
EPOCHS=${EPOCHS:-2}
FROM=${FROM:-2}
THREADS=${THREADS:-4}
export ENERGY_MODEL_DATA_ROOT=$DATA OMP_NUM_THREADS=$THREADS
mkdir -p "$OUT"
stage() { [[ $1 -ge $FROM ]]; }
stamp() { echo "$1 $(date '+%F %T')"; }
INIT=$OUT/initialization.pt
ADDITIVE=$OUT/additive/v3_pipeline_additive_best.pt
PANEL=${PANEL:-$OUT/log_discount.npy}

# 2. Discount panel (product x day, centred on training days) and starting values.
if stage 2; then
  [[ -e $PANEL ]] || python3 scripts/build_discount_panel.py --output "$PANEL" > "$OUT/2_discount_panel.log" 2>&1
  [[ -e $INIT ]] || python3 scripts/initialize.py --output "$INIT" --threads "$THREADS" > "$OUT/2b_initialize.log" 2>&1
  stamp STAGE2_DONE
fi

# 3. Exact additive base model: chain-only prices, discount split, elasticity anchor (skipped if
#    $ADDITIVE exists, so an existing base model is never overwritten).
if stage 3 && [[ -e $ADDITIVE ]]; then
  echo "stage 3 skipped: $ADDITIVE exists"
elif stage 3; then
  WARM_ARGS=(); [[ -n ${WARM:-} ]] && WARM_ARGS=(--warm-start "$WARM")
  python3 -u scripts/fit_additive.py --artifact "$INIT" --output-dir "$OUT/additive" ${WARM_ARGS[@]+"${WARM_ARGS[@]}"} \
    --discount-panel "$PANEL" --disc-kappa --disc-common --require-convergence --threads "$THREADS" \
    > "$OUT/3_additive.log" 2>&1
  stamp STAGE3_DONE
fi

# 4. The mixture model from the base model: purchase history (mean-centred), within-category
#    competition by the price instrument on a fixed 40,000 training trips, common display and
#    mailer slopes, at most 40 products per category.
if stage 4; then
  python3 -u scripts/fit_mixture.py --initialization "$INIT" --additive "$ADDITIVE" --out "$OUT/$NAME" \
    --recency --centre-history mean --within-competition --comp-iv-trips 40000 --promo-common \
    --max-per-category 40 --epochs "$EPOCHS" --threads "$THREADS" > "$OUT/4_$NAME.out" 2>&1
  stamp STAGE4_DONE
fi
