#!/usr/bin/env bash
# The whole pipeline on Dunnhumby, stage by stage.  Run detached, e.g.
#   nohup run/dunnhumby.sh > runs/dunnhumby/pipeline.out 2>&1 &
# Each stage writes its own log under $OUT.  Set FROM=<stage number> to start later.
#
#   DATA       the model-data bundle (basket_input/, data/), built by run/prepare_dunnhumby.sh (required)
#   OUT        where everything is written (default runs/dunnhumby)
#   REFERENCE  reference test trips for stage 5 (default: made by scripts/reference_trips.py)
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=${DATA:?set DATA to the model-data bundle}
OUT=${OUT:-runs/dunnhumby}
FROM=${FROM:-1}
THREADS=${THREADS:-4}
export ENERGY_MODEL_DATA_ROOT=$DATA OMP_NUM_THREADS=$THREADS
mkdir -p "$OUT"
stage() { [[ $1 -ge $FROM ]]; }

# 1. Categories from training co-purchases (only for a new bundle; Dunnhumby's exists, and
#    rebuilding it changes the bundle fingerprint, so earlier checkpoints stop loading).
# python3 scripts/build_affinity_partition.py

# 2. Discount panel (product x day), centred on training days.
stage 2 && python3 scripts/build_discount_panel.py --output "$OUT/log_discount.npy" > "$OUT/2_discount_panel.log" 2>&1

# 2b. Starting values.  INIT=<file> reuses an existing initialization instead (a fresh one
#     differs from the one the Step 3 refit used only in rho_0, which that older initializer
#     set with a random phi and store prices).
INIT=${INIT:-$OUT/initialization.pt}
stage 2 && [[ ! -e $INIT ]] && python3 scripts/initialize.py --output "$INIT" --threads "$THREADS" \
  > "$OUT/2b_initialize.log" 2>&1

# 3. Exact additive base model: chain-only prices, discount split, elasticity anchor.
stage 3 && python3 -u scripts/fit_additive.py --artifact "$INIT" --output-dir "$OUT/additive" \
  --discount-panel "$OUT/log_discount.npy" --disc-kappa --disc-common --require-convergence \
  --threads "$THREADS" > "$OUT/3_additive.log" 2>&1
ADDITIVE=$OUT/additive/v3_pipeline_additive_best.pt

# 3c. Discount and promotion check of the base model on validation events.
stage 3 && python3 checks/discount_promotion.py events --out-dir "$OUT/discount_events" > "$OUT/3c_events.log" 2>&1
stage 3 && python3 checks/discount_promotion.py model --events "$OUT/discount_events" --checkpoint "$ADDITIVE" \
  --out "$OUT/additive/discount_check.json" --threads "$THREADS" > "$OUT/3c_discount_check.log" 2>&1

# 4. Version-4 with recency, without and with M4c; choose on their validation log-likelihood.
for VARIANT in rec m4c; do
  EXTRA=(); [[ $VARIANT == m4c ]] && EXTRA=(--m4c)
  stage 4 && python3 -u scripts/fit_interactions.py --initialization "$INIT" --additive "$ADDITIVE" \
    --out "$OUT/v4_$VARIANT" --recency "${EXTRA[@]}" --threads "$THREADS" > "$OUT/4_v4_$VARIANT.out" 2>&1
  stage 4 && python3 checks/discount_promotion.py model --events "$OUT/discount_events" --checkpoint "$OUT/v4_$VARIANT.pt" \
    --out "$OUT/v4_${VARIANT}_discount_check.json" --threads "$THREADS" > "$OUT/4_v4_${VARIANT}_discount.log" 2>&1
  stage 4 && python3 checks/support_cap.py --checkpoint "$OUT/v4_$VARIANT.pt" --threads "$THREADS" \
    --out "$OUT/v4_${VARIANT}_support_cap.json" > "$OUT/4_v4_${VARIANT}_support_cap.log" 2>&1
done

# 5. Test weeks, once, for the chosen model only (likelihood on the reference test trips and
#    realism); set CHOSEN=rec or m4c.
if [[ -n ${CHOSEN:-} ]]; then
  REFERENCE=${REFERENCE:-$OUT/reference_test_trips.npz}
  [[ -e $REFERENCE ]] || python3 scripts/reference_trips.py --initialization "$INIT" --base-model "$ADDITIVE" \
    --out "$REFERENCE"
  python3 -u scripts/evaluate.py --checkpoint "$OUT/v4_$CHOSEN.pt" \
    --reference "$REFERENCE" \
    --threads "$THREADS" > "$OUT/5_evaluate_$CHOSEN.log" 2>&1
fi
