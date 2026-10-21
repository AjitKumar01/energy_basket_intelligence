#!/usr/bin/env bash
# The pipeline on the synthetic stress world, where the true price responses are known.
#   nohup run/stress_world.sh > runs/stress/pipeline.out 2>&1 &
# The world has no regular-price series, so there is no discount split.
#   WORLD      the stress-world directory (its bundle is WORLD/model_input_category)
#   REFERENCE  reference test trips for stage 5 (default: made by scripts/reference_trips.py)
set -euo pipefail
cd "$(dirname "$0")/.."
WORLD=${WORLD:?set WORLD to the stress-world directory}
DATA=$WORLD/model_input_category
OUT=${OUT:-runs/stress}
FROM=${FROM:-2}
THREADS=${THREADS:-4}
export ENERGY_MODEL_DATA_ROOT=$DATA OMP_NUM_THREADS=$THREADS
mkdir -p "$OUT"
stage() { [[ $1 -ge $FROM ]]; }

stage 2 && python3 scripts/initialize.py --output "$OUT/initialization.pt" --threads "$THREADS" > "$OUT/2b_initialize.log" 2>&1
stage 3 && python3 -u scripts/fit_additive.py --artifact "$OUT/initialization.pt" --output-dir "$OUT/additive" \
  --require-convergence --threads "$THREADS" > "$OUT/3_additive.log" 2>&1
ADDITIVE=$OUT/additive/v3_pipeline_additive_best.pt
for VARIANT in rec m4c; do
  EXTRA=(); [[ $VARIANT == m4c ]] && EXTRA=(--m4c)
  stage 4 && python3 -u scripts/fit_interactions.py --initialization "$OUT/initialization.pt" --additive "$ADDITIVE" \
    --out "$OUT/v4_$VARIANT" --recency "${EXTRA[@]}" --threads "$THREADS" > "$OUT/4_v4_$VARIANT.out" 2>&1
done
# Scoring on the world's test trips (known truth): likelihood, realism and the 32 price scenarios.
REFERENCE=${REFERENCE:-$OUT/reference_test_trips.npz}
stage 5 && [[ ! -e $REFERENCE ]] && python3 scripts/reference_trips.py --initialization "$OUT/initialization.pt" \
  --base-model "$ADDITIVE" --out "$REFERENCE"
for VARIANT in rec m4c; do
  stage 5 && python3 -u scripts/evaluate.py --checkpoint "$OUT/v4_$VARIANT.pt" \
    --reference "$REFERENCE" \
    --world "$WORLD" --threads "$THREADS" > "$OUT/5_evaluate_$VARIANT.log" 2>&1
done
