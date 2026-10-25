#!/usr/bin/env bash
# Base-model-only variant of small_world.sh: generate the world, fit the exact base model,
# and score the base-model promotion environment against the world's true revenue.
#   OUT      folder to write (required)
#   GEN      extra generator flags, e.g. "--seed 1 --category-growth 0.45 --departments 6 --categories-per-department 8"
#   ADD      extra fit_additive.py flags (optional)
#   THREADS  threads per job (default 2)
#   WORLD    an existing world to use instead of OUT/world (then start at FROM=2 or later)
#   FROM     first stage: 1 generate, 2 base model, 4 base promotion check (default 1)
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:?set OUT to the folder to write}
GEN=${GEN:-}
ADD=${ADD:-}
THREADS=${THREADS:-2}
FROM=${FROM:-1}
export OMP_NUM_THREADS=$THREADS
W=${WORLD:-$OUT/world}; D=$W/model_input_category
mkdir -p "$OUT"
stage() { [[ $1 -ge $FROM ]]; }
timed() {  # name, command...
  local name=$1; shift
  local t0; t0=$(date +%s)
  "$@"
  printf '%s\t%s\n' "$name" $(( $(date +%s) - t0 )) >> "$OUT/timings.tsv"
}

# 1. The world: defaults to 3 departments x 4 categories, 1,000 households, 5 stores (overridden by GEN).
if stage 1; then
  timed generate python3 -u synthetic/generate_stress_world.py --output "$W" --households 1000 --stores 5 \
    --departments 3 --categories-per-department 4 $GEN > "$OUT/1_generate.log" 2>&1
  timed bundle python3 -u data_prep/canonical/prepare_bundle.py --config data_prep/configs/stress_world.json \
    --canonical "$W/canonical" --out "$D" > "$OUT/1_bundle.log" 2>&1
fi
export ENERGY_MODEL_DATA_ROOT=$D
I=$OUT/initialization.pt; B=$OUT/additive/v3_pipeline_additive_best.pt

# 2. The exact base model.
if stage 2; then
  if [[ "$ADD" == *"--discount-panel"* ]]; then
    timed discount_panel python3 scripts/build_discount_panel.py --data-root "$D" --output "$OUT/log_discount.npy" > "$OUT/2_discount_panel.log" 2>&1
  fi
  NMAX=$(python3 -c 'import numpy as np, os; d = np.load(os.environ["ENERGY_MODEL_DATA_ROOT"] + "/basket_input/v3_index_affinity.npz"); print(int(d["trip_nlines"][d["trip_split"] == 0].max()))')
  timed initialize python3 scripts/initialize.py --output "$I" --nmax "$NMAX" --threads "$THREADS" > "$OUT/2_initialize.log" 2>&1
  timed additive python3 -u scripts/fit_additive.py --artifact "$I" --output-dir "$OUT/additive" \
    --require-convergence --threads "$THREADS" $ADD > "$OUT/2_additive.log" 2>&1
fi

# 4. Base-model promotion environment against the truth.
if stage 4; then
  timed check_base python3 -u sim/stress_check.py --world "$W" --initialization "$I" --threads "$THREADS" \
    --truth-cache "$OUT/truth_cache.json" --out "$OUT/env_base.json" \
    --checkpoint "$B" --simulator base > "$OUT/4_env_base.log" 2>&1
fi
echo ALL_DONE
