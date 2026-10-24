#!/usr/bin/env bash
# One small synthetic world end to end: generate it, fit the base model and two Version-4
# models (no competition; competition with lambda from the price instrument), and score the
# promotion environment against the world's true revenue (sim/stress_check.py), with each
# trip's latent state reweighted and held fixed.  Each stage's wall time goes to OUT/timings.tsv.
#   OUT      folder to write (required)
#   GEN      extra generator flags, e.g. "--seed 3 --category-growth 0.45 --display-separate"
#   THREADS  threads per job (default 2)
#   ADD      extra fit_additive.py flags (optional; record them, worlds are only comparable under the same fit)
#   WORLD    an existing world to use instead of OUT/world (then start at FROM=2 or later)
#   INIT     initialization to use (default OUT/initialization.pt)
#   BASE     base model to use (default OUT/additive/v3_pipeline_additive_best.pt)
#   FROM     first stage: 1 generate, 2 base model, 3 Version 4, 4 promotion check (default 1)
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

# 1. The world: 3 departments x 4 categories (about 120 products), 1,000 households, 5 stores.
if stage 1; then
  timed generate python3 -u synthetic/generate_stress_world.py --output "$W" --households 1000 --stores 5 \
    --departments 3 --categories-per-department 4 $GEN > "$OUT/1_generate.log" 2>&1
  timed bundle python3 -u data_prep/canonical/prepare_bundle.py --config data_prep/configs/stress_world.json \
    --canonical "$W/canonical" --out "$D" > "$OUT/1_bundle.log" 2>&1
fi
export ENERGY_MODEL_DATA_ROOT=$D
I=${INIT:-$OUT/initialization.pt}; B=${BASE:-$OUT/additive/v3_pipeline_additive_best.pt}

# 2. The exact base model.
if stage 2; then
  timed initialize python3 scripts/initialize.py --output "$I" --threads "$THREADS" > "$OUT/2_initialize.log" 2>&1
  timed additive python3 -u scripts/fit_additive.py --artifact "$I" --output-dir "$OUT/additive" \
    --require-convergence --threads "$THREADS" $ADD > "$OUT/2_additive.log" 2>&1
fi

# 3. Version 4: no competition, and competition with lambda from the price instrument.
if stage 3; then
  timed fit_hs_rec python3 -u scripts/fit_interactions.py --initialization "$I" --additive "$B" \
    --out "$OUT/hs_rec" --recency --threads "$THREADS" > "$OUT/3_hs_rec.out" 2>&1
  timed fit_v4_comp python3 -u scripts/fit_interactions.py --initialization "$I" --additive "$B" \
    --out "$OUT/v4_comp" --recency --within-competition --comp-estimator instrument \
    --threads "$THREADS" > "$OUT/3_v4_comp.out" 2>&1
fi

# 4. The promotion environment against the truth.
if stage 4; then
  check() {  # tag, flags...
    local tag=$1; shift
    timed "check_$tag" python3 -u sim/stress_check.py --world "$W" --initialization "$I" --threads "$THREADS" \
      --truth-cache "$OUT/truth_cache.json" --out "$OUT/env_$tag.json" "$@" > "$OUT/4_env_$tag.log" 2>&1
  }
  check base --checkpoint "$B" --simulator base
  for m in hs_rec v4_comp; do
    for l in fixed reweight; do
      check "${m}_$l" --checkpoint "$OUT/$m.pt" --simulator v4 --latent $l
    done
  done
fi
echo ALL_DONE
