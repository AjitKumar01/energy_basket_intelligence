#!/usr/bin/env bash
# Version 4 with within-category competition (--within-competition, lambda_c learned by
# likelihood) on the stress world, then the halo check and the held-out evaluation.  Same
# fit settings as the recency fit it is compared with.
#   WORLD   the stress-world directory (with model_input_category/)
#   INIT    initialization;  ADDITIVE  the fitted base model;  REFERENCE  reference test trips
#   OUT     output folder (default runs/stress)
set -euo pipefail
cd "$(dirname "$0")/.."
WORLD=${WORLD:?set WORLD}; INIT=${INIT:?set INIT}; ADDITIVE=${ADDITIVE:?set ADDITIVE}
REFERENCE=${REFERENCE:?set REFERENCE}; OUT=${OUT:-runs/stress}
export ENERGY_MODEL_DATA_ROOT=$WORLD/model_input_category OMP_NUM_THREADS=4
mkdir -p $OUT
python3 -u scripts/fit_interactions.py --initialization "$INIT" --additive "$ADDITIVE" \
  --out $OUT/v4_complik --recency --within-competition --threads 4 > $OUT/v4_complik.out 2>&1
python3 -u checks/halo.py --checkpoint $OUT/v4_complik.pt --world "$WORLD" \
  --out $OUT/halo_v4_complik.json --threads 4 > $OUT/halo_v4_complik.log 2>&1
python3 -u scripts/evaluate.py --checkpoint $OUT/v4_complik.pt --reference "$REFERENCE" \
  --world "$WORLD" --threads 4 --out-dir $OUT > $OUT/v4_complik.eval.log 2>&1
