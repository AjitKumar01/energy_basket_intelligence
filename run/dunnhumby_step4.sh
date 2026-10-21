#!/usr/bin/env bash
# Step 4 on Dunnhumby: Version 4 with recency, without and with M4c, from a fitted base model.
# Both variants run in parallel, 4 threads each.
#   DATA      the model-data bundle (required)
#   INIT      the initialization the base model started from (required)
#   ADDITIVE  the fitted base model, e.g. additive/v3_pipeline_additive_best.pt (required)
#   OUT       where everything is written (default runs/dunnhumby)
#
# Schedule matched to the stress-world recipe by number of updates, not epochs: a Dunnhumby
# epoch is 625 updates (stress: 102), so 6 SML epochs ~ the stress run's 2,900 updates.
# Validation every epoch on 1,024 held-out training trips with 128 importance draws.
# At most 40 products from one category (the data's maximum is 30; the base model puts at
# most 2e-9 of a trip's probability above 40).
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT=${DATA:?set DATA to the model-data bundle} OMP_NUM_THREADS=4
INIT=${INIT:?set INIT to the initialization file}
ADDITIVE=${ADDITIVE:?set ADDITIVE to the fitted base model}
OUT=${OUT:-runs/dunnhumby}
mkdir -p $OUT
SCHEDULE=(--pl-epochs 1 --sml-epochs 6 --min-sml-epochs 3 --val-every 1 --cal-epochs 1
          --val-trips 1024 --val-draws 128 --max-per-category 40)
python3 -u scripts/fit_interactions.py --initialization $INIT --additive $ADDITIVE --out $OUT/v4_rec --recency "${SCHEDULE[@]}" \
  --threads 4 > $OUT/4_v4_rec.out 2>&1 &
python3 -u scripts/fit_interactions.py --initialization $INIT --additive $ADDITIVE --out $OUT/v4_m4c --recency --m4c "${SCHEDULE[@]}" \
  --threads 4 > $OUT/4_v4_m4c.out 2>&1 &
wait
