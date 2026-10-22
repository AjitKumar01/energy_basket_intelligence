#!/usr/bin/env bash
# Refit Version 4 (recency) with the history shift centred within each trip and its learned
# size term (--centre-history; see v4/model.py), warm-started from a fitted recency model.
# Skips the pseudo-likelihood warm-up; same validation and cap as run/dunnhumby_step4.sh.
#   DATA   the model-data bundle (required)
#   FROM   the fitted Version-4 recency checkpoint to start from (required)
#   OUT    where everything is written (default runs/dunnhumby)
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT=${DATA:?set DATA to the model-data bundle} OMP_NUM_THREADS=4
FROM=${FROM:?set FROM to a fitted Version-4 recency checkpoint}
OUT=${OUT:-runs/dunnhumby}
mkdir -p $OUT
python3 -u scripts/fit_interactions.py --initialization "$(python3 -c "import torch,sys; print(torch.load(sys.argv[1], weights_only=False)['initialization'])" "$FROM")" \
  --additive "$(python3 -c "import torch,sys; print(torch.load(sys.argv[1], weights_only=False)['additive'])" "$FROM")" \
  --out $OUT/v4_hist --recency --centre-history --warm-start "$FROM" \
  --pl-epochs 0 --sml-epochs 4 --min-sml-epochs 2 --val-every 1 --cal-epochs 1 \
  --val-trips 1024 --val-draws 128 --max-per-category 40 --threads 4 > $OUT/4_v4_hist.out 2>&1
