#!/usr/bin/env bash
# Refit Version 4 (recency) with the history shift centred within each trip and its learned
# size term (--centre-history; see v4/model.py), warm-started from a fitted recency model.
# Skips the pseudo-likelihood warm-up; same validation and cap as run/dunnhumby_step4.sh.
#   DATA   the model-data bundle (required)
#   FROM   the fitted Version-4 recency checkpoint to start from (required)
#   OUT    where everything is written (default runs/dunnhumby)
#   MODE   centring: mean (default) or inclusive
#   NAME   output name (default v4_hist, or v4_hist_inclusive)
#   PL     pseudo-likelihood epochs first (default 0; 1 helps when the mode changes the level)
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT=${DATA:?set DATA to the model-data bundle} OMP_NUM_THREADS=4
FROM=${FROM:?set FROM to a fitted Version-4 recency checkpoint}
OUT=${OUT:-runs/dunnhumby}
MODE=${MODE:-mean}
NAME=${NAME:-v4_hist$([[ $MODE == mean ]] || echo "_$MODE")}
PL=${PL:-0}
mkdir -p $OUT
python3 -u scripts/fit_interactions.py --initialization "$(python3 -c "import torch,sys; print(torch.load(sys.argv[1], weights_only=False)['initialization'])" "$FROM")" \
  --additive "$(python3 -c "import torch,sys; print(torch.load(sys.argv[1], weights_only=False)['additive'])" "$FROM")" \
  --out $OUT/$NAME --recency --centre-history $MODE --warm-start "$FROM" \
  --pl-epochs $PL --sml-epochs 4 --min-sml-epochs 2 --val-every 1 --cal-epochs 1 \
  --val-trips 1024 --val-draws 128 --max-per-category 40 --threads 4 > $OUT/4_$NAME.out 2>&1
