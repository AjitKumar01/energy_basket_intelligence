#!/usr/bin/env bash
# Version 4 on the commodity partition with within-category competition (lambda_c from the
# price instrument, display and mailer left out of it) and common display and mailer slopes
# (--promo-common), fitted afresh from the commodity base model with the schedule of
# run/commodity.sh stage 4; then the discount check, basket size by period and the real halo.
# Run detached, e.g.
#   nohup caffeinate -i run/commodity_comp.sh > runs/commodity/comp_promo.out 2>&1 &
#   DATA      the commodity bundle (default runs/bundle_commodity)
#   OUT       where the base model is and everything is written (default runs/commodity)
#   EVENTS    the validation discount events (checks/discount_promotion.py events)
#   NAME      output name (default v4_comp_promo)
#   FROM      1 fit and checks (default), 2 checks only
set -euo pipefail
cd "$(dirname "$0")/.."
DATA=$(cd "${DATA:-runs/bundle_commodity}" && pwd -P)
OUT=${OUT:-runs/commodity}
EVENTS=${EVENTS:-runs/dunnhumby/discount_events}
NAME=${NAME:-v4_comp_promo}
FROM=${FROM:-1}
export ENERGY_MODEL_DATA_ROOT=$DATA OMP_NUM_THREADS=4
INIT=$OUT/initialization.pt
ADDITIVE=$OUT/additive/v3_pipeline_additive_best.pt
V4=$OUT/$NAME
stamp() { echo "$1 $(date '+%F %T')"; }

if [[ $FROM -le 1 ]]; then
  stamp FIT_START
  python3 -u scripts/fit_interactions.py --initialization "$INIT" --additive "$ADDITIVE" \
    --out "$V4" --recency --centre-history mean \
    --within-competition --comp-estimator instrument --promo-common \
    --pl-epochs 1 --sml-epochs 6 --min-sml-epochs 3 --val-every 1 --cal-epochs 1 \
    --val-trips 1024 --val-draws 128 --max-per-category 40 --threads 4 > "$OUT/4_$NAME.out" 2>&1
  stamp FIT_DONE
fi
python3 checks/discount_promotion.py model --events "$EVENTS" --checkpoint "$V4.pt" \
  --out "${V4}_discount_check.json" --threads 4 > "${V4}_discount.log" 2>&1
stamp DISCOUNT_CHECK_DONE
python3 checks/size_by_period.py --checkpoint "$V4.pt" --out "${V4}_size_by_period.json" \
  > "${V4}_size_by_period.log" 2>&1
stamp SIZE_DONE
python3 checks/halo_real.py --checkpoint "$V4.pt" --model-trips 12 --out "${V4}_halo_real.json" \
  > "${V4}_halo_real.log" 2>&1
stamp ALL_DONE
