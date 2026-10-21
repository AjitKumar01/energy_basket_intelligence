#!/bin/zsh
# Pilot: refit only lam, price, display and mailer weights of the Dunnhumby additive parent on
# the chain-only price panel; arm "el" keeps the parent's elasticity penalty, arm "free" drops it.
# Usage: run_price_pilot.sh TAG ELAST_W [extra pilot args, e.g. --discount-panel PATH]
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT=/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
A=$ENERGY_MODEL_DATA_ROOT/artifacts/corrected_complete_rank5_20260913
O=runs/dunnhumby_price
mkdir -p $O
python3 -u scripts/v4_training/price_refit_pilot.py --initialization $A/artifacts/initialization.pt \
  --additive $A/out/v3_pipeline_additive_best.pt --output $O/pilot_$1.pt --elast-w $2 "${@:3}" > $O/pilot_$1.log 2>&1
