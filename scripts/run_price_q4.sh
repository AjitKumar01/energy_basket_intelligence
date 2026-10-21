#!/bin/zsh
# Discount check (Q4 only) of an additive checkpoint on the validation discount events.
# Usage: run_price_q4.sh TAG ADDITIVE_CHECKPOINT [SPLIT]
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT=/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
A=$ENERGY_MODEL_DATA_ROOT/artifacts/corrected_complete_rank5_20260913
python3 -u scripts/version4/fit_discount_response.py --q4-only --q4-split ${3:-validation} --threads 4 \
  --source-dir runs/dunnhumby/discount_evidence --output-dir runs/dunnhumby_price/q4_$1 \
  --additive-checkpoint $2 --init-checkpoint $A/artifacts/initialization.pt \
  --rank5-checkpoint $A/artifacts/candidate_rank1.pt > runs/dunnhumby_price/q4_$1.log 2>&1
