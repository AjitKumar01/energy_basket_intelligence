#!/bin/zsh
# Score a SHOPPER stress-world fit like stress.py --eval scores the energy model.  Usage: TAG
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT=$PWD/data/stress_world/model_input_category
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
python3 -u scripts/shopper/eval_shopper_stress.py --checkpoint runs/shopper/$1.pt \
  --output runs/shopper/$1.eval.json > runs/shopper/$1.eval.log 2>&1
