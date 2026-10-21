#!/bin/zsh
# SHOPPER on the stress world (same bundle, trips, assortment and prices as the energy model).
# Usage: run_shopper_stress.sh TAG K [extra fit args]
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT=$PWD/data/stress_world/model_input_category
export OMP_NUM_THREADS=${SHOPPER_THREADS:-4} MKL_NUM_THREADS=${SHOPPER_THREADS:-4} OPENBLAS_NUM_THREADS=${SHOPPER_THREADS:-4} VECLIB_MAXIMUM_THREADS=${SHOPPER_THREADS:-4}
python3 -u scripts/shopper/fit_shopper.py --bundle --K $2 --P 10 --D 10 --lr 0.01 --batch 256 \
  --iters 20000 --eval-every 500 --val-trips 2048 --val-perms 32 --out runs/shopper/$1.pt "${@:3}" \
  > runs/shopper/$1.log 2>&1
