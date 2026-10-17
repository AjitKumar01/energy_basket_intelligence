#!/usr/bin/env bash
# Training-only convergence run (hs_conv): held-out validation subset of training trips,
# decoupled lam/base learning rates and moment-matching preconditioning, rank 16 tied Phi.
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT="$PWD/data/stress_world/model_input_category"
export DYLD_INSERT_LIBRARIES="/tmp/libshm_shim.dylib"
/opt/homebrew/bin/python3.11 -u scripts/v4_training/stress.py train \
  --out runs/stress/hs_conv --cpp-sampler --tie-phi --rank 16 \
  --phi-prec 20 --delta-prec 200 --pool-prod 0.5 \
  --lr-base 0.0015 --lr-lam 0.008 --mm-lam 0.5 --mm-rho 0.2 --mm-rho0 0.5 \
  --sml-epochs 28 --min-sml-epochs 16 --cal-epochs 2 \
  --val-trips 2048 --val-every 4 --patience 2 --val-tol 0.005 --threads 4
/opt/homebrew/bin/python3.11 -u scripts/v4_training/stress.py eval \
  --out runs/stress/hs_conv --cpp-sampler --threads 4
