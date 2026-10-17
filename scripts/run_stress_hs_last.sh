#!/usr/bin/env bash
# Last Version-4 attempt: separate rho learning rates, no manual rho_c steps, 10x stronger Phi prior.
# Progress: tail -f runs/stress/hs_last.out
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT="$PWD/data/stress_world/model_input_category"
python -u scripts/v4_training/stress.py train --out runs/stress/hs_last --cpp-sampler --mm-rho 0 --phi-prec 20
python -u scripts/v4_training/stress.py eval --out runs/stress/hs_last --cpp-sampler
