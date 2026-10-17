#!/usr/bin/env bash
# Category-tied Phi (phi_j = psi_c + delta_j), rank 16, otherwise as hs_tied.
# Progress: tail -f runs/stress/hs_tied_r16.out
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT="$PWD/data/stress_world/model_input_category"
python -u scripts/v4_training/stress.py train --out runs/stress/hs_tied_r16 --cpp-sampler --mm-rho 0 --phi-prec 20 --tie-phi --rank 16
python -u scripts/v4_training/stress.py eval --out runs/stress/hs_tied_r16 --cpp-sampler
