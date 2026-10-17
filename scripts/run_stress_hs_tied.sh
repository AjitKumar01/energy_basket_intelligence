#!/usr/bin/env bash
# Category-tied Phi (phi_j = psi_c + delta_j), otherwise as hs_last.
# Progress: tail -f runs/stress/hs_tied.out
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT="$PWD/data/stress_world/model_input_category"
python -u scripts/v4_training/stress.py train --out runs/stress/hs_tied --cpp-sampler --mm-rho 0 --phi-prec 20 --tie-phi
python -u scripts/v4_training/stress.py eval --out runs/stress/hs_tied --cpp-sampler
