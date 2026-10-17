#!/usr/bin/env bash
# Fixed HS training (separate rho rates, C++ sampler) on the stress world, then evaluation.
# Progress: tail -f runs/stress/hs_fixed.out
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT="$PWD/data/stress_world/model_input_category"
python -u scripts/v4_training/stress.py train --out runs/stress/hs_fixed --cpp-sampler
python -u scripts/v4_training/stress.py eval --out runs/stress/hs_fixed --cpp-sampler
