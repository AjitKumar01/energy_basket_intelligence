#!/usr/bin/env bash
# HS training of Version-4 with its interactions on the stress world, then evaluation.
# Progress: tail -f runs/stress/hs10.out
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT="$PWD/data/stress_world/model_input_category"
python -u scripts/v4_training/stress.py train --out runs/stress/hs10 --pl-epochs 6 --sml-epochs 40 --sweeps 10
python -u scripts/v4_training/stress.py eval --out runs/stress/hs10
