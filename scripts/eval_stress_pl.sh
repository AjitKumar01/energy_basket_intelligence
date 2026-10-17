#!/usr/bin/env bash
# Score the pseudo-likelihood-only checkpoint, after the running HS fit finishes.
# Progress: tail -f runs/stress/hs_pl.eval.out
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT="$PWD/data/stress_world/model_input_category"
while pgrep -f run_stress_hs10.sh >/dev/null; do sleep 30; done
python -u scripts/v4_training/stress.py eval --out runs/stress/hs_pl --checkpoint runs/stress/hs_pl.pt
