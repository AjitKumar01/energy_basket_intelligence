#!/usr/bin/env bash
# Generate the synthetic stress world and build its model-data bundle.
#   OUT   folder to write (required): OUT/canonical, truth files, OUT/model_input_category
# About 2 minutes.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=${OUT:?set OUT to the folder to write}
python3 -u synthetic/generate_stress_world.py --output "$OUT"
python3 -u data_prep/canonical/prepare_bundle.py --config data_prep/configs/stress_world.json \
  --canonical "$OUT/canonical" --out "$OUT/model_input_category"
