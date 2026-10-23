#!/usr/bin/env bash
set -euo pipefail
cd /Users/ajit/Projects/nf_dunnhumby/ebi_v4_competition

export ENERGY_MODEL_DATA_ROOT=/Users/ajit/Projects/nf_dunnhumby/ebi_v4t/data/stress_world/model_input_category
export OMP_NUM_THREADS=4
export DYLD_INSERT_LIBRARIES=/Users/ajit/Projects/nf_dunnhumby/ebi_v4t/scripts/v4_training/libshm_shim.dylib

PY=/opt/homebrew/Cellar/python@3.11/3.11.13/libexec/bin/python3
INIT=/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/stress_clean_confirm/artifacts/initialization.pt
ADD=/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/stress_clean_confirm/out/v3_pipeline_additive_best.pt
WORLD=/Users/ajit/Projects/nf_dunnhumby/ebi_v4t/data/stress_world
REF=/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/stress_clean_confirm/reports/likelihood_test_paired_per_trip.npz

rm -f runs/stress/v4_comp.log
"$PY" -u scripts/fit_interactions.py --initialization "$INIT" --additive "$ADD" \
  --out runs/stress/v4_comp --recency --m4c --threads 4

"$PY" -u checks/halo.py --checkpoint runs/stress/v4_comp.pt --world "$WORLD" \
  --out runs/stress/halo_v4_comp.json --threads 4

"$PY" -u scripts/evaluate.py --checkpoint runs/stress/v4_comp.pt --reference "$REF" \
  --world "$WORLD" --threads 4
