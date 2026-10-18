#!/usr/bin/env bash
# One ERIM check of the stress-world M4a recipe (T1 price unfreezing + M2/M3 coupling + household
# taste per category), unchanged except for the data: ERIM category-partition base model and its
# certified Version-4 test reference. ERIM has no causal price truth, so price scenarios are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
A=/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/erim_clean_category
export ENERGY_MODEL_DATA_ROOT=/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/data/erim_basket/model_input_clean_category
SHIM_DYLIB="$PWD/scripts/v4_training/libshm_shim.dylib"
if [[ ! -f "$SHIM_DYLIB" || "$PWD/scripts/v4_training/shm_shim.c" -nt "$SHIM_DYLIB" ]]; then
  clang -dynamiclib -O2 "$PWD/scripts/v4_training/shm_shim.c" -o "$SHIM_DYLIB"
fi
export DYLD_INSERT_LIBRARIES="$SHIM_DYLIB"
BASE=(--initialization "$A/artifacts/initialization.pt" --additive "$A/out/v3_pipeline_additive_best.pt")
mkdir -p runs/erim
/opt/homebrew/bin/python3.11 -u scripts/v4_training/stress.py train "${BASE[@]}" \
  --out runs/erim/hs_m4a --cpp-sampler --tie-phi --f-extra --presence-phi --cat-rho0 --rank 16 \
  --unfreeze-price --lr-price 0.003 --price-prec 2.0 \
  --u-cat --lr-u-cat 0.005 --u-cat-prec 2.0 \
  --phi-prec 20 --delta-prec 200 --pool-prod 0.5 \
  --lr-base 0.0015 --lr-lam 0.008 --mm-lam 0.5 --mm-rho 0.2 --mm-rho0 0.5 \
  --sml-epochs 28 --min-sml-epochs 16 --cal-epochs 2 \
  --val-trips 2048 --val-every 4 --patience 2 --val-tol 0.005 --threads 4
/opt/homebrew/bin/python3.11 -u scripts/v4_training/stress.py eval "${BASE[@]}" \
  --reference "$A/reports/likelihood_test_per_trip.npz" --phi0-refit none --no-price-scenarios \
  --out runs/erim/hs_m4a --cpp-sampler --threads 4
/opt/homebrew/bin/python3.11 -u scripts/v4_training/eval_3seed_auc.py "${BASE[@]}" \
  --checkpoint runs/erim/hs_m4a.pt --output runs/erim/hs_m4a_3seed_auc.json --threads 4
/opt/homebrew/bin/python3.11 -u scripts/v4_training/diagnose_multicount_pull.py "${BASE[@]}" \
  --checkpoint runs/erim/hs_m4a.pt --output runs/erim/hs_m4a_multicount_diagnostic.json --threads 4
