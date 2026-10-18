#!/usr/bin/env bash
# Step 1 (T1, hs_t1): training only. Unfreeze gamma, beta, price_kappa and psi in SML and
# calibration stages with their own learning rate (--lr-price 0.003) and weak prior (--price-prec 2.0)
# centred on the initial Phi=0 values; everything else is the hs_m3 recipe.
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT="$PWD/data/stress_world/model_input_category"
SHIM_DYLIB="$PWD/scripts/v4_training/libshm_shim.dylib"
if [[ ! -f "$SHIM_DYLIB" || "$PWD/scripts/v4_training/shm_shim.c" -nt "$SHIM_DYLIB" ]]; then
  clang -dynamiclib -O2 "$PWD/scripts/v4_training/shm_shim.c" -o "$SHIM_DYLIB"
fi
export DYLD_INSERT_LIBRARIES="$SHIM_DYLIB"
/opt/homebrew/bin/python3.11 -u scripts/v4_training/stress.py train \
  --out runs/stress/hs_t1 --cpp-sampler --tie-phi --f-extra --presence-phi --cat-rho0 --rank 16 \
  --unfreeze-price --lr-price 0.003 --price-prec 2.0 \
  --phi-prec 20 --delta-prec 200 --pool-prod 0.5 \
  --lr-base 0.0015 --lr-lam 0.008 --mm-lam 0.5 --mm-rho 0.2 --mm-rho0 0.5 \
  --sml-epochs 28 --min-sml-epochs 16 --cal-epochs 2 \
  --val-trips 2048 --val-every 4 --patience 2 --val-tol 0.005 --threads 4
/opt/homebrew/bin/python3.11 -u scripts/v4_training/stress.py eval \
  --out runs/stress/hs_t1 --cpp-sampler --threads 4
/opt/homebrew/bin/python3.11 -u scripts/v4_training/eval_3seed_auc.py \
  --checkpoint runs/stress/hs_t1.pt --output runs/stress/hs_t1_3seed_auc.json --threads 4
/opt/homebrew/bin/python3.11 -u scripts/v4_training/diagnose_multicount_pull.py \
  --checkpoint runs/stress/hs_t1.pt --output runs/stress/hs_t1_multicount_diagnostic.json --threads 4
/opt/homebrew/bin/python3.11 -u scripts/v4_training/diagnose_offset_gap.py \
  --m3-checkpoint runs/stress/hs_t1.pt --m3-per-trip runs/stress/hs_t1.per_trip.npz \
  --out-dir runs/stress/hs_t1_gap --threads 4
