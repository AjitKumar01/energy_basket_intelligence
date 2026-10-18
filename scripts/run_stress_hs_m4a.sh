#!/usr/bin/env bash
# Step 2 (M4a, hs_m4a): household taste per category b_tj += theta_h . U_c(j),
# initialized at 0 with a weak prior (--u-cat-prec 2.0) and its own learning rate (--lr-u-cat 0.005),
# on top of Step 1 (T1).
set -euo pipefail
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT="$PWD/data/stress_world/model_input_category"
SHIM_DYLIB="$PWD/scripts/v4_training/libshm_shim.dylib"
if [[ ! -f "$SHIM_DYLIB" || "$PWD/scripts/v4_training/shm_shim.c" -nt "$SHIM_DYLIB" ]]; then
  clang -dynamiclib -O2 "$PWD/scripts/v4_training/shm_shim.c" -o "$SHIM_DYLIB"
fi
export DYLD_INSERT_LIBRARIES="$SHIM_DYLIB"
/opt/homebrew/bin/python3.11 -u scripts/v4_training/stress.py train \
  --out runs/stress/hs_m4a --cpp-sampler --tie-phi --f-extra --presence-phi --cat-rho0 --rank 16 \
  --unfreeze-price --lr-price 0.003 --price-prec 2.0 \
  --u-cat --lr-u-cat 0.005 --u-cat-prec 2.0 \
  --phi-prec 20 --delta-prec 200 --pool-prod 0.5 \
  --lr-base 0.0015 --lr-lam 0.008 --mm-lam 0.5 --mm-rho 0.2 --mm-rho0 0.5 \
  --sml-epochs 28 --min-sml-epochs 16 --cal-epochs 2 \
  --val-trips 2048 --val-every 4 --patience 2 --val-tol 0.005 --threads 4
/opt/homebrew/bin/python3.11 -u scripts/v4_training/stress.py eval \
  --out runs/stress/hs_m4a --cpp-sampler --threads 4
/opt/homebrew/bin/python3.11 -u scripts/v4_training/eval_3seed_auc.py \
  --checkpoint runs/stress/hs_m4a.pt --output runs/stress/hs_m4a_3seed_auc.json --threads 4
/opt/homebrew/bin/python3.11 -u scripts/v4_training/diagnose_multicount_pull.py \
  --checkpoint runs/stress/hs_m4a.pt --output runs/stress/hs_m4a_multicount_diagnostic.json --threads 4
/opt/homebrew/bin/python3.11 -u scripts/v4_training/diagnose_offset_gap.py \
  --m3-checkpoint runs/stress/hs_m4a.pt --m3-per-trip runs/stress/hs_m4a.per_trip.npz \
  --out-dir runs/stress/hs_m4a_gap --threads 4
