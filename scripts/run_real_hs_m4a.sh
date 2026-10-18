#!/usr/bin/env bash
# The stress-world M4a recipe (T1 price unfreezing + M2/M3 coupling + household taste per category)
# on a real dataset, starting from that dataset's Version-4 base model and scored against its
# certified Version-4 test reference. Real data has no causal price truth, so price scenarios are skipped.
#   usage: run_real_hs_m4a.sh OUT ARTIFACT_DIR DATA_ROOT [VAL_EVERY]
#   ARTIFACT_DIR holds artifacts/initialization.pt, out/v3_pipeline_additive_best.pt, reports/likelihood_test_per_trip.npz
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=$1; A=$2; export ENERGY_MODEL_DATA_ROOT=$3; VAL_EVERY=${4:-4}
SHIM_DYLIB="$PWD/scripts/v4_training/libshm_shim.dylib"
if [[ ! -f "$SHIM_DYLIB" || "$PWD/scripts/v4_training/shm_shim.c" -nt "$SHIM_DYLIB" ]]; then
  clang -dynamiclib -O2 "$PWD/scripts/v4_training/shm_shim.c" -o "$SHIM_DYLIB"
fi
export DYLD_INSERT_LIBRARIES="$SHIM_DYLIB"
BASE=(--initialization "$A/artifacts/initialization.pt" --additive "$A/out/v3_pipeline_additive_best.pt")
mkdir -p "$(dirname "$OUT")"
/opt/homebrew/bin/python3.11 -u scripts/v4_training/stress.py train "${BASE[@]}" \
  --out "$OUT" --cpp-sampler --tie-phi --f-extra --presence-phi --cat-rho0 --rank 16 \
  --unfreeze-price --lr-price 0.003 --price-prec 2.0 \
  --u-cat --lr-u-cat 0.005 --u-cat-prec 2.0 \
  --phi-prec 20 --delta-prec 200 --pool-prod 0.5 \
  --lr-base 0.0015 --lr-lam 0.008 --mm-lam 0.5 --mm-rho 0.2 --mm-rho0 0.5 \
  --sml-epochs 28 --cal-epochs 2 \
  --val-trips 2048 --val-every "$VAL_EVERY" --patience 2 --val-tol 0.005 --threads 4
# likelihood + 3-seed classifier (writes OUT_3seed_auc.json); no price truth on real data
/opt/homebrew/bin/python3.11 -u scripts/v4_training/stress.py eval "${BASE[@]}" \
  --reference "$A/reports/likelihood_test_per_trip.npz" --phi0-refit none --no-price-scenarios \
  --out "$OUT" --cpp-sampler --threads 4
/opt/homebrew/bin/python3.11 -u scripts/v4_training/diagnose_multicount_pull.py "${BASE[@]}" \
  --checkpoint "$OUT.pt" --output "${OUT}_multicount_diagnostic.json" --max-trips 8192 --threads 4
