#!/bin/zsh
# M4b on the stress world: two starting values for lambda (0.5 and ~0.95), train only; choose on validation.
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT="$PWD/data/stress_world/model_input_category"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
for init in 0.0 3.0; do
  tag=hs_m4b_i${init%.*}
  python3 -u scripts/v4_training/rerun_with.py runs/stress/hs_rec.log --out runs/stress/$tag --skip-eval \
    --set iv_slope=True iv_slope_init=$init > runs/stress/$tag.out 2>&1 || exit 1
done
