#!/bin/zsh
# Score the chosen M4b stress run once on test, then the substitution split and the paired comparison.
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT="$PWD/data/stress_world/model_input_category"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
tag=${1:-hs_m4b_i0}
python3 -u scripts/v4_training/rerun_with.py runs/stress/$tag.log --out runs/stress/$tag --skip-train > runs/stress/$tag.eval.out 2>&1 || exit 1
python3 -u scripts/v4_training/compare_per_trip.py --new $tag=runs/stress/$tag.per_trip.npz \
  --base hs_rec=runs/stress/hs_rec.per_trip.npz --output runs/stress/$tag.compare.json >> runs/stress/$tag.eval.out 2>&1 || exit 1
echo DONE >> runs/stress/$tag.eval.out
python3 -u scripts/v4_training/diagnose_substitution.py --checkpoint runs/stress/$tag.pt --eval-json runs/stress/$tag.eval.json \
  --output runs/stress/$tag.substitution.json >> runs/stress/$tag.eval.out 2>&1 || exit 1
echo DIAG_DONE >> runs/stress/$tag.eval.out
