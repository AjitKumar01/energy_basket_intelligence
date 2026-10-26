# Source this before any bench command:  source bench/env.sh
# Every value can be set beforehand to override the default.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")/.." && pwd)
export REPO
export PY=${PY:-python3}                         # the Python that has requirements.txt
export V4_BUILD_DIR=${V4_BUILD_DIR:-$REPO/build/native}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
# Synthetic discount world, seed 3 (nmax 31, 48 categories, 522 products).  Its checkpoint
# records paths relative to SYN_RUN_DIR, so the scripts change into that directory first.
export SYN_RUN_DIR=${SYN_RUN_DIR:-$REPO}
export SYN_DATA=${SYN_DATA:-$SYN_RUN_DIR/runs/disc_c48_s3/world/model_input_category}
export SYN_CK=${SYN_CK:-runs/disc_c48_s3_promo/v4_comp.pt}
# Dunnhumby commodity model (nmax 120, 187 categories, 5,455 products).
export REAL_RUN_DIR=${REAL_RUN_DIR:-$REPO}
export REAL_DATA=${REAL_DATA:-$REAL_RUN_DIR/runs/bundle_commodity}
export REAL_CK=${REAL_CK:-runs/commodity/v4_comp_promo.pt}
