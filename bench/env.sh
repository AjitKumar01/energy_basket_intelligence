# Source this before any bench command:  source bench/env.sh
# One self-contained Python for everything (standalone CPython 3.13.4 plus the pinned
# requirements.txt, all inside the project folder; no pyenv or Homebrew needed).
export PY=/Users/ajit/Projects/nf_dunnhumby/.bench-python/venv/bin/python3
export REPO=/Users/ajit/Projects/nf_dunnhumby/ebi_v4_pipeline
# Its own build of the C++ kernels (built on first use with clang++).
export V4_BUILD_DIR=$REPO/build/native-bench
export OMP_NUM_THREADS=4
# Synthetic world, seed 3 (nmax 31, 48 categories, 522 products).  Its checkpoint records paths
# relative to SYN_RUN_DIR, so the scripts change into that directory before loading.
export SYN_RUN_DIR=/Users/ajit/Projects/nf_dunnhumby/ebi_v4_scale
export SYN_DATA=$SYN_RUN_DIR/runs/disc_c48_s3/world/model_input_category
export SYN_CK=runs/disc_c48_s3_promo/v4_comp.pt
# Real data (nmax 120, 187 categories, 5,455 products).
export REAL_RUN_DIR=$REPO
export REAL_DATA=$REPO/runs/bundle_commodity
export REAL_CK=runs/commodity/v4_comp_promo.pt
