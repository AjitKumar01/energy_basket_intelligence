#!/bin/zsh
# Step 3: refit the Dunnhumby exact additive parent with the original settings
# (artifacts/corrected_fit_full_20260913) plus chain-only prices and the discount split.
cd "$(dirname "$0")/.."
export ENERGY_MODEL_DATA_ROOT=/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
A=$ENERGY_MODEL_DATA_ROOT/artifacts/corrected_complete_rank5_20260913
O=runs/dunnhumby_refit/out
mkdir -p $O
python3 -u scripts/version4/fit_exact_additive.py --artifact $A/artifacts/initialization.pt \
  --output-dir $O --label pipeline_additive --iters 30000 --batch 128 --lr 0.002 \
  --weight-decay 1e-05 --validation-trips 1024 --validation-chunk 128 --eval-every 100 \
  --size-kl 1.0 --lr-patience 4 --lr-factor 0.5 --min-lr 6.25e-05 --convergence-patience 8 \
  --convergence-min-updates 4000 --validation-min-delta 0.001 --rkl-w 10.0 --elast-w 20.0 \
  --elast-target -0.121 --pool-prod 1.45 --lam-centre 1 --seed 29001 --threads 4 \
  --rho-c-max-category-reward 1.5 --require-convergence \
  --price-feature-contract chain_product_week_only \
  --discount-panel runs/dunnhumby_price/log_discount.npy --disc-kappa --disc-common \
  > runs/dunnhumby_refit/fit.out 2>&1
