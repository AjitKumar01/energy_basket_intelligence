#!/bin/zsh
cd /Users/ajit/Projects/nf_dunnhumby/ebi_v4t
export ENERGY_MODEL_DATA_ROOT=/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence PYTHONPATH=scripts/v4_training:scripts/version4 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 VECLIB_MAXIMUM_THREADS=4
for w in discount kappa; do
  python3 -u scripts/v4_training/price_scale_profile.py --initialization /Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/corrected_complete_rank5_20260913/artifacts/initialization.pt --additive /Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/corrected_complete_rank5_20260913/out/v3_pipeline_additive_best.pt     --trips 2048 --what $w --scales 0.5 1 1.5 2 3 4 6 --output runs/dunnhumby_price/additive_${w}_scale.json > runs/dunnhumby_price/additive_${w}_scale.out 2>&1
done
