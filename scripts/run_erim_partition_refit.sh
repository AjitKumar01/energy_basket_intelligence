#!/bin/bash
# Four-stage ERIM refit for one dataset config, as used for the partition comparisons:
#   1. price-zero parent  2. joint price utility  3. price alignment audit  4. full pipeline.
# Usage: scripts/run_erim_partition_refit.sh <config suffix> [threads]
#   e.g. scripts/run_erim_partition_refit.sh evidence_residual 3
# builds configs/datasets/erim_availability_<suffix>.json into its model_data_root and writes
# the run to artifacts/erim_<suffix>_refit/. Do not edit scripts/ or tests/ while it runs
# (the pipeline's source-freeze guard aborts on any change).
set -u
cd "$(dirname "$0")/.."
suffix=$1
threads=${2:-4}
config=configs/datasets/erim_availability_$suffix.json
NEW=$(python3 -c "import json,pathlib;print(pathlib.Path(json.load(open('$config'))['model_data_root']).resolve())")
R=$PWD/artifacts/erim_${suffix}_refit
P=$R/price_zero_parent/out/v3_pipeline_additive_best.pt
mkdir -p "$R"
python -u scripts/prepare_model_bundle.py --dataset-config "$config" --force > "$R/00_bundle_build.log" 2>&1 || { echo "bundle build failed"; exit 1; }
( python -u scripts/run_pipeline.py --model-data-root "$NEW" --run-dir "$R/price_zero_parent" --profile full --threads "$threads" --stop-after additive > "$R/01_price_zero_parent.console.log" 2>&1; echo EXIT=$? >> "$R/01_price_zero_parent.console.log" ) && grep -q "EXIT=0" "$R/01_price_zero_parent.console.log" && \
( ENERGY_MODEL_DATA_ROOT=$NEW V3_AFFINITY=1 python -u scripts/version4/fit_joint_price_utility.py --checkpoint "$P" --output-dir "$R/joint_price_utility" --threads "$threads" > "$R/02_joint_price_utility.console.log" 2>&1; echo EXIT=$? >> "$R/02_joint_price_utility.console.log" ) && grep -q "EXIT=0" "$R/02_joint_price_utility.console.log" && \
( ENERGY_MODEL_DATA_ROOT=$NEW V3_AFFINITY=1 python -u scripts/version4/audit_joint_price_alignment.py --checkpoint "$P" --fit-report "$R/joint_price_utility/report.json" --output "$R/joint_price_utility/alignment_audit.json" --threads "$threads" > "$R/03_alignment_audit.console.log" 2>&1; echo EXIT=$? >> "$R/03_alignment_audit.console.log" ) && \
( python -u scripts/run_pipeline.py --model-data-root "$NEW" --run-dir "$R/full" --profile full --threads "$threads" --price-coefficients "$R/joint_price_utility/joint_price_coefficients.json" > "$R/04_full_pipeline.console.log" 2>&1; echo EXIT=$? >> "$R/04_full_pipeline.console.log" )
echo "$suffix: $(grep -h EXIT= "$R"/0*.console.log | tr '\n' ' ')"
