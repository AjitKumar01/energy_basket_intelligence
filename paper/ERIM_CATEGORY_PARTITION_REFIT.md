# ERIM refit with the merchandise-category partition

Date: 2026-09-17

## Question

On the synthetic world, the co-purchase affinity partition could not represent
within-category substitution. The merchandise-category partition could (see
[SYNTHETIC_CAPABILITY_VALIDATION.md](SYNTHETIC_CAPABILITY_VALIDATION.md)). Does switching
ERIM to the category partition improve the real-data model?

## Setup

- **Bundle.** `configs/datasets/erim_availability_category.json` builds
  `data/erim_basket/model_input_availability_category`.
  - Its baskets, items, prices and availability are byte-identical to the current
    availability bundle.
  - Only the partition differs: 8 merchandise categories instead of affinity groups (one
    residual group of 290 products plus 19 small groups).
  - Data fingerprint: `5d67c480…`.
- **Refit.** The same four stages as the availability refit, under
  `artifacts/erim_category_refit/`:
  1. price-zero parent;
  2. joint price utility;
  3. alignment audit;
  4. full pipeline with `--price-coefficients`.

  All stages exited 0, and every likelihood, numerical and population gate passed.
- **Comparison.** Both runs are scored on the same held-out trips. Paired intervals use
  household-cluster inference.
  - Comparison scripts: `comparison/compare_runs.py` and `comparison/price_ablation.py`.
  - Baseline run: `artifacts/erim_availability_refit/full` (affinity partition).

## Results

| Area | Measure | Affinity (current) | Category | Paired difference (95%) |
|---|---|---:|---:|---:|
| **Likelihood** | validation, final model | | | **+0.035** (0.027, 0.043) nats/basket |
| | test, final model | | | **+0.029** (0.021, 0.038) |
| | test, parent model | | | +0.031 (0.023, 0.040) |
| | interaction gain over parent (test) | 0.0049 | 0.0029 (0.0012, 0.0046) | |
| **Recommendation** | MRR | 0.2508 | **0.2596** | **+0.0087** (0.0011, 0.0163) |
| | recall@5 | 0.351 | 0.374 | |
| | popularity MRR | 0.128 | 0.128 | |
| **Generation** | category total variation | 0.109 | **0.039** | |
| | mean basket size (observed 1.703) | 1.677 | 1.681 | |
| **Price** | price-ablation gain (test) | | +0.048 (0.039, 0.057) | |
| | joint price stage: test gain; selected category ridge | | +0.0455 (0.041, 0.050); 0.01 | |
| | 20% cut: own-product purchase multiplier | ×1.345 | ×1.348 | |
| | 20% cut: basket size change | +0.40 | +0.34 | |
| | minimum reweighting ESS in scenarios | 0.31 | **0.52** | |
| **Population** | gates | pass | pass | |
| | strict tail calibration | inconclusive | inconclusive | |
| **Segments** | number selected | 4 | 3 | |
| **Promotion MDP** | actions with positive 95% lower bound | 0 of 24 | 4 of 18 | |
| | chosen policy | no promotion | segment 2, peanut-butter bundle, 10%/20% off | |
| **Runtime** | total stage seconds | 619 | **511** | |

### Fitted within-group penalties ρ_c (category run)

Positive ρ_c means substitution: a second product from the same category is less
likely.

| Category | ρ_c | Reading |
|---|---:|---|
| ketchup | 1.66 | strong substitution |
| tuna | 1.32 | strong substitution |
| tissue | 1.01 | substitution (matches the brand switching in the coupon lottery) |
| margarine | 0.70 | substitution |
| peanut butter | 0.69 | substitution |
| brownie mix | −0.03 | none |
| dry dinner | −0.03 | none |
| sugar | −0.03 | none |

In the affinity run, most ρ_c were negative (attraction). Affinity groups hold products
bought together, so their penalties measure complementarity, not substitution.

### Promotion recommendation

Segment 2 is offered a five-product peanut-butter bundle at 10% or 20% off.
- **Per trip, 20% off:** incremental post-discount sales 0.0045 (lower bound 0.0015),
  against a markdown of 0.0215. Reweighting ESS is 0.95.
- **Budget scenarios:**
  - at 25% of the maximum spend: 11.6 incremental sales, robust lower bound 4.3;
  - at 50% and 75%: 17.7, robust lower bound 5.9.
- **Frozen-policy test evaluation:** incremental sales 11.5 (6.6, 16.5) against a
  markdown of 52.1 (40.2, 64.0).

This is a model-conditional scenario. The independent evaluation marks causal policy
value as `not_identifiable` from observational data. At the modeled margins, the markdown
exceeds the incremental sales. So "positive incremental sales" does not mean profit.

## Findings

1. **The category partition is better on every held-out measure that compares cleanly.**
   - Likelihood rises about 0.03 nats per basket.
   - MRR rises significantly.
   - Generated category mix is about 3× closer to observed.
   - Price scenarios are more stable (higher ESS).
   - The run is 17% faster.
2. **ERIM shows substitution in five of eight categories.** The affinity partition could
   not express this. Brownie mix, dry dinner and sugar show none.
3. **The interaction gain shrinks** (0.0049 → 0.0029). Part of what the low-rank φ term
   had absorbed is now carried by ρ_c in the parent model. The total likelihood still
   improves.
4. **Promotion conclusions depend on the partition.** The affinity run recommended no
   promotion. The category run finds a small positive, substitution-aware action. It
   remains an observational model scenario, not a causal or profit claim.

## Recommendation

- Adopt `erim_availability_category.json` as the ERIM standard configuration.
- Moving the retail API to the category checkpoint needs a separate step: rerun the
  corrected completion audit on the new checkpoint and update the API defaults.
- Nested substitution groups
  ([NESTED_SUBSTITUTION_GROUPS.md](NESTED_SUBSTITUTION_GROUPS.md)) are not needed to
  obtain this gain. ERIM's flat categories are already brand-level substitute sets.

## Reproduce

```bash
python -u scripts/prepare_model_bundle.py --dataset-config configs/datasets/erim_availability_category.json
# stages 01–04 as in ERIM_AVAILABILITY_REFIT.md, with
#   --model-data-root data/erim_basket/model_input_availability_category
#   and run directories under artifacts/erim_category_refit/
python artifacts/erim_category_refit/comparison/compare_runs.py
python artifacts/erim_category_refit/comparison/price_ablation.py
```

Logs are `artifacts/erim_category_refit/0*_*.log`. Comparison outputs are
`artifacts/erim_category_refit/comparison/run_comparison.json` and
`price_ablation.json`.
