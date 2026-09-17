# Synthetic capability validation of the dataset-agnostic pipeline

Date: 2026-09-17

## Question

Is the pipeline dataset-agnostic? And when the true generating model is known, which of its
capabilities are recovered correctly?

## Method

1. **Contract.** The pipeline's input is now one declared canonical directory, validated
   by `canonical_contract.py`. Dataset choices go in a JSON config; see
   [CANONICAL_INPUT_CONTRACT.md](CANONICAL_INPUT_CONTRACT.md). ERIM rebuilt from its config
   reproduces every model file byte for byte.
2. **Synthetic world.** `scripts/synthetic/generate_canonical_world.py` writes the
   canonical format from a known energy basket law (`basket_world.py`, written
   independently of the pipeline). It is checked against brute-force enumeration: the
   sampler's total variation is below 0.03, and the normalizer agrees to 1e-8.

   | Truth component | Value |
   |---|---|
   | Households, stores, products, weeks | 3,000 in 3 latent taste segments; 20 stores; 120 products in 8 categories; 52 weeks (32 train / 10 validation / 10 test) |
   | Household effects | rank-3 taste plus a basket-size propensity |
   | Prices | randomized weekly chain prices (±25%) with category sensitivities 0.6–2.4 |
   | Stock | 76% of product-store pairs from week 1; store-level launches, 69 after training |
   | Interactions | two cross-category complement "missions" (rank-2 φ) |
   | Substitution | within-category penalty ρ_c 0.4–1.2 per pair |
   | Size law | quadratic basket-size potential |
   | Store sales feed | non-panel shoppers plus the panel's own units |
   | Scale | 85,521 trips; mean basket 3.47 products |
3. **Pipeline runs.** Both used `prepare_model_bundle.py` and
   `run_pipeline.py --model-data-root`, with no code changes:
   - **run A** used the default co-purchase affinity partition, `maximum_group_size` 24;
   - **run B** used the new `"partition": "category"` option (merchandise categories).
4. **Oracle comparison.** `scripts/synthetic/evaluate_capabilities.py` compares every
   report with the truth on the same trips, products and actions:
   - log-likelihood by tensor Gauss–Hermite quadrature (error 3e-8);
   - expectations as exact derivatives of log Z₊;
   - recommendations as true add-one scores.

## Results

| Capability | Measure | Run A (affinity) | Run B (category) | Truth / reference |
|---|---|---:|---:|---:|
| **Pipeline** | contract, all stages, likelihood/numerical/population gates | pass | pass | — |
| **Availability** | confirmed cell precision / recall | 0.9985 / 0.9999 | same bundle rule | — |
| | never-stocked pairs found | 424 | 424 | 424 |
| | launch week error | 0.94 weeks | | |
| **Price response** | evidence stage | passed (product level) | passed | randomized prices |
| | sensitivity correlation / mean | 0.970 / 1.34 | 0.970 / 1.34 | 1 / 1.50 |
| | 20% cut: own-product purchase multiplier | ×1.333 | ×1.333 | ×1.375 |
| | 20% cut: basket size change | +0.71 | +0.73 | +0.63 |
| | 20% rise: own-product purchase multiplier | ×0.794 | ×0.793 | ×0.775 |
| **Held-out likelihood (test)** | final model, nats per basket | −12.908 | **−12.802** | −12.275 |
| | gap to truth (95%) | 0.633 (0.599–0.667) | **0.527 (0.495–0.558)** | 0 |
| | interaction gain over parent | +0.042 | +0.037 | |
| **Interaction structure** | pairwise effect correlation with truth | 0.09 | **0.95** | 1 |
| | slope (fitted on true) | 0.01 | **0.86** | 1 |
| | mean within-category pair effect | 0.00 | **−0.81** | −0.85 |
| | top true complements among fitted top 10% | 31% | **53%** | |
| | selected rank | 7 | 6 | 2, plus category blocks |
| **Households** | taste correlation (double-centred) | 0.68 | 0.69 | |
| | size propensity correlation | 0.69 | 0.68 | |
| | product appeal correlation | 0.76 | 0.76 | |
| **Segments** | number / adjusted Rand index | 3 / 0.65 | 3 / 0.66 | 3 / 1 |
| **Recommendation** | MRR | 0.232 | **0.238** | oracle 0.258; popularity 0.143 |
| | share of attainable lift over popularity | 78% | **83%** | |
| **Size and generation** | expected size (64 trips) | 3.29 | 3.32 | oracle 3.26; observed 3.38 |
| | generated mean size / category TV | 3.36 / 0.047 | 3.32 / 0.036 | |
| | strict tail-calibration gate | failed | failed | |
| **Promotion policy** | action value correlation with truth | 0.91 | **0.98** | |
| | model's best action is the true best | no | **yes** | |
| | chosen policy: model estimate (95% lower bound) → true value | +83 (52) → +20; +155 (94) → +26; +199 (109) → **−11** | **+42 (8) → +56** at all budgets | |

Run A's three policy entries are for budgets of 25%, 50% and 75%.

### Where the remaining 0.53 nats per basket goes (run B, test)

These oracle ablations remove a component without refitting, so they are indicative.

| True model variant | Log-likelihood |
|---|---:|
| Full truth | −12.275 |
| Without price response | −12.372 |
| Without interactions or category penalties | −12.524 |
| Household taste replaced by its segment centre | −12.496 |
| … and without the household size propensity | −12.581 |
| Without any household taste or size propensity | −14.988 |
| **Fitted model (run B)** | **−12.802** |

Households have a median of 18 training trips. Individual taste deviations and size
propensity are worth about 0.31 nats together, and they can only be partly estimated from
so few trips. The fitted model captures about 81% of the likelihood that household and
basket structure add over a model without household taste (−14.99 → −12.80, against
−12.28 for the truth). The rest is shrunken household and product estimates, not a
missing mechanism.

## Findings

1. **The pipeline is dataset-agnostic in practice.**
   - A third, never-seen dataset ran end to end with only a canonical directory and a
     config.
   - One parameter was genuinely dataset-dependent and belonged in the config: the
     affinity group-size cap. With 128 and a 120-product catalogue, everything fell into
     one group.
2. **Well recovered:**
   - availability and launches;
   - randomized price response (correlation 0.97, about 10% attenuated, so scenarios are
     slightly conservative for purchases but overstate basket growth about 12%);
   - basket size and generation;
   - the number of customer segments, and most of their membership;
   - recommendations, with 78–83% of the attainable lift over popularity.
3. **Main finding: within-category substitution depends on the partition.**
   - The default co-purchase affinity partition groups complements. Substitutes, rarely
     bought together, land in different groups, where ρ_c cannot act. The
     positive-semidefinite interaction term cannot make a whole group repel.
   - Run A therefore missed substitution entirely (pair correlation 0.09).
   - With the merchandise-category partition, substitution is recovered (−0.81 against
     −0.85; correlation 0.95).
4. **Missing substitution makes promotion decisions wrong.**
   - Run A over-valued a within-category bundle discount. Its estimates were 4–18× the
     true value, and at the largest budget the chosen policy **lost** money even though
     its 95% lower bound was positive.
   - Run B ranked promotions almost perfectly (correlation 0.98), chose the truly best
     action, and was conservative (+42 predicted, +56 true).
   - Discounting several substitutes shifts purchases among them rather than adding
     sales. A model without substitution counts that shift as new sales.
5. **Still not recovered:**
   - Individual household taste beyond segment membership is only partly recovered.
   - The strict size-tail calibration gate fails in both runs; the broad population gates
     pass.
   - The residual likelihood gap is 0.53 nats per basket.

## Recommendations

- **Choose the partition per dataset.** Use `category` when merchandise categories hold
  substitutes, which is typical of branded grocery, and when category sizes keep the exact
  within-group dynamic program affordable. Keep `affinity` only where categories are too
  large or uninformative, and know that it cannot represent substitution.
- **Re-test ERIM with the category partition.** Its eight categories are brands of one
  product type, and the coupon lottery showed brand switching. The categories hold about
  60 products each, so check dynamic-program cost first.
- **Report the partition** next to any promotion recommendation, and treat affinity-partition
  promotion values as unreliable for within-category bundles.

## Reproduce

```bash
python -u scripts/synthetic/generate_canonical_world.py --output data/synthetic_capability_world
python -u scripts/prepare_model_bundle.py --dataset-config configs/datasets/synthetic_capability_world.json
python -u scripts/prepare_model_bundle.py --dataset-config configs/datasets/synthetic_capability_world_category.json
python -u scripts/run_pipeline.py --model-data-root data/synthetic_capability_world/model_input \
    --run-dir artifacts/synthetic_capability_test/full --profile full --threads 4
python -u scripts/run_pipeline.py --model-data-root data/synthetic_capability_world/model_input_category \
    --run-dir artifacts/synthetic_capability_test_category/full --profile full --threads 4
python -u scripts/synthetic/evaluate_capabilities.py
python -u scripts/synthetic/evaluate_capabilities.py \
    --bundle data/synthetic_capability_world/model_input_category \
    --run artifacts/synthetic_capability_test_category/full \
    --output artifacts/synthetic_capability_test_category/capability_report.json
```

Reports:
- `artifacts/synthetic_capability_test/capability_report.json`
- `artifacts/synthetic_capability_test_category/capability_report.json`
- `artifacts/synthetic_capability_test_category/oracle_ablation_test.json`
