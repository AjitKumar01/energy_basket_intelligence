# ERIM retail-only price panel: diagnosis and full refit

Date: 2026-09-16

## Question

The ERIM model price panel (`log_price.npy`, `log_price_dev.npy`) was built from every
store-week price cell in the canonical bundle. Where the ERIM retail file lacked a
store-week, the adapter filled the cell from panel-household purchases. Such a cell exists
only because a panel household bought the product. Using it as a covariate conditions the
price on the modeled outcome: the same failure the Dunnhumby price research found in
transaction-derived store prices ([REAL_PRICE_EVIDENCE_RESEARCH.md](REAL_PRICE_EVIDENCE_RESEARCH.md)).

## Diagnosis

Counts are item-week cells of the chain price grid (464 products x 51 weeks).

| Finding | Evidence |
|---|---|
| Purchase-only cells are outcome-selected | 580 cells; 563 of them have a modeled basket buying that product that week. Median support is 3 units. |
| Contamination is wider than purchase-only cells | 523 further *mixed* cells, all in price-supported categories, blended retail and panel-purchase prices. |
| Frozen dinners have no retail price | All 568 of their observed cells are purchase-only. Their fitted price sensitivity is zero and kappa = 1 cancels the common component, so their utilities were unaffected. |
| Effect on price inputs | A retail-only panel changes 224 supported products; maximum change 0.31 log price and 0.43 nats of price energy, mean 0.002 nats on changed product-days. |
| Manufactured price/purchase association | Within product, bought weeks were 1.07% cheaper (current panel) versus 1.06% (retail-only). The contamination barely alters the association the price coefficient learns. |
| Held-out leakage through backfill | Leading gaps are backfilled from later weeks, but no training week was filled from a validation or test price. |
| Mislabelled basis | The bundle declared `erim_purchase_unit_expenditure...`, but the model never uses transaction unit prices. |

## Corrections

- `CanonicalBasketModelInputBuilder` takes `model_price_sources`, default
  `("retail_aggregate",)`. Products without any retail price receive a constant
  training-period reference price with exactly zero deviation. A training price may not
  be backfilled from a held-out period. The build audit records excluded cells and
  constant-price products. Passing both sources reproduces the previous bundle byte for
  byte.
- `run_pipeline.py` treats the price-evidence stage's exit status 2 as a verdict ("fit
  with price fixed to zero"), not a crash. Previously that branch was unreachable and the
  ERIM price-zero parent had to be fitted stage by stage.
- `audit_customer_segments.py` no longer ranks segments by price sensitivity when every
  household shares one frozen price response. That label had flipped between "high" and
  "low" on floating-point rounding. Clustering itself was unaffected: the constant price
  block standardizes to zero.

## Refit

1. New bundle `data/erim_basket/model_input_retail_prices` (fingerprint `8b7319da...`).
   Baskets, affinity partition and ragged index are byte-identical to the previous bundle;
   only price files differ, so both runs score identical trips.
2. Price-zero additive parent. The event-based price evidence failed as before (test loss
   delta +0.00283 versus +0.00293).
3. Joint current-price coefficients, fitted on training weeks, selected on validation and
   opened once on test. Category pooling (ridge 0.01) was selected again; sensitivities
   are 1.20-1.59 (previously 1.21-1.48). Test gain over zero price is +0.0425 nats/basket
   (household-clustered 95% interval 0.0382-0.0468). Correct product/week alignment beats
   every product and week placebo. The previous coefficients came from a smaller
   (16-dimension taste) parent, so this step alone does not isolate the panel change.
4. Full pipeline with the new coefficients: every likelihood, numerical and
   population-safety gate passed.

## Paired comparison with the previous full run

Both runs use the same model size, support 1..10 and trip manifests.

| Quantity | Previous (mixed sources) | Retail-only refit |
|---|---:|---:|
| Test log likelihood, final model | -7.6248 | -7.6220 |
| Paired test change, new minus previous | | +0.0029 (0.0021 to 0.0037) |
| Paired validation change | | +0.0027 (0.0021 to 0.0033) |
| Interaction gain over additive parent, test | +0.0075 (0.0059 to 0.0091) | +0.0076 (0.0060 to 0.0091) |
| Price contribution, price on minus off, test | +0.0417 (0.0335 to 0.0498) | +0.0449 (0.0362 to 0.0536) |
| Recommendation MRR (popularity 0.128) | 0.2623 | 0.2638; paired +0.0015 (-0.0012 to 0.0043) |
| Generated / observed mean size | 1.627 / 1.703 | 1.635 / 1.703 |
| 20% price cut: own incidence multiplier, size change | x1.303, +0.296 | x1.330, +0.300 |
| 20% price rise | x0.804, -0.167 | x0.790, -0.163 |
| Population tail gates | pass | pass |
| Promotion MDP, net-sales objective | no promotion; 0/18 actions with positive lower bound | no promotion; 0/18 |

## Conclusion

The purchase-derived price cells were a genuine outcome-selection defect, but a small one.
Removing them slightly improves held-out likelihood (+0.003 nats/basket, interval clear
of zero) and slightly strengthens the price contribution, while leaving interaction,
recommendation, generation, tail and policy conclusions unchanged. The retail-only panel
is the corrected ERIM contract. Price scenarios remain observational model scenarios, not
causal effects.

## Artifacts

- Diagnosis output and bundle build: `artifacts/erim_retail_price_refit/00_bundle_build.log`
- Price-zero parent: `artifacts/erim_retail_price_refit/price_zero_parent/`
- Joint coefficients and alignment audit: `artifacts/erim_retail_price_refit/joint_price_utility/`
- Full run: `artifacts/erim_retail_price_refit/full/` (console `04_full_pipeline.console.log`)
- Comparison: `artifacts/erim_retail_price_refit/comparison/run_comparison.json` and
  `price_ablation.json`
