# ERIM Multi-Category Basket Run

## Purpose and honest scope

This run tests the basket model on the eight product categories available from the
official Chicago Booth ERIM archive: brownies, frozen dinners, ketchup, margarine,
peanut butter, sugar, tissue and tuna. A modeled basket is the nonempty set of products
from these eight categories bought by one household at one store, on one day and trip.
It is not the household's complete grocery checkout. Any retail interpretation must say
“tracked-category sub-basket,” not “whole basket.”

The source archives and every extracted file are bound to SHA-256 digests in
`data/erim_basket/source_manifest.json`. The canonical build requires a household-store-
week to occur in the observation panels of all eight selected categories before it can
contribute a purchase. This produces a common 51-week window.

## Canonical build

The source-specific `ERIMMultiCategoryBasketAdapter` ends at a dataset-neutral contract:

- transactions identified by household, store, period, day, trip and product;
- a product catalogue with stable composite identifiers (`category:UPC`);
- shopping opportunities;
- store-period prices with their source recorded; and
- promotion observations.

Composite identifiers are necessary because masked/private-label UPC values can collide
between category archives. The train/validation/test split is chronological. Products
and households are selected using training data only; validation and test cannot create
new support.

The retained cohort contains 464 products, 3,862 households, 35 stores, 139,021
tracked-category baskets and 231,694 basket-product rows. Mean distinct basket size is
1.667, the maximum is 10, 37.66% of baskets span more than one tracked category, and
12.77% span at least three.

## Price and promotion treatment

**Correction (16 September 2026).** The sentence below describes the original bundle's
declared label; the model actually used chain store-week sales prices that mixed retail
aggregates with panel-purchase fallbacks. Those fallbacks are outcome-selected. The
corrected contract prices the model from retail aggregates only; see
[ERIM_RETAIL_PRICE_REFIT.md](ERIM_RETAIL_PRICE_REFIT.md).

The basket incidence fit uses purchase unit expenditure as its modeled paid-price basis.
Nineteen retained nonpositive purchase lines keep their observed incidence but receive a
reference price estimated from positive store/product/week purchases; this is recorded in
the canonical audit. Prices are never silently replaced by zero.

Promotion regressors are disabled in the first basket fit. Seven ERIM categories contain
retail promotion aggregates, but frozen dinners do not. Encoding its missing promotion
history as “no promotion” would be false. The original observed promotion panel is still
preserved separately for price-evidence filtering.

Price counterfactual support is tested independently before basket fitting. Only price
cells from the ERIM retail aggregate are eligible; purchase-derived fallback prices are
excluded. Before/after events with display, advertising or special-price activity are
excluded. The price hierarchy is selected on validation data and opened once on the
chronological test data. If no candidate improves over the no-price predictor under the
product-, store- and week-clustered gates, every basket-model price coefficient is fixed
to zero. A failed price gate does not prevent evaluation of price-free basket composition,
completion or generation, but it forbids price counterfactual claims.

## Reproducible data stages

Run from the repository root:

```bash
python -u scripts/data/fetch_erim_categories.py
python -u scripts/build_erim_basket_dataset.py --config configs/erim_basket.json
python -u scripts/build_canonical_basket_input.py \
  --canonical-dir data/erim_basket/canonical \
  --output-root data/erim_basket/model_input \
  --price-basis erim_chain_week_retail_aggregate_unit_price_carried_forward_constant_training_reference_without_retail_price \
  --promotion-feature disabled --model-price-sources retail_aggregate

ENERGY_MODEL_DATA_ROOT="$PWD/data/erim_basket/model_input" V3_AFFINITY=1 \
  python -u scripts/version4/build_affinity_partition.py \
  --minimum-pair-count 8 --maximum-group-size 128 \
  --output data/erim_basket/model_input/basket_input/items_affinity.parquet

ENERGY_MODEL_DATA_ROOT="$PWD/data/erim_basket/model_input" V3_AFFINITY=1 \
  python -u scripts/version4/data.py --force

ENERGY_MODEL_DATA_ROOT="$PWD/data/erim_basket/model_input" \
  python -u scripts/version4/provenance.py
```

`ENERGY_MODEL_DATA_ROOT` isolates every model-facing ERIM file from the Dunnhumby input
bundle. The resulting fingerprint covers the canonical model inputs, training-only
affinity partition, ragged index, independent price-evidence panel and observed-promotion
panel.

## Run artifacts and decision rules

All fitted artifacts and reports are under `artifacts/erim_basket_run/`. The additive fit
uses exact normalization over nonempty tracked-category baskets with support from 1 to 16
items. Six items beyond the observed maximum prevent the fitted law from being forced to
truncate at the sample maximum.

The final report must state separate verdicts for:

1. price evidence;
2. additive held-out likelihood and size calibration;
3. independently stable interaction rank;
4. interaction likelihood gain;
5. basket completion/recommendation metrics;
6. basket generation and numerical diagnostics; and
7. tail-size calibration.

Passing one verdict never implies another. In particular, good basket completion does not
validate price counterfactuals, and a failed price test does not invalidate price-free
basket completion.

## Logs

- Download: `artifacts/erim_category_download.log`
- Canonical build: `artifacts/erim_basket_build.log`
- Model-input build: `artifacts/erim_model_input_build.log`
- Affinity partition: `artifacts/erim_affinity_build.log`
- Ragged index: `artifacts/erim_ragged_index_build.log`
- Data fingerprint: `artifacts/erim_data_fingerprint.log`
- Price evidence: `artifacts/erim_basket_run/03_supported_price.log`
- Full additive console: `artifacts/erim_basket_run/05_additive_full_console.log`
- Full additive internal log: `artifacts/erim_basket_run/additive_full/v3_erim_additive.log`

- Initialization: `artifacts/erim_basket_run/01_initialize.log`
- Rank selection: `artifacts/erim_basket_run/06_rank_selection.log`
- Interaction fit: `artifacts/erim_basket_run/07_interaction_fit.log`
- Household-size residual: `artifacts/erim_basket_run/08_household_size.log`
- Validation/test likelihood: `artifacts/erim_basket_run/09_likelihood_validation.log`
  and `artifacts/erim_basket_run/10_likelihood_test.log`
- Recommendation: `artifacts/erim_basket_run/11_recommendation.log`
- Counterfactual/generation mechanics: `artifacts/erim_basket_run/12_generation_counterfactual.log`
- Customer segments: `artifacts/erim_basket_run/13_customer_segments.log`
- Interaction embedding audit: `artifacts/erim_basket_run/14_interaction_embedding.log`
- Train/validation/test population tails:
  `artifacts/erim_basket_run/15_population_size.log`,
  `artifacts/erim_basket_run/19_population_size_validation.log`, and
  `artifacts/erim_basket_run/20_population_size_test.log`
- Size-phase diagnosis: `artifacts/erim_basket_run/16_size_phase_diagnostic.log`
- Full generation calibration: `artifacts/erim_basket_run/17_real_generation_calibration.log`
- Retail application audit: `artifacts/erim_basket_run/18_retail_applications.log`

## Completed-run results

The validation-selected ERIM price hierarchy failed the locked-test comparison: its
conditional log loss was 0.695127 versus 0.692200 for the no-price predictor. The global
price coefficient made only a 0.000117 improvement and this was not robust to clustering
by product, store and week. All basket-model price coefficients are consequently fixed to
zero. Price counterfactual and pricing-policy capabilities are unavailable for this run.

The selected interaction rank is eight. The stratified interaction estimator has
cross-fit gain 0.01305, minimum half gain 0.01250 and minimum within-band ESS fraction
0.565. Against the exact additive parent, the final model improves held-out log
likelihood by 0.008603 on validation and 0.007391 on test; both clustered confidence
intervals are strictly positive after the numerical-error allowance.

On 752 test hide-one cases, MRR is 0.2381 versus 0.1284 for training popularity, and
recall@20 is 0.5625 versus 0.3816. The interaction block's incremental MRR over its fitted
parent is not resolved: gain 0.00212 with 95% interval [-0.00161, 0.00584]. The principal
evidence for the interaction block is therefore held-out likelihood, not MRR.

The 1,024-context, four-replicate generator audit passes sampler correctness. Minimum ESS
is 0.904, particle expected-size error is 0.00241 items, and the maximum q10/q11 expected-
size discrepancy is 0.0100. Fitted mean size is 1.6135 versus 1.7002 observed. This passes
the declared +/-0.25-item equivalence margin but is a statistically visible conservative
bias. Size/category/item total variation is 0.0427/0.0321/0.2468, compared with observed
split-half sampling references 0.0430/0.0626/0.2713.

The broad population safety gates pass on train, validation and test, but the strict
absolute 0.0005 tail-calibration gate for sizes at least five fails on every split. The
test fitted mean is nevertheless close to observed (1.6185 versus 1.6256). Sizes 11--16
extend beyond the observed training maximum of ten. Their corrected aggregate mass is
about 0.026% on validation/test, but reaches 5.71% for the most exposed confirmed test
context. This support extension is model extrapolation and remains not identifiable.

The larger retailer audit supports offline cross-sell ranking: MRR 0.2522 versus 0.1393
for popularity on 128 partial validation baskets. Matched-pair bundle and same-category
SKU retrieval are promising, but small/constructed. Remaining-item RMSE improves only
from 1.305 to 1.284 and is inconclusive. Checkout-stop probabilities fail calibration
against the training-size baseline even though their AUC is higher. Real stockout
substitution is not identified because availability and lost demand are absent.

The complete machine-readable verdict is
`artifacts/erim_basket_run/verification_status.json`; the narrative audit is
`artifacts/erim_basket_run/FINAL_AUDIT.md`. The complete test suite passes: 186 tests and
zero failures.
