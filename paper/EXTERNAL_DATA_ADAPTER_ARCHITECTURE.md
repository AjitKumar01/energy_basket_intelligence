# External retail data architecture

Date: 2026-09-15

## Why there is more than one pipeline

Retail datasets do not all observe the same random object. Treating them as if they
do would create invalid training examples rather than reusable software.

- Bayesm margarine observes one selected margarine and the prices of ten offered
  alternatives. It does not contain dates, quantities, empty category choices, or
  multi-product baskets.
- The downloaded ERIM ketchup files contain household purchase lines, household
  store-weeks, store-product-week sales and promotion flags, and product labels.
  The current prepared panel conditions on buying ketchup. It has time and
  quantities, but not a verified complete shelf assortment or a multi-category
  basket catalogue.
- Dunnhumby contains multi-product checkout baskets, but its historical price and
  promotion construction has different evidentiary limitations.

The correct common layer is therefore a validated data contract, not a forced
conversion of every row into a Dunnhumby-shaped basket.

## Implemented design

The external validation code now has four separate responsibilities:

1. `ChoiceDatasetAdapter` is the source boundary. It reads a source and returns a
   `PreparedChoiceDataset` with one canonical padded choice representation.
2. `SplitStrategy` owns splitting. `GroupedRandomSplit` is used when time is absent;
   `TemporalSplit` keeps entire periods together.
3. `ExternalChoiceExperiment` owns all model selection, fitting, metrics,
   uncertainty, price-support checks, placebos, and output formatting. It contains
   no Bayesm or ERIM branches.
4. `DatasetCapabilities` declares what the source actually observes. Unsupported
   basket and causal stages are reported as blocked rather than guessed.

Bayesm and ERIM are now thin adapters. A third `LongChoiceCsvAdapter` accepts any
one-row-per-offered-product CSV through column mappings in JSON. It validates:

- positive finite offered prices;
- at least two unique products per occasion;
- exactly one chosen product per occasion;
- constant customer, period, and covariate values within an occasion;
- unique product labels;
- complete required fields and valid split inputs.

Adding another already-long choice dataset needs a configuration entry, not a new
training script. A genuinely different raw format needs one adapter that emits the
same contract; the experiment engine remains unchanged.

## Capability gate for the complete basket model

The complete basket pipeline requires actual multi-product basket incidence,
repeated customers, and temporal ordering. Neither current external prepared panel
meets that contract:

| Capability | Bayesm margarine | ERIM ketchup panel |
| --- | --- | --- |
| Conditional product choice | Yes | Yes |
| Complete multi-product baskets | No | No |
| Temporal order | No | Yes |
| Repeated customers | Yes | Yes |
| Explicit category non-purchase | No | No |
| Known offered assortment | Yes | No; positive-sales approximation |
| Quantity fields | No | Yes |
| Promotion fields | No | Yes |
| Verified randomized prices | No | No |

Running the Dunnhumby basket stages on either prepared panel would therefore be
wrong. It would invent basket interactions for Bayesm and mistake a conditioned
ketchup choice for a complete shopping trip in ERIM.

To run the whole basket pipeline on ERIM, first download and join multiple ERIM
categories, construct household-store-day-trip baskets, build a training-only
catalogue, and audit cross-category price coverage. That new adapter must emit a
separate canonical basket contract. Only after its `full_baskets` capability passes
should interaction fitting, basket completion, generation, and basket-level price
counterfactuals be enabled.

**Update (16 September 2026).** This multi-category basket contract now exists.
`external_basket.ERIMMultiCategoryBasketAdapter` joins eight ERIM categories into
household-store-day baskets with a training-only cohort and temporal splits, and
`canonical_basket_input.CanonicalBasketModelInputBuilder` emits an isolated model-data
bundle that the unchanged Version-4 pipeline consumes through `ENERGY_MODEL_DATA_ROOT`.
See [ERIM_MULTI_CATEGORY_BASKET_RUN.md](ERIM_MULTI_CATEGORY_BASKET_RUN.md). About 2.6% of
item-week cells in that bundle's price panel are priced only from panel purchases (mostly
frozen dinners, which have no retail file); the builder's price audit reports the count.

## Configuration

The current two datasets and all common experiment settings live in
`configs/external_price_validation.json`. Run them with:

```bash
python -u scripts/run_external_price_validation.py \
  --config configs/external_price_validation.json
```

The common runner accepts any number of configured adapters and returns results
under `datasets[dataset_id]`. Dataset identifiers are not hard-coded in the engine.

For a generic long CSV, use a configuration entry shaped as follows:

```json
{
  "type": "long_choice_csv",
  "id": "retailer_category_choice",
  "sources": {"choices": "../data/retailer/choices.csv"},
  "columns": {
    "occasion": "visit_id",
    "group": "customer_id",
    "product": "sku",
    "price": "displayed_price",
    "chosen": "was_selected",
    "label": "product_name",
    "period": "week",
    "covariates": ["customer_tenure", "store_format"]
  },
  "split": {"type": "temporal", "block_name": "week"},
  "capabilities": {
    "temporal_order": true,
    "known_assortment": true,
    "outside_option": true
  }
}
```

Capabilities must describe evidence in the source, not desired functionality. In
particular, setting `randomized_prices` to true is valid only when experiment
assignment and compliance have been independently verified.
