# Canonical input contract: running the pipeline on any dataset

The energy-basket pipeline is dataset-agnostic. A dataset enters through one canonical
directory. Everything dataset-specific lives in a dataset configuration file, not in code.

```text
source data ──adapter──▶ canonical directory ──prepare_model_bundle.py + config──▶ model-data bundle
                                                                                      │
                                           run_pipeline.py --model-data-root ◀───────┘
```

The adapter is the only dataset-specific code; ERIM's is `external_basket.py`. Dunnhumby
predates the contract and keeps its raw-CSV route at the repository root.

## 1. Canonical directory

The validator is `scripts/version4/canonical_contract.py`. It runs automatically at the
start of every bundle build.

### `transactions.parquet` (one row per purchased product in a basket)

| Column | Meaning |
|---|---|
| `basket_id` | one shopping trip |
| `customer_id` | integer 0..N−1, contiguous |
| `store_id` | integer 0..S−1, contiguous |
| `period` | integer week 1..127 |
| `day` | integer day 0..1023, with `period == day // 7 + 1` |
| `product_id` | key into `products.product_id` |
| `item_id` | model index of the product (must agree with `products`) |
| `category` | category label (must agree with `products`) |
| `quantity` | positive integer units |
| `unit_price` | positive price paid per unit |
| `split` | `train`, `validation` or `test`, chronological by period |

Invariants:
- **Basket rows.** A basket belongs to one customer, store, day and split, and lists each
  product once.
- **Splits.** Each period belongs to one split, and train < validation < test.
- **Training support.** Every product and every held-out customer has training purchases.

### `products.parquet`

| Column | Meaning |
|---|---|
| `item_id` | 0..J−1, contiguous |
| `product_id` | unique key |
| `category` | category label |
| `label` | readable description |
| optional `subcategory`, `brand`, `manufacturer`, `department` | catalogue metadata; defaults come from the config |

### `store_week_prices.parquet`

| Column | Meaning |
|---|---|
| `product_id`, `store_id`, `period` | cell |
| `units`, `revenue` | positive sales in the cell |
| `price_source` | `retail_aggregate`: an independent store feed. `purchase_aggregate`: derived from the modeled panel's purchases |

Only the sources listed in the config price the model. By default that is
`retail_aggregate` alone, because panel-derived prices exist only when the panel bought
the product. Retail-aggregate cells also drive the optional store-availability rule.

### `promotions.parquet`

Columns: `product_id`, `store_id`, `period`, `display`, `advertised`, `special_price`
(booleans). It may be empty.

### Optional `shopping_opportunities.parquet`

Columns: `customer_id`, `store_id`, `period`.

### `build_audit.json`

It must contain `source_sha256` and
`audit.cohort_policy.minimum_product_training_lines`.

### Limits

- **Periods:** at most 127. **Days:** at most 1023. Both come from the sparse feature key
  strides.
- **Conditioning on a trip.** A basket is the nonempty set of tracked products bought on
  one trip. Trips with no tracked purchase are not modeled.

## 2. Dataset configuration

Example: `configs/datasets/erim_availability.json`.

| Key | Meaning | Default |
|---|---|---|
| `dataset_name` | name recorded in the bundle | required |
| `canonical_dir`, `model_data_root` | input directory and bundle output (relative to repository root) | required |
| `price_basis` | declared description of the modeled price | required |
| `promotion_feature` | `disabled`, `advertised` or `special_price` | `disabled` |
| `model_price_sources` | canonical price sources allowed to price the model | `["retail_aggregate"]` |
| `availability.rule` | `disabled` (declared catalogue) or `retail_first_sale` | `disabled` |
| `availability.left_censor_periods` | first sales this close to the feed start count from the start | 13 |
| `affinity.minimum_pair_count`, `affinity.maximum_group_size` | training-only affinity partition | 8, 128 |
| `metadata_defaults` | `MANUFACTURER` and `DEPARTMENT` when the catalogue lacks them | `UNKNOWN` |
| `promotion_coverage_note` | declared coverage of the promotion feed | generic |

## 3. Commands

```bash
python -u scripts/prepare_model_bundle.py --dataset-config configs/datasets/<name>.json
python -u scripts/run_pipeline.py --model-data-root <model_data_root> \
    --run-dir artifacts/<run> --profile full
```

`prepare_model_bundle.py` runs these steps and writes `bundle_preparation.json`:
1. contract validation;
2. bundle build;
3. affinity partition;
4. ragged index;
5. data fingerprint.

Verification: rebuilding ERIM from `configs/datasets/erim_availability.json` reproduces
every model file of `data/erim_basket/model_input_availability` byte for byte. Only
`meta.json` gains `dataset_name`.
