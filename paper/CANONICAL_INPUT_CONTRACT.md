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
| `affinity.partition` | `affinity` (co-purchase groups), `category` (merchandise categories) or `catalogue_hierarchy` (finest declared catalogue level meeting fixed floors; §4) | `affinity` |
| `affinity.minimum_pair_count`, `affinity.maximum_group_size` | `affinity` only: co-purchase partition settings; keep the group cap well below the catalogue size | 8, 128 |
| `affinity.minimum_group_products`, `affinity.minimum_group_training_lines` | `catalogue_hierarchy` only: floors a declared subcategory must meet to form its own group | 3, 300 |
| `product_metadata` | optional parquet keyed by `product_id` adding declared catalogue columns (`subcategory`, `brand`, `manufacturer`, `department`) without changing the canonical directory | none |
| `metadata_defaults` | `MANUFACTURER` and `DEPARTMENT` when the catalogue lacks them | `UNKNOWN` |
| `promotion_coverage_note` | declared coverage of the promotion feed | generic |

## 3. Commands

```bash
python -u scripts/prepare_model_bundle.py --dataset-config configs/datasets/<name>.json
python -u scripts/run_pipeline.py --model-data-root <model_data_root> \
    --run-dir artifacts/<run> --profile full
```

`prepare_model_bundle.py` runs these steps and writes `bundle_preparation.json`:
1. contract validation (after merging any `product_metadata`);
2. bundle build;
3. partition (co-purchase, category or catalogue hierarchy; model-free, before training);
4. ragged index;
5. data fingerprint.

Verification:
- Rebuilding ERIM from `configs/datasets/erim_availability.json` reproduces every model
  file of `data/erim_basket/model_input_availability` byte for byte. Only `meta.json` gains
  `dataset_name`.
- Rebuilding from `erim_availability_category.json` after the catalogue-hierarchy change
  reproduces `model_input_availability_category` byte for byte (fingerprint `5d67c480…`).

## 4. Choosing the partition

The partition is an **input** to the frozen model. It changes which products share a
penalty, never the model's form. There are three options:

| Option | Groups | Settings | Use when | Measured |
|---|---|---|---|---|
| `affinity` (default) | products frequently bought together, one large residual group | `minimum_pair_count`, `maximum_group_size` | there is no usable catalogue, or categories are too large for the exact program | cannot represent substitution. Synthetic pair correlation 0.09; ERIM −0.029 nats per basket against `category` |
| `category` | one group per merchandise category | none | categories hold competing products (typical branded grocery) | synthetic correlation 0.95; **ERIM standard** (+0.029 over `affinity`, significant MRR gain) |
| `catalogue_hierarchy` | category × declared subcategory meeting floors; the rest pooled per category | `minimum_group_products`, `minimum_group_training_lines`, optional `product_metadata` | subcategories are distinct substitute sets that do **not** substitute across each other | equals `category` without subcategories; ERIM category × type tied with `category` (−0.001) because types there also substitute weakly across each other |

The ERIM `affinity` figure is from the availability refit against the category refit on
the same test trips.

The partition sets which product groups share the exact within-group penalty ρ_c. Co-purchase
affinity groups put complements together, so substitutes land in different groups and
**within-category substitution cannot be represented**. On the known-truth synthetic
world, the category partition raised the pairwise-effect correlation with the truth from
0.09 to 0.95. It also turned an over-valued, loss-making promotion policy into a
correctly ranked, conservative one. See
[SYNTHETIC_CAPABILITY_VALIDATION.md](SYNTHETIC_CAPABILITY_VALIDATION.md).

### Catalogue hierarchy (`catalogue_hierarchy`)

A fixed, model-free rule declared in the config, applied once before training:
- **Own groups.** Within each category, every declared subcategory with at least
  `minimum_group_products` products and `minimum_group_training_lines` training purchase
  lines becomes its own exact group.
- **Remainder.** Products without a declared subcategory (subcategory equal to the
  category) and products in subcategories below either floor stay together in one category
  remainder group.
- **No subcategories.** Without declared subcategories the rule gives exactly the category
  partition. This was checked on ERIM and on the synthetic world: identical
  `items_affinity.parquet` and ragged index.
- **Manifest.** `affinity_manifest.json` records the floors and, per category, every
  declared subcategory with its products, training lines and whether it formed a group.

**Where subcategories come from.** The pipeline never interprets catalogue text.
Subcategories come either from the canonical `subcategory` column, or from a
`product_metadata` file written by dataset-specific adapter code:
- **Dunnhumby:** has sub-commodities natively.
- **ERIM:** its product types (for example oil- or water-packed tuna) are parsed from labels
  in `scripts/version4/erim_catalogue.py`, and written by `scripts/build_erim_product_types.py`
  to `data/erim_basket/catalogue/product_types.parquet`. See
  [ERIM_SUBSTRUCTURE_SCREEN.md](ERIM_SUBSTRUCTURE_SCREEN.md) for why types matter.
- **Why a separate file:** it keeps the canonical directory, and every bundle already built
  from it, unchanged.

**Limitation.** The partition is one level. Substitution between groups of the same
category, for example across product types, is not represented. A nested tree would
restore it ([NESTED_SUBSTITUTION_GROUPS.md](NESTED_SUBSTITUTION_GROUPS.md)).
