# Canonical input contract: running the pipeline on any dataset

The pipeline is dataset-agnostic. A dataset enters through one **canonical directory**, and
every dataset-specific choice lives in a dataset configuration file, not in code.

```text
source data ──adapter──▶ canonical directory ──prepare_model_bundle.py + config──▶ model-data bundle
                                                                                      │
                                           run_pipeline.py --model-data-root ◀───────┘
```

The adapter is the only dataset-specific code. This repository ships one:
`scripts/synthetic/generate_stress_world.py`, which writes a synthetic retailer directly in
canonical form (see `SYNTHETIC_STRESS_WORLD.md`).

## 1. Canonical directory

The validator is `scripts/version4/canonical_contract.py`. It runs automatically at the start
of every bundle build.

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
`retail_aggregate` alone, because panel-derived prices exist only when the panel bought the
product. Retail-aggregate cells also drive the optional store-availability rule.

### `promotions.parquet`

Columns: `product_id`, `store_id`, `period`, `display`, `advertised`, `special_price`
(booleans). It may be empty.

### Optional `shopping_opportunities.parquet`

Columns: `customer_id`, `store_id`, `period`.

### `build_audit.json`

It must contain `source_sha256` and `audit.cohort_policy.minimum_product_training_lines`.

### Limits

- **Periods:** at most 127. **Days:** at most 1023. Both come from the sparse feature key
  strides.
- **Conditioning on a trip.** A basket is the nonempty set of tracked products bought on one
  trip. Trips with no tracked purchase are not modeled.

## 2. Dataset configuration

Example: `configs/datasets/stress_world_category.json`.

| Key | Meaning | Default |
|---|---|---|
| `dataset_name` | name recorded in the bundle | required |
| `canonical_dir`, `model_data_root` | input directory and bundle output (relative to the repository root) | required |
| `price_basis` | declared description of the modeled price | required |
| `promotion_feature` | `disabled`, `advertised` or `special_price` | `disabled` |
| `model_price_sources` | canonical price sources allowed to price the model | `["retail_aggregate"]` |
| `availability.rule` | `disabled` (declared catalogue) or `retail_first_sale` | `disabled` |
| `availability.left_censor_periods` | first sales this close to the feed start count from the start | 13 |
| `affinity.partition` | `affinity`, `category`, `finest_catalogue_level`, `catalogue_hierarchy` or `substitution_evidence`; see §4 | `affinity` |
| `affinity.category_boundary`, `minimum_expected_cooccurrence`, `evidence_threshold`, `complement_threshold`, `maximum_group_size`, `unassigned`, `maximum_dense_pairs` | `substitution_evidence` only; see `SUBSTITUTION_EVIDENCE_PARTITION.md` | true, 5, −0.3, 0.3, 24, `singleton`, 5e7 |
| `affinity.minimum_pair_count`, `affinity.maximum_group_size` | `affinity` only: co-purchase partition settings; keep the group cap well below the catalogue size | 8, 128 |
| `affinity.minimum_group_products`, `affinity.minimum_group_training_lines` | `catalogue_hierarchy` only: floors a declared subcategory must meet to form its own group | 3, 300 |
| `product_metadata` | optional parquet keyed by `product_id` adding declared catalogue columns (`subcategory`, `brand`, `manufacturer`, `department`) without changing the canonical directory; `meta.json` records it by name and hash only | none |
| `metadata_defaults` | `MANUFACTURER` and `DEPARTMENT` when the catalogue lacks them | `UNKNOWN` |
| `promotion_coverage_note` | declared coverage of the promotion feed | generic |

Keys that do not apply to the chosen partition are rejected when the config loads.

## 3. Commands

```bash
python -u scripts/prepare_model_bundle.py --dataset-config configs/datasets/<name>.json
python -u scripts/run_pipeline.py --model-data-root <model_data_root> --run-dir artifacts/<run> --profile full
```

`prepare_model_bundle.py` runs these steps and writes `bundle_preparation.json`:
1. contract validation (after merging any `product_metadata`);
2. bundle build;
3. partition (model-free, before training);
4. ragged index;
5. data fingerprint.

`run_pipeline.py` re-verifies the fingerprint and re-hashes every model-facing file before
any stage runs.

## 4. Choosing the partition

The partition is an **input** to the model. It sets which products share the exact
within-group penalty \(\rho_c\); it never changes the model's form.

| Option | Groups | Settings | Use when |
|---|---|---|---|
| `affinity` (default) | products frequently bought together, one large residual group | `minimum_pair_count`, `maximum_group_size` | there is no usable catalogue, or categories are too large for the exact program |
| `category` | one group per merchandise category | none | categories hold competing products (typical branded grocery) |
| `finest_catalogue_level` | one group per (category, subcategory) path; products without a subcategory form their category's remainder group | none | the catalogue's finest level is trusted as the substitute unit |
| `catalogue_hierarchy` | category × declared subcategory meeting floors; the rest pooled per category | `minimum_group_products`, `minimum_group_training_lines`, optional `product_metadata` | subcategories are distinct substitute sets that do not substitute across each other |
| `substitution_evidence` | products bought instead of each other (household-level co-purchase shortfall on training trips) | see `SUBSTITUTION_EVIDENCE_PARTITION.md` | no trusted catalogue, or substitutes cut across the catalogue |

**Why it matters.** Co-purchase affinity groups put complements together, so substitutes land
in different groups and within-group substitution cannot be represented. The interaction term
\(\Phi\) expresses only complements (its matrix is positive semidefinite), so substitution comes
from \(\rho_c\) alone. Use a partition whose groups are substitute sets.

### Catalogue partitions and the partition check

`category`, `finest_catalogue_level` and `catalogue_hierarchy` share one implementation
(`scripts/version4/catalogue_partition.py`).

- **Group key.** Groups are keyed by the **full path** (category, subcategory), never by a
  subcategory label alone, because catalogues reuse labels under different parents.
- **Partition check.** Every grouping is validated before it is written: every product has
  exactly one group, group ids are contiguous, and every group lies inside one category.
- **Diagnostics.** Singleton groups, reused labels, and categories filed under more than one
  department are recorded. Departments are reported, not used.
- **Singletons.** A single-product group has no pairs, so its penalty is inert.
- **Catalogue hierarchy.** Within each category, every declared subcategory with at least
  `minimum_group_products` products and `minimum_group_training_lines` training lines becomes
  its own group; the rest stay in one category remainder group. Without declared
  subcategories the rule gives exactly the category partition. `affinity_manifest.json`
  records the floors and every subcategory's decision.
- **Where subcategories come from.** The pipeline never interprets catalogue text.
  Subcategories come from the canonical `subcategory` column or from a `product_metadata`
  file written by adapter code.

**Limitation.** The partition is one level. Substitution between groups of the same category
is not represented.
