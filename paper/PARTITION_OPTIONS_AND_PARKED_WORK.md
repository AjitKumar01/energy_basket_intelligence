# Product partition options and parked partition work

Date: 2026-09-17

The energy basket model is **frozen**. It uses one within-group penalty ρ_c per product
group, and the partition only decides which products share a penalty. Every option below
is model-free and is fixed before training, in `scripts/prepare_model_bundle.py`, step 3.
It is chosen with `affinity.partition` in a dataset config. Bundles built outside that
script (the Dunnhumby route) use `scripts/version4/build_catalogue_partition.py` for the
catalogue and evidence rules.

## 1. Options for building the groups

| # | `affinity.partition` | How groups are formed | Settings (defaults) | Needs | Measured |
|---|---|---|---|---|---|
| 1 | `affinity` (default) | products often bought **together**: pairs co-occurring at least `minimum_pair_count` times are linked, and connected products are grouped up to a size cap; the rest form one large residual group | `minimum_pair_count` 8, `maximum_group_size` 128 | baskets only | groups complements, so substitution cannot be represented. Synthetic pair-effect correlation 0.09; ERIM 0.029 nats per basket worse than `category` |
| 2 | `category` | one group per merchandise category | none | category column | synthetic correlation 0.95; **ERIM standard and API checkpoint** (+0.029 over `affinity`, significant MRR gain) |
| 3 | `finest_catalogue_level` | one group per declared (category, subcategory) **path**; products without a subcategory form their category's remainder group; no floors | none | subcategory where declared | equals `category` on ERIM and the synthetic world. Dunnhumby: 802 groups in 187 commodities, 244 single-product groups (not fitted) |
| 4 | `catalogue_hierarchy` | as option 3, but a subcategory gets its own group only if it meets floors; the rest are pooled per category | `minimum_group_products` 3, `minimum_group_training_lines` 300 | subcategory where declared | ERIM category × product type (17 groups): tie with `category` (−0.001), 34% slower |
| 5 | `substitution_evidence` | products bought **instead of** each other: household-level co-purchase shortfall on training trips (trip size and stocking adjusted, empirical-Bayes shrinkage), constrained average-linkage merging and single-product moves; leftovers as singletons or one residual group per category | `category_boundary` true, `minimum_expected_cooccurrence` 5, `evidence_threshold` −0.3, `complement_threshold` 0.3, `maximum_group_size` 24, `unassigned` singleton, `maximum_dense_pairs` 5e7 | baskets; availability panel if present | synthetic: recovers the true sets (ARI 1.0, also without a boundary). ERIM: 18 evidence groups, 240 products without evidence, stability 0.52–0.75. **Not fitted yet (parked)** |

The catalogue options (2, 3, 4) key groups by the full path and run one partition check
(`scripts/version4/catalogue_partition.py`). The check confirms that each product is in
exactly one group, that group ids are contiguous, and that every group nests in one
category. It also reports reused labels, single-product groups and department
inconsistencies. Option 5 runs the same check, and requires nesting only when
`category_boundary` is true.

**Catalogue columns.** Subcategories come either from the canonical `subcategory` column or
from an optional `product_metadata` file. That file is written by dataset-specific adapter
code: ERIM's product types come from `scripts/build_erim_product_types.py`.

Details:
- [CANONICAL_INPUT_CONTRACT.md](CANONICAL_INPUT_CONTRACT.md) §4
- [ERIM_CATEGORY_PARTITION_REFIT.md](ERIM_CATEGORY_PARTITION_REFIT.md)
- [ERIM_CATALOGUE_PARTITION_REFIT.md](ERIM_CATALOGUE_PARTITION_REFIT.md)
- [SUBSTITUTION_EVIDENCE_PARTITION.md](SUBSTITUTION_EVIDENCE_PARTITION.md)

### Not available in the frozen model

| Idea | Why not | Record |
|---|---|---|
| Two penalty levels at once (category + type nested tree) | changes the model's energy and every computation using it | [NESTED_SUBSTITUTION_GROUPS.md](NESTED_SUBSTITUTION_GROUPS.md) (deferred) |
| Sign constraint ρ_c ≥ 0 for substitute groups | changes the model's parameter constraints | [SUBSTITUTION_EVIDENCE_PARTITION.md](SUBSTITUTION_EVIDENCE_PARTITION.md) §1 |
| Groups learned during training | makes the partition depend on the fitted model (circular) | nested design §7 |

## 2. Parked work

### P1. ERIM refits with `substitution_evidence` (parked 2026-09-17)

**Status.**
- **Configs:** `configs/datasets/erim_availability_evidence.json` (singleton leftovers) and
  `erim_availability_evidence_residual.json` (one residual group per category).
- **Bundles:** built with the current code at `data/erim_basket/model_input_availability_evidence` and
  `data/erim_basket/model_input_availability_evidence_residual`.
- **Pre-training checks:** `artifacts/evidence_partition_checks/`.
- **Earlier refits:** the first attempt was stopped during stage 2, to fix scalability first.
  Its partial outputs were removed.

**Run (about 15–20 minutes each; both can run side by side with 3 threads each):**

```bash
scripts/run_erim_partition_refit.sh evidence 3
scripts/run_erim_partition_refit.sh evidence_residual 3
```

**Compare with the category run.**
- Copy `artifacts/erim_catalogue_refit/comparison/compare_runs.py` and `price_ablation.py`
  into `artifacts/erim_<suffix>_refit/comparison/`.
- Set `NEW` (or `RUN`) to `artifacts/erim_<suffix>_refit/full`.
- Run the ablation with `ENERGY_MODEL_DATA_ROOT` set to the bundle.
- Also break the likelihood down by trip type, as in ERIM_CATALOGUE_PARTITION_REFIT.md.

**What to decide.**
- Does either variant beat `category` on held-out likelihood and MRR?
- Expectation: `singleton` probably loses, because 240 products get no penalty.
  `residual_per_category` is the fair comparison.
- Report negative fitted penalties, since the sign constraint is out of scope.

### P2. Dunnhumby with catalogue or evidence partitions (parked 2026-09-17)

**Status.**
- No Dunnhumby fit uses options 2–5. Dunnhumby still uses `affinity` (300 groups).
- Partition builds only, not written into any bundle:
  - `finest_catalogue_level`: 802 groups, valid partition;
  - `substitution_evidence`: 4.1 s with a category boundary (304 evidence groups, 4,114
    single-product groups), 16.3 s without (409, 4,005).

**Before fitting,** run a model-free screen (minutes). Compare co-purchase shortfall for:
- pairs in the same sub-commodity;
- pairs in the same commodity but different sub-commodities;
- pairs in different commodities (the anchor).

The screen shows whether sub-commodities also substitute across each other. That is the
case that made the ERIM category × type partition a tie.

**Then fit** two or three options side by side (`category`, `finest_catalogue_level`,
`substitution_evidence` with `residual_per_category`), and choose on validation likelihood.
Each Dunnhumby fit takes hours. A new partition needs a new bundle directory: do not
overwrite `basket_input/items_affinity.parquet`, which existing checkpoints depend on.

### P3. Synthetic nested-group decision experiment (parked 2026-09-17)

**Status.**
- Prototype: `scripts/verification/nested_rho/decision_experiment.py`.
- Worlds: `artifacts/nested_rho_decision/{A,B}`.
- Stopped by a session restart before any fit completed. It is only relevant if the nested
  model is ever unfrozen.

**Run:** `artifacts/nested_rho_decision/run_all.sh` (about 4–5 hours).

### Resolved

The category × type bundle was rebuilt on 2026-09-18 after the product-metadata identity
fix, and its refit was rerun (`artifacts/erim_catalogue_refit`, fingerprint `3bf89a5a…`).
Per-trip likelihoods are bit-identical to the first run and every comparison output matches,
so no bundle now records an absolute path. Rerun with:

```bash
scripts/run_erim_partition_refit.sh catalogue 4
```
