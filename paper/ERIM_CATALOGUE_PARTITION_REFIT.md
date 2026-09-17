# ERIM refit with the catalogue-hierarchy partition (category × product type)

Date: 2026-09-17

## Question

The model-free screen ([ERIM_SUBSTRUCTURE_SCREEN.md](ERIM_SUBSTRUCTURE_SCREEN.md)) found
strong substitution within product types (peanut butter, sugar, margarine) and co-purchase
across types. Does a flat partition into category × type groups improve on the category
partition?

## Setup

- **Bundle.** `configs/datasets/erim_availability_catalogue.json` builds
  `data/erim_basket/model_input_availability_catalogue` (fingerprint `71aeaab7…`).
  - **Partition.** `catalogue_hierarchy`, with floors of 3 products and 300 training lines.
  - **Types.** Taken from `data/erim_basket/catalogue/product_types.parquet`, which ERIM's
    adapter derives from product labels.
  - **Unchanged.** Baskets, prices and availability are the same as in the category bundle.
- **Groups (17).**

  | Category | Groups |
  |---|---|
  | sugar | granulated, brown, powdered, substitute |
  | tuna | oil, water |
  | tissue | 1-ply, 2-ply |
  | peanut butter | creamy, chunky, remainder (2 products) |
  | margarine | stick, tub, remainder (2 squeeze products and 1 untyped) |
  | brownie, ketchup, dry dinner | one group each |
- **Refit and comparison.** The same four stages as the category refit
  (`artifacts/erim_catalogue_refit/`). All stages exited 0 and all gates passed. The run is
  paired against `artifacts/erim_category_refit/full` on the same trips.

## Results

| Measure | Category (8 groups) | Category × type (17 groups) | Paired difference (95%) |
|---|---:|---:|---:|
| Validation log-likelihood, final | −7.1348 | −7.1352 | −0.0004 (−0.0060, 0.0053) |
| Test log-likelihood, final | −7.2705 | −7.2715 | −0.0009 (−0.0071, 0.0052) |
| Test log-likelihood, parent | −7.2735 | −7.2734 | +0.0000 (−0.0062, 0.0063) |
| Interaction gain (test) | 0.0029 | 0.0020 | |
| MRR | 0.2596 | 0.2596 | −0.0000 (−0.0058, 0.0058) |
| Recall@5 / @10 | 0.374 / 0.489 | 0.366 / 0.496 | |
| Generated mean size (observed 1.703) | 1.681 | 1.663 | |
| Generated category total variation | **0.039** | 0.133 | |
| 20% cut: own-product multiplier / size change / min ESS | ×1.348 / +0.34 / 0.52 | ×1.347 / +0.38 / 0.50 | |
| Price ablation gain (test) | +0.048 | +0.048 (0.039, 0.058) | |
| Strict tail calibration | inconclusive | failed (broad population gates pass) | |
| Segments | 3 | 5 | |
| MDP actions with positive 95% lower bound | 4 of 18 | 3 of 30 | |
| Runtime, stage seconds | **511** | 685 | |

### Fitted group penalties

| Category | Category partition ρ_c | Category × type ρ |
|---|---:|---|
| sugar | −0.03 | granulated 2.06, brown 1.35, powdered 1.07, substitute −0.03 |
| peanut butter | 0.69 | creamy 1.77, chunky 1.18, remainder −0.56 |
| margarine | 0.70 | tub 1.32, stick 1.19, remainder 3.89 |
| tuna | 1.32 | water 1.82, oil 1.19 |
| tissue | 1.01 | 2-ply 0.60, 1-ply 0.44 |
| ketchup | 1.66 | 1.61 |

The type groups recover the within-type substitution the screen predicted. For example,
granulated sugar is at 2.06, where the whole sugar category was fitted at −0.03.

### Where the likelihood goes (test, `comparison/likelihood_by_trip_type.json`)

| Test trips | Trips | Mean change per trip | Contribution per basket |
|---|---:|---:|---:|
| Two or more products of one category, **mixed types** | 268 | +0.52 (margarine), +0.66 (tissue), +1.10 (tuna), +0.58 (peanut butter), +0.04 (sugar) | **+0.025** |
| Two or more products of one category, **same type** | 115 | −0.76 (margarine), −0.50 (tuna); +0.25 (tissue) | −0.005 |
| No category with two or more products | 3,720 | −0.021 (−0.025, −0.018) | **−0.019** |
| All (a trip can fall in more than one row) | 4,096 | | −0.001 |

## Findings

1. **Overall, the flat category × type partition is not better than the category
   partition.**
   - Likelihood and MRR are unchanged within ±0.006.
   - Generation matches the category mix worse (0.133 against 0.039).
   - It is 34% slower.
2. **The two effects cancel.**
   - Trips with mixed types gain a lot. They are no longer penalized as if brown and
     powdered sugar were substitutes.
   - But 91% of trips hold at most one product per category, and each loses about
     0.02 nats. With no penalty across types, the model puts more probability on
     multi-product baskets within a category than the data support.
3. **Cross-type pairs are therefore not free of substitution.** They substitute less than
   same-type pairs, but more than no penalty at all implies. The screen measured only the
   *relative* difference, so it could not show this.
4. **A single level cannot hold both effects:**
   - a category-level penalty for any two products of a category;
   - an additional type-level penalty within a type.

   That is exactly the nested tree (category → type) of
   [NESTED_SUBSTITUTION_GROUPS.md](NESTED_SUBSTITUTION_GROUPS.md). This refit is the
   first real-data evidence that the nested form, not a finer flat partition, is what ERIM
   needs.

## Decision

- **Keep the category partition** as the ERIM standard and API checkpoint.
- **Keep `catalogue_hierarchy` as a generic option.** It is correct and dataset-agnostic,
  but for ERIM a flat finer partition is not an improvement.
- **The next modelling step, if pursued, is the two-level nested tree** with declared
  catalogue levels: a category node and type nodes, all with ρ ≥ 0. The tree comes from the
  catalogue, so the rare-product coverage problem of the data-built tree does not arise.

## Reproduce

```bash
python scripts/build_erim_product_types.py
python -u scripts/prepare_model_bundle.py --dataset-config configs/datasets/erim_availability_catalogue.json
# stages 01–04 as in ERIM_CATEGORY_PARTITION_REFIT.md, with
#   --model-data-root data/erim_basket/model_input_availability_catalogue
#   and run directories under artifacts/erim_catalogue_refit/
python artifacts/erim_catalogue_refit/comparison/compare_runs.py
ENERGY_MODEL_DATA_ROOT=data/erim_basket/model_input_availability_catalogue V3_AFFINITY=1 \
    python artifacts/erim_catalogue_refit/comparison/price_ablation.py
```
