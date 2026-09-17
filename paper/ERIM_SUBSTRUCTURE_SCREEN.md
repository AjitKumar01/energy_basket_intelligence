# Do ERIM categories contain nested substitute groups?

Date: 2026-09-17

## Question

The nested substitution-group extension ([NESTED_SUBSTITUTION_GROUPS.md](NESTED_SUBSTITUTION_GROUPS.md))
lets sub-groups inside a category substitute **more strongly** than the category as a
whole (ρ_v ≥ 0 at every node). Before spending hours on a synthetic decision experiment
or days on implementation, this screen asks whether ERIM shows that pattern at all.

## Method (model-free, `scripts/verification/nested_rho/substructure_check.py`)

- **Pair shortfall.** For products j, k in one category, from training trips only:
  - O_jk is the number of trips that contain both products.
  - E_jk = Σ_h q_h a_hj a_hk is the household-adjusted expectation: each household's
    purchases placed independently across its own trips.
  - A set of pairs has shortfall log((ΣO + 1)/(ΣE + 1)).
  - A pair penalty ρ lowers the shortfall by roughly ρ. So the **contrast** between two pair
    sets reads as a difference in substitution strength: negative means the first set
    substitutes more.
- **Test 1, split-half clustering.** Cluster products on half of the households, then
  compare within-cluster with between-cluster shortfall on the other half.
  - Null: random groupings with the same cluster sizes.
  - Ten household splits; 2 and 3 clusters.
  - Only products with at least 30 training lines are included.
- **Test 2, manufacturer.** Compare same-manufacturer pairs with different-manufacturer
  pairs.
  - Manufacturer is the UPC manufacturer prefix.
  - The 95% interval comes from a household bootstrap.
- **Calibration.** Both tests were also run on the two known-truth worlds of the decision
  experiment (`artifacts/nested_rho_decision/{A,B}`):
  - world A has flat category substitution;
  - world B has brand lines, a mixed category and rare products.

## Calibration: the screen finds nested substitution when it exists

| World / category | Truth | Split-half: splits with p < 0.05 (best k) | Declared first-five vs last-five contrast (95%) |
|---|---|---:|---:|
| B cat0 | two lines, ρ 1.0 / 0.9 over 0.3 | **10/10** | **−0.96** (−1.12, −0.81) |
| B cat2 | two unrelated sets, ρ 1.2 / 1.1, category 0 | **10/10** | **−0.99** (−1.14, −0.84) |
| B cat3 | three lines (3/3/4), ρ ≈ 1 over 0.2 | **10/10** | −0.30 (−0.44, −0.17), the split cuts across lines |
| B cat1, cat4, cat5 | flat or none | 0–2/10 | −0.06 to +0.05 |
| A, all categories | flat | 0–5/10 | −0.20 to +0.15 |

Reading rules taken from calibration:
- **Split-half:** nested substitution gives ≥ 9 of 10 splits; flat truth gives at most 5.
- **Declared groups:** noise is about ±0.2; a true sub-group with Δρ ≈ 1 gives about −1.

## ERIM results

| Category | Products (≥ 30 lines) | Split-half: splits with p < 0.05 (k=2 / k=3) | Same − different manufacturer contrast (95%) | Fitted model: same − different mean φ-Gram |
|---|---:|---:|---:|---:|
| brownie | 22 | 1 / 4 | **+0.84** (0.48, 1.21) | 0.000 |
| ddinner | 11 | 0 / 0 | **+0.97** (0.36, 2.47) | 0.000 |
| ketchup | 21 | 1 / 6 | **+0.61** (0.03, 1.20) | 0.022 |
| marg | 96 | 0 / 1 | **+0.50** (0.43, 0.58) | 0.027 |
| pbutter | 63 | 1 / 1 | **+0.92** (0.69, 1.15) | 0.003 |
| sugar | 67 | 1 / 0 | **+1.45** (1.38, 1.51) | 0.069 |
| tissue | 77 | 0 / 0 | **+1.88** (1.78, 1.98) | 0.068 |
| tuna | 32 | 1 / 1 | **+2.05** (1.81, 2.30) | 0.156 |

## Findings of the first screen

1. **No brand-level or clustering-detectable nested substitution.** No category reaches the
   calibrated split-half level (≥ 9/10). Ketchup's 6/10 is the highest.
2. **Same-manufacturer products are co-purchased more, not less.** In every category the
   contrast is +0.5 to +2.0, far beyond the ±0.2 noise.
3. **The fitted category model barely captures this.** Its φ-Gram difference is 0.00–0.16.

The first version of this note concluded "do not implement nested groups." The two
follow-ups below revise that conclusion: the brand pattern is mostly a **product-type**
pattern.

## Follow-up 1: promotions and shifting loyalty (`promotion_check.py`)

- **Expectation.** A leave-one-trip-out expectation is used, so that finer adjustment cells
  cannot absorb the observed counts.
- **Calibration.** Synthetic worlds with no promotions:
  - world A contrasts stay within ±0.22 under both adjustments;
  - world B nested lines stay at −0.96 to −1.05.
- **Scope.** All splits; products with ≥ 30 lines.
- **Promotion definitions.** A pair counts as unpromoted on a trip only if neither product
  had display, feature advertising or a special-price flag in that store-week. The *strict*
  version also excludes weeks where either product sold ≥ 5% below its store's median price.

| Category | All trips | Unpromoted (flags) | Unpromoted (strict) | Household × 13-week adjustment | Both | Purchase lines promoted (strict) |
|---|---:|---:|---:|---:|---:|---:|
| brownie | +1.00 (0.71, 1.37) | +0.83 | +0.69 (0.16, 1.12) | +0.83 | +0.59 (0.07, 1.02) | 42% |
| ddinner | +1.05 (0.64, 1.54) | no feed | no feed | +0.60 (−0.05, 1.18) | — | — |
| ketchup | +0.58 (0.14, 1.12) | +0.49 | +0.31 (−0.30, 0.94) | +0.67 | +0.44 (−0.18, 1.04) | 54% |
| marg | +0.55 (0.49, 0.61) | +0.64 | +0.75 (0.64, 0.85) | +0.51 | +0.70 (0.60, 0.79) | 46% |
| pbutter | +0.92 (0.72, 1.10) | +0.81 | +0.92 (0.68, 1.20) | +0.94 | +0.94 (0.67, 1.18) | 44% |
| sugar | +1.44 (1.37, 1.50) | +1.26 | +1.26 (1.17, 1.35) | +1.27 | +1.07 (0.99, 1.16) | 40% |
| tissue | +1.75 (1.66, 1.83) | +1.61 | +1.59 (1.44, 1.73) | +1.72 | +1.53 (1.42, 1.69) | 57% |
| tuna | +2.01 (1.82, 2.19) | +1.55 | +1.46 (1.19, 1.74) | +1.96 | +1.44 (1.14, 1.76) | 61% |

**Result.**
- Promotions explain a modest share: tuna −0.55, brownie −0.31, ketchup −0.27, sugar
  −0.18, tissue −0.16. For margarine and peanut butter, none.
- Shifting loyalty explains little: at most −0.45 (ddinner), and −0.17 for sugar.
- Excluding promoted weeks and adjusting for quarterly loyalty together still leaves +0.44
  to +1.53. Only ketchup's interval includes zero.

## Follow-up 2: product types (`type_check.py`)

**Why.** The same-manufacturer pairs with the largest excess were different *types* from
one brand: brown + powdered sugar, oil + water tuna, chunky + creamy peanut butter.

**Declared types.** The rules come from catalogue labels and were fixed before computing
any contrast:

| Category | Types |
|---|---|
| tuna | oil / water |
| peanut butter | creamy / chunky |
| sugar | granulated / brown / powdered / sugar substitute |
| margarine | stick / tub / squeeze |
| tissue | 1-ply / 2-ply |

Brownie, ketchup and dry dinner have no clear type attribute.

**Classes.** Pairs are classed by (same type?, same manufacturer?). Each class is compared
with the reference class "different type, different manufacturer". Negative means the
class substitutes more strongly than the reference.

| Category (products) | Same type, different manufacturer | Same type, same manufacturer | Different type, same manufacturer | Fitted ρ_c | Fitted model φ-Gram, same type vs reference |
|---|---:|---:|---:|---:|---:|
| pbutter (75) | **−0.93** (−1.21, −0.68) | **−2.18** (−2.70, −1.81) | +1.29 (1.05, 1.54) | 0.69 | 0.00 |
| sugar (75) | **−2.21** (−2.43, −2.00) | **−0.85** (−1.11, −0.65) | +1.27 (1.22, 1.33) | −0.03 | 0.00 |
| marg (106) | **−0.96** (−1.06, −0.88) | **−0.74** (−0.86, −0.61) | +0.38 (0.31, 0.44) | 0.70 | −0.04 |
| tuna (36) | −0.34 (−0.65, 0.02) | +1.63 (1.28, 1.98) | +1.80 (1.47, 2.11) | 1.32 | −0.08 |
| tissue (80) | −0.17 (−0.33, −0.05) | +1.82 (1.71, 1.94) | +0.88 (0.72, 1.04) | 1.01 | 0.00 |

The φ-Gram column shows the same-type, different-manufacturer class.

**Result.**
1. **Clear nested substitution by product type in peanut butter, sugar and margarine.**
   - Products of the same type substitute about 1–2 units of log co-purchase more strongly
     than products of different types. Calibration reads −1 as Δρ ≈ 1.
   - Different types from one brand are bought together.
   - Sugar is exactly the synthetic "mixed category":
     - strong substitution within a type (−2.2), co-purchase across types (+1.3);
     - so the flat category penalty averages to ρ_c ≈ 0.
2. **Tuna and tissue are weaker and mixed.**
   - Type substitution across brands is small (−0.34, −0.17).
   - Same-brand variants of one type are bought together (+1.6, +1.8): colours, can styles,
     stock-up across a brand's line.
3. **The fitted category model captures almost none of this.** Its φ-Gram differences are
   ≤ 0.13 in absolute value, against 1–2 observed.
4. **Why the earlier tests missed it.**
   - Types cut across brands, so the manufacturer contrast mixed within-type substitutes with
     cross-type co-purchases.
   - Clustering on noisy pair evidence did not recover a two- to four-way type split among
     36–106 products.

## Revised decision

- **Nested substitution exists in ERIM, at the product-type level.** The ρ_v ≥ 0 tree
  (category → type) can represent the main effect in peanut butter, sugar and margarine.
- **The builder needs no data-driven clustering for this level.** The type nodes come from
  declared catalogue attributes. That avoids the rare-product coverage failure seen in the
  synthetic builder.
- **Brand-within-type effects have both signs:**
  - attraction in tuna and tissue;
  - weaker substitution than across brands in sugar.

  A nonnegative tree cannot represent them. Leave them to φ, or later allow signed nodes.
- **Cheapest real test first.** Refit ERIM with a flat **category × type** partition.
  - Products without a type stay in their category group.
  - This is a one-level partition, so the current exact code runs it unchanged. Only the
    bundle's group column changes: a `subcategory`-style partition option in
    `prepare_model_bundle.py`.
  - It drops the cross-type penalty, which the data say is not substitution anyway.
  - If it beats the category partition on held-out likelihood and recommendation, the full
    nested tree becomes the next step.
- **Synthetic experiment.** The decision experiment (`decision_experiment.py`) is not
  needed before that refit. The real-data refit is cheaper and answers the question
  directly.

## Reproduce

```bash
python scripts/verification/nested_rho/decision_experiment.py generate --world A
python scripts/verification/nested_rho/decision_experiment.py generate --world B
python scripts/verification/nested_rho/substructure_check.py
python scripts/verification/nested_rho/promotion_check.py
python scripts/verification/nested_rho/type_check.py
```

Outputs:
- `artifacts/nested_rho_decision/substructure_check.json`
- `metadata_contrast_calibration.json`
- `erim_manufacturer_vs_model.json`
- `promotion_check.json`
- `type_check.json`
- `type_check_model_gram.json`
