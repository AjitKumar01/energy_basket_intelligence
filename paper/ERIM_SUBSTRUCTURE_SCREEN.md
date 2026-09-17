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

## Findings

1. **No evidence of nested substitution in ERIM.** No category reaches the calibrated
   split-half level (≥ 9/10). Ketchup's 6/10 is the highest, just above the flat-truth
   maximum of 5.
2. **The within-category structure goes the other way.** In every category,
   same-manufacturer products are bought on the same trip **more** often than
   different-manufacturer products (contrast +0.5 to +2.0, far beyond the ±0.2 noise):
   - substitution is concentrated **across** brands;
   - products of one brand behave like mild complements (flavours, sizes, variants).

   A ρ_v ≥ 0 tree cannot express this, because nesting can only add substitution inside a
   brand. The extension as designed does not fit ERIM.
3. **The fitted category model barely captures this brand co-purchase.** The difference in
   its low-rank interaction term is 0.00–0.16, against an observed 0.5–2.0. Its taste terms
   are household-level, and the screen already adjusts for household.
4. **Likely explanations, not yet separated:**
   - **Joint manufacturer promotions.** ERIM's promotion feed is disabled in the model, and
     deals often cover a whole brand line.
   - **Brand loyalty that shifts over time within a household.**
   - **Genuine same-trip variety buying.**

   The first two are not complementarity in the causal sense.

## Decision

- **Do not restart the nested ρ ≥ 0 decision experiment for ERIM, and do not implement the
  extension now.** The data show the opposite of the structure it models.
  - The experiment script and worlds are kept
    (`scripts/verification/nested_rho/decision_experiment.py`,
    `artifacts/nested_rho_decision/`).
  - Its first batch was stopped by a session restart before any fit completed.
- **Keep the flat category partition** (current ERIM standard and API checkpoint).
- **Worth investigating next (cheap first):**
  1. Does the same-manufacturer excess persist in store-weeks without display or feature
     advertising? If it vanishes, the gap is a missing promotion input, not missing
     structure.
  2. If it persists, a **signed** brand level inside categories could represent it: an
     attractive brand node under a repulsive category node. The tree program stays exact
     with a negative node factor under the nmax truncation, but the sign constraint in
     §7 and risk 8 of the design would have to change. A model-based check with generated
     baskets should confirm the gap first.

## Reproduce

```bash
python scripts/verification/nested_rho/decision_experiment.py generate --world A
python scripts/verification/nested_rho/decision_experiment.py generate --world B
python scripts/verification/nested_rho/substructure_check.py
```

Outputs:
- `artifacts/nested_rho_decision/substructure_check.json`
- `metadata_contrast_calibration.json`
- `erim_manufacturer_vs_model.json`
