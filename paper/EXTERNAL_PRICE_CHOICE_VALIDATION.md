# External price-choice validation

Date: 2026-09-15

## Decision

The price term is learnable from held-out observations in both external datasets
tested here. This resolves the narrow question that remained ambiguous in the
Dunnhumby data: when alternatives and their prices are observed, price adds real
predictive information beyond product preference alone.

It does **not** establish a causal price elasticity. Neither experiment verifies
that the observed prices were randomly assigned. The fitted scenarios may be used
for prediction, candidate generation, and experiment design, but a retailer must
run a randomized price or promotion trial before treating the changes as causal.

## What was tested

The benchmark is the size-one restriction of the basket energy law:

`P(chosen product | offered products, prices, context) proportional to`
`exp(product utility + context-product utility - beta * log(price))`.

The fitted price coefficient is constrained to be nonnegative. Consequently, for
a fixed choice set, increasing only one product's price cannot increase that
product's model probability. This structural rule is tested again numerically.

Two separately fitted models are compared on held-out choices:

1. A no-price model with product preference and available household context.
2. The same model with the common log-price term.

Regularization is selected on validation data. The test data are used only after
selection. Negative differences below mean the price model has lower held-out log
loss and therefore predicts the observed choice better.

## Results

| Result | `bayesm` margarine | ERIM ketchup |
| --- | ---: | ---: |
| Training / validation / test choices | 2,277 / 676 / 1,065 | 13,066 / 4,348 / 4,318 |
| Products | 10 | 75 |
| Split | Random within household; no date in source | Whole-week temporal |
| No-price test log loss | 1.8103 | 2.3440 |
| Price-model test log loss | 1.6358 | 2.1724 |
| Improvement | 0.1745 | 0.1717 |
| No-price / price top-1 accuracy | 38.22% / 44.98% | 25.08% / 30.59% |
| Household-bootstrap 95% interval for difference | [-0.2140, -0.1365] | [-0.1873, -0.1558] |
| Whole-week-bootstrap 95% interval for difference | Not available | [-0.2428, -0.0941] |
| Product-specific price-shuffle check | Passed; 1/201 tail fraction | Passed; 1/201 tail fraction |
| Supported +10% price queries | 9,046 / 10,650 (84.94%) | 70,891 / 74,030 (95.76%) |
| Own-probability monotonicity violations | 0 | 0 |
| Maximum probability-mass error | 6.66e-16 | 6.66e-16 |

The household intervals retain repeat observations from the same household as a
unit. ERIM is additionally resampled by whole week, preserving common weekly
shocks. Both ERIM intervals are wholly below zero.

The shuffle check preserves each product's held-out price distribution but assigns
those prices to the wrong choice occasions. In 200 shuffles, none predicted as
well as the actual price alignment. This rules out the trivial explanation that
the gain comes only from permanent price differences between cheap and expensive
brands.

## Worked predictive scenario

For one supported ERIM held-out choice set, the model gives Heinz Ketchup a 59.29%
choice probability at $0.97. At $1.067, a 10% increase, its probability becomes
52.58%. The largest modeled recipient is Heinz Ketchup PLS, which moves from
9.19% to 10.70%.

This is a model-implied substitution scenario, not evidence that imposing that
price would cause precisely those changes. Its legitimate use is to propose a
specific action for a randomized retailer trial and to forecast the range of
responses the trial should be powered to detect.

## Leakage and support controls

- ERIM is split by whole week: train through 1987 week 02, validate through week
  12, and test on the following 11 weeks.
- A focal household's units and expenditure are subtracted before calculating
  each store-product-week price used for that household. This prevents the chosen
  household's own purchase from mechanically setting its predictor.
- ERIM prices are average **pre-coupon** transaction prices. The source's extended
  price field is explicitly pre-coupon; coupon fields remain separate.
- A +10% query is reported only when both the factual and changed prices lie
  inside that product's train-plus-validation price range.
- Input file hashes are recorded in the result JSON.

## Limits that remain

The ERIM retail file contains products with positive recorded store-week sales.
It does not prove shelf availability for products with zero sales. The choice set
is therefore an observed-sales approximation, not a complete inventory record.

Both datasets condition on a category purchase. They answer, “which margarine or
ketchup is chosen, given that the category is bought?” They do not model whether
the shopper buys nothing in the category, basket size, visits, quantities, or
cross-category substitution. This benchmark validates the price-choice component;
it is not a validation of the complete multi-product basket model.

Average observational price remains related to promotions, store decisions,
stock, and demand from other households. Product intercepts, a temporal holdout,
focal-household removal, block uncertainty, and the shuffle test make the
predictive result credible. They do not remove those causal confounders.

## Reproduce the run

Required inputs are stored under the ignored directory
`data/external_price_validation/`. Run:

```bash
python -u scripts/run_external_price_validation.py \
  --config configs/external_price_validation.json \
  2>&1 | tee artifacts/external_price_validation_oop.log
```

Machine-readable results:
`artifacts/external_price_validation_audited.json`.

Complete console log:
`artifacts/external_price_validation_oop.log`.

The final adapter-based audited run took 27.70 seconds on the current machine.

## Retail next step

Use the fitted model to rank a small number of plausible, in-range price actions.
Randomly assign eligible store-weeks to the current price or one proposed price,
hold displays and coupons fixed where feasible, and record category non-purchase
as well as the selected product. Estimate substitution and total category demand
from that untouched trial. Only trial-supported effects should feed a production
pricing decision or profit calculation.
