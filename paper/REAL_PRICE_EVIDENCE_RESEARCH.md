# Real-price evidence and learnability: corrected audit

## Final conclusion

The present real dataset does **not** identify a stable, correctly directed own-price
response. The earlier apparent pooled signal was an artifact of selecting store-price
cells that existed only when the product sold. It must not be cited as evidence that the
basket model learned real price counterfactuals.

The corrected analysis first builds a purchase-independent exposure panel, then fits on
training weeks, selects shrinkage on validation weeks, and opens the test period once.
The validation-selected model is worse than ignoring price on test. No price coefficient
artifact is certified, and the full basket pipeline now stops before training rather than
injecting unsupported price effects.

This finding does not contradict the synthetic experiment. Synthetic offers were
randomized, all opportunities—including non-purchases—were recorded, and the alternative
outcomes were known. Those conditions are absent from the real panel.

## 1. Root cause in the original evidence panel

`data/price_store_week.parquet` is reconstructed from transaction lines. Every row is a
product sale summarized to product/store/week. Direct inspection gives:

- 2,349,168 price cells;
- minimum `n_tx` = 1;
- median `n_tx` = 1; and
- zero cells with `n_tx = 0`.

Therefore a store/product/week price is observed only after at least one sale. Requiring
prices in adjacent weeks silently requires the product to sell in both weeks. The prior
same-store comparison consequently removed almost all zero-purchase outcomes before it
estimated demand response. That is selection on the dependent variable.

The earlier panel had only 41 zero-total training events among 8,296 events. Its negative
price association and the fitted sensitivity near 0.32 are invalid for price-response
inference.

The historical implementation has been retired: calling `build_store_events()` in
`scripts/version4/research_real_price_evidence.py` now fails with an explanation instead
of regenerating the biased analysis.

## 2. Corrected exposure construction

The replacement uses the same price information available to the basket model without
requiring a product sale at a particular store:

1. Start from the chain-level weekly modal loyalty price.
2. Carry it through missing weeks exactly as preprocessing does.
3. Expose it to every store in which the product belongs to the static assortment built
   from training data.
4. Count all baskets in that store/week, including explicit zero product purchases.
5. Compare adjacent weeks for the same product and store.
6. Keep only price movements of at least 5% and at most 0.70 log points.
7. Require display and mailer status to be unchanged and recorded promotion depth to move
   by less than one percentage point.
8. Require at least 16 baskets per store/week and at least 500 training product lines.
9. Only after those filters, learn each product's eligible training price-change range;
   require held-out movements to lie within it.

Sparse store-price deviations are deliberately excluded from this evidence design: those
cells are themselves revealed by sales. This leaves measurement error because the chain
price is only a proxy for a posted store price, but it removes the direct outcome-selection
bug.

The complete construction initially contains 7,370,901 store/product price-change rows.
After the prespecified filters and training-product support check:

| Period | Store events | Informative strata | Zero-total strata | Informative products |
|---|---:|---:|---:|---:|
| Training | 134,028 | 34,416 | 99,612 | 193 |
| Validation | 13,475 | 3,376 | 10,099 | 84 |
| Test | 18,317 | 4,485 | 13,832 | 94 |

An informative stratum has at least one purchase across the two adjacent weeks. A
zero-total stratum contributes exactly zero to the conditional likelihood; retaining it
proves that exposure construction no longer depends on a sale.

## 3. Focused statistical model

For a store/product price event, let `y0` and `y1` be product-purchase basket counts and
`n0` and `n1` be all eligible basket counts before and after. The fitted conditional
Poisson model is equivalent to

```text
y1 | (y0 + y1) ~ Binomial(y0 + y1,
                          sigmoid(log(n1/n0) - sensitivity * log_price_change))
```

Conditioning on the total purchase count removes the event-specific baseline rate. A
nonnegative sensitivity encodes the ordinary own-price direction. This is still an
observational predictive model, not a causal estimator: inventory, competitor activity,
demand-led price setting, and other time-varying conditions remain unobserved.

Three nested models were trained only on weeks 10–82:

- one global sensitivity;
- category sensitivities shrunk to the global value; and
- product sensitivities shrunk to their category values.

The category and product penalty grid was selected only on weeks 83–90. Weeks 91–101 were
then evaluated once. Analytic gradients were checked against finite differences, and a
separate synthetic conditional-count test recovers its known sensitivity.

## 4. Corrected results

### 4.1 There is no positive pooled response

The nonnegative global training optimum is exactly zero. Allowing the coefficient to have
either sign gives:

| Period | Unconstrained sensitivity | 95% interval clustered by product |
|---|---:|---:|
| Training | -0.0070 | [-0.0638, 0.0498] |
| Validation | -0.2150 | [-0.4369, 0.0068] |
| Test | 0.0229 | [-0.1183, 0.1642] |

With the model convention, a positive sensitivity is ordinary demand: a price increase
reduces purchase incidence. The negative estimates instead associate higher measured
prices with higher purchase rates. All three intervals contain zero, and the point estimate
changes sign between validation and test. This is not evidence for ordinary price response
and is not a stable drift from one positive elasticity to another.

### 4.2 Product flexibility overfits validation

Validation selected the product hierarchy with category penalty 0.1 and product penalty
0.001. It improved conditional log loss slightly on validation:

```text
no-price validation loss     0.6855100819
selected validation loss     0.6853813400
```

On the untouched test period it reversed:

```text
no-price test loss           0.6889408453
selected test loss           0.6899150632
selected minus no-price     +0.0009742179  (worse)
```

Every cluster bootstrap interval includes zero and extends in the harmful direction. None
supports a test improvement:

| Resampling unit | 95% interval for selected minus no-price loss |
|---|---:|
| Product | [-0.000178, 0.003008] |
| Store | [-0.000253, 0.002228] |
| Week | [-0.000300, 0.002867] |

The point estimate is worse overall, and no clustering view establishes a benefit. The
product model assigned zero sensitivity to 74 of 193 training products and values as high
as 4.06 to others. That heterogeneity did not generalize.

## 5. What is—and is not—learnable

The corrected data support these statements:

- A model can reconstruct historical basket probabilities conditional on observed prices.
- It cannot learn a stable price-to-demand response from the available transaction-derived
  exposure alone.
- It cannot support household-specific or catalogue-wide SKU-specific elasticities.
- The previous aggregate target of -0.121 is an external calibration assumption, not a
  result recovered from this evidence.
- The relative-price multiplier cannot be identified by a uniform-price target.
- Interaction fitting cannot repair the issue because it copies the additive-stage price
  parameters unchanged.

The likely causes are price endogeneity, incomplete posted-price measurement, absent
stock/availability histories, unmeasured promotions and competitor conditions, and the
fact that only nonempty loyalty-card baskets are observed. Merely increasing model rank,
particles, epochs, or catalogue-scale embeddings cannot manufacture this missing
identification.

## 6. Implemented correction and fail-closed behavior

`scripts/version4/fit_supported_price_response.py` now:

- constructs the complete exposure panel;
- verifies counts, support, and duplicate-free event keys;
- fits global/category/product hierarchies with bound-aware convergence checks;
- chooses hierarchy and shrinkage using validation only;
- compares the frozen choice with no-price on locked test data;
- bootstraps loss differences by product, store, and week; and
- writes a non-certified marker instead of coefficients when gates fail.

The full pipeline runs this evidence stage before the expensive additive fit. The additive
trainer can load, hash, freeze, and preserve a future certified price component with
relative-price multiplier equal to one. On the present data the evidence stage exits
nonzero, so expensive basket training does not start with unsupported coefficients.

## 7. What a retailer must supply next

To estimate causal or operational price effects, collect a store/SKU/time panel containing
the offered price even when no sale occurs, plus inventory/stockout state, all promotions,
competitor prices where relevant, store traffic, units, costs, and margin. Then create
exogenous variation through a randomized store-SKU-week or eligible-customer offer test
inside safe business bounds.

Training must use only the randomized or otherwise identified exposure period. Validation
selects pooling. A locked test must beat no-price and pooled baselines with uncertainty
clustered at the randomization unit. Only after that should counterfactual price endpoints
be enabled. Until then, the model's non-price basket-completion and generation functions
can be evaluated independently, but price scenarios are simulations under an assumed
coefficient—not evidence-backed forecasts.

## Reproducibility

- Corrected implementation: `scripts/version4/fit_supported_price_response.py`
- Focused tests: `tests/test_supported_price_response.py`
- Run log: `artifacts/supported_price_response_20260915/run.log`
- Machine report: `artifacts/supported_price_response_20260915/report.json`
- Fail-closed coefficient marker:
  `artifacts/supported_price_response_20260915/coefficients.json`

The artifacts are local generated outputs and are intentionally not treated as source
files. The JSON report records the implementation hash and every split/gate value.
