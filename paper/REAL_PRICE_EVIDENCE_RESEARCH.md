# Why Real Price Counterfactuals Did Not Match the Synthetic Results

## Executive conclusion

The original 32-event real-data comparison was too noisy and did not compare like with
like. It should not have been used to conclude that the model had learned—or failed to
learn—real price effects.

A corrected same-product, same-store study changes the conclusion:

1. The real data contain a small but repeatable **average** relationship between price and
   product purchasing.
2. Product-specific relationships are suggested for a small, well-observed part of the
   catalogue, but they are not stable enough across all held-out periods to be established.
3. The fitted model captures a little test-period price information, but its
   product-by-product sensitivities do not reproduce consistently across periods.
4. Therefore the weak result is partly a data problem and partly a fitting problem. It is
   not logical to demand 5,455 reliable product effects from this panel. It is logical to
   require the model to learn the supported shared average and to report that most
   product-specific effects are unavailable.
5. The synthetic experiment answered a much easier identification question: offers were
   randomized, every opportunity was recorded, the catalogue had only 20 products, and
   the true alternative outcomes were known.

The current price component should be refitted hierarchically against the same-store
training evidence and judged against the untouched test period. Price scenarios should
remain research-only until that refit beats a simple pooled-price baseline and is then
validated in a randomized retailer trial.

## 1. Why the earlier 32-event result was weak

The earlier evaluator detected a price change from the chain-wide weekly modal price. It
then predicted a response while holding the original shoppers and circumstances fixed,
but compared that prediction with purchases made by a different collection of shoppers
in the next week. Those are different questions.

The audit now quantifies the problem:

- There were 3,176 nominally clean chain-level price events in the test period.
- Only 1,087, or 34.2%, had even one store with a price observation in both weeks.
- Of the 32 evaluated events, only 9 had any same-store price observation in both weeks.
- The median product was purchased only 3 times before and 3 times after the change.
- The median model effect was only 0.0014 of the ordinary sampling error of the observed
  difference. Even the largest was only 0.034 of that error.
- The median observed movement was 518 times the median predicted movement.
- The interaction and parent predictions differed by at most 0.00000367 in this panel.

The observed differences were therefore dominated by which shoppers happened to visit
and buy in each week. A 56.25% sign-agreement score from these 32 events is not an
informative test of the model.

There is also a price-measurement problem. The available prices are reconstructed from
transactions, not supplied as a complete posted-price history. A chain-wide modal price
can change because the stores contributing purchases changed. A valid price feed would
record the price offered even when nobody bought the product.

## 2. Corrected same-store research design

The new audit constructs store-product price events and holds both store and product
fixed. For each event it compares purchasing among baskets at that store in adjacent
weeks. It retains only events satisfying all of the following:

- the price moved by at least 5%, with absolute log change no greater than 0.70;
- the recorded discount depth did not change by more than one percentage point;
- display and mailer status remained unchanged;
- the product had at least 500 training purchases;
- each store-week contained at least 16 modeled shopping trips;
- the test-period price movement lay inside that product's training-period range; and
- product selection did not inspect validation or test purchases.

This still is not a causal experiment. Advertising, stock, competition, and other weekly
changes remain unobserved. It is, however, a much better test of whether the same
observational relationship repeats outside training.

The complete same-store construction found 97,521 adjacent-week price events covering
4,134 products and 113 stores. The median event still contained only one product purchase
per side, demonstrating the catalogue's severe long tail.

After the stricter support filters, the evidence panel contained:

| Period | Events | Products |
|---|---:|---:|
| Training | 8,296 | 177 |
| Validation | 997 | 82 |
| Test | 1,332 | 99 |

## 3. Is there price evidence in the real data?

Yes, at an aggregate observational level. The estimated change in product incidence per
unit change in log price was negative and similar in all three periods:

| Period | Estimated slope | Product-clustered 95% interval |
|---|---:|---:|
| Training | -0.02189 | [-0.03385, -0.00994] |
| Validation | -0.03373 | [-0.06288, -0.00458] |
| Test | -0.02444 | [-0.04287, -0.00601] |

In the test period, product incidence fell by 0.00836 on average after price rises and
increased by 0.00693 after price cuts. These are associations, but their direction and
magnitude reproduce across time.

A single response coefficient fitted only on training weeks achieved the following on all
1,332 eligible test events:

- Pearson correlation: 0.175;
- rank correlation: 0.171;
- sign agreement: 57.8%;
- mean-squared-error improvement over predicting no response: 3.0%; and
- mean-absolute-error improvement: 0.5%.

This is weak predictive information, not an absence of information. It establishes a
reasonable shared price response but not precise event-level forecasts.

## 4. Can product-specific responses be learned?

The evidence becomes stronger with repetition, but not conclusive across every held-out
comparison. Among products observed in every period:

| Minimum usable events in each period | Products | Training-to-test correlation | Validation-to-test correlation |
|---:|---:|---:|---:|
| 1 | 69 | 0.043 | 0.126 |
| 3 | 51 | 0.061 | 0.370 |
| 5 | 32 | 0.329 | 0.515 |
| 8 | 22 | 0.339 | 0.541 |
| 10 | 16 | 0.286 | 0.690 |

For the 32-product, five-event panel, product-bootstrap 95% intervals were:

- training versus validation: [0.157, 0.741];
- training versus test: [-0.010, 0.683]; and
- validation versus test: [0.057, 0.746].

The validation-to-test result suggests that some stable product differences may exist,
but the training-to-test interval includes zero. This is not enough to certify even the
32-product subset, much less separate elasticities for all 5,455 products. It provides
still less support for a separate response for every household-product pair.

## 5. Did the fitted model learn the available evidence?

Not adequately.

The frozen model was evaluated on 80 validation and 80 test events, with one event per
product. Selection used price and training support only and did not inspect the held-out
purchase response. All numerical calculations passed; minimum absolute ESS was 18.5 out
of 64.

On the 80-product test panel, the interaction model achieved:

- Pearson correlation: 0.179;
- rank correlation: 0.086;
- sign agreement: 51.9%;
- mean-squared-error improvement over predicting no response: 2.6%; and
- mean-absolute-error improvement: 1.8%.

That is a small amount of held-out predictive value. However, it did not reproduce on the
80-product validation panel: Pearson correlation was -0.085, rank correlation was -0.165,
and both absolute and squared errors became worse.

The simple training-only pooled rule had a higher test rank correlation, 0.172, but only a
1.0% squared-error improvement and worse absolute error. Neither method reproduced on the
80-product validation sample. The fitted model has therefore not demonstrated consistent
superiority over the simplest price baseline.

Most importantly, for the 32 products with at least five usable events in every period,
the fitted product price coefficient correlated with the independently measured response
as follows:

| Comparison | Correlation |
|---|---:|
| Fitted coefficient versus training evidence | -0.253 |
| Fitted coefficient versus validation evidence | 0.057 |
| Fitted coefficient versus test evidence | 0.284 |

The direction changes across periods, so the product allocation is not reliably learned.

## 6. Why the present training path misses it

### 6.1 The overall price strength is imposed, not independently learned

Training penalizes deviation from an aggregate elasticity target of -0.121. Repository
documentation already records that this target is a calibration assumption rather than
independent real-data evidence. The basket likelihood contains too little price movement
to control the overall magnitude reliably on its own.

### 6.2 The target does not identify which product is sensitive

A single average constraint can control the average response while allocating it to the
wrong products. In the current checkpoint, the product price coefficient is extremely
uneven: its median is 0.00642, its 90th percentile is 0.04495, and its maximum is 0.2114.
The relative-price multiplier is 15.0456. Most evaluated product responses are consequently
almost zero, with a small number carrying most of the total response.

### 6.3 The relative-price multiplier is not identified by the aggregate target

When every price moves together, the common/relative construction causes the relative
part to cancel. Therefore an aggregate all-price elasticity target cannot identify the
relative-price multiplier. That multiplier must be calibrated using supported
single-product or within-category price movements. It should not be allowed to drift based
only on basket likelihood.

### 6.4 The interaction fit does not refit the price parameters

The parent and interaction checkpoints have bit-identical household price factors,
product price factors, relative-price multiplier, display effects, mailer effects, and
product intercepts. The interaction stage changes basket interactions and size terms, but
it cannot repair the price coefficients learned by the additive stage. Synthetic
interaction success therefore does not imply improved real price estimation.

### 6.5 There are too many requested effects for the available repetitions

The current price block represents variation across 1,920 households and 5,455 products.
The same-store audit found tentative product evidence for tens of products, not reliable
evidence for thousands. Household-specific price response is even more weakly supported.
Without aggressive pooling, the model can fit arbitrary patterns that barely affect
likelihood.

## 7. Why synthetic success does not transfer automatically

| Synthetic experiment | Real retailer panel |
|---|---|
| 20 products | 5,455 products |
| 1,200 customers | 1,920 modeled households |
| 216,000 customer-day opportunities | 200,698 observed nonempty trips |
| 7 offers randomly assigned at customer-day level | Historical prices were not randomly assigned |
| Known assignment probability | No documented assignment mechanism |
| 25,732 trips with repeated action support | Median store-product-week event has one purchase per side |
| Visits, non-visits, quantities, cost, and profit recorded | Primarily purchase baskets; no complete opportunity, stock, or cost panel |
| True alternative-price outcomes computable | Only one factual price is observed per shopping occasion |
| Exact enumeration over a six-product maximum basket | Approximate inference over 5,455 products and baskets up to 120 products |
| Mostly correct fitted structure plus a declared mild misspecification | Unknown confounding and unknown structural misspecification |

The synthetic work verifies the mathematics and shows that the estimator works when given
the information required to identify price response. It does not demonstrate that the
real panel contains equally informative treatment variation.

## 8. What should be changed

### 8.1 Replace the price evidence panel

Use store-SKU-week posted prices covering offered products whether or not they sold. Add
stock availability, display, mailer, coupon, competitor price, and store traffic. Until
that feed exists, use the corrected same-store panel and label it observational.

### 8.2 Fit price response hierarchically

Start with one shared response. Add category-level deviations where the training data
support them. Add product deviations only after a minimum repetition and reliability gate.
Unsupported products should shrink back to their category or global response. Do not fit
household-specific price response unless repeated randomized exposure supports it.

The loss should compare the model's predicted price response directly with training-period
same-store response moments. Validation should choose the pooling strength. The test period
must remain untouched until the final comparison.

### 8.3 Identify common and relative price effects separately

Retain the common/relative theory, but estimate its two parts with different evidence:

- common price movement from broad price-level variation; and
- relative product response from supported within-store, within-category movements.

Keep the relative-scale-equals-one model as an explicit ablation. Do not select the larger
relative multiplier unless it improves held-out price-response calibration, not merely
basket likelihood.

### 8.4 Use appropriate acceptance gates

A corrected price model should be accepted only if it:

1. beats the zero-response baseline;
2. beats the training-only pooled-price baseline;
3. reproduces direction and calibration in both validation and test periods;
4. aligns product sensitivities for the prespecified supported-product subset;
5. reports unsupported products as pooled estimates rather than confident individual
   effects; and
6. passes the existing numerical and ESS checks.

### 8.5 Make the synthetic test resemble the real identification problem

Add a 5,455-product long tail, transaction-derived missing prices, store-level price
variation, promotion confounding, stockouts, and many products with no repeat price event.
The estimator should learn shared/category effects, shrink unsupported products, and state
that unsupported individual effects are unavailable.

### 8.6 Obtain causal evidence before deployment

For a production claim, randomize store-SKU-week or customer-offer prices inside safe
bounds. Record every eligible opportunity, including no visit and no purchase, along with
stock, units, cost, margin, display, and competitor conditions. Evaluate the locked model
against the randomized average effect and report uncertainty by the unit that was
randomized.

## 9. Final verdict

The real data do not justify saying “there is no price signal.” They support a modest
shared response and only tentative product differences for a small high-support subset.
They also do not justify saying “the current model learned product-level
counterfactuals.” It did not.

The scientifically defensible conclusion is:

> Real price response is learnable at a pooled observational level. Product-specific
> response is not established by the present panel, and the current
> likelihood-plus-aggregate-calibration fit does not reproduce consistently. Refitting
> with same-store response supervision and hierarchical shrinkage is warranted; causal
> price deployment still requires randomized retailer data.

## Reproducibility

- Research implementation:
  [`scripts/version4/research_real_price_evidence.py`](../scripts/version4/research_real_price_evidence.py)
- Machine-readable result:
  [`artifacts/real_price_evidence_research_20260915/report.json`](../artifacts/real_price_evidence_research_20260915/report.json)
- Per-event frozen model comparison:
  [`artifacts/real_price_evidence_research_20260915/per_event.parquet`](../artifacts/real_price_evidence_research_20260915/per_event.parquet)
- Superseded 32-event evaluation:
  [`artifacts/remaining_verification_audited_20260914/real_price_response_evaluation.json`](../artifacts/remaining_verification_audited_20260914/real_price_response_evaluation.json)
- Synthetic comparison:
  [`artifacts/remaining_verification_audited_20260914/synthetic_verification/synthetic_retailer_experiment.json`](../artifacts/remaining_verification_audited_20260914/synthetic_verification/synthetic_retailer_experiment.json)
