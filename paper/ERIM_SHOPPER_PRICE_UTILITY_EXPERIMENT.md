# ERIM direct-price basket experiment

## Question

Does contemporaneous normalized price improve the ERIM joint basket model when it is
placed directly in product utility, as in SHOPPER, rather than estimated from a separate
before/after price-change regression?

## Model and identification boundary

For product (j) in week (t), the fitted additive basket utility is augmented by

\[
    -\tau_{c(j)}\{\log p_{jt}-\overline{\log p_j}^{\,train}\},
    \qquad \tau_{c(j)}\geq 0.
\]

The term is included in both the observed-basket numerator and every alternative basket in
the exact category/cardinality normalizer. This is therefore a joint basket likelihood,
not a separate binary purchase regression. The non-price parameters are held at the
validated price-zero additive checkpoint for this diagnostic. Price coefficients are fit
on weeks 1--30, their pooling level is selected on weeks 31--40, and weeks 41--51 are
opened only after selection.

Only 449 products with a retail-aggregate product/week price panel receive a price
coefficient. The 15 frozen-dinner products are fixed to zero because that category has no
retail aggregate file. Store/product/week price deviations are not used: those ERIM cells
exist only when a sale occurred and would make the feature outcome-selected.

This design establishes incremental held-out predictive information. It does **not** turn
the observational price series into randomized price interventions, so it does not by
itself identify causal revenue or profit effects.

## Fit selected by validation

The global sensitivity converged at 1.3241. Validation selected category pooling with a
ridge of 0.01. The supported-category sensitivities were:

| Category | Sensitivity |
|---|---:|
| Brownie | 1.4272 |
| Ketchup | 1.4780 |
| Margarine | 1.2144 |
| Peanut butter | 1.4121 |
| Sugar | 1.4724 |
| Tissue | 1.3077 |
| Tuna | 1.2666 |
| Frozen dinner | Not estimated; fixed to zero |

The category model improved over the global model by 0.00112 nats per validation basket
(household-clustered 95% interval 0.00089 to 0.00134), and by 0.00279 on test (0.00253 to
0.00305). The extra pooling level is therefore small but supported; it is not merely the
largest mean among noisy candidates.

## Held-out evidence

| Split | Baskets | Gain over zero-price model | Clustered 95% interval |
|---|---:|---:|---:|
| Validation | 26,279 | +0.04562 nats/basket | +0.04244 to +0.04880 (household) |
| Test | 26,706 | +0.03895 nats/basket | +0.03497 to +0.04294 (household) |
| Validation | 26,279 | +0.04562 nats/basket | +0.03330 to +0.05794 (household and week) |
| Test | 26,706 | +0.03895 nats/basket | +0.02108 to +0.05683 (household and week) |

Validation gains were positive in every one of its ten weeks. Test gains were positive in
all eleven weeks, although weeks 50 and 51 were individually close to zero. The result is
therefore not produced by one exceptional week.

The 2.5% of test shopping contexts with the most extreme offered-price panels gained
0.2988 nats per basket, with a
household-clustered 95% interval of 0.2628 to 0.3348. This is the expected ordering if the
added term is using price information rather than contributing an arbitrary intercept.

## Alignment falsification tests

The selected coefficients were evaluated without refitting after deliberately damaging
the price panel:

| Test-panel price input | Gain over zero-price model |
|---|---:|
| Correct product and correct week | +0.03895 |
| Category/week average only | -0.00354 |
| Price paths assigned to wrong products, mean of 19 permutations | -0.01867 |
| Correct product prices assigned to wrong weeks, mean of 19 permutations | -0.00996 |

The actual alignment beat every product and time permutation in both validation and test.
With 19 permutations the conservative one-sided randomization bound is 0.05. Thus the gain
is not explained by a generic category/week price level, a calendar trend, or the marginal
distribution of prices alone. The correct product-price and week-price alignment matters.

## Meaning for a retailer

For a supported category, a 10% price increase changes that product's conditional log-odds
by (-\tau\log(1.10)). With the fitted coefficients, its conditional odds multiplier is
approximately 0.87 to 0.89. This statement is exact for the model's probability of adding
that product conditional on the rest of the basket. The full joint model can also compute
how the probability of every complete basket changes, including whether other products
rise as substitutes or fall as complements.

This now supports realistic *observational price-scenario* queries such as:

- holding a shopper, store, week, assortment, and current cart fixed, increase one
  supported product's price and compare its inclusion probability;
- compare the probability of retaining versus dropping that product from a planned
  basket;
- rank likely substitute and complementary basket changes under the new price;
- aggregate those probability changes over historical shopping contexts for demand
  planning.

It is not yet appropriate to label those changes causal profit lift. A retailer should
first validate proposed actions with randomized price or promotion trials, including
availability and margin data.

## Reproducible artifacts

- Fit report: `artifacts/erim_basket_run/joint_price_utility/report.json`
- Coefficients: `artifacts/erim_basket_run/joint_price_utility/joint_price_coefficients.json`
- Per-basket scores: `artifacts/erim_basket_run/joint_price_utility/joint_price_per_trip.npz`
- Fit log: `artifacts/erim_basket_run/joint_price_utility/joint_price_utility.log`
- Alignment audit: `artifacts/erim_basket_run/joint_price_utility/alignment_audit.json`
- Alignment log: `artifacts/erim_basket_run/joint_price_utility/alignment_audit.log`

The exact-normalizer implementation is independently tested against complete enumeration
on small catalogues. The repository regression suite passes 188 tests.

## Retail-only refit

These coefficients were estimated on a price panel that still contained panel-purchase
fallback prices. The retail-only rebuild, re-estimated coefficients and full refit are in
[ERIM_RETAIL_PRICE_REFIT.md](ERIM_RETAIL_PRICE_REFIT.md).

## Required next step before replacing the production checkpoint

The result justifies a fresh additive fit in which the certified price block is present
while the non-price parameters are re-optimized, followed by new interaction, calibration,
recommendation, and counterfactual audits. The current production candidate should not be
silently mutated: it was trained under the explicit contract that price response was zero.
The price-feature contract must also travel with the new checkpoint so every consumer uses
chain product/week prices and cannot re-enable outcome-selected store deviations.
