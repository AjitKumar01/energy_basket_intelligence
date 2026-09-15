# Type B and Type C coupon campaign EDA

## Decision

The Type B/C data are reliable enough to reconstruct a **binary coupon-offer exposure**:
which household received a campaign, when it was active, and which products each coupon
could be used on. They are not reliable enough to reconstruct the monetary net price for
the catalogue, and they do not identify a causal coupon effect.

No basket-model use is authorized by this EDA. A later, separate experiment may evaluate
binary offer availability as an observational predictor, provided it splits by whole
campaign and does not call the coefficient a price elasticity. Realized paid price must
not be used as a product-choice regressor.

## Data used

The audit joins:

- `campaign_table.csv`: household/campaign receipt;
- `campaign_desc.csv`: campaign start and end day;
- `coupon.csv`: campaign/coupon/product applicability;
- `coupon_redempt.csv`: redemption records;
- cleaned transaction lines in `data/tx.parquet`; and
- the 1,920-household, 5,455-product basket-model cohort.

The Dunnhumby guide states that Type B and Type C participants receive every coupon in
their campaign. This makes offer availability reconstructable for these types. Type A is
excluded because the database supplies only a pool and does not identify the 16 coupons
actually sent to each household.

## 1. Assignment and mapping integrity

There are 25 Type B/C campaigns: 19 Type B and 6 Type C. They contain 3,229
household-campaign assignments covering 1,076 distinct households. Assignment integrity is
strong:

- zero duplicate household/campaign assignments;
- zero disagreements between assignment and campaign-description type;
- all 527 Type B/C redemption rows belong to a recorded recipient;
- all 527 occur inside the recorded campaign window; and
- all 527 refer to a coupon present in that campaign's coupon map.

The coupon-product map needs deterministic cleaning. Its Type B/C subset contains 21,655
rows, of which 3,406 are exact duplicate campaign/coupon/product rows. After removing
those duplicates there are 18,249 mappings. Fifty-four coupon codes occur in more than one
campaign, so `COUPON_UPC` alone is not a safe identifier. Every lookup must use
`(CAMPAIGN, COUPON_UPC)`.

After cleaning, Type B/C contains 410 campaign/coupon keys representing 330 coupon codes
and 11,272 eligible raw products.

## 2. Relevance to the fitted basket catalogue

Only 1,008 of the 5,455 modeled products appear in a Type B/C coupon mapping: 18.5% of the
catalogue. The campaigns cover 1,036 of the 1,920 modeled households.

The complete Type B/C data imply 53,617 household/campaign/coupon offer instances. Only
27,683 of these involve a coupon that applies to at least one modeled product.

| Campaign type | Campaigns | Recipient assignments | Coupon keys | Eligible raw products | Modeled products | Model-cohort recipient baskets during campaigns | Baskets containing eligible modeled product |
|---|---:|---:|---:|---:|---:|---:|---:|
| Type B | 19 | 2,655 | 296 | 10,356 | 945 | 23,383 | 2,135 (9.13%) |
| Type C | 6 | 574 | 114 | 1,064 | 80 | 9,475 | 416 (4.39%) |

Campaign 16 has 43 valid redemptions but none of its eligible products is in the current
5,455-product catalogue. It cannot contribute to the present product model.

## 3. Redemption linkage reliability

Redemption integrity at the campaign level is excellent, but the redemption file has no
`PRODUCT_ID`. Product linkage therefore has to be inferred by intersecting the coupon's
eligible products with that household's purchases on the redemption day.

| Linkage check | Redemptions |
|---|---:|
| Valid Type B/C redemption rows | 527 |
| At least one applicable product purchased that day | 422 |
| At least one applicable transaction line with `COUPON_DISC < 0` | 321 |
| Exactly one discounted applicable line | 299 |
| Exactly one coupon redemption that day and exactly one discounted applicable line | 107 |
| No applicable product found in cleaned transactions | 105 |
| Applicable product found, but no discounted applicable line | 101 |

Only 107 redemptions have the conservative single-redemption/single-line linkage needed
for a relatively unambiguous observed discount. Eighty of those are relevant to the
modeled households and coupon keys. This is too little support for catalogue-wide coupon
amount inference.

The valid redemption count is only 0.983% of the 53,617 offered coupon instances. That is
not a data error: non-redemption is an important outcome. It does mean the monetary value
cannot be recovered for most offered coupons from redeemed transactions.

## 4. Coupon amount reliability

`coupon.csv` records applicability but contains no face value, minimum quantity, or other
offer terms. Transaction `COUPON_DISC` records the realized discount after purchase. It is
not a pre-purchase coupon catalogue.

Among the 410 campaign/coupon keys:

- 77 have any conservative single-line observed discount;
- 7 have at least three such observations; and
- only 4 have at least three observations with at least 90% at the same cent amount.

Only 2 of those 4 keys apply to the modeled catalogue. They cover 378 of 27,683
model-relevant offered coupon instances, or 1.37%. Even these are labelled *stable observed
amounts*, not verified face values: quantity restrictions and other discount rules are
not recorded.

Therefore an effective net price cannot be safely constructed across Type B/C. Reading
`paid_price` from purchased lines would condition the explanatory variable on purchase
and redemption, recreating the outcome-selection error found in the price audit.

## 5. Campaign overlap and evaluation leakage

The 3,229 assignments cover 115,822 unique recipient-days. On 18,496 of those days
(16.0%), a household has more than one active Type B/C campaign; the maximum is four.
Any later predictor must represent all simultaneously active campaigns rather than assign
the basket to one arbitrary campaign.

Seven campaigns cross the basket model's existing train/validation/test boundaries:

- campaign 14 and 16 cross training and validation;
- campaign 19, 20, 21 and 22 cross validation and test; and
- campaign 15 crosses all three periods.

A normal basket-time split would therefore place the same coupon offer on both sides of
an evaluation boundary. Coupon evaluation must hold out whole campaigns. Campaign 24 also
ends after the transaction observation window, and campaigns 15 and 24 are incomplete in
the model-aligned window.

## 6. Evidence of selection and temporal instability

Before each campaign, recipient households already purchase eligible modeled products at
different rates from nonrecipients:

- median pre-period standardized difference: 0.325;
- 18 of 25 campaigns have absolute imbalance of at least 0.10; and
- 15 of 25 have absolute imbalance of at least 0.25.

There is no documented randomized assignment mechanism. This imbalance means a simple
recipient-versus-nonrecipient comparison would largely measure customer selection.

For the 23 campaigns with a complete model-aligned campaign window, a purely descriptive
difference-in-differences check is evenly divided: 11 positive, 11 negative and one zero.
The median is zero, with a range from -0.0374 to 0.0255 in eligible-product basket
incidence. These values do not establish a stable causal response; they are diagnostics,
not fitted effects.

## 7. Reliability assessment

| Information needed | Reliability | Decision |
|---|---|---|
| Type B/C campaign recipient | High | Usable |
| Campaign validity window | High | Usable, except observation-window truncation |
| Coupon/product applicability | High after exact deduplication | Usable with `(campaign, coupon)` identity |
| Non-redemption | High for Type B/C offers | Retain as an outcome |
| Product attached to a redemption | Partial | Only 107 conservative links |
| Coupon monetary value | Very low | Do not construct catalogue-wide net price |
| Realized paid price as choice input | Invalid | Do not use |
| Causal campaign effect | Not identified | Do not claim |
| Held-out binary offer prediction | Feasible research question | Requires whole-campaign split and explicit baselines |

## Reproducibility

- EDA implementation: `scripts/version4/eda_typebc_coupon_campaigns.py`
- Focused tests: `tests/test_coupon_campaign_eda.py`
- Machine report: `artifacts/typebc_coupon_eda_20260915/report.json`
- Execution log: `artifacts/typebc_coupon_eda_20260915/run.log`

The JSON records SHA-256 hashes for the implementation and every source input. Generated
artifacts remain local and are not model checkpoints.
