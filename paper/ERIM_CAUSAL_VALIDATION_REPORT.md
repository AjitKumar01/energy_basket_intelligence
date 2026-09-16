# Can the basket model's price what-ifs be trusted? Evidence from a randomized ERIM experiment

Date: 2026-09-17

This report consolidates five linked analyses: the experiment and four model tests. Details, tables and reproduction
commands are in the linked documents.

## Summary

- **The question.** The energy basket model answers what-if questions such as "what happens
  to baskets if this product is 20% cheaper?". Its price effects were learned from
  historical prices set by retailers, so they are observational. Only a lottery that
  assigns prices or discounts can confirm them.
- **The evidence.** The ERIM archive contains one such lottery. In 1985–86 ERIM assigned
  2,975 Sioux Falls households at random to 15 cells that received different Cottonelle
  toilet-tissue coupon schedules. The groups were balanced before the campaign. The coupons
  **caused** about **14–15% more Cottonelle volume**, about **+0.34 purchase-weeks per
  household** in each of two treated arms. Total tissue buying did not measurably change,
  and nothing persisted after the campaign.
- **The model tests.** We asked a tissue-only version of the basket model, fitted only on
  pre-campaign data, to predict those effects. Four versions were tested:

| Test | Model | Blind? | Verdict | What it showed |
|---|---|---|---|---|
| 1 | Basket model as-is | Yes | Pass | The pass came from two errors cancelling |
| 2 | + product launch availability, redemption-based coupons, value-scaled response | No | Fail | Normal purchasing now predicted within 10%; coupon effects badly under-predicted |
| 3 | + trial-and-repeat purchasing | No | Fail | No habit effect in the data; nothing changed |
| 4 | + coupon delivery borrowed from the other arm | No | Fail | Arm A's lift predicted exactly; arm B's lift was not coupon-driven |

**Conclusion.** Once product availability is modelled, the basket model predicts
*ordinary* purchasing well. Once a trial has measured coupon reach, it predicts the
purchasing that *coupon use* causes. It cannot predict a promotion's full causal effect from
historical data alone. Coupon delivery varies by channel, redemption does not scale with
face value, and part of a promotion's effect is awareness rather than price. Promotion
what-ifs from the model are therefore **screening tools that need calibration from a
randomized trial**, not forecasts of causal lift.

## 1. What causal data ERIM contains

The official ERIM documentation lists eleven data files. Comparing it with the Kilts
category archives shows the following.

| Data | Available | Causal use |
|---|---|---|
| Store features, displays, special prices, shelf prices (store causal file, retail file) | Yes, all categories | Retailer-chosen, so observational |
| Household cable and TV-meter flags | Yes | Descriptive |
| **Household test-cell assignment** | Yes; **only tissue** has more than one cell | **Randomized** |
| TV telemeter viewing and commercial-exposure files (1987–88) | **Not in the archives** | — |

The tissue cells are the only randomized treatment in the data.

## 2. The randomized coupon experiment

Source: [ERIM_RANDOMIZED_COUPON_EXPERIMENT.md](ERIM_RANDOMIZED_COUPON_EXPERIMENT.md)

**Design.** The 15 cells form three arms repeated in three size strata. The arms were
inferred from the test-coupon codes each arm redeemed. The full treatment package per cell
is undocumented.

| Arm | Coupons |
|---|---|
| Control | $0.70 (1985 wk 37), $0.70 (wk 48) |
| A | Control's coupons + $1.00 (wk 33) + $0.70 (1986 wk 5) |
| B | $1.00 (wk 33) and $0.75 (wk 49), mostly instead of control's |

**Method.** Intention-to-treat on all assigned households, with stratum-weighted
differences and p-values from 4,999 within-stratum re-randomizations.

**Results over the campaign, per household:**

| Outcome | Control | A − control | B − control |
|---|---|---|---|
| Cottonelle units | 3.02 | +0.41 (+13.6%), p = 0.045 | +0.46 (+15.2%), p = 0.034 |
| Cottonelle purchase-weeks | 2.16 | +0.34, p = 0.011 | +0.34, p = 0.012 |
| Cottonelle shelf spend | $3.60 | +$0.54, p = 0.03 | +$0.56, p = 0.03 |
| All tissue units | 15.05 | −0.22, p = 0.69 | +0.44, p = 0.45 |

- Pre-campaign balance held on all 42 checks.
- The post-campaign difference was about zero (p ≈ 0.7–0.9).
- The effect on total tissue buying is under-powered, so these data do not show whether the
  coupons grew the category or only moved shoppers between brands.

## 3. Can the model predict it?

All tests use the energy basket model restricted to tissue: 11 brands, size potentials,
household and brand tastes, and price utility −τ·log-price deviation. The empty basket is
included so the model also predicts whether a household buys tissue at all. The model is
fitted on Sioux Falls weeks 1985-14 to 1985-32, before any arm-specific coupon, and every
experimental household is simulated under every schedule.

### Test 1: blind prediction

Source: [ERIM_TISSUE_MODEL_PREDICTION_TEST.md](ERIM_TISSUE_MODEL_PREDICTION_TEST.md)

- **Setup.** A coupon is modelled as a price cut for households holding it. Predictions
  were written to file before the cell file was read.
- **Result.**
  - The pre-registered criteria passed: A +0.41 and B +0.25, against +0.34 each.
  - The control level was 45% too low (1.19 against 2.16 purchase-weeks).
  - Redemptions were 2–5× too low.
  - The model ranked A above B (the two were equal) and predicted category growth.
  - With levels corrected, A's prediction overshot the interval.
- **Reading.** A too-low baseline offset a too-strong coupon effect.

### Discovery between tests 1 and 2

**Cottonelle was a new brand, launched in Sioux Falls during the training window.** It was
never sold in Springfield. Its first store sale was 1985 week 21, reaching 16 of 17 stores
by week 30. Test 1 treated pre-launch weeks as refusals. The coupon lottery was launch
support.

### Test 2: three fixes

Source: [ERIM_TISSUE_MODEL_RETEST.md](ERIM_TISSUE_MODEL_RETEST.md)

**Fixes.**
1. Household-level availability, from store launch dates and pre-campaign store use, plus
   an awareness trend.
2. A coupon becomes a separate "buy with coupon" option, so it matters only when used.
3. The coupon's pull is proportional to face value ÷ price, separate from shelf-price τ.

**Result.**
- Control Cottonelle level within 10% (+10%), tissue +3%.
- Arm A lift +0.13 (just inside the interval); arm B lift +0.02 (fail).
- Redemptions under-predicted 3–6× (control 0.17 against 0.73).
- The pre-campaign coupon implied about three weeks of coupon life.

### Test 3: trial and repeat

Source: [ERIM_TISSUE_MODEL_TRIAL_REPEAT_TEST.md](ERIM_TISSUE_MODEL_TRIAL_REPEAT_TEST.md)

- **Change.** A household that has bought Cottonelle gets a utility shift λ.
- **Result.**
  - λ = −0.38: after buying, households bought *less*, consistent with stocking up.
  - Even the upper-bound version left arm B at +0.02.
  - The verdict was unchanged.

### Test 4: cross-arm transport

Source: [ERIM_TISSUE_TRANSPORT_TEST.md](ERIM_TISSUE_TRANSPORT_TEST.md)

- **Setup.** Coupon notice and retention were calibrated on one arm's redemptions only, and
  used to predict the other arm.
- **Result.**
  - Both arms imply the same delivery: every coupon noticed, retained 92–93% per week.
  - **Calibrated on B, the model predicts A's lift at +0.34, exactly as observed.**
  - Calibrated on A, it predicts B's lift at +0.07 against +0.34 observed.
  - Redemptions stay under-predicted even with full reach, which raises the control
    Cottonelle level to +22%.

**Why B fails (exploratory, after the verdict):**

| Extra Cottonelle units per household | A − control | B − control |
|---|---|---|
| Bought with a test coupon | +0.50 | +0.09 |
| Bought with an ordinary coupon | +0.02 | +0.17 |
| Bought without a coupon | −0.11 | +0.20 |

- **A's lift is redemption-driven**, which is the mechanism the model encodes.
- **B's lift is mostly not redemption-driven.**
  - B households bought more at full price and with ordinary coupons, mostly in the 13
    weeks after B's $1.00 mailing.
  - This was equally strong in unmetered households, so it is not TV advertising.
  - Shelf prices paid did not differ between arms.
- **Redemption was also flatter in face value than assumed.** $0.70 coupons were redeemed
  about twice as often as predicted; $1.00 coupons came close.

## 4. What we learned

**Established:**
1. **Product availability is essential.** A new product's sales cannot be modelled without
   when and where it was stocked. With it, ordinary purchasing was predicted within 10%.
2. **The price-response structure transports for redemption-driven effects.** When
   delivery is known, the pre-campaign coupon response predicted arm A's causal lift
   exactly.
3. **Promotion effects include non-price channels.** One arm's lift came through extra
   purchasing without its coupons, which no price-cut representation can produce.
4. **Pre-campaign data do not reveal promotion delivery.** Coupon life went from 65% to
   92–93% weekly retention, and reach from partial to full.
5. **A single passing test can be misleading.** The blind test passed through compensating
   errors; only the mechanism checks (levels, redemptions, arm ranking) exposed this.

**Not established:**
- Whether promotions grow the tissue category or only move shoppers between brands. The
  estimates are too imprecise.
- Why arm B's mailing raised non-coupon purchasing: awareness, format or an unrecorded
  sample.
- Anything about shelf-price changes, which were not randomized, or about other categories.

## 5. Implications

**For using the model**

| Question | Trust level |
|---|---|
| Basket composition, recommendations, ordinary purchase levels (with availability) | Supported |
| "What if customers redeem this coupon?", once a trial has measured reach and redemption | Supported by test 4 (arm A) |
| "What if we run this promotion?" before any trial | **Not supported**; use only to rank candidates for a trial |
| Category growth from promotions | Not identified |
| Profit or net-sales lift from pricing policies (the pricing decision process) | Not validated causally; keep as screening |

**For the pipeline**
- Add a product availability or assortment input (first store sale per product) wherever
  new products occur. Test 1's error shows its absence silently biases baselines.
- Keep promotion effects separate from price effects. A promotion should have its own
  calibrated response, not a price-equivalent, as the retailer guide already recommends.
- Use a randomized trial to calibrate promotion reach and redemption before the pricing
  decision process recommends promotions. The protocol is in
  [RETAILER_API_USER_MANUAL.md](RETAILER_API_USER_MANUAL.md).

## 6. Caveats

- **Only test 1 was blind.** Tests 2–4 were designed after the outcomes were known. Each
  disclosed this, fixed its criteria and model before running, fitted no parameter on
  test-period purchases, and stated its expected result in advance.
- **The arm definitions are reconstructed.** They come from redemption codes; the complete
  cell treatments are undocumented. Arm B still redeemed some control coupons.
- **The evidence is narrow.** One brand, one market, one 1985–86 campaign, 2,975
  households. Intervals are wide (about ±0.26 purchase-weeks), so modest prediction errors
  cannot be detected.
- **Prediction intervals understate uncertainty.** They reflect parameter uncertainty only,
  not model misspecification.

## Reproduce

| Analysis | Command | Runtime | Outputs |
|---|---|---|---|
| Experiment | `python -u scripts/run_erim_coupon_experiment.py --permutations 4999` | about 1 minute | `artifacts/erim_coupon_experiment/` |
| Test 1 | `python -u scripts/run_erim_tissue_prediction_test.py` | about 17 minutes | `artifacts/erim_tissue_prediction_test/` |
| Test 2 | `python -u scripts/run_erim_tissue_prediction_test_v2.py` | about 10 minutes | `artifacts/erim_tissue_prediction_test_v2/` |
| Test 3 | `python -u scripts/run_erim_tissue_prediction_test_v3.py` | about 12 minutes | `artifacts/erim_tissue_prediction_test_v3/` |
| Test 4 | `python -u scripts/run_erim_tissue_transport_test.py` | about 30 minutes | `artifacts/erim_tissue_transport_test/` |

Unit and exactness tests are in `tests/test_erim_coupon_experiment.py` and
`tests/test_erim_tissue_*.py`. The full suite passes: 192 tests.
