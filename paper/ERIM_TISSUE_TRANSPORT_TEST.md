# Tissue model test 4: cross-arm transport

Date: 2026-09-17

Follows [ERIM_TISSUE_MODEL_TRIAL_REPEAT_TEST.md](ERIM_TISSUE_MODEL_TRIAL_REPEAT_TEST.md).

## Question

Tests 2 and 3 predicted normal purchasing well but under-predicted campaign redemptions
about four-fold. They learned coupon reach and life from the one pre-campaign coupon. If
coupon *delivery* is the only missing piece, borrowing it from one experimental arm should
let the model predict the other arm's lift.

## Design

The protocol is fixed in the docstring of `scripts/run_erim_tissue_transport_test.py`. Not
blind.

- **Purchase model.** The test-2 model, fitted only on pre-campaign weeks: availability,
  awareness trend, direct redemption, coupon response κ·value/price, shelf-price τ and
  household tastes.
- **Delivery.** Arm-specific coupon codes get their own notice probability q_c and weekly
  retention r_c. They are calibrated by Poisson deviance on **one arm's redemptions** by
  code and 4-week block, using no purchase outcomes. The market-wide coupons keep their
  pre-campaign values.
- **Transport.** Calibrate on A, predict B − control. Calibrate on B, predict A − control.
  The control arm is never used for calibration.
- **Criteria.**
  - T1 / T2: held-out arm Cottonelle lift positive and inside the 95% interval.
  - T3: its redemption effect inside the interval.
  - T4: its tissue effect inside the interval.
  - T5: control levels within ±20% for Cottonelle and ±25% for redemptions.
- **Expectation, stated before running.** If delivery is the missing piece, T1–T3 pass;
  if not, the purchase response to coupons is also wrong.

## Results

**Calibrated delivery.** Both arms give the same answer: q_c = 1.00 (at the upper bound)
and r_c = 0.92 (A) or 0.93 (B). Refit ranges were 0.91–0.93 and 0.93–0.94. Campaign coupons
lasted far longer than the pre-campaign coupon (r = 0.65).

| | Calibrated on A → predict B | Calibrated on B → predict A |
|---|---|---|
| Cottonelle purchase-weeks: observed | +0.34 (0.08 to 0.61) | +0.34 (0.08 to 0.60) |
| Cottonelle purchase-weeks: predicted (90%) | **+0.07** (0.06–0.08) ✗ | **+0.34** (0.33–0.38) ✓ |
| Redemptions: observed | +0.14 (0.04 to 0.23) | +0.52 (0.42 to 0.62) |
| Redemptions: predicted | +0.07 ✓ | +0.42 ✗ (just below) |
| Tissue purchase-weeks: observed / predicted | +0.05 / +0.05 ✓ | −0.16 / +0.23 ✓ |
| Control Cottonelle level (2.16 observed) | 2.63 (+22%) ✗ | 2.65 (+23%) ✗ |
| Control redemptions (0.73 observed) | 0.46 (−38%) ✗ | 0.49 (−34%) ✗ |

**Verdict: fail.**
- T2 passes: A's lift is predicted almost exactly.
- T1 fails: B's lift is under-predicted five-fold.
- T3 fails for the B → A direction; T5 fails in both directions.
- Even with every coupon noticed, the model under-predicts redemptions: control 0.47
  against 0.73, and the calibration arm A 0.86 against 1.26.

## Why B cannot be predicted

This analysis is exploratory, done after the verdict. It splits the lift by how Cottonelle
was bought, in units per assigned household during the test.

| Purchase type | A − control | B − control |
|---|---|---|
| With a test coupon | **+0.50** | +0.09 |
| With an ordinary manufacturer coupon | +0.02 | **+0.17** |
| Store coupon | 0.00 | 0.00 |
| No coupon | −0.11 | **+0.20** |

- **A's lift is redemption-driven.** Extra test-coupon purchases, partly offset by fewer
  full-price ones. This is exactly the mechanism the model encodes, and calibrated on B it
  predicts A correctly.
- **B's lift is mostly *not* redemption-driven.**
  - B households bought more Cottonelle at full price and with ordinary newspaper-type
    coupons (median value $0.20, the same as other arms).
  - The full-price increase is concentrated in 1985 weeks 33–45, right after B's $1.00 drop
    (+0.16, then +0.04, then about 0).
  - It appears in the mostly unmetered cells 13–15 as strongly as elsewhere, so it is not TV
    cut-in advertising.
  - Shelf prices paid did not differ between arms.
- **Something about B's first mailing raised Cottonelle purchasing beyond redemptions.**
  Possible explanations include:
  - the coupon acting as a reminder or advertisement;
  - a different mailing format;
  - an unrecorded sample.

  The archive does not say which. A model in which coupons work only through redemption,
  as tests 2–4 assume, cannot produce this. Test 1's persistent held-coupon boost could, but
  it applied that boost to every arm and got A and B the wrong way round.
- **Redemption does not scale with face value.**
  - The $0.70 coupons were redeemed about twice as often as predicted: control 019 0.44
    observed vs 0.25 predicted.
  - The $1.00 coupons were close: A 001 0.35 vs 0.30.

  The assumption that the coupon's pull is proportional to its value is too steep.

## Conclusion after four tests

| Test | What changed | Levels | A lift | B lift | Verdict |
|---|---|---|---|---|---|
| 1 | Blind basket model | −45% | ✓ (compensating errors) | ✓ (compensating errors) | pass |
| 2 | Launch availability, direct redemption, value response | ✓ | ✓ barely | ✗ | fail |
| 3 | + trial and repeat | ✓ | ✓ barely | ✗ | fail |
| 4 | + delivery borrowed from the other arm | ✗ (coupon-driven over-prediction) | ✓ | ✗ | fail |

What is established:

1. **Ordinary purchasing is predictable** once product availability is modelled: tests
   2–3 were within 10%.
2. **Redemption-mediated coupon lift transports across arms.** Once delivery is known, the
   pre-campaign purchase response predicts arm A's lift: +0.34 against +0.34.
3. **Pre-campaign data cannot predict a campaign's effect by themselves.** Delivery
   differed (retention 0.65 before the campaign against 0.92–0.93 during it), redemption
   propensity was flatter in face value than assumed, and one arm's lift came mainly
   through non-redemption purchasing that no coupon-as-price-cut model represents.

For the retailer this means the basket model can answer *"what if a coupon is redeemed?"*
once a trial has measured reach and redemption. It should not be trusted to answer *"what
if we run this promotion?"* before any trial: part of a promotion's effect comes from
awareness, which is not a price effect at all.

## Reproduce

```bash
python -u scripts/run_erim_tissue_transport_test.py     # about 30 minutes
```

Outputs are in `artifacts/erim_tissue_transport_test/`:
- `prediction.json`: calibration and predictions, written before the arm contrasts are
  computed;
- `report.json`;
- `run.log`.

Tests are in `tests/test_erim_tissue_transport_test.py`. They check that delivery scales
redemptions, that by-code redemptions sum to the campaign total, that zero notice gives no
redemptions, and that the Poisson deviance behaves correctly.
