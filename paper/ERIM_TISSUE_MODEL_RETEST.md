# Tissue model re-test with three fixes

Date: 2026-09-17

Follows [ERIM_TISSUE_MODEL_PREDICTION_TEST.md](ERIM_TISSUE_MODEL_PREDICTION_TEST.md).

## Disclosure

The first test was blind. This re-test was designed after its results and the experiment's
outcomes were known, so it is not blind. To limit that:
- every fix is motivated by a mechanism found in pre-campaign or store data;
- no parameter is fitted on test-period panel outcomes, and no level anchoring is used;
- the pass criteria are stricter and include every quantity the first model got wrong;
- the protocol is fixed in the docstring of `scripts/run_erim_tissue_prediction_test_v2.py`.

The model was checked for bugs only on training-window data before the comparison ran.

## A discovery that changed fix 1

Cottonelle was **not sold in Springfield at all**. In Sioux Falls it **launched during the
training window**:
- its first store sale was in 1985 week 21, reaching 16 of 17 chain stores by week 30;
- panel purchase rates were zero before week 20;
- the first $1.00 coupon (week 29) lifted its share of store tissue units from 8% to 55%.

The coupon lottery was launch support for a new brand. The first model treated pre-launch
weeks as weeks in which shoppers refused Cottonelle. That, more than a missing trend,
explains its too-low baseline.

## The three fixes

1. **Availability and awareness.** Cottonelle's utility for household h in week t gains
   log(a_ht) + θ·log(1 + weeks since launch). Here a_ht is the household's pre-campaign
   share of purchase-days, across all eight ERIM categories, at stores already selling
   Cottonelle.
2. **Direct redemption.** Holding a coupon adds a separate "Cottonelle with coupon"
   alternative. Using it redeems and consumes the coupon, so a coupon raises purchasing
   only when it is used.
3. **Separate coupon response.** The alternative has weight κ·(face value ÷ shelf price),
   independent of shelf-price τ. A worthless coupon has no effect.

## Fitted parameters

| Parameter | First test | Re-test (90% refit range) |
|---|---|---|
| Shelf-price τ | 1.20 | 2.45 (2.34–2.59) |
| Awareness trend θ | — | −0.08 (−0.15 to −0.03) |
| Coupon noticed q | 1.00 | 1.00 |
| Weekly coupon retention r | 0.90 | **0.65** (0.60–0.70) |
| Coupon response κ | — | 1.22 (1.13–1.34); a $1.00 coupon is used on about half of Cottonelle purchases |

In-sample, the re-test model reproduces the $1.00 coupon's weekly redemptions (observed
233/177/77/19, predicted 249/155/65/32). It does not capture the post-launch fade in
Cottonelle purchases in weeks 24–28.

## Results

Effects are per assigned household over 45 test weeks; observed intervals are
randomization 95% intervals.

| Criterion | Quantity | Observed | Re-test prediction (90%) | First test | Pass |
|---|---|---|---|---|---|
| C1 | Cottonelle purchase-weeks, A | +0.34 (0.08 to 0.60) | +0.13 (0.12–0.15) | +0.41 | ✓ (barely) |
| C1 | Cottonelle purchase-weeks, B | +0.34 (0.08 to 0.61) | +0.02 (0.02–0.02) | +0.25 | **✗** |
| C2 | Tissue purchase-weeks, A / B | −0.16 / +0.05 (±0.65) | +0.09 / +0.01 | +0.28 / +0.17 | ✓ |
| C3 | Arm coupon redemptions, A | +0.52 (0.42 to 0.62) | +0.18 | +0.22 | **✗** |
| C3 | Arm coupon redemptions, B | +0.14 (0.04 to 0.23) | +0.02 | +0.08 | **✗** |
| C4 | A − B, Cottonelle purchase-weeks | +0.00 (−0.27 to 0.27) | +0.11 | +0.16 | ✓ |
| C5 | Control Cottonelle purchase-weeks | 2.16 | 2.38 (+10%) | 1.19 (−45%) | ✓ |
| C5 | Control coupon redemptions | 0.73 | 0.17 (−77%) | 0.16 | **✗** |

Also: control tissue purchase-weeks were 10.06 observed vs 10.33 predicted (+3%). Other-brand
purchase-weeks were 8.16 vs 8.27 (+1%). Without the awareness trend, predictions barely
change: +0.14 and +0.02.

**Verdict: fail**, on C1 for B, C3 for both arms and the redemption part of C5.

### First vs second half of the test (never examined before)

| | Observed, first half | Predicted | Observed, second half | Predicted |
|---|---|---|---|---|
| A: Cottonelle purchase-weeks | +0.20 (0.03 to 0.37) | +0.08 | +0.13 (0.01 to 0.26) | +0.05 |
| A: redemptions | +0.29 | +0.10 | +0.23 | +0.08 |
| B: Cottonelle purchase-weeks | **+0.28 (0.12 to 0.45)** | +0.02 | +0.06 (−0.07 to 0.18) | +0.00 |
| B: redemptions | +0.12 | +0.02 | +0.02 | +0.00 |

## What the re-test shows

1. **Fix 1 worked.** Once the model knows when and where Cottonelle was on the shelf, it
   predicts control-group purchase levels within 10% with no anchoring. The first model
   was off by 45%.
2. **Fixes 2 and 3 made the coupon predictions honest, and they fail.**
   - With redemption tied to use, the pre-campaign coupon implies a short coupon life
     (about 3 weeks) and modest reach.
   - The campaign coupons were used far more: the control group redeemed 0.73 coupons per
     household against 0.17 predicted. They were also used over months, not weeks.
   - So the model under-predicts the campaign's redemptions 3–6× and its brand lift
     2.5–17×.
3. **The first test's pass was mostly compensating errors.** Its persistent unredeemed
   coupon boost (r = 0.90, π = 0.33) inflated effects, and its missing launch timing
   deflated the baseline.
4. **Two things pre-campaign data cannot tell the model.**
   - *Coupon delivery.* The only pre-campaign coupon behaved like a short-lived
     newspaper insert. The campaign coupons, targeted by cell, had much higher reach and
     longer life, likely because of a different delivery channel that is not documented
     in the archive.
   - *Trial and repeat for a new brand.* Group B's lift arrived in the first half:
     +0.28 purchase-weeks from only +0.12 extra redemptions, about 2.4 extra purchase-weeks
     per extra redemption. That pattern fits an early $1.00 coupon producing trial that
     turned into repeat buying. The basket model has no purchase-history dependence, so it
     cannot produce this. This is a hypothesis from the data, not a tested result.

## Answer

After the fixes, the model predicts *normal* purchasing well: brand and category levels
within 10%. It does **not** predict the causal effect of a new coupon campaign from
pre-campaign data. The obstacles are coupon reach and life that differ by delivery
channel, and trial and repeat dynamics for a newly launched brand.

This is a stronger and more trustworthy conclusion than the first test's pass. The basket
model's what-if answers for promotions should not be used without calibration from a
randomized test.

Directions that could be tested next, each under a new fixed protocol:
- **Trial and repeat.** Let a household's first Cottonelle purchase raise its later
  Cottonelle utility, estimated from post-launch pre-campaign weeks.
- **Cross-arm transport.** Calibrate coupon reach and life on one arm's redemptions, then
  predict the other arm's lift. This tests the purchase response separately from the
  unknowable delivery.

## Reproduce

```bash
python -u scripts/run_erim_tissue_prediction_test_v2.py     # about 10 minutes
```

Outputs are in `artifacts/erim_tissue_prediction_test_v2/`:
- `prediction.json`: predictions, frozen before the assignment file is read;
- `report.json`;
- `run.log`.

Tests are in `tests/test_erim_tissue_prediction_test_v2.py`. They cover availability
mixing, weeks since launch, the coupon alternative against explicit enumeration, and
zero-value coupons having no effect.
