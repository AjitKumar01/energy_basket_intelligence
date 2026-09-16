# Can a pre-campaign basket model predict a randomized coupon test?

Date: 2026-09-17

## Question

[ERIM_RANDOMIZED_COUPON_EXPERIMENT.md](ERIM_RANDOMIZED_COUPON_EXPERIMENT.md) measured the
causal effect of randomized Cottonelle coupon schedules. This test asks whether the
energy basket model, fitted only on data from before the campaign and never shown the cell
labels, predicts those causal effects. It is a real out-of-sample test of the model's
what-if answers: a model with no price response predicts zero effect and fails.

## Protocol

The full protocol is fixed in the docstring of `scripts/run_erim_tissue_prediction_test.py`.

- **Model.** The additive energy basket model for one category: 11 tissue brands, size
  potentials, household and household-brand taste, and price utility −τ·(log price
  deviation) from the retail file.
  - The empty basket is included, so the model also predicts whether a household buys
    tissue at all that week.
  - A held coupon of value v is a Cottonelle price cut, entering through the same τ.
  - A dropped coupon is noticed with probability q, kept each week with probability r,
    and redeemed with probability π when Cottonelle is bought while held. The holding
    state is latent and integrated out exactly.
- **Training data.** 4,158 Sioux Falls households, weeks 1985-14 to 1985-32, before any
  arm-specific coupon. The one coupon every household received ($1.00, week 29) is what
  identifies q, r and π. Penalty strength was chosen on held-out weeks: 1.0, the smallest
  value in the grid.
- **Prediction.** Every experimental household is simulated under all three schedules
  over 1985-33 to 1986-25, using its observed shopping weeks and shelf prices. The
  predicted effect of a schedule is the average difference from control. Predictions were
  written to `prediction.json` before the assignment file was read.
- **Criteria.**
  - C1: the predicted effect on Cottonelle purchase-weeks is positive and inside the
    observed 95% interval, for both arms A and B.
  - C2: the predicted effect on tissue purchase-weeks is inside the observed interval.
- **Uncertainty.** Prediction intervals are 90% intervals from 20 household-weighted
  refits. They reflect parameter uncertainty only, not model misspecification.
- **Amendment.** A pooled check that used no arm labels, made before any arm comparison
  was seen, showed that Cottonelle was growing: 0.043 → 0.072 purchase-weeks per shopping
  week. The model has no trend. A secondary **level-anchored** prediction therefore adds
  Cottonelle and category shifts in each 4-week block, chosen to match the pooled outcomes
  of all experimental households. The primary verdict remains the pre-registered one.

The model is the basket model's structure refitted on 1985 tissue data. It is not the
1986–87 eight-category checkpoint, whose window starts after the campaign ended. That
checkpoint's tissue sensitivity (τ = 1.49) is used as a sensitivity run.

## Fitted parameters

| Parameter | Estimate (90% refit range) |
|---|---|
| Price sensitivity τ | 1.20 (1.20–1.26); the 1986–87 eight-category model found 1.49 |
| Coupon noticed, q | 1.00, at the boundary |
| Weekly retention, r | 0.90 (0.86–0.92) |
| Redemption when buying Cottonelle while holding, π | 0.33 (0.31–0.35) |

## Results

Effects are per assigned household over the 45 test weeks; observed intervals are
randomization 95% intervals.

| Outcome | Arm | Observed | Primary prediction | Level-anchored | τ = 1.49 |
|---|---|---|---|---|---|
| Cottonelle purchase-weeks | A | **+0.34** (0.08 to 0.60), p = 0.011 | **+0.41** (0.36–0.48) ✓ | +0.61 (0.56–0.68) ✗ | +0.32 |
| | B | **+0.34** (0.08 to 0.61), p = 0.012 | **+0.25** (0.21–0.32) ✓ | +0.33 (0.28–0.42) ✓ | +0.19 |
| Tissue purchase-weeks | A | −0.16 (−0.82 to 0.48) | +0.28 ✓ | +0.43 ✓ | +0.22 |
| | B | +0.05 (−0.63 to 0.69) | +0.17 ✓ | +0.24 ✓ | +0.13 |
| Other-brand purchase-weeks | A | −0.44 (−1.00 to 0.14) | −0.08 | −0.11 | −0.07 |
| | B | −0.20 (−0.78 to 0.38) | −0.05 | −0.06 | −0.04 |
| Arm-specific coupon redemptions | A | +0.52 (0.42 to 0.62) | +0.22 ✗ | +0.33 ✗ | +0.16 |
| | B | +0.14 (0.04 to 0.23) | +0.08 ✓ | +0.09 ✓ | +0.06 |
| A − B, Cottonelle purchase-weeks | | +0.00 (−0.25 to 0.27) | +0.16 ✓ | +0.27 (just outside) | +0.13 |

Control-arm levels:

| Outcome | Observed | Primary model | Level-anchored |
|---|---:|---:|---:|
| Cottonelle purchase-weeks | 2.16 | 1.19 | 2.07 |
| Tissue purchase-weeks | 10.06 | 9.49 | 9.81 |
| Test-coupon redemptions | 0.73 | 0.16 | 0.29 |

**Pre-registered verdict: C1 and C2 pass for both arms.** A no-effect model fails C1,
because zero lies below both observed intervals.

## What this does and does not show

**Supported.**

1. A model fitted only on pre-campaign weeks, using only shelf-price variation and one
   common coupon, predicted the causal brand lift of randomized coupon schedules within
   the experiment's intervals. It predicted +0.41 and +0.25 purchase-weeks; the
   experiment measured +0.34 and +0.34.
2. The price response is roughly right in size. τ fitted on 1985 prices (1.20) and τ from
   the separate 1986–87 model (1.49) give similar predictions, and both pass C1.

**Not supported. These are the reasons not to over-read the pass.**

1. **The test has low resolution.** The observed intervals are wide (about ±0.26), so any
   prediction between roughly +0.08 and +0.60 passes. C2 is weaker still: the category
   intervals are about ±0.65 wide, so C2 would pass for almost any plausible prediction.
2. **The model predicts the wrong mechanism.**
   - It predicts category growth (+0.28 and +0.17 tissue purchase-weeks) and little
     brand switching (−0.08 and −0.05). The experiment points the other way: no category
     change and larger, though not significant, losses for other brands.
   - The model under-predicts redemptions by 2.4× in arm A and 2.5×–4.6× at control
     levels. So it predicts about the right extra Cottonelle purchasing from far too few
     redemptions.
   - The low π = 0.33 lets a held but unredeemed coupon keep raising Cottonelle
     purchasing for weeks. That is probably a compensating error, not real behaviour.
3. **The model misranks the schedules.** It predicts that more coupons (A) beat
   substituted coupons (B) by 0.16–0.27 purchase-weeks. The experiment found them equal
   (difference +0.00). The primary prediction is still inside the interval; the
   level-anchored one is just outside.
4. **Levels are wrong without anchoring.** Cottonelle was growing and the model has no
   trend: it predicted 1.19 control-arm purchase-weeks against 2.16 observed. When levels
   are anchored to the truth, the arm A prediction overshoots (+0.61, outside the
   interval), because predicted effects scale with the base rate. The primary pass
   partly comes from an under-predicted base rate offsetting an over-strong coupon
   response.

## Answer to the counterfactual question

For "what if these households get this coupon schedule?", the model's answer on brand
volume was in the right range, a genuine out-of-sample causal check it could have
failed. It is not reliable for questions that need the mechanism:
- whether the category grows;
- how much comes from competitors;
- how many coupons are redeemed;
- which of two schedules is better.

Those answers need a randomized test, or a model calibrated on one.

Concrete improvements this test points to:
- a time trend or brand-growth term;
- modelling redemption directly instead of a latent coupon holding that persists;
- a model that separates coupons from shelf prices, with its own response instead of
  sharing τ.

Each change should be re-tested against this experiment under a new, pre-registered
protocol.

## Reproduce

```bash
python -u scripts/run_erim_tissue_prediction_test.py        # about 17 minutes
```

Outputs are in `artifacts/erim_tissue_prediction_test/`:
- `prediction.json`: predictions frozen before the assignment is read;
- `report.json`: comparison and verdicts;
- `run.log`.

`--predict-only` stops before the assignment file is read. Exactness tests are in
`tests/test_erim_tissue_prediction_test.py`; the basket probabilities, marginals and
forward recursion are checked against brute-force enumeration.

## Re-test

A re-test with launch availability, direct redemption and a separate coupon response is in
[ERIM_TISSUE_MODEL_RETEST.md](ERIM_TISSUE_MODEL_RETEST.md). It fixes the level error but
fails the coupon criteria. It shows the pass above was mostly compensating errors.
