# ERIM randomized coupon experiment

Date: 2026-09-16

## Why this exists

Earlier ERIM price results are observational: shelf prices were set by the retailers, so
they can be confounded with seasons, advertising and stock. A causal price effect needs
prices, or discounts, assigned by lottery. A historical archive cannot be re-run, so the
only way to perform a randomized test on ERIM data is to find a lottery that ERIM itself
ran. This document records that search and the one experiment it found.

## What causal and media data ERIM contains

The official documentation (`data/erim_basket/documentation/erim_doc.pdf`) lists eleven
tape files. The Kilts category archives use their own numbering:

| Documented tape file | Archive file | In our archives | Used by the basket model |
|---|---|---|---|
| 1 Purchase history (display, feature ad, special price, coupon codes, **cell identifier**) | `f1` | all 8 categories | purchases only |
| 2 Shopping occasions (from 1986 wk 25) | `f2` | all | yes |
| 3 Yearly shopping-card usage | `f3` | all | no |
| — Household test-cell assignment (undocumented) | `f4` | all; only tissue has more than one cell | no |
| 4 Store causal (weekly display / ad / special price / shelf price) | `f5` | all | no |
| 5 UPC data | `f6` (`pbut_f7` for peanut butter) | all | yes |
| 6 Weekly shopping summary | `f7` | all | no |
| 7 Retail tracking (store-week sales with display/ad/special-price flags) | `f10` | all except frozen dinners | yes (prices) |
| 8 Household demographics (cable and **meter status** flags) | `f8` | all | no |
| 9 Household members | `f9` | all | no |
| 10 Working telemeters (8/87–9/88) | — | **absent** | — |
| 11 Commercial exposure (9/87–9/88) | — | **absent** | — |

So advertising is present as *store* features, displays and special prices, and households
are flagged as metered or on cable. The TV telemetry and commercial-exposure files are
not in the Kilts category archives. They would also fall outside our basket window and
cover other categories (yogurt, catsup, soup, detergent). Store promotions are chosen by
retailers, so they are observational too.

## The randomized experiment

The tissue archive's `f4` assigns 2,975 Sioux Falls households to 15 cells. Purchase
records carry the same cell identifier (zero mismatches) and a "special code used to track
test coupons". Test-coupon redemptions show a three-arm design repeated five times:

| Arm | Cells | Cottonelle test coupons redeemed |
|---|---|---|
| Control | 1, 4, 7, 10, 13 | market drops: $0.70 (1985 wk 37), $0.70 (wk 48) |
| A: extra coupons | 2, 5, 8, 11, 14 | control drops **plus** $1.00 (1985 wk 33) and $0.70 (1986 wk 5) |
| B: different schedule | 3, 6, 9, 12, 15 | $1.00 (1985 wk 33) and $0.75 (wk 49), mostly **instead of** the control drops |

Coupons of $1.00 (1985 wk 29) and $0.50 (1986 wk 6) were redeemed equally in every
arm. The cells fall into size strata of about 252, 78 and 331 households; the largest
stratum is mostly unmetered, so delivery was by print or mail, not TV. Each arm has 991–992
households. A cell's full treatment package is not documented, so the arm effect includes
anything else that differed by cell. The coupon codes are the observed difference.

**Method.** Intention-to-treat on every assigned household, with zero purchases for
households who bought nothing. Effects are stratum-weighted differences in means.
P-values come from 4,999 random re-assignments of households within strata. A more
conservative exact test treats the 15 cells as the randomized units (minimum attainable
p = 0.0625). Windows: pre 1985 wk 5–32, test 1985 wk 33–1986 wk 25, post 1986 wk
26–1987 wk 23. The post window is the basket model's training period.

## Results

**Balance before the test.** No pre-period outcome differs between arms: Cottonelle
units, buyers, spend, coupon use, all tissue units, shopping weeks, cable, meter,
income or household size. The smallest p-value is 0.06, for coupon value in B versus
control; 42 comparisons were made.

**During the test, per assigned household:**

| Outcome | Control mean | A − control | p | B − control | p |
|---|---:|---:|---:|---:|---:|
| Cottonelle units | 3.02 | **+0.41 (+13.6%)** | 0.045 | **+0.46 (+15.2%)** | 0.034 |
| Bought Cottonelle at all | 66.2% | +1.1 pt | 0.59 | **+4.6 pt** | 0.028 |
| Cottonelle shelf spend (retailer revenue) | $3.60 | **+$0.54** | 0.030 | **+$0.56** | 0.030 |
| Manufacturer coupon value redeemed | $0.73 | +$0.37 | <0.001 | +$0.32 | <0.001 |
| Cottonelle spend net of coupons | $2.87 | +$0.17 | 0.45 | +$0.24 | 0.27 |
| Other tissue brands, units | 12.04 | −0.63 | 0.19 | −0.02 | 0.96 |
| All tissue units | 15.05 | −0.22 | 0.69 | +0.44 | 0.45 |
| All tissue shelf spend | $16.12 | +$0.03 | 0.96 | +$0.57 | 0.36 |
| Shopping weeks (placebo) | 33.4 | −0.56 | 0.37 | −0.57 | 0.39 |

Cell-level exact tests point the same way. Cottonelle units rose in 4 of 5 A-cell pairs
(p = 0.125) and in 5 of 5 B-cell pairs (p = 0.0625, the smallest attainable).

**After the test**, effects disappear: Cottonelle units +0.04 (p = 0.86) and +0.07
(p = 0.73). Coupons moved purchases forward but left no lasting brand habit.

**Interpretation.**

1. Randomized coupons caused about 14–15% more Cottonelle volume. Test-inversion 95%
   intervals are roughly ×1.00 to ×1.27 (A) and ×1.02 to ×1.29 (B).
2. Arm B achieved the same lift with fewer coupons than arm A. The earlier $1.00 drop in
   place of a later $0.70 one was the more efficient schedule.
3. Category volume and category spend did not measurably change. The brand gain is
   consistent with switching from other tissue brands, but the category-level result is
   under-powered: the ±1.1-unit null band is larger than the brand effect. A retailer
   should not assume these coupons grew the category.
4. The retailer's Cottonelle revenue rose by about $0.55 per household, reimbursed by the
   manufacturer. The manufacturer's net revenue change was +$0.17–0.24 and is not
   significant.

## Comparison with the basket model

The retail-only basket model gives Cottonelle a price sensitivity of 1.49 on log price.
In the test, the average price paid per Cottonelle unit after manufacturer coupons was
$0.951 (control), $0.887 (A) and $0.894 (B). Treating those as price cuts of 7.0% and 6.2%,
the model implies an own-purchase odds multiplier of ×1.11 and ×1.10. The experiment
measured ×1.14 and ×1.15 in units, with intervals that include the model values.

This is a consistency check, not a validation. A coupon is not a shelf-price cut: only
redeemers pay less, coupons also advertise, and the model was fitted on the following
year's non-experimental prices. The fair statement: the observational price sensitivity
has the right sign and magnitude, compared with one randomized discount in one brand.
It is not calibrated by this experiment.

## Why no other randomized price test is possible here

- No other category has more than one cell (`f4` is all cell `01`), so the other seven
  categories contain no household lottery.
- Store prices, features and displays were set by retailers, so they are not randomized.
- The experiment ended at 1986 week 25. The joint eight-category basket window starts at
  week 26, so the basket model cannot be refitted on treated weeks. A tissue-only model
  covering 1985–86 could be.

A true randomized *shelf-price* test requires a live retailer trial. That protocol (lottery
assignment of stores or shoppers, fixed success metric, profit after cost) is in
[RETAILER_API_USER_MANUAL.md](RETAILER_API_USER_MANUAL.md).

## Reproduce

```bash
python -u scripts/run_erim_coupon_experiment.py --permutations 4999
```

Outputs: `artifacts/erim_coupon_experiment/report.json` (all estimates, null bands and
coupon-code tables), per-window household tables `households_{pre,test,post}.parquet`, and
`run.log`. Tests: `tests/test_erim_coupon_experiment.py`.

## Model prediction test

A tissue-only basket model fitted before the campaign was asked to predict these effects
blind to the cell labels. See [ERIM_TISSUE_MODEL_PREDICTION_TEST.md](ERIM_TISSUE_MODEL_PREDICTION_TEST.md).
