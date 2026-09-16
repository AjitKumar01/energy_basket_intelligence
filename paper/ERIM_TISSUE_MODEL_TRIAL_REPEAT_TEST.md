# Tissue model test 3: trial and repeat

Date: 2026-09-17

Follows [ERIM_TISSUE_MODEL_RETEST.md](ERIM_TISSUE_MODEL_RETEST.md).

## Question

The re-test predicted purchase levels well but almost no coupon lift for arm B. B's
observed lift came early and was large relative to its extra redemptions. Cottonelle was a
new brand, so a plausible cause is that a coupon-driven first purchase raises later
buying. Does adding that trial-and-repeat effect, learned from pre-campaign weeks, let the
model predict B's lift?

## Design

The protocol is fixed in the docstring of `scripts/run_erim_tissue_prediction_test_v3.py`.
Designed after the earlier tests' results were known, so not blind. Nothing else changed
from the re-test.

- **Change.** Cottonelle utility gains λ once the household has bought Cottonelle. The
  brand launched inside the training window, so every household's history is observed
  from zero. In simulation the history evolves with simulated purchases; at the test start
  it is the observed status (43% of experimental households had tried Cottonelle).
- **Identification risk.** Persistent taste can masquerade as habit. The primary model
  keeps household Cottonelle taste. Sensitivities: λ = 0, which should reproduce the
  re-test, and no household Cottonelle taste, which gives an upper bound on λ.
- **Criteria.** C1–C5 are unchanged from the re-test.
- **Expectation, stated before running.** This targets C1 for arm B. It should not fix the
  redemption misses (C3, C5).

## Results

**Fitted history effect.**
- Main fit: λ = **−0.38**.
- Refits: −0.74 to −0.62, while the trend θ moved from +0.03 to between +0.07 and +0.18.
  The history effect and the awareness trend trade off, so neither is well identified.
- Without household taste: λ = +0.36.
- Held-out log-likelihood improved only from −1.1372 to −1.1365 per shopping week.

A negative λ means that after buying Cottonelle a household is *less* likely to buy it in
following weeks. That fits pantry stocking of a storable product, or trial that did not
convert, better than habit formation.

| Criterion | Quantity | Observed | Prediction (90%) | λ = 0 | No household taste | Pass |
|---|---|---|---|---|---|---|
| C1 | Cottonelle purchase-weeks, A | +0.34 (0.08 to 0.60) | +0.12 (0.10–0.13) | +0.13 | +0.14 | ✓ (barely) |
| C1 | Cottonelle purchase-weeks, B | +0.34 (0.08 to 0.61) | +0.02 (0.01–0.02) | +0.02 | +0.02 | **✗** |
| C2 | Tissue purchase-weeks, A / B | −0.16 / +0.05 | +0.08 / +0.01 | | | ✓ |
| C3 | Redemptions, A / B | +0.52 / +0.14 | +0.17 / +0.02 | | | **✗** |
| C4 | A − B Cottonelle purchase-weeks | +0.00 (−0.27 to 0.27) | +0.11 | | | ✓ |
| C5 | Control Cottonelle purchase-weeks | 2.16 | 2.35 (+9%) | 2.38 | 2.33 | ✓ |
| C5 | Control redemptions | 0.73 | 0.17 (−77%) | | | **✗** |

B's first-half lift was +0.28 observed against +0.02 predicted.

**Verdict: fail**, on the same criteria as the re-test. The λ = 0 run reproduces the
re-test (+0.13, +0.02, 2.38), which confirms the implementation.

## Conclusion

1. **Trial and repeat learned before the campaign does not explain arm B.** Even the upper
   bound, which attributes all persistence to habit, moves B's predicted lift only from
   +0.02 to +0.02.
2. **Pre-campaign purchase data point the other way.** After a Cottonelle purchase,
   households bought it less in the following weeks, consistent with stocking up.
3. **The binding constraint is coupon exposure.** Every version predicts that coupons are
   used by few households within about three weeks. The campaign's coupons were redeemed
   about four times as often and over months. With exposure that short, B's schedule is
   nearly identical to control's in the model, whatever the purchase response.
4. **Normal purchasing is still predicted well:** Cottonelle within 9%, all tissue within 3%.

Across three versions, the basket model predicts ordinary purchase levels once product
availability is modelled. It cannot predict this campaign's causal effect from
pre-campaign data. The missing piece is not the purchase response but how the campaign's
coupons reached households, and that is not recorded in the archive.

The only remaining route on this data is the cross-arm transport test: calibrate coupon
reach and life on one arm's redemptions, then predict the other arm's purchase lift.

## Reproduce

```bash
python -u scripts/run_erim_tissue_prediction_test_v3.py     # about 12 minutes
```

Outputs are in `artifacts/erim_tissue_prediction_test_v3/`:
- `prediction.json`;
- `report.json`, which includes a weekly in-sample check over the training window;
- `run.log`.

Tests are in `tests/test_erim_tissue_prediction_test_v3.py`. They check that λ = 0 matches
the re-test likelihood, that the history indicator is strictly lagged, and that a positive
λ raises simulated repeat purchasing.
