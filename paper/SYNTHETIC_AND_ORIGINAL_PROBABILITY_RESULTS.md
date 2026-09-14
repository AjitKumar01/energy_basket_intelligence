# Probability and price corrections: synthetic and original-data results

Analysis date: 2026-09-13. All results below concern the current
`theory-probability-foundations` branch. No original-data model was retrained or
overwritten during this analysis.

## Conclusion

The synthetic run supports the implemented probability identities, actual price
map, additive response penalty, and tested sampling procedures. The original-data
check independently confirms that the previous elasticity proxy was materially
wrong: it understates the magnitude of the additive model's uniform-price response
by about 50% on 512 locked validation contexts.

The saved interaction model still improves held-out likelihood after allowing for
within-household dependence. This does not make it a fully calibrated generator or
a validated pricing policy. Its historical particle bank fails the new absolute
ESS gate in two tail-size bands, and its existing generation diagnostics show a
basket-size mismatch. A fresh corrected fit and independent calibration audit are
needed before claiming the new pipeline is validated on original data.

## Reproducibility and evidence

The user's full synthetic run completed all five stages in 301.17 seconds:

```sh
python -u scripts/run_synthetic_experiment.py --profile full --threads 4
```

- [Synthetic pipeline log](../artifacts/synthetic_probability_full_20260912T184533Z/pipeline.log)
- [Synthetic manifest](../artifacts/synthetic_probability_full_20260912T184533Z/manifest.json)
- [Probability foundations results](../artifacts/synthetic_probability_full_20260912T184533Z/probability_foundations.json)

Every recorded synthetic report hash was verified. The synthetic manifest's
recorded source files matched the working tree when the results were inspected;
the original-data audit and its test were added subsequently. The user's dtype
fix is present in the successful run. Earlier failed smoke logs are not evidence
of failure of this completed full run.

The new, read-only checkpoint audit was run as:

```sh
python -u scripts/version4/audit_original_probability.py --contexts 512 --threads 4
```

- [Original-data pipeline log](../artifacts/original_probability_audit_20260912T215117Z/pipeline.log)
- [Original-data report](../artifacts/original_probability_audit_20260912T215117Z/report.json)
- [Per-context responses](../artifacts/original_probability_audit_20260912T215117Z/per_context.npz)

It completed in 5.16 seconds. Input hashes were checked before and after execution;
checkpoint loading also checked the data fingerprint. The report distinguishes
new numerical checks, reanalysed saved likelihood scores, and historical generation
and ESS diagnostics. These are not interchangeable forms of evidence.

## 1. Synthetic probability and sampling checks

There were three seeds, with additive/interacting and kappa=1/kappa=2.4 cases:
12 deterministic cases, each with 637 exhaustively enumerated supported baskets.
All declared gates passed.

| Calculation against enumeration | Maximum absolute error |
| --- | ---: |
| Actual model energy | 1.78e-15 |
| Actual common/relative price Jacobian | 1.50e-10 |
| Finite price-action reweighting | 4.16e-17 |
| Single-utility finite-tilt formula | 1.33e-15 |
| Additive log normalizer | 4.44e-16 |
| Additive uniform-price elasticity | 1.12e-13 |
| Elasticity-penalty parameter gradients | 8.53e-14 |
| Interacting HS quadrature log normalizer | 6.66e-14 |
| Interacting HS quadrature score | 3.32e-12 |

The additive generator passed its simultaneous checks using 12,000 draws per
seed. Interacting SMC used 48 independent replicates of 256 particles; its largest
tested moment error was 0.02488, within the predefined uncertainty/bias allowance.
Its largest normalizer-ratio deviation from one was 0.0002184. Independence is
across SMC runs, not assumed among particles within a run.

The randomized opportunity experiment used 216,000 opportunities per seed and
included no-purchase outcomes. All twelve reported outcome-effect intervals
covered their known oracle effects. Three successful seeds do not establish
long-run interval coverage. The adversarial ESS checks rejected concentrated
weights even when a fractional ESS threshold alone would pass.

These are implementation validations on the tested worlds, not a proof of
universal finite-particle accuracy or exact quadrature for arbitrary parameters.

## 2. Does correcting the price penalty help estimation?

The controlled comparison fitted the actual native-DP model with 54,000 training
and 54,000 validation trips per seed. The elasticity target was supplied by the
known synthetic oracle; only utilities and price loadings were fitted in this
comparison. This isolates the correction but is not full-model identification.

| Fit | Mean absolute elasticity error, three seeds | Mean oracle KL |
| --- | ---: | ---: |
| Likelihood only | 0.002105 | 0.000205 |
| Legacy proxy penalty | 0.006370 | 0.000291 |
| Corrected response penalty | 0.001118 | 0.000156 |

The correction reduced calibration error relative to the proxy in every seed,
with an 82.5% reduction in the average error. It also reduced oracle KL in every
seed versus both alternatives. Validation likelihood was not uniformly better
than likelihood-only fitting, and no broad superiority claim follows from three
seeds with an oracle-supplied target. Real calibration needs a defensible external
target and sensitivity to its uncertainty; the correction does not identify that
target from observational checkouts.

## 3. Recovery and pricing are not equally strong

[Exact interaction recovery](../artifacts/synthetic_probability_full_20260912T184533Z/synthetic_exact_certification.json)
behaved sensibly: fitting interactions slightly hurt held-out likelihood under
the null (approximately -0.0011 and -0.0015 nats/trip), while gains were positive
at strengths 0.35 and 0.7. This supports retaining an independent gain gate rather
than assuming interactions must help.

The complete-retailer reports expose remaining limitations:

| Metric | Well-specified world | Misspecified world |
| --- | ---: | ---: |
| Interaction-minus-additive test log likelihood, nats/trip | +0.2321 | +0.1163 |
| Interaction-kernel correlation | 0.9764 | 0.9313 |
| Price-elasticity correlation | 0.4032 | 0.4266 |
| Generated/observed size total variation | 0.0346 | 0.0711 |
| Counterfactual basket-size MAE | 0.0280 | 0.0149 |
| Predicted profit of selected policy | 398.70 | 411.46 |
| Oracle profit of that selected policy | 160.39 | 140.09 |
| Oracle optimal profit | 177.98 | 178.31 |
| Policy regret / oracle optimal profit | 9.9% | 21.4% |

Sources: [well-specified report](../artifacts/synthetic_probability_full_20260912T184533Z/synthetic_retailer_experiment.json)
and [misspecified report](../artifacts/synthetic_probability_full_20260912T184533Z/synthetic_retailer_misspecified.json).

Good basket likelihood and kernel recovery coexist with weak price-coefficient
recovery and substantial profit overprediction. Neither policy violated its
budget, but budget feasibility is not value calibration. The two retailer worlds
use different seeds, so their metric differences are not a paired estimate of
the effect of misspecification. In particular, the smaller counterfactual size
MAE in the misspecified run does not establish robustness. These recovery reports
are descriptive; their completion is not a pass gate for every economic metric.

## 4. Original-data numerical checks

The original catalogue has 5,455 products and basket-size support 1..120. We
retained the fitted common/relative price parameter, kappa=11.1513.

The additive checkpoint was evaluated on the first 512 contexts of the previously
locked validation manifest. Its aggregate elasticity is the derivative of mean
expected basket size divided by mean expected size, not the unweighted average
of per-context elasticities.

| Calculation | Elasticity |
| --- | ---: |
| Legacy mean-coefficient/variance proxy | -0.1039476741 |
| Exact covariance from native first adjoint | -0.2078408773 |
| Implemented fourth-order response | -0.2078408773 |

The independent reference differentiates E[N] with respect to slot utilities once,
obtaining Cov(N,Y_j), then contracts with the actual heterogeneous price
coefficients. It does not use finite differences. The largest per-context slope
discrepancy was 6.02e-10; halving the finite-difference step changed a slope by at
most 2.32e-9. Both actual uniform-price and single-product price maps matched the
analytic common/relative specification to less than 2e-15.

Eight locked validation scores from the final rank-five checkpoint were also
recomputed at their original quadrature level 7, matching the saved scores to
1.42e-14. This checks replay compatibility, not exactness of that quadrature.

Uniform price changes cancel the relative-price component; this cancellation
does not hold for a single-product action. The kappa=1 synthetic ablation therefore
does not justify silently replacing the existing fitted price model.

## 5. Original-data uncertainty, ESS and generation

Saved paired scores were reanalysed without refitting or selecting new models:

| Split | Trips / households | Mean gain, nats/trip | Household-clustered 95% interval |
| --- | --- | ---: | --- |
| Validation | 4096 / 1432 | +0.02675 | [0.02175, 0.03176] |
| Test | 4096 / 1401 | +0.03101 | [0.02510, 0.03691] |

Subtracting the empirical adjacent-quadrature-rule allowance leaves positive
lower diagnostics of 0.02140 and 0.02457. These allowances are not rigorous
exact-integral error bounds, nor do the adjusted endpoints have a demonstrated
joint 95% coverage guarantee. Household clustering permits within-household
dependence; it does not account for shared shocks across households or model-fit
and model-selection uncertainty.

The saved full-fit particle diagnostics imply:

| Size band | Draws | Minimum absolute ESS | New minimum 2 |
| --- | ---: | ---: | --- |
| 41..59 | 5 | 2.846 | Pass |
| 60..80 | 4 | 1.601 | Fail |
| 81..120 | 3 | 1.341 | Fail |

All smaller-size bands pass. These absolute values were reconstructed from saved
per-band minima, not estimated using a new bank. Their failure prevents treating
the historical fit as accepted under the new ESS contract; it does not erase the
separate held-out likelihood evidence.

The historical 64-context generation audit reports generated mean size 7.262,
model expected size 7.618, and observed mean size 10.031. Generated and observed
size variances were 72.97 and 136.28. There were no invalid-assortment or duplicate
item baskets, but structural validity does not imply distributional calibration.
This small-context discrepancy merits a larger independently replicated audit;
it is not by itself a population-level rejection test.

## Recommended next experiment

1. Fit a fresh additive model with the corrected response objective, preserving
   the current common/relative price specification. Keep the historical checkpoint
   as a comparator; do not continue its legacy-proxy optimizer state as if the
   objective were unchanged. State the elasticity target's source and uncertainty.
2. Rebuild interaction banks with the new tail allocation and enforce absolute
   and fractional ESS on both cross-fit and final banks. More draws are not an
   automatic pass; require the measured diagnostics to pass.
3. Recheck independent held-out likelihood, population and segment-level basket
   calibration, and independently replicated generation. Freeze all choices
   before a final test evaluation; do not tune on the test results reported here.
4. Validate action-level elasticities, profit and policy regret explicitly. Treat
   current original-data price actions as conditional model scenarios, not
   identified causal effects. Arrival/no-purchase and quantity components are
   not trained capabilities of the saved original-data checkpoint.

The probability law is a coherent conditional data generator. The remaining
work is empirical calibration, numerical reliability at deployment scale, and
identification of the price interventions one intends to use it for.
