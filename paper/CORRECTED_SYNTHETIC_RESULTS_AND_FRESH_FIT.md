# Corrected synthetic results and fresh original-data fit

Date: 2026-09-13. Implementation and acceptance criteria are described in
[the experiment design](CORRECTED_FIT_EXPERIMENT_DESIGN.md).

## Synthetic outcome

The full run completed all nine stages in 359.18 seconds. Every recorded report
hash was verified. All probability-foundation gates and all six retailer-stage
generation/policy gates passed.

- [Aggregate synthetic log](../artifacts/synthetic_probability_full_20260912T221556Z/pipeline.log)
- [Manifest and stage logs](../artifacts/synthetic_probability_full_20260912T221556Z/manifest.json)
- [Probability and tail-ESS results](../artifacts/synthetic_probability_full_20260912T221556Z/probability_foundations.json)

The synthetic source snapshot was frozen during the run. Subsequently, the
original-data driver's top-level provenance import and tail allocation were
corrected and regression-tested before starting the fresh fit; those changes do
not alter the completed synthetic calculations. The fit has its own source hashes.

| World / seed | Observed mean size | Generated mean size | Policy plug-in profit | Oracle value of selected policy |
| --- | ---: | ---: | ---: | ---: |
| Well-specified / 73021 | 3.4370 | 3.4377 | 684.38 | 808.91 |
| Misspecified / 73122 | 3.3189 | 3.3248 | 941.58 | 924.29 |
| Well-specified / 93021 | 3.4184 | 3.4344 | 853.03 | 1006.46 |
| Misspecified / 93021 | 3.4354 | 3.4341 | 796.51 | 987.24 |
| Well-specified / 113021 | 3.5802 | 3.5579 | 648.59 | 825.84 |
| Misspecified / 113021 | 3.5618 | 3.5777 | 734.63 | 843.97 |

The largest profit overprediction was 1.87%; five cases underpredicted value.
All six selected policies satisfied the oracle expected-budget check. This is
not a claim of exact profit recovery: underprediction reached about 21.5%, policy
regret remains, and the 10% spending reserve is not a stochastic budget guarantee.

The independent randomized held-out profit intervals supported positive value
in only one of the six cases. The others remain statistically inconclusive, even
though their synthetic oracle values are positive. The implementation now reports
this distinction instead of treating the optimizer's own prediction as evidence
of realized profit.

The enumerated tail stress cases had legacy minimum ESS values of 1.626, 1.279,
and 1.642. Pilot budget increases followed by independent 2048-per-band audit banks
raised those minima to 893.34, 440.42, and 834.57. Maximum absolute log-normalizer
ratio errors were 0.01228, 0.01275, and 0.01371, within the declared 0.08 tolerance.
These small-world experiments validate the recovery procedure; they do not
substitute for measuring ESS in the original-data fit.

## Recommendation metrics

The synthetic base well-specified run reports:

| Metric | Additive | Interaction |
| --- | ---: | ---: |
| MRR | 0.33251 | 0.34991 |
| Interaction Recall@5 | — | 0.57517 |
| Interaction Recall@10 | — | 0.79504 |

The interaction MRR exceeded additive MRR in all six retailer runs. These are
descriptive paired-manifest comparisons, not a universal recommendation guarantee.

The fresh original-data evaluation is configured for 2000 test trips, hiding one
bought item per eligible basket and ranking the full contemporaneous assortment.
It reports MRR and Recall@5/10/20/100 against popularity, within-model ablations,
and the separately fitted additive parent. Full-minus-parent gains use
household-clustered paired uncertainty, with per-case ranks retained for replay.
That evaluator was also exercised on 32 historical trips as a compatibility
check; those small-sample scores are not fresh-fit results.

## Original-data fit: launched, not completed

The fresh full fit was launched in detached screen session
`corrected_fit_full_20260913`, with a fresh initializer and optimizer. It retains
the common/relative price model and calibration target -0.121, now applied to
the corrected response. The original historical checkpoints and reports were
not replaced.

```sh
python -u scripts/run_pipeline.py --profile full --threads 4 \
  --run-dir artifacts/corrected_fit_full_20260913 --start-at initialize
```

- [Live aggregate fit log](../artifacts/corrected_fit_full_20260913/invocation_20260912T222337Z/pipeline.log)
- [Live stage/status manifest](../artifacts/corrected_fit_full_20260913/invocation_20260912T222337Z/manifest.json)
- [Additive training log](../artifacts/corrected_fit_full_20260913/out/v3_pipeline_additive.log)

The full run proceeds through additive fitting, rank selection, interaction and
household-size refinement, recommendation/likelihood/generation evaluation and
population audits. It uses 100 draws per context, with 16 draws in each of the
three largest-size bands, while preserving absolute ESS >=2 and fractional
ESS >=0.20. A failed gate stops the pipeline; passing the synthetic suite does
not waive any original-data gate.

Do not edit Python/C++ source or tests while this run is active: the driver
checks its source snapshot before downstream stages. The manifest, not this
launch note, is authoritative for current status. The original-data recommendation
report will be written under `artifacts/corrected_fit_full_20260913/reports/` only
after its evaluation stage finishes.

Verification before launch: 102 tests passed, an isolated original-data smoke
fit completed, and the recommendation compatibility check completed. The first
smoke invocation exposed a missing top-level provenance import; it was fixed,
regression-tested, and the full fit successfully passed that startup path.
