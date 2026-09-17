# ERIM store availability input: implementation and full refit

Date: 2026-09-17

## Why

The causal validation report ([ERIM_CAUSAL_VALIDATION_REPORT.md](ERIM_CAUSAL_VALIDATION_REPORT.md))
found that a new product's purchases cannot be modelled without knowing when and where it
was stocked. The basket model had no such input: every store was assumed to offer the whole
464-product catalogue in every week. This change adds a store availability input to the main
pipeline and refits ERIM with it.

## What ERIM's store sales feed shows

- Of the 15,715 tracked product–store pairs, only 7,275 ever sold at that store.
- 279 of those pairs first sold after the training weeks, so products were launched
  mid-window.
- No panel purchase occurred at a store whose sales feed never shows that product.
- A strict "available from its first sale" rule would put 0.54% of purchase lines before
  availability, almost all slow sellers early in the feed.
- A static assortment built from training-period sales would exclude 2,072 held-out
  purchases of later launches.

## Rule (`availability="retail_first_sale"`)

Implemented in `CanonicalBasketModelInputBuilder._write_availability`:

1. **Launch.** A product is confirmed at a store from the first period in which the store's
   retail aggregate records units **net of the modeled cohort's own units**. A purchase
   therefore cannot confirm its own availability.
2. **Start of the feed.** A first sale within 13 periods of the store's feed start counts as
   available from the start.
3. **Before launch.** The product stays in the choice set. Its utility gains log ε, because
   a missing sale does not prove absence. ε is the training-period purchase rate in
   unconfirmed cells relative to confirmed cells, clipped to [1e-4, 1]. ERIM gives
   **ε = 0.0030**.
4. **No feed.** Products or stores without a feed are always available. That covers the 15
   frozen-dinner products, and every product in Dunnhumby, which has no availability panel
   and is unchanged.

**Result for ERIM:**
- 54% of training choice slots are unconfirmed.
- 517 training purchase lines and 1 validation line fall in unconfirmed cells; the test
  split has none.

## Model integration

- **Utility.** `RaggedModel.b_at` adds `log_avail`. The ragged support is unchanged, so the
  energy and every normalizer basket scale an unconfirmed product by exactly ε. A new test
  checks this against brute-force enumeration.
- **Plumbing.**
  - `Features.log_availability` looks up the offset from `availability.npz`.
  - `Batcher`, the baselines and the retail API forward it.
  - The panel is part of the data fingerprint as `availability_panel`.
  - The pipeline logs the contract at the data stage.
- **Existing bundles.** Dunnhumby and the retail-only ERIM bundle load with availability
  disabled and produce identical contexts.
- **CLI.** `build_canonical_basket_input.py --availability retail_first_sale`. The default,
  `disabled`, keeps old builds reproducible.

## Refit

- **Bundle.** `data/erim_basket/model_input_availability`. Its baskets, affinity partition
  and price panels are byte-identical to the retail-only bundle; only availability
  differs.
- **Stages.** Same as the retail-only refit:
  1. Price evidence (failed its held-out gate again).
  2. Parent fit with price fixed at zero.
  3. Joint price coefficients and alignment audit.
  4. Full pipeline.

  Every stage exited cleanly, and the pipeline's likelihood, numerical and population-safety
  gates passed.
- **Aborted first attempt.** The first attempt stopped because a test file was edited
  during the run, triggering the source-freeze guard. It is kept as
  `price_zero_parent_aborted_source_edit` and produced no results.

## Paired comparison with the retail-only refit

Same trips, households and trip manifests; 95% household-clustered intervals.

| Quantity | Retail-only | With availability | New − old |
|---|---:|---:|---:|
| Test log-likelihood, parent | −7.6295 | −7.3047 | **+0.325 (0.297 to 0.353)** |
| Test log-likelihood, final model | −7.6220 | −7.2998 | **+0.322 (0.294 to 0.350)** |
| Validation log-likelihood, final model | −7.4109 | −7.1698 | **+0.241 (0.215 to 0.267)** |
| Interaction gain over parent, test | +0.0076 (0.0060–0.0091) | +0.0049 (0.0033–0.0066) | smaller |
| Price contribution (on − off), test | +0.045 (0.036–0.054) | +0.048 (0.039–0.057) | similar |
| Joint price test gain | +0.0425 | +0.0454 (0.041–0.050) | similar |
| Recommendation MRR (popularity 0.128) | 0.2638 | 0.2508 | **−0.013 (−0.023 to −0.003)** |
| Generated / observed mean basket size | 1.635 / 1.703 | 1.677 / 1.703 | closer |
| Generated category total variation | 0.118 | 0.109 | better |
| 20% price cut: own incidence, size change | ×1.33, +0.30 | ×1.35, +0.40 | larger size response |
| 20% price cut, minimum reweighting ESS | 0.73 | 0.31 | less stable |
| Population tail gates | pass | pass | |
| Promotion MDP (net sales) | no promotion (0/18 with positive lower bound) | no promotion (0/24) | unchanged |

The price alignment placebos passed on both splits: actual product and week alignment beat
every placebo. Price coefficients rose by 3–5% in every category.

## Reading the results

1. **Availability is the largest held-out likelihood improvement found for ERIM:**
   +0.32 nats per basket on test. That is about 40× the interaction terms' contribution and
   7× the price term's. Most of the old model's error came from normalizing over products
   the store did not stock.
2. **Interaction gain fell from 0.0076 to 0.0049.** Part of what the interactions captured
   was shared store assortment. For example, products stocked by the same stores are more
   often bought together. That pattern now belongs to availability, so the remaining
   interaction evidence is cleaner.
3. **Recommendation MRR fell by 0.013, and the decline is statistically significant.**
   Broken down by how widely the hidden product was stocked:

   | Hidden product stocked in | Cases | MRR before | MRR after |
   |---|---:|---:|---:|
   | ≤10 stores | 141 | 0.215 | 0.231 |
   | 11–25 stores | 108 | 0.173 | 0.195 |
   | 26–34 stores | 159 | 0.122 | 0.127 |
   | All 35 stores | 344 | 0.378 | 0.334 |

   The old model mistook limited distribution for low appeal. Correcting that ranks
   limited-distribution products better but costs ubiquitous best-sellers some top-1
   positions (16.5% → 14.5% top-1 hits). More cases improved (39%) than worsened (35%), and
   median rank improved slightly (12 → 11.5), but MRR is dominated by rank-1 hits. The
   likelihood gain shows the new model is the better description of purchases. A retailer
   who only cares about the single top suggestion should compare both on its own data.
4. **Price-cut scenarios became less stable.** The 20% cut's reweighting ESS fell to 0.31,
   and its predicted size change rose. Scenarios far from observed prices should be read
   with more caution. They remain observational model scenarios, not causal effects.
5. **Generation is closer to observed baskets** in mean size and category mix.

## Decision

Availability-aware ERIM (`data/erim_basket/model_input_availability`,
`artifacts/erim_availability_refit/full`) is the better model of purchases and becomes the
recommended ERIM contract. Two caveats go with it: the MRR trade-off above, and weaker
reweighting stability for large price cuts.

## Artifacts

| Item | Location |
|---|---|
| Bundle build log | `artifacts/erim_availability_refit/00_bundle_build.log` |
| Parent, price coefficients, full run | `artifacts/erim_availability_refit/{price_zero_parent,joint_price_utility,full}/` and the `0*_*.console.log` files |
| Paired comparison | `artifacts/erim_availability_refit/comparison/run_comparison.json`, `price_ablation.json` |
| Code | `scripts/version4/canonical_basket_input.py`, `features.py`, `fit.py`, `ragged.py`, `baselines.py`, `provenance.py`, `scripts/run_pipeline.py`, `retail_api/service.py` |
| Tests | `tests/test_availability.py`, `tests/test_pipeline_hardening.py` |
