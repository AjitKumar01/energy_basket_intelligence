# Remaining verification: instruction manual

Date: 2026-09-14

## 1. Objective and scope

Finish the requested sequence: **verify the theory and its implementation on
synthetic data with known truth, then evaluate the supported claims on real data**.

Do not interpret execution of every existing pipeline stage as completion of this
objective. Separate these questions throughout the work:

1. Does the code evaluate and sample the specified probability law correctly?
2. Does the fitted law predict observed real baskets and price-associated changes?
3. Does changing price cause the predicted change in outcomes?
4. Does an independently evaluated policy improve the intended business outcome?

Success on an earlier question does not establish the next one. Finite synthetic
experiments are implementation evidence under their tested worlds, not a universal
proof or evidence that real prices were randomized.

This manual is an implementation and experiment specification. New stages and
acceptance criteria described below are **not already implemented or passed**.
Writing this document does not launch experiments or authorize deployment.

## 2. Frozen starting point

Retain the completed run as a benchmark; do not overwrite it:

- Run: `artifacts/corrected_complete_rank5_20260913/`.
- Final checkpoint: `artifacts/candidate_rank1.pt` inside that run.
- Final checkpoint SHA256:
  `919b9d300be3f97cf1f72d7cfeb2a1f0e9cec456d6bb03e75281fe8b45a0b837`.
- Additive parent: `out/v3_pipeline_additive_best.pt` inside that run.
- Parent SHA256:
  `5a2ef61e32c52002ff65bf8bb75ec4b4c7cb0b643d839f0a24566921d4a1d243`.
- Data fingerprint:
  `ffad638b6bb4bac06371c5bc2d4e82bc0f4a32e9adcfb394bd849c8b9b569d0a`.
- Completed synthetic suite:
  `artifacts/synthetic_probability_full_20260912T221556Z/`.
- Detailed baseline assessment:
  [completed-run audit](../artifacts/corrected_complete_rank5_20260913/RESULTS_AUDIT.md).

The current original-data law concerns distinct-product incidence in a nonempty
basket, with size support 1..120 and 5,455 modeled products. It does not model
visits, null baskets, quantities, costs, inventory, or customer retention. The
catalogue is a declared support, not verified contemporaneous inventory.

Preserve the common/relative-price specification. Treat kappa=1 as a separately
labeled ablation, not a replacement introduced merely to simplify verification.
The existing -0.121 price calibration target is an assumption, not independent
real-data evidence that the fitted elasticity is correct.

### What has and has not been established

| Component | Existing evidence | Remaining obligation |
| --- | --- | --- |
| Probability/price mathematics | Exact-enumeration and synthetic tests | Verify every newly changed production path and numerical estimator |
| Interaction fitting | Rank 5; ridge 0.01; unchanged ESS gates passed | Preserve provenance and training-only selection |
| Held-out likelihood | Validation gain +0.01358; test gain +0.02210 nats/basket, positive paired intervals | Preserve comparator; do not attribute joint corrections solely to embeddings |
| Recommendation | MRR 0.09448; parent 0.09426; popularity 0.04272 | Parent contrast remains inconclusive |
| Generation | Valid support; mean 7.075 versus observed 10.031 on 64 validation contexts | Larger, uncertainty-aware, prespecified calibration study |
| Tails | Training-population safety tolerance passed | Held-out calibration and higher numerical fidelity |
| Price scenarios | Finite responses and high reweighting ESS | Independent numerical agreement, uncertainty, observed-response evaluation |
| Promotion policy | Expected-budget arithmetic works | Correct weighting, independent selection/evaluation, valid uncertainty and outcome scope |

## 3. Required execution order

1. Freeze provenance, estimands, panels, budgets, and acceptance rules.
2. Implement the missing evaluation infrastructure and regression tests.
3. Run synthetic verification, including independent oracle checks and failures.
4. On real data, first verify numerical responses and factual calibration.
5. Audit real price variation and choose a defensible response-evaluation design.
6. Evaluate price responses under that design, with an explicit claim level.
7. Only then evaluate a frozen promotion policy with the corrected estimand.
8. Produce separate capability verdicts and an evidence-linked final report.

Read-only feasibility checks on available data can happen early. Do not start an
expensive refit merely to discover later that the required intervention data are
absent. Reuse the frozen fit to diagnose evaluation and numerical problems first.

## 4. Phase A — provenance and experiment contract

### A1. Protect existing work

- Inspect `git status --short` and preserve unrelated changes. Remain on the
  current branch unless instructed otherwise; no implicit commit or push.
- Use a new, uniquely named artifact directory per experiment. Never use the
  completed run or historical root-level outputs as destinations.
- Record source hashes, dirty diff, checkpoint and data hashes, Python/package
  versions, commands, seeds, threads, timing, exit status, and per-stage logs.
- Freeze source files while a run is active. Resume only when source, data,
  checkpoint, panel and sampling-cache compatibility checks pass.
- Store context IDs, household IDs, weights, actions and exclusions alongside
  numerical results. An aggregate JSON alone is insufficient for an audit.

### A2. Declare the estimand and splits

Create `verification_protocol.md` and a machine-readable configuration in the new
run directory before examining new confirmatory results. Specify:

- Outcomes: incidence of which products, expected size, category composition,
  conditional basket value, or another explicitly supported quantity.
- Population: trip-weighted or household-weighted; eligibility and exclusions.
- Action: price only, or price plus display/mailer/other promotion components.
- Which covariates are fixed, which are pre-action, and which may respond to action.
- Training, selection, diagnostic and final-evaluation panels; selection seeds.
- Minimum practically important effects and acceptable errors, sampling precision,
  multiplicity family, maximum compute budget and stopping rules.

The existing test split has already been examined. Do not tune against it and
call a later result untouched confirmation. Prefer genuinely unexamined future
data. If unavailable, use a documented nested/rolling-origin design with refitting
and temporal separation, and label the resulting evidence appropriately. A new
random seed alone does not restore a previously used test set.

Practical tolerances must be fixed before the final evaluation. If they are not
specified, report diagnostics but mark practical acceptance `not_assessed`.
Do not equate a confidence interval containing zero with proof of calibration.

## 5. Phase B — synthetic verification before real-data conclusions

### B1. Verify the law and the actual price map

Use small catalogues for exact enumeration of every allowed nonempty basket.
Use an independently written oracle, not the same production routine on both
sides of a comparison. Check:

- Normalization, finite nonnegative probabilities, support and no duplicate items.
- Item incidences, pair incidences, size law, means and covariances.
- Energy differences, likelihood values, gradients, and finite price tilts.
- Uniform, one-SKU, and bundle price changes with heterogeneous coefficients.
- Common/relative-price coupling: update both slot price deviations and the actual
  assortment mean; a one-price action need not be a one-utility action.
- Zero action, composition of sequential actions, and return to the factual law.
- Multiple supports, catalogue sizes, rare items, price depths and interaction
  strengths, including deliberately poor-overlap cases.

For a fixed support and a statistic f without explicit action dependence, the
reference derivative is:

`d E[f(Y)] / da = Cov(f(Y), d E_theta(Y, a) / da)`.

For a price-dependent outcome such as sales, also include `E[df(Y,a)/da]`.
If assortment/support changes, the simple fixed-support reweighting identity is
not sufficient. Do not infer cross-price signs directly from a Gram coefficient.

Compare finite differences at several steps with the independent derivative.
Use scaled absolute/relative tolerances suitable for float64; freeze them from
development tests, including near-zero responses, before acceptance runs.

### B2. Separate three errors

Report independently:

1. **Numerical/sampling error:** production estimates versus the exact fitted law.
2. **Estimation error:** the fitted law versus the known data-generating law.
3. **Model misspecification:** performance when the generating law is outside the
   fitted family.

For known-law sampling, use independent SMC runs and increasing particle budgets.
Do not call resampled particles independent replicates. For fitted-model recovery,
use independent training/selection/evaluation draws, multiple world seeds, and
rare-event stress tests. Compare identified quantities, such as Gram matrices and
predictions; raw embedding coordinates are rotation-nonidentified.

Keep oracle parameters, moments, propensities and costs out of fitting/selection
except where the experimental design explicitly makes a quantity observed. Oracle
truth belongs in evaluation. For correction ablations, hold sample and compute
budgets fixed; do not attribute all improvement to code if data volume also changes.

### B3. Verify policy evaluation and failure detection

- In synthetic opportunity data with randomized offers and known costs, freeze
  action/policy selection before held-out evaluation.
- Check response bias, interval coverage across worlds, selected-policy value,
  regret against the oracle, and expected versus realized budget violations.
- Include null outcomes where the synthetic experiment observes opportunities.
  This does not authorize adding an arrival capability to the real-data model.
- Test that low ESS, wrong lineage, missing reports, inadequate precision and
  invalid probability masses fail closed rather than produce an accepted policy.
- Report inconclusive profit intervals as inconclusive; positive oracle value is
  not evidence that the finite holdout established positive value.

Run the existing full suite after changes, then the additional tests required by
this specification. The existing suite alone does not implement every item above.

## 6. Phase C — real-data numerical and factual verification

### C1. Verify responses of the frozen fitted law

Extend the original-data audit to accept explicit checkpoint, parent and run
paths. Its current defaults refer to historical root-level artifacts; running it
unchanged would not reliably audit the completed isolated run.

For locked representative and separately labeled high-risk contexts:

- Audit the additive parent's exact-DP response against a covariance/adjoint
  calculation and step-halved finite differences.
- Audit the final rank-5 interaction model, not just the additive parent.
- Compare factual-bank importance reweighting with fresh sampling directly at
  the changed price, across independent seeds. Use higher-rule quadrature on a
  tractable panel as another numerical check.
- Preserve the complete action transformation, support and factual context.
- Save per-context factual and changed expectations, differences, ratio definition,
  absolute and fractional ESS, particle counts, MC standard errors and fidelity.
- For poor overlap, regenerate at the target or bridge with intermediate actions;
  increasing particle counts is not a guaranteed fix for low fractional ESS.

Engineering starting budgets, not automatic acceptance thresholds: 128 contexts,
64/128/256 particles, four independent seed replicates. Scale only after profiling.
Acceptance requires agreement within prespecified combined numerical tolerances
and sufficient precision to distinguish the practically important response.
If the budget is exhausted, report `inconclusive`, not `passed`.

### C2. Distinguish sampler correctness from generator calibration

First compare generated samples to the fitted law on the same contexts. Then
compare the fitted law to observed baskets. This distinguishes a faulty sampler
from a miscalibrated model.

- Start with approximately 1,024 representative validation contexts if available,
  subject to a precision-based design; add a separately labeled stress panel.
- Keep the context weights identical between model and observations. Use
  household-cluster inference for repeated trips and account separately for MC
  noise. More baskets per context cannot replace more observed contexts.
- Compare mean, variance, size CDF/TV, category and item incidence, selected pair
  frequencies, and tail probabilities. Include segments, price bins and size-risk
  bins, with simultaneous uncertainty for prespecified subgroup claims.
- Assess equivalence to practical tolerances, not only failure to reject equality.
- For sparse distributions, simulate finite-sample reference discrepancies under
  the fitted law using the same context mix and observed sample size.
- Inspect selection/truncation: claims concern modeled nonempty baskets up to 120,
  not all retailer shopping opportunities or unmodeled products.

The existing 64-context panel underpredicts size, while segment 0 overpredicts it.
Do not impose a blanket upward correction. If a model change is needed, diagnose
which context/size component is wrong, fit on training data only, and repeat
synthetic checks and the fixed parent-comparison protocol.

### C3. Resolve tail fidelity separately from safety

- Re-evaluate the failed screen/confirm contexts at a higher rule and/or independent
  target sampler. Save probability masses and numerical agreement, not only means.
- Treat the observed 3.505-item maximum rule gap as unresolved until explained.
- Extend the population audit to validation and a legitimate final-evaluation
  population; the completed 160,007-context screen was training data.
- Keep high-risk selection separate from representative calibration estimates.
- Preserve the current safety gate, but add a distinct calibration gate. The
  current allowance of twice the observed tail rate plus 0.0005 is not equality
  of predicted and observed tails.
- Label the screen-error envelope empirical. Do not describe its sampled maximum
  as a rigorous bound for every unsampled context.

## 7. Phase D — the missing real-data price-response evaluation

### D1. Audit whether the treatment and outcomes are actually observed

Build a price-event provenance report before selecting an estimator. Inspect
`scripts/data/22_basket_data.py`, `scripts/version4/features.py`, data manifests,
`data/price_week.parquet`, `basket_input/store_price.npz`, and price/promotion panels.

For each prospective event record SKU, store, date/week, old/new price, price source,
observation/imputation flag, display/mailer flags, eligibility, available outcomes,
and event/assignment-unit ID. Explicitly audit:

- Loyalty versus base/posted price and coupon adjustments. A transaction-derived
  unit price can change with purchaser composition; it is not automatically an
  exogenous offered-price treatment.
- Chain fallback, carried-forward prices, missing store prices and stale observations.
  Do not treat an imputation jump as a real intervention.
- Observed offer/assortment availability; absence of purchases is not evidence of
  a stockout or of an observed no-purchase opportunity.
- Within-product/store variation, depth support, seasonality and concurrent promotion.
- Number of independent events and assignment clusters, not merely basket count.
- Leakage in baseline means, feature construction, controls and event selection.

Price-only and price-plus-display/mailer are different treatments. Define which
is being evaluated and change the corresponding model context consistently.

### D2. Choose and document the strongest defensible design

| Available evidence | Permitted evaluation | Required restriction |
| --- | --- | --- |
| Documented randomized offers with outcomes for eligible opportunities | Experimental contrasts; propensity-based policy evaluation if support exists | Verify assignment, eligibility, missing outcomes and interference |
| Defensible natural experiment or comparison design | Design-specific causal estimate, conditional on its assumptions | Document timing, controls, pretrends/placebos, spillovers and sensitivity |
| Ordinary observational prices and baskets | Held-out conditional predictive response assessment | Label associative; no automatic causal claim |
| Prices/exposure or controls inadequate | Numerical scenarios and factual diagnostics only | Mark real intervention validation blocked and list missing data |

Do not assume that fixed effects, matching, a propensity model or doubly robust
estimation remove unmeasured confounding. Do not apply inverse-propensity methods
to unsupported continuous/bundle actions or fabricate randomization probabilities.

For a proposed event-study/comparison design, declare event windows and controls
using pre-action information, account for overlapping events and calendar effects,
and check pretrends, placebo dates, concurrent promotions and sensitivity to the
control set. These checks support assumptions; they cannot prove them.

### D3. Match the model prediction to the empirical estimand

For every held-out event:

1. Freeze the model, event eligibility and background-context distribution.
2. Predict outcomes at factual and changed prices using the same eligible context
   panel, weights and declared action components.
3. Obtain the empirical contrast allowed by D2. An unadjusted before/after change
   is not a counterfactual ground truth.
4. Compare the estimated empirical contrast to the model contrast, with uncertainty
   from both sources. If they share observations, retain their dependence in the
   resampling/inference scheme.
5. Aggregate only over supported events, report exclusions, and compare with no
   response, fitted-parent response, and a training-fitted simple response baseline.

Report signed bias, MAE/RMSE, effect calibration, sign agreement for effects that
are resolvable, intervals, support/overlap and heterogeneity. Weight by the declared
target population; do not allow a few high-volume products to silently redefine it.

Cluster at the treatment-assignment unit, such as store/event, and account for
repeated households or time dependence when applicable. Household clustering alone
does not address a price shock shared by many households. With few independent
events, report limited precision rather than relying on basket-level sample size.

**Selection warning:** price may change who visits or buys. Conditioning on observed
nonempty baskets after treatment can change the population. Define the conditional
comparison honestly; do not reinterpret it as an effect on total demand or on a
fixed shopper population without the required assumptions and opportunity data.

Acceptance: the frozen model's errors must meet predeclared practical tolerances
with adequate uncertainty, on supported actions and a genuine evaluation panel.
If identification fails, stop the causal branch; do not substitute high ESS or
synthetic recovery as a real-data causal pass.

## 8. Phase E — correct and independently evaluate the policy

Do this only after the numerical and factual prerequisites are satisfactory.

1. **Fix weighting.** `run_segment_pricing_mdp.py` selects contexts round-robin by
   household but scales average responses by trip counts. Either sample trips
   representatively or apply documented inclusion/traffic weights. If targeting
   household value instead, change both the estimand and scaling consistently.
2. **Separate selection and evaluation.** Select bundles, discounts, budgets and
   schedules on training/selection data. Freeze the policy hash and evaluate on
   an independent panel. The current test-context valuations were also used to
   select schedules and are not an independent final policy evaluation.
3. **Fix uncertainty.** Replace sums of selected per-action 1.96-SE lower bounds
   with an evaluation of the frozen policy. Use simultaneous bounds for any
   screening family, and separate context uncertainty, SMC error and parameter
   uncertainty. A nested bootstrap/refit or documented sensitivity analysis may
   be needed; state explicitly when inference conditions on a fixed fit.
4. **Use the correct reward.** Conditional list-price basket value, post-discount
   incidence-weighted sales, and profit are different. Do not label the first two
   profit. Costs, quantities and opportunity outcomes require separate data/models.
5. **Handle budget risk.** Check expected and, if claimed, stochastic realized
   budgets. Uncertain markdown spend requires uncertainty propagation. Expected
   feasibility alone is not a hard-spend guarantee. Include no promotion and do
   not force minimum utilization for a non-beneficial campaign.
6. **Limit generalization.** Retain supported price depths and products. The
   current 28-day allocation has no learned visit, inventory or retention dynamics.

Use randomized/off-policy evaluation only where D2's design and action support
permit it. Otherwise label outputs simulated conditional value, not verified
policy improvement. Deployment or a new live experiment requires separate approval.

## 9. Engineering changes and required artifacts

The following changes are work to perform, not capabilities already available:

| Existing code | Required change |
| --- | --- |
| `audit_original_probability.py` | Explicit isolated-run/checkpoint paths; final interaction-model response audit; no historical-path fallback |
| `audit_particle_counterfactual_generation.py` | Locked panels, per-context outputs, independent replicate runner, fidelity and calibration criteria |
| `audit_customer_segments.py` | Same-context/weighted comparisons with appropriate observed and MC uncertainty |
| `audit_population_size.py` | Distinguish safety/calibration/fidelity; higher-rule follow-up and held-out populations |
| `run_segment_pricing_mdp.py` | Correct estimand weights, freeze/select/evaluate separation and valid policy uncertainty |
| `run_pipeline.py` | Enforce declared capability-specific gates; remove ambiguous universal certification |
| `summarize_pipeline_results.py` | Report passed/failed/inconclusive/not-assessed/not-identifiable separately, with lineage |
| New price-event evaluation module | D1–D3 data audit, event manifests, supported design, observed-versus-predicted contrasts |

Each stage should emit a report, per-context/event arrays, a persistent log and a
manifest with inputs and hashes. Save the following logical outputs under the new
run root; names here specify deliverables, not existing scripts:

- `verification_protocol.md` and machine-readable protocol.
- `synthetic_verification/` with oracle, production, recovery and coverage results.
- `price_data_provenance.json` and locked event/panel manifests.
- `real_numerical_price_audit.json` and replicate/per-context results.
- `real_generation_calibration.json` and size/segment/distribution diagnostics.
- `real_tail_fidelity.json` and confirmation arrays.
- `real_price_response_evaluation.json` with claim level and empirical contrasts.
- `frozen_policy.json` and `independent_policy_evaluation.json`.
- `VERIFICATION_REPORT.md` and `verification_status.json` linking all evidence.

Add regression tests for historical-artifact rejection, mismatched contexts,
support changes, null/missing acceptance values, low ESS, MC replication,
household-versus-trip weighting, selection/evaluation leakage, absent propensities,
and the inability to claim causal/profit capabilities from scenario-only reports.

## 10. Commands available now

Run from the repository root with the configured Python environment. Commands in
this section exist today. They do not by themselves implement the missing phases.

Read-only starting checks:

```sh
git status --short
python -m pytest -q
```

After implementation changes, rerun the existing synthetic full suite. It creates
a fresh timestamped directory with a pipeline log and per-stage reports:

```sh
python -u scripts/run_synthetic_experiment.py --profile full --threads 4
```

Before launching a diagnostic, inspect its actual interface:

```sh
python scripts/version4/audit_particle_counterfactual_generation.py --help
python scripts/version4/audit_population_size.py --help
python scripts/version4/run_segment_pricing_mdp.py --help
```

Do not run `audit_original_probability.py` with its present historical defaults
for this checkpoint. Implement explicit paths first. Do not fabricate CLI flags
for proposed new stages. A plain rerun of `run_pipeline.py --profile full` still
has the old verification gaps unless those stages and gates have been changed.

Budget policy: profile small panels, then scale only the limiting dimension.
Record time per context/particle/rule. Higher quadrature levels can grow rapidly.
At the compute cap, preserve partial outputs and report the unresolved criterion;
do not weaken a gate or repeatedly change seeds to obtain a pass.

## 11. Final acceptance and handoff checklist

- [ ] Synthetic exact-law, derivative, sampling and recovery checks pass after all changes.
- [ ] All reports trace to frozen code, data, checkpoints, panels and actions.
- [ ] Final-model real price responses agree across independent numerical methods.
- [ ] Factual generator calibration meets declared tolerances, or failures are explicit.
- [ ] Tail safety, tail calibration and numerical fidelity have separate verdicts.
- [ ] Real price-event provenance and identification feasibility are documented.
- [ ] Predicted responses are compared to supported empirical contrasts, with uncertainty.
- [ ] Causal claims are either supported under named assumptions or explicitly withheld.
- [ ] Policy weights match its estimand; selection and final evaluation are separate.
- [ ] Reward, costs, budget risk and unmodeled capabilities are stated accurately.
- [ ] Recommendation parent comparisons retain uncertainty; no significance claim from likelihood alone.
- [ ] Final report lists all failures, inconclusive results, exclusions, logs and commands.

Use separate verdicts for `numerical_price_verified`, `factual_calibration`,
`observed_price_response_evaluated`, `causal_price_identification`, and
`policy_value_evaluated`. A completed execution must not set all of them to true.

If real-data identification is impossible with the available inputs, a valid
handoff states exactly what was verified, what remains unidentifiable, and what
additional data or design is required. It must not say that real price
counterfactuals were verified merely because the simulator returned numbers.

## References within this repository

- [Probability foundations and price counterfactuals](PROBABILITY_FOUNDATIONS_AND_PRICE_COUNTERFACTUALS.md)
- [Synthetic verification design](CORRECTED_FIT_EXPERIMENT_DESIGN.md)
- [Existing synthetic pipeline](PROBABILITY_FOUNDATIONS_SYNTHETIC_PIPELINE.md)
- [Completed real-data audit](../artifacts/corrected_complete_rank5_20260913/RESULTS_AUDIT.md)
- [Completed real-data results](../artifacts/corrected_complete_rank5_20260913/RESULTS.md)
