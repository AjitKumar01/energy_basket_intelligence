# Version-4 energy basket model

This branch is the minimal, reproducible implementation of the model in
[`paper/version4.html`](paper/version4.html). It contains one selected end-to-end
pipeline—not a collection of numbered experiment runs.

The statistical law is unchanged:

\[
p_\theta(S\mid x)=\frac{\exp E_\theta(S,x)}{Z_\theta(x)},
\qquad 1\leq |S|\leq120,
\]

over the complete 5,455-product catalogue, with the original additive utility,
household/context effects, price response, category term, total-size potential, and
low-rank Gram interactions. Recommendation, basket generation, price counterfactuals,
and retailer simulation all come from this same joint law; there is no separate
recommendation training objective.

## Why this pipeline

The selected pipeline is:

1. deterministic data preparation and a training-only affinity partition;
2. fresh initialization—no learned checkpoint is loaded;
3. exact additive joint maximum likelihood using the category/cardinality dynamic
   program, with validation-driven learning-rate decay, a convergence gate, and a
   complete-support bound on the existing category coefficient;
4. one split-half spectral pass that certifies the largest stable rank from 8 down to 4;
5. one cross-fitted constrained Monte Carlo likelihood solve for the PSD interaction
   kernel and the original total-size potential correction, using fixed exact draws from
   the additive law;
6. one strictly concave household-size block update in an identified catalogue-common
   direction of the existing household/product utility, selected by within-household
   cross-fit and capped by the population tail screen;
7. locked complete-support likelihood, recommendation, generation, price, and
   population-tail audits.

This is the corrected end-to-end pipeline. It never searches over \(\rho_0\)
initializations and does not run stochastic Smolyak gradients. In the certified product
basis it writes \(K=UCU^\top\), optimizes \(C\) directly under
\(0\preceq C\preceq I\), and jointly solves a linear/quadratic correction inside the
original \(\rho_0(n)\). Fixed common draws make the sampled likelihood deterministic and
concave in these natural parameters. PSD/tail projection and Armijo backtracking make
every accepted step monotone. Cross-fit gain and proposal effective sample size are
checked before the independent Smolyak audit. Certification remains fail-closed: if the
optimizer does not converge, numerical fidelity fails, or the model
puts excessive mass on anomalously large baskets, the last stage exits nonzero and keeps
the candidate artifacts for diagnosis rather than calling them production-ready.

The category bound is

\[
(-\rho_c)_+{m_c\choose2}\le1.5,
\]

where (m_c) is the largest supported offered count in category (c). It preserves the
Version-4 energy exactly while preventing a broad learned affinity group from creating an
unobserved large-basket clique phase. Narrow two-item groups retain the original
(-1.5) coefficient range.

The evidence and decision rule are in [`paper/PIPELINE.md`](paper/PIPELINE.md). The full
model derivation is in [`paper/THEORY.md`](paper/THEORY.md), and estimator details are in
[`paper/ESTIMATOR.md`](paper/ESTIMATOR.md).
The completed corrected-data fit and its fail-closed production decision are reported in
[`paper/CORRECTED_PIPELINE_RESULTS.md`](paper/CORRECTED_PIPELINE_RESULTS.md).
The completed rank-one successor, including its locked likelihood, recommendation,
generation, external-baseline, interaction-embedding and population-certification results,
is reported in
[`paper/RANK1_PIPELINE_RESULTS.md`](paper/RANK1_PIPELINE_RESULTS.md).
The exact-enumeration interaction recovery benchmark and its real-data diagnosis are in
[`paper/SYNTHETIC_INTERACTION_AUDIT.md`](paper/SYNTHETIC_INTERACTION_AUDIT.md).
The conditional size failure, rank-one household reparameterization, constrained pilot,
and unchanged sampling recursion are derived in
[`paper/HOUSEHOLD_SIZE_AUDIT.md`](paper/HOUSEHOLD_SIZE_AUDIT.md).
The finite-horizon, budget-constrained three-segment promotion environment and its
limitations are documented in
[`paper/SEGMENT_PROMOTION_MDP.md`](paper/SEGMENT_PROMOTION_MDP.md).
The complete downstream theory—exact conditional draws, interaction-tempered SMC,
normalizer and consistency guarantees, Rao--Blackwell incidences, recommendation,
counterfactual reweighting, segmentation, calibration, quantity boundaries and the MDP—is
in [`paper/INFERENCE_AND_SIMULATION.md`](paper/INFERENCE_AND_SIMULATION.md).
The complete small-retailer synthetic experiment—explicit no-purchase opportunities,
exact basket enumeration, randomized promotions, quantities, costs, counterfactuals and
oracle policy regret—is in
[`paper/SYNTHETIC_RETAILER_EXPERIMENT.md`](paper/SYNTHETIC_RETAILER_EXPERIMENT.md).

## Synthetic retailer demonstration

The synthetic driver needs no dunnhumby files. It runs an exact no-interaction control,
a well-specified complete retailer and a deliberately misspecified retailer:

```bash
python scripts/run_synthetic_experiment.py --profile full --threads 8 \
  2>&1 | tee artifacts/synthetic_experiment_full.log
```

The experiment contains explicit customer-day no-purchase outcomes, randomized logged
offers, store choice, the Version-4 basket law, shifted-negative-binomial quantities,
costs and a 28-day promotion budget. Every one of the 60,459 supported baskets in the
20-product retailer worlds is enumerated, so their likelihood and basket counterfactuals
have no numerical-integration error. Synthetic results validate recovery under declared
truth; they do not replace real held-out evaluation or randomized commercial trials.

## Current empirical status

The rank-one pipeline has completed from fresh initialization. It selected rank 5 and, on
locked 4,096-trip panels, improves over its matched exact additive parent by
\(0.02671\pm0.00211\) nats/basket on validation and \(0.03275\pm0.00239\) on test. The
q8 numerical-error upper bounds are \(0.000318\) and \(0.000468\) nats respectively, so
the positive likelihood gains are not quadrature artifacts. Locked test MRR is
\(0.09525\pm0.00607\). The interaction-only MRR gain is positive but its 95% interval
still crosses zero.

The three declared external baselines were then trained from fresh lineages to their
validation convergence certificates and scored on the identical locked 4,096-trip test
manifest. The model gains \(2.25204\pm0.09842\) nats over Bernoulli,
\(2.26164\pm0.09767\) over DPP, and \(1.81225\pm0.09239\) over NDPP. These are paired
standard errors; all three likelihood advantages are statistically clear. Multinomial and
the exact additive parent are retained as ablations, while SHOPPER is not included in this
external headline because its posterior/sequential fitting protocol is not matched to the
three direct-likelihood baselines.

An orientation-invariant audit also finds held-out structure in the interaction kernel.
The 2,000 strongest training-selected cross-affinity Gram pairs have 29,911 test
co-incidences versus 24,589.4 under a frequency-and-size configuration null (lift 1.216),
whereas matched controls have lift 0.998. This supports aggregate interaction information,
not causal or uniformly reliable SKU-level complement claims.

The former parent pipeline's localized extreme-basket failure is resolved: the complete
160,007-context q6 screen and 2,048-context q7 confirmation find no context with majority
probability on \(N\ge60\), and the calibrated population tail upper bound is
\(0.002013\), below the allowed \(0.004250\). Every declared technical certification gate
passes.

That pass is not a claim that the model is already a commercial digital twin. On the
64-context-per-segment generation panel, generated baskets remain too small and
under-dispersed (mean \(7.34\), variance \(63.83\)) relative to the selected observed
baskets (mean \(10.03\), variance \(136.28\)). The promotion MDP is therefore restricted
to campaign shortlisting and A/B-test design until visit probability, quantities, costs,
inventory and causal intervention evidence are added. Historical baseline numbers from
the old preprocessing are not treated as corrected-data comparisons.

## Requirements

- Python 3.11 or newer
- macOS or Linux (Windows users should use WSL)
- a C++ compiler compatible with the active Python installation
- about 12 GB free RAM for the full-population stages
- at least 5 GB free disk in addition to the raw CSV bundle
- dunnhumby *The Complete Journey* CSV files (not redistributed here)

CUDA is not currently a valid backend for the certified likelihood fit. The dominant
normalizer is an exact float64 elementary-symmetric/category-polynomial dynamic program
with a custom probability adjoint, and its compiled implementation is CPU-only. The rank
stage also uses SciPy sparse and convex CPU solvers. Merely moving the small dense utility
blocks to a GPU would retain the CPU bottleneck, add transfers, and would not constitute a
CUDA implementation of this estimator. Apple MPS is excluded for the same reason.

The pipeline is nevertheless hardware-portable: `--device auto` detects CUDA/MPS, RAM,
CPU count and PyTorch build information, selects the fastest *eligible* exact backend, and
chooses a conservative CPU thread count. It writes the complete decision to
`artifacts/runtime_capabilities.json`. Inspect a new machine without reading data or
starting a fit:

```bash
python scripts/inspect_hardware.py
```

An explicit unsupported request such as `--device cuda` fails before training instead of
falling back silently or swapping in a less accurate likelihood. `--threads N` overrides
the automatic CPU choice when a machine-specific benchmark justifies it.

Install:

```bash
python --version  # must report 3.11 or newer
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

## Data

Download the dunnhumby CSV bundle and point `NF_RAW_DIR` at the directory containing
`transaction_data.csv`, `product.csv`, and `causal_data.csv`:

```bash
export NF_RAW_DIR="/absolute/path/to/dunnhumby_The-Complete-Journey CSV/"
```

Derived files under `data/` and `basket_input/` are deliberately ignored by Git because
the source-data license does not permit redistribution.

The pipeline accepts the original release files only. Before training, the preprocessing
audit checks their locked SHA-256 digests; a modified, partially extracted, or different
release fails closed instead of silently producing a different cohort.

The cohort is defined using training weeks 9--82 only: the top 5,455 products by training
line frequency, followed by households with 20--300 training shopping days. Validation is
weeks 83--90 and test is weeks 91--101. Weeks outside 9--101 are excluded because the
promotion file does not cover them. No held-out purchase line is removed after cohort
definition; in the absence of a stock feed, every modeled store uses the complete declared
5,455-product chain catalogue.

Every pipeline invocation runs `scripts/data/audit_preprocessing.py` before fitting. It
checks the locked raw-file digests, independently reconstructs cohort membership and every
basket-product outcome, verifies modal chain/store prices and promotion coverage, and
writes `basket_input/preprocessing_manifest.json`. See
[`paper/PREPROCESSING_AUDIT.md`](paper/PREPROCESSING_AUDIT.md).

## Check before spending compute

Print the exact command graph without training:

```bash
python scripts/run_pipeline.py --from-raw --dry-run
```

The dry run is the recommended first command after cloning. Its preflight checks the
Python version, required modules, compiler, raw input filenames, and—when the raw rebuild
flag is omitted—the required derived files. It then shows every stage, fixed panel size,
rank gate, quadrature level, and output location. It does not compile the extension, hash
the raw files, or fit a model.

## Fresh-clone integration check

Use the smoke profile before committing a long full fit:

~~~bash
git clone --branch version4-household-size-rank1 --single-branch \
  https://github.com/AjitKumar01/nf_dunnhumby.git
cd nf_dunnhumby
python --version  # must report 3.11 or newer
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export NF_RAW_DIR="/absolute/path/to/dunnhumby_The-Complete-Journey CSV/"
pytest -q
python scripts/run_pipeline.py --from-raw --profile smoke \
  2>&1 | tee artifacts/pipeline_smoke.log
~~~

This exact clean-clone workflow was executed on 2026-09-01. It rebuilt the corrected
5,455-product data from the three raw CSVs, passed the preprocessing audit, compiled the
native extension, initialized a fresh model, and exercised additive fitting, rank
selection, interaction and household-size fitting, likelihood, recommendation,
counterfactual generation, customer segmentation, complement auditing, population-tail
screening, and the promotion MDP. The smoke model is deliberately undertrained: its
statistical certification may fail and its scores must not be reported as research
results. A zero process exit means the software path completed, not that the smoke model
passed production gates.

## Full end-to-end execution

From raw CSVs:

```bash
mkdir -p artifacts
python scripts/run_pipeline.py --from-raw 2>&1 | tee artifacts/pipeline.log
```

If `data/` and `basket_input/` already exist:

```bash
python scripts/run_pipeline.py 2>&1 | tee artifacts/pipeline.log
```

If the machine stops during the exact-additive stage, continue the same from-scratch run
without resetting Adam or replaying minibatches:

```bash
python scripts/run_pipeline.py --resume-additive out/v3_pipeline_additive.pt \
  2>&1 | tee -a artifacts/pipeline.log
```

The full profile has a 30,000-update safety ceiling but must satisfy the convergence gate;
reaching that ceiling is a nonzero pipeline failure and no interaction/evaluation stage is
then allowed to run.

To stop after a stage:

```bash
python scripts/run_pipeline.py --stop-after additive
python scripts/run_pipeline.py --stop-after rank
python scripts/run_pipeline.py --stop-after evaluation
```

`--profile smoke` uses tiny panels, starts with a maximum basis rank of 4, and relaxes
statistical fit gates solely to exercise every model-pipeline code path. The convex solve
may reduce the active rank. Smoke still writes the population-tail and promotion-policy
reports but does not fail the integration test when the tail report rejects the
deliberately undertrained model. Smoke output is not statistically valid and must never be
used for reporting results.

## Outputs and logs

| Path | Meaning |
|---|---|
| `artifacts/pipeline.log` | complete console log when invoked with `tee` as above |
| `out/v3_pipeline_additive.log` | exact additive optimizer log |
| `out/v3_pipeline_additive_best.pt` | best additive parent |
| `artifacts/interaction_basis_rank*.{npz,json}` | rank audits; rejected ranks remain auditable |
| `artifacts/candidate.{pt,json}` | interaction candidate before the household-size block update |
| `artifacts/candidate_rank1.{pt,json}` | final rank-one household-size candidate and cross-fit/safety audit |
| `reports/likelihood_{validation,test}.json` | paired, complete-support likelihood |
| `reports/recommendation.json` | locked exact add-one MRR, MRR@5/10/20 and Recall@5/10/20; no normalizer is used |
| `reports/generation_counterfactual.json` | SMC validity and price response |
| `reports/customer_segments.json` | segment structure, generation, and price response |
| `reports/population_size.json` | full-population size/tail certification |
| `reports/segment_promotion_mdp.json` | three-segment finite-horizon promotion policy and action-response audit |
| `reports/interaction_embedding_audit.{json,md}` | invariant Gram scores, held-out co-incidence audit and complement candidates |
| `reports/baselines_converged_bernoulli_dpp_ndpp.json` | locked paired external-baseline likelihood comparison and convergence certificates |

The population audit checkpoints its screen under
`reports/population_size.screen-<digest>.*`. An interruption resumes at the last durable
boundary. If the cheap signed rule is invalid for an individual context, only that context
is escalated to the next quadrature level and the trip ID is recorded in the final report.

The last audit may reject a fitted candidate. In the full profile that rejection makes
the pipeline exit nonzero, while preserving the reports and candidate for diagnosis. It
means the fitted law is not safe for production simulation; it is not an estimator crash.
Only the smoke profile converts that statistical rejection into a successful integration
exit, because smoke is deliberately too small to certify a model.

The main command runs the complete Version-4 model fit and its declared evaluations.
Converged external competitors are intentionally a second command because they are three
additional long fits and are not required to obtain the model checkpoint. Run the
external suite from the later section when reproducing the comparison table.

## Reading the recommendation metrics

For each eligible held-out basket, one bought item is hidden. Every contemporaneously
available product is ranked by its exact conditional add-one energy. `MRR` averages
\(1/r\), where \(r\) is the hidden product's rank. `MRR@k` is \(1/r\) only when
\(r\leq k\), otherwise zero. The normalizer cancels in this conditional ranking, so MRR
does not use or tune the Smolyak level.

## Inspecting learned complements

The embedding columns may rotate without changing the model, so individual coordinates
must not be named or interpreted. The invariant pair-specific coefficient is

\[
\gamma_{ij}=\phi_i^\top\phi_j-
\rho_{c(i)}\mathbf 1\{c(i)=c(j)\}.
\]

The full energy cross-difference also contains the common size-curvature term
\(-\Delta^2\rho_0(|T|)\); it does not change pair ordering at a fixed background size.

Recreate the structural and held-out audit with:

```bash
python scripts/version4/audit_interaction_embeddings.py
```

Products are selected using the fitted training parameters only. Test baskets validate
the selected panel against a configuration null and matched controls; test outcomes never
select candidates. Positive scores are predictive complement hypotheses after the other
energy terms are held fixed. They are not causal cross-price estimates, so promotion use
still requires a randomized experiment.

## External baseline suite

The repository includes independently trainable implementations of five basket models.
For the declared external comparison, Bernoulli, DPP and NDPP are competitors; the
multinomial and exact additive laws are ablations, and SHOPPER remains available as a
separate protocol comparison.

| Baseline | Normalized basket law |
|---|---|
| Multinomial | empirical training-only \(P(n)\) times an exact distinct-item ESP composition, restricted to \(1\le n\le120\) |
| Bernoulli | exact independent-product set law restricted to \(1\le n\le120\) |
| DPP | exact non-empty low-rank determinantal law; probability beyond size 120 is bounded in the report |
| NDPP | exact non-empty nonsymmetric determinantal law; probability beyond size 120 is bounded in the report |
| SHOPPER | sequential interaction model converted to a set probability by summing small order spaces exactly and using reproducible sampled orderings otherwise; checkout is forced at 120 |

These baselines are deliberately a separate command from the main pipeline. This prevents
an ordinary model fit from silently spending compute on five competitors. Print the full
command graph first:

```bash
python scripts/run_baselines.py --dry-run
```

The default profile trains every baseline from a fresh lineage until it satisfies the
same explicit optimization contract: at least two epoch-equivalents of minibatch exposure,
trips, validation-driven learning-rate reductions, the declared learning-rate floor, and
four further stale validation checks at that floor. It then scores each validation-selected
checkpoint on the main model's locked test trips:

```bash
python scripts/run_baselines.py --profile converged \
  --models bernoulli,dpp,ndpp \
  2>&1 | tee artifacts/baselines_external_converged.log
```

The completed comparison is:

| Model | Selected / terminal update | Test nats/basket | Main-model paired gain |
|---|---:|---:|---:|
| Version-4 rank-one | -- | \(-46.064895\) | -- |
| Bernoulli | 52,000 / 56,000 | \(-48.316937\) | \(2.252042\pm0.098425\) |
| DPP | 46,500 / 52,000 | \(-48.326534\) | \(2.261638\pm0.097672\) |
| NDPP | 55,500 / 59,000 | \(-47.877142\) | \(1.812247\pm0.092393\) |

All rows use manifest SHA-256
`60e591ee6da37ad2e22a9e0ce1eb6896eac384158c5e7f96337bba465bb97caf`.
DPP and NDPP place at most \(7.1\times10^{-29}\) and \(6.7\times10^{-26}\) probability
above size 120 under their certified bounds, so their support difference is numerically
irrelevant here. The Bernoulli utility receives the same observed contextual feature
families but fresh parameters; it does not load the Version-4 checkpoint, Gram embedding,
category coefficient, size potential, or household-size correction.

The baseline implementation caches static store-assortment layouts and uses four CPU
threads for these small rank/cardinality kernels. When the Bernoulli category cap is at
least the global size cap, it evaluates
\(\prod_j(1+w_jx)\) directly instead of redundantly multiplying category polynomials:
\(\prod_c\prod_{j\in c}(1+w_jx)=\prod_j(1+w_jx)\). A unit test checks both likelihood and
every parameter gradient against the category-factorized path. These are exact runtime
refactors, not changes to a baseline objective.

The 60,000-update setting is only a safety ceiling. Reaching it without satisfying the
certificate exits nonzero and produces no test comparison. An interrupted fresh lineage
can resume without resetting its optimizer, validation state, or random streams:

```bash
python scripts/run_baselines.py --profile converged \
  --models bernoulli,dpp,ndpp --resume-training \
  2>&1 | tee -a artifacts/baselines_external_converged.log
```

The historical equal-update experiment remains reproducible with
`--profile published-1000`, but it is explicitly not a converged-model comparison. Use
`--skip-training` to rescore checkpoints belonging to the selected profile.

For a quick installation check, `--profile smoke` trains each model for two updates and
scores only the first 16 locked trips. Its relaxed certificate exercises the control flow
only and must not be reported as model quality. The converged comparison writes
`reports/baselines_converged_bernoulli_dpp_ndpp.json` plus its `_per_trip.npz` companion;
the JSON
records checkpoint hashes, selected and terminal iterations, convergence certificates,
manifest hash, likelihoods, paired standard errors, and support diagnostics.

## Reproducibility contract

- The first learned stage starts only from `artifacts/initialization.pt`.
- The interaction rank is selected using training split halves, not validation/test.
- All 5,455 product rows remain active in the interaction fit.
- Final likelihood uses support 1 through 120 and fixed identical trip panels.
- Test data never selects a checkpoint or estimator.
- A candidate is not production-certified unless every final gate passes.
