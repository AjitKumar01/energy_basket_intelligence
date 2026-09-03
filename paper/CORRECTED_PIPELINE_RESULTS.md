# Corrected end-to-end pipeline: measured results

Status: **historical parent fit; production certification rejected for this checkpoint**
Date: 2026-08-31

This is the immutable empirical report for the first corrected Version-4 parent fit. It
reports one fresh execution of the pipeline defined in [PIPELINE.md](PIPELINE.md), after the raw
data and preprocessing audit described in
[PREPROCESSING_AUDIT.md](PREPROCESSING_AUDIT.md). The basket law in
[THEORY.md](THEORY.md) was not changed.

This checkpoint's rejection is retained because it motivated the identified household-size
correction. It is not the current pipeline decision. The successor subsequently completed
from fresh initialization and passed the declared population gates; its canonical result
is [RANK1_PIPELINE_RESULTS.md](RANK1_PIPELINE_RESULTS.md).

The execution reached every stage. Its final nonzero exit was an intentional fail-closed
decision by the population safety gate, not a process abort and not a quadrature error.
The fitted checkpoint is retained as a research candidate.

## 1. Data contract

The cohort is defined without held-out leakage:

| Quantity | Corrected value |
|---|---:|
| Products | 5,455 |
| Households | 1,920 |
| Stores | 115 |
| Affinity groups | 300 |
| Train, weeks 9--82 | 160,007 baskets; 1,223,933 lines |
| Validation, weeks 83--90 | 17,351 baskets; 129,731 lines |
| Test, weeks 91--101 | 23,340 baskets; 181,342 lines |
| Total | 200,698 baskets; 1,535,006 lines |

The product catalogue and household cohort are selected with training weeks only. No
validation or test purchase is deleted after cohort selection. Because the raw source has
no stock feed, the declared support is the complete 5,455-product catalogue at all 115
stores and all basket sizes \(1,\ldots,120\).

## 2. Exact additive fit

The run started from a fresh initialization; it did not load learned parameters from an
older experiment. With \(\Phi=0\), the category/cardinality recursion evaluates the
partition function and gradients exactly.

On its fixed 1,024-trip validation panel, the log likelihood moved from

\[
-49.622960
\quad\text{to a best of}\quad
-44.748944\ \text{nats/basket}.
\]

The best checkpoint occurred at update 13,500. Validation-driven learning-rate reduction
continued until the declared convergence condition was reached at update 14,700. Runtime
was 6,387.7 seconds. The safety ceiling of 30,000 updates was not reached.

## 3. Interaction rank and natural-parameter solve

At the additive parent, the ordinary score in a factor \(\Phi\) is zero because the Gram
energy is quadratic in \(\Phi\). The pipeline therefore estimates the pair-statistic score,
tests its leading eigenspaces on independent training halves, and accepts the largest rank
whose mean squared overlap is at least \(0.5\).

The rank-8 audit selected rank 5:

| Candidate rank | Split-half mean squared overlap | Accepted |
|---:|---:|:---:|
| 4 | 0.499320 | no |
| 5 | 0.549018 | yes |
| 6 | 0.485841 | no |
| 7 | 0.487540 | no |
| 8 | 0.461541 | no |

In that basis, \(K=UCU^\top\) was fitted with \(0\preceq C\preceq I\) on 12,000
contexts and 64 fixed exact draws per context. The selected ridge was \(3\times10^{-4}\).
The swapped-half gains were \(0.024789\) and \(0.023863\) nats/basket; the mean was
\(0.024326\). Median importance-sampling ESS was \(63.878/64=0.9981\), and the first
percentile was \(59.446\). The full-data sampled gain was

\[
0.024517\pm0.001085\ \text{nats/basket}.
\]

The fixed-draw model size moments on this fit panel were \(7.7676\) and \(84.9713\),
versus observed \(7.7738\) and \(84.8368\). All five eigenvalues of \(C\) reached the
declared spectral cap \(1\). This is an active model-capacity boundary and must be reported;
raising it is not justified until the local large-basket failure below is controlled.

## 4. Locked complete-support likelihood

Every row below uses the same fixed trip panel for parent and child. The uncertainty for
the interaction claim is the standard error of the *paired difference*, not the much
larger cross-trip standard error of either absolute mean.

| Split | Trips | Exact additive parent | Rank-5 child | Paired child gain | 95% interval |
|---|---:|---:|---:|---:|---:|
| Validation | 4,096 | \(-43.702634\pm0.674506\) | \(-43.681003\pm0.673908\) | \(+0.021630\pm0.001581\) | \([0.018532,0.024729]\) |
| Test | 4,096 | \(-46.085337\pm0.699393\) | \(-46.061429\pm0.698896\) | \(+0.023908\pm0.001647\) | \([0.020679,0.027137]\) |

The higher-level quadrature audit bounded target-rule error by \(0.000510\) nats on
validation and \(0.000720\) nats on test, both below the declared \(0.01\)-nat tolerance.
Even after subtracting that numerical allowance, the lower confidence bounds on the
interaction gains remain positive. Thus the corrected fit establishes a statistically
positive held-out likelihood contribution from the Gram interaction over its matched
exact additive parent.

These numbers do **not** by themselves re-certify historical external baselines trained
on earlier preprocessing. Such a comparison requires fresh baseline convergence on this
same corrected cohort and locked trip manifests.

## 5. Recommendation

Recommendation hides one item in each eligible test basket and ranks all products by the
exact add-one conditional energy. The partition function cancels, so this evaluation uses
no Smolyak approximation and no recommendation-specific training objective.

On 1,615 valid cases:

| Metric | Full interaction model |
|---|---:|
| MRR | \(0.095144\pm0.006083\) |
| MRR@5 | \(0.082136\) |
| MRR@10 | \(0.087112\) |
| MRR@20 | \(0.090338\) |
| Recall@5 | \(0.124458\) |
| Recall@10 | \(0.162229\) |
| Recall@20 | \(0.209907\) |

The matched additive parent has MRR \(0.094897\). The paired interaction increment is

\[
+0.000247\pm0.000372,
\qquad 95\%\ \mathrm{CI}=[-0.000481,0.000976],
\]

which is not statistically established. Interactions reduce mean rank by 2.79 positions,
but the present test does not support a positive MRR claim. The strong total MRR and the
significant likelihood gain are distinct facts; one does not mathematically imply the
other because likelihood scores the entire basket distribution while MRR tests one
conditional ranking functional.

## 6. Generation, price counterfactuals and segments

The sequential Monte Carlo generator used 64 particles per context. Minimum normalized
ESS was \(0.99945\); no generated basket contained a duplicate or an unavailable product.
On the 64-context generation panel, 4,096 generated baskets had mean size \(7.3899\) and
variance \(70.2408\), versus observed mean \(10.0313\) and variance \(136.2803\). Category
total variation was \(0.2001\). This panel is under-dispersed, although the more serious
population failure is localized upper-tail mass rather than excessive aggregate variance.

Uniform price counterfactuals have the required direction:

| Price multiplier | Own incidence retained | Change in expected basket size |
|---:|---:|---:|
| 0.8 | 1.06272 | +0.33752 |
| 0.9 | 1.02821 | +0.15396 |
| 1.0 | 1.00000 | 0 |
| 1.1 | 0.97654 | -0.13152 |
| 1.2 | 0.95674 | -0.24560 |

Training-only household surfaces select three stable segments. The labels summarize their
held-out category over-indexing: soft drinks/tropical fruit, refrigerated/salad bar, and
fluid milk/isotonic drinks. Segment-specific generation, KL/JS/TV comparisons and price
responses are stored in `reports/customer_segments.json`; segment choice uses no test
outcomes.

## 7. Full-population safety gate

The screening pass covered all 160,007 training contexts. No context required escalation
because of an invalid signed rule. At the aggregate level:

\[
\widehat P_{\rm obs}(N\ge60)=0.001875,
\qquad
\widehat P_{\rm model,q7}(N\ge60)=0.002633,
\]

with a 95% upper bound \(0.002745\), below the allowed \(0.004250\). Therefore the
aggregate tail-rate gate passes.

The localized conditional-tail gate fails. In the 384 highest-risk contexts, the more
accurate rule gives model mean size \(41.50\) versus observed \(21.80\), and tail probability
\(P(N\ge60)=0.22025\) versus observed frequency \(0.02083\). Twelve contexts assign at least
half of their mass to \(N\ge60\); the worst such probability for a context whose observed
size is below 40 is \(0.73837\). This violates the declared safety condition.

The cheap/high rule discrepancy is also material in that risk panel: mean absolute
expected-size gap \(1.95\), maximum gap \(9.27\). The safety decision therefore uses the
higher rule and a conservative error envelope, not the cheap screen alone.

## 8. Verdict

The corrected pipeline demonstrates all of the following:

- reproducible fresh additive convergence;
- stable rank-5 interaction directions covering all 5,455 products;
- a numerically certified and statistically positive held-out likelihood gain over the
  exact additive parent;
- valid recommendation and generation mechanics; and
- correctly directed price counterfactuals.

It does not establish a significant interaction MRR improvement, and it is not safe for
production basket generation or retailer policy simulation because of localized extreme
size phases. The next model-development task is therefore to identify and constrain the
context directions responsible for that phase *inside the existing Version-4 size and
interaction parameterization*, followed by a fresh end-to-end fit and the same locked
gates. It is not valid to certify this checkpoint by averaging away those contexts.

Primary machine-readable evidence:

- `out/v3_pipeline_additive_history.json`
- `artifacts/interaction_basis_rank8.json`
- `artifacts/candidate.json`
- `reports/likelihood_validation.json`
- `reports/likelihood_test.json`
- `reports/recommendation.json`
- `reports/generation_counterfactual.json`
- `reports/customer_segments.json`
- `reports/population_size.json`
- `artifacts/pipeline_corrected_full.log`
