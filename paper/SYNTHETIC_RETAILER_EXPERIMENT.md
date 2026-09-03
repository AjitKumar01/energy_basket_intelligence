# Complete synthetic retailer experiment

Status: **executed, deterministic recovery and robustness audit**

Branch: `version4-synthetic-retailer-demo`

Seed: `73021` for the well-specified world and `73122` for the misspecified world

This experiment answers a question that the transaction-only dunnhumby panel cannot:

> If customer opportunities, no-purchase outcomes, randomized promotions, true basket
> interactions, quantities, costs and policy rewards were all observed, could the
> application recover them and use them coherently?

The answer is **yes for prediction, basket recovery, generation and useful policy
selection in this controlled experiment**, with an important qualification: point
estimates of incremental policy profit remain optimistic. Synthetic recovery is evidence
about implementation under known truth. It is not evidence that the same causal effects
hold in dunnhumby or at a retailer.

---

## 1. Why there are three worlds

One synthetic world generated exactly from the fitted family would be an easy but
incomplete test. The driver therefore runs three distinct checks.

1. **No-interaction exact control.** The true Gram kernel is zero. A correctly selected
   interaction model should not obtain a held-out gain merely because it has more
   parameters.
2. **Well-specified complete retailer.** The full purchase-arrival, Version-4 basket and
   shifted-negative-binomial quantity laws match the fitted families.
3. **Misspecified retailer.** The truth adds local pair/triple basket terms, nonlinear
   recency response and quadratic quantity response to discounts. These terms are absent
   from the fitted model.

The exact interaction-strength audit also includes two nonzero interaction levels. Thus
the experiment checks both false-positive control and power.

---

## 2. Complete data-generating process

The stakeholder-scale worlds contain:

| Quantity | Value |
|---|---:|
| Customers | 240 |
| Products | 20 |
| Categories | 5 |
| Customer segments | 3 |
| Stores | 3 |
| Interaction rank | 3 |
| Maximum distinct products | 6 |
| Calendar | 180 days |
| Train / validation / test | 120 / 30 / 30 days |
| Promotion actions | 7 |
| Customer-day opportunities | 43,200 |
| Enumerated nonempty baskets | 60,459 |

All offers are randomized at the **customer-day** level and their propensities are known.
This matters. Segment-day assignment would produce only 120 independent treatment draws
per segment even if 80 household rows shared every draw. Customer-level randomization
gives the action model genuine independent support.

### 2.1 Purchase arrival

For household $h$ on day $t$,

\[
A_{ht}\sim\operatorname{Bernoulli}(v_{ht}),
\tag{SR.1}
\]

\[
\operatorname{logit}(v_{ht})
=a_{g(h)}+u_h+s_1(t)+s_2(t)
+\gamma R_{ht}+\tau_{g(h),d_{ht}},
\tag{SR.2}
\]

where $R_{ht}$ is capped recency and $d_{ht}$ is the randomized promotion action. The
misspecified world additionally contains a recency-season interaction and quadratic
recency term. The fitted hazard does not contain those nonlinear terms.

The fitted hazard includes household effects, segment-action cells, calendar terms and
recency. Its ridge constant is selected only on the validation period and then refitted on
train plus validation before the locked test.

### 2.2 Store choice

Conditional on $A_{ht}=1$, a three-store categorical law depending on segment generates
the store. The fitted model uses a smoothed training-only segment-store table.

### 2.3 Basket incidence

Conditional on a purchasing trip, the well-specified world draws

\[
p(S\mid g,d,A=1)
=\frac{\exp E(S;g,d)}
{\sum_{1\leq |T|\leq6}\exp E(T;g,d)},
\tag{SR.3}
\]

with

\[
\begin{aligned}
E(S;g,d)
={}&\sum_{j\in S}
\left(\lambda_j+\alpha_{gj}
-\eta_j\Delta\log p_{jd}\right)\\
&+\sum_{j<k;\,j,k\in S}\phi_j^\top\phi_k
-\sum_c\rho_c{n_c(S)\choose2}
-\rho_0(|S|).
\end{aligned}
\tag{SR.4}
\]

The misspecified world adds a catalog-relative local term

\[
0.45I_{0}I_{1}-0.35I_{2}I_{3}
+0.40I_{0}I_{\lfloor J/3\rfloor}I_{\lfloor2J/3\rfloor},
\tag{SR.5}
\]

which includes a third-order effect outside the fitted pairwise Version-4 family.

Every basket in Eq. (SR.3) is enumerated. Consequently, the synthetic likelihood,
gradient, generator and counterfactual oracle contain **no quadrature, QMC or SMC error**.

### 2.4 Quantities

For each included product,

\[
Q_{jht}=1+X_{jht},
\qquad
X_{jht}\sim\operatorname{NB}(r_j,\mu_{jht}),
\tag{SR.6}
\]

\[
\log\mu_{jht}
=\kappa_j+\delta_{g(h)}+\beta_q d_{jht}.
\tag{SR.7}
\]

The misspecified world adds a quadratic discount term. The fitted quantity model retains
only Eq. (SR.7).

### 2.5 Retail economics and policy

Every SKU has a known list price and unit cost. For one customer opportunity,

\[
\Pi(a)
=v(a)\sum_jP(j\in S\mid a)\,
\mathbb E[Q_j\mid j\in S,a]\,[p_j(a)-c_j].
\tag{SR.8}
\]

The 28-day policy chooses no promotion or one segment-bundle discount per day. A discrete
budget dynamic program has complexity

\[
O(TBA),
\tag{SR.9}
\]

where $T=28$, $B=240$ budget bins and $A=19$ candidate segment-actions. Positive
costs are rounded upward, so the chosen policy cannot exceed its predicted budget.

---

## 3. Exact certification sandbox

The independent small audit enumerates 3,472 baskets for 14 products through size 5.
Each row below is a separate fresh fit on 10,000 locked test baskets.

| True interaction scale | Replicate | Interaction minus additive | Gram correlation |
|---:|---:|---:|---:|
| 0 | 0 | $-0.00114\pm0.00046$ | N/A |
| 0 | 1 | $-0.00152\pm0.00051$ | N/A |
| 0.35 | 0 | $+0.00272\pm0.00095$ | 0.795 |
| 0.35 | 1 | $+0.00457\pm0.00121$ | 0.863 |
| 0.70 | 0 | $+0.03214\pm0.00274$ | 0.979 |
| 0.70 | 1 | $+0.04768\pm0.00324$ | 0.978 |

The zero-interaction controls do not manufacture a positive held-out interaction gain.
As signal increases, both likelihood gain and orientation-invariant kernel recovery rise.

---

## 4. Well-specified complete-retailer result

The simulation produced 5,215 trips from 43,200 opportunities, for a purchase rate of
0.12072. The locked test contains 7,200 opportunities and 895 trips.

### 4.1 Arrival and store

| Metric | Value |
|---|---:|
| Arrival log loss | 0.36398 |
| Brier score | 0.10623 |
| Expected calibration error | 0.00559 |
| Probability MAE to oracle | 0.01884 |
| Probability correlation to oracle | 0.8930 |
| Store accuracy | 0.68045 |

Validation selected logistic ridge $C=0.1$.

### 4.2 Basket likelihood and interactions

| Model | Test nats/trip |
|---|---:|
| Oracle | $-8.19381$ |
| Exact additive | $-8.44672$ |
| Fitted interaction | $-8.21462$ |

The paired interaction gain is

\[
0.23210\pm0.02794,
\qquad
95\%\ \mathrm{CI}=[0.17734,0.28685].
\tag{SR.10}
\]

The learned Gram-kernel correlation is $0.9764$. Individual price-elasticity
correlation is only $0.4032$, while action-level counterfactual incidence is much more
accurate. This distinction is important: bundle-randomized data identifies bundle
response more strongly than every SKU's separate elasticity.

### 4.3 Recommendation and generation

| Metric | Additive | Interaction |
|---|---:|---:|
| MRR | 0.3212 | 0.3379 |
| Recall@5 | 0.5240 | 0.5508 |
| Recall@10 | 0.7721 | 0.8000 |

Generation on identical test contexts gives:

| Statistic | Observed | Generated |
|---|---:|---:|
| Mean size | 3.4212 | 3.3810 |
| Size variance | 1.7094 | 1.5806 |

Size total variation is $0.03464$, size Jensen--Shannon divergence is $0.00165$, and
item-incidence RMSE is $0.02038$.

### 4.4 Quantities and counterfactuals

Observed/predicted mean quantity is $1.3615/1.3700$. The true/fitted discount
coefficient is $2.15/2.0621$, and true/fitted dispersion is $2.6/2.4267$.

Across all 21 segment-action contexts:

| Counterfactual error | Value |
|---|---:|
| Mean basket-size MAE | 0.02801 products |
| Item-incidence MAE | 0.01056 |
| Arrival-probability MAE | 0.01938 |
| Maximum expected-size error | 0.10228 products |

### 4.5 Policy result

Under the known oracle, the fitted 28-day policy earns $160.389$ incremental profit,
versus the oracle optimum $177.985$:

\[
\frac{160.389}{177.985}=0.9011.
\tag{SR.11}
\]

It therefore captures 90.1% of oracle value with zero budget overspend. However, its own
point estimate is $398.697$, which is materially optimistic. Action-value correlation is
only $0.519$. The correct interpretation is:

- the policy ranking is useful in this synthetic world;
- the raw profit estimate is not calibrated enough for autonomous deployment;
- lower-confidence-bound selection and a real randomized pilot remain necessary.

---

## 5. Misspecified-world result

The misspecified world produces 5,116 trips and 830 locked test trips.

| Metric | Result |
|---|---:|
| Interaction minus additive | $+0.11631\pm0.01968$ nats/trip |
| Gram correlation | 0.9313 |
| Additive / interaction MRR | 0.3365 / 0.3490 |
| Observed / generated mean size | 3.1819 / 3.3084 |
| Size total variation | 0.07108 |
| Basket-size counterfactual MAE | 0.01494 |
| Arrival-probability MAE | 0.02001 |
| Oracle realized / oracle-optimal policy profit | 140.094 / 178.306 |
| Fraction of oracle policy value | 0.7857 |
| Budget overspend | 0 |

The model retains 78.6% of oracle policy value under mild misspecification, but again
overstates point profit. The generation and policy degradation relative to the
well-specified world are visible rather than hidden.

---

## 6. Cross-world results and statistical interpretation

The table below separates recovery under the fitted family from robustness when the
truth contains omitted structure.

| Quantity | Well-specified | Misspecified | Interpretation |
|---|---:|---:|---|
| Opportunities | 43,200 | 43,200 | Identical observation budget |
| Test trips | 895 | 830 | Purchase incidence differs through the simulated truth |
| Interaction gain (nats/trip) | $0.23210\pm0.02794$ | $0.11631\pm0.01968$ | Positive paired held-out gain in both worlds |
| Lower endpoint of 95% interval | 0.17734 | 0.07773 | Neither gain is explained by test-trip variation alone |
| Gram-kernel correlation | 0.9764 | 0.9313 | Pairwise interaction geometry remains recoverable |
| Interaction MRR | 0.3379 | 0.3490 | Recommendation is conditional and is not directly comparable across worlds |
| Interaction minus additive MRR | 0.01677 | 0.01244 | Interactions improve hidden-item ranking |
| Generation size TV | 0.03464 | 0.07108 | Omitted structure visibly worsens generation |
| Counterfactual size MAE | 0.02801 | 0.01494 | Aggregate size can remain accurate despite local misspecification |
| Policy fraction of oracle value | 0.9011 | 0.7857 | Policy is more sensitive than predictive averages |

The likelihood uncertainty is paired. If $S_i$ is locked test basket $i$, define

\[
D_i=\log p_{\mathrm{int}}(S_i\mid x_i)
-\log p_{\mathrm{add}}(S_i\mid x_i).
\tag{SR.12}
\]

The reported standard error is

\[
\operatorname{se}(\bar D)
=\frac{\operatorname{sd}(D_1,\ldots,D_m)}{\sqrt m}.
\tag{SR.13}
\]

This is more informative than subtracting two independent marginal standard errors,
because both models score exactly the same test baskets. In the well-specified world the
95% interval is $[0.17734,0.28685]$ nats/trip; in the misspecified world it is
$[0.07773,0.15490]$. The interaction improvement is therefore statistically resolved in
both experiments. The exact zero-interaction controls in Section 3 are the complementary
specificity check: when the data-generating kernel is zero, the more flexible model does
not obtain a positive held-out gain.

The kernel comparison uses $K=\Phi\Phi^\top$, not the coordinates of $\Phi$. This is
necessary because replacing $\Phi$ by $\Phi R$ for any orthogonal matrix $R$ leaves all
basket energies unchanged. A coordinate-wise embedding correlation would incorrectly
penalize an equivalent rotation; the Gram correlation measures the identifiable object.

The recommendation result answers a narrower question than likelihood. For every test
basket one product is hidden and all eligible products are ranked by their conditional
add-one energy. The partition function cancels in this ranking. Consequently, higher MRR
shows that the learned interaction geometry helps conditional completion, while the
likelihood result additionally certifies normalized probability over the complete
basket support.

Generation is evaluated on the same held-out contexts rather than by drawing arbitrary
segments or prices. The total-variation and Jensen--Shannon statistics compare the full
empirical and generated basket-size distributions; item-incidence RMSE compares all 20
marginal product probabilities. These are distributional checks, not just visual samples.

## 7. Why policy-value calibration remains a separate problem

The policy layer consumes several learned objects at once. For segment $g$ and action
$a$, its one-opportunity value estimate is

\[
\widehat V(g,a)
=\widehat v(g,a)
\sum_j \widehat\pi_j(g,a)
\widehat q_j(g,a)\,[p_j(a)-c_j],
\tag{SR.14}
\]

where $\hat v$ is purchase-arrival probability, $\hat\pi_j$ is basket incidence and
$\hat q_j$ is expected purchased quantity. Small errors in these three factors can align
and become a much larger error in incremental profit

\[
\widehat\Delta V(g,a)=\widehat V(g,a)-\widehat V(g,0).
\tag{SR.15}
\]

Subtraction makes the problem harder: the incremental signal can be much smaller than
either absolute value. Maximization then introduces a winner's-curse effect because the
budget solver deliberately selects actions with the largest estimated gains.

The observed calibration evidence is:

| Policy quantity | Well-specified | Misspecified |
|---|---:|---:|
| Estimated value of selected policy | 398.697 | 411.460 |
| Oracle value of selected policy | 160.389 | 140.094 |
| Oracle-optimal value | 177.985 | 178.306 |
| Optimism factor, estimate / realized | 2.486 | 2.937 |
| Action-value correlation | 0.519 | 0.448 |
| Action-value MAE | 6.062 | 4.610 |
| Oracle regret | 17.596 | 38.212 |
| Budget violation | 0 | 0 |

Thus two statements coexist:

1. the selected policy has positive oracle value and respects the budget; and
2. its own numerical profit forecast is not calibrated.

The second problem is **not automatically solved by PPO, SAC or another policy optimizer**.
Those algorithms optimize rewards supplied by an environment. If the learned environment
overstates a promotion effect, a more powerful optimizer can exploit that error more
aggressively. In the present experiment the state transition is deliberately simple and
daily rewards are conditionally independent apart from the remaining budget. The finite
dynamic program therefore solves the declared optimization problem exactly; replacing it
with PPO or SAC would add approximation error without repairing the value model.

RL becomes appropriate after the retailer model contains genuine sequential state, for
example inventory depletion, promotion carryover, changing customer recency, stockouts,
competitor response or long-run retention. Even then, causal value calibration remains a
prerequisite. The appropriate development order is:

1. estimate action effects from logged randomized propensities;
2. use held-out inverse-propensity or doubly robust policy-value estimates;
3. propagate uncertainty and optimize a lower confidence bound rather than a raw mean;
4. validate the shortlisted policy in a controlled retailer experiment; and
5. introduce constrained offline RL or model-based control only when sequential effects
   are present and identifiable.

For this synthetic audit, oracle regret is therefore the decision-quality metric and the
uncalibrated point estimate is reported as a diagnostic. For real deployment, a policy
must not be advertised with a monetary return until randomized or otherwise causally
identified data calibrates that return.

## 8. Time and memory complexity

Let

\[
M=\sum_{n=1}^{n_{\max}}{J\choose n}
\tag{SR.16}
\]

be the enumerated support and $K=GA$ the segment-action contexts. One exact basket
likelihood evaluation costs

\[
O\!\left(MJ(K+r)\right),
\tag{SR.17}
\]

with storage $O(MJ+KM)$. For $J=20$, $n_{\max}=6$, $M=60{,}459$ and $K=21$, so exact
enumeration is practical. It would not be practical at 5,455 products; the production
estimator remains necessary there.

Two final full invocations completed all exact controls and both retailer worlds in
20.17 and 25.28 seconds on the audited CPU machine. The scientific outputs were
identical; the wall-clock difference reflects ordinary machine load. The experiment is
intentionally cheap enough to rerun in continuous integration or a stakeholder
demonstration.

---

## 9. Independent synthetic pipeline and reproduction

The synthetic pipeline is intentionally independent of `scripts/run_pipeline.py`. It
does not read raw dunnhumby data, load a production checkpoint, reuse a real-data split or
write into the real pipeline's checkpoint lineage. Its seeds, support, actions, splits and
output reports are declared by `scripts/run_synthetic_experiment.py`.

Run the complete experiment:

```bash
python scripts/run_synthetic_experiment.py --profile full --threads 8 \
  2>&1 | tee artifacts/synthetic_experiment_full.log
```

Run the software-path check:

```bash
python scripts/run_synthetic_experiment.py --profile smoke --threads 4
```

Machine-readable outputs are written to the ignored `reports/` directory:

- `synthetic_exact_certification.json`;
- `synthetic_retailer_experiment.json`;
- `synthetic_retailer_misspecified.json`; and
- `synthetic_experiment_manifest.json`.

The source entry points are:

- `scripts/version4/audit_synthetic_interactions.py`;
- `scripts/version4/audit_synthetic_retailer.py`; and
- `scripts/run_synthetic_experiment.py`.

---

## 10. Decision

The experiment establishes that the application can, under observed opportunities and
randomized offers:

- learn purchase/no-purchase propensity;
- recover a low-rank basket-interaction kernel;
- improve full joint likelihood and recommendation;
- generate calibrated baskets;
- recover quantity response;
- answer one-step counterfactuals; and
- choose a budget-feasible policy with useful oracle value.

It simultaneously demonstrates why synthetic success is not a production certificate:
policy point values are optimistic, item-level elasticity is weakly identified by bundle
variation, and misspecification degrades generation and policy performance. Those are
precisely the uncertainties that real randomized experiments must resolve.
