# Retail Application Evaluation Audit

## Executive conclusion

The joint basket law now answers the correct cart-conditional query `P(T | A ⊆ S, x)` directly. Revealed items are forced into the basket; the unobserved completion may be empty, so stopping is a proper probability rather than a heuristic. Exact enumeration certifies the transformation on synthetic catalogues, and bridge ESS diagnoses the real-data Monte Carlo calculation.

The synthetic certification passed: exact forced-completion score error 1.78e-15, SMC stopping error 1.38e-05, maximum incidence error 0.0174, and minimum ESS 0.999999. The new informative-mask quadrature matched enumerated stopping, size, item incidence, and normalization to at worst 7.22e-16.

On real data, the corrected adjacent deterministic rules used 71 and 341 nodes, with a 1341-node follow-up for 6 contexts. The final maximum stop-probability and item-incidence gaps were 4.3e-07 and 5.31e-06; the independent SMC-versus-quadrature mean absolute stop gap was 3.97e-06.

The corrected natural-panel audit no longer supports the earlier conclusion that basket completion size fails. That conclusion came from a balanced tail panel and an unmodeled random-pair selection mechanism. The corrected masked-pair mean is well aligned and MAE improves on the training-only size baseline. Stopping gains are not statistically resolved, and scenario outputs must not be presented as causal effects, expected profit, or individual customer transitions.

## Results by application

| Application | Concrete retailer output | Evidence in these data | Verdict |
|---|---|---|---|
| Real-time cross-sell | Top-k products after a two-item masked cart | Masking-corrected natural panel (200 non-stop cases): MRR 0.3667 vs 0.2184 popularity; hidden-set recall@20 23.47% vs 11.54%. Locked 1,615-case test: MRR 0.0945 vs 0.0427 popularity. | Promising offline; A/B test before deployment. |
| Basket completion | Distribution over remaining products and count | Natural 256-trip panel: actual remaining mean 6.688, predicted 6.519; MAE 6.119 vs 6.867 training-size baseline. Paired MAE improvement 95% interval -1.215 to -0.219 items. | Mean calibrated; MAE improves, full distribution not yet certified. |
| Stopping | Masking-aware `P(T=∅ | A,x)` | Brier 0.1663 vs 0.1729; log loss 0.5227 vs 0.5320; AUC 0.6086. Both paired score intervals include zero. | Modest discrimination; improvement unresolved. |
| Stockout substitution | Alternatives within desired SKU's subcommodity | Held-out-choice proxy: model MRR 0.5492 vs popularity 0.3803 across 32 cases. No stockout labels. | Proxy only; actual substitution not identified. |
| Price scenarios | Basket probabilities under a declared price vector | 32 observational events: sign agreement 56.25%; child MAE 0.001022, parent MAE 0.001022; model accuracy explicitly not assessed. | Numerically usable scenario, not validated causal effect. |
| Personalized bundles | Candidate bundle odds conditional on cart | Matched-negative proxy: model MRR 0.5392 vs popularity 0.2699 across 27 cases. | Offline proxy; randomized offers needed. |
| Promotion targeting | Segment/bundle/discount scenario table | Existing MDP has 13 numerically admissible actions, but excludes profit, inventory, visits, switching, and quantities. | Do not deploy as policy. |
| Assortment planning | Recompute basket law after SKU addition/removal | No historical availability, planogram, cost, capacity, or lost-demand intervention labels. | Not evaluable with current data. |
| Demand forecasting | Conditional SKU/category incidence | Fitted mean size 6.794 vs 7.864 observed; error -1.071; item TV 0.351. | Failed factual calibration. |
| Customer segmentation | Stable descriptive audience IDs | 3 clusters; silhouette 0.151, stability ARI 0.994; segment item TV spans 0.288–0.382. | Useful for description/experiment strata, not response targeting. |

## What the conditional probability means

For a revealed cart `A`, the engine sums over every allowable completion `U`:

```text
P(T | A ⊆ S, x) = exp(score(A ∪ T)) / Σ_U exp(score(A ∪ U))
```

`U=∅` is included. Therefore `P(stop | A,x)=P(T=∅ | A ⊆ S,x)`. This is an eventual-basket completion law. The Dunnhumby transactions do not contain scan/cart order, so it is not a chronological next-item law.

A retrospective validation case is built by uniformly selecting two products from the final basket. That observation mechanism is informative about final size. Its correct evaluation law is `q(T|A,x) ∝ P(T|A⊆S,x) / C(|A|+|T|,2)`. The earlier balanced 64-case result omitted this factor and is superseded.

The earlier worked price examples remain candidate-set calculations: a 15% increase in the declared soy SKU moved dairy's two-candidate probability by 0.0354 percentage points. A 20% increase in the declared butter SKU moved its exact-rest addition probability from 0.000879 to 0.000837. These small changes are model scenarios, not evidence that individual shoppers switch.

The independent 128-particle SMC and deterministic quadrature agreed closely on stopping (mean absolute difference 3.97e-06). Expected remaining size was noisier under SMC (mean absolute difference 0.534 items; maximum 2.264). A high bridge ESS establishes stable importance weights; it does not by itself guarantee that 128 draws precisely estimate every downstream moment.

## Deployment sequence

1. Use current outputs only for candidate generation and analyst scenario review.
2. Repeat the masking-corrected audit on a larger locked panel and add calibration curves; require held-out size, Brier, log-loss, and calibration gates.
3. Log cart order, recommendation exposure, stock availability, shelf price, promotion assignment, units, costs, margin, and trip/no-trip opportunities.
4. Run randomized tests for recommendation, price, bundle, and promotion decisions.
5. Promote an application only when its own decision metric passes; a good joint likelihood or stable ESS is not a substitute for application validation.

## Reproducibility

New application evaluation: `/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/retail_application_audit_20260915/real_application_evaluation.json`
Deterministic cart-conditional audit: `/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/retail_application_audit_20260915/real_cart_quadrature_evaluation_followup.json`
Corrected basket-completion audit: `/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/retail_application_audit_20260915/real_basket_completion_corrected.json`
Corrected basket-completion log: `/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/retail_application_audit_20260915/real_basket_completion_corrected.log`
New evaluation log: `/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/retail_application_audit_20260915/real_application_evaluation.log`
Machine-readable consolidated audit: `/Users/ajit/Projects/nf_dunnhumby/energy_basket_intelligence/artifacts/retail_application_audit_20260915/consolidated_application_audit.json`
