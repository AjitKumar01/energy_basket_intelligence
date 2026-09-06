# Synthetic certification of the complete-demand extension

Status: **all predeclared synthetic checks passed**

Date: 6 September 2026

## 1. Question tested

The experiment tests whether the Version-4 partition function can link store visitation,
purchase activation, and basket choice inside one joint likelihood while preserving:

- normalization over no visit, empty visits, and nonempty baskets;
- the original conditional Version-4 basket law;
- interaction recovery when the signal is identified;
- rejection of unsupported interaction capacity;
- extensive-margin price response; and
- polynomial catalogue scaling.

The synthetic evidence is an implementation test under known truth. It is not real-data
evidence and does not establish causal price effects.

## 2. Experimental design

Every full replicate contains:

| Quantity | Value |
|---|---:|
| Households | 180 |
| Days per household | 150 |
| Household-day opportunities | 27,000 |
| Products | 12 |
| Stores | 3 |
| Segments | 3 |
| Price actions per store | 4 |
| Interaction rank | 2 |
| Maximum basket size | 4 |
| Enumerated nonempty baskets | 793 |
| Training / validation / test days | 95 / 25 / 30 |

Exact enumeration is used only in this small world so every probability and likelihood
has an oracle. A separate polynomial dynamic program is benchmarked through 10,000
products.

Three model fits are compared:

1. **Linked additive:** joint visit/store/purchase/basket law with zero Gram interaction.
2. **Linked interaction:** the same joint law with the rank-2 interaction block.
3. **Independent hurdle:** visit and purchase heads do not receive the basket inclusive
   value; conditional basket fitting remains available.

The experiment matrix contains three strong-interaction seeds, one weak-interaction
world, and one exactly null-interaction world. Interaction acceptance requires a positive
paired 95% validation lower bound. Rejected children revert exactly to the linked additive
parent.

## 3. Algebraic and unit-level verification

The automated tests verify:

1. probabilities sum to one over the full outcome space;
2. conditioning on store and purchase recovers Version-4 exactly;
3. \(\partial\log Z_+/\partial b_j\) equals conditional item incidence;
4. reported observed probabilities equal the explicit joint formula;
5. no-visit observations send gradients into basket parameters only in the linked model;
6. discounts change visit probability in the linked model but not the independent visit
   head;
7. the ESP dynamic program equals exhaustive subset summation;
8. the affinity-category dynamic program equals exhaustive subset summation;
9. simulated data contain no visits, empty visits, and nonempty purchases; and
10. recurrence state memory does not grow with catalogue size for fixed \(n_{\max}\).

All 11 complete-demand unit tests pass.

## 4. Strong-interaction recovery

| Seed | Validation lower 95% | Test interaction gain | Test lower 95% | Kernel correlation |
|---:|---:|---:|---:|---:|
| 104017 | +0.000110 | +0.004308 | +0.001322 | 0.99950 |
| 104018 | +0.003244 | +0.009494 | +0.005758 | 0.99911 |
| 104019 | +0.003487 | +0.004385 | +0.001265 | 0.99874 |

All gains are nats per household-day opportunity. Across the three seeds:

\[
\text{mean test interaction gain}=0.006062
\quad\text{nats/opportunity},
\]

and mean recovered-kernel correlation is \(0.99912\). Every strong child passed the
validation gate before test evaluation.

For seed 104017, the true interaction eigen-scales were \((2.34,1.32)\) and the fitted
values were \((2.131,1.117)\). The model therefore recovered both the interaction geometry
and useful magnitude rather than merely producing a favorable likelihood from unrelated
parameters.

## 5. Weak and null safeguards

| World | Validation lower 95% | Accepted? | Selected test gain |
|---|---:|:---:|---:|
| Weak interaction, seed 105017 | -0.000838 | No | 0 after fallback |
| Null interaction, seed 106017 | -0.000167 | No | 0 after fallback |

The optimizer can propose nonzero interaction scales even under the null. The validation
gate prevents those noisy parameters from entering the selected model. This is why raw
parameter magnitude is not an acceptance criterion.

The weak world deliberately measures the finite-sample detection boundary. Rejecting it
is the correct result: the experiment does not claim identifiable interaction improvement
when the lower confidence bound crosses zero.

## 6. Linked model versus independent hurdle

In every one of the five worlds, the linked selected model beat the independent hurdle at
the 95% level. For the three strong seeds, the linked-minus-independent test gains were:

| Seed | Gain | 95% interval |
|---:|---:|---:|
| 104017 | +0.004275 | [0.001344, 0.007205] |
| 104018 | +0.005825 | [0.003109, 0.008541] |
| 104019 | +0.008403 | [0.005357, 0.011449] |

The mean gain was \(0.006168\) nats per opportunity. This comparison isolates the
inclusive-value link: both competitors retain a conditional basket law, but only the
linked model allows assortment and price changes to affect visitation and purchase
activation through \(Z_+\).

## 7. Counterfactual demand

In the oracle, moving every store from no discount to the largest tested action changed
both visit and purchase probability. The independent hurdle predicted exactly zero change
in those two margins because price never enters its arrival heads.

Across the three strong seeds:

| Counterfactual error | Linked model | Independent hurdle |
|---|---:|---:|
| Mean visit-probability MAE | 0.00337 | 0.01322 |
| Mean expected-distinct-items MAE | 0.00847 | 0.03065 |

The linked expected-demand error was about 29% of the independent error. It did not
recover every magnitude perfectly, but it recovered the missing extensive-margin
direction and materially improved total-demand accuracy.

For seed 104017, the oracle no-discount-to-largest-action changes were:

\[
\Delta P(\text{visit})=0.03724,
\qquad
\Delta P(\text{purchase})=0.04406.
\]

The fitted linked model recovered \(0.03326\) and \(0.03948\); the independent model
returned zero for both.

## 8. Unconditional item demand

For seed 104017, fitted item-level unconditional probabilities had:

\[
\operatorname{MAE}=0.000879,
\qquad
\operatorname{corr}=0.99621
\]

against the oracle. Across strong seeds, the correlations were 0.99621, 0.99136, and
0.97949. These quantities incorporate no visit, store choice, purchase activation, and
conditional basket inclusion.

## 9. Scalability audit

The small-world fit enumerates baskets only for exact certification. Two non-enumerating
recurrences were benchmarked independently.

### 9.1 Basic ESP recurrence

With \(n_{\max}=120\), the measured log--log time slope over the three largest catalogues
was \(0.992\), consistent with linear dependence on product count for fixed size support.

### 9.2 Affinity-category recurrence

| Products | Categories | \(n_{\max}\) | Median seconds | Coefficient workspace |
|---:|---:|---:|---:|---:|
| 50 | 10 | 50 | 0.00056 | 1,632 bytes |
| 200 | 40 | 120 | 0.00545 | 3,872 bytes |
| 1,000 | 200 | 120 | 0.03628 | 3,872 bytes |
| 5,455 | 300 | 120 | 0.06968 | 3,872 bytes |
| 10,000 | 300 | 120 | 0.09220 | 3,872 bytes |

These timings belong to a standalone Python/NumPy log-domain reference recurrence on the
current machine. They demonstrate scaling shape and exactness, not production throughput.
The optimized real pipeline uses a native custom-adjoint implementation.
The table reports only temporary coefficient workspace. Input product weights and
category labels require additional \(O(J)\) storage, while model parameters require their
own storage.

At rank 5, the current Smolyak screen, reported, and audit rules contain 11, 71, and 341
nodes. Hence a reported full interaction likelihood repeats related partition work 71
times per store context. The visit extension adds store contexts but does not change this
rank-dependent multiplier.

## 10. Predeclared decision

All eight experiment-level checks passed:

- all three strong interactions were accepted;
- all three strong test gains had positive 95% lower bounds;
- all three strong kernels had correlation above 0.95;
- the weak interaction was rejected as underpowered;
- the null interaction was rejected;
- linked likelihood beat the independent hurdle in every world;
- all joint distributions normalized within (10^{-12}); and
- linked counterfactual item demand beat the independent hurdle in every world.

The frozen-code certification matrix used fresh seeds 104017--106017 after the design and
gates were fixed; earlier development seeds are not included in the headline evidence.
The full five-world matrix completed in approximately 28.9 seconds on the recorded local
machine. Raw generated reports are written under `reports/` and are intentionally ignored
by Git; the reproducible commands and immutable summarized findings are retained here.

## 11. Conclusion and boundary

The synthetic experiment supports the proposed theory:

1. the inclusive-value construction is a normalized joint probability law;
2. it preserves Version-4 conditional basket probabilities exactly;
3. visit and no-visit outcomes inform basket parameters through the joint normalizer;
4. identifiable interactions can be recovered and improve held-out joint likelihood;
5. weak and null interactions can be rejected safely;
6. linking \(Z_+\) is necessary to recover price-driven extensive-margin demand; and
7. catalogue dependence remains polynomial, with interaction rank—not basket count—as
   the main certification limit.

This does not yet justify fitting the model to a retailer without the opportunity,
no-purchase, feasible-store, availability, and unchosen-store price data listed in
[`COMPLETE_DEMAND_MODEL.md`](COMPLETE_DEMAND_MODEL.md). It also does not make
observational price effects causal.
