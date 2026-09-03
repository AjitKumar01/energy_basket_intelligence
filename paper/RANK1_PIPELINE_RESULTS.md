# Certified rank-one pipeline result

## 1. Scope and lineage

This document records the completed result of the selected Version-4 pipeline on the
corrected dunnhumby cohort. The execution began from `artifacts/initialization.pt`; no
learned checkpoint from an earlier numbered run was resumed. The fitted law remains the
law in `version4.html`. The only structural reparameterization assigns one existing
household/product utility direction to a catalogue-common loading, so that the household
coordinate changes the size marginal without changing composition conditional on size.

The final checkpoint is `artifacts/candidate_rank1.pt` and the complete console record is
`artifacts/pipeline_household_rank1_full.log`.

## 2. Training result

The exact additive parent converged at update 14,300; its best fixed-panel validation
score was \(-44.736334\) at update 13,100. The split-half spectral audit selected rank 5
with overlap \(0.516813\); ranks 6--8 did not satisfy the stability gate.

The constrained natural-parameter interaction solve selected ridge \(0.0003\). Its
cross-fitted likelihood gain was \(0.024333\) nats/basket, the weaker split gained
\(0.023898\), and median proposal ESS fraction was \(0.99821\). All five eigenvalues of
the fitted interaction matrix reached the declared spectral cap. That saturation is
reported as a capacity diagnostic, not used to relax the cap after looking at test data.

The identified household-size block selected ridge 4,800 by within-household cross-fit:

\[
\Delta\ell_{\mathrm{size}}^{\mathrm{CF}}
=0.00320684\pm0.00022071,
\qquad
\operatorname{LCB}_{95}=0.00277425.
\]

The full-panel gain was \(0.01026735\) nats/basket and five households reached the
predeclared conditional-tail cap.

## 3. Locked likelihood

All values below use complete support \(1\le |S|\le120\), the same 4,096 trips for parent
and child, q7 for the interaction model, and an exact dynamic program for the additive
parent.

| Split | Rank-one model | Exact additive parent | Paired gain |
|---|---:|---:|---:|
| Validation | \(-43.687816\) | \(-43.714530\) | \(0.026714\pm0.002107\) |
| Test | \(-46.064895\) | \(-46.097646\) | \(0.032750\pm0.002393\) |

On independent 128-trip panels, q7-minus-q8 error had 95% absolute upper bound
\(0.000318\) nats on validation and \(0.000468\) on test. After charging those bounds
against the paired gains, both lower confidence bounds remain positive. The interaction
model therefore has a statistically and numerically established likelihood gain over its
matched additive parent.

## 4. Recommendation

The locked test protocol hides one bought product and ranks all contemporaneously
available products by conditional add-one energy. The normalizer cancels, so these metrics
do not depend on q7.

| Metric | Result |
|---|---:|
| Eligible cases | 1,615 |
| MRR | \(0.095246\pm0.006075\) |
| MRR@5 | \(0.081899\) |
| MRR@10 | \(0.087099\) |
| MRR@20 | \(0.090316\) |
| Recall@5 | \(0.122601\) |
| Recall@10 | \(0.160991\) |
| Recall@20 | \(0.207430\) |

The interaction-minus-additive MRR gain is
\(0.0011649\pm0.0006271\), with 95% interval
\([-0.000064,0.002394]\). Total recommendation quality is useful, but this panel does not
establish a separate interaction-only recommendation improvement at the 5% level.

## 5. Interaction-embedding audit

Embedding axes are rotationally unidentified, so the audit uses row norms and the Gram
matrix, not coordinate labels. For products \(i,j\), the identified pair-specific
coefficient is

\[
\gamma_{ij}=\phi_i^\top\phi_j-
\rho_{c(i)}\mathbf 1\{c(i)=c(j)\}.
\]

For a background basket \(T\), the full energy cross-difference additionally contains
the common term \(-\Delta^2\rho_0(|T|)\). Thus \(\gamma_{ij}\) orders product-specific
interaction effects at a fixed background size but is not, by itself, the complete energy
contrast.

The fitted checkpoint has active rank 5. All five singular values equal the declared
spectral cap of 1. The rank-5 split-half mean squared subspace overlap is \(0.516813\),
and 919/1,963/3,954 products carry 90%/95%/99% of squared row-norm mass. Saturation and
moderate subspace stability make the ranking useful for candidate retrieval, but prevent
strong claims about precise product-level magnitudes.

The 2,000 strongest cross-affinity pairs were selected from training parameters only and
then evaluated on all 23,340 test baskets. The null randomly allocates the observed
product-incidence stubs to the observed basket-size sequence, preserving marginal product
frequency and basket-size moments. Controls were matched using training product frequency
and household support.

| Panel | Observed co-incidences | Null expectation | Aggregate lift | Fraction above null |
|---|---:|---:|---:|---:|
| Top Gram pairs | 29,911 | 24,589.4 | 1.216 | 67.95% |
| Matched controls | 27,908 | 27,959.4 | 0.998 | 32.70% |

Examples with both a positive model coefficient and held-out excess co-incidence include
bananas--strawberries, cucumbers--green peppers, cucumbers--broccoli, premium hot-dog
meat--hot-dog buns, and premium meat--hamburger buns. Individual counterexamples also
exist: the high-scoring extra-large-eggs--bananas pair has 280 observed co-incidences
versus 317.6 expected. The conclusion is therefore aggregate: the Gram kernel contains
held-out complement information, but every SKU pair is a hypothesis requiring support,
replication and ultimately randomized promotion validation. The configuration lift is
not the model odds multiplier and neither quantity is a causal effect.

The reproducible entry point is
`scripts/version4/audit_interaction_embeddings.py`; it writes the detailed JSON and
Markdown reports under `reports/`.

## 6. Converged external baselines

Bernoulli, DPP and NDPP were trained from independent fresh lineages with validation-only
checkpoint selection. Each had to complete at least two epoch-equivalents, reach its
learning-rate floor, and satisfy the stale-validation convergence certificate before test
scoring. The selected checkpoints were evaluated on the identical locked 4,096-trip test
manifest used by the main model.

| Model | Selected update | Terminal convergence update | Test nats/basket | Main-model paired gain |
|---|---:|---:|---:|---:|
| Version-4 rank-one | -- | -- | \(-46.064895\) | -- |
| Bernoulli | 52,000 | 56,000 | \(-48.316937\) | \(2.252042\pm0.098425\) |
| DPP | 46,500 | 52,000 | \(-48.326534\) | \(2.261638\pm0.097672\) |
| NDPP | 55,500 | 59,000 | \(-47.877142\) | \(1.812247\pm0.092393\) |

The main model is statistically better than all three; NDPP is the strongest external
baseline. The Bernoulli model receives the same observed contextual utility families but
fits fresh weights and contains no interaction, category, total-size or household-size
block from the Version-4 checkpoint. DPP and NDPP are normalized over all nonempty sizes;
their certified maximum probability bounds above size 120 are \(7.1\times10^{-29}\) and
\(6.7\times10^{-26}\), so this support distinction cannot explain the result.

The exact additive parent and multinomial are treated as ablations rather than external
competitors. SHOPPER is excluded from this headline because its sequential posterior and
ordering-marginalization protocol is not matched to these three direct-likelihood fits.

## 7. Population basket-size certification

The cheap q6 rule screened all 160,007 modeled contexts. It estimated observed/model mean
sizes \(7.64925/7.59039\) and observed/model \(N\ge60\) rates
\(0.001875/0.001660\). No screened context had majority extreme-tail probability.

The q7 confirmation evaluated 2,048 high-risk contexts and a separate random calibration
panel of 2,048 contexts:

- maximum confirmed \(P(N\ge60\mid x)\): \(0.402682\);
- contexts with \(P(N\ge60\mid x)\ge0.5\): zero;
- contexts with confirmed \(E[N\mid x]\ge60\): zero; and
- calibrated population-tail 95% upper bound: \(0.002013\), below the allowed
  \(0.004250\).

All four declared population gates pass. The mean absolute q6--q7 expected-size gap is
\(0.8002\) products, with maximum \(5.2889\); the one-item q6 fidelity diagnostic therefore
does not pass. Safety is based on q7 confirmation and conservative coverage envelopes,
not on pretending q6 is exact.

## 8. Generation and counterfactuals

Generation produced no unavailable products and no duplicate items. Minimum normalized
SMC ESS was \(0.99938\), and the price audit showed the required monotone direction.

The small segment-balanced generation panel nevertheless exposes a material calibration
caveat:

| Size moment | Generated | Selected observed baskets |
|---|---:|---:|
| Mean | 7.3406 | 10.0313 |
| Variance | 63.8349 | 136.2803 |

This does not contradict the full-population mean/tail pass: the 64-context-per-segment
panel is deliberately balanced and has a different empirical context distribution. It
does mean that segment-level rollout claims need a larger, distribution-weighted
generation audit before deployment.

## 9. Promotion-policy experiment

The three-segment, 28-day budget-constrained environment reweights common factual SMC
particles under 10% and 20% five-SKU discounts. It optimizes a paired 95% lower-confidence
reward subject to expected markdown budget, ESS and extreme-tail gates. The evaluated
policies used 98.9--99.4% of their scenario budgets without overspend. The strongest
supported action was a 10% milk-bundle discount for segment 1; its per-segment-trip
incremental list-price basket value was \(0.9764\pm0.1511\), with lower bound \(0.6803\).

These results support campaign shortlisting, not autonomous pricing. The model conditions
on an arriving nonempty trip and does not contain visit probability, quantities,
wholesale costs, inventory or causal treatment identification. Full details are in
[SEGMENT_PROMOTION_MDP.md](SEGMENT_PROMOTION_MDP.md).

## 10. Decision

The pipeline passes its declared technical likelihood, numerical and population-tail
certification. It is the accepted implementation for continued research and controlled
rollout experiments. Commercial production readiness remains withheld until the
segment-level generation mismatch is resolved or bounded on a larger weighted panel and
promotion response is validated with randomized interventions and economic state.
