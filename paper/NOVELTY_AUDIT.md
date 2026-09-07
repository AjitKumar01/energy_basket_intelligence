# Novelty audit of the Version-4 energy-basket work

**Audit date:** 5 September 2026
**Scope:** research and publication novelty, not a legal patentability opinion

## 1. Executive conclusion

The work is **moderately novel in its present form and has the ingredients of a strong
applied-methods contribution**. Its defensible novelty is the complete construction, not
any one ingredient.

The following ingredients all have substantial prior art:

- joint energy or Ising models for binary inclusion variables;
- low-rank interaction matrices;
- the Hubbard--Stratonovich transformation;
- exact dynamic programs for cardinality potentials;
- elementary symmetric polynomials for fixed-cardinality subset sums;
- Smolyak sparse-grid quadrature;
- Monte Carlo maximum likelihood; and
- annealed importance sampling or sequential Monte Carlo bridges.

The strongest potentially original contribution is instead the combination of:

1. a normalized, order-invariant, contextual law for complete nonempty retail baskets;
2. household, price, promotion, season, and store effects at SKU level;
3. a low-rank Gram interaction together with affinity-group and total-size potentials;
4. an exact Hubbard--Stratonovich/elementary-symmetric-polynomial reduction of the
   exponential subset sum to a rank-dimensional Gaussian expectation;
5. a staged spectral and natural-parameter interaction estimator; and
6. law-consistent likelihood evaluation, recommendation, generation, and
   counterfactual analysis over the declared 5,455-product support.

A concise assessment is:

| Aspect | Assessment |
|---|---|
| Novelty of individual mathematical ingredients | Low |
| Novelty of the Version-4 model specification | Moderate |
| Novelty of the combined H--S/category/size polynomial reduction | Moderate to high; the strongest candidate contribution |
| Novelty of the staged interaction estimator | Moderate to high, but not yet supported by a complete statistical theorem |
| Novelty of retailer applications | Low to moderate |
| Full-catalogue implementation and certification | Moderate to high as an engineering and empirical contribution |
| Overall novelty currently demonstrated | Approximately 3/5 |
| Potential after stronger theory and closest-model comparisons | Approximately 4/5 |

The appropriate position is therefore:

> A novel structured basket law and scalable inference synthesis, built from established
> mathematical tools.

It should not be positioned as the invention of energy-based basket modeling,
Hubbard--Stratonovich inference, sparse-grid quadrature, or SMC.

## 2. The object being assessed

For an observed context $x$, assortment $\mathcal A_x$, and supported nonempty basket
$S$, Version-4 uses

\[
p_\Theta(S\mid x,S\ne\varnothing)
=
\frac{\exp\{E_\Theta(S,x)\}}{Z_{+,\Theta}(x)},
\tag{N.1}
\]

with energy

\[
\begin{aligned}
E_\Theta(S,x)
={}&
\sum_{j\in S}b_j(x)
+\sum_{j<k\,;\,j,k\in S}\phi_j^\top\phi_k\\
&-\sum_c\rho_c{n_c(S)\choose2}
-\rho_0(|S|).
\end{aligned}
\tag{N.2}
\]

The full model and training flow are derived in
[PIPELINE_TEXTBOOK.md](PIPELINE_TEXTBOOK.md). The normalizer theorem gives

\[
Z_+(x)
=
\mathbb E_{z\sim\mathcal N(0,I_r)}
\left[
\sum_{n=1}^{n_{\max}}e^{-\rho_0(n)}A_n(z,x)
\right],
\tag{N.3}
\]

where $A_n(z,x)$ is obtained exactly, for fixed $z$, from affinity-group elementary
symmetric polynomials and truncated polynomial convolution. The external integral has
dimension $r$, the active interaction rank, rather than dimension $J$, the catalogue
size.

The selected estimator first fits the exact $\Phi=0$ parent, obtains stable interaction
directions from an observed-minus-parent pair residual, fits a PSD natural parameter
$C$ in $K=UCU^\top$ using fixed exact parent draws, and certifies the fitted child with
independent Smolyak likelihood. Positive interaction-tempered SMC is used when complete
draws from the fitted law are required.

## 3. Closest established work

### 3.1 Joint binary and basket models

Quadratic-exponential regression already defines joint distributions of correlated binary
variables using marginal and pairwise terms. This makes the mathematical family underlying
Eq. (N.2) related to an Ising or quadratic-exponential model rather than a wholly new
probability family. Zhao and Prentice also noted the computational difficulty of fitting
large dependent blocks.

- [Zhao and Prentice, “Correlated binary regression using a quadratic exponential model,”
  *Biometrika*, 1990](https://academic.oup.com/biomet/article-pdf/77/3/642/705505/77-3-642.pdf)

Simultaneous interdependent multi-category basket choice, household heterogeneity,
complementarity, price effects, and basket simulation were studied in the marketing
literature well before Version-4. The 1999 Shopping Basket model used a multivariate
probit and hierarchical Bayes fit, although only on four categories rather than a
full SKU catalogue.

- [Manchanda, Ansari and Gupta, “The Shopping Basket,” *Marketing Science*,
  1999](https://pubsonline.informs.org/doi/abs/10.1287/mksc.18.2.95)

More recently, Ising models have been used directly for multi-item basket choice and
retail assortment optimization. This is especially important prior art against any claim
that Version-4 is the first Ising-style model for retailer decisions.

- [Vasilyev, Maier and Seifert, “Assortment optimization given basket shopping behavior
  using the Ising model,” 2025](https://arxiv.org/abs/2502.16260)

### 3.2 Large-scale contextual retail models

SHOPPER models large retail baskets sequentially, with household preferences, prices,
product interactions, substitutes and complements, generation, and price
counterfactuals. Its principal distinction from Version-4 is that SHOPPER defines a
sequential choice-and-checkout process, whereas Version-4 defines an order-invariant
probability for a set and normalizes over the declared set support.

- [Ruiz, Athey and Blei, “SHOPPER: A Probabilistic Model of Consumer Choice with
  Substitutes and Complements,” *Annals of Applied Statistics*,
  2020](https://doi.org/10.1214/19-AOAS1265)

Donnelly et al. model heterogeneous consumer taste and price sensitivity across many
product categories. Their consumer generally selects at most one product per category,
and utility is additive across categories. Version-4's multi-product category counts and
explicit cross-product interaction structure are materially different, but contextual
price and household factorization are not by themselves new.

- [Donnelly, Ruiz, Blei and Athey, “Counterfactual Inference for Consumer Choice Across
  Many Product Categories,” 2021](https://arxiv.org/abs/1906.02635)

### 3.3 Cardinality and subset inference

Cardinality potentials and efficient exact marginalization or joint sampling have known
dynamic-program and tree constructions. Elementary symmetric polynomials are the natural
coefficients of a product $\prod_j(1+w_ju)$ and therefore a standard way to sum products
of weights over fixed-size subsets.

- [Tarlow et al., “Fast Exact Inference for Recursive Cardinality Models,”
  2012](https://arxiv.org/abs/1210.4899)

This precedent means that the ESP recursion alone is not a novelty claim. The potentially
new part is its nesting with Version-4's affinity-group count potentials, global size
potential, and the H--S-reweighted SKU utilities.

### 3.4 Low-rank Ising inference and H--S tempering

The H--S transformation of quadratic interactions is classical. More specifically,
Koehler, Lee and Risteski already use an H--S extended state and simulated tempering to
sample approximately low-rank Ising models. This is close mathematical prior art for the
combination of low-rank binary interactions, an auxiliary Gaussian variable, and a
temperature bridge.

- [Koehler, Lee and Risteski, “Sampling Approximately Low-Rank Ising Models: MCMC Meets
  Variational Methods,” COLT 2022](https://proceedings.mlr.press/v178/koehler22a.html)

Therefore neither the H--S identity nor “tempering an H--S representation” can be claimed
as new. Version-4's possible distinction is the exact conditional basket recursion with
retail-specific category and size structure and its integration into one fitted retail
law.

### 3.5 Monte Carlo likelihood and sequential bridges

Constructing a Monte Carlo approximation to an exponential-family likelihood from fixed
draws has established precedent.

- [Geyer and Thompson, “Constrained Monte Carlo Maximum Likelihood for Dependent Data,”
  *JRSS B*, 1992](https://rss.onlinelibrary.wiley.com/doi/abs/10.1111/j.2517-6161.1992.tb01443.x)

SMC samplers already define particle approximations across a sequence of unnormalized
target laws and estimate ratios of normalizing constants.

- [Del Moral, Doucet and Jasra, “Sequential Monte Carlo Samplers,” *JRSS B*,
  2006](https://academic.oup.com/jrsssb/article-abstract/68/3/411/7110641)

Smolyak quadrature is likewise a classical sparse-grid construction. Its use for the
remaining Gaussian expectation is an implementation choice and certification mechanism,
not a new quadrature family.

## 4. Claim-by-claim novelty assessment

### 4.1 Contextual joint basket energy: moderate novelty

The broad concept is not new. The more specific structure is useful and reasonably
distinct:

- it is a normalized, order-invariant set law rather than a sequence model;
- it admits more than one item per category;
- it includes an explicit flexible total-size potential;
- it separates low-rank product geometry from shared within-affinity effects; and
- it conditions on contemporaneous assortment and context.

The contribution should be the structured specification and the consequences it enables,
not the claim that an energy model can represent a basket.

### 4.2 H--S plus affinity/category ESP plus total size: strongest potential novelty

The exact reduction in Eq. (N.3) is the most promising technical contribution. A targeted
search found no exact published match containing all of the following in one retail model:

1. contextual SKU utilities;
2. a low-rank Gram pair interaction;
3. multiple affinity-group cardinality potentials;
4. a separate arbitrary total-size potential;
5. H--S reduction to an $r$-dimensional integral; and
6. exact fixed-$z$ evaluation through nested ESP and degree convolution.

Failure to find an exact match is not proof of priority. A formal related-work search and
citation chaining are still required before using “first” language. The safe claim is:

> We derive a tractable H--S/ESP representation for this structured contextual basket law.

The unsafe claim is:

> We introduce the Hubbard--Stratonovich or elementary-symmetric-polynomial method.

### 4.3 Spectral residual and natural-parameter interaction fit: promising but incomplete

The estimator addresses real structural problems:

- $\nabla_\Phi L=0$ at $\Phi=0$ even when the Gram-space score is nonzero;
- $\Phi$ is rotationally unidentified;
- joint fitting would repeatedly pay for rank-dimensional integration while basic
  additive effects are still moving; and
- fixing the parent draws makes the natural-parameter objective deterministic and
  concave over a convex PSD constraint.

Spectral initialization, score residuals, and Monte Carlo likelihood are established
ideas. The particular combination may be novel for this basket law. To turn it into a
strong statistical-method contribution, the paper needs:

- a population characterization of the residual score matrix;
- conditions under which its leading positive eigenspace identifies the Gram subspace;
- finite-parent-draw error bounds for the fitted $C$;
- a consistency or profile-likelihood argument for the staged estimator;
- an explicit runtime and memory theorem; and
- comparison with warm-started joint MLE, alternating MLE, and pseudolikelihood.

The existing proof of concavity establishes one global optimum of the **fixed-draw sampled
objective**. It does not by itself establish consistency for the full Version-4 MLE.

### 4.4 Smolyak and SMC: low component novelty, useful system design

Using deterministic Smolyak likelihood for certification and positive SMC for generation
is a sound separation of tasks. It avoids treating signed sparse-grid weights as sampling
probabilities. This is valuable systems design, but the individual algorithms are not new.

A defensible contribution is the inference architecture:

\[
\text{exact no-Gram DP}
\longrightarrow
\text{fixed-draw interaction fit}
\longrightarrow
\text{deterministic likelihood certification}
\longrightarrow
\text{positive full-law generation}.
\tag{N.4}
\]

### 4.5 Recommendation, generation, and counterfactuals: consequences rather than separate inventions

These functions are important because they all follow from the same joint law:

- recommendation uses exact conditional add-one energy and cancels $Z_+$;
- generation samples the H--S augmented law and the exact conditional basket recursion;
- price counterfactuals change the contextual utility and query or reweight the resulting
  law; and
- segment and policy layers aggregate those predictions for decisions.

Retail recommendation, basket generation, product complements, and price
counterfactuals all have prior art. Their contribution here is internal coherence: they
are not separately trained modules with incompatible probabilities.

## 5. What the empirical evidence currently establishes

The repository records a historical full execution in which the rank-one candidate
improves over its matched exact additive parent by

\[
0.02671\pm0.00211
\quad\text{validation nats/basket}
\]

and

\[
0.03275\pm0.00239
\quad\text{test nats/basket}.
\tag{N.5}
\]

The corresponding reported quadrature-error bounds are $0.000318$ and $0.000468$ nats,
well below the gains. This is meaningful evidence that the interaction child improves
the normalized held-out basket distribution over the matched parent.

The same historical experiment reports large paired test-likelihood margins over
converged Bernoulli, DPP, and NDPP models:

| External model | Version-4 gain in nats/basket |
|---|---:|
| Bernoulli | $2.25204\pm0.09842$ |
| DPP | $2.26164\pm0.09767$ |
| NDPP | $1.81225\pm0.09239$ |

Those comparisons demonstrate predictive value of the full Version-4 specification, but
they do not isolate the Gram embedding: the competing laws also differ in size,
affinity-group, and contextual structure. The exact additive parent remains the primary
interaction ablation.

The synthetic audits strengthen the mechanism check:

- a zero-interaction world does not manufacture a positive held-out interaction gain;
- increasing true interaction strength increases recovered likelihood and MRR;
- Gram-kernel correlations reach $0.966$ and $0.993$ in moderate and strong signal
  settings; and
- larger retailer simulations recover Gram correlations of $0.9764$ when well specified
  and $0.9313$ under mild misspecification.

See [SYNTHETIC_INTERACTION_AUDIT.md](SYNTHETIC_INTERACTION_AUDIT.md) and
[SYNTHETIC_RETAILER_EXPERIMENT.md](SYNTHETIC_RETAILER_EXPERIMENT.md).

These synthetic results demonstrate identifiability and estimator behavior under their
declared designs. They do not by themselves establish novelty or generalization to other
retail datasets.

## 6. Important evidentiary qualifications

### 6.1 The current hardened code has not reproduced the historical headline yet

The values in Eq. (N.5) were produced before the latest artifact-lineage and simultaneous
ridge-selection hardening. The model law and main estimators were not intentionally
changed, but the current revision still needs one fresh full execution before its outputs
can replace the historical headline.

### 6.2 The recommendation evaluator now separates the interaction estimand

The locked evaluator produces three nested scores:

1. `additive_utility`;
2. `structured_no_gram`, which adds the affinity-group increment; and
3. `full_interaction`, which additionally adds the Gram increment.

The corrected evaluator reports three different paired contrasts and names both candidate
and reference in every output record. Its primary interaction estimand compares item 3
with item 2. Certification rejects a recommendation artifact that contains only the old
ambiguous `mrr_gain_full_minus_additive` field.

For the historical rank-one panel, the archived MRR means imply the exact paired point
gain

\[
0.0952461197-0.0939008375=0.0013452822.
\tag{N.6}
\]

The old report did not retain case ranks, so its clean paired standard error cannot be
reconstructed. Equation (N.6) is positive descriptive evidence, not a significance claim.
This issue does not affect the matched likelihood comparison in Eq. (N.5).

### 6.3 Generation is not fully calibrated

The historical population tail gates pass after the household-size correction, but the
small segment-balanced generation panel remains too small and under-dispersed relative to
its selected observed baskets. That does not invalidate likelihood, but it prevents a
strong production-simulator claim.

### 6.4 The real-data model is conditional on a purchase trip

The model estimates

\[
p(S\mid x,S\ne\varnothing),
\]

not the probability that a household visits or buys nothing. Quantities, wholesale cost,
inventory constraints, and causal treatment effects are also not fitted by the selected
real-data pipeline. Synthetic code containing those variables does not make them learned
real-data capabilities.

### 6.5 Observational counterfactuals are not automatically causal

Changing price variables inside the fitted law produces internally coherent structural
predictions. It does not prove that an actual intervention will have the predicted effect
without identification assumptions or randomized evidence.

## 7. Claims that are defensible now

The following contribution statements are supportable if written carefully.

1. **Structured basket law.** A normalized, order-invariant contextual distribution over
   nonempty retail baskets that combines SKU utility, low-rank product interactions,
   affinity-group count effects, and a flexible total-size potential.
2. **Tractable discrete reduction.** An exact reduction of the exponential subset sum to
   an $r$-dimensional Gaussian expectation whose fixed-node integrand is evaluated by
   nested category and cardinality polynomial recursions.
3. **Staged estimator.** A spectral Gram-space rank audit followed by a constrained,
   fixed-draw natural-parameter likelihood fit that avoids the zero factor-gradient at the
   additive point.
4. **Unified inference.** Likelihood, recommendation, generation, and price response are
   derived from one joint law rather than independently trained output heads.
5. **Scale and auditability.** A full declared 5,455-product implementation with
   deterministic likelihood certification, numerical-error checks, matched ablations,
   and synthetic recovery experiments.

## 8. Claims that should not be made

The current evidence does not support saying that the work is:

- the first probabilistic model of shopping baskets;
- the first retail model with complements or substitutes;
- the first use of an Ising model for basket or assortment decisions;
- a new Hubbard--Stratonovich, ESP, Smolyak, AIS, or SMC algorithm;
- state of the art in recommendation;
- a causal pricing engine;
- a complete model of store visits, unit demand, or profit;
- a production-ready retailer digital twin; or
- certified on the latest pipeline revision before a fresh full run is completed.

“First” language for the exact combined H--S/ESP construction should also be withheld
until a systematic database search and backward/forward citation review are completed.

## 9. Work needed for a convincing publication claim

### 9.1 Theory

1. State the exact H--S/ESP theorem with computational and memory complexity.
2. Separate exact algebraic reduction from numerical quadrature approximation.
3. Derive a usable Smolyak error condition for the actual integrand family or state the
   empirical adjacent-rule certification contract as such.
4. Formalize the population interaction-score matrix and its relation to the true Gram
   subspace.
5. Bound the finite-draw error of the natural-parameter MCLE.
6. State precisely what the staged estimator estimates when additive parameters are not
   reoptimized after the Gram block enters.
7. Give finite-particle or replicate diagnostics for the SMC generation path.

### 9.2 Empirical comparisons

1. Run the hardened pipeline from fresh initialization and archive all receipts.
2. Compare against the closest conceptual competitors, not only Bernoulli, DPP, and NDPP:
   - a low-rank Ising model fitted by pseudolikelihood or another scalable method;
   - SHOPPER under a clearly reconciled sequential-to-set evaluation protocol; and
   - a competitive autoregressive or permutation-invariant neural basket model.
3. Report separate ablations for contextual utility, affinity-group structure, total-size
   structure, household-size coordinate, and Gram interaction.
4. Re-estimate the clean Gram-only MRR uncertainty on the next hardened full checkpoint;
   the evaluator and certification schema are already corrected.
5. Demonstrate the method on at least one additional retail dataset.
6. Expand generation evaluation to a distribution-weighted held-out context panel with
   size, category, incidence, and pair calibration.

### 9.3 Retail interpretation

1. Keep recommendation and descriptive complement discovery separate from causal bundle
   effects.
2. Present promotion policies as experiment shortlists until randomized treatment data
   are available.
3. Add a visit/no-purchase component before claiming unconditional demand or production
   traffic simulation.
4. Fit quantities, costs, and inventory before claiming profit optimization.

## 10. Recommended paper positioning

A defensible high-level contribution statement is:

> We introduce a normalized, order-invariant contextual basket law with low-rank SKU
> interactions and nested affinity and total-cardinality structure. For this law, we derive
> an exact H--S/ESP reduction that replaces an exponential subset sum by a
> rank-dimensional Gaussian expectation with an exact fixed-node dynamic program. We fit
> interactions through a staged spectral natural-parameter estimator, certify likelihood
> deterministically, and generate baskets through a positive interaction-tempered SMC
> bridge.

This wording makes the actual new combination explicit and credits the standard tools as
ingredients. It also distinguishes the work from SHOPPER's sequential model, additive
many-category choice models, generic cardinality inference, and generic low-rank Ising
sampling.

## 11. Final assessment

There is genuine research value in the Version-4 work. The most credible novelty is the
structured H--S/ESP normalizer and the end-to-end inference architecture around the same
full-basket probability law. The staged estimator may become a second strong contribution
if its statistical target and finite-draw error are formalized.

Today, the project is best described as a **novel synthesis with a potentially original
tractability result**, supported by one real-data study and useful synthetic recovery
evidence. It is not yet established as a new fundamental class of energy model or a new
general-purpose sampling method. Stronger closest-model comparisons, a fresh hardened
run, corrected Gram-only recommendation evaluation, and another dataset would materially
raise the strength of the novelty claim.
