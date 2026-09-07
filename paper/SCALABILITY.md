# Scalability of the Version-4 energy-basket model

## 1. Executive conclusion

The model is computationally scalable in catalogue size, household count, and number of
observed trips because it never enumerates all possible baskets and never stores a dense
product-by-product interaction matrix. Its main scalability limit is interaction rank:
high-accuracy likelihood evaluation requires a Gaussian quadrature whose node count grows
rapidly with rank. Full interaction-aware basket generation is a second expensive path
because it repeats a conditional dynamic program across SMC particles and bridge levels.

The practical conclusion is:

- training and recommendation are viable on the complete 5,455-product catalogue;
- rank 4--8 interaction models are practical when high-accuracy quadrature is restricted
  to controlled evaluation panels;
- recommendation is suitable for online or near-online serving;
- likelihood certification, generation, and policy simulation are primarily offline
  batch operations; and
- very high interaction ranks, exhaustive price searches, and real-time full-basket SMC
  generation are not practical with the current implementation.

The model has removed exponential dependence on the number of possible baskets. The
remaining challenge is the repeated polynomial work required by numerical integration
and sampling.

---

## 2. Quantities controlling computational cost

Let:

| Symbol | Meaning |
|---|---|
| $T$ | Number of observed checkout trips |
| $J$ | Total number of modeled products; currently 5,455 |
| $J_x$ | Number of products offered in context $x$ |
| $H$ | Number of modeled households; currently 1,920 |
| $C_x$ | Number of nonempty affinity groups in context $x$ |
| $n_{\max}$ | Maximum supported basket size; currently 120 |
| $K$ | Additive household-taste rank |
| $K_p$ | Household--product price-response rank |
| $r$ | Active Gram-interaction rank |
| $M_q(r)$ | Number of Smolyak nodes for rank $r$ and level $q$ |
| $N_{\mathrm{part}}$ | Number of SMC particles |
| $L$ | Number of SMC bridge transitions |

The effects of these dimensions are not interchangeable. Catalogue size primarily
controls dynamic-program work, while interaction rank controls the number of times that
work must be repeated during high-accuracy integration.

---

## 3. Catalogue scalability

### 3.1 The model avoids subset enumeration

A catalogue of $J$ products has $2^J$ possible subsets. Explicit enumeration is
impossible even at modest $J$. Version-4 instead uses elementary-symmetric-polynomial and
category-convolution recursions. For one context, the exact no-Gram normalizer costs
approximately

\[
O\!\left(J_xn_{\max}+C_xn_{\max}^2\right).
\tag{1}
\]

The dependence on catalogue size is therefore approximately linear rather than
exponential. Increasing the offered assortment from 5,455 to 10,910 products would be
expected to roughly double the product-recursion component, not square it or enumerate
additional subsets.

The $C_xn_{\max}^2$ term comes from combining affinity-group polynomials. Increasing
$n_{\max}$ can consequently be more expensive than increasing $J_x$: doubling
$n_{\max}$ may increase this convolution term by approximately four times.

### 3.2 The interaction representation is low rank

The product interaction matrix is

\[
K=\Phi\Phi^\top,
\qquad
\Phi\in\mathbb R^{J\times r}.
\tag{2}
\]

The implementation stores $Phi$, not the dense $J\times J$ matrix $K$. Interaction
storage is therefore $O(Jr)$ rather than $O(J^2)$. With $J=5,455$ and $r=5$, the
embedding contains 27,275 values, whereas a dense interaction matrix would contain
approximately 29.8 million values.

The spectral rank audit similarly uses sparse co-incidence operations and a sparse
leading eigensolver. It does not construct a dense matrix containing every product pair.

### 3.3 Effect of real assortment information

The current data do not contain a reliable stock feed, so the declared support uses the
complete 5,455-product catalogue in every context. A production retailer normally has
store- and time-specific availability. Using the true offered assortment would reduce
$J_x$ and often $C_x$, accelerating training, likelihood evaluation, recommendation, and
generation while making the probability support more realistic.

Catalogue expansion is therefore computationally plausible into the tens of thousands of
products, provided that actual assortment filtering is available. Statistical sparsity
and cold-start products may become limiting before memory for the low-rank embeddings
does.

---

## 4. Household and data-volume scalability

Household-specific storage is approximately

\[
O\!\left(H(K+K_p+1)\right),
\tag{3}
\]

covering taste coordinates, price coordinates, and the common basket-size coordinate.
It is linear in the number of households. The post-interaction household-size correction
is one strictly concave, one-dimensional problem per household, so these solves are
independent and naturally parallelizable.

Trip-dependent training and evaluation are also approximately linear in $T$. Contexts can
be split into batches and many evaluation operations can be parallelized across trips.
The present pipeline is principally a single-machine CPU implementation rather than a
distributed trainer, so the algorithm exposes more horizontal parallelism than the
current runtime uses.

At very large $H$, the primary difficulty is statistical rather than computational.
Households with few trips cannot support unconstrained individual embeddings. Hierarchical
pooling, segment priors, or feature-based cold-start mappings would then be required.

---

## 5. Scalability of exact additive training

When the Gram interaction is zero, the category/cardinality dynamic program computes the
normalizer and its gradient exactly. One minibatch update costs approximately

\[
O\!\left(
B[J_xn_{\max}+C_xn_{\max}^2]
\right),
\tag{4}
\]

where $B$ is the minibatch size. No Smolyak nodes, QMC samples, particles, retries, or
skipped trips are involved.

A historical corrected-data execution performed 14,700 additive updates in 6,387.7
seconds:

\[
\frac{6387.7}{14700}\approx0.435
\quad\text{seconds per update},
\tag{5}
\]

or about 138 updates per minute. This is evidence for the scale of the selected exact
additive path, not a hardware-independent benchmark and not the duration of every
pipeline stage.

This stage is currently CPU-oriented. The exact float64 ESP/category-polynomial
normalizer and its custom adjoint are native CPU operations, while rank construction uses
SciPy sparse and convex CPU solvers. Moving only the smaller dense utility calculations
to a GPU would add host/device transfers without accelerating the dominant work.

---

## 6. Scalability of interaction fitting

### 6.1 The selected interaction solve

The pipeline first estimates a stable product subspace $U\in\mathbb R^{J\times r}$ and
then fits a small positive-semidefinite matrix $C_{\mathrm{int}}\in\mathbb R^{r\times r}$.
For $M$ selected contexts and $D$ exact additive-parent draws per context, forming the
natural interaction statistics costs approximately

\[
O(MDr^2).
\tag{6}
\]

At rank 5, the symmetric matrix has only

\[
\frac{r(r+1)}2=15
\]

unique coordinates. The proposal baskets are generated once and their sufficient
statistics are reused. Consequently, the projected optimization does not repeatedly
estimate a noisy high-dimensional partition function.

This advantage depends on overlap between the additive proposal and the fitted
interaction child. In the recorded rank-one pipeline, the median proposal ESS fraction
was approximately 0.998. That is excellent overlap. If much stronger interactions were
allowed, proposal ESS could fall, requiring more draws $D$ and increasing both cost and
finite-sample error.

### 6.2 Statistical scalability of rank selection

Increasing rank adds capacity only when the data contain repeatable residual pair
structure. Candidate interaction subspaces are compared across two training halves. In
the recorded pipeline, rank 5 passed the stability criterion while ranks 6--8 did not.

Rank 5 is therefore not merely a computational fallback. It is the largest interaction
subspace supported by the current split-half evidence. Increasing rank without additional
stable information would increase numerical cost while fitting unstable directions.

---

## 7. Interaction rank and Smolyak node growth

Final full-support likelihood requires an $r$-dimensional Gaussian expectation. Smolyak
quadrature evaluates the complete dynamic program at every quadrature node. Its cost is

\[
O\!\left(
T M_q(r)[J_xn_{\max}+C_xn_{\max}^2]
\right).
\tag{7}
\]

The node counts generated by the selected implementation are:

| Interaction rank | Cheap screen | Reported likelihood | Higher audit |
|---:|---:|---:|---:|
| 4 | 9 | 49 | 201 |
| 5 | 11 | 71 | 341 |
| 6 | 13 | 97 | 533 |
| 7 | 15 | 127 | 785 |
| 8 | 17 | 161 | 1,105 |

The three columns use $q=r+1$, $q=r+2$, and $q=r+3$, respectively. Increasing $q$ does
not add parameters or make a checkpoint more trained; it improves numerical integration
for fixed parameters.

This table identifies the main scalability boundary. Rank 5 likelihood requires 71
dynamic-program evaluations per context at the reporting level and 341 at the audit
level. Rank 8 requires 161 and 1,105. Ranks such as 16 or 32 are not practical with the
current isotropic Smolyak certification rule.

The pipeline controls this cost by:

1. avoiding Smolyak during exact additive training;
2. fitting interactions from reused proposal statistics;
3. using the cheaper rule for population screening;
4. using the reporting rule on locked likelihood panels; and
5. reserving the highest rule for a small numerical-error audit.

This allocation preserves full-support likelihood accuracy while preventing the most
expensive quadrature from entering every optimization update.

---

## 8. Recommendation scalability

For add-one recommendation, the full partition function cancels. If $R$ is a partial
basket, its interaction summary

\[
m_R=\sum_{k\in R}\phi_k
\]

is computed once. Candidate product $j$ receives interaction contribution
$\phi_j^\top m_R$. Scoring all offered candidates therefore costs

\[
O(J_xr)
\tag{8}
\]

beyond contextual utility construction.

At $J_x=5,455$ and $r=5$, this requires roughly 27,000 interaction-coordinate
operations per recommendation case. It uses no Smolyak quadrature, SMC particles, or
basket enumeration. This is the model's most scalable production function and can be
batched efficiently. For much larger catalogues, a retrieval stage can shortlist
candidates before exact Version-4 reranking.

---

## 9. Basket-generation scalability

Full interaction-aware basket generation uses an SMC bridge with $L$ transitions and
$N_{\mathrm{part}}$ particles. Its dominant cost is approximately

\[
O\!\left(
L N_{\mathrm{part}}
[J_xn_{\max}+C_xn_{\max}^2]
\right)
\tag{9}
\]

per context, because exact conditional basket recursions are repeated across particles
and bridge levels. The interaction-statistic component itself is cheaper,
approximately $O(LN_{\mathrm{part}}\bar n r)$ for average basket size $\bar n$.

Generation is naturally parallel across contexts, particles, customer segments, and
counterfactual scenarios. Nevertheless, the current exact recursions are CPU-oriented,
and the implementation does not yet provide a fully distributed or CUDA-native sampler.
Generation is consequently appropriate for offline simulation, campaign studies, and
batched synthetic-data production, but not currently for low-latency generation on every
customer request.

Sampler ESS measures numerical overlap, not model calibration. The recorded minimum
normalized SMC ESS was high, but the small held-out generation panel still showed a
basket-size calibration difference. Increasing particles would reduce Monte Carlo error;
it would not automatically correct statistical misspecification in the fitted law.

---

## 10. Price-counterfactual and policy scalability

Small price changes can reuse factual samples by importance weighting. This is relatively
cheap when factual and counterfactual laws overlap. Large changes can reduce effective
sample size and require fresh interaction-aware generation.

An exhaustive search over every product, discount, household, and campaign period would
therefore be expensive. A scalable retailer workflow is hierarchical:

1. use fitted elasticity, recommendation, and interaction evidence to shortlist actions;
2. aggregate households into economically meaningful segments;
3. apply counterfactual basket simulation only to shortlisted actions;
4. solve the budget-constrained policy problem over that restricted action set; and
5. use larger particle counts and repeated seeds only for final candidates.

This procedure uses inexpensive model queries to reduce the number of expensive rollout
queries. It does not change the underlying basket law.

---

## 11. Current implementation boundary

The algorithm and implementation have different scalability properties. The mathematical
construction is batch-parallel across trips, households, quadrature nodes, particles, and
scenarios. The selected implementation currently uses a certified CPU path because its
dominant float64 polynomial recursion and sparse/convex components are CPU-native.

The current implementation is suitable for:

- the complete 5,455-product catalogue;
- approximately 200,000 checkout baskets;
- thousands to tens of thousands of households;
- statistically supported interaction ranks around 4--8;
- online or near-online recommendation;
- offline full-support likelihood certification; and
- offline segment-level generation and policy experiments.

It is not yet suitable for:

- interaction ranks such as 16--32 under isotropic Smolyak certification;
- high-accuracy likelihood over millions of contexts;
- real-time high-particle SMC generation for every customer;
- exhaustive evaluation of every SKU-discount combination;
- GPU-first execution of the complete exact pipeline; or
- distributed training without additional orchestration.

---

## 12. Practical priorities for further scaling

The following changes would improve runtime without changing the Version-4 theory:

1. **Use true context-specific assortments.** This reduces $J_x$ and $C_x$ everywhere.
2. **Batch independent contexts and particles.** Likelihood panels and SMC rollouts expose
   substantial parallelism.
3. **Cache context-only polynomial components.** Counterfactuals often change only a
   small subset of item utilities.
4. **Use candidate retrieval before recommendation reranking.** This is useful when the
   catalogue becomes much larger than 5,455 products.
5. **Keep high-order quadrature as certification, not training.** The current staged
   interaction solve already follows this principle.
6. **Investigate anisotropic or dimension-adaptive certification.** This could exploit
   unequal importance across latent interaction directions, but it would require new
   deterministic error and signed-cancellation checks before replacing the current rule.
7. **Develop a native parallel conditional sampler.** Vectorizing and parallelizing the
   exact polynomial recursion would directly reduce generation and policy latency.

The final assessment is that Version-4 has solved the catalogue combinatorics problem but
not the general high-dimensional integration problem. It scales well by keeping the
interaction rank small and statistically justified. Recommendation inherits excellent
scaling because the normalizer cancels; likelihood and generation remain controlled
offline computations whose cost must be budgeted explicitly.
