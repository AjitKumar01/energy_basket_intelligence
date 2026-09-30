# Pipeline decision for the Version-4 model

## 1. The unit of comparison

A checkpoint is an output, not a method. A numbered experiment can look better because it
used a different parent, panel, split, support, rank, or quadrature rule. Therefore this
project compares **pipelines** under one contract:

\[
\mathcal P=(\text{data},\text{initialization},\text{objective},
\text{normalizer},\text{optimizer},\text{acceptance gates}).
\]

Two outputs are comparable only when these components and the evaluation manifest agree.
Historical numeric labels are retained only in old provenance; they are not model names.

## 2. Non-negotiable model contract

Every admissible pipeline fits the joint law of `THEORY.md` §4:

\[
p_\theta(S\mid x)=\frac{\exp\{E_\theta(S,x)\}}{Z_\theta(x)},
\quad
Z_\theta(x)=\sum_{1\leq |A|\leq120}\exp\{E_\theta(A,x)\}.
\]

The catalogue has 5,455 products. The Gram term is

\[
\sum_{i<j\in S}\phi_i^\top\phi_j
=\frac12\left(\left\|\sum_{j\in S}\phi_j\right\|^2
-\sum_{j\in S}\|\phi_j\|^2\right).
\]

No pipeline may replace the joint law with a conditional-size model, remove interactions,
truncate the catalogue to convenient products, or optimize MRR instead of likelihood.

## 3. Decision order

Pipeline selection is lexicographic, not a hand-tuned weighted average:

1. unchanged law and complete support;
2. fresh, reproducible completion without estimator aborts;
3. numerical score fidelity at the accepted rank;
4. population basket-size and generation calibration;
5. paired held-out likelihood against the exact additive parent and external baselines;
6. interaction contribution to held-out recommendation;
7. wall time and memory.

This ordering prevents a small likelihood gain from buying a pathological simulator.

## 4. Pipelines evaluated

| Pipeline family | Normalizer/training idea | What the evidence establishes | Decision |
|---|---|---|---|
| End-to-end RQMC joint SGD | randomized Gaussian integral on every update | high latency, retries/aborts, and no completed convincing convergence path | reject as default |
| Long joint Smolyak SGD from an arbitrary checkpoint | q8 large-batch score plus q9 correction and q10 audit | stable quadrature and positive likelihood gain, but poor initialization wastes updates and rare size phases remain | reject as a standalone pipeline |
| Unconstrained post-fit scalar/context size corrections | tilt existing \(\rho_0\) or a household-common direction without a conditional-tail gate | changes are cheap, but the unconstrained versions can move extreme mass to the wrong contexts | reject |
| Exact additive + rank score + constrained natural-parameter MCLE + identified household size block | exact parent draws fit \(C\) and a correction inside \(\rho_0\); a one-dimensional, ridge-regularized household block is then solved with a deterministic conditional-tail cap | full-catalogue rank learning, monotone block solves, cross-fit selection, no extra latent variable, and direct control of localized size phases | **selected; full validation passed** |

The selected pipeline, with the optional joint refinement stage, completes a fresh fit from
initialization and passes every declared likelihood, numerical and population-tail gate on the
synthetic stress world (`JOINT_REFINEMENT_STAGE.md` §3). This is a technical pipeline
certification, not a causal or commercial deployment certificate.

## 5. Selected pipeline

### Stage A — data and support

A dataset adapter writes the canonical directory, and `prepare_model_bundle.py` converts it
into a fingerprinted model-data bundle: checkout baskets, product/store/week prices and price
deviations, promotions, the declared availability rule, and a training-only product partition
(`CANONICAL_INPUT_CONTRACT.md`). The product and household cohorts are defined from training
periods only. An observed basket in any split outside the declared support is a hard error;
outcomes are never deleted to fit the support. The pipeline re-verifies the bundle fingerprint
before any stage runs.

### Stage B — exact additive maximum likelihood

Set \(\Phi=0\). The H--S integral disappears and the category/cardinality dynamic program
computes \(Z(x)\) and its gradients exactly for all products and sizes 1 through 120. This
stage fits all original non-Gram incidence parameters. It is the fast convergence phase.
The 30,000-update setting is only a safety ceiling: validation plateaus lower the learning
rate, and convergence is declared only after the minimum rate stops producing new bests.

The original category parameter is fitted subject to the complete-support admissibility
constraint

\[
(-\rho_c)_+{m_c\choose2}\le1.5,
\]

where (m_c) is the largest available count of category (c) on support 1 through 120.
This prevents a broad affinity group from being extrapolated as a 120-product attractive
clique. It changes neither the Version-4 energy nor the exact dynamic program and costs
(O(C)) per update.

### Stage C — rank identification

At \(\Phi=0\), the ordinary gradient with respect to \(\Phi\) is zero because the energy is
quadratic in \(\Phi\). The informative local object is instead the pair-statistic score

\[
R=\mathbb E_{\rm data}[P(S)]-\mathbb E_{p_{\rm add}}[P(S)].
\]

Positive eigenvectors of \(R\) are locally improving positive-semidefinite Gram
directions. Ranks 8 down to 4 are tested on independent training halves. The largest rank
whose mean squared subspace overlap is at least 0.5 is accepted. Rank is capacity; it is
not quadrature accuracy.

### Stage D — constrained interaction and size MCLE

In the accepted basis \(U\), write

\[
K=\Phi\Phi^\top=UCU^\top,\qquad C\succeq0.
\]

Partition sizes \(1{:}120\) into the seven bands

\[
1{:}4,\;5{:}10,\;11{:}20,\;21{:}40,\;41{:}59,\;60{:}80,\;81{:}120.
\]

For each context, the additive dynamic program computes each band's exact probability
\(p_{m\ell}\). It then draws baskets exactly conditional on that band. Let
\(S_{m\ell d}\) denote those fixed draws and define

\[
h_{C,q}(S)=\operatorname{tr}\{C F_U(S)\}-q(|S|),
\]

where \(q\) is piecewise linear between 12 knots and is written back into the existing
\(\rho_0(1{:}120)\) table. The exact partition-ratio identity is

\[
\frac{Z_{C,q,+}(x_m)}{Z_{0,+}(x_m)}
=
\sum_\ell p_{m\ell}
\mathbb E_0[e^{h_{C,q}(S)}\mid N\in\mathcal B_\ell,x_m].
\]

Its fixed-bank estimate gives the sampled log-likelihood gain

\[
\widehat G(C,q)=\frac1M\sum_{m=1}^M\left[
h(S_m^{\rm obs})-\log\left\{
\sum_\ell\frac{p_{m\ell}}{D_\ell}
\sum_{d=1}^{D_\ell}e^{h(S_{m\ell d})}
\right\}\right].
\]

The ratio estimate is unbiased before taking its logarithm. Its logarithm is not exactly
unbiased at finite draw count, which is why independent Smolyak evaluation remains the
acceptance authority.

With the bank fixed, the objective is concave because a linear term minus log-sum-exp is
concave. The feasible set

\[
0\preceq C\preceq \sigma_{\max}^2I,\qquad
q(1)=0,\qquad |q(n_k)|\le q_{\max}
\]

is convex. Quadratic magnitude and second-difference penalties regularize the size curve.
The already fitted category penalty \(\rho_c\) is frozen: an explicit
\(C,\rho_c,\rho_0\) experiment failed cross-validation. Alternating bounded L-BFGS for
\(q\) and projected Armijo ascent for \(C\) optimize the same concave objective.

Ridge is selected by swapped context halves. Both held-out directions must improve and
within-band effective sample size must pass its declared floor. The full solve then
recovers \(\Phi=U C^{1/2}\). This trains an interaction vector for every one of the 5,455
products without optimizing 5,455 by \(r\) unidentified factor coordinates.

The correction is not a new size factor or a change to the Version-4 joint law. It uses
capacity already declared in the unrestricted size potential. The fixed draw bank is
content-addressed and automatically reused after interruption. Hard conditional-Bernoulli
draws fall back to exact log-domain dynamic programming rather than aborting.

### Stage E — identified household-size block

Reserve one existing household-taste coordinate for the fixed all-product loading. Then

\[
b_{jh}=\widetilde b_{jh}+\kappa_h,
\qquad \sum_{j=1}^{J}\widetilde\alpha_j=0.
\]

For a basket of size \(n\), the new coordinate contributes \(n\kappa_h\). Equivalently,
it is the household-specific linear direction

\[
\rho_{0h}(n)=\rho_0(n)-n\kappa_h
\]

inside the original Version-4 energy. It does not introduce a conditional-size model.
For fixed other parameters, each \(\kappa_h\) is fitted by a one-dimensional strictly
concave penalized size likelihood. Ridge is selected on alternating chronological trips
within household, and the result is projected onto an upper bound obtained from the
complete-population low-rule tail screen. This is an exact block update of the same joint
likelihood; the penalty stabilizes sparse households, while the cap prevents a localized
large-basket phase.

### Optional stage — joint refinement

With `--joint-refinement`, a stage between E and F moves \(\lambda,\theta,\alpha,\rho_c,\rho_0\) and
\(C\) jointly from the staged optimum by iterated Monte Carlo maximum likelihood, choosing each
step on held-out selection trips, and re-applies Stage E's household-size cap after every round.
The refined checkpoint replaces the staged one only if it has a
positive paired validation gain, passes a numerical audit one level finer, and passes the
population-size audit below. See `JOINT_REFINEMENT_STAGE.md`.

### Stage F — certification

The candidate must pass:

- paired validation and test likelihood at \(q=r+2\), with a \(q=r+3\) numerical audit;
- exact conditional add-one MRR and recall on a fixed test manifest;
- SMC validity: no duplicates or unavailable products and adequate ESS;
- monotone uniform-price counterfactual response;
- segment-specific generation and price-response diagnostics;
- an orientation-invariant Gram/complement audit on held-out baskets;
- \(q=r+1\) screening over every supported training context and \(q=r+2\)
  confirmation of the highest-risk size laws; and
- after tail certification, the three-segment budget-constrained promotion-policy audit.

The population gate rejects a candidate when its mean tail probability is incompatible
with observed \(N\geq60\) frequency or when a context with observed size below 40 assigns
more than half its mass to \(N\geq60\). This directly targets the rare phase transition
that small validation panels missed.

## 6. Complexity

Let \(B\) be batch size, \(J_x\) the offered products in a context, \(C_x\) its nonempty
affinity groups, \(n_{\max}\) the declared maximum basket size, accepted rank \(r\), and
Smolyak node count \(M_q(r)\).

- Exact additive update: approximately
  \(O(B[J_x n_{\max}+C_x n_{\max}^2])\), with no quadrature nodes.
- Pair-score construction: sparse observed/generated co-incidence plus a sparse leading
  eigensolve; the dense \(J\times J\) Gram matrix is never materialized.
- Projected Fisher accumulation:
  \(O(TD[r^2+r^4])\) for \(T\) contexts and \(D\) draws, with only
  \(r(r+1)/2\) fitted coordinates.
- Household-size solve after size laws are cached:
  \(O(Tn_{\max}\log(1/\epsilon))\) time for one-dimensional bisections and
  \(O(Tn_{\max})\) storage; it adds no H--S dimension and no
  product-by-household tensor.
- Smolyak likelihood audit:
  \(O(T M_q(r)[J_x n_{\max}+C_x n_{\max}^2])\).
- Recommendation: \(O(J_x r)\) add-one scoring; \(\log Z\) cancels, so no Smolyak cost.

For rank 7, q8/q9/q10 use 15/127/785 nodes. In general the selected pipeline uses
\(q=r+1\) for the population screen, \(q=r+2\) for reported likelihood and high-risk
confirmation, and \(q=r+3\) only for a small numerical certification panel. The corrected
rank-5 fit therefore uses q6/q7/q8; the rank-relative accuracy contract is unchanged.

## 7. What “best” means now

The selected pipeline is the smallest revision targeted by the failure audit: the
interaction block still learns full-catalogue interactions, while the identified
household coordinate addresses local size bias without changing fixed-size composition or
increasing quadrature rank. It is selected because a fresh end-to-end execution passed the
predeclared gates—not because one numbered run or checkpoint happened to look favorable.
