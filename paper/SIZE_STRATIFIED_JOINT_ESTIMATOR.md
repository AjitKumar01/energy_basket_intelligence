# Size-Stratified Joint Estimation for the Version-4 Basket Model

## 1. Purpose

This document develops an estimator improvement for the existing Version-4 energy basket
model. It does **not** alter the basket probability, catalogue support, interaction term,
Hubbard--Stratonovich identity, or elementary-symmetric-polynomial calculation.

The problem is narrower. The interaction stage estimates a partition-function ratio from
a finite bank of baskets sampled from the exact additive parent. Ordinary sampling spends
almost all draws on common basket sizes. A rare large-size region can therefore receive no
draw even when a proposed interaction update makes that region important. An ESS computed
from the baskets that happened to be drawn cannot reveal an omitted region.

The solution is to represent basket-size regions deliberately, retain their exact parent
probability weights, and optimize a larger but still concave natural-parameter block.
The implementation contains an optional category update, but the canonical estimator fits
only \(C\) and \(\rho_0\). This distinction matters: a bounded experiment found that freeing
all category penalties overfit the sampled bank and failed cross-validation. The already
trained additive \(\rho_c\) is therefore retained unless a separate audit supports changing
it.

## 2. The unchanged probability law

For context \(x\), a nonempty basket \(S\) has probability

\[
p_\Theta(S\mid x)
=
\frac{\exp\{E_\Theta(S;x)\}}{Z_{\Theta,+}(x)},
\qquad
1\le |S|\le n_{\max},
\]

where

\[
E_\Theta(S;x)
=
\sum_{j\in S}b_j(x)
+\sum_{j<k;\,j,k\in S}K_{jk}
-\sum_c\rho_c {n_c(S)\choose2}
-\rho_0(|S|).
\]

Here \(K=\Phi\Phi^\top\succeq0\), \(n_c(S)\) is the number of products from category \(c\),
and \(n_{\max}=120\). All 5,455 products remain in the offered catalogue normalizer.

Let \(p_0(S\mid x)\) denote the converged additive parent obtained by setting \(K=0\).
The additive category/cardinality dynamic program computes its normalizer and its size law
exactly.

## 3. A concave natural-parameter block

Let \(U\in\mathbb R^{J\times r}\) be the rank-\(r\) product basis selected by the split-half
spectral audit, and write

\[
K=UCU^\top,
\qquad
0\preceq C\preceq \sigma_{\max}^2I.
\]

For a basket \(S\), define

\[
F_U(S)
=\frac12\left[
\left(\sum_{j\in S}u_j\right)
\left(\sum_{j\in S}u_j\right)^\top
-\sum_{j\in S}u_ju_j^\top
\right].
\]

Then

\[
\sum_{j<k;\,j,k\in S}K_{jk}
=\operatorname{tr}\{CF_U(S)\}.
\]

For the general natural block, consider an interaction matrix \(C\), an optional category
correction \(d=(d_1,\ldots,d_C)\), and an existing size-potential correction \(q(n)\). The log-density
increment relative to the additive parent is

\[
h_\eta(S)
=
\operatorname{tr}\{CF_U(S)\}
-\sum_c d_c {n_c(S)\choose2}
-q(|S|),
\]

where \(\eta=(C,d,q)\). This is affine in the natural parameters. In the default tested
configuration \(d=0\), so the optimized block is \((C,q)\). Omitting the frozen category
coordinates also removes their sparse matrix multiplication and storage from training.

The implementation represents \(q\) by its values at a modest set of size knots and uses
piecewise-linear interpolation between them. This does not create a different size model:
the interpolated values are written into the original \(\rho_0(1),\ldots,\rho_0(120)\)
table. It is an estimation regularizer for sparsely observed sizes.

## 4. Exact stratification identity

Partition the supported sizes into disjoint bands

\[
\mathcal B_1,\ldots,\mathcal B_L,
\qquad
\bigcup_{\ell=1}^L\mathcal B_\ell=\{1,\ldots,n_{\max}\}.
\]

The implemented default bands are

\[
1{:}4,\quad 5{:}10,\quad 11{:}20,\quad 21{:}40,\quad
41{:}59,\quad 60{:}80,\quad 81{:}120.
\]

The split at 60 is tied to the declared production audit \(P(N\ge60\mid x)\). A single
\(41{:}80\) band would still be unbiased, but its finite bank could spend every draw below
60 and estimate that particular tail functional poorly. The default allocation
\((16,16,12,8,5,4,3)\) keeps the total at 64 draws while explicitly representing both
sides of the safety boundary.

Put

\[
p_{0\ell}(x)=P_0(N\in\mathcal B_\ell\mid x).
\]

### Proposition 1 — stratified partition-ratio identity

\[
\frac{Z_{\eta,+}(x)}{Z_{0,+}(x)}
=
\sum_{\ell=1}^L
p_{0\ell}(x)
\mathbb E_0\!\left[
e^{h_\eta(S)}
\mid N\in\mathcal B_\ell,x
\right].
\]

### Proof

Because the bands partition the complete nonempty support, the law of total expectation
gives

\[
\mathbb E_0[e^{h_\eta(S)}\mid x]
=
\sum_\ell P_0(N\in\mathcal B_\ell\mid x)
\mathbb E_0[e^{h_\eta(S)}\mid N\in\mathcal B_\ell,x].
\]

Direct substitution of \(p_0(S\mid x)=e^{E_0(S;x)}/Z_{0,+}(x)\) shows that the left side is

\[
\sum_S\frac{e^{E_0(S;x)}}{Z_{0,+}(x)}e^{h_\eta(S)}
=\frac{Z_{\eta,+}(x)}{Z_{0,+}(x)}.
\]

This proves the identity. \(\square\)

## 5. The estimator and its expectation

For each context and band, draw independently

\[
S_{\ell d}\sim
p_0(S\mid N\in\mathcal B_\ell,x),
\qquad d=1,\ldots,D_\ell.
\]

The exact additive DP performs this in two steps:

1. draw \(N\) from the exact parent size law restricted to the selected band;
2. reverse-sample category counts and products exactly conditional on \(N\).

Define

\[
\widehat R_x(\eta)
=
\sum_{\ell=1}^L
\frac{p_{0\ell}(x)}{D_\ell}
\sum_{d=1}^{D_\ell}e^{h_\eta(S_{\ell d})}.
\]

### Proposition 2 — unbiasedness on the partition-function scale

For fixed allocations \(D_\ell\ge1\),

\[
\mathbb E[\widehat R_x(\eta)]
=\frac{Z_{\eta,+}(x)}{Z_{0,+}(x)}.
\]

### Proof

Each within-band sample mean is unbiased for its corresponding conditional expectation.
Multiplying by the exact band probability and summing gives Proposition 1. \(\square\)

This proposition does not say that \(\log\widehat R_x\) is unbiased. Jensen's inequality
gives

\[
\mathbb E[\log\widehat R_x]
\le
\log\mathbb E[\widehat R_x]
=
\log\frac{Z_{\eta,+}}{Z_{0,+}}.
\]

Consequently the sampled likelihood gain is optimistic at finite draw count. Independent
Smolyak evaluation remains the acceptance authority.

## 6. Concavity of the fixed-bank objective

For observed context-basket pairs \((x_m,S_m^{\mathrm{obs}})\), define

\[
\widehat G(\eta)
=
\frac1M\sum_{m=1}^M
\left[
h_\eta(S_m^{\mathrm{obs}})
-\log\widehat R_{x_m}(\eta)
\right].
\]

### Proposition 3 — global sampled target

With the conditional draws and their allocations frozen, \(\widehat G\) is concave in
\(\eta=(C,d,q)\). It remains concave after subtracting nonnegative quadratic ridge and
second-difference penalties.

### Proof

The observed term is affine in \(\eta\). The estimated ratio is a positive weighted sum of
exponentials of affine functions. Its logarithm is therefore a log-sum-exp and is convex.
An affine function minus a convex function is concave. Negative quadratic penalties are
also concave. The PSD spectral interval for \(C\), box constraints, and the fixed
\(q(1)=0\) gauge form a convex feasible set. \(\square\)

The implementation alternates two optimization blocks only for numerical conditioning:

1. bounded L-BFGS for the size coefficients and, only when enabled, category coefficients;
2. projected Armijo ascent for the small PSD matrix \(C\).

Both update the same concave objective. There is no Cholesky reparameterization and no
nonconvex optimizer hidden inside this stage.

## 7. Variance and allocation

Let

\[
\sigma_\ell^2(x;\eta)
=
\operatorname{Var}_0
\left[e^{h_\eta(S)}\mid N\in\mathcal B_\ell,x\right].
\]

Independence between strata gives

\[
\operatorname{Var}(\widehat R_x)
=
\sum_{\ell=1}^L
\frac{p_{0\ell}(x)^2\sigma_\ell(x;\eta)^2}{D_\ell}.
\]

If one draw in band \(\ell\) costs \(c_\ell\), minimizing this variance subject to a fixed
cost yields

\[
D_\ell
\propto
\frac{p_{0\ell}\sigma_\ell}{\sqrt{c_\ell}}.
\]

An independent pilot may estimate these quantities. The allocation is then frozen for the
optimization bank. Every declared band retains a positive draw floor. Current code starts
with a fixed conservative allocation; pilot adaptation is not yet part of the certified
pipeline.

Overall ESS is misleading under deliberate unequal stratum weights. The implementation
therefore reports ESS within each size band, where it diagnoses missing product-composition
modes rather than the intended difference in band probability.

## 8. Time and memory complexity

Let \(D=\sum_\ell D_\ell\), let \(r\) be interaction rank, and let \(k(S)\) be the number of occupied
categories in a sampled basket. The draw-bank statistics cost approximately

\[
O\left(MD\,[r^2+k(S)]\right).
\]

Stratification need not increase \(D\); it redistributes the existing draw budget. The
full \(\rho_0\) curve is represented by integer size indices and spline lookup. When an
experimental category update is enabled, its statistics are stored in CSR form. In the
default configuration those 300 frozen coordinates are omitted entirely. The
implementation does not allocate a dense

\[
M\times D\times(r^2+C+n_{\max})
\]

array.

The expensive forward category DP is evaluated once per minibatch. All conditional-size
draws reuse its tables. Reverse sampling has the same order as ordinary repeated parent
sampling. Within a category, the fast path uses mean-matched conditional-Bernoulli
rejection. If a sharp law has not accepted after 256 attempts, the sampler switches to an
exact log-domain elementary-symmetric backtrack. The fallback costs
\(O(J_c k + J_c D_c)\) for \(J_c\) offered products, requested count \(k\), and \(D_c\)
pending draws. It prevents a difficult but valid basket from terminating the run and does
not approximate or change the conditional law.

## 9. Size-phase diagnostic

Define the non-size catalogue pressure

\[
H_x(n)=\log\mathbb E_z[A_n(z,x)].
\]

Then

\[
P(N=n\mid x)
\propto
e^{H_x(n)-\rho_0(n)},
\]

so the adjacent-size log odds are

\[
\log\frac{P(N=n+1\mid x)}{P(N=n\mid x)}
=
\big[H_x(n+1)-H_x(n)\big]
-\big[\rho_0(n+1)-\rho_0(n)\big].
\]

This identifies whether an excessive size phase is created by catalogue pressure or by an
insufficient size penalty. The implemented report also separates the Gram contribution by
comparing the full child with its \(\Phi=0\) counterpart.

On the current accepted checkpoint's 64 highest-expected-size confirmed contexts, the
mean predicted size is 46.32 while the observed mean is 26.75. The Gram term adds only
about \(0.010\) to the average adjacent-size pressure over sizes 41--80, whereas the total
catalogue and \(\rho_0\) increments are each about \(5.4\) and nearly cancel. Thus the
remaining localized sensitivity is not principally caused by a large Gram contribution;
it arises because two large terms cancel and a small size-potential error changes the tail
shape. The exact output is `reports/size_phase_diagnostic.json`.

## 10. Required validation before a full run

The following tests must pass before this estimator may replace the current stage:

1. fixed-size reverse draws always have their requested size and valid assortment;
2. exact enumeration confirms the stratified ratio expectation;
3. finite differences confirm the complete sparse gradient;
4. line-segment tests confirm concavity;
5. accepted optimization steps are monotone;
6. adversarial rare-tail experiments show lower error at equal draw cost;
7. both cross-fit halves improve over the additive parent;
8. within-band ESS passes its predeclared floor;
9. independent Smolyak likelihood value and score audits pass;
10. complete-population local and aggregate tail audits pass; and
11. generation size, category and pair calibration do not regress.

The existing accepted checkpoint remains authoritative until all these gates pass. A
small software probe is not evidence of statistical improvement.

## 11. What the bounded experiment established

The final seven-band estimator was tested on a fixed 1,000-context training manifest with
64 draws per context and an independently scored 512-trip validation manifest. This is a
bounded gate for deciding whether a full run is justified; it is not the final full-panel
research result.

An earlier controlled experiment first freed \(C\), \(\rho_c\), and \(\rho_0\) together.
It failed cross-validation: its mean cross-fit gain was \(-0.00072\) nats/basket, one
half lost \(0.05585\) nats/basket, and
several category corrections reached their bounds. This is evidence against making the
largest proposed block the default; it is not evidence against the unchanged basket law.

The seven-band experiment froze \(\rho_c\) and fitted \(C\) with a smooth size correction
using
the predeclared penalties were

\[
\lambda_{\mathrm{size}}=10^{-3},
\qquad
\lambda_{\Delta^2}=10^{-1}.
\]

The two held-out halves gained \(0.01741\) and \(0.01260\) nats/basket, for a mean
cross-fit gain of \(0.01501\). The minimum within-band ESS fraction was \(0.564\), above
the predeclared \(0.20\) floor. On 512 independently evaluated validation trips, the
pre-household child gained

\[
0.01956\pm0.00645
\quad\text{nats/basket}
\]

over the exact additive parent. After the level-7 versus level-8 numerical allowance,
the lower 95% bound was \(0.00651\).

Before household recalibration, however, this candidate failed the population tail audit.
The already defined cross-fitted rank-one household size stage then removed the localized
failure. On 5,000 contexts, the pre-household model had five confirmed majority-tail
contexts and a calibrated aggregate-tail upper bound of \(0.00403\). After the
cross-fitted household correction, these became zero contexts and \(0.00267\),
respectively, below the \(0.0037\) limit. The household update itself had a
cross-fitted likelihood gain of \(0.00770\) nats/basket and a simultaneous lower bound of
\(0.00051\).

The final child retained an independently audited validation gain of

\[
0.01936\pm0.00928
\quad\text{nats/basket}.
\]

Its ordinary lower 95% bound was \(0.00116\), and its lower bound after the quadrature
allowance was \(0.00070\). Thus the complete bounded pipeline passes both likelihood and
tail gates. The margin is narrow enough that adoption still requires the predeclared
full-panel pipeline, but full-run compute is now justified.

## 12. Canonical full-run decision

The predeclared full execution completed on 2026-09-08. It used 12,000 training contexts,
64 draws per context, the seven bands above, rank five, frozen \(\rho_c\), and 37 active
natural parameters. The two cross-fit gains were \(0.02013\) and \(0.01960\)
nats/basket; mean cross-fit gain was \(0.01986\), and minimum within-band ESS fraction was
\(0.3993\).

On the locked 4,096-trip panels, the final post-household checkpoint gained

\[
0.026754\pm0.002376
\quad\text{on validation}
\]

and

\[
0.031009\pm0.002645
\quad\text{on test}
\]

over the exact additive parent. The respective lower 95% bounds after adjacent-rule
allowances were \(0.021744\) and \(0.025290\).

The complete 160,007-context population audit found zero majority-tail contexts. Its
bias-corrected \(P(N\ge60)\) was \(0.003445\), with upper 95% bound \(0.003570\) below
the predeclared \(0.003700\) limit. The full pipeline therefore passed and this estimator
is now canonical.

This decision does not convert unresolved downstream evidence into a theorem. Overall MRR
was \(0.095246\), while the interaction-only increment remained statistically
inconclusive. Generated basket-size mean and variance improved to \(7.26\) and \(72.97\)
but remain below the observed \(10.03\) and \(136.28\). Consequently the estimator is
accepted for model fitting and full-support likelihood, while unconditional generation
calibration remains an explicit development target.
