# Version-4 basket model: end-to-end theory for review

Status: **current theory and empirical contract**
Date: 2026-09-01

The foundational basket law is the one stated in [model.html](model.html) and
[version4.html](version4.html). The HTML files also contain historical empirical material,
which is not a description of the corrected cohort. This document reorganizes the theory
into one logical flow and states the current empirical contract. It does not propose a
replacement basket law.

---

## 1. The complete idea

A basket is a set \(S\), not a sequence. For a household and shopping occasion \(x\), the
model assigns an energy \(E_\Theta(S;x)\) to every feasible basket. Its probability is

\[
p_\Theta(S\mid x)
=
\frac{\exp\{E_\Theta(S;x)\}}
{\sum_{T\in\Omega_x^+}\exp\{E_\Theta(T;x)\}}.
\tag{1}
\]

The numerator scores one observed basket. The denominator scores all possible baskets and
is the partition function \(Z_+(x)\). With 5,455 products, direct enumeration is impossible.
The central theorem transforms \(Z_+\) into a low-dimensional Gaussian integral whose
integrand is computed exactly by elementary symmetric polynomials.

The logical flow is

1. context \(x\) determines item utilities \(b_j(x)\);
2. item utilities, interactions, category counts and total size determine basket energy;
3. the H--S/ESP theorem computes the normalizer and its derivatives;
4. observed sufficient statistics minus model-expected statistics give the likelihood
   score;
5. all original non-Gram incidence parameters are fitted exactly in the additive block,
   then the PSD interaction and confounded size directions are profiled together;
6. the fitted joint law supplies recommendation, generation and counterfactuals.

The statistical law, numerical estimator and optimizer are separate layers. Improving the
estimator must not change the statistical law.

---

## 2. Support, assortment and notation

For shopping occasion \(x\), let

\[
\mathcal A_x=\{\text{products offered on occasion }x\},
\]

\[
c(j)=\text{category of product }j,
\qquad
n(S)=|S|,
\qquad
n_c(S)=|S\cap\mathcal A_{xc}|.
\]

The declared non-empty support is

\[
\Omega_x^+
=
\left\{
S\subseteq\mathcal A_x:
1\le |S|\le n_{\max}
\right\},
\qquad n_{\max}=120.
\tag{2}
\]

Every observed basket used in likelihood evaluation must lie in its own assortment and
support. Numerical difficulty is not a reason to remove a basket or product from
\(\Omega_x^+\).

For the current dunnhumby experiment there is no observed stock/assortment feed. A sale is
evidence that an item was available, but a non-sale is not evidence that it was absent.
Accordingly the declared experimental support is

\[
\mathcal A_x=\{1,\ldots,5455\}
\]

at every modeled store. This support is fixed from the training-defined catalogue and does
not use held-out basket contents. A production deployment with a real stock feed should
replace \(\mathcal A_x\) by the externally observed offered set; this changes the context
support, not the energy law.

The context \(x\) contains the household, prices, promotions, calendar, store and any
declared history features. During simulation, every context variable must be externally
provided or be a deterministic function of simulated history.

### 2.1 Corrected empirical construction

The current experiment fixes all cohort decisions with training weeks 9--82:

1. rank products by training purchase-line frequency, breaking ties by product ID, and
   retain exactly 5,455;
2. retain households with 20--300 distinct training shopping days;
3. use weeks 83--90 only for validation and weeks 91--101 only for test; and
4. discard weeks outside 9--101 because the causal/promotion source does not cover them.

This gives 1,920 households and the following immutable split counts:

| Split | Weeks | Baskets | Purchase lines |
|---|---:|---:|---:|
| Training | 9--82 | 160,007 | 1,223,933 |
| Validation | 83--90 | 17,351 | 129,731 |
| Test | 91--101 | 23,340 | 181,342 |

No held-out outcome is used to select the catalogue or households, and no held-out line is
deleted after selection. Modal chain-week and store-week prices are computed from observed
transactions, with the training distribution supplying centering constants. The source
contains no inventory feed, so non-purchase is not evidence of unavailability; this is why
the declared support is the complete catalogue rather than a store-sales proxy.

The category index in the current implementation is a training-only affinity partition,
not the raw merchandising taxonomy. The unchanged deterministic construction produces 300
groups; 1,724 products form the residual group and the largest non-residual group contains
128 products. These values affect the empirical feature map \(c(j)\), but they do not alter
the category term in the energy.

The preprocessing manifest locks raw-file hashes, cohort membership, prices, promotion
coverage, split counts and full support. Details and independently reconstructed checks are
in [PREPROCESSING_AUDIT.md](PREPROCESSING_AUDIT.md).

---

## 3. Item utility

The version-4 item value can be written

\[
\begin{aligned}
b_{jht}
={}&
\lambda_j
+\theta_h^\top\alpha_j
-g_{hj}\,\Delta\log p_{jt} \\
&+w^{\mathrm{dsp}}_jD_{jt}
+w^{\mathrm{mlr}}_jM_{jt}
+\mu_j^\top\delta_{w(t)}
+\zeta_j^\top\xi_{s(t)}
+\psi_j^\top r_{jht},
\end{aligned}
\tag{3}
\]

where

\[
g_{hj}
=
\operatorname{softplus}(\gamma_h)^\top
\operatorname{softplus}(\beta_j)
\ge 0.
\tag{4}
\]

Therefore

\[
\frac{\partial b_{jht}}
{\partial\,\Delta\log p_{jt}}
=-g_{hj}\le0.
\tag{5}
\]

The sign of the own-price response is structural rather than left for the optimizer to
guess.

The implementation decomposes price into a trip-common movement \(m_t\) and a
product-relative deviation \(e_{jt}\):

\[
\Delta\log p_{jt}=m_t+e_{jt},
\tag{6}
\]

\[
b^{\mathrm{price}}_{jht}
=
-g_{hj}\left(m_t+\kappa e_{jt}\right),
\qquad
\kappa>0.
\tag{7}
\]

This is a parameterization within \(b_j(x)\); it does not alter the basket energy theorem.
A uniform-price counterfactual must update the common component as well as each raw price.

Recency is disabled in the current experiment because its train/test distribution is not
stable. That is a declared covariate choice, not a change to the normalizer theorem.

---

## 4. Basket energy and joint probability

For \(S\in\Omega_x^+\), the original version-4 energy is

\[
\boxed{
E_\Theta(S;x)
=
\sum_{j\in S}b_j(x)
+\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k
-\sum_c\rho_c{n_c(S)\choose2}
-\rho_0(|S|)
}.
\tag{8}
\]

The non-empty partition function and probability law are

\[
Z_+(x)
=
\sum_{T\in\Omega_x^+}
\exp\{E_\Theta(T;x)\},
\tag{9}
\]

\[
\boxed{
p_\Theta(S\mid x,S\neq\varnothing)
=
\frac{\exp\{E_\Theta(S;x)\}}{Z_+(x)}
}.
\tag{10}
\]

The four components have distinct roles:

- \(\sum b_j\) controls individual propensity;
- \(\phi_j^\top\phi_k\) gives low-rank complementarity;
- \(-\rho_c{n_c\choose2}\) gives structured within-category attraction or repulsion;
- \(-\rho_0(n)\) governs total basket size.

The matrix

\[
K=\Phi\Phi^\top
\tag{11}
\]

is positive semidefinite. Thus the Gaussian interaction represents attractive low-rank
structure. The explicit category term can represent structured repulsion that a real
Gaussian Gram matrix cannot.

### 4.1 Identified pair-specific complement coefficient

The coordinates of an interaction vector are not individually identified. For every
orthogonal matrix \(Q\), replacing \(\Phi\) by \(\Phi Q\) leaves

\[
(\Phi Q)(\Phi Q)^\top=\Phi\Phi^\top
\tag{11a}
\]

and hence the complete basket law unchanged. Interpretation must therefore use the Gram
matrix \(K=\Phi\Phi^\top\), not labels assigned to its axes.

Let \(T\) be a background basket containing neither product \(i\) nor \(j\), and put
\(t=|T|\). The exact energy cross-difference is

\[
\begin{aligned}
\Delta_{ij}E(T;x)
&=E(T\cup\{i,j\};x)-E(T\cup\{i\};x)
-E(T\cup\{j\};x)+E(T;x)\\
&=K_{ij}-\rho_{c(i)}\mathbf 1\{c(i)=c(j)\}
-\Delta^2\rho_0(t),
\end{aligned}
\tag{11b}
\]

where

\[
\Delta^2\rho_0(t)=\rho_0(t+2)-2\rho_0(t+1)+\rho_0(t).
\tag{11c}
\]

The last term is common to every candidate product pair at the same background size. The
identified *pair-specific* coefficient is therefore

\[
\boxed{
\gamma_{ij}=K_{ij}-\rho_{c(i)}\mathbf 1\{c(i)=c(j)\}
=\phi_i^\top\phi_j-\rho_{c(i)}\mathbf 1\{c(i)=c(j)\}.
}
\tag{11d}
\]

For products in different affinity groups, \(K_{ij}>0\) is the model's direct complement
coefficient. At fixed background size, it multiplies pair-specific conditional odds
relative to the common size curvature by \(\exp(K_{ij})\). For products in the same group,
the explicit category coefficient must be included; reporting \(K_{ij}\) alone would
misstate the Version-4 law. This is a predictive association parameter. Neither a
positive Gram entry nor observed co-incidence identifies a causal cross-price response.

### 4.2 Complete-support admissibility of the category coefficient

The affinity partition is an estimation device, not a claim that every pair in a broad
group is equally complementary. Because the category statistic is quadratic, a weakly
negative coefficient on a broad group can otherwise create a remote large-basket phase
that is almost invisible to ordinary minibatches.

Let

\[
m_c=\max_x\min\{|A_x\cap c|,n_{\max}\},
\tag{11e}
\]

where (A_x) is the offered assortment. The fitted parameter is restricted to

\[
\boxed{(-\rho_c)_+{m_c\choose2}\le B},
\qquad B=1.5\text{ nats in the declared pipeline}.
\tag{11f}
\]

Equivalently, for (m_c\ge2),

\[
\rho_c\ge -\frac{B}{{m_c\choose2}}.
\tag{11g}
\]

This is an optimization-domain constraint on the coefficient already present in Eq. (8). It
does not modify the energy, joint law, support, incidence formula, or normalizer theorem.
A two-item group retains the old lower bound \(-1.5\), whereas a 120-item group has lower
bound \(-1.5/{120\choose2}\). Strong attraction therefore remains available for a small
specific bundle, but cannot be extrapolated as a complete clique across a broad group.

**Proposition 1 (category support bound).** Under Eq. (11f), category \(c\)'s attractive
contribution to every supported basket is at most \(B\).

**Proof.** For every supported \(S\), \(0\le n_c(S)\le m_c\), and
\({n\choose2}\) is nondecreasing for integer \(n\ge0\). If \(\rho_c\ge0\), the term
\(-\rho_c{n_c(S)\choose2}\le0\) is not attractive. If \(\rho_c<0\), then

\[
-\rho_c{n_c(S)\choose2}
\le(-\rho_c){m_c\choose2}
\le B.
\]

This proves the claim. \(\square\)

Projection is coordinate-wise clipping and costs (O(C)) after an optimizer update. At
an active lower bound, outward Adam first moments are cleared so stale momentum does not
keep proposing the same inadmissible move. No quadrature node or dynamic-program state is
added.

### 4.2 The empty basket

If \(Z_0\) includes the empty basket, then \(E(\varnothing)=0\) and

\[
Z_+(x)=Z_0(x)-1.
\tag{12}
\]

Numerically, \(Z_+\) must be computed by omitting the degree-zero contribution. Computing
\(\exp(\log Z_0)-1\) subtracts nearly equal quantities when \(Z_0\approx1\).

---

## 5. The H--S/ESP normalizer theorem

The pair interaction satisfies

\[
\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k
=
\frac12
\left\|
\sum_{j\in S}\phi_j
\right\|^2
-\frac12
\sum_{j\in S}\|\phi_j\|^2.
\tag{13}
\]

Define

\[
\widetilde b_j(x)
=
b_j(x)-\frac12\|\phi_j\|^2,
\tag{14}
\]

\[
w_j(z;x)
=
\exp\left\{
\widetilde b_j(x)+z^\top\phi_j
\right\},
\qquad
z\sim\mathcal N(0,I_r),
\tag{15}
\]

where \(r\) is the active interaction rank.

For category \(c\), let \(e_k\) denote the elementary symmetric polynomial of degree \(k\)
in its offered product weights. Define the category polynomial

\[
G_c(u;z,x)
=
\sum_{k=0}^{\min(|\mathcal A_{xc}|,n_{\max})}
\exp\left\{-\rho_c{k\choose2}\right\}
e_k\!\left(
\{w_j(z;x):j\in\mathcal A_{xc}\}
\right)u^k.
\tag{16}
\]

Let \(A_n(z,x)\) be defined by

\[
\prod_cG_c(u;z,x)
=
\sum_{n=0}^{n_{\max}}A_n(z,x)u^n.
\tag{17}
\]

### Theorem 1 — exact reduction of the partition function

\[
\boxed{
Z_+(x)
=
\mathbb E_{z\sim\mathcal N(0,I_r)}
\left[
\sum_{n=1}^{n_{\max}}
e^{-\rho_0(n)}A_n(z,x)
\right]
}.
\tag{18}
\]

### Proof

Let

\[
v_S=\sum_{j\in S}\phi_j.
\tag{19}
\]

Using (13),

\[
\begin{aligned}
\exp\{E_\Theta(S;x)\}
={}&
\exp\left\{
-\rho_0(n)
-\sum_c\rho_c{n_c\choose2}
\right\}\\
&\times
\prod_{j\in S}e^{\widetilde b_j(x)}
\exp\left\{\frac12\|v_S\|^2\right\}.
\end{aligned}
\tag{20}
\]

The Gaussian moment-generating identity gives

\[
\mathbb E_{z\sim\mathcal N(0,I_r)}
\left[e^{z^\top v}\right]
=
e^{\|v\|^2/2}.
\tag{21}
\]

Substituting (21) into (20) converts each product contribution to
\(w_j(z;x)\). The subset sum is finite, so expectation and summation may be exchanged.
Within category \(c\), the sum over all \(k\)-product subsets equals the elementary
symmetric polynomial \(e_k\). The variable \(u\) records total degree, so multiplying the
category polynomials enforces

\[
n=\sum_c n_c.
\tag{22}
\]

Finally \(e^{-\rho_0(n)}\) supplies the total-size potential, and omitting \(n=0\) enforces
non-empty support. This proves (18). \(\square\)

### Consequence

The theorem replaces a sum over \(2^{5455}\) subsets by:

1. an exact ESP/category dynamic program at fixed \(z\); and
2. an \(r\)-dimensional Gaussian integral.

Catalogue size affects the fixed-node dynamic program. The dimension of the outer integral
is the active interaction rank.

---

## 6. Elementary symmetric polynomials

For weights \(a_1,\ldots,a_m\),

\[
\prod_{j=1}^m(1+a_ju)
=
\sum_{k=0}^m e_k(a_1,\ldots,a_m)u^k.
\tag{23}
\]

The subtraction-free recursion is

\[
e_k^{(j)}
=
e_k^{(j-1)}
+a_je_{k-1}^{(j-1)},
\qquad
e_0^{(0)}=1.
\tag{24}
\]

This recursion is numerically preferable to alternating Newton identities, whose large
terms cancel at the required degrees. Per-category polynomials are multiplied in a
balanced tree and truncated at \(n_{\max}\).

---

## 7. Likelihood and gradients

Ignoring the separate quantity factor, one observed basket contributes

\[
\boxed{
\ell_x(\Theta)
=
E_\Theta(S_x;x)-\log Z_+(x)
}.
\tag{25}
\]

This is one joint objective. There is no independent recommendation objective and no
separately weighted interaction objective.

Because \(\log Z_+\) is a cumulant-generating function,

\[
\frac{\partial\log Z_+(x)}{\partial b_j}
=
\mathbb E_\Theta[x_j\mid x,S\neq\varnothing]
=
P_\Theta(j\in S\mid x,S\neq\varnothing)
=:\pi_j(x),
\tag{26}
\]

\[
\frac{\partial^2\log Z_+(x)}
{\partial b_j\,\partial b_k}
=
\operatorname{Cov}_\Theta(x_j,x_k\mid x,S\neq\varnothing).
\tag{27}
\]

Hence the utility score is

\[
\frac{\partial\ell_x}{\partial b_j}
=
x_j^{\mathrm{obs}}-\pi_j(x).
\tag{28}
\]

The size-potential score is

\[
\frac{\partial\ell_x}{\partial\rho_0(n)}
=
-\mathbf1\{|S_x|=n\}
+P_\Theta(|S|=n\mid x).
\tag{29}
\]

The category score compares observed and expected \({n_c\choose2}\). The interaction score
compares observed and model-expected pair structure. Every block has the same
positive-phase minus negative-phase interpretation.

Two exact checks follow:

\[
\sum_j\pi_j(x)
=
\mathbb E_\Theta[|S|\mid x],
\tag{30}
\]

\[
\frac{\partial\log Z_+}{\partial\rho_0(n)}
=
-P_\Theta(|S|=n\mid x).
\tag{31}
\]

### 7.1 Quantity likelihood

If \(q_j\ge1\) is the purchased quantity, the complete trip likelihood is

\[
\ell_x^{\mathrm{complete}}
=
E_\Theta(S_x;x)-\log Z_+(x)
+\sum_{j\in S_x}\log P(q_j\mid j\in S_x,x).
\tag{32}
\]

The present conditional quantity law is shifted negative binomial. Since it sums to one
conditional on incidence, it does not alter \(Z_+\).

---

## 8. Nesting relative to the no-interaction model

The matched no-interaction model is obtained by setting

\[
\Phi=0.
\tag{33}
\]

### Proposition 2 — training-likelihood nesting

If the two models use the same training baskets, support, covariates and admissible
non-interaction parameters, and both exact objectives are globally optimized, then

\[
\max_{\Theta,\Phi}
\ell_{\mathrm{train}}(\Theta,\Phi)
\ge
\max_\Theta
\ell_{\mathrm{train}}(\Theta,0).
\tag{34}
\]

### Proof

The feasible set on the left contains every feasible point \((\Theta,0)\) on the right.
\(\square\)

This does not guarantee that finite stochastic optimization finds the larger maximum. It
also does not guarantee a validation or MRR improvement. If the interaction model loses on
the exact training likelihood, the implementation or optimizer is defective. If it wins
on training and loses on validation, overfitting or distribution shift remains possible.

---

## 9. Why an exactly zero \(\Phi\) cannot learn by first-order descent

Let

\[
\Phi=\varepsilon V.
\tag{35}
\]

The pair energy becomes

\[
\sum_{j<k}\phi_j^\top\phi_kx_jx_k
=
\varepsilon^2
\sum_{j<k}v_j^\top v_kx_jx_k.
\tag{36}
\]

Therefore

\[
\left.
\frac{\partial\ell}{\partial\Phi}
\right|_{\Phi=0}
=0.
\tag{37}
\]

The zero interaction point is a singular coordinate point, not an ordinary point at which
a nonzero first derivative can reveal pair signal.

Let \(X(S)\) be the off-diagonal pair-incidence matrix, and at an additive fit define

\[
R
=
\mathbb E_{\mathrm{data}}[X(S)]
-\mathbb E_{p_{\Theta,0}}[X(S)].
\tag{38}
\]

For a small rank-one activation, the second-order likelihood coefficient is

\[
\frac12v^\top Rv.
\tag{39}
\]

Positive eigenvectors of \(R\) are locally improving PSD Gram directions. Split-sample
subspace agreement determines whether those directions are stable enough to initialize.

This justifies a rank audit after the exact additive optimum. It does not require a noisy
factor-coordinate optimizer. The interaction block can instead be optimized in its
identifiable natural parameter \(C\), while the size direction most strongly confounded
with positive interactions is profiled in the same solve.

---

## 10. Correct end-to-end fitting flow

### Phase A — fresh initialization

Initialize every original version-4 parameter from a fixed seed or declared data summary.
A run called fresh must not load a trained checkpoint.

### Phase B — exact additive warm start

Set \(\Phi=0\). Then the Gaussian integrand is constant and the category/cardinality DP
computes the normalizer exactly. Optimize all other declared incidence parameters.

This phase cheaply places utilities, category terms and the size potential in a sensible
region. It is not the final interaction fit.

### Phase C — interaction rank identification

Estimate (38), test eigen-directions on split samples, choose the largest stable rank and
obtain an orthonormal product basis \(U\). This phase selects coordinates; it does not
assign the final interaction magnitude.

### Phase D — deterministic natural-parameter likelihood

Write the Gram matrix in the certified basis as

\[
K=UCU^\top,\qquad 0\preceq C\preceq \sigma_{\max}^2I.
\tag{40}
\]

For a basket \(S\), define

\[
F_U(S)=\frac12\left[
\left(\sum_{j\in S}u_j\right)
\left(\sum_{j\in S}u_j\right)^\top
-\sum_{j\in S}u_ju_j^\top
\right].
\tag{40a}
\]

Then the original Version-4 interaction energy is exactly

\[
\sum_{j<k\,;\,j,k\in S}\phi_j^\top\phi_k
=\operatorname{tr}\{C F_U(S)\}.
\tag{40b}
\]

No interaction term has been removed or approximated in (40b). Only its coordinates have
changed from a nonidentifiable factor to its PSD natural parameter.

Let \(S_{md}\) be fixed exact draws from the converged additive parent for context \(m\).
To profile the size response induced by positive interactions, use directions already
contained in the original free size potential,

\[
\Delta\rho_0(n)=a(n/10)+c(n/10)^2.
\tag{40c}
\]

Set

\[
h_{C,a,c}(S)=\operatorname{tr}\{CF_U(S)\}-\Delta\rho_0(|S|).
\tag{40d}
\]

The common-random-number Monte Carlo likelihood ratio is

\[
\widehat G(C,a,c)
=\frac1M\sum_{m=1}^M\left[
h_{C,a,c}(S_m^{\mathrm{obs}})
-\log\left\{\frac1D\sum_{d=1}^D
\exp h_{C,a,c}(S_{md})\right\}
\right].
\tag{40e}
\]

Each summand is a linear function minus log-sum-exp, hence concave. With

\[
c\ge0,\qquad a+(n_{\max}/10)c\ge0,
\tag{40f}
\]

the feasible set is convex and the size correction cannot make the maximum supported
size more attractive than under its additive parent. Therefore projected ascent with
backtracking has one global sampled target; it is not a search over \(\rho_0\)
initializations. After solving, an eigendecomposition \(C=LL^\top\) gives
\(\Phi=UL\), including an interaction embedding for every product.

All non-Gram incidence parameters were already trained from scratch in Phase B. Holding
them at their exact additive optimum while fitting (40e) is a block-coordinate/profile
optimization choice, not a change to the joint law. The size block is updated jointly
because it is the first-order nuisance direction created by a positive Gram kernel.
Independent high-order Smolyak likelihood, rather than the sampled training objective,
decides whether the resulting block update is accepted.

### Phase E — convergence and test

Use a fixed representative validation panel for scheduling and checkpoint selection. Read
the untouched test panel once after all modeling and estimator decisions are frozen.

Calling Phases B--C “end-to-end training” without fitting and certifying Phase D would be
incorrect.

---

## 11. Smolyak quadrature

Define the positive integrand

\[
F_+(z;x)
=
\sum_{n=1}^{n_{\max}}
e^{-\rho_0(n)}A_n(z,x).
\tag{41}
\]

Then

\[
Z_+(x)
=
\int_{\mathbb R^r}
F_+(z;x)\varphi_r(z)\,dz.
\tag{42}
\]

Let \(U^i\) be the probabilists' one-dimensional Gauss--Hermite rule at level \(i\). The
isotropic Smolyak combination rule is

\[
\mathcal A(q,r)
=
\sum_{q-r+1\le|\boldsymbol i|\le q}
(-1)^{q-|\boldsymbol i|}
{r-1\choose q-|\boldsymbol i|}
\bigotimes_{k=1}^rU^{i_k}.
\tag{43}
\]

For active rank seven:

| Level | Nodes | Proper role |
|---|---:|---|
| \(q=8\) | 15 | coarse training score |
| \(q=9\) | 127 | target likelihood and target score |
| \(q=10\) | 785 | independent fidelity audit |

Smolyak weights are signed. Consequently the implementation must:

1. scale node contributions before summation;
2. perform a true signed sum, not a log-sum-exp;
3. reject a non-positive estimated partition rather than clamp it; and
4. report the cancellation condition.

All 5,455 products remain in each fixed-node ESP calculation. A checkpoint stored with 32
columns but only seven active directions must use a seven-dimensional rule embedded into
those columns, not a 32-dimensional Smolyak grid.

---

## 12. The estimator must reproduce the score, not only \(\log Z\)

Optimization uses

\[
\nabla_\Theta\ell
=
\nabla_\Theta E(S;x)
-\nabla_\Theta\log Z_+(x).
\tag{44}
\]

Close scalar normalizers can have different slopes. Nested-level audits must therefore
measure, for every important parameter block \(b\),

\[
\delta_b^{(q)}
=
\frac{
\left\|g_b^{(q)}-g_b^{(q+1)}\right\|
}{
\left\|g_b^{(q+1)}\right\|+\epsilon
},
\tag{45}
\]

\[
\operatorname{cos}_b^{(q)}
=
\frac{
\left(g_b^{(q)}\right)^\top g_b^{(q+1)}
}{
\left\|g_b^{(q)}\right\|
\left\|g_b^{(q+1)}\right\|
+\epsilon
}.
\tag{46}
\]

At minimum, report these for \(\Phi,\lambda,\rho_c,\rho_0\) and the concatenated incidence
score. Identity (30) independently checks the common-utility derivative.

The primary error contract concerns the population mean likelihood and score. The maximum
tripwise discrepancy remains a diagnostic and identifies contexts requiring investigation,
but it must not silently redefine a mean objective. Difficult trips may not be dropped.

---

## 13. Coarse and fine gradient theory

### 13.1 Direct level-8 optimization is inexact

Let \(L(\Theta)\) be the negative level-9 likelihood and suppose level 8 supplies

\[
\widehat g_t
=
\nabla L(\Theta_t)+e_t,
\qquad
\|e_t\|\le\delta.
\tag{47}
\]

### Proposition 3 — error floor of a fixed biased score

If \(L\) has \(M\)-Lipschitz gradient, fixed score error generally permits convergence only
to a neighbourhood satisfying

\[
\|\nabla L(\Theta)\|=O(\delta).
\tag{48}
\]

Exact stationarity requires \(\delta_t\to0\), or an unbiased stochastic error together with
the usual diminishing-step conditions.

### Proof sketch

For an update \(\Theta^+=\Theta-\eta\widehat g\), smoothness gives

\[
L(\Theta^+)
\le
L(\Theta)
-\eta\nabla L(\Theta)^\top\widehat g
+\frac{M\eta^2}{2}\|\widehat g\|^2.
\tag{49}
\]

The adverse inner product is bounded by

\[
\left|\nabla L(\Theta)^\top e\right|
\le
\delta\|\nabla L(\Theta)\|.
\tag{50}
\]

When \(\|\nabla L\|\) is of order \(\delta\), this error can cancel the descent term.
\(\square\)

Thus level 8 is defensible for coarse movement while its score error is small relative to
the learning signal. Level 8 alone is not enough for an unqualified final level-9 MLE claim.

### 13.2 Unbiased multifidelity level-9 score

Let \(g_8(x)\) and \(g_9(x)\) be negative-phase gradients from the two rules. On a minibatch
of \(B\) trips, select a uniformly random subset \(U\) of size \(m\) and define

\[
\widehat g_9
=
\frac1B\sum_{i=1}^Bg_8(x_i)
+\frac1m\sum_{i\in U}
\left[g_9(x_i)-g_8(x_i)\right].
\tag{51}
\]

Conditioned on the minibatch,

\[
\mathbb E_U[\widehat g_9]
=
\frac1B\sum_{i=1}^Bg_9(x_i).
\tag{52}
\]

### Proof

Each minibatch index has inclusion probability \(m/B\). Therefore

\[
\mathbb E_U
\left[
\frac1m\sum_{i\in U}(g_9-g_8)(x_i)
\right]
=
\frac1B\sum_{i=1}^B(g_9-g_8)(x_i).
\tag{53}
\]

Adding the level-8 mean proves (52). \(\square\)

The observed positive phase is common to both rules and cancels in the correction.

### 13.3 Proposed accuracy schedule

The defensible next-run schedule is:

1. use direct \(q=8\) only during the coarse phase;
2. switch to (51) before declaring convergence;
3. increase \(m\), or use direct \(q=9\), as the net score shrinks;
4. use \(q=10\) only on a fixed audit panel; and
5. stop if quadrature score uncertainty is not smaller than the remaining learning signal.

This is how speed is gained without changing the final target.

### 13.4 Telescoping three-level score

For a lower centre rule, write the target score exactly as

\[
g_{r+2}=g_r+(g_{r+1}-g_r)+(g_{r+2}-g_{r+1}).
\tag{53a}
\]

Compute \(g_r\) on all \(B\) minibatch contexts and estimate the two differences on
independently uniform subsets \(U_1,U_2\), of sizes \(m_1,m_2\):

\[
\widehat g_{r+2}
=\bar g_r^{\,B}
+\overline{(g_{r+1}-g_r)}^{\,U_1}
+\overline{(g_{r+2}-g_{r+1})}^{\,U_2}.
\tag{53b}
\]

By the same inclusion-probability argument as (52)--(53), each correction mean is
conditionally unbiased for its full-minibatch counterpart. Linearity of expectation then
gives

\[
\mathbb E[\widehat g_{r+2}\mid B]=\bar g_{r+2}^{\,B}.
\tag{53c}
\]

This changes only the computation schedule. It does not change the Version-4 likelihood,
the target quadrature rule, or any model parameter. The two correction sizes control
variance and can be increased without changing the expectation.

---

## 14. Complexity

Let

- \(B\) be minibatch trips;
- \(T\) be offered product slots in the batch;
- \(C_B\) be trip-category rows;
- \(R=n_{\max}\);
- \(r\) be active rank; and
- \(P\) be quadrature nodes.

For one node, embedding projection costs \(O(Tr)\). The subtraction-free category ESP costs
approximately \(O(TR)\), subject to bucket and product-tree constants. Category polynomial
products are truncated at \(R\). Denote one measured fixed-node full-catalogue cost by

\[
\mathcal C_F(B,T,C_B,R,r).
\tag{54}
\]

A \(P\)-node rule then costs approximately

\[
O\!\left(
P\,\mathcal C_F(B,T,C_B,R,r)
\right).
\tag{55}
\]

For rank seven, \(B=24\) and correction size \(m=4\), the node-trip work is

\[
\begin{aligned}
\text{direct }q8
&:\quad 24(15)=360,\\
\text{multifidelity }q9
&:\quad 24(15)+4(127+15)=928,\\
\text{direct }q9
&:\quad 24(127)=3048,\\
\text{direct }q10
&:\quad 24(785)=18840.
\end{aligned}
\tag{56}
\]

This explains why direct \(q=8\) is suitable early, multifidelity is suitable late, and
direct \(q=10\) is restricted to small audits.

The tested 128-node SVD-aligned positive tensor rule was less accurate than the 127-node
level-9 Smolyak rule on identical trips and is rejected.

---

## 15. Numerical envelopes are not foundational theory

The recent probes used

\[
\|\Phi\|_2\le1.5
\tag{57}
\]

to remain within the empirically audited Smolyak region. Equation (57) is not a theorem of
version 4. It is a numerical restriction. If the validation likelihood score points
outward at this boundary, the estimator is suppressing a potentially useful model
direction.

The Euclidean projection onto an operator-norm ball is

\[
\Phi
=
U\operatorname{diag}(s_1,\ldots,s_r)V^\top
\longmapsto
U\operatorname{diag}\!\left(
\min(s_1,c),\ldots,\min(s_r,c)
\right)V^\top.
\tag{58}
\]

Uniformly scaling every singular value when only \(s_1>c\) is not this projection and
destroys feasible interaction directions.

At a constrained optimum the correct diagnostic is the projected/KKT gradient. We must
measure whether the score at the boundary is outward normal, tangential or inward before
calling a boundary checkpoint converged.

---

## 16. Gauge fixing and identifiability

Several parameterizations have exact flat directions. For example,

\[
\lambda_j+\mu_j^\top\delta_w
\tag{59}
\]

is invariant to compensating transfers between \(\lambda_j\) and the mean of
\(\delta_w\). Household taste and store effects have analogous gauges.

The context-side factors \(\theta,\delta,\xi\) are therefore centred after an optimizer
step. This leaves every value \(b_j(x)\) unchanged while choosing a unique representation.

There is also a utility/size gauge. Adding a constant \(a\) to every product utility adds
\(a|S|\) to a basket energy. The transformation

\[
\lambda_j^+=\lambda_j-a,
\qquad
\rho_0^+(n)=\rho_0(n)-an
\tag{60}
\]

leaves (8) unchanged. Any centring operation must transfer this level exactly.

Finally, for orthogonal \(Q\),

\[
(\Phi Q)(\Phi Q)^\top=\Phi\Phi^\top.
\tag{61}
\]

Only the Gram matrix is substantively identified; the orientation of latent columns is not.

---

## 17. Basket-size response and price counterfactuals

Consider a utility perturbation

\[
b\longmapsto b+\varepsilon d.
\tag{62}
\]

### Proposition 4 — covariance response identity

\[
\boxed{
\frac{d\,\mathbb E_\Theta[n]}{d\varepsilon}
=
\operatorname{Cov}_\Theta
\left(n,d^\top x\right)
}.
\tag{63}
\]

For a uniform utility shift \(d=\mathbf1\),

\[
\boxed{
\frac{d\,\mathbb E_\Theta[n]}{d\varepsilon}
=
\operatorname{Var}_\Theta(n)
}.
\tag{64}
\]

### Proof

Since \(n=\mathbf1^\top x\), equation (27) gives

\[
\begin{aligned}
\frac{d\,\mathbb E[n]}{d\varepsilon}
&=
\mathbf1^\top
\operatorname{Cov}(x)d\\
&=
\operatorname{Cov}
\left(\mathbf1^\top x,d^\top x\right).
\end{aligned}
\tag{65}
\]

Setting \(d=\mathbf1\) proves (64). \(\square\)

The size distribution and aggregate price response are therefore linked. Underestimating
\(\operatorname{Var}(n)\) limits the aggregate response to a common price movement,
regardless of how expressive the product price loadings are.

A correct price counterfactual must:

1. change the action prices;
2. recompute the common and relative price components in \(b_j(x)\);
3. recompute \(Z_+\), size probabilities and incidences;
4. report paired changes in size, incidence, units, revenue and margin;
5. verify small positive and negative perturbations; and
6. remain inside the estimator's audited context and parameter envelope.

These are structural responses of the fitted probability law. They become causal
elasticities only under a credible identification design. Observational likelihood alone
does not eliminate price endogeneity, promotion targeting or stockout confounding.

---

## 18. Recommendation follows from the joint law

### 18.1 Unconditional incidence

For a household and occasion, the natural propensity score is

\[
\pi_j(x)
=
\frac{\partial\log Z_+(x)}{\partial b_j}.
\tag{66}
\]

Raw \(b_j\) is only standalone utility. Incidence \(\pi_j\) includes competition for basket
slots, the category and size laws, and all interactions.

### 18.2 Conditioning on a revealed basket

For revealed set \(R\), define a completion \(T\) with \(T\cap R=\varnothing\). Then

\[
p_\Theta(T\mid R,x)
\propto
\exp\left\{
E_\Theta(R\cup T;x)-E_\Theta(R;x)
\right\}.
\tag{67}
\]

For candidate \(j\notin R\),

\[
\log
\frac{
p_\Theta(T\cup\{j\}\mid R,x)
}{
p_\Theta(T\mid R,x)
}
=
b_j(x)
+\sum_{k\in R\cup T}\phi_j^\top\phi_k
-\rho_{c(j)}n_{c(j)}(R\cup T)
-\Delta\rho_0,
\tag{68}
\]

where

\[
\Delta\rho_0
=
\rho_0(|R\cup T|+1)-\rho_0(|R\cup T|).
\tag{69}
\]

The H--S/ESP calculation can be conditioned on \(R\) by shifting remaining utilities and
support. Recommendation should rank conditional incidences from (67), not an ad-hoc sum of
utility and interaction terms.

### 18.3 MRR definitions

Let \(r_i\) be the rank of held-out target product \(j_i^*\) among the same eligible
candidates for every method. Untruncated MRR is

\[
\operatorname{MRR}
=
\frac1N\sum_{i=1}^N\frac1{r_i}.
\tag{70}
\]

At cutoff \(K\),

\[
\operatorname{MRR}@K
=
\frac1N
\sum_{i=1}^N
\frac{\mathbf1\{r_i\le K\}}{r_i},
\tag{71}
\]

\[
\operatorname{Recall}@K
=
\frac1N
\sum_{i=1}^N
\mathbf1\{r_i\le K\}.
\tag{72}
\]

An unqualified logged MRR means (70), not MRR@5, @10 or @20.

### 18.4 Why likelihood and MRR are related but not equivalent

Likelihood scores the calibrated probability of the entire basket over complete support.
MRR uses only the rank of one or more held-out products. A likelihood gain may come from
better size, category or tail calibration and barely change top ranks. A ranking change can
improve MRR while worsening probability calibration.

Product-discrimination improvements may raise both, but there is no theorem making MRR a
monotone function of joint likelihood.

---

## 19. Basket generation

This section gives the generating-law decomposition. The sampler actually used by the
pipeline—interaction tempering, exact conditional category/cardinality draws, positive
SMC resampling, Hubbard--Stratonovich blocked mutation, finite-particle guarantees,
counterfactual reweighting and diagnostics—is derived in
[INFERENCE_AND_SIMULATION.md](INFERENCE_AND_SIMULATION.md).

The augmented latent density corresponding to (18) is

\[
p_\Theta(z\mid x,S\neq\varnothing)
=
\frac{
\varphi_r(z)F_+(z;x)
}{
Z_+(x)
}.
\tag{73}
\]

Given an exact draw of \(z\), generation is:

### Step 1 — total size

\[
P(n\mid z,x)
=
\frac{
e^{-\rho_0(n)}A_n(z,x)
}{
\sum_{m=1}^{n_{\max}}e^{-\rho_0(m)}A_m(z,x)
}.
\tag{74}
\]

### Step 2 — category counts

Given \(n\), draw \((r_c)_c\) satisfying

\[
\sum_cr_c=n
\tag{75}
\]

by a backward pass through prefix/suffix category-polynomial products.

### Step 3 — products within categories

For a category containing weights \(w_1,\ldots,w_M\), conditional on selecting \(r\)
products, the inclusion probability for item \(j\) is proportional to

\[
w_j
e_{r-1}(w_1,\ldots,w_{j-1},w_{j+1},\ldots,w_M).
\tag{76}
\]

Sequential suffix-ESP sampling produces an exact \(r\)-product subset.

### Step 4 — quantities

For every included product, draw \(q_j\) from its conditional shifted-negative-binomial
law.

Steps 1--4 after \(z\) are exact and retain complete declared support.

### 19.1 The outer \(z\) draw is a separate numerical problem

Step 4 belongs to the complete theoretical factorization. The selected fresh pipeline
currently fits and certifies Steps 1--3 only; its quantity parameters are not trained or
authorized for unit-sales claims.

Signed Smolyak weights do not define a probability distribution and must never be treated
as posterior resampling probabilities. Importance resampling for (73) is asymptotically
exact as proposal count grows, but finite-sample correctness depends on global mode
coverage. A high ESS only says sampled weights agree; it does not prove that an unsampled
mode has negligible mass.

Consequently:

- likelihood quadrature does not automatically certify generation;
- a positive outer sampler needs its own nested or replicate error audit;
- MCMC targeting (73) is valid only with convergence diagnostics; and
- generated size, incidence, pair, category and tail moments must be compared with the
  model moments and held-out data.

The selected implementation satisfies the positivity requirement by starting from exact
draws at the no-Gram bridge and moving to the full interaction law with positive-weight
SMC. It never interprets signed Smolyak nodes as sampling probabilities.

---

## 20. Retailer simulation and an MDP

A repeated retailer simulator can define:

\[
\text{state}_t
=
(\text{household},\text{calendar},\text{store},
\text{observed history},\text{exogenous inputs}),
\tag{77}
\]

\[
\text{action}_t
=
(\text{prices},\text{coupons},\text{display},
\text{mailer},\text{assortment}),
\tag{78}
\]

\[
\text{transition}_t:
\quad
S_t\sim p_\Theta(S\mid x_t,a_t),
\qquad
q_{jt}\sim P(q_j\mid j\in S_t,x_t,a_t),
\tag{79}
\]

followed by deterministic history updates.

A retailer reward may be

\[
R_t
=
\sum_{j\in S_t}
q_{jt}
\left(
p_{jt}-c_{jt}
\right)
-\text{promotion cost}
-\text{stockout cost}.
\tag{80}
\]

This is an MDP only if the state is closed. If inventory or consumption is latent and not
observed, the correct object is a POMDP over beliefs, not an MDP with invented inventory.

Before policy optimization, the environment needs:

1. multi-step rollout stability;
2. causal or experimental validation of interventions;
3. action-support and extrapolation checks;
4. assortment and stockout handling;
5. revenue and quantity calibration; and
6. uncertainty analysis.

A predictive basket model is not automatically a causal digital twin.

### 20.1 Implemented finite-budget promotion environment

The implemented first policy problem deliberately uses a smaller state than (77). A
promotion lasts \(T=28\) days and the retailer supplies a fixed expected markdown budget
\(B\). Its planning state is

\[
s_t=(\tau_t,B_t),
\tag{80a}
\]

where \(\tau_t\) is the remaining duration. An action is no promotion or one
segment-specific five-SKU bundle at a 10% or 20% discount. Candidate bundles are formed
from training data only. For a factual SMC particle \(S^{(m)}\), an action is evaluated by
the exact energy ratio

\[
w_m(a)\propto
\exp\!\left(E_{p'}(S^{(m)},x)-E_p(S^{(m)},x)\right).
\tag{80b}
\]

Thus all actions reuse common particles, producing paired counterfactual estimates. An
action is admissible only when its minimum normalized reweighting ESS is at least 0.2 and
its audited contexts avoid majority probability on \(N\ge60\).

If \(C_a\) is expected markdown spend and \(L_a\) is the 95% lower confidence bound on
incremental list-price basket value, the finite-horizon recursion is

\[
V_\tau(b)=
\max_{a:\widetilde C_a\le b}
\left\{L_a+V_{\tau-1}(b-\widetilde C_a)\right\}.
\tag{80c}
\]

Upward cost rounding and a tightened terminal grid guarantee at least 95% planned budget
utilization without overspending the supplied budget. The result is a conservative
budget-allocation policy among modeled arriving trips. Because the basket law is
conditional on a nonempty trip and represents incidence rather than units, the reward is
not causal profit: it omits visit response, unit quantities, wholesale cost, inventory and
competitor reactions. The mathematical contract, measured action responses and executable
entry point are in [SEGMENT_PROMOTION_MDP.md](SEGMENT_PROMOTION_MDP.md).

---

## 21. Fair baseline comparisons

Every method must use:

- identical trip IDs;
- identical revealed and held-out products;
- identical candidate assortment;
- identical prices, promotions, week and store;
- identical support and non-empty conditioning;
- the same likelihood unit, in nats per basket; and
- the same MRR and Recall definitions.

The most important ablation is the matched additive parent: the same version-4 utilities,
size law, category term and support, with \(\Phi=0\). This isolates the interaction value.

A conventional multinomial, Bernoulli model, SHOPPER, DPP or other baseline must be fitted
using its own correct normalizer and likelihood. Equal update count alone is not fairness
when update costs differ. Report both:

\[
\text{equal-update/equal-compute comparisons}
\tag{81}
\]

and

\[
\text{converged-model comparisons}.
\tag{82}
\]

For the latter, every fitted external baseline uses a fresh lineage and the same fixed
validation manifest. Let \(v_k\) be validation log likelihood at evaluation \(k\). A gain
is material when

\[
v_k > \max_{j<k}v_j + \delta,
\qquad \delta=0.002\ \text{nats/basket}.
\tag{82a}
\]

Validation plateaus reduce the learning rate geometrically. Convergence is certified only
after all of the following hold simultaneously:

1. training exposure is at least two epoch-equivalents over the supported training trips;
2. the learning rate has reached \(0.02\) times its initial value;
3. no material validation improvement has occurred for eight evaluations; and
4. at least four stale evaluations have occurred after reaching the learning-rate floor.

The maximum update count is a fail-closed safety ceiling, not a stopping definition. If a
baseline reaches it without this certificate, no test score or model-superiority claim is
produced. The validation-selected checkpoint—not necessarily the terminal checkpoint—is
then scored once on the common locked test manifest. SHOPPER uses a fixed ordering stream
for validation and a higher-ordering audit for its final set likelihood, separating
optimizer convergence from ordering Monte Carlo error.

---

## 22. Convergence contract

The next approved run should not declare convergence from a flat noisy trace alone.

### 22.1 Model contract

1. Energy (8) remains unchanged.
2. All offered products remain in each likelihood normalizer.
3. Support remains \(1\le n\le120\).
4. Rank seven remains fixed unless a new split-stability audit approves another rank.
5. No empirical size factorization or recommendation loss is introduced.

### 22.2 Estimator contract

1. ESP/category computation is exact at every node.
2. Direct \(q=8\) is permitted only in the coarse phase.
3. The final phase uses the unbiased level-9 score (51), or direct level 9.
4. Level 10 audits both value and gradient.
5. No estimator-difficult trip is skipped.
6. A non-positive signed partition is rejected, never clamped.
7. Fidelity failure stops gracefully and preserves the last certified best.

### 22.3 Optimizer contract

1. Phase B fits every declared non-Gram incidence block from scratch, and Phase D fits the
   identifiable PSD interaction kernel together with its confounded size directions.
2. Mature Adam moments are retained.
3. Gauge transformations preserve item utilities exactly.
4. Spectral projection uses singular-value clipping.
5. A numerical envelope is reported as a restriction, not as foundational theory.
6. A learning-rate rollback resumes from the best checkpoint, not from a regressed tail.
7. Training latency and validation latency are logged separately.

### 22.4 Formal convergence conditions

Require all of:

\[
\text{no material fixed-panel improvement over the patience window},
\tag{83}
\]

\[
\eta_t=\eta_{\min},
\tag{84}
\]

\[
\left\|
\Pi_{\mathcal T}
\nabla\ell_{q9}(\Theta_t)
\right\|
\le\tau_{\mathrm{opt}},
\tag{85}
\]

\[
\left\|
\nabla\ell_{q10}(\Theta_t)
-\nabla\ell_{q9}(\Theta_t)
\right\|
\le\tau_{\mathrm{quad}}
<\tau_{\mathrm{opt}},
\tag{86}
\]

where \(\Pi_{\mathcal T}\) projects onto the feasible tangent cone when a numerical boundary
is active.

---

## 23. Final evaluation contract

After convergence:

1. select on a large fixed validation panel;
2. evaluate once on the untouched test panel;
3. score the matched additive parent and all baselines on identical trips;
4. report paired confidence intervals for likelihood differences;
5. report MRR, MRR@5, MRR@10, MRR@20 and Recall at the same cutoffs;
6. audit size mean, variance, quantiles and large-basket tail;
7. audit category-count and pair moments;
8. run symmetric price counterfactuals;
9. validate generation separately from likelihood; and
10. report training, evaluation and generation latency.

---

## 24. What is currently established

Executable enumeration tests establish that:

- the H--S identity reproduces the original Gram interaction;
- ESP normalizers reproduce direct subset enumeration;
- gradients with respect to \(\lambda,\Phi,\rho_c,\rho_0\) match enumeration;
- size probabilities and incidence derivatives match enumeration;
- the common-utility response equals \(\operatorname{Var}(n)\);
- revealed-set conditioning is the conditional of the same joint law; and
- conditional generation matches enumerable laws within Monte Carlo error.

The corrected end-to-end execution additionally establishes the following empirical facts.
The exact additive parent converged from \(-49.622960\) to \(-44.748944\) nats/basket on
its fixed validation panel. A split-half score audit selected rank five, with mean squared
subspace overlap \(0.549018\). The constrained fixed-draw natural-parameter solve then
found a cross-fitted gain of approximately \(0.0243\) nats/basket with median effective
sample fraction \(0.9981\).

On locked 4,096-trip panels, the interaction child improves over its matched exact
additive parent by

\[
\widehat\Delta_{\mathrm{val}}
=0.021630\pm0.001581
\quad\text{and}\quad
\widehat\Delta_{\mathrm{test}}
=0.023908\pm0.001647
\quad\text{nats/basket}.
\tag{87}
\]

The higher-rule numerical error bounds are \(0.000510\) and \(0.000720\) nats,
respectively, so both paired gains remain positive after numerical allowance. This is the
first corrected-data result that statistically establishes a Gram-interaction likelihood
gain over the exact additive parent.

Recommendation is also evaluated from the same law. Total MRR is

\[
0.095144\pm0.006083,
\tag{88}
\]

but the interaction increment over additive MRR is only

\[
0.000247\pm0.000372,
\qquad
95\%\ \mathrm{CI}=[-0.000481,0.000976].
\tag{89}
\]

Therefore an interaction recommendation gain is not established. This does not contradict
Eq. (87): log likelihood is a proper score for the whole basket distribution, whereas MRR
is a discontinuous one-hidden-item ranking functional.

Generation has no invalid-assortment or duplicate-item baskets and price counterfactuals
move in the theoretically required direction. Production certification nevertheless
fails. The aggregate \(N\ge60\) rate passes its calibrated bound, but the high-accuracy
audit finds 12 high-risk contexts with \(P(N\ge60)\ge0.5\), including a maximum of
\(0.73837\) when observed size is below 40. Hence the current fitted parameters are a
research candidate, not a safe retailer simulator.

All exact numbers, panels and artifact paths are in
[CORRECTED_PIPELINE_RESULTS.md](CORRECTED_PIPELINE_RESULTS.md).

---

## 25. Current decision

The model law remains fixed by Eqs. (8)--(10), with all 5,455 products and sizes
\(1,\ldots,120\) in support. The corrected pipeline is now empirically validated as a
reproducible fitting and certification procedure. Its fail-closed terminal decision must
also be respected:

1. the likelihood interaction claim over the matched additive parent is accepted;
2. the interaction MRR claim is not accepted;
3. historical baseline results are not comparable until those baselines converge on the
   corrected cohort and identical manifests;
4. the current candidate is not authorized for production generation or policy
   simulation; and
5. the next fit must address the localized extreme-size phase within the existing
   \(\rho_0,\rho_c,\Phi\) theory and then rerun every locked gate.

No numerical averaging over contexts may replace the local tail gate, and no post-hoc
checkpoint choice may replace a fresh end-to-end execution.
