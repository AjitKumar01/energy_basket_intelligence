# The Version-4 energy-basket pipeline

## A textbook explanation of the model, estimator, training stages, and outputs

## 1. The complete story in one page

The data contain completed shopping baskets. For each observed checkout context $x$,
the model assigns a probability to every nonempty basket $S$ drawn from the offered
catalogue $\mathcal A_x$:

\[
p_\Theta(S\mid x)
=
\frac{\exp\{E_\Theta(S,x)\}}
{Z_{+,\Theta}(x)},
\qquad
Z_{+,\Theta}(x)
=
\sum_{\substack{A\subseteq\mathcal A_x\\1\le |A|\le n_{\max}}}
\exp\{E_\Theta(A,x)\}.
\tag{1}
\]

The energy $E_\Theta(S,x)$ combines product popularity, household taste, price,
promotions, season, store, total basket size, affinity-group counts, and low-rank
product interactions. The normalizer $Z_+(x)$ turns these relative scores into a
proper probability distribution over all supported nonempty baskets.

This one probability law supports all of the following:

- likelihood evaluation asks how much probability the model assigned to an observed
  basket;
- recommendation conditions the law on the known part of a basket;
- generation draws a complete basket from the law;
- price counterfactuals change prices inside $x$ and recompute the law; and
- retailer-policy simulation compares decisions through the counterfactual laws.

There is no separately trained recommender. Training is staged, but the model is not a
collection of disconnected models. The stages are a numerically and statistically
controlled way to estimate the parameters of Eq. (1):

\[
\boxed{
\text{audited data}
\longrightarrow
\text{exact additive parent}
\longrightarrow
\text{stable interaction directions}
\longrightarrow
\text{interaction fit}
\longrightarrow
\text{residual size calibration}
\longrightarrow
\text{locked evaluation}.}
\tag{2}
\]

Two similarly named operations must not be confused:

1. The **affinity partition** groups products using training-only co-purchases. It does
   not fit any household parameter.
2. The **rank-one household-size coordinate** is learned in the exact additive fit. A
   later stage tests only an incremental correction to that coordinate after
   interactions have changed the size distribution.

The foundational Version-4 energy and the Hubbard--Stratonovich/elementary-symmetric-
polynomial result are unchanged by this staged optimization.

---

## 2. What the raw data become

### 2.1 Unit of observation

The modeled outcome is a set of distinct products at a recorded checkout. It is not a
customer-day arrival model: the absence of a transaction is not represented as an empty
basket. It is also not, in the real-data fit, a complete demand-and-profit model:
quantities, wholesale costs, inventory, and causal treatment effects require additional
information.

Let:

- $j\in\{1,\ldots,J\}$ identify a product;
- $h\in\{1,\ldots,H\}$ identify a household;
- $t$ identify a checkout occasion;
- $x_t$ contain household, day, week, store, price, and promotion information;
- $S_t\subseteq\mathcal A_{x_t}$ be the distinct-product basket;
- $N_t=|S_t|$ be its size; and
- $n_c(S_t)$ be its number of products in affinity group $c$.

### 2.2 The verified data in this checkout

The raw bundle supplied with this repository was checked against the locked release
digests and rebuilt end to end. The audit found:

| Quantity | Audited value |
|---|---:|
| Raw transaction rows | 2,595,732 |
| Raw product rows | 92,353 |
| Raw causal/promotion rows | 36,786,524 |
| Modeled products | 5,455 |
| Eligible households | 1,920 |
| Modeled basket-product rows | 1,535,006 |
| Modeled baskets | 200,698 |
| Training baskets | 160,007 |
| Validation baskets | 17,351 |
| Test baskets | 23,340 |
| Observed mean basket size | 7.648 |
| Observed basket-size variance | 80.691 |
| Maximum supported basket size | 120 |

The transaction cleaning removes 42,326 of 2,595,732 raw lines: 18,917 have nonpositive
quantity or sales value, and 23,409 have bulk quantity, extreme reconstructed unit price,
or a sign-anomalous discount. Only two lines have materially sign-anomalous discounts.
This accounting is performed before cohort selection; large checkout baskets are not
deleted merely because their normalization is difficult.

Products and households are selected from training weeks only. Training uses weeks
9--82, validation uses weeks 83--90, and test uses weeks 91--101. Every modeled product
and household occurs in training, and the independent preprocessing audit found zero
held-out purchase lines outside the declared 5,455-product support.

The three source SHA-256 digests, preprocessing choices, affinity partition, and model
index are combined into one data fingerprint. A checkpoint from different data is
rejected even if its array dimensions happen to match.

---

## 3. The Version-4 basket law

### 3.1 Energy of a basket

The model energy is

\[
\begin{aligned}
E_\Theta(S,x)
={}&
\sum_{j\in S} b_j(x)
+\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k \\
&-\sum_c \rho_c {n_c(S)\choose 2}
-\rho_0(|S|).
\end{aligned}
\tag{3}
\]

Each block has a distinct interpretation:

| Block | Role |
|---|---|
| $b_j(x)$ | Contextual value of product $j$ before basket interactions |
| $\phi_j^\top\phi_k$ | Low-rank product-specific complementarity |
| $-\rho_c{n_c(S)\choose2}$ | Shared within-affinity-group pair effect |
| $-\rho_0(|S|)$ | Flexible global potential for total basket size |

The Gram interaction matrix is $K=\Phi\Phi^\top\succeq0$. Consequently the Gram
part represents attractive low-rank directions. The explicit category-count term can
represent broader attraction or repulsion within a group, depending on the sign of
$\rho_c$.

### 3.2 Contextual product utility

The fitted utility can be written schematically as

\[
\begin{aligned}
b_j(x)
={}&\lambda_j
+\theta_h^\top\alpha_j
-a_{hj}\,d_{jt}^{\mathrm{price}}
+w_j^{\mathrm{dsp}}D_{jt}
+w_j^{\mathrm{mlr}}M_{jt} \\
&+\mu_j^\top\delta_{w(t)}
+\zeta_j^\top\xi_{s(t)}.
\end{aligned}
\tag{4}
\]

Here:

- $\lambda_j$ is product popularity after the other effects are controlled;
- $\theta_h^\top\alpha_j$ is household-specific taste;
- $a_{hj}$ is a nonnegative price-response coefficient;
- $D_{jt}$ and $M_{jt}$ indicate display and mailer exposure;
- $\mu_j^\top\delta_w$ captures low-rank seasonal variation; and
- $\zeta_j^\top\xi_s$ captures low-rank store variation.

The selected real-data fit excludes recency because the previously constructed recency
feature was not stable under the temporal split. This is a declared fitted-capability
choice, not a change to Eq. (1).

### 3.3 Why $\rho_0$ and a common household coordinate can coexist

The vector $\rho_0(1),\ldots,\rho_0(n_{\max})$ describes the population-level shape
of the size distribution. A household-common utility $\kappa_h$ contributes

\[
\sum_{j\in S}\kappa_h=|S|\kappa_h.
\tag{5}
\]

It therefore tilts the global size law for household $h$:

\[
\rho_{0h}(n)=\rho_0(n)-n\kappa_h.
\tag{6}
\]

The flexible $\rho_0$ captures overall nonlinearity in size; $\kappa_h$ captures a
parsimonious household-specific tendency toward larger or smaller baskets. It does not
change the relative probability of two baskets of the same size.

---

## 4. The role of $\beta$

### 4.1 $\beta_j$ is a price factor, not an interaction factor

The household-product price coefficient is

\[
a_{hj}
=
\operatorname{softplus}(\gamma_h)^\top
\operatorname{softplus}(\beta_j)
\ge 0.
\tag{7}
\]

The three product embeddings have different jobs:

\[
\underbrace{\alpha_j}_{\text{household taste}},
\qquad
\underbrace{\beta_j}_{\text{price response}},
\qquad
\underbrace{\phi_j}_{\text{basket interaction}}.
\tag{8}
\]

$\gamma_h$ describes how household $h$ responds along the price dimensions, while
$\beta_j$ describes how product $j$ loads on those dimensions. The economically
meaningful object is their nonnegative dot product $a_{hj}$, not an isolated coordinate
of either embedding. The nonnegative coordinates can, for example, be jointly permuted or
oppositely rescaled between the two factors without changing the fitted dot products.

### 4.2 Why nonnegativity matters

For a simple log-price deviation $d_{jt}=\Delta\log p_{jt}$, the price contribution is

\[
b_j(x)=\cdots-a_{hj}d_{jt}.
\tag{9}
\]

Hence

\[
\frac{\partial b_j(x)}{\partial\log p_{jt}}=-a_{hj}\le0.
\tag{10}
\]

Softplus makes the sign restriction structural: a higher own price cannot directly make
the product more attractive. For a price change that affects only this utility term,
the derivative of the normalized log basket probability has the familiar observed-minus-
expected form

\[
\frac{\partial\log p(S\mid x)}{\partial\log p_{jt}}
=
-a_{hj}\left[
\mathbf 1\{j\in S\}-P(j\in S\mid x)
\right].
\tag{11}
\]

This equation also explains why price learning uses the normalizer: changing one utility
changes both the observed basket energy and the probabilities of every alternative basket.

### 4.3 Common and relative price movement

The implementation decomposes each log-price deviation as

\[
d_{jt}=\bar d_t+e_{jt},
\tag{12}
\]

where $\bar d_t$ is the average movement over the offered catalogue and $e_{jt}$ is
the product's relative movement. The fitted contribution is

\[
-a_{hj}\left[
\bar d_t+\operatorname{softplus}(\kappa_p)e_{jt}
\right].
\tag{13}
\]

A common movement shifts many product utilities and can strongly affect total size. A
relative movement mainly reallocates share among products. The scalar $\kappa_p$ lets
these responses have different scales without changing the foundational energy model.

$\beta_j$, $\gamma_h$, and $\kappa_p$ are learned in the exact additive stage.
They are frozen during the later interaction solve, but they continue to affect the
additive proposal law, the final normalizer, generated baskets, recommendations under a
new price context, counterfactuals, and retailer decisions. A poor additive price fit can
also leak price-driven co-purchase structure into the interaction residual; this is one
reason the additive parent must converge first.

### 4.4 A separate symbol sometimes also called beta

Annealed samplers often call their inverse temperature $\beta\in[0,1]$. That is not
the learned product parameter $\beta_j$. This document calls the annealing temperature
$\tau$ throughout:

\[
\pi_\tau(S,z\mid x)
\propto
\pi_0(S,z\mid x)
\exp\{\tau\,\Delta E_{\mathrm{int}}(S,z,x)\}.
\tag{14}
\]

---

## 5. How the normalizer becomes tractable

Direct computation of $Z_+(x)$ would enumerate an exponential number of subsets. The
Version-4 proposition separates the low-rank interaction from the discrete basket sum.

### 5.1 Hubbard--Stratonovich transformation

Define

\[
v(S)=\sum_{j\in S}\phi_j.
\tag{15}
\]

Then

\[
\sum_{j<k\in S}\phi_j^\top\phi_k
=
\frac12\|v(S)\|^2
-\frac12\sum_{j\in S}\|\phi_j\|^2.
\tag{16}
\]

For $z\sim\mathcal N(0,I_r)$, the Hubbard--Stratonovich identity is

\[
\exp\left\{\frac12\|v\|^2\right\}
=
\mathbb E_z\left[\exp\{z^\top v\}\right].
\tag{17}
\]

Define the conditional product weight

\[
w_j(z,x)
=
\exp\left\{
b_j(x)-\frac12\|\phi_j\|^2+z^\top\phi_j
\right\}.
\tag{18}
\]

Conditional on $z$, the product-specific Gram coupling has disappeared. Product choices
can now be collected by polynomial degree.

### 5.2 Elementary-symmetric and category polynomials

For affinity group $c$, define

\[
G_c(u;z,x)
=
\sum_{k=0}^{\min(|\mathcal A_{xc}|,n_{\max})}
\exp\left\{-\rho_c{k\choose2}\right\}
e_k\!\left(\{w_j(z,x):j\in\mathcal A_{xc}\}\right)u^k,
\tag{19}
\]

where $e_k$ is the elementary symmetric polynomial of degree $k$. Multiplying the
group polynomials and truncating at $n_{\max}$ gives

\[
\prod_cG_c(u;z,x)
=
\sum_{n=0}^{n_{\max}}A_n(z,x)u^n.
\tag{20}
\]

Therefore

\[
\boxed{
Z_+(x)
=
\mathbb E_{z\sim\mathcal N(0,I_r)}
\left[
\sum_{n=1}^{n_{\max}}e^{-\rho_0(n)}A_n(z,x)
\right].}
\tag{21}
\]

### Proposition 1: exact discrete summation for fixed $z$

For fixed $z$, every coefficient $A_n(z,x)$, and hence the integrand in Eq. (21),
can be computed without enumerating baskets.

**Proof.** For weights $w_1,\ldots,w_m$, elementary symmetric polynomials satisfy

\[
e_k^{(q)}
=
e_k^{(q-1)}+w_qe_{k-1}^{(q-1)},
\qquad e_0^{(q)}=1.
\tag{22}
\]

This recursion visits products and degrees rather than subsets. Multiplication by the
exact group-count factor produces Eq. (19). Truncated polynomial convolution combines
the groups to produce Eq. (20), and multiplication by the total-size factor produces
Eq. (21). Thus the discrete sum is exact for fixed $z$. $\square$

When $\Phi=0$, the weights no longer depend on $z$, so the Gaussian expectation
vanishes. This yields an exact and comparatively fast additive likelihood over the full
5,455-product support.

---

## 6. Why $\Phi$ is not trained jointly from the first update

The tractability proposition permits joint likelihood evaluation in principle. It does
not say that raw factor-coordinate optimization is identified, cheap, or able to leave
the additive point. The current staging follows from three facts.

### Proposition 2: the factor gradient is zero at the additive point

The likelihood depends on $\Phi$ through $K=\Phi\Phi^\top$. If $L(K)$ denotes
the likelihood as a function of $K$, then

\[
\nabla_\Phi L
=
\left(\nabla_KL+\nabla_KL^\top\right)\Phi.
\tag{23}
\]

Consequently,

\[
\nabla_\Phi L\big|_{\Phi=0}=0,
\tag{24}
\]

even if a positive-semidefinite direction in $K$ would improve likelihood.

**Proof.** Substitute $\Phi=0$ into the right side of Eq. (23). The score with respect
to $K$ may be nonzero, but multiplication by $\Phi$ makes the factor-coordinate
gradient zero. $\square$

Thus an optimizer initialized at the exact additive model cannot discover interactions
through an ordinary first derivative. Random nonzero initialization avoids exact zero,
but introduces arbitrary orientation and pays interaction-normalizer cost before basic
popularity, household, price, and context effects are learned.

### 6.1 Rotational non-identification

For every orthogonal $Q$,

\[
(\Phi Q)(\Phi Q)^\top=\Phi\Phi^\top.
\tag{25}
\]

Many factor matrices therefore describe the same basket distribution. A raw joint
optimizer can spend movement in directions that change the coordinates but not the law.

### 6.2 Computational separation

With $\Phi=0$, one exact dynamic-program pass evaluates each context. With active
interactions, the same pass must be repeated across a numerical rule in $r$-dimensional
$z$-space. It is inefficient to pay that cost while thousands of additive parameters
are still learning first-order structure.

### 6.3 What the pipeline does instead

The pipeline first fits the additive model, then calculates an interaction score in the
natural Gram space. If $P(S)$ is the off-diagonal co-incidence matrix, define

\[
R
=
\mathbb E_{\mathrm{data}}[P(S)]
-\mathbb E_{p_{\mathrm{add}}}[P(S)].
\tag{26}
\]

Positive eigenvectors of $R$ are locally improving PSD directions. Split-half overlap
checks whether those directions repeat across two training subsets. With the selected
basis $U\in\mathbb R^{J\times r}$, the model then uses

\[
K=UCU^\top,
\qquad C\succeq0,
\tag{27}
\]

and optimizes the small natural matrix $C$, rather than every coordinate of a randomly
oriented $\Phi$.

This is an optimization design, not a new probability theory. Joint warm-started or
alternating refinement is possible in principle, but it is not part of the certified
pipeline. The current estimator is deliberately a staged residual fit: additive
parameters are not unrestrictedly reoptimized after interactions enter. That limitation
must remain explicit when interpreting the fitted interaction gain.

---

## 7. The executable pipeline, stage by stage

### 7.1 Data audit and deterministic preprocessing

The driver checks the three raw file digests, reconstructs the declared loyalty-price
basis, constructs checkout baskets, creates the temporal split, selects the cohort using
training only, builds price/promotion/store panels, and independently reconstructs the
outcomes as an audit. It then writes the content-addressed model-data fingerprint.

### 7.2 Training-only affinity partition

Supported positive-lift co-purchase edges from training baskets are used to form product
groups. These groups define $n_c(S)$ and the polynomial blocks in Eq. (19).

This stage learns none of $\theta$, $\alpha$, $\kappa_h$, $\rho_0$, $\rho_c$,
$\beta$, or $\phi$. The word “affinity” describes a fixed product partition, not an
affinity model fit and not household-size training.

### 7.3 Fresh initialization

The pipeline creates a new artifact from training-only moments. One coordinate of the
existing household-taste factorization is reserved as a catalogue-common loading:

\[
\alpha_{j,K}=1\quad\text{for every }j,
\qquad
\kappa_h=\theta_{h,K}.
\tag{28}
\]

The remaining product-taste loadings are centered across the catalogue. $\Phi$ is set
exactly to zero before additive training.

### 7.4 Exact additive maximum likelihood

With $\Phi=0$, the pipeline maximizes

\[
\ell_{\mathrm{add}}(\Theta)
=
\sum_t
\left[
E_{\mathrm{add}}(S_t,x_t)
-\log Z_{+,\mathrm{add}}(x_t)
\right]
\tag{29}
\]

using the exact category/cardinality dynamic program. This stage jointly learns:

- product intercepts;
- household and product taste factors, including $\kappa_h$;
- household and product price factors $\gamma_h$ and $\beta_j$;
- the relative-price scale $\kappa_p$;
- promotion, season, and store effects;
- the affinity-group coefficients $\rho_c$; and
- the flexible global size potential $\rho_0(1),\ldots,\rho_0(120)$.

Regularizers encode declared pooling, aggregate size calibration, and aggregate price-
elasticity calibration. Validation drives learning-rate reduction and the convergence
gate. The interaction matrix remains exactly zero throughout this stage.

### 7.5 Stable-rank audit

The spectral procedure estimates Eq. (26) on the full training sample and on two halves.
It chooses the largest candidate rank whose positive interaction subspace is sufficiently
stable. If rank 8 is unstable, lower ranks are tested rather than jumping blindly to a
fixed rank. Rank measures model capacity; it is unrelated to quadrature level $q$,
which measures numerical integration accuracy for a fixed fitted model.

### 7.6 Constrained interaction MCLE

For a fixed basis $U$, define the interaction statistic

\[
F_U(S)
=
\frac12\left[
\left(\sum_{j\in S}u_j\right)
\left(\sum_{j\in S}u_j\right)^\top
-\sum_{j\in S}u_ju_j^\top
\right].
\tag{30}
\]

Then $\operatorname{tr}\{CF_U(S)\}$ is exactly the pair interaction under
$K=UCU^\top$. The fitted correction also permits a linear and quadratic update inside
the original size potential:

\[
h_{C,a,c}(S)
=
\operatorname{tr}\{CF_U(S)\}
-a\frac{|S|}{10}
-c\left(\frac{|S|}{10}\right)^2.
\tag{31}
\]

For each selected training context $m$, the pipeline draws $D$ exact baskets
$S_{md}$ from the additive parent. It keeps these common draws fixed while optimizing

\[
\widehat G(C,a,c)
=
\frac1M\sum_{m=1}^M
\left[
h(S_m^{\mathrm{obs}})
-\log\left{
\frac1D\sum_{d=1}^D e^{h(S_{md})}
\right\}
\right].
\tag{32}
\]

The logarithm estimates the child-to-parent normalizer ratio. With the samples fixed,
Eq. (32) is a deterministic concave function of the natural parameters: a linear term
minus log-sum-exp. PSD and size-tail constraints define a convex feasible set. Projected
ascent and backtracking accept only improving steps.

The finite-$D$ log-mean-exp has Monte Carlo error and finite-sample bias. The pipeline
therefore cross-fits the gain, checks proposal effective sample size, and independently
certifies the accepted checkpoint with higher-accuracy Smolyak likelihood. Finally it
factors $C$ and stores

\[
\Phi=UC^{1/2}.
\tag{33}
\]

### 7.7 Why household rank one appears again

This later step is not a repeated fit of the same objective.

Let $\kappa_h^{(0)}$ be the household-common coordinate already learned in the exact
additive stage. After the interaction and global size corrections are fitted, define the
new parent size law

\[
q_x(n)=p_{\mathrm{post-int}}(N=n\mid x).
\tag{34}
\]

The later stage tests only a residual increment $\delta_h$:

\[
q_{x,\delta_h}(n)
=
\frac{q_x(n)e^{n\delta_h}}
{\sum_{m=1}^{n_{\max}}q_x(m)e^{m\delta_h}}.
\tag{35}
\]

For household $h$, the penalized objective is

\[
L_h(\delta_h)
=
\sum_{t:h(t)=h}
\left[
N_t\delta_h
-\log\sum_mq_{x_t}(m)e^{m\delta_h}
\right]
-\frac{\lambda}{2}\delta_h^2.
\tag{36}
\]

Its curvature is

\[
L_h''(\delta_h)
=
-\sum_{t:h(t)=h}
\operatorname{Var}_{\delta_h}(N\mid x_t)-\lambda<0.
\tag{37}
\]

Therefore every household problem has a unique solution. Ridge is selected by swapped
chronological household-day folds and tested with household-cluster uncertainty.

Why can $\delta_h$ be nonzero if $\kappa_h^{(0)}$ was already optimized? The score
condition is

\[
\frac{\partial\ell}{\partial\kappa_h}
=
\sum_{t:h(t)=h}
\left[N_t-\mathbb E_\Theta(N\mid x_t)\right].
\tag{38}
\]

The expectation in Eq. (38) changes when $\Phi$ and the global size correction are
introduced. The old additive optimum need not remain the post-interaction optimum. The
late one-dimensional block cheaply restores this score direction without reopening all
additive parameters:

\[
\kappa_h^{\mathrm{final}}
=
\kappa_h^{(0)}+\delta_h.
\tag{39}
\]

The increment is not forced. If its simultaneous cross-fitted lower confidence bound is
not positive, $\delta_h=0$ and the parent is preserved. A minimal downward safety
projection can still be applied if a localized context violates the declared extreme-
basket tail limit. Independent likelihood and population checks follow either outcome.

### 7.8 Locked evaluation

The final candidate is evaluated on fixed manifests:

- validation and test likelihood on complete support;
- recommendation MRR and recall;
- generation and price-counterfactual diagnostics;
- customer segments and interaction-embedding diagnostics; and
- full-population and high-accuracy extreme-size screens.

Validation decides model acceptance. Test reports performance and is not used to tune the
candidate. External Bernoulli, DPP, and NDPP baselines have their own convergence checks
and are compared on the identical locked test manifest; they are not trained inside the
main model pipeline.

### 7.9 Production certification and policy calculation

The complete supported training population is screened at a cheaper Smolyak rule, and the
highest-risk contexts are recomputed at a more accurate rule. Certification fails closed
if localized large-basket mass, numerical discrepancy, lineage, convergence, or required
likelihood evidence fails. The segment promotion calculation then uses the certified
checkpoint as a structural environment for campaign shortlisting.

---

## 8. Numerical integration after training

For active rank $r$, Smolyak quadrature approximates only the Gaussian expectation in
Eq. (21). It does not truncate products or basket sizes. The pipeline uses:

- $q=r+1$ for the large population screen;
- $q=r+2$ for reported likelihood and high-risk confirmation; and
- $q=r+3$ on a smaller panel to bound numerical error.

Increasing $q$ adds integration nodes; it does not add parameters, increase rank, or
make the checkpoint “more trained.” The comparison between adjacent levels measures
quadrature error only. It does not measure minibatch noise, variation across baskets,
optimization error, or model misspecification.

---

## 9. How the fitted law produces retail functions

### 9.1 Recommendation follows by conditioning

Let $R$ be the observed remainder of a basket when exactly one product is hidden. For
a candidate $j\notin R$, define $s_j(R,x)=E(R\cup\{j\},x)$. Then

\[
\begin{aligned}
P(j\text{ completes }R\mid x,R,\text{one missing})
&=
\frac{p(R\cup\{j\}\mid x)}
{\sum_{k\notin R}p(R\cup\{k\}\mid x)} \\
&=
\frac{e^{s_j(R,x)}/Z_+(x)}
{\sum_{k\notin R}e^{s_k(R,x)}/Z_+(x)} \\
&=
\frac{e^{s_j(R,x)}}
{\sum_{k\notin R}e^{s_k(R,x)}}.
\end{aligned}
\tag{40}
\]

The normalizer cancels. All completed candidates have the same size, so common total-size
terms cancel as well. Recommendation therefore needs neither Smolyak quadrature nor a
separate recommendation objective. A better average joint likelihood need not guarantee
a better finite-sample MRR, however: likelihood scores the probability of the complete
basket, while MRR tests the rank of one hidden item among many alternatives.

### 9.2 Basket generation follows directly from the joint law

The Hubbard--Stratonovich representation defines the augmented law

\[
p(S,z\mid x)
\propto
\varphi_r(z)
\exp\left\{
-\rho_0(|S|)-\sum_c\rho_c{n_c(S)\choose2}
\right\}
\prod_{j\in S}w_j(z,x).
\tag{41}
\]

Marginalizing $z$ gives exactly Eq. (1). Conditional on $z$, a basket is drawn by:

1. computing the size and affinity-group polynomial coefficients;
2. drawing the total size $N$;
3. drawing the number selected from each affinity group; and
4. drawing the actual distinct products by reverse ESP recursion.

The nontrivial step is drawing $z$. Its target marginal is the Gaussian density times
the complete polynomial integrand, not a standard Gaussian. The implemented SMC bridge
starts from exact additive basket draws, gradually turns on the interaction contribution,
reweights particles, and applies invariant rejuvenation moves. The bridge is a sampling
algorithm for Eq. (1); it is not an added model term and does not correct the basket after
generation.

### 9.3 Price counterfactuals

For a proposed price change, update the price features in $x$, recompute Eq. (13), and
query the resulting basket law. With factual samples from $p_0$, a counterfactual
expectation can also be estimated by

\[
\mathbb E_1[g(S)]
=
\frac{
\mathbb E_0\left[g(S)e^{E_1(S)-E_0(S)}\right]
}{
\mathbb E_0\left[e^{E_1(S)-E_0(S)}\right]
}.
\tag{42}
\]

Effective sample size must be checked because large interventions can make factual
samples unrepresentative of the counterfactual law. These are structural predictions
under the fitted observational model, not automatically causal treatment effects.

### 9.4 Customer segments and bundles

Households are clustered using fitted, economically meaningful summaries such as taste,
price response, and common size tendency. Segment labels summarize heterogeneity; they do
not alter the underlying household model. Candidate bundles come from products and
interaction pairs that are relevant to a segment and supported by held-out co-incidence
audits. A large $\phi_j^\top\phi_k$ is evidence of fitted complementarity, not by itself
causal proof that discounting one item creates demand for the other.

### 9.5 Budget-constrained retailer simulation

The current policy layer gives a retailer a finite markdown budget over a fixed campaign
horizon. A state records time and remaining expected budget. An action chooses no offer or
a segment-specific bundle and discount. The fitted counterfactual basket law supplies
expected incidence and markdown use, and dynamic programming selects a feasible action
sequence.

Because the real-data checkpoint does not identify visits, quantities, costs, inventory,
or causal treatment response, this layer is suitable for campaign shortlisting and A/B-
test design, not a guaranteed profit-optimal deployment. The separate synthetic-retailer
pipeline adds those missing variables under known truth so that policy recovery can be
measured honestly.

---

## 10. Computational cost and why the stages save work

Let $B$ be contexts in a minibatch, $J_x=|\mathcal A_x|$, $C_x$ be active affinity
groups, $r$ be interaction rank, and $M_q(r)$ be the number of Smolyak nodes.

| Operation | Leading work |
|---|---|
| Exact additive normalizer | $O\!\left(B[J_xn_{\max}+C_xn_{\max}^2]\right)$ |
| Sparse interaction score | Sparse observed/generated pair counts plus leading eigensolve |
| Natural interaction statistics | $O(MDr^2)$ for $M$ contexts and $D$ draws |
| One-dimensional size recalibration | $O(Tn_{\max}\log(1/\varepsilon))$ |
| Smolyak likelihood | $O\!\left(TM_q(r)[J_xn_{\max}+C_xn_{\max}^2]\right)$ |
| Add-one recommendation | $O(J_xr)$, without $Z_+(x)$ |

The most important multiplier is $M_q(r)$. Staging performs the high-dimensional
additive learning without that multiplier, identifies a small interaction subspace once,
and reserves high-accuracy integration for fixed validation, test, and certification
panels.

---

## 11. What a successful pipeline establishes

A successful full execution establishes that:

- the raw data, derived data, and checkpoint lineage are consistent;
- the exact additive optimizer met its declared convergence rule;
- the selected interaction subspace was repeatable across training halves;
- the constrained interaction block passed its cross-fit and proposal-quality gates;
- the validation likelihood gain survived an independent numerical-integration audit;
- downstream functions loaded the same final checkpoint; and
- localized as well as aggregate basket-size safety checks passed.

It does not establish that:

- every learned complement pair is causal;
- observational price response equals randomized treatment response;
- a conditional checkout model predicts whether a household visits;
- product incidence alone predicts purchased units or profit; or
- a promotion policy can be deployed without controlled business validation.

---

## 12. Compact answers to the three central questions

### Is rank-one household size trained twice?

No. The additive stage learns the base value $\kappa_h^{(0)}$ jointly with additive
utility, price, context, and size parameters. Interactions later change
$P(N\mid x)$. The final rank-one stage tests a residual $\delta_h$ against that changed
law. The final value is $\kappa_h^{(0)}+\delta_h$; unsupported residual corrections are
exactly zero. The earlier affinity-partition stage trains neither value.

### Why is the interaction embedding not trained jointly from initialization?

At $\Phi=0$, its ordinary factor gradient is exactly zero; at arbitrary nonzero
initialization, its orientation is unidentified; and every interaction update multiplies
normalizer cost by a multidimensional integration rule. The pipeline instead fits the
exact additive parent, estimates stable locally improving directions in Gram space, and
fits a small PSD natural matrix in that basis. This is an optimization strategy consistent
with the same Version-4 law, not a replacement of its theorem.

### What is the role of $\beta$?

The learned $\beta_j$ is the product side of household-product price sensitivity.
Together with $\gamma_h$, it creates $a_{hj}\ge0$, which controls how product utility
responds to log price. It affects the original fit and every price-dependent downstream
query, but it does not represent taste or complementarity. An annealing temperature that
some sampling papers also call beta is a separate algorithmic quantity, denoted $\tau$
here.
