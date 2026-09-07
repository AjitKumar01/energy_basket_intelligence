# The Version-4 energy-basket model and pipeline

## A connected textbook account of the probability law, computation, training, and retail use

## 1. The problem and the complete route to a solution

A retailer observes completed shopping trips. On trip $t$, household $h(t)$ chooses a
nonempty set of distinct products $S_t$ from the products available on that occasion.
The central statistical problem is:

> Given the household, assortment, prices, promotions, store, and time, assign a
> probability to every possible nonempty basket.

Solving this one problem provides a common foundation for five tasks:

1. **Likelihood evaluation:** measure how well the fitted law explains held-out baskets.
2. **Recommendation:** condition the law on the observed part of a basket and rank a
   missing product.
3. **Basket generation:** draw complete synthetic baskets for a specified context.
4. **Price counterfactuals:** alter prices in the context and recompute the basket law.
5. **Retail policy analysis:** compare promotion decisions through those counterfactual
   laws, subject to a budget and campaign horizon.

The model must represent product interactions without summing explicitly over all
$2^{5,455}$ subsets of the catalogue. Version-4 achieves this by combining a low-rank
interaction representation with the Hubbard--Stratonovich identity and polynomial
dynamic programming. The remaining low-dimensional Gaussian integral is evaluated
numerically.

The complete dependency order is

\[
\boxed{
\text{raw transactions}
\to \text{audited baskets}
\to \text{one joint basket law}
\to \text{tractable normalizer}
\to \text{staged parameter fitting}
\to \text{numerical certification}
\to \text{retail queries}.}
\tag{1}
\]

The chapters follow exactly this order. Every numerical device is introduced only after
the probability quantity that it computes has been defined. The foundational Version-4
law is never replaced by a separate recommender or generator.

---

## 2. Data, observation unit, and notation

### 2.1 The modeled observation

One observation is one recorded checkout basket. Duplicate units of the same product are
collapsed, so the outcome is a **set of distinct products**, not a vector of purchased
quantities. The model is conditional on a checkout having occurred. It therefore assigns
probability to nonempty baskets but does not estimate the probability that a household
visits the store or buys nothing.

The distinction matters. If a retailer wants unconditional demand, the conditional
basket law must later be combined with a separate visit/no-purchase model. Likewise,
unit demand, inventory, wholesale cost, and causal treatment response require data not
contained in a set-valued checkout outcome.

### 2.2 Indices, sets, and dimensions

The following notation is used throughout.

| Symbol | Type or range | Meaning |
|---|---|---|
| $t$ | $1,\ldots,T$ | Checkout occasion or trip |
| $j,k$ | $1,\ldots,J$ | Products; here $J=5,455$ |
| $h$ | $1,\ldots,H$ | Households; here $H=1,920$ |
| $c$ | $1,\ldots,C_{\mathrm g}$ | Training-derived affinity groups; $C_{\mathrm g}$ is their number |
| $x_t$ | context | All information known before basket $t$ is chosen |
| $\mathcal A_{x_t}$ | subset of $\{1,\ldots,J\}$ | Products offered in context $x_t$ |
| $S_t$ | subset of $\mathcal A_{x_t}$ | Observed distinct-product basket |
| $N_t=|S_t|$ | integer | Number of distinct products in basket $t$ |
| $n_c(S)$ | integer | Number of products from affinity group $c$ in basket $S$ |
| $n_{\max}$ | integer | Largest basket size supported by the law; here 120 |
| $K$ | integer | Dimension of household--product taste factors |
| $K_p$ | integer | Dimension of household--product price factors |
| $r$ | integer | Rank of the Gram interaction representation |
| $q$ | integer | Smolyak quadrature level; it is not an interaction rank |

The context $x_t$ includes the household identity $h(t)$, store $s(t)$, week $w(t)$,
the offered assortment, product prices, and promotion indicators. A subscript such as
$b_j(x_t)$ means that product $j$ has a context-dependent utility on trip $t$.

Three distinct meanings of “rank” must not be mixed:

- $K$ controls the dimension of additive household taste;
- $K_p$ controls the dimension of heterogeneous price response; and
- $r$ controls the dimension of product interactions.

The quadrature level $q$ controls only numerical integration accuracy for an already
specified rank-$r$ model.

### 2.3 Audited data used by the pipeline

The supplied data were reconstructed from raw files and checked against locked release
digests. The resulting cohort is:

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

Training uses weeks 9--82, validation uses weeks 83--90, and test uses weeks 91--101.
Products and households are selected using training weeks only. This prevents future
transactions from influencing cohort construction. The independent preprocessing audit
found no held-out purchase line outside the declared product support.

The cleaning step removes nonpositive quantities or sales values, implausible bulk
quantities, extreme reconstructed unit prices, and sign-anomalous discounts. It does not
delete large but valid checkout baskets merely because they are numerically difficult.
Raw-file digests, preprocessing choices, product groups, and model indices are combined
into a data fingerprint. A checkpoint with a different fingerprint is rejected even
when its array dimensions happen to match.

---

## 3. One probability law for complete baskets

### 3.1 From an energy score to a probability

For a nonempty basket $S\subseteq\mathcal A_x$ with $|S|\le n_{\max}$, define

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
\tag{2}
\]

$E_\Theta(S,x)$ is a real-valued score called the energy, and $\Theta$ denotes all
learned parameters. A larger energy means a larger relative basket probability. The
normalizer $Z_+(x)$ sums the unnormalized score over every supported nonempty basket, so
the probabilities in Eq. (2) add to one. The plus sign reminds us that the empty basket
is excluded because the data are conditional on an observed checkout.

### 3.2 The Version-4 energy

The complete energy is

\[
\begin{aligned}
E_\Theta(S,x)
={}&
\sum_{j\in S} b_j(x)
+\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k \\
&-\sum_{c=1}^{C_{\mathrm g}}\rho_c {n_c(S)\choose 2}
-\rho_0(|S|).
\end{aligned}
\tag{3}
\]

Each term answers a different modeling question.

| Quantity | Dimension | Interpretation |
|---|---:|---|
| $b_j(x)$ | scalar | Contextual utility of product $j$ before pair interactions |
| $\phi_j$ | $r$-vector | Interaction embedding of product $j$ |
| $\phi_j^\top\phi_k$ | scalar | Product-specific pair interaction |
| $\rho_c$ | scalar | Shared pair penalty or reward within affinity group $c$ |
| $\rho_0(n)$ | scalar for each $n=1,\ldots,n_{\max}$ | Flexible population-level basket-size potential |

The interaction embeddings form a matrix $\Phi\in\mathbb R^{J\times r}$ whose $j$th
row is $\phi_j^\top$. Their Gram matrix is

\[
K=\Phi\Phi^\top\succeq 0.
\tag{4}
\]

Although $K$ is positive semidefinite, an off-diagonal entry
$K_{jk}=\phi_j^\top\phi_k$ may be positive, zero, or negative. A positive value raises
the relative score of baskets containing both products; a negative value lowers it. The
affinity term is separate: ${n_c(S)\choose2}$ counts the number of within-group pairs,
and $-\rho_c$ supplies a common effect for those pairs.

### 3.3 Contextual utility of one product

The contextual item utility is

\[
\begin{aligned}
b_j(x_t)
={}&\lambda_j
+\theta_{h(t)}^\top\alpha_j
-a_{h(t)j}\left[
\bar d_t+
\operatorname{softplus}(\kappa_p)(d_{jt}-\bar d_t)
\right] \\
&+w_j^{\mathrm{dsp}}D_{jt}
+w_j^{\mathrm{mlr}}M_{jt}
+\mu_j^\top\delta_{w(t)}
+\zeta_j^\top\xi_{s(t)}.
\end{aligned}
\tag{5}
\]

Here $\operatorname{softplus}(u)=\log(1+e^u)$ is always positive. The variables and
parameters in Eq. (5) are:

| Symbol | Dimension | Meaning |
|---|---:|---|
| $\lambda_j$ | scalar per product | Baseline popularity after other effects are controlled |
| $\theta_h$ | $K$-vector per household | Household taste coordinates |
| $\alpha_j$ | $K$-vector per product | Product taste coordinates |
| $\theta_h^\top\alpha_j$ | scalar | Household $h$'s fitted affinity for product $j$ |
| $d_{jt}$ | scalar | Product $j$'s log-price deviation on trip $t$ |
| $\bar d_t$ | scalar | Mean log-price deviation over the offered assortment |
| $a_{hj}$ | nonnegative scalar | Household--product price sensitivity |
| $\kappa_p$ | scalar | Learned relative-price scaling parameter |
| $D_{jt}$ | binary indicator | Whether product $j$ is on display |
| $M_{jt}$ | binary indicator | Whether product $j$ appears in a mailer |
| $w_j^{\mathrm{dsp}},w_j^{\mathrm{mlr}}$ | scalars per product | Display and mailer effects |
| $\mu_j,\delta_w$ | equal-length vectors | Product and week factors for seasonal variation |
| $\zeta_j,\xi_s$ | equal-length vectors | Product and store factors for store variation |

The price coefficient uses a nonnegative low-rank factorization:

\[
a_{hj}
=
\sum_{k=1}^{K_p}
\operatorname{softplus}(\gamma_{hk})
\operatorname{softplus}(\beta_{jk})
\ge 0.
\tag{6}
\]

$\gamma_h\in\mathbb R^{K_p}$ is a household price factor and
$\beta_j\in\mathbb R^{K_p}$ is a product price factor. Their softplus transforms make
$a_{hj}$ nonnegative. The minus sign in Eq. (5) then prevents the fitted own-price
contribution from increasing utility when the relevant price deviation increases.

The decomposition into $\bar d_t$ and $d_{jt}-\bar d_t$ separates two effects. Moving
all prices together changes the common price level and may change total basket size.
Moving one product relative to the trip average mainly reallocates probability across
products. The learned positive factor $\operatorname{softplus}(\kappa_p)$ permits these
responses to have different magnitudes.

The symbol $\beta_j$ in Eq. (6) is a learned **product price factor**. Later, an
unsubscripted scalar $\beta\in[0,1]$ will denote an SMC bridge coordinate. They are
unrelated quantities that happen to share a conventional Greek letter.

### 3.4 Global and household-specific basket size

The vector
$\rho_0(1),\ldots,\rho_0(n_{\max})$ represents the nonlinear population-level shape of
the size distribution. Household variation enters through one reserved coordinate in
the additive taste factorization:

\[
\theta_h=
\begin{bmatrix}\widetilde\theta_h\\ \kappa_h\end{bmatrix},
\qquad
\alpha_j=
\begin{bmatrix}\widetilde\alpha_j\\ 1\end{bmatrix}.
\tag{7}
\]

Consequently,

\[
\theta_h^\top\alpha_j
=
\widetilde\theta_h^\top\widetilde\alpha_j+\kappa_h,
\qquad
\sum_{j\in S}\kappa_h=|S|\kappa_h.
\tag{8}
\]

The effective size potential for household $h$ is therefore

\[
\rho_{0h}(n)=\rho_0(n)-n\kappa_h.
\tag{9}
\]

This equation does not create or store a separate $H\times n_{\max}$ parameter table.
$\rho_{0h}$ is only a convenient way to interpret the sum of the global curve and the
household-common utility. The global curve can be nonlinear in $n$, while $\kappa_h$
provides a parsimonious linear tilt toward larger or smaller baskets. Because $\kappa_h$
adds the same amount to every product, it cannot change the relative probabilities of
two baskets with the same size.

The phrase **rank-one household-size component** refers to the $H\times J$ utility
matrix $\kappa\mathbf 1_J^\top$, which has rank one. It is not the Gram interaction rank
$r$. Centering constraints on household and product coordinates separate this common
size tendency from intercepts and the remaining taste factors.

At this point the probability law is fully defined. Its only apparent obstacle is the
normalizer in Eq. (2), which seems to require an exponential enumeration of baskets.
The next chapter removes that obstacle without changing the law.

---

## 4. Making the normalizer tractable

### 4.1 Rewriting the quadratic interaction

For any basket $S$, let

\[
v(S)=\sum_{j\in S}\phi_j\in\mathbb R^r.
\tag{10}
\]

Expanding the squared norm gives

\[
\|v(S)\|^2
=
\sum_{j\in S}\|\phi_j\|^2
+2\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k.
\]

Rearranging,

\[
\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k
=
\frac12\|v(S)\|^2
-\frac12\sum_{j\in S}\|\phi_j\|^2.
\tag{11}
\]

The second term is additive over products. The first term is the remaining quadratic
coupling.

### 4.2 Hubbard--Stratonovich identity

Let $z\sim\mathcal N(0,I_r)$ be an $r$-dimensional standard Gaussian vector. Its moment
generating function states that, for any fixed $v\in\mathbb R^r$,

\[
\mathbb E_z[e^{z^\top v}]=e^{\|v\|^2/2}.
\tag{12}
\]

This is the Hubbard--Stratonovich identity in the form needed here. It replaces an
exponential of a quadratic expression with an expectation of an exponential that is
linear in $v$. Substituting Eq. (10) makes that exponent additive over products.

Define the conditional weight

\[
w_j(z,x)
=
\exp\left\{
b_j(x)-\frac12\|\phi_j\|^2+z^\top\phi_j
\right\}>0.
\tag{13}
\]

For a fixed $z$, all product-specific interaction information is now contained in
ordinary positive item weights $w_j(z,x)$. The remaining sum over baskets can therefore
be organized by basket size and affinity-group counts.

### 4.3 Elementary symmetric polynomials

For numbers $w_1,\ldots,w_m$, the elementary symmetric polynomial of degree $k$ is

\[
e_k(w_1,\ldots,w_m)
=
\sum_{1\le i_1<\cdots<i_k\le m}
w_{i_1}\cdots w_{i_k},
\qquad e_0=1.
\tag{14}
\]

It is exactly the sum of products of weights over all size-$k$ subsets. Thus it sums all
possible choices of $k$ distinct products without listing the subsets one by one.

For the products in affinity group $c$ that are offered in context $x$, define
$\mathcal A_{xc}=\mathcal A_x\cap\mathcal G_c$, where $\mathcal G_c$ is the set of
products assigned to group $c$. The group polynomial is

\[
G_c(u;z,x)
=
\sum_{k=0}^{\min(|\mathcal A_{xc}|,n_{\max})}
e^{-\rho_c{k\choose2}}
e_k\!\left(\{w_j(z,x):j\in\mathcal A_{xc}\}\right)u^k.
\tag{15}
\]

The formal variable $u$ is only a bookkeeping device: the exponent of $u$ records how
many products were selected. Multiplying the group polynomials and discarding degrees
above $n_{\max}$ gives

\[
\prod_{c=1}^{C_{\mathrm g}}G_c(u;z,x)
=
\sum_{n=0}^{n_{\max}}A_n(z,x)u^n.
\tag{16}
\]

$A_n(z,x)$ is the exact conditional total weight of every basket of size $n$, including
all item weights and affinity-count effects but not yet the global size potential.

### 4.4 Proposition 1: exact discrete summation for fixed latent state

**Proposition.** For a fixed $z$, every coefficient $A_n(z,x)$ can be computed in
polynomial time in catalogue size and $n_{\max}$, without enumerating baskets.

**Proof.** Suppose products are processed one at a time. Let $e_k^{(q)}$ be the degree
$k$ elementary symmetric polynomial after the first $q$ weights. Any size-$k$ subset
either excludes product $q$ or includes it. These two disjoint cases give

\[
e_k^{(q)}
=e_k^{(q-1)}+w_qe_{k-1}^{(q-1)},
\qquad e_0^{(q)}=1.
\tag{17}
\]

The recursion stores only degrees $0,ldots,n_{\max}$. Applying it within each affinity
group computes Eq. (15). Truncated polynomial convolution then combines groups to obtain
Eq. (16). Every admissible subset appears exactly once according to its selected count
in each group. Therefore the discrete summation is exact for fixed $z$. $\square$

Combining Eqs. (12)--(16) yields the Version-4 normalizer:

\[
\boxed{
Z_+(x)
=
\mathbb E_{z\sim\mathcal N(0,I_r)}
\left[
\sum_{n=1}^{n_{\max}}e^{-\rho_0(n)}A_n(z,x)
\right].}
\tag{18}
\]

The enormous discrete sum is exact. Only an $r$-dimensional Gaussian expectation
remains. When $\Phi=0$, the weights no longer depend on $z$, so the expectation vanishes
and the entire nonempty-basket normalizer is computed exactly by dynamic programming.

This mathematical separation suggests the training strategy: learn the large additive
part with the exact normalizer, then learn interactions in a stable low-dimensional
subspace.

---

## 5. Mathematical basis for staged interaction estimation

### 5.1 Proposition 2: the factor gradient vanishes at the additive model

The likelihood depends on $\Phi$ through $K=\Phi\Phi^\top$. Let $L(K)$ denote the
log-likelihood expressed as a function of the Gram matrix. By the chain rule,

\[
\nabla_\Phi L
=
\left(\nabla_KL+\nabla_KL^\top\right)\Phi.
\tag{19}
\]

Therefore

\[
\nabla_\Phi L\big|_{\Phi=0}=0.
\tag{20}
\]

**Proof.** At $\Phi=0$, the matrix multiplying the Gram-space score is multiplied by the
zero factor matrix. The product is zero even when $\nabla_KL$ contains a direction that
would improve likelihood. $\square$

This result explains why joint training from an exactly additive initialization cannot
discover interactions through an ordinary first derivative. Random nonzero
initialization removes the exact zero but introduces arbitrary orientation and expensive
interaction normalization before the much larger additive structure has been learned.

### 5.2 Rotation does not identify a unique factor matrix

For every orthogonal matrix $Q\in\mathbb R^{r\times r}$ satisfying $Q^\top Q=I_r$,

\[
(\Phi Q)(\Phi Q)^\top=\Phi\Phi^\top.
\tag{21}
\]

Thus many numerical factor matrices represent the same probability law. Individual
coordinates of $\phi_j$ have no unique meaning; pairwise dot products and the subspace
are the identified objects. Directly optimizing arbitrary coordinates can waste effort
moving through rotations that do not change any basket probability.

### 5.3 Stable directions in natural Gram space

Let $P(S)$ be the symmetric matrix whose off-diagonal entry $(j,k)$ records whether
products $j$ and $k$ co-occur in $S$ and whose diagonal is zero. Compare observed pair
incidence with the additive parent's expected pair incidence:

\[
R
=
\mathbb E_{\mathrm{data}}[P(S)]
-\mathbb E_{p_{\mathrm{add}}}[P(S)].
\tag{22}
\]

A positive eigenvector of $R$ is a locally improving positive-semidefinite interaction
direction. The pipeline computes candidate subspaces on two chronological training
halves and checks their overlap. This asks whether the interaction directions repeat in
independent portions of training data instead of reflecting sampling noise.

Let $U\in\mathbb R^{J\times r}$ contain the accepted orthonormal directions. To avoid
confusing this matrix with the number $C_{\mathrm g}$ of affinity groups, denote the
small fitted interaction matrix by $C_{\mathrm{int}}$. The Gram matrix is parameterized as

\[
K=UC_{\mathrm{int}}U^\top,
\qquad C_{\mathrm{int}}\in\mathbb R^{r\times r},
\qquad C_{\mathrm{int}}\succeq0.
\tag{23}
\]

The small positive-semidefinite matrix $C_{\mathrm{int}}$ is the natural parameter
optimized during the interaction stage. After fitting, $C_{\mathrm{int}}^{1/2}$ is any
symmetric positive-semidefinite square
root and

\[
\Phi=UC_{\mathrm{int}}^{1/2}
\tag{24}
\]

recovers an embedding with the desired Gram matrix. This procedure changes the
optimization coordinates, not the Version-4 probability law.

---

## 6. The training pipeline from raw data to a locked model

The stage order below follows the mathematics just established. Earlier stages estimate
broad, well-supported structure cheaply. Later stages add the smaller residual
components that require more expensive computation. Every stage records its data
fingerprint, parent checkpoint, configuration, and acceptance evidence so that a resumed
run cannot silently combine incompatible artifacts.

### 6.1 Stage A: audit and preprocess the raw data

The pipeline verifies raw-file digests, reconstructs checkout baskets, creates the
chronological split, selects products and households from training only, and builds the
price, promotion, week, store, and assortment panels. It independently reconstructs key
outcomes as an audit. Its output is a deterministic modeled dataset and fingerprint.

### 6.2 Stage B: construct the affinity partition from training data

Supported positive-lift co-purchase edges group products into $C_{\mathrm g}$ affinity blocks. These
fixed blocks define $n_c(S)$ and the group polynomials in Eq. (15). This stage estimates
no household, price, size, or interaction-embedding parameter. “Affinity partition” is
only the name of a product grouping used by the model and dynamic program.

### 6.3 Stage C: initialize the complete parameter structure

Training-only moments initialize product intercepts and other additive blocks. One taste
coordinate is reserved for the common household loading in Eq. (7), and the remaining
product taste coordinates are centered across the catalogue. The interaction matrix is
set exactly to $\Phi=0$.

### 6.4 Stage D: fit the exact additive parent

With $\Phi=0$, the pipeline maximizes

\[
\ell_{\mathrm{add}}(\Theta)
=
\sum_{t\in\mathrm{train}}
\left[
E_{\mathrm{add}}(S_t,x_t)
-\log Z_{+,\mathrm{add}}(x_t)
\right].
\tag{25}
\]

The dynamic program computes $Z_{+,\mathrm{add}}$ exactly. This stage jointly learns
$\lambda_j$, $\theta_h$, $\alpha_j$, $\gamma_h$, $\beta_j$, $\kappa_p$, promotion
effects, seasonal factors, store factors, $\rho_c$, and the vector $\rho_0$. It therefore
already learns the base household-size coordinate $\kappa_h$ contained in $\theta_h$.

Regularization pools weakly observed parameters and calibrates aggregate basket-size and
price-elasticity behavior. Validation likelihood determines learning-rate reductions and
the convergence gate. No Gram interaction is active in this stage.

### 6.5 Stage E: select a stable interaction rank

The residual matrix in Eq. (22) is computed on the full training set and on two halves.
Candidate ranks are evaluated by positive eigenvalue evidence and subspace stability.
The largest accepted rank is used. If rank 8 is unstable, the procedure checks each lower
candidate rather than treating rank 5 as a theoretical default. The selected $r$ is a
data-supported capacity decision; it is not a quadrature setting.

### 6.6 Stage F: fit interactions by fixed-proposal conditional likelihood

Let $u_j^\top$ be row $j$ of $U$. For a basket $S$, define the symmetric statistic

\[
F_U(S)
=
\frac12\left[
\left(\sum_{j\in S}u_j\right)
\left(\sum_{j\in S}u_j\right)^\top
-\sum_{j\in S}u_ju_j^\top
\right].
\tag{26}
\]

The interaction energy under $K=UC_{\mathrm{int}}U^\top$ is
$\operatorname{tr}\{C_{\mathrm{int}}F_U(S)\}$. Denote the two residual size
coefficients by $a_{\mathrm s}$ and $c_{\mathrm s}$ so they cannot be confused with a
product or affinity-group index. The small correction fitted at this stage is

\[
h_{C_{\mathrm{int}},a_{\mathrm s},c_{\mathrm s}}(S)
=
\operatorname{tr}\{C_{\mathrm{int}}F_U(S)\}
-a_{\mathrm s}\frac{|S|}{10}
-c_{\mathrm s}\left(\frac{|S|}{10}\right)^2.
\tag{27}
\]

$a_{\mathrm s}$ and $c_{\mathrm s}$ are scalar residual adjustments inside the original flexible size
potential. They are not a new basket law; they prevent a newly introduced pair energy
from being confused with a broad linear or quadratic increase in size.

Choose $M$ training contexts. For context $m$, draw $D$ exact proposal baskets
$S_{md}$ from the additive parent and keep those draws fixed during the solve. If
$S_m^{\mathrm{obs}}$ is the observed basket, optimize

\[
\widehat G(C_{\mathrm{int}},a_{\mathrm s},c_{\mathrm s})
=
\frac1M\sum_{m=1}^{M}
\left[
h_{C_{\mathrm{int}},a_{\mathrm s},c_{\mathrm s}}(S_m^{\mathrm{obs}})
-\log\left{
\frac1D\sum_{d=1}^{D}e^{h_{C_{\mathrm{int}},a_{\mathrm s},c_{\mathrm s}}(S_{md})}
\right\}
\right].
\tag{28}
\]

The second term estimates the logarithm of the child-to-parent normalizer ratio. Fixed
draws make Eq. (28) a deterministic concave function of the natural parameters: a
linear term minus a log-sum-exp term. The constraint $C_{\mathrm{int}}\succeq0$ and declared tail
constraints form a convex feasible set, so projected ascent with backtracking has a
clear acceptance rule.

The finite-$D$ log-mean-exp still has Monte Carlo error and finite-sample bias. The
pipeline consequently cross-fits the estimated gain, checks proposal effective sample
size, and later verifies the accepted checkpoint with an independent Smolyak likelihood.

### 6.7 Stage G: recalibrate only the residual household-size direction

Let $\kappa_h^{(0)}$ be the common household coordinate already learned in Stage D. After
interactions and the global size correction have changed the distribution, let

\[
q_x(n)=p_{\mathrm{post-int}}(N=n\mid x)
\tag{29}
\]

be the new parent size law. The old additive optimum need not remain optimal because its
expected basket size has changed. The score for the common household coordinate is

\[
\frac{\partial\ell}{\partial\kappa_h}
=
\sum_{t:h(t)=h}
\left[N_t-\mathbb E_\Theta(N\mid x_t)\right].
\tag{30}
\]

Rather than reopening every additive parameter, Stage G tests a one-dimensional residual
$\delta_h$ for each household:

\[
q_{x,\delta_h}(n)
=
\frac{q_x(n)e^{n\delta_h}}
{\sum_{m=1}^{n_{\max}}q_x(m)e^{m\delta_h}}.
\tag{31}
\]

With ridge parameter $\lambda>0$, its objective is

\[
L_h(\delta_h)
=
\sum_{t:h(t)=h}
\left[
N_t\delta_h
-\log\sum_{m=1}^{n_{\max}}q_{x_t}(m)e^{m\delta_h}
\right]
-\frac{\lambda}{2}\delta_h^2.
\tag{32}
\]

Its second derivative is

\[
L_h''(\delta_h)
=
-\sum_{t:h(t)=h}
\operatorname{Var}_{\delta_h}(N\mid x_t)-\lambda<0.
\tag{33}
\]

Therefore each household problem is strictly concave and has one unique optimum. Ridge
strength is selected using swapped chronological household-day folds. The final common
coordinate is

\[
\kappa_h^{\mathrm{final}}=\kappa_h^{(0)}+\delta_h.
\tag{34}
\]

This is not a second independent fit of the original rank-one component. Stage D learns
the base coordinate under the additive law; Stage G tests only the residual correction
required after the interaction law changes expected size. If the simultaneous
cross-fitted lower confidence bound is not positive, every $\delta_h$ is set to zero and
the parent checkpoint is preserved.

### 6.8 Stage H: lock evaluation and production certification

Validation decides whether a candidate is accepted. The untouched test set reports
performance but does not tune the model. Fixed manifests support paired comparisons for
complete-support likelihood, recommendation, generation, counterfactual behavior,
segments, and interaction diagnostics.

Certification then screens the full supported population with a cheaper quadrature rule
and recomputes high-risk contexts more accurately. It fails closed when lineage,
convergence, likelihood evidence, quadrature agreement, or localized extreme-basket
probability fails. This localized test matters even when the population-average tail rate
looks acceptable: a small set of contexts can otherwise place most probability on
implausibly large baskets.

---

## 7. Numerical integration and likelihood certification

Equation (18) reduced the original discrete problem to an $r$-dimensional Gaussian
expectation. Smolyak quadrature approximates only this continuous integral. It never
limits the catalogue to 20 products and never truncates baskets below the declared
$n_{\max}$.

For fixed interaction rank $r$, a level-$q$ Smolyak rule supplies nodes
$z^{(1)},\ldots,z^{(M_q)}$ and weights $\omega_1,\ldots,\omega_{M_q}$. Schematically,

\[
\mathbb E_z[f_x(z)]
\approx
\sum_{m=1}^{M_q(r)}\omega_m f_x(z^{(m)}),
\tag{35}
\]

where

\[
f_x(z)=\sum_{n=1}^{n_{\max}}e^{-\rho_0(n)}A_n(z,x).
\]

Rank and level have separate roles:

| Quantity | What it controls | Changes model capacity? |
|---|---|---:|
| $r$ | Dimension of $\phi_j$ and interaction subspace | Yes |
| $q$ | Accuracy and cost of Gaussian integration | No |
| $M_q(r)$ | Number of quadrature nodes implied by $q$ and $r$ | No |

The selected pipeline uses $q=r+1$ for broad screening, $q=r+2$ for reported likelihood
and high-risk confirmation, and $q=r+3$ on a smaller audit panel. Agreement between
adjacent levels estimates quadrature error for fixed parameters. It does not measure
variation across test baskets, optimizer convergence, minibatch noise, or model
misspecification.

The per-basket held-out log-likelihood is

\[
\ell_t=E_\Theta(S_t,x_t)-\log Z_{+,\Theta}(x_t).
\tag{36}
\]

Paired model comparisons use the same baskets and contexts. If model A and model B have
scores $\ell_t^A$ and $\ell_t^B$, the estimated mean gain is the average of
$d_t=\ell_t^A-\ell_t^B$, and its uncertainty is computed from the paired differences,
not from two unrelated standard deviations. This pairing removes much of the trip-level
difficulty shared by both models.

With the law trained and its likelihood numerically certified, we can now derive each
retail function directly from that same law.

---

## 8. Retail functions derived from the fitted law

### 8.1 Missing-item recommendation

Suppose $R$ is the observed remainder of a basket after exactly one item is hidden. For
candidate $j\in\mathcal A_x\setminus R$, define

\[
s_j(R,x)=E_\Theta(R\cup\{j\},x).
\]

Conditioning Eq. (2) on the event that one offered item completes $R$ gives

\[
\begin{aligned}
P(j\text{ completes }R\mid x,R,\text{one missing})
&=
\frac{p_\Theta(R\cup\{j\}\mid x)}
{\sum_{k\in\mathcal A_x\setminus R}p_\Theta(R\cup\{k\}\mid x)}\\
&=
\frac{e^{s_j(R,x)}/Z_+(x)}
{\sum_{k\in\mathcal A_x\setminus R}e^{s_k(R,x)}/Z_+(x)}\\
&=
\frac{e^{s_j(R,x)}}
{\sum_{k\in\mathcal A_x\setminus R}e^{s_k(R,x)}}.
\end{aligned}
\tag{37}
\]

The full normalizer cancels. Every candidate completion also has the same total size, so
common size terms cancel. Recommendation therefore uses the trained joint law but needs
neither Smolyak integration nor a separately trained recommendation objective.

MRR, or mean reciprocal rank, evaluates where the hidden item appears in the ranking. If
its rank is $r_t$, that case contributes $1/r_t$; MRR averages these values. MRR@$K$
assigns zero when $r_t>K$. Recall@$K$ is the fraction of cases with $r_t\le K$. A better
joint likelihood need not mechanically produce a better finite-sample MRR because
likelihood scores all aspects of the complete basket, whereas MRR tests one particular
conditional ranking.

### 8.2 Complete basket generation from the probability law

Equation (18) defines the augmented joint density

\[
p_\Theta(S,z\mid x)
\propto
\varphi_r(z)
e^{-\rho_0(|S|)}
\prod_{c=1}^{C_{\mathrm g}}e^{-\rho_c{n_c(S)\choose2}}
\prod_{j\in S}w_j(z,x),
\tag{38}
\]

where $\varphi_r$ is the standard $r$-dimensional Gaussian density. Integrating out $z$
recovers exactly the basket law in Eq. (2).

If $z$ were already drawn from its correct marginal distribution, the four conditional
generation steps would be:

1. Compute all group coefficients in Eq. (15) and their convolution in Eq. (16).
2. Draw total size $N=n$ with probability proportional to
   $e^{-\rho_0(n)}A_n(z,x)$ for $n=1,\ldots,n_{\max}$.
3. Conditional on $N$, draw the count allocated to each affinity group using the forward
   and backward polynomial coefficients.
4. Within each group, draw the requested number of distinct products by reverse
   elementary-symmetric-polynomial recursion.

These steps produce a complete basket. No post-generation correction changes its
constituents. The only unresolved task is drawing $z$ from its model-implied marginal,
which is proportional to the Gaussian density multiplied by the complete polynomial
integrand. A standard Gaussian draw is generally incorrect once interactions are active.
The interaction-tempered SMC construction below solves precisely this sampling problem.

### 8.3 Interaction-tempered sequential Monte Carlo

#### 8.3.1 Particle representation and bridge law

Sequential Monte Carlo, abbreviated SMC, represents a target distribution by a population
of random candidates called particles. Here one particle is one complete basket for a
fixed context; it is not a product and not a learned parameter.

Separate the Gram interaction from the rest of the fitted energy:

\[
E_\Theta(S,x)=E_0(S,x)+V_\Phi(S),
\qquad
V_\Phi(S)=\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k.
\tag{39}
\]

$E_0$ retains contextual utility, household effects, price, promotions, affinity counts,
and the final total-size potential. Define intermediate laws

\[
p_\beta(S\mid x)
=
\frac{\exp\{E_0(S,x)+\beta V_\Phi(S)\}}
{Z_\beta(x)},
\qquad 0\le\beta\le1.
\tag{40}
\]

At $\beta=0$, only the Gram interaction is absent and exact dynamic-program sampling is
available. At $\beta=1$, the distribution is the complete fitted Version-4 law.
Intermediate values are numerical stepping stones, not separately fitted models.

#### 8.3.2 Exact augmented representation at every bridge value

Let $m(S)=\sum_{j\in S}\phi_j$. At bridge value $\beta$, consider

\[
\widetilde p_\beta(S,z\mid x)
\propto
\exp\left\{
E_0(S,x)-\frac12\|z\|^2
+\sqrt\beta\,z^\top m(S)
-\frac\beta2\sum_{j\in S}\|\phi_j\|^2
\right\}.
\tag{41}
\]

Completing the square,

\[
-\frac12\|z\|^2+\sqrt\beta\,z^\top m(S)
=
-\frac12\|z-\sqrt\beta\,m(S)\|^2
+\frac\beta2\|m(S)\|^2.
\tag{42}
\]

After $z$ is integrated out, Eqs. (11) and (42) leave exactly
$E_0(S,x)+\beta V_\Phi(S)$. Thus Eq. (41) has basket marginal Eq. (40), and

\[
z\mid S,x,\beta
\sim
\mathcal N\!\left(\sqrt\beta\,m(S),I_r\right).
\tag{43}
\]

Conditional on $z$, product $j$ has log weight

\[
\eta_j(z,\beta,x)
=
b_j(x)-\frac\beta2\|\phi_j\|^2
+\sqrt\beta\,z^\top\phi_j.
\tag{44}
\]

The four exact conditional basket steps from Section 8.2 can therefore be used at every
bridge value.

#### 8.3.3 Weighting, resampling, and rejuvenation

Choose a schedule $0=\beta_0<\beta_1<\cdots<\beta_L=1$. A particle basket from the
previous level receives incremental weight

\[
W_\ell^{(p)}
=
\exp\left\{
(\beta_\ell-\beta_{\ell-1})V_\Phi(S_{\ell-1}^{(p)})
\right\}.
\tag{45}
\]

The three SMC operations have separate purposes:

| Operation | Purpose in this model |
|---|---|
| Weighting | Identifies baskets favored by the next interaction increment |
| Resampling | Gives high-weight baskets more descendants and removes low-weight ones |
| Rejuvenation | Restores diversity through a transition that preserves $p_{\beta_\ell}$ |

Rejuvenation first samples $z$ from Eq. (43), then samples a new complete basket from
Eq. (44) using the exact size, group-count, and reverse-ESP recursions. This blocked Gibbs
step may change basket size, group allocation, and actual products while leaving the
current bridge law invariant.

The normalized effective sample size is

\[
\frac{\operatorname{ESS}_\ell}{N_{\mathrm{part}}}
=
\frac{1}{N_{\mathrm{part}}\sum_{p=1}^{N_{\mathrm{part}}}(\overline W_\ell^{(p)})^2},
\tag{46}
\]

where $N_{\mathrm{part}}$ is the number of particles, $p$ indexes a particle, and
$\overline W_\ell^{(p)}$ is its normalized weight.
A value near one indicates balanced weights at that transition. A low value indicates
weight concentration. High ESS alone is not proof of correctness because all particles
could still miss a distant mode; repeated-seed and held-out calibration checks remain
necessary.

#### 8.3.4 Normalizer estimate and statistical limit

Let $Z_0(x)$ and $Z_1(x)$ denote the normalizers of the endpoint laws $p_0$ and $p_1$
from Eq. (40). $Z_0(x)$ is available from the exact no-Gram dynamic program; $Z_1(x)$
is the full interaction normalizer that SMC estimates. The estimate is

\[
\widehat Z_1(x)
=
Z_0(x)
\prod_{\ell=1}^{L}
\left[
\frac1{N_{\mathrm{part}}}\sum_{p=1}^{N_{\mathrm{part}}}W_\ell^{(p)}
\right].
\tag{47}
\]

With exact $p_0$ initialization, unbiased resampling, and invariant bridge kernels,
$\widehat Z_1$ is unbiased on the $Z$ scale. Its logarithm is not unbiased:

\[
\mathbb E[\log\widehat Z_1]\le\log Z_1
\tag{48}
\]

by Jensen's inequality. Increasing particles, improving bridge overlap, and using
independent repetitions reduce and diagnose finite-particle error. Terminal particles
converge to $p_1$ as $N_{\mathrm{part}}$ grows, but a finite resampled population is not independent and
identically distributed because particles may share ancestors.

The implemented quadratic schedule has 17 reported levels and 16 transitions,

\[
\beta_\ell
=1-\left(1-\frac{\ell}{16}\right)^2,
\qquad \ell=0,\ldots,16,
\tag{49}
\]

with closer levels near the full interaction endpoint. Schedule and particle count alter
Monte Carlo cost and error, not the definition of the target law. The bridge does not
train, shrink, or repair $\Phi$.

### 8.4 Price counterfactuals

To study a proposed price change, modify the price variables inside context $x$, recompute
$b_j(x)$ through Eq. (5), and query the new basket law. For a statistic $g(S)$ and two
energies $E_0,E_1$, importance reweighting gives

\[
\mathbb E_1[g(S)]
=
\frac{\mathbb E_0\!\left[g(S)e^{E_1(S)-E_0(S)}\right]}
{\mathbb E_0\!\left[e^{E_1(S)-E_0(S)}\right]}.
\tag{50}
\]

This identity supports small counterfactual changes when factual samples overlap the new
law. Effective sample size must be checked; a large price intervention may require fresh
sampling from the counterfactual model. Because the fitted data are observational, these
are structural predictions under the fitted law, not automatic causal treatment effects.

### 8.5 Household segments and bundle candidates

Households can be summarized using fitted taste coordinates, price sensitivity, and the
common size tendency $\kappa_h$. Clustering those summaries produces descriptive
segments. The segment label does not replace the household-level model; it is a reporting
and policy layer that makes heterogeneous behavior easier to communicate.

A fitted pair score $\phi_j^\top\phi_k$ measures residual co-occurrence after the modeled
additive context is controlled. Candidate bundles should require both a favorable fitted
interaction and empirical held-out support. A high dot product alone is not causal proof
that promoting one product creates demand for the other.

### 8.6 Budget-constrained promotion simulation

For a campaign of finite duration, a state may contain the current period, remaining
expected markdown budget, and customer segment. An action chooses no promotion or a
segment-specific product or bundle discount. The counterfactual basket law supplies
expected product incidence and markdown use. Dynamic programming can then select a
feasible sequence of actions.

The real-data model does not identify visits, purchased units, wholesale costs, inventory,
or causal promotion response. Consequently, this calculation is appropriate for
shortlisting campaigns and designing controlled tests, not for claiming guaranteed
profit optimization. The separate synthetic-retailer experiment adds these variables
under known truth to test policy recovery.

---

## 9. Computational cost and the reason for each separation

Let $B$ be contexts in a minibatch, $J_x=|\mathcal A_x|$, $C_x$ the number of active
affinity groups, $M_q(r)$ the number of Smolyak nodes, $M$ the number of interaction-fit
contexts, and $D$ the fixed proposal draws per context.

| Operation | Leading computational work |
|---|---|
| Exact additive normalizer | $O\!\left(B[J_xn_{\max}+C_xn_{\max}^2]\right)$ |
| Sparse interaction score | Sparse observed/generated pair counts plus a leading eigensolve |
| Natural interaction statistics | $O(MDr^2)$ |
| Household residual size fit | $O(Tn_{\max}\log(1/\varepsilon))$ for numerical tolerance $\varepsilon$ |
| Smolyak likelihood | $O\!\left(TM_q(r)[J_xn_{\max}+C_xn_{\max}^2]\right)$ |
| Add-one recommendation | $O(J_xr)$ without evaluating $Z_+(x)$ |

The expensive multiplier is $M_q(r)$. The staged pipeline avoids paying it while the
large additive parameter blocks learn first-order structure. It then estimates a small
interaction matrix $C_{\mathrm{int}}$, and reserves high-accuracy quadrature for fixed evaluation and
certification panels. This separation follows from the factor-gradient result and the
normalizer complexity; it is not an arbitrary freezing convention.

---

## 10. Evaluation, interpretation, and limits

### 10.1 What a successful execution establishes

A successful full pipeline establishes that:

- raw, processed, and checkpoint fingerprints agree;
- the exact additive optimizer met its declared convergence rule;
- the chosen interaction subspace was stable across training halves;
- the constrained interaction correction passed cross-fit and proposal-quality gates;
- validation likelihood gain survived independent quadrature audit;
- recommendation, generation, counterfactual, and segment evaluations used the same
  final checkpoint; and
- aggregate and localized basket-size safety checks passed.

It does not establish causal complementarity, randomized price response, store-visit
probability, purchased quantities, profit, or deployment-ready policy value.

### 10.2 The three central design decisions in connected form

The household-size component first enters through the common coordinate $\kappa_h$ in
the exact additive fit. Interactions then alter expected basket size, so the later
$\delta_h$ step tests only a residual correction. Unsupported corrections revert to
zero. The component is therefore calibrated twice against two different parent laws but
is not independently relearned twice.

The interaction embedding is staged because the factor gradient is zero at $\Phi=0$,
the factor orientation is rotationally unidentified, and active interactions require a
multidimensional integral. Stable Gram-space directions and a small PSD natural matrix
address these three facts without changing the model.

The SMC coordinate $\beta$ exists only after fitting, when complete samples from the
full interaction law are required. It connects an exactly sampleable no-Gram law to the
same fitted Version-4 target. It controls sampling overlap, variance, and runtime, but it
does not add a model parameter or modify the endpoint distribution.

### 10.3 Current implementation qualifications

The selected implementation was re-audited on 5 September 2026. The basket energy,
normalizer, exact additive dynamic program, spectral rank selection, constrained natural
interaction fit, residual household-size fit, Smolyak evaluation, locked recommendation
calculation, and interaction-tempered SMC agree with the derivations above.

Four qualifications must remain visible:

1. Historical numerical results predate the newest artifact-lineage and simultaneous
   ridge-selection hardening. They describe the same model law, but a fresh complete run
   is required to certify the latest code revision.
2. The recommendation evaluator reports additive utility, structured no-Gram, and full
   interaction scores. The clean Gram-only comparison is full interaction versus
   structured no-Gram. A full-versus-utility contrast also includes non-Gram structure
   and must not be labelled interaction-only.
3. The full-profile certification records generation diagnostics, but segment-level
   generation calibration is not yet a hard gate. Any held-out generation-size mismatch
   must remain explicit in a production-readiness statement.
4. The real-data checkpoint models conditional nonempty product incidence. It does not
   by itself provide quantities, visit/no-purchase probability, inventory, cost, or
   causal treatment effects.

---

## 11. Implementation map and reproducibility checks

The mathematical chapters above deliberately avoid requiring code knowledge. This final
map helps a reader connect the theory to the selected executable components.

| Mathematical responsibility | Selected implementation |
|---|---|
| Contextual utility, observed-basket energy, and common parameterization | `scripts/version4/ragged.py` |
| End-to-end stage graph, artifact lineage, and gates | `scripts/run_pipeline.py` |
| Exact additive likelihood | `scripts/version4/fit_exact_additive.py` |
| Stable interaction subspace and rank selection | `scripts/version4/build_spectral_phi_initialization.py` |
| Natural PSD interaction fit | `scripts/version4/fit_convex_natural_interactions.py` |
| Residual household-size correction | `scripts/version4/fit_household_size_rank1.py` |
| Smolyak parent/child likelihood comparison | `scripts/version4/compare_rank8_parent_likelihood.py` |
| Locked missing-item recommendation | `scripts/version4/eval_smolyak_rank8_mrr.py` |
| Interaction-tempered generation and counterfactual audit | `scripts/version4/audit_particle_counterfactual_generation.py` |
| SMC bridge mechanics | `scripts/version4/tempered_ais.py` |

The practical pipeline is resumable stage by stage, but reuse is permitted only when the
required artifact exists and its fingerprint, parent lineage, schema, and acceptance
metadata match the current run. Otherwise the dependent stage is recomputed. This rule
makes automated resurrection safe without requiring a person to monitor and manually
choose checkpoints.

The single conceptual summary is therefore:

\[
\boxed{
\begin{array}{c}
\text{Version-4 defines one conditional law over complete nonempty baskets.}\\
\text{The theorem makes its discrete normalizer exact for each latent state.}\\
\text{Staged fitting solves an optimization and identification problem.}\\
\text{Smolyak certifies likelihood; SMC samples the full fitted law.}\\
\text{Recommendation and counterfactuals are conditional queries of that same law.}
\end{array}}
\]
