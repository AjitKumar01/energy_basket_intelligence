# The Version-4 energy-basket repository: an end-to-end textbook

## Purpose, scope, and reading order

This chapter explains the complete real-data workflow implemented by this repository. It
starts with the retail question and the observed data, defines the probability model,
derives the tractable normalizer, explains why estimation is staged, proves the main
properties of the estimators, and then derives recommendation, basket generation, price
counterfactuals, segmentation, and policy simulation from the fitted probability law.
The final sections map the mathematics to the repository and state the computational
cost and the limits of the evidence.

The mathematical model is the Version-4 model in `paper/version4.html`. Nothing in this
chapter replaces that law. The selected pipeline changes how its parameters are estimated
and how its predictions are audited. This distinction is important:

- a **model identity** is an equality that follows from the probability law;
- an **estimator property** describes a numerical approximation to that identity;
- an **optimization guarantee** describes the objective actually optimized; and
- an **empirical gate** is a test that a fitted checkpoint must pass.

These four statements are not interchangeable. For example, an unbiased estimator of a
partition-function ratio does not make its logarithm unbiased, and a concave sampled
objective does not prove that the statistical model is correctly specified.

The complete flow is

\[
\boxed{
\begin{gathered}
\text{raw transactions and contexts}
\longrightarrow \text{audited nonempty baskets}\\
\longrightarrow \text{Version-4 probability law}
\longrightarrow \text{exact additive fit}\\
\longrightarrow \text{stable interaction subspace}
\longrightarrow \text{size-stratified interaction fit}\\
\longrightarrow \text{household size recalibration}
\longrightarrow \text{independent certification}\\
\longrightarrow \text{recommendation, generation, counterfactuals, and policy analysis}.
\end{gathered}}
\tag{1}
\]

Each arrow is developed in that order below. No later object is required to understand an
earlier definition.

---

## 1. The retail problem represented by the model

### 1.1 The observed outcome

A checkout occasion supplies a context \(x\) and a basket \(S\). The context records the
household, store, date, available price and promotion features, and the declared offered
catalogue. The basket is the set of distinct stock-keeping units purchased at that
checkout. If a product is bought multiple times, it still appears once in the incidence
basket; unit counts are stored but are not fitted by the selected real-data objective.

Let

- \(J\) be the number of modeled products;
- \(\mathcal A_x\subseteq\{1,\ldots,J\}\) be the offered products in context \(x\);
- \(S\subseteq\mathcal A_x\) be a basket;
- \(N=|S|\) be its number of distinct products;
- \(n_{\max}=120\) be the declared maximum supported size; and
- \(n_c(S)\) be the number of products in affinity group \(c\).

The real-data observations are completed purchase trips. They do not contain every
opportunity on which a household could have visited but bought nothing. The fitted law is
therefore conditional on a nonempty checkout:

\[
P(S\mid x,\,N\ge 1).
\tag{2}
\]

This conditioning is not a technical omission. It determines what may be claimed. The
model can compare products and baskets conditional on a trip. Without opportunity or
visit data it cannot estimate the unconditional probability that a household shops or
the probability of the empty basket.

### 1.2 The decisions supported after fitting

One coherent basket law is used for several retail questions:

1. **Likelihood:** how much probability was assigned to a held-out basket?
2. **Recommendation:** which missing product best completes a partially observed basket?
3. **Generation:** what complete baskets are plausible for a household and context?
4. **Price counterfactuals:** how does the fitted distribution change when price features
   are changed?
5. **Segmentation:** how do household taste, price response, and size tendency differ?
6. **Bundle discovery:** which fitted pair effects are strong and supported by held-out
   co-incidence evidence?
7. **Policy shortlisting:** under a markdown budget, which segment-specific actions merit
   a controlled business test?

There is no separately trained recommendation model. These functions are different
conditionings, expectations, or simulations under the same fitted law.

### 1.3 Audited data represented in the repository

The verified real-data build contains:

| Quantity | Value |
|---|---:|
| Raw transaction rows | 2,595,732 |
| Raw product rows | 92,353 |
| Raw promotion rows | 36,786,524 |
| Modeled products | 5,455 |
| Eligible households | 1,920 |
| Modeled baskets | 200,698 |
| Training baskets | 160,007 |
| Validation baskets | 17,351 |
| Test baskets | 23,340 |
| Mean distinct-product basket size | 7.648 |
| Basket-size variance | 80.691 |
| Maximum supported size | 120 |

Products and households are selected using training weeks only. Weeks 9--82 are
training, weeks 83--90 are validation, and weeks 91--101 are test. A content digest binds
the raw sources, preprocessing policy, product ordering, affinity partition, and ragged
index. Checkpoints with a different digest are rejected even when their tensor shapes
match.

---

## 2. The Version-4 probability law

### 2.1 Energy and normalization

For a supported nonempty basket \(S\), the model defines an energy

\[
\begin{aligned}
E_\Theta(S,x)
={}&\sum_{j\in S}b_j(x)
+\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k\\
&-\sum_c\rho_c {n_c(S)\choose 2}
-\rho_0(|S|).
\end{aligned}
\tag{3}
\]

The basket probability is

\[
p_\Theta(S\mid x)
=\frac{\exp\{E_\Theta(S,x)\}}{Z_{+,\Theta}(x)},
\qquad
Z_{+,\Theta}(x)
=\sum_{\substack{A\subseteq\mathcal A_x\\1\le |A|\le n_{\max}}}
\exp\{E_\Theta(A,x)\}.
\tag{4}
\]

The numerator gives a positive relative weight to every supported basket. The
normalizer is the sum of those weights, so

\[
\sum_{S}p_\Theta(S\mid x)
=\frac{1}{Z_{+,\Theta}(x)}\sum_S\exp\{E_\Theta(S,x)\}=1.
\tag{5}
\]

This calculation explains why the normalizer is required: without it, energies compare
baskets but do not define probabilities.

The plus sign in \(Z_+\) records that the empty basket is excluded. If an opportunity
model later supplies a purchase probability \(\pi(x)=P(N\ge1\mid x)\), a complete demand
law can be formed as

\[
P(S\mid x)=
\begin{cases}
1-\pi(x),&S=\varnothing,\\
\pi(x)p_\Theta(S\mid x,N\ge1),&S\ne\varnothing.
\end{cases}
\tag{6}
\]

That extension must be trained on observed purchase opportunities; it cannot be inferred
from checkout-only data.

### 2.2 Contextual product utility

The item utility implemented by the selected model is

\[
\begin{aligned}
b_j(x)={}&\lambda_j+\theta_h^\top\alpha_j
-g_{hj}\{\bar d_x+\kappa_p(d_{jx}-\bar d_x)\}\\
&+w_j^{\mathrm{dsp}}D_{jx}+w_j^{\mathrm{mlr}}M_{jx}
+\mu_j^\top\delta_{w(x)}+\zeta_j^\top\xi_{s(x)},
\end{aligned}
\tag{7}
\]

where:

| Symbol | Meaning |
|---|---|
| \(\lambda_j\) | baseline popularity of product \(j\) |
| \(\theta_h\in\mathbb R^K\) | taste coordinates of household \(h\) |
| \(\alpha_j\in\mathbb R^K\) | taste coordinates of product \(j\) |
| \(d_{jx}\) | log-price deviation for product \(j\) in context \(x\) |
| \(\bar d_x\) | average price movement across the offered catalogue |
| \(\kappa_p\) | scale separating relative from common price movement |
| \(D_{jx},M_{jx}\) | display and mailer indicators |
| \(w_j^{\mathrm{dsp}},w_j^{\mathrm{mlr}}\) | product promotion effects |
| \(\delta_{w(x)},\mu_j\) | week and product seasonal factors |
| \(\xi_{s(x)},\zeta_j\) | store and product store factors |

Heterogeneous own-price response is

\[
g_{hj}
=\sum_{k=1}^{K_p}
\operatorname{softplus}(\gamma_{hk})
\operatorname{softplus}(\beta_{jk})\ge0.
\tag{8}
\]

Every term in the sum is nonnegative because
\(\operatorname{softplus}(u)=\log(1+e^u)>0\). Holding the other features fixed,
increasing the price deviation therefore changes utility with derivative

\[
\frac{\partial b_j(x)}{\partial d_{jx}}
=-g_{hj}\kappa_p
\quad\text{for the relative-price component}.
\tag{9}
\]

When \(\kappa_p\ge0\), the modeled own-price contribution is non-increasing. Common
catalogue-wide movement enters through \(\bar d_x\), allowing price level to affect total
basket size as well as item substitution. The symbol \(\beta_j\) in Eq. (8) is a learned
product price factor. It is unrelated to the scalar SMC bridge coordinate introduced in
Section 8.

### 2.3 Product and group interactions

Let \(\Phi\in\mathbb R^{J\times r}\) have row \(\phi_j^\top\), and define

\[
K=\Phi\Phi^\top.
\tag{10}
\]

For every vector \(v\),

\[
v^\top Kv=v^\top\Phi\Phi^\top v=\|\Phi^\top v\|^2\ge0,
\tag{11}
\]

so \(K\) is positive semidefinite. This does **not** mean that every off-diagonal
\(K_{jk}=\phi_j^\top\phi_k\) is positive: two embedding vectors may have a negative inner
product. Positive semidefiniteness constrains the interaction matrix globally; it does
not impose entrywise attraction.

The term \(-\rho_c{n_c(S)\choose2}\) counts all unordered pairs inside group \(c\),
because

\[
{n_c\choose2}=\frac{n_c(n_c-1)}2
\tag{12}
\]

is exactly the number of distinct pairs among \(n_c\) selected products. It provides a
shared group-level pair effect, while \(K_{jk}\) provides product-specific low-rank
departures. A positive or negative fitted pair score is predictive association under the
model, not by itself a causal complement or substitute effect.

### 2.4 Total-size potential and household-specific size tendency

The table

\[
\rho_0(1),\ldots,\rho_0(n_{\max})
\tag{13}
\]

can represent a nonlinear population-level basket-size shape. One coordinate of the
household taste factor is reserved so that its product loading is one for every product.
Calling that household coordinate \(\kappa_h\), its contribution to a basket is

\[
\sum_{j\in S}\kappa_h=|S|\kappa_h.
\tag{14}
\]

Combining Eqs. (3) and (14) shows that household \(h\) experiences the effective size
potential

\[
\rho_{0h}(n)=\rho_0(n)-n\kappa_h.
\tag{15}
\]

The conclusion follows by algebra: the energy contains
\(n\kappa_h-\rho_0(n)=-[\rho_0(n)-n\kappa_h]\). Thus \(\rho_0\) describes the flexible
population curve, while one scalar \(\kappa_h\) tilts that curve toward larger or smaller
baskets for a household. Because the added amount is the same for every basket of size
\(n\), it does not change the relative probability of two baskets having equal size.

### 2.5 Identifiability and gauges

A bilinear term is not uniquely parameterized. For an orthogonal matrix \(Q\),

\[
(\Phi Q)(\Phi Q)^\top=\Phi QQ^\top\Phi^\top=\Phi\Phi^\top.
\tag{16}
\]

Therefore the columns of \(\Phi\) may rotate without changing the law. Interpretation
must use rotation-invariant quantities such as \(K_{jk}\), row norms, eigenvalues, or
subspaces rather than a named embedding coordinate.

There is also a size-interaction gauge. If the constraints permit a constant \(a\), then
replacing

\[
K\leftarrow K+a\mathbf 1\mathbf 1^\top,
\qquad
\rho_0(n)\leftarrow\rho_0(n)+a{n\choose2}
\tag{17}
\]

leaves every basket energy unchanged, because the added off-diagonal pair contribution is
\(a{n\choose2}\) and the size potential subtracts the same amount. Gauge constraints and
centered factor conventions are consequently necessary for stable parameter reporting.

---

## 3. Why direct normalization is difficult

For \(J\) offered products there are \(2^J\) subsets. Even after excluding the empty set
and sizes greater than \(n_{\max}\), direct enumeration requires

\[
\sum_{n=1}^{n_{\max}}{J\choose n}
\tag{18}
\]

terms. With \(J=5{,}455\), this number is far beyond direct computation. The difficulty
is not evaluating the energy of one observed basket; it is summing the weights of every
possible alternative basket in Eq. (4).

Version-4 becomes practical because the interaction is low rank and the remaining
couplings depend on category counts and total size. The next section proves how those
structures replace subset enumeration by a low-dimensional Gaussian expectation and
polynomial dynamic programs.

---

## 4. The tractability theorem

### 4.1 Rewriting the pair interaction

Define the summed embedding of a basket

\[
m(S)=\sum_{j\in S}\phi_j.
\tag{19}
\]

Expanding its squared norm gives

\[
\begin{aligned}
\|m(S)\|^2
&=\left(\sum_{j\in S}\phi_j\right)^\top
  \left(\sum_{k\in S}\phi_k\right)\\
&=\sum_{j\in S}\|\phi_j\|^2
  +2\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k.
\end{aligned}
\tag{20}
\]

Solving Eq. (20) for the pair sum yields

\[
\sum_{j<k\in S}\phi_j^\top\phi_k
=\frac12\|m(S)\|^2-rac12\sum_{j\in S}\|\phi_j\|^2.
\tag{21}
\]

No approximation has been made.

### 4.2 Hubbard--Stratonovich identity

**Lemma 1.** If \(z\sim\mathcal N(0,I_r)\), then for every \(v\in\mathbb R^r\),

\[
\exp\left\{\frac12\|v\|^2\right\}
=\mathbb E_z[\exp\{z^\top v\}].
\tag{22}
\]

**Proof.** The standard Gaussian density is
\((2\pi)^{-r/2}\exp(-\|z\|^2/2)\). Hence

\[
\begin{aligned}
\mathbb E_z[e^{z^\top v}]
&=(2\pi)^{-r/2}\int_{\mathbb R^r}
  \exp\left\{-\frac12\|z\|^2+z^\top v\right\}\,dz.
\end{aligned}
\tag{23}
\]

Complete the square:

\[
-\frac12\|z\|^2+z^\top v
=-\frac12\|z-v\|^2+\frac12\|v\|^2.
\tag{24}
\]

Substitution into Eq. (23) separates the constant factor:

\[
\mathbb E_z[e^{z^\top v}]
=e^{\|v\|^2/2}(2\pi)^{-r/2}
\int_{\mathbb R^r}e^{-\|z-v\|^2/2}\,dz.
\tag{25}
\]

The remaining integral is one because it is the integral of the density of
\(\mathcal N(v,I_r)\). This proves Eq. (22). \(\square\)

Apply Lemma 1 with \(v=m(S)\) and combine it with Eq. (21). Define

\[
w_j(z,x)=
\exp\left\{b_j(x)-\frac12\|\phi_j\|^2+z^\top\phi_j\right\}.
\tag{26}
\]

Then the product-specific part of a basket weight becomes \(\prod_{j\in S}w_j(z,x)\)
inside the Gaussian expectation. Conditional on \(z\), the product-specific pair coupling
has become additive in selected products.

### 4.3 Elementary symmetric polynomials

For numbers \(w_1,\ldots,w_m\), the elementary symmetric polynomial of degree \(k\) is

\[
e_k(w_1,\ldots,w_m)
=\sum_{1\le i_1<\cdots<i_k\le m}w_{i_1}\cdots w_{i_k}.
\tag{27}
\]

It is precisely the total weight of choosing \(k\) distinct products. It may be computed
without enumerating \({m\choose k}\) subsets.

**Lemma 2.** Let \(e_k^{(q)}\) denote the degree-\(k\) polynomial after processing the
first \(q\) products. Then

\[
e_k^{(q)}=e_k^{(q-1)}+w_qe_{k-1}^{(q-1)},
\quad e_0^{(q)}=1,
\tag{28}
\]

with impossible degrees equal to zero.

**Proof.** Every size-\(k\) subset of the first \(q\) products belongs to exactly one of
two disjoint cases. It either excludes product \(q\), contributing to
\(e_k^{(q-1)}\), or includes product \(q\), leaving a size-\(k-1\) subset among the first
\(q-1\) products and contributing \(w_qe_{k-1}^{(q-1)}\). Adding the two cases gives
Eq. (28). \(\square\)

### 4.4 Affinity-group and total-size polynomials

For affinity group \(c\), let \(\mathcal A_{xc}\) be its offered products and define

\[
G_c(u;z,x)
=\sum_{k=0}^{\min(|\mathcal A_{xc}|,n_{\max})}
e^{-\rho_c{k\choose2}}
e_k(\{w_j(z,x):j\in\mathcal A_{xc}\})u^k.
\tag{29}
\]

The coefficient of \(u^k\) is the total conditional weight of selecting exactly \(k\)
products from group \(c\), including the group-count energy. Multiplying all group
polynomials performs the sum over every allocation of total size among groups:

\[
\prod_cG_c(u;z,x)=\sum_{n=0}^{n_{\max}}A_n(z,x)u^n,
\tag{30}
\]

where multiplication is truncated above \(n_{\max}\). The coefficient \(A_n\) is thus
the total conditional weight of every basket of size \(n\) before applying \(\rho_0(n)\).

### 4.5 Main normalizer theorem

**Theorem 1.** The Version-4 nonempty-basket normalizer is

\[
\boxed{
Z_{+,\Theta}(x)
=\mathbb E_{z\sim\mathcal N(0,I_r)}
\left[
\sum_{n=1}^{n_{\max}}e^{-\rho_0(n)}A_n(z,x)
\right].}
\tag{31}
\]

For a fixed \(z\), its complete discrete basket sum is calculated exactly by the
recursions in Eqs. (28)--(30).

**Proof.** Start with Eq. (4), insert Eq. (21), and apply Lemma 1. All summands are
nonnegative, so the finite basket sum and Gaussian expectation may be interchanged:

\[
\begin{aligned}
Z_+(x)
&=\mathbb E_z\sum_S
e^{-\rho_0(|S|)}
e^{-\sum_c\rho_c{n_c(S)\choose2}}
\prod_{j\in S}w_j(z,x).
\end{aligned}
\tag{32}
\]

Within each group, Lemma 2 sums products having each possible group count; this gives
Eq. (29). Multiplication of the group polynomials sums over all compatible group-count
allocations and yields \(A_n\) in Eq. (30). Finally, multiplying size \(n\) by
\(e^{-\rho_0(n)}\) and summing \(n=1,\ldots,n_{\max}\) gives Eq. (31). Every discrete
choice has been summed exactly once. \(\square\)

### 4.6 What the theorem does and does not make exact

If \(\Phi=0\), then every \(w_j\) is independent of \(z\). The expectation in Eq. (31)
is unnecessary, and the entire normalizer is exact after one polynomial dynamic program.

If \(\Phi\ne0\), the dynamic program is still exact for each fixed \(z\), but the
remaining \(r\)-dimensional Gaussian expectation must be integrated or sampled. The
theorem has therefore replaced exponential dependence on \(J\) with polynomial discrete
work and a numerical problem whose dimension is the interaction rank \(r\), not the
catalogue size.

### 4.7 Derivatives of the normalizer are model moments

Suppose a scalar parameter \(\eta\) enters the energy and the finite-support derivative
may be interchanged with the sum. Differentiating gives

\[
\begin{aligned}
\frac{\partial\log Z_\Theta(x)}{\partial\eta}
&=\frac{1}{Z_\Theta(x)}
  \sum_S e^{E_\Theta(S,x)}
  \frac{\partial E_\Theta(S,x)}{\partial\eta}\\
&=\mathbb E_\Theta\left[
  \frac{\partial E_\Theta(S,x)}{\partial\eta}\mid x
  \right].
\end{aligned}
\tag{33}
\]

For an item intercept \(\lambda_j\),
\(\partial E/\partial\lambda_j=\mathbf1\{j\in S\}\), so

\[
\frac{\partial\log Z}{\partial\lambda_j}=P_\Theta(j\in S\mid x).
\tag{34}
\]

This proves that incidence probabilities are derivatives of \(\log Z\). MRR is not such
a derivative; it is a rank metric computed from held-out hidden-item predictions in
Section 9.

For one observation, the log-likelihood score is

\[
\frac{\partial\ell}{\partial\eta}
=\frac{\partial E(S^{\mathrm{obs}},x)}{\partial\eta}
-\mathbb E_\Theta\left[
\frac{\partial E(S,x)}{\partial\eta}\mid x\right].
\tag{35}
\]

Thus maximum likelihood matches observed sufficient statistics to their model
expectations. This identity explains every later gradient calculation.

---

## 5. Why estimation is staged without changing the model

Theorem 1 makes likelihood evaluation possible, but it does not imply that all parameters
should be optimized from an arbitrary joint initialization. The selected stages address
one derivative degeneracy, one identification problem, and one large computational
multiplier.

### 5.1 The factor gradient cannot leave the exact additive origin

The interaction likelihood depends on \(\Phi\) through \(K=\Phi\Phi^\top\). Let
\(L(K)\) be any differentiable objective expressed in the Gram matrix.

**Proposition 2.** If \(K=\Phi\Phi^\top\), then

\[
\nabla_\Phi L=(\nabla_KL+\nabla_KL^\top)\Phi.
\tag{36}
\]

Consequently \(\nabla_\Phi L=0\) at \(\Phi=0\), even when a positive-semidefinite
direction in \(K\) would improve \(L\).

**Proof.** A perturbation \(d\Phi\) produces

\[
dK=d\Phi\,\Phi^\top+\Phi\,d\Phi^\top.
\tag{37}
\]

Using the Frobenius inner product,

\[
\begin{aligned}
dL
&=\langle\nabla_KL,dK\rangle\\
&=\langle\nabla_KL,d\Phi\,\Phi^\top\rangle
 +\langle\nabla_KL,\Phi\,d\Phi^\top\rangle\\
&=\langle(\nabla_KL+\nabla_KL^\top)\Phi,d\Phi\rangle.
\end{aligned}
\tag{38}
\]

The coefficient of \(d\Phi\) is Eq. (36). Substituting \(\Phi=0\) gives zero.
\(\square\)

This is why the pipeline does not simply fit the additive model with \(\Phi=0\) and then
ask an ordinary factor gradient to discover interactions. A random nonzero initialization
avoids the exact zero, but it chooses an arbitrary rotation and pays the interaction
normalizer cost before the large first-order effects have been estimated.

### 5.2 The natural interaction score identifies useful directions

Let \(X(S)\) be the symmetric off-diagonal co-incidence matrix of a basket. At the
additive parent, the natural score for an interaction perturbation is the difference

\[
R=mathbb E_{\mathrm{data}}[X(S)]
-\mathbb E_{p_{\mathrm{add}}}[X(S)].
\tag{39}
\]

For a rank-one positive-semidefinite perturbation \(K(\epsilon)=\epsilon vv^\top\), the
directional derivative at zero is proportional to

\[
\left.\frac{dL}{d\epsilon}\right|_{\epsilon=0}=v^\top Rv.
\tag{40}
\]

If \(Rv=\lambda v\) with \(\lambda>0\), then \(v^\top Rv=\lambda\|v\|^2>0\). A leading
positive eigenvector is therefore a locally improving PSD direction. The pipeline repeats
the eigenspace calculation on two training halves and selects the largest candidate rank
whose subspace is stable. This guards against treating a sample-specific pair residual as
a persistent interaction direction.

The full \(5{,}455\times5{,}455\) residual matrix need not be stored. For a sparse basket
incidence vector \(s\),

\[
X(S)v=s(s^\top v)-s\odot v.
\tag{41}
\]

The first term forms the basket sum \(s^\top v\); the second removes diagonal
self-pairs. Accumulating such matrix-vector products permits a sparse leading-eigenvalue
method. The cost follows observed basket lines and requested eigenvectors rather than a
dense \(J^2\) matrix.

### 5.3 Fixed-basis natural parameters remove rotational optimization

Let \(U\in\mathbb R^{J\times r}\) contain the selected orthonormal directions. The
interaction is written

\[
K=UCU^\top,
\qquad C\succeq0.
\tag{42}
\]

The basis \(U\) fixes the interaction subspace, while the small \(r\times r\) matrix
\(C\) learns its strength and mixtures. For row \(u_j^\top\) of \(U\), define

\[
F_U(S)=\frac12\left[
\left(\sum_{j\in S}u_j\right)
\left(\sum_{j\in S}u_j\right)^\top
-\sum_{j\in S}u_ju_j^\top
\right].
\tag{43}
\]

Using Eq. (20) with \(u_j\) in place of \(\phi_j\),

\[
\operatorname{tr}\{CF_U(S)\}
=\sum_{j<k\in S}u_j^\top Cu_k
=\sum_{j<k\in S}K_{jk}.
\tag{44}
\]

The fitted matrix is finally factored as \(C=LL^\top\) and stored through
\(\Phi=UL\). This storage factor is not a second statistical fit.

### 5.4 Computational reason for fitting the additive parent first

With \(\Phi=0\), the Gaussian variable disappears from Theorem 1. Every additive update
uses one exact full-catalogue dynamic program per context. With active rank \(r\), a
deterministic likelihood audit repeats related work at many \(r\)-dimensional quadrature
nodes. It is therefore efficient to learn product popularity, household taste, price,
promotion, week, store, group counts, and the global size curve before paying the
interaction-integration multiplier.

Staging does not assert that the separate phases are unrelated. They estimate parameters
of the same Eq. (3), and every accepted descendant is scored as the complete joint model.
The selected interaction phase is a constrained residual refinement, not unrestricted
joint reoptimization of every earlier nonlinear factor. That limitation is explicit.

---

## 6. The real-data training pipeline

### 6.1 Stage A: deterministic data construction and audit

The data stage reconstructs prices, removes only declared invalid transaction lines,
forms checkout-level distinct-SKU baskets, creates the temporal split, selects the cohort
from training data, builds promotion and store panels, and independently reconstructs key
outcomes. It records the chosen price basis and source digests.

This stage is logically prior to fitting. A perfect optimizer cannot repair leakage from
selecting products on test data, a price column with the wrong meaning, or a mismatch
between numerator and normalizer assortments.

### 6.2 Stage B: training-only affinity partition and ragged index

Training co-purchases define a fixed product partition used by \(n_c(S)\) and the group
polynomials in Eq. (29). The partition is neither a learned household representation nor
a claim that the groups are official merchandising categories.

The ragged index stores each context's offered catalogue and each observed basket without
padding all groups to the largest group. It makes the mathematical sets and group blocks
available to the dynamic program.

### 6.3 Stage C: fresh initialization

A fresh run creates a new initialization artifact derived from training-only moments.
One household-taste coordinate is reserved for the common product loading in Eq. (14),
other product-taste columns are centered, and \(\Phi\) is exactly zero. The artifact is
bound to the data fingerprint.

Starting from this artifact matters. Resuming an optimizer checkpoint continues the same
fit; it is not a fresh experiment. The driver validates that distinction.

### 6.4 Stage D: exact additive maximum likelihood

With \(\Phi=0\), the pipeline maximizes

\[
L_{\mathrm{add}}(\Theta)
=\sum_t\left[E_{\mathrm{add}}(S_t,x_t)
-\log Z_{+,\mathrm{add}}(x_t)\right]-\mathcal R(\Theta),
\tag{45}
\]

where \(\mathcal R\) contains the declared regularization and calibration penalties.
The normalizer is exact over all 5,455 products and sizes 1--120. The fitted blocks are
product intercepts, household/product taste, the common household size coordinate,
price factors, promotion effects, week/store factors, \(\rho_c\), and the full
\(\rho_0\) table. Recency and quantity parameters are not trained in the selected
real-data path.

Validation controls learning-rate reductions and the convergence gate. The best and
latest checkpoints must belong to the same fit, and a full-profile stage may not be
resurrected from a checkpoint that merely reached its update ceiling without satisfying
the convergence contract.

### 6.5 Stage E: stable-rank audit

The spectral calculation in Section 5.2 evaluates ranks from the requested maximum
downward. If rank 8 is unstable, the method tests lower ranks; rank 5 is not a universal
fallback selected without evidence. The output records the parent checkpoint, selected
rank, split-half evidence, basis digest, and data fingerprint.

Rank \(r\) is model capacity. It is distinct from a Smolyak level \(q\), which controls
the accuracy and cost of a numerical integral for an already fixed rank and checkpoint.

### 6.6 Stage F: canonical size-stratified interaction estimation

Let \(p_0(S\mid x)\) be the converged additive parent. In the selected basis, the child
increment is

\[
h_\eta(S)=\operatorname{tr}\{CF_U(S)\}-q(|S|),
\tag{46}
\]

where \(C\) is constrained PSD and \(q(n)\) is a smooth correction written into the
existing \(\rho_0(n)\) table. The implementation uses 12 size knots with piecewise-linear
interpolation. This is a regularized parameterization of the same Version-4 size
potential, not a new size model.

The already fitted \(\rho_c\) is frozen in the canonical estimator. A bounded experiment
that freed all category corrections allowed sparse coordinates to reach their bounds and
failed cross-validation. Freezing them is therefore a predeclared estimator decision
supported by that test; the underlying Eq. (3) still contains and uses \(\rho_c\).

Ordinary parent sampling can miss a rare size region. A finite bank may contain no large
basket even when a proposed interaction makes such baskets important. The canonical
method partitions sizes into

\[
\mathcal B_1=1{:}4,\quad
\mathcal B_2=5{:}10,\quad
\mathcal B_3=11{:}20,\quad
\mathcal B_4=21{:}40,\quad
\mathcal B_5=41{:}59,\quad
\mathcal B_6=60{:}80,\quad
\mathcal B_7=81{:}120.
\tag{47}
\]

The boundary at 60 matches the production tail audit. The draw allocation
\((16,16,12,8,5,4,3)\) uses 64 baskets per context while deliberately representing each
region.

#### Proposition 3: exact stratified partition-ratio identity

Put \(p_{0\ell}(x)=P_0(N\in\mathcal B_\ell\mid x)\). Then

\[
\frac{Z_{\eta,+}(x)}{Z_{0,+}(x)}
=\sum_{\ell=1}^{7}p_{0\ell}(x)
\mathbb E_0\left[e^{h_\eta(S)}
\mid N\in\mathcal B_\ell,x\right].
\tag{48}
\]

**Proof.** Direct substitution of the parent probability gives

\[
\mathbb E_0[e^{h_\eta(S)}\mid x]
=\sum_S\frac{e^{E_0(S,x)}}{Z_{0,+}(x)}e^{h_\eta(S)}
=\frac{Z_{\eta,+}(x)}{Z_{0,+}(x)}.
\tag{49}
\]

The seven bands form a disjoint partition of every supported size. Applying the law of
total expectation to the left side of Eq. (49) gives the right side of Eq. (48).
\(\square\)

For every context and band, draw

\[
S_{\ell d}\sim p_0(S\mid N\in\mathcal B_\ell,x),
\qquad d=1,\ldots,D_\ell,
\tag{50}
\]

using the exact parent size law followed by exact reverse category/product sampling. The
ratio estimator is

\[
\widehat R_x(\eta)
=\sum_{\ell=1}^{7}\frac{p_{0\ell}(x)}{D_\ell}
\sum_{d=1}^{D_\ell}e^{h_\eta(S_{\ell d})}.
\tag{51}
\]

#### Proposition 4: unbiasedness on the ratio scale

For fixed positive allocations,

\[
\mathbb E[\widehat R_x(\eta)]
=\frac{Z_{\eta,+}(x)}{Z_{0,+}(x)}.
\tag{52}
\]

**Proof.** Each sample mean in Eq. (51) is unbiased for its within-band conditional
expectation. Multiplication by the exact band probability and summation gives Eq. (48).
\(\square\)

This result is deliberately stated on the ratio scale. Because logarithm is concave,
Jensen's inequality gives

\[
\mathbb E[\log\widehat R_x]
\le\log\mathbb E[\widehat R_x]
=\log\frac{Z_{\eta,+}}{Z_{0,+}}.
\tag{53}
\]

The finite-bank log likelihood is consequently optimistic. Independent deterministic
quadrature, rather than the training bank, decides final likelihood acceptance.

#### Proposition 5: concavity of the fixed-bank objective

For observed context-basket pairs \((x_m,S_m^{\mathrm{obs}})\), define

\[
\widehat G(\eta)=\frac1M\sum_{m=1}^{M}
\left[h_\eta(S_m^{\mathrm{obs}})-\log\widehat R_{x_m}(\eta)\right].
\tag{54}
\]

With the conditional draws frozen, \(\widehat G\) is concave in the natural parameters
\((C,q)\). It remains concave after nonnegative quadratic ridge and second-difference
penalties.

**Proof.** The observed term is affine in \((C,q)\). Equation (51) is a positive weighted
sum of exponentials of affine functions. Its logarithm is a log-sum-exp and is convex.
An affine function minus a convex function is concave. A negative quadratic form with a
positive-semidefinite penalty matrix is concave. Finally, the spectral interval
\(0\preceq C\preceq\sigma_{\max}^2I\), coefficient boxes, and a fixed size-potential
gauge form a convex feasible set. \(\square\)

The fixed draw bank therefore creates a deterministic global sampled target. Bounded
L-BFGS updates the size coefficients; projected Armijo ascent updates the small PSD matrix
\(C\). Both optimize Eq. (54). Common draws are cached, so interruption does not require
rebuilding the statistical experiment.

#### Variance allocation and within-band diagnostics

Let

\[
\sigma_\ell^2(x;\eta)=
\operatorname{Var}_0[e^{h_\eta(S)}\mid N\in\mathcal B_\ell,x].
\tag{55}
\]

Independence across bands implies

\[
\operatorname{Var}(\widehat R_x)
=\sum_{\ell=1}^{7}
\frac{p_{0\ell}(x)^2\sigma_\ell(x;\eta)^2}{D_\ell}.
\tag{56}
\]

To derive the ideal allocation under cost \(c_\ell\) per draw, minimize Eq. (56) subject
to \(\sum_\ell c_\ell D_\ell=C\). Differentiating the Lagrangian with respect to
\(D_\ell\) gives

\[
-\frac{p_{0\ell}^2\sigma_\ell^2}{D_\ell^2}+\lambda c_\ell=0,
\tag{57}
\]

and therefore

\[
D_\ell\propto\frac{p_{0\ell}\sigma_\ell}{\sqrt{c_\ell}}.
\tag{58}
\]

The current certified pipeline uses the fixed conservative allocation in Eq. (47), not
an adaptive pilot. Every band has a positive floor. Effective sample size is reported
within bands; one aggregate ESS would confound deliberately unequal band probability with
undesired concentration among compositions inside a band.

### 6.7 Stage G: residual household-size recalibration

The additive stage already learns a base common household coordinate
\(\kappa_h^{(0)}\). Interactions and the global size correction alter the parent size law,
so the old score condition need not remain zero. Let

\[
q_x(n)=P_{\mathrm{post-int}}(N=n\mid x).
\tag{59}
\]

The later stage tests only a residual increment \(\delta_h\):

\[
q_{x,\delta_h}(n)
=\frac{q_x(n)e^{n\delta_h}}
{\sum_{m=1}^{n_{\max}}q_x(m)e^{m\delta_h}}.
\tag{60}
\]

For household \(h\), the penalized objective is

\[
L_h(\delta_h)=
\sum_{t:h(t)=h}\left[
N_t\delta_h-
\log\sum_mq_{x_t}(m)e^{m\delta_h}
\right]-\frac\lambda2\delta_h^2.
\tag{61}
\]

Differentiating once gives

\[
L_h'(\delta_h)=
\sum_{t:h(t)=h}
\left[N_t-\mathbb E_{\delta_h}(N\mid x_t)\right]-\lambda\delta_h.
\tag{62}
\]

Differentiating the tilted expectation gives its variance, so

\[
L_h''(\delta_h)=
-\sum_{t:h(t)=h}\operatorname{Var}_{\delta_h}(N\mid x_t)-\lambda<0
\tag{63}
\]

when \(\lambda>0\). Hence each household problem is strictly concave and has at most one
maximizer. The ridge is selected by swapped chronological household-day folds. The
increment is accepted only if its simultaneous cross-fitted lower bound is positive;
otherwise \(\delta_h=0\) preserves the parent. The final coordinate is

\[
\kappa_h^{\mathrm{final}}=\kappa_h^{(0)}+\delta_h.
\tag{64}
\]

Thus household size is not fitted twice to the same target. The first value belongs to
the additive joint optimum; the second is a one-dimensional residual score repair after
the law has changed.

### 6.8 Stage H: independent likelihood and production certification

Interaction likelihood is evaluated through Theorem 1 with deterministic Smolyak
Gauss--Hermite quadrature. For active rank \(r\), the pipeline uses:

- \(q=r+1\) for a broad population screen;
- \(q=r+2\) for the target likelihood and high-risk confirmation; and
- \(q=r+3\) on a smaller audit panel to estimate adjacent-rule discrepancy.

Changing \(q\) changes numerical integration accuracy and node count. It does not change
rank, support, parameters, or training progress. Agreement between adjacent rules is an
error diagnostic, not a formal theorem that both rules cannot miss the same feature.

Validation decides whether a candidate is accepted. Test reports its performance and is
not used to tune it. The population screen evaluates aggregate and localized probability
of \(N\ge60\), escalates risky contexts to the more accurate rule, and fails closed on
lineage, numerical, likelihood, or tail violations.

---

## 7. Exact conditional inference from the fitted law

This section derives retailer-facing probabilities before discussing randomized basket
generation. The derivations use the fitted Eq. (4), so no auxiliary prediction model is
introduced.

### 7.1 Recommendation by hiding one item

Let \(R\) be the revealed part of a basket and suppose the evaluation protocol states
that exactly one item is missing. Candidate \(j\notin R\) completes the basket as
\(R\cup\{j\}\). By the definition of conditional probability,

\[
P(j\text{ is missing}\mid R,x,\text{one missing})
=\frac{P(R\cup\{j\}\mid x)}
{\sum_{k\notin R}P(R\cup\{k\}\mid x)}.
\tag{65}
\]

Substitute Eq. (4):

\[
\begin{aligned}
P(j\mid R,x,\text{one missing})
&=\frac{e^{E(R\cup\{j\},x)}/Z_+(x)}
{\sum_{k\notin R}e^{E(R\cup\{k\},x)}/Z_+(x)}\\
&=\frac{e^{E(R\cup\{j\},x)}}
{\sum_{k\notin R}e^{E(R\cup\{k\},x)}}.
\end{aligned}
\tag{66}
\]

The same normalizer appears in every term and cancels. Every candidate completion also
has size \(|R|+1\), so the common \(\rho_0(|R|+1)\) term cancels. Terms independent of the
candidate may be subtracted from every exponent without changing the softmax. The
candidate-dependent score is

\[
\begin{aligned}
s_j(R,x)={}&b_j(x)
+\sum_{k\in R}\phi_j^\top\phi_k\\
&-\rho_{c(j)}n_{c(j)}(R),
\end{aligned}
\tag{67}
\]

where the final term is the increment

\[
-\rho_c\left[{n_c(R)+1\choose2}-{n_c(R)\choose2}\right]
=-\rho_c n_c(R).
\tag{68}
\]

Therefore

\[
\boxed{
P(j\mid R,x,\text{one missing})
=\frac{e^{s_j(R,x)}}{\sum_{k\notin R}e^{s_k(R,x)}}.}
\tag{69}
\]

This is why add-one recommendation requires neither Smolyak quadrature nor a separately
trained recommendation loss.

The evaluation chooses a held-out test basket with at least two products, hides one item
using a fixed seed, excludes the still-revealed products from the candidate set, and ranks
every other supported product. If the hidden item has rank \(r_t\), its reciprocal rank is
\(1/r_t\). Over \(T\) cases,

\[
\operatorname{MRR}=\frac1T\sum_{t=1}^T\frac1{r_t},
\qquad
\operatorname{MRR@}K=\frac1T\sum_{t=1}^T
\frac{\mathbf1\{r_t\le K\}}{r_t}.
\tag{70}
\]

Recall@\(K\) is \(T^{-1}\sum_t\mathbf1\{r_t\le K\}\). Likelihood and MRR are related
through the same law but are not the same objective: likelihood scores the entire basket
including size probability, whereas MRR tests the relative ordering of one item among
many same-size completions. Better average likelihood does not mathematically force
higher finite-sample MRR.

### 7.2 Incidence, size, and other expectations

Equation (34) gives a product's marginal incidence. More generally, if a coefficient
\(a\) multiplies a statistic \(T(S)\) in the energy, then

\[
\frac{\partial\log Z}{\partial a}=\mathbb E[T(S)\mid x].
\tag{71}
\]

For example, derivatives with respect to pair natural parameters yield pair moments, and
derivatives with respect to size-potential coordinates yield signed size probabilities.
These identities support model diagnostics, but calculating them with active interactions
still requires the normalizer machinery or a valid sampler.

### 7.3 Price counterfactuals

Let \(E_0\) be the fitted energy under factual prices and \(E_1\) the energy after a
declared price intervention. For any basket statistic \(g(S)\),

\[
\begin{aligned}
\mathbb E_1[g(S)]
&=\sum_Sg(S)\frac{e^{E_1(S)}}{Z_1}\\
&=\frac{\sum_Sg(S)e^{E_0(S)}e^{E_1(S)-E_0(S)}}
{\sum_Se^{E_0(S)}e^{E_1(S)-E_0(S)}}\\
&=\frac{\mathbb E_0[g(S)e^{\Delta E(S)}]}
{\mathbb E_0[e^{\Delta E(S)}]},
\end{aligned}
\tag{72}
\]

where \(\Delta E(S)=E_1(S)-E_0(S)\). The second equality multiplies numerator and
denominator by \(1/Z_0\), which turns both sums into factual expectations. Equation (72)
is an exact identity. Its self-normalized Monte Carlo implementation is approximate and
must report effective sample size; a large intervention may require fresh sampling from a
closer proposal.

The output is a structural counterfactual under the fitted observational law. It is not
automatically a causal estimate of a randomized discount because historical price and
promotion assignment may be confounded.

### 7.4 Segments and bundle candidates

Households are clustered on rotation-invariant fitted summaries of taste, price response,
and size tendency. Clustering summarizes heterogeneity; it does not replace the
household-level model. Segment descriptions are then computed on held-out outcomes.

For products \(j,k\), the Gram contribution \(K_{jk}=\phi_j^\top\phi_k\) and the relevant
group-count increment provide a fitted pair score. Candidate pairs are chosen without test
outcomes, then compared with frequency/support-matched controls on held-out co-incidence.
This supports the statement that a pair is predictively associated beyond modeled
first-order effects. It does not prove that promoting one product causally increases
demand for the other.

---

## 8. Basket generation from the Version-4 law

Recommendation conditions on an almost complete basket. Generation is a different task:
draw an entire random \(S\) from Eq. (4). It follows directly from the joint law and uses
the fitted interaction embedding. The sampling bridge below is a numerical method for
drawing from that law, not a post-generation correction or an additional model.

### 8.1 The augmented joint distribution

Split the final energy into

\[
E(S,x)=E_0(S,x)+V_\Phi(S),
\qquad
V_\Phi(S)=\sum_{j<k\in S}\phi_j^\top\phi_k,
\tag{73}
\]

where \(E_0\) contains every fitted term except the Gram interaction. Define

\[
p(S,z\mid x)\propto
\varphi_r(z)
e^{-\rho_0(|S|)-\sum_c\rho_c{n_c(S)\choose2}}
\prod_{j\in S}w_j(z,x),
\tag{74}
\]

with \(w_j\) from Eq. (26) and \(\varphi_r\) the standard Gaussian density.

**Proposition 6.** The basket marginal of Eq. (74) is exactly the Version-4 law in
Eq. (4).

**Proof.** Integrating Eq. (74) over \(z\) applies Lemma 1 to
\(m(S)=\sum_{j\in S}\phi_j\). The resulting exponent is the pair term in Eq. (21), while
all terms in \(E_0\) remain unchanged. The normalizing constant is therefore exactly
\(Z_+(x)\), so the basket marginal is Eq. (4). \(\square\)

### 8.2 The four conditional basket steps

For a fixed latent \(z\), Theorem 1 supplies the complete conditional sampler:

1. Compute product weights \(w_j(z,x)\), group elementary-symmetric polynomials, and
   combined size coefficients \(A_n(z,x)\).
2. Draw total size \(N=n\) with probability

   \[
   P(N=n\mid z,x)=
   \frac{e^{-\rho_0(n)}A_n(z,x)}
   {\sum_{m=1}^{n_{\max}}e^{-\rho_0(m)}A_m(z,x)}.
   \tag{75}
   \]

3. Conditional on \(N=n\), reverse the group-polynomial convolutions to draw counts
   \((n_1,\ldots,n_C)\) satisfying \(\sum_cn_c=n\).
4. In every group, reverse the elementary-symmetric recursion to draw the requested
   number of actual distinct products.

Each reverse choice uses a normalized pair of contributions from the forward recursion.
For example, in Eq. (28), the probability of including product \(q\), conditional on
degree \(k\), is proportional to \(w_qe_{k-1}^{(q-1)}\), while exclusion is proportional
to \(e_k^{(q-1)}\). The two contributions sum to \(e_k^{(q)}\), so their normalized
probabilities are exact. Repeating backward samples exactly from the conditional
distribution represented by the polynomial.

These four steps already produce a basket once a valid \(z\) has been drawn. The remaining
problem is that the correct marginal law of \(z\) is not \(\mathcal N(0,I_r)\); it is the
Gaussian density multiplied by the complete polynomial integrand in Eq. (31).

### 8.3 The interaction bridge

Introduce an algorithmic coordinate \(\beta\in[0,1]\):

\[
p_\beta(S\mid x)
=\frac{\exp\{E_0(S,x)+\beta V_\Phi(S)\}}{Z_\beta(x)}.
\tag{76}
\]

At \(\beta=0\), only the Gram interaction is absent; household, price, promotion, group,
and size terms remain. This no-Gram law has an exact normalizer and sampler. At
\(\beta=1\), Eq. (76) is the complete fitted model. Intermediate values are not learned
models and do not change the final law.

The corresponding augmented density is

\[
\widetilde p_\beta(S,z\mid x)\propto
\exp\left\{
E_0(S,x)-\frac12\|z\|^2
+\sqrt\beta z^\top m(S)
-\frac\beta2\sum_{j\in S}\|\phi_j\|^2
\right\}.
\tag{77}
\]

Completing the square,

\[
-\frac12\|z\|^2+\sqrt\beta z^\top m(S)
=-\frac12\|z-\sqrt\beta m(S)\|^2
+\frac\beta2\|m(S)\|^2.
\tag{78}
\]

After integrating \(z\), the final two terms in Eqs. (77)--(78) combine through
Eq. (21) to give \(\beta V_\Phi(S)\). Hence Eq. (77) has basket marginal Eq. (76), and

\[
z\mid S,x,\beta\sim
\mathcal N(\sqrt\beta m(S),I_r).
\tag{79}
\]

Conditional on \(z\), product \(j\) has log weight

\[
\eta_j(z,\beta,x)=b_j(x)
-\frac\beta2\|\phi_j\|^2
+\sqrt\beta z^\top\phi_j.
\tag{80}
\]

Thus the same exact four conditional steps work at every bridge value.

### 8.4 Sequential Monte Carlo and the purpose of the bridge

Sequential Monte Carlo, or SMC, represents a distribution by a population of complete
candidate baskets called particles. It moves that population through

\[
0=\beta_0<\beta_1<\cdots<\beta_L=1.
\tag{81}
\]

If a particle at level \(\beta_{\ell-1}\) contains basket \(S\), its incremental weight
is

\[
W_\ell(S)=
\exp\{(\beta_\ell-\beta_{\ell-1})V_\Phi(S)\}.
\tag{82}
\]

This formula is the ratio of the two unnormalized bridge densities, because their
\(E_0\) terms cancel. One direct step from zero to one can concentrate nearly all weight
on a few baskets. Multiple overlapping steps reduce each exponent and permit particles
to move toward interaction-favored regions.

At each level the algorithm:

1. **weights** particles by Eq. (82);
2. **resamples** according to normalized weights, allocating descendants to baskets that
   are plausible at the next level; and
3. **rejuvenates** by drawing \(z\mid S\) from Eq. (79) and a new \(S\mid z\) through
   Eqs. (75) and the reverse recursions.

Weighting alone cannot create a basket absent from the population. Resampling alone
duplicates high-weight baskets and reduces diversity. Rejuvenation changes size, group
counts, and products while preserving the current target.

**Proposition 7.** The blocked rejuvenation kernel leaves
\(\widetilde p_\beta(S,z\mid x)\) invariant.

**Proof.** Suppose the current \(S\) has marginal \(p_\beta(S\mid x)\). Drawing
\(z\sim p_\beta(z\mid S,x)\) produces the joint law
\(p_\beta(S,z\mid x)\) by the definition of conditional probability. Drawing
\(S'\sim p_\beta(S\mid z,x)\) and marginalizing the old \(S\) gives

\[
\sum_Sp_\beta(S,z\mid x)p_\beta(S'\mid z,x)
=p_\beta(z\mid x)p_\beta(S'\mid z,x)
=p_\beta(S',z\mid x).
\tag{83}
\]

Therefore the output pair has the same joint law. \(\square\)

The bridge normalizer estimate is

\[
\widehat Z_1(x)=Z_0(x)
\prod_{\ell=1}^{L}
\left[\frac1P\sum_{p=1}^{P}W_\ell^{(p)}\right].
\tag{84}
\]

Under exact initialization, unbiased resampling, and invariant kernels, standard SMC
induction gives \(\mathbb E[\widehat Z_1]=Z_1\). Its logarithm is still downward biased by
Jensen's inequality. The terminal particles converge to \(p_1\) as particle count grows,
but a finite resampled population is neither exact IID output nor proof that a remote mode
was represented.

The normalized bridge diagnostic is

\[
\frac{\operatorname{ESS}_\ell}{P}
=\frac{1}{P\sum_{p=1}^{P}(\bar W_\ell^{(p)})^2}.
\tag{85}
\]

It equals one for uniform normalized weights and approaches \(1/P\) when one particle
holds almost all weight. High ESS is necessary but not sufficient because equally
weighted particles can all miss the same remote region. Generation audits must therefore
also compare size, tail, category, item, and pair summaries with held-out data and repeat
the calculation across seeds.

The scalar bridge \(\beta\) is not learned, does not shrink \(\Phi\), and does not alter
the final probability model. It only controls the numerical path used to sample from the
already fitted law.

### 8.5 Basket-size phase diagnostic

Define the non-size catalogue pressure

\[
H_x(n)=\log\mathbb E_z[A_n(z,x)].
\tag{86}
\]

Theorem 1 implies

\[
P(N=n\mid x)\propto e^{H_x(n)-\rho_0(n)}.
\tag{87}
\]

Taking the ratio of adjacent probabilities cancels the common normalizer:

\[
\log\frac{P(N=n+1\mid x)}{P(N=n\mid x)}
=H_x(n+1)-H_x(n)
-[\rho_0(n+1)-\rho_0(n)].
\tag{88}
\]

This identity separates increasing catalogue pressure from the increment in the fitted
size penalty. It explains why a small error in either increment can cause a large tail
change when two large terms nearly cancel. The diagnostic also compares the full child
with \(\Phi=0\) to isolate the Gram contribution.

---

## 9. Retail policy calculations and their limits

### 9.1 Segment-specific promotion planning

For a fixed campaign horizon, the implemented planning state records time and remaining
expected markdown budget. An action selects no offer or a segment-specific candidate
bundle and discount. The fitted counterfactual basket law supplies expected product
incidence, basket value, and markdown use. Backward dynamic programming selects a
feasible sequence of actions.

This is a structural decision aid for shortlisting actions and designing controlled
tests. The real-data law is conditional on a trip and does not contain wholesale cost,
inventory, units, store choice, visit probability, or randomized treatment assignment.
It therefore does not establish realized profit optimality.

### 9.2 A complete unconditional-demand extension

If future data contain purchase opportunities, including visits with no transaction and
non-visits, the conditional basket law can be linked rather than replaced. One coherent
factorization is

Let \(\pi_V(x)=P(V=1\mid x;\Theta,\omega)\) and let
\(\pi_B(x)=P(N\ge1\mid V=1,x;\Theta,\omega)\). A normalized joint law is

\[
P(V,S\mid x)=
\begin{cases}
1-\pi_V(x),&V=0,\ S=\varnothing,\\
\pi_V(x)[1-\pi_B(x)],&V=1,\ S=\varnothing,\\
\pi_V(x)\pi_B(x)p_\Theta(S\mid x,N\ge1),
  &V=1,\ S\ne\varnothing,\\
0,&V=0,\ S\ne\varnothing.
\end{cases}
\tag{89}
\]

where \(V\) is a visit indicator. Shared summaries from \(\Theta\), such as an inclusive
value derived from \(Z_+\), may enter the visit and purchase logits. The joint objective
then contains the log visit probability, purchase/no-purchase probability, and conditional
basket likelihood. Sharing parameters through a defined inclusive-value term makes the
arrival component economically linked to basket attractiveness while retaining a
normalized joint probability.

This is a future model requiring opportunity data. It must not be silently fitted by
treating unobserved days as no-purchase events.

### 9.3 Synthetic validation

The separate synthetic workflow uses a small product universe where every basket can be
enumerated. It generates known visits, store choices, randomized offers, baskets,
quantities, costs, and an oracle policy. Because ground truth is known, it can test
interaction recovery, counterfactual calibration, and action recovery both under correct
specification and declared misspecification. Passing a synthetic experiment verifies an
implementation and identification scenario; it does not prove that the real observational
data satisfy those assumptions.

---

## 10. How the mathematics maps to the repository

The real-data pipeline is driven by `scripts/run_pipeline.py`. It owns a seven-stage graph:

\[
\text{data}\to\text{initialize}\to\text{additive}\to\text{rank}
\to\text{interaction}\to\text{evaluation}\to\text{certification}.
\tag{90}
\]

The table below maps each conceptual step to its main implementation and output. Generated
data, artifacts, reports, and logs are intentionally excluded from Git.

| Conceptual step | Primary implementation | Principal output |
|---|---|---|
| Raw-data construction | `scripts/data/01_build_base.py`, `22_basket_data.py`, `23_promo_data.py` | `data/`, `basket_input/` panels |
| Independent data audit | `scripts/data/audit_preprocessing.py` | preprocessing manifest and data fingerprint |
| Affinity partition | `scripts/version4/build_affinity_partition.py` | affinity items and manifest |
| Ragged index | `scripts/version4/data.py` | `v3_index_affinity.npz` |
| Fresh model state | `scripts/version4/initialize_version4.py` | `artifacts/initialization.pt` |
| Exact additive fit | `scripts/version4/fit_exact_additive.py` | additive best/latest checkpoints and history |
| Stable interaction basis | `scripts/version4/build_spectral_phi_initialization.py` | basis NPZ and rank report |
| Canonical interaction fit | `scripts/version4/fit_stratified_natural_interactions.py` | `artifacts/candidate.pt` and draw-bank cache |
| Household size residual | `scripts/version4/fit_household_size_rank1.py` | `artifacts/candidate_rank1.pt` |
| Likelihood comparison | `scripts/version4/compare_rank8_parent_likelihood.py` | paired validation/test reports |
| Recommendation | `scripts/version4/eval_smolyak_rank8_mrr.py` | MRR and recall report |
| Generation/counterfactual | `audit_particle_counterfactual_generation.py` | particle audit report |
| Segment interpretation | `audit_customer_segments.py`, `audit_interaction_embeddings.py` | segment and interaction reports |
| Tail certification | `audit_population_size.py` | population-size report and caches |
| Promotion policy | `run_segment_pricing_mdp.py` | segment MDP report |

The main shared statistical objects are:

- `ragged.py`: full-catalogue ragged representation and basket model;
- `fit.py`: batching and shared contextual utilities;
- `interaction_particles.py`: exact no-Gram size law and conditional sampling helpers;
- `stratified_natural.py`: sparse sufficient statistics and stratified objective;
- `tempered_ais.py` and `tempered_block_gibbs.py`: full-model SMC generation;
- `pipeline_support.py`: split, quadrature, size-law, and inference contracts;
- `checkpoint_io.py` and `provenance.py`: lineage and capability validation; and
- `poly_degree_native.cpp`: the differentiable native polynomial kernel.

The older `fit_convex_natural_interactions.py` is retained only behind
`--interaction-estimator legacy-ordinary` for reproduction. It is not the default path.
Research scripts in the same directory do not become part of the selected pipeline merely
because they are tracked.

### 10.1 Running from scratch

From the repository root, create an isolated Python environment, install the declared
dependencies, and point the driver to the raw-data directory if it is not in the default
sibling location:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export NF_RAW_DIR=/absolute/path/to/dunnhumby_The-Complete-Journey\ CSV
mkdir -p artifacts
python scripts/run_pipeline.py --from-raw --profile full \
  2>&1 | tee artifacts/pipeline.log
```

The command creates a fresh initialization and runs every stage in Eq. (90). The pipeline
console can be watched in `artifacts/pipeline.log`; the additive optimizer also maintains
`out/v3_pipeline_additive.log`.

A software smoke run uses the same graph with reduced manifests:

```bash
python scripts/run_pipeline.py --from-raw --profile smoke \
  2>&1 | tee artifacts/pipeline_smoke.log
```

Smoke mode tests integration, not convergence or scientific performance.

### 10.2 Resuming without confusing recovery and a fresh fit

The stage suffix can be restarted with

```bash
python scripts/run_pipeline.py --profile full --start-at interaction
```

Before running, the driver verifies the exact prerequisites of the requested stage:
artifact type, data fingerprint, initialization ancestry, profile, selected rank, and
completion state. It fails rather than silently using an unrelated checkpoint.

To move a completed prefix to another machine, copy the generated `data/`,
`basket_input/`, `artifacts/`, and `out/` files required by that stage in addition to
cloning the repository. Git does not contain licensed data or trained checkpoints. If
only the Git clone is available, the pipeline must rebuild from the earliest stage whose
inputs are present.

Continuing an interrupted additive optimizer is a separate operation:

```bash
python scripts/run_pipeline.py --profile full --start-at additive \
  --resume-additive out/v3_pipeline_additive.pt
```

This restores optimizer, scheduler, and minibatch state. It is not a fresh training run
and should not be reported as one.

### 10.3 Baseline and synthetic workflows

Baselines have a separate driver so every model receives its own convergence evidence and
is scored on the identical locked test manifest:

```bash
python scripts/run_baselines.py --profile converged
```

The matched external baselines are Bernoulli, DPP, and NDPP. The exact additive parent and
multinomial are ablations; SHOPPER uses a protocol that is not identical to the matched
external comparison.

The small known-truth experiments are run with

```bash
python scripts/run_synthetic_experiment.py
```

They are independent of the licensed retail files.

---

## 11. Computational scaling

Let \(B\) be the number of contexts in a batch, \(J_x=|\mathcal A_x|\), \(C_x\) the
number of active groups, \(r\) the selected interaction rank, \(M_q(r)\) the number of
Smolyak nodes, \(D=\sum_\ell D_\ell\) the stratified draws per interaction context,
\(P\) the number of SMC particles, and \(L\) the number of bridge transitions.

| Operation | Leading work | Reason |
|---|---|---|
| Exact additive normalizer | \(O(B[J_xn_{\max}+C_xn_{\max}^2])\) | ESP recursions and truncated group convolutions |
| Spectral audit | sparse pair matrix-vector products plus leading eigensolve | Eq. (41) avoids dense \(J^2\) storage |
| Draw-bank statistics | about \(O(MD[r^2+k(S)])\) after parent DP | each basket stores small interaction and occupied-group statistics |
| Natural-parameter solve | linear in cached bank size per objective/gradient call | no new full-catalogue DP inside each optimizer step |
| Household size fit | \(O(Tn_{\max}\log(1/\epsilon))\) after caching | independent one-dimensional root solves |
| Smolyak likelihood | \(O(TM_q(r)[J_xn_{\max}+C_xn_{\max}^2])\) | one discrete DP per quadrature node |
| Add-one ranking | \(O(J_xr)\) beyond additive utilities | \(Z_+\) cancels |
| SMC generation | approximately \(O(LP)\) conditional basket recursions per context | repeated bridge transitions and particles |

The exact polynomial stage is linear in catalogue size for its within-group ESP work and
quadratic in the capped degree for group convolution. It is not exponential in products.
The interaction audit's main multiplier is \(M_q(r)\), which grows rapidly with rank and
quadrature level. This is why a stable rank 5 model can be preferable to rank 8: unused
dimensions increase numerical cost without guaranteeing information gain.

The size-stratified estimator does not increase the total 64-draw budget merely to cover
large baskets; it redistributes the budget across exact-probability bands. Fixed banks
make repeated optimizer iterations inexpensive relative to regenerating full-catalogue
samples. Reverse sampling includes an exact log-domain fallback, so a sharp conditional
law can slow a draw but does not abort the run or change the target distribution.

The certified backend is currently CPU float64 because the dominant differentiable
polynomial operation is a CPU native extension and the rank stage uses CPU sparse solvers.
CUDA and Apple MPS are detected but are not silently selected for an uncertified mixed
implementation. Machine portability means explicit capability detection and a correct
fallback, not a claim that every stage is GPU-accelerated.

---

## 12. What the current evidence establishes

The canonical full run completed on 8 September 2026 with rank five, 12,000 interaction
training contexts, 64 stratified draws per context, frozen \(\rho_c\), and 37 active
natural parameters. Its interaction cross-fit gains were 0.02013 and 0.01960
nats/basket. Minimum within-band ESS fraction was 0.3993.

On locked 4,096-trip panels, the final checkpoint improved over its exact additive parent
by

\[
0.026754\pm0.002376\quad\text{nats/basket on validation}
\tag{91}
\]

and

\[
0.031009\pm0.002645\quad\text{nats/basket on test}.
\tag{92}
\]

After charging adjacent-quadrature discrepancies, the lower 95% bounds were 0.021744 and
0.025290. The full 160,007-context population screen found no majority-tail context. Its
bias-corrected \(P(N\ge60)\) was 0.003445 with upper 95% bound 0.003570, below the
predeclared 0.003700 threshold.

Overall add-one MRR was 0.095246. The interaction-only MRR increment was
0.001165\(\pm\)0.000627 and was not statistically established under the declared paired
criterion. Generated basket-size mean and variance were 7.26 and 72.97, compared with
10.03 and 136.28 in the audit data. The size distribution improved and passed the extreme
tail safety gate, but unconditional generation calibration remains incomplete.

These results justify the canonical estimator for full-support likelihood fitting and
show a statistically supported likelihood improvement over the additive parent. They do
not establish a recommendation improvement caused by interactions, exact generative
matching of all held-out moments, causal price effects, or a production-ready profit
policy. Historical external-baseline margins must not be presented as if they were
recomputed against this exact checkpoint unless that locked comparison is rerun.

---

## 13. Logical guarantees and empirical gates

The following table prevents theoretical and empirical claims from being mixed.

| Statement | Status | Basis |
|---|---|---|
| Eq. (31) equals the Version-4 normalizer | Exact theorem | H--S identity plus ESP/group convolution proof |
| Additive normalizer is exact on declared support | Exact algorithmic consequence | \(\Phi=0\) removes Gaussian integration |
| Stratified ratio estimate is unbiased | Exact estimator statement | Fixed positive allocation and exact conditional draws |
| Log of the ratio estimate is unbiased | False at finite draw count | Jensen inequality in Eq. (53) |
| Fixed-bank natural objective is concave | Exact optimization statement | Affine minus log-sum-exp on convex constraints |
| Household residual solve is unique | Exact optimization statement | Strictly negative curvature in Eq. (63) |
| Smolyak target equals the exact Gaussian integral | Approximate numerical claim | Adjacent-rule audits estimate, but do not prove, residual error |
| Interaction likelihood gain is positive | Empirically supported for the recorded run | Paired locked panels and numerical allowance |
| Interaction improves MRR | Not established | Declared uncertainty interval includes zero |
| Generation is fully calibrated | Not established | mean/variance mismatch remains |
| Price simulations are causal | Not established | observational training and missing randomized treatment design |

---

## 14. Glossary

**Additive parent.** The Version-4 law with the Gram interaction set to zero. It retains
household, price, promotion, group-count, and size effects.

**Affinity group.** A training-only product partition used by the group-count potential
and polynomial computation. It is not a household embedding or an official category.

**Basket support.** Every declared nonempty subset of the offered catalogue with size at
most 120. Numerical rank or quadrature does not reduce this support.

**Bridge coordinate \(\beta\).** An algorithmic number between zero and one that turns on
the fitted Gram interaction during SMC. It is not learned and is unrelated to the product
price factor \(\beta_j\).

**Elementary symmetric polynomial.** The sum of products of weights over every distinct
subset of a fixed degree. It is the exact combinatorial object needed for set-valued
baskets.

**Energy.** An unnormalized log score. Probabilities arise only after exponentiation and
division by the support-wide normalizer.

**Interaction rank.** The dimension of the product Gram embedding. It controls model
capacity and the dimension of the Gaussian integral.

**MCLE.** Monte Carlo maximum likelihood: optimization of a likelihood whose normalizer
ratio is represented using a fixed Monte Carlo bank.

**MRR.** Mean reciprocal rank of a held-out item. It is a ranking metric, not a partial
derivative of \(\log Z\).

**Particle.** One complete candidate basket in SMC.

**Quadrature level \(q\).** Numerical integration fidelity for a fixed rank and model. It
does not add parameters or training iterations.

**Rank-one household size coordinate.** A common item-utility component whose contribution
to a size-\(n\) basket is \(n\kappa_h\).

**SMC.** A population sampling method that weights, resamples, and rejuvenates particles
through a sequence of bridge distributions.

**Size potential \(\rho_0\).** The fitted table that controls population-level nonlinear
basket-size preference after catalogue combinatorics and other energy terms are included.

---

## 15. Further repository references

This chapter is the canonical continuous explanation. More specialized details are in:

- `paper/THEORY.md` for the formal model contract;
- `paper/SIZE_STRATIFIED_JOINT_ESTIMATOR.md` for the canonical interaction estimator and
  full-run decision;
- `paper/INFERENCE_AND_SIMULATION.md` for expanded sampling and counterfactual analysis;
- `paper/CODEBASE_ARCHITECTURE.md` for file-level ownership, artifacts, and audit findings;
- `paper/PIPELINE.md` for the concise executable-stage specification; and
- Section 11 of this chapter for the consolidated scalability analysis.

When a numerical result and a timeless identity appear in different documents, the
identity is governed by Version-4 theory and the numerical claim is governed by the
checkpoint, manifest, estimator level, and date recorded with that result.
