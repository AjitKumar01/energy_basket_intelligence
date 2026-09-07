# The Version-4 energy-basket pipeline

> **Archived draft.** The dependency-ordered replacement is
> [`PIPELINE_TEXTBOOK_RESTRUCTURED.md`](PIPELINE_TEXTBOOK_RESTRUCTURED.md). It defines
> notation before use and presents the model, tractability result, training stages,
> numerical estimators, and retail functions in one connected sequence.

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

The Gram interaction matrix is $K=\Phi\Phi^\top\succeq0$. This constrains the global
geometry of the interaction kernel, but it does **not** make every off-diagonal pair
coefficient positive: an individual dot product $\phi_j^\top\phi_k$ may be positive,
zero, or negative. Positive values raise the relative score of baskets containing the
pair; negative values lower it. The explicit category-count term represents an additional
shared attraction or repulsion within an affinity group, depending on the sign of
$\rho_c$. Thus Version-4 is more restrictive than an arbitrary indefinite $J\times J$
pair-interaction matrix, while still allowing pair-specific effects of both signs.

### 3.2 Contextual product utility

The fitted utility is

\[
\begin{aligned}
b_j(x)
={}&\lambda_j
+\theta_h^\top\alpha_j
-a_{hj}\left[
\bar d_t+\operatorname{softplus}(\kappa_p)(d_{jt}-\bar d_t)
\right]
+w_j^{\mathrm{dsp}}D_{jt}
+w_j^{\mathrm{mlr}}M_{jt} \\
&+\mu_j^\top\delta_{w(t)}
+\zeta_j^\top\xi_{s(t)}.
\end{aligned}
\tag{4}
\]

The nonnegative household--product price coefficient is

\[
a_{hj}
=
\sum_{k=1}^{K_p}
\operatorname{softplus}(\gamma_{hk})
\operatorname{softplus}(\beta_{jk})
\ge 0.
\tag{4a}
\]

Here $d_{jt}$ is the product's log-price deviation and $\bar d_t$ is its mean over the
trip's offered assortment. The common component $\bar d_t$ moves all offered-item
utilities together and therefore affects total basket size. The centered component
$d_{jt}-\bar d_t$ primarily reallocates choice between products. The learned positive
scale $\operatorname{softplus}(\kappa_p)$ lets those two empirically different responses
have different magnitudes without permitting a positive own-price derivative.

Here:

- $\lambda_j$ is product popularity after the other effects are controlled;
- $\theta_h^\top\alpha_j$ is household-specific taste;
- $a_{hj}$ is the nonnegative household--product price-response coefficient in Eq. (4a);
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

## 4. The role of $\beta$ in the interaction bridge

This is the $\beta$ used by the basket-generation and interaction-tempering algorithm.
It is an algorithmic bridge coordinate, not a learned model parameter.

### 4.1 What “Sequential Monte Carlo” means

Sequential Monte Carlo, abbreviated SMC, represents a probability distribution by a
population of random candidates.

- **Monte Carlo** means that random draws are used to approximate probabilities and
  expectations.
- **Sequential** means that the population is passed through a sequence of intermediate
  probability distributions rather than being moved from an easy law to a difficult law
  in one step.
- A **particle** is one complete candidate basket for one customer context. It is not one
  product and it is not a model parameter.

In this application, SMC can be summarized as

\[
\boxed{
\text{exact no-Gram base baskets}
\longrightarrow
\text{slightly interaction-weighted baskets}
\longrightarrow\cdots\longrightarrow
\text{full Version-4 interaction baskets}.}
\]

The important SMC terms are:

| Term | Meaning in this basket model |
|---|---|
| Particle | One complete nonempty basket |
| Particle population | Several candidate baskets for the same context |
| Weight | How much more compatible a basket is with the next interaction level |
| Resampling | Copy high-weight baskets more often and remove low-weight baskets |
| Ancestor | The earlier particle from which a resampled particle was copied |
| Mutation or rejuvenation | Generate a new basket through an invariant blocked update |
| ESS | Effective number of meaningfully weighted particles before resampling |

SMC is not an additional predictive model. It is also not the procedure that originally
learns the interaction embedding. It is an inference algorithm used after fitting when we
need complete interaction-aware basket draws or a randomized normalizer estimate.

### 4.2 Separate the interaction from the rest of the fitted energy

Write the final fitted energy as

\[
E(S,x)=E_0(S,x)+V_\Phi(S),
\qquad
V_\Phi(S)=\sum_{j<k\in S}\phi_j^\top\phi_k.
\tag{7}
\]

$E_0$ retains everything except the Gram interaction: contextual item utilities,
household effects, prices, promotions, affinity-group counts, and the final total-size
potential. In particular, $\beta=0$ does not discard household information or replace the
model by a multinomial distribution. It switches off only $V_\Phi$.

Define the bridge law

\[
p_\beta(S\mid x)
=
\frac{\exp\{E_0(S,x)+\beta V_\Phi(S)\}}{Z_\beta(x)},
\qquad 0\le\beta\le1.
\tag{8}
\]

Its endpoints have exact meanings:

- $p_0$ is the no-Gram law. Its normalizer and exact basket sampler are available through
  the category/cardinality dynamic program.
- $p_1$ is the complete fitted Version-4 law.
- Intermediate values do not describe additional fitted models. They form a numerical
  path between the tractable endpoint and the target endpoint.

Therefore the final generated distribution is independent of how the intermediate
$\beta$ values are named. Schedule and particle count affect Monte Carlo error and cost,
not the definition of $p_1$.

### 4.3 Why $\sqrt{\beta}$ appears in the latent Gaussian representation

Let

\[
m(S)=\sum_{j\in S}\phi_j.
\]

At bridge value $\beta$, use the augmented density

\[
\widetilde p_\beta(S,z\mid x)
\propto
\exp\left\{
E_0(S,x)-\frac12\|z\|^2
+\sqrt{\beta}\,z^\top m(S)
-\frac{\beta}{2}\sum_{j\in S}\|\phi_j\|^2
\right\}.
\tag{9}
\]

Completing the square gives

\[
-\frac12\|z\|^2+\sqrt{\beta}\,z^\top m(S)
=
-\frac12\|z-\sqrt{\beta}\,m(S)\|^2
+\frac{\beta}{2}\|m(S)\|^2.
\tag{10}
\]

After integrating out $z$, the basket-dependent remainder is

\[
E_0(S,x)
+\frac{\beta}{2}\|m(S)\|^2
-\frac{\beta}{2}\sum_{j\in S}\|\phi_j\|^2
=
E_0(S,x)+\beta V_\Phi(S).
\]

Thus Eq. (9) has exactly the bridge marginal in Eq. (8), and

\[
z\mid S,x,\beta
\sim
\mathcal N\!\left(\sqrt{\beta}\,m(S),I_r\right).
\tag{11}
\]

Conditional on $z$, product $j$ has additive log weight

\[
\eta_j(z,\beta,x)
=
b_j(x)-\frac{\beta}{2}\|\phi_j\|^2
+\sqrt{\beta}\,z^\top\phi_j.
\tag{12}
\]

The same exact size, affinity-count, and reverse-ESP recursions can therefore draw a
basket conditional on $z$ at every bridge value.

### 4.4 Why a bridge is needed

A direct proposal would draw $S^{(p)}\sim p_0$ and weight it by
$\exp\{V_\Phi(S^{(p)})\}$. When interactions are strong, a few baskets can receive nearly
all the weight. The proposal effective sample size then collapses, and unrepresented
high-interaction regions cannot be recovered by simply normalizing the available weights.

The bridge breaks that difficult change into overlapping increments. For a schedule

\[
0=\beta_0<\beta_1<\cdots<\beta_L=1,
\]

the incremental weight is

\[
W_\ell^{(p)}
=
\exp\left\{
(\beta_\ell-\beta_{\ell-1})V_\Phi(S_{\ell-1}^{(p)})
\right\}.
\tag{13}
\]

Each incremental exponent is smaller than the one-step exponent. After weighting, the
particles are resampled and moved with a kernel that preserves $p_{\beta_\ell}$. This
allows the particle population to move toward interaction-favored baskets before the next
increment is applied.

The normalized diagnostic is

\[
\frac{\operatorname{ESS}_\ell}{P}
=
\frac{1}{P\sum_{p=1}^P(\overline W_\ell^{(p)})^2}.
\tag{14}
\]

A value near one means weights at that particular bridge are balanced. A small value
means severe concentration. A large ESS is necessary but not sufficient: particles can
all miss a remote mode and still have similar weights.

### 4.5 Why weighting, resampling, and rejuvenation are all necessary

Weighting alone retains the original no-Gram baskets. It changes their importance but
cannot create an interaction-favored basket that was never drawn.

Resampling converts unequal weights into an approximately equally weighted population.
For example, if baskets containing a supported complement pair receive large weights,
those baskets obtain more descendants. But resampling alone creates duplicates and can
reduce diversity, a phenomenon called particle impoverishment.

Rejuvenation repairs this problem. At the current $\beta_\ell$, the algorithm draws a
latent state from the exact conditional law in Eq. (11), then draws a new complete basket
using the conditional weights in Eq. (12) and the exact polynomial recursion. This
blocked Gibbs step may change size,
affinity-group allocation, and actual products while preserving the current bridge law.
It can therefore create nearby interaction-compatible alternatives rather than merely
copying the same basket.

The three operations have complementary roles:

\[
\underbrace{\text{weight}}_{\text{identify promising particles}}
\quad\longrightarrow\quad
\underbrace{\text{resample}}_{\text{allocate particles to them}}
\quad\longrightarrow\quad
\underbrace{\text{rejuvenate}}_{\text{restore movement and diversity}}.
\]

### 4.6 The bridge used by the pipeline

With 17 reported levels, there are 16 transitions. The default schedule is

\[
\beta_\ell
=
1-\left(1-\frac{\ell}{16}\right)^2,
\qquad \ell=0,\ldots,16.
\]

This places progressively closer levels near $\beta=1$, where the full interaction is
active. The complete procedure is:

1. Compute $Z_0(x)$ exactly and draw independent baskets exactly from $p_0$ by one
   forward dynamic program and repeated reverse draws.
2. At every transition, calculate Eq. (13) and accumulate the log average weight.
3. Resample baskets according to normalized incremental weights.
4. At each nonterminal bridge, draw
   $z\mid S,x,\beta_\ell$ from Eq. (11), then draw a complete
   $S'\mid z,x,\beta_\ell$ using the exact conditional recursion. This blocked Gibbs
   update leaves $p_{\beta_\ell}$ invariant.
5. After the terminal weighting and resampling, apply an additional $\beta=1$ blocked
   update when actual generated baskets are required. This improves diversity without
   changing the target law.

### 4.7 What $\beta$ guarantees—and what it does not

The SMC normalizer estimate is

\[
\widehat Z_1(x)
=
Z_0(x)
\prod_{\ell=1}^L
\left[
\frac1P\sum_{p=1}^P W_\ell^{(p)}
\right].
\]

With exact $p_0$ initialization, unbiased resampling, and invariant bridge kernels,
$\widehat Z_1$ is unbiased on the $Z$ scale. Its logarithm is not unbiased:

\[
\mathbb E[\log\widehat Z_1]\le\log Z_1
\]

by Jensen's inequality. More particles, better overlap, and repeated independent runs
reduce and diagnose this finite-particle error.

The terminal particles consistently approximate $p_1$ as the particle count grows, but a
finite resampled population is not IID. Shared ancestors remain possible even after the
final rejuvenation.

Most importantly, $\beta$ does not train or shrink $\Phi$, and it does not alter the
final model. It controls the numerical path used to reach the already fitted interaction
law. Changing the schedule can improve ESS or runtime; it cannot improve a poorly learned
interaction embedding. The constrained interaction MCLE fits $C$ and hence $\Phi$ from
fixed $p_0$ draws before this generation bridge is used; bridge $\beta$ is not an
additional interaction-training parameter.

### 4.8 How SMC differs from the other estimators in the pipeline

The pipeline deliberately uses different numerical tools for different questions:

| Method | Main role | Random? | Produces baskets? |
|---|---|---:|---:|
| Exact dynamic program | Normalize and sample the no-Gram law | No for normalization | Yes |
| Fixed-draw interaction MCLE | Fit the small natural interaction block | Draws fixed during solve | Uses proposal baskets |
| Smolyak quadrature | Deterministic final likelihood certification | No | No |
| SMC bridge | Generate from the full interaction law and estimate $Z_1/Z_0$ | Yes | Yes |

SMC is also different from running a single Markov chain. It maintains a population,
reweights that population across successive targets, and resamples. Its rejuvenation step
is a short Gibbs/Markov transition inside SMC, but the overall algorithm is a population
method rather than one uninterrupted chain.

The generation result should be judged using more than one number. Relevant checks are:

1. minimum bridge ESS, to detect immediate weight collapse;
2. repeated-seed stability, to detect Monte Carlo sensitivity;
3. duplicate and ancestry concentration, to detect particle impoverishment;
4. generated size, tail, category, item, and pair moments against held-out data; and
5. agreement with deterministic Smolyak likelihood on manageable audit panels.

Passing ESS alone is insufficient because an entire remote mode can be absent from every
particle.

### 4.9 The separate price parameter with the same letter

The codebase also stores a learned product price factor named model.beta. It enters

\[
a_{hj}
=
\operatorname{softplus}(\gamma_h)^\top
\operatorname{softplus}(\beta_j)\ge0
\]

inside the item price response. That $\beta_j$ is unrelated to the scalar bridge
$\beta\in[0,1]$. In this document, an unsubscripted scalar $\beta$ means the interaction
bridge; $\beta_j$ always means the product price factor.

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

Let $R$ be the observed remainder of a basket when exactly one product is hidden. For an
offered candidate $j\in\mathcal A_x\setminus R$, define
$s_j(R,x)=E(R\cup\{j\},x)$. Then

\[
\begin{aligned}
P(j\text{ completes }R\mid x,R,\text{one missing})
&=
\frac{p(R\cup\{j\}\mid x)}
{\sum_{k\in\mathcal A_x\setminus R}p(R\cup\{k\}\mid x)} \\
&=
\frac{e^{s_j(R,x)}/Z_+(x)}
{\sum_{k\in\mathcal A_x\setminus R}e^{s_k(R,x)}/Z_+(x)} \\
&=
\frac{e^{s_j(R,x)}}
{\sum_{k\in\mathcal A_x\setminus R}e^{s_k(R,x)}}.
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
starts from exact no-Gram base-basket draws, gradually turns on the interaction
contribution, reweights particles, and applies invariant rejuvenation moves. The bridge
is a sampling algorithm for Eq. (1); it is not an added model term and does not correct
the basket after generation. Section 4 defines every SMC operation and gives the bridge
invariance and normalizer guarantees.

### 9.3 Price counterfactuals

For a proposed price change, update the price features in $x$, recompute the contextual
item utility in Eq. (4), and
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

The scalar $\beta\in[0,1]$ forms the interaction bridge
$p_\beta(S\mid x)\propto\exp\{E_0(S,x)+\beta V_\Phi(S)\}$. At zero, only the Gram
interaction is absent and exact basket draws are available. At one, the law is the full
fitted Version-4 model. Intermediate values split a difficult importance-weighting step
into overlapping steps, permitting resampling and invariant blocked updates before the
next interaction increment. Thus $\beta$ controls SMC overlap, variance, and runtime; it
is not learned and does not change the final model. The subscripted $\beta_j$ appearing
elsewhere in the code is a separate product price factor.

---

## 13. Verification against the selected implementation

This document was re-audited against the selected executable pipeline on 5 September
2026. The audit checked the probability law and fitted utility in `ragged.py`, the stage
graph and gates in `run_pipeline.py`, the exact no-Gram dynamic program, the spectral-rank
builder, the constrained natural-parameter fit, the household-size residual fit, the
Smolyak likelihood comparison, the locked recommendation evaluator, and the
interaction-tempered SMC implementation.

| Textbook claim | Implementation evidence | Result |
|---|---|---|
| Stage order is data, initialization, additive, rank, interaction, evaluation, certification | `scripts/run_pipeline.py` | Verified |
| The observed-basket energy and normalizer use the same contextual item utility | `RaggedModel.b_at`, `RaggedModel.energy`, and `differentiable_logz_beta0` | Verified |
| The additive stage fixes $\Phi=0$ and fits the listed incidence blocks with an exact normalizer | `fit_exact_additive.py` | Verified |
| Rank selection tests every candidate rank from 4 through 8 and returns the largest accepted one | `build_spectral_phi_initialization.py` and `run_pipeline.rank_selection` | Verified |
| The interaction solve is deterministic and concave after fixing the parent draws | `fit_convex_natural_interactions.py` | Verified for the sampled objective |
| The late household block fits only an incremental common-utility tilt and can fall back to zero | `fit_household_size_rank1.py` | Verified |
| Validation requires a positive audited gain; test is reported without a positive-gain acceptance requirement | the two `compare_rank8_parent_likelihood.py` calls in `run_pipeline.py` | Verified |
| Reported likelihood uses $q=r+2$ and audits a smaller panel at $q=r+3$ | `run_pipeline.py` and `compare_rank8_parent_likelihood.py` | Verified |
| SMC uses the stated quadratic 17-level schedule, resamples at every bridge, mutates at nonterminal bridges, and uses a final $\beta=1$ update for generated baskets | `audit_particle_counterfactual_generation.py` and `tempered_ais.py` | Verified |
| Locked add-one recommendation does not evaluate $Z_+$ | `eval_smolyak_rank8_mrr.py` | Verified |

Four qualifications remain important.

1. The historical numerical results in the repository predate the latest artifact-lineage
   and simultaneous ridge-selection hardening. They are evidence about the same model law,
   but a fresh full execution is required to certify the current code revision.
2. The recommendation evaluator computes all of `additive_utility`,
   `structured_no_gram`, and `full_interaction`. It now reports three explicitly named
   paired contrasts. The primary Gram-only estimand compares `full_interaction` with
   `structured_no_gram`; the broader full-versus-utility contrast is retained separately
   and is never labelled interaction-only. From the archived historical means, the clean
   Gram paired point gain is \(0.0013452822\). Its historical paired standard error is
   unavailable because the legacy report did not retain the case ranks, so significance
   awaits a hardened-pipeline evaluation.
3. The current full-profile certification checks checkpoint lineage, likelihood evidence,
   numerical quadrature error, rank/interaction gates, and localized population-size
   safety. It records generation diagnostics, but segment-level generation calibration is
   not presently a hard pass/fail gate. The held-out generation-size mismatch must
   therefore remain visible in any production-readiness statement.
4. The real-data checkpoint fits conditional nonempty product incidence. Quantity,
   visit/no-purchase, inventory, cost, and causal intervention capabilities are not implied
   by the presence of corresponding experimental or synthetic code paths.

Subject to these qualifications, the mathematical derivations and the selected stage
flow in Sections 1--12 match the implementation.
