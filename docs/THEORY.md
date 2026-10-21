# Theory of the Version-4 basket model

This document defines the model and derives everything the code computes from it: the exact
normaliser, the treatment of pairwise interactions, exact sampling, basket generation,
counterfactuals, the estimate of $\log Z$, and the training objectives. Every formula here is
what the code in `v4/` implements; section 11 maps formulas to functions.

## 1. Setting and notation

A **trip** $t$ is one visit of a household $h$ to a store $s$ on day $d$ (week $w$). The store
offers a set of products $\mathcal J_t$ (here the whole training catalogue, $J = 5{,}455$
products for Dunnhumby). The products are partitioned into **categories**
$c \in \{1,\dots,C\}$; $c(j)$ is the category of product $j$ and $\mathcal J_{tc}$ the offered
products of category $c$. A **basket** is a set $S \subseteq \mathcal J_t$. For a basket write

$$
n = |S|, \qquad n_c = |S \cap \mathcal J_{tc}|, \qquad k = \#\{c : n_c > 0\},
$$

for its size, its count in category $c$, and its number of categories present. The model's
**support** is

$$
\mathcal S_t = \{ S \subseteq \mathcal J_t : 1 \le n \le n_{\max},\; n_c \le R \ \forall c \},
$$

with $n_{\max}$ the largest training basket (120 for Dunnhumby) and $R \le n_{\max}$ an optional
cap on products from one category (default $R = n_{\max}$). Everything below is conditional on
the trip context $x_t$ (household, store, day, prices, promotions, purchase history).

Notation: $\sigma(a) = 1/(1+e^{-a})$, $\operatorname{sp}(a) = \log(1+e^a)$ (softplus),
$\binom{n}{2} = n(n-1)/2$, $(a)_+ = \max(a, 0)$, $N(z; m, I)$ the Gaussian density.

## 2. The model

### 2.1 Joint law

$$
p_\theta(S \mid x_t) = \frac{\exp E_\theta(S, x_t)}{Z_\theta(x_t)}, \qquad
Z_\theta(x_t) = \sum_{A \in \mathcal S_t} \exp E_\theta(A, x_t), \qquad S \in \mathcal S_t .
$$

One law gives the basket likelihood, recommendations (conditionals of one product given the
rest), generated baskets (draws from $p$), and counterfactuals (the same law at a changed
context).

### 2.2 Product utilities

For product $j$ on trip $t$ of household $h$:

$$
\begin{aligned}
b_{tj} ={}& \lambda_j + \kappa_h + \tilde\theta_h^\top \tilde\alpha_j
  && \text{popularity, household size shift, taste} \\
&- g_{hj}\big[\bar m_t + \kappa\,(\Delta_{tj} - \bar m_t)\big]
  && \text{price} \\
&- g_{hj}(\kappa_D - \kappa)\, D_{tj} - c_D\, D_{tj}
  && \text{discount split} \\
&+ w^{\text{dsp}}_j\, \mathrm{dsp}_{tj} + w^{\text{mlr}}_j\, \mathrm{mlr}_{tj}
  && \text{display, mailer} \\
&+ \mu_j^\top \delta_{w} + \zeta_j^\top \xi_{s}
  && \text{season (week of year), store} \\
&+ \psi_j^\top r_{tj} + \log a_{js w}
  && \text{recency, availability} \\
&+ \theta_h^\top u_{c(j)} + \psi^{\text{item}\,\top}_{c(j)} r^{\text{item}}_{tj}
  && \text{taste per category (M4a), purchase history}
\end{aligned}
$$

- **Taste.** $\theta_h \in \mathbb R^K$, $\alpha_j \in \mathbb R^K$ ($K = 32$). The last coordinate is
  a household size shift: $\alpha_{jK} \equiv 1$, so $\kappa_h = \theta_{hK}$ adds the same amount
  to every product. The remaining loadings $\tilde\alpha_j = \alpha_{j,1:K-1}$ are centred over
  products, so they carry composition only.
- **Price.** $\Delta_{tj}$ is the log price minus the product's training-mean log price, from the
  chain-wide weekly price (the same in every store). $\bar m_t = |\mathcal J_t|^{-1}\sum_{j \in \mathcal J_t}\Delta_{tj}$
  is the trip's mean over its whole assortment. The slopes are non-negative and low rank,
  $$g_{hj} = \textstyle\sum_{q=1}^{K_p} \operatorname{sp}(\gamma_{hq})\,\operatorname{sp}(\beta_{jq}) \ \ge 0, \qquad \kappa = \operatorname{sp}(\kappa^{\text{raw}}) > 0 .$$
  A change of all prices together moves every utility by $-g_{hj}$ (the level); $\kappa$ scales
  the response to a price *relative* to the rest of the assortment.
- **Discount split.** With $D_{tj} = \min(\log(\text{price}/\text{regular price}), 0)$, centred on the
  product's training days, $\Delta = (\Delta - D) + D$ splits each price deviation into a regular
  part and a discount part. The discount part gets its own multiplier $\kappa_D = \operatorname{sp}(\kappa_D^{\text{raw}})$
  and a slope $c_D$ common to all households and products. With $\kappa_D = \kappa$ and $c_D = 0$
  the model is exactly the one-slope model, which is where training starts.
- **Promotions.** $\mathrm{dsp}, \mathrm{mlr} \in \{0, 1\}$, per product, store and week.
- **Context factors.** $\delta_w$ ($w = (\text{WEEK\_NO}-1) \bmod 52$) and $\xi_s$ are centred
  over weeks and stores; $\theta$ is centred over households (section 9.4).
- **Recency.** $r_{tj} \in \mathbb R^4$: whether the household never bought in $j$'s subcommodity
  before day $d$, and $e^{-\tau/7}$, $e^{-\tau/\bar g}$, $\log(1+\tau)/\log 100$ of the days $\tau$
  since it last did ($\bar g$ is the subcommodity's typical gap).
  $r^{\text{item}}_{tj} \in \mathbb R^3$: bought $j$ before; $j$ was in the household's latest
  purchase of $j$'s category; $e^{-\tau_j/28}$ with $\tau_j$ the days since it last bought $j$.
  All are computed strictly before the trip's day, so they are fixed data.
- **Availability.** For canonical bundles that record it, $\log a_{jsw} = 0$ once a product is
  confirmed at a store and $\log \epsilon$ before; otherwise $0$.

Every term is a fixed function of the context, so utilities never depend on the basket.

### 2.3 Energy

$$
E(S) = \sum_{j \in S} b_{tj}
 \;+\; \mathcal P(S)
 \;-\; \sum_{c} \Big[ \rho_c \binom{n_c}{2} + f_c(n_c) \Big]
 \;+\; \sum_{c:\, n_c > 0} \iota_{tc}
 \;-\; \rho_0(\,\cdot\,)
$$

**Pair term $\mathcal P(S)$.** Products carry vectors $\phi_j \in \mathbb R^r$ ($r = 16$). Without
M2, $\mathcal P(S) = \sum_{i<j \in S} \phi_i^\top \phi_j = \tfrac12\big(\|\sum_{j\in S}\phi_j\|^2 - \sum_{j \in S}\|\phi_j\|^2\big)$.
With **M2** the vectors are tied to categories, $\phi_j = \psi_{c(j)} + \delta_j$, and the category
vector counts **once per category present** rather than once per product:

$$
s(S) = \sum_{c:\, n_c>0} \psi_c + \sum_{j \in S} \delta_j
     = \sum_{j \in S}\phi_j - \sum_c (n_c - 1)_+\, \psi_c ,
$$

$$
\mathcal P(S) = \tfrac12 \|s(S)\|^2 - \tfrac12\sum_{j\in S}\|\phi_j\|^2 + \tfrac12\sum_c (n_c-1)_+ \|\psi_c\|^2 .
$$

So two categories that go together raise the probability of the pair of categories, without
also rewarding many products from each.

**Within-category terms.** $\rho_c\binom{n_c}{2}$ is a quadratic penalty ($\rho_c > 0$, substitutes)
or reward ($\rho_c < 0$, complements) on several products from one category. **M1** adds a free
potential $f_c(n_c)$ for $n_c = 2$, $3$, and $\ge 4$ ($f_c(0) = f_c(1) = 0$).

**Size potential $\rho_0$.** In the base model it acts on the basket size, $-\rho_0(n)$.
In Version 4 (**M3**) it acts on the number of categories present, $-\rho_0(k)$. The number of
products per category is then governed by $\rho_c$ and $f_c$. $\rho_0(0) = 0$ and baskets are
non-empty.

**Inclusive-value term $\iota_{tc}$ (M4c).** With the category's inclusive value
$\mathrm{IV}_{tc}(b) = \log\sum_{j \in \mathcal J_{tc}} e^{b_{tj}}$ and $\bar b$ the utilities at
regular prices (no price deviation, discount, display or mailer),

$$
\iota_{tc} = (\lambda_c - 1)\big[\mathrm{IV}_{tc}(b) - \mathrm{IV}_{tc}(\bar b)\big], \qquad \lambda_c = \sigma(\lambda^{\text{raw}}_c) \in (0,1).
$$

A discount raises $b_{tj}$ and so the product's share *within* its category. It also raises
$\mathrm{IV}_{tc}$ by $\Delta\mathrm{IV}$, but $\iota_{tc}$ takes back $(1-\lambda_c)\Delta\mathrm{IV}$
from every basket that contains the category. To first order, the category's incidence
therefore moves by only a fraction $\lambda_c$ of what the product utility alone would give,
as in a nested logit with dissimilarity $\lambda_c$. Real Dunnhumby discounts are about 90%
within-category switching and 10% category expansion. $\iota$ depends on the context only, so it
is a per-category constant on each trip and keeps the normaliser exact.

### 2.4 Base model and Version 4

| | Base (additive) model | Version 4 |
|---|---|---|
| pair term | $\phi = 0$ | $\phi_j = \psi_{c(j)} + \delta_j$, rank 16, M2 |
| within category | $\rho_c$ | $\rho_c$ and $f_c$ (M1) |
| size potential | $\rho_0(n)$ | $\rho_0(k)$ (M3) |
| extra utilities | — | $\theta_h^\top u_c$ (M4a); $r^{\text{item}}$, $r$ (recency); $\iota$ (M4c, optional) |
| normaliser | exact dynamic program | exact given a latent $z$ (section 3) |

## 3. Removing the pair term with a Gaussian latent

For any vector $s$, $\;e^{\frac12\|s\|^2} = \mathbb E_{z \sim N(0, I_r)}\, e^{z^\top s}$. Applying this to
$\tfrac12\|s(S)\|^2$ in the energy gives

$$
e^{E(S)} = \mathbb E_{z \sim N(0,I)} \, e^{E_z(S)},
$$

$$
E_z(S) = \sum_{j \in S} u_{tj}(z) \;-\; \sum_c \Big[\rho_c\tbinom{n_c}{2} + f_c(n_c) + (n_c - 1)_+\, a_{tc}(z)\Big]
 \;+\; \sum_{c:\, n_c>0}\iota_{tc} \;-\; \rho_0(\cdot),
$$

$$
u_{tj}(z) = b_{tj} - \tfrac12\|\phi_j\|^2 + z^\top \phi_j, \qquad
a_{tc}(z) = z^\top\psi_c - \tfrac12\|\psi_c\|^2 \quad (\text{M2 only}).
$$

$E_z$ has **no pair term**: given $z$, products interact only through their category counts and
the size potential. It follows that

$$
Z = \mathbb E_{z\sim N(0,I)}\, Z_z, \qquad Z_z = \sum_{A \in \mathcal S_t} e^{E_z(A)},
$$

and that the joint law of $(S, z)$,

$$
p(S, z) \propto N(z; 0, I)\, e^{E_z(S)},
$$

has the basket law $p(S)$ as its $S$-marginal, with simple conditionals:

$$
z \mid S \sim N\big(s(S), I\big), \qquad
p(S \mid z) = e^{E_z(S)} / Z_z, \qquad
q(z) := p(z) = N(z; 0, I)\, Z_z / Z .
$$

The first holds because $N(z;0,I)\,e^{z^\top s} \propto N(z; s, I)$, and only $z^\top s(S)$ in $E_z(S)$
involves both $z$ and $S$. Indeed $E_z(S) = E(S) - \tfrac12\|s(S)\|^2 + z^\top s(S)$, so
$p(S, z) = p(S)\, N(z; s(S), I)$ and the $z$-marginal is exactly a mixture over the model's baskets:

$$
q(z) = \sum_{S} p(S)\, N\big(z; s(S), I\big).
$$

**Tempering.** For $\beta \in [0,1]$, $u_{tj} = b_{tj} - \tfrac\beta2\|\phi_j\|^2 + \sqrt\beta\, z^\top\phi_j$
(and $a_{tc}$ likewise) multiplies the pair term by $\beta$. $\beta = 0$ is the model without
interactions; it is used to start chains and to anneal them (section 6).

## 4. The exact normaliser given $z$

Given $z$ (or with $\phi = 0$), $Z_z$ is a sum over all supported baskets. That is up to
$\binom{5455}{120}$ terms, yet it is computed exactly in polynomial time.

### 4.1 Within a category: elementary symmetric polynomials

For category $c$ on trip $t$, with weights $w_j = e^{u_{tj}}$, the total weight of all $k$-subsets is
the elementary symmetric polynomial

$$
e_{c,k} = \sum_{A \subseteq \mathcal J_{tc},\ |A| = k}\ \prod_{j \in A} w_j ,
\qquad
e^{(i)}_{k} = e^{(i-1)}_{k} + w_i\, e^{(i-1)}_{k-1},
$$

computed by this recursion over the category's products ($k \le R$). The recursion has no
subtraction, so it is stable at any degree. Every term of $E_z$ other than $\sum u$ depends on the
basket only through $n_c$, so category $c$ contributes the **category polynomial**

$$
G_c(y) = \sum_{k=0}^{\min(R,\,|\mathcal J_{tc}|)} g_{c,k}\, y^k, \qquad
g_{c,k} = e_{c,k}\, \exp\!\Big(-\rho_c\tbinom{k}{2} - f_c(k) - (k-1)_+ a_{tc}(z) + \mathbb 1[k\ge 1]\,\iota_{tc}\Big),
$$

with $g_{c,0} = 1$.

### 4.2 Across categories: the size dynamic program

Categories are independent given $z$ except through the size potential. With $\rho_0$ on the
basket size (base model),

$$
Z_z = \sum_{n=1}^{n_{\max}} e^{-\rho_0(n)}\ [y^n] \prod_c G_c(y),
$$

where $[y^n]$ is the coefficient of $y^n$. The product is formed category by category, truncated
at degree $n_{\max}$: $F_0 = 1$, $F_c(y) = F_{c-1}(y)\,G_c(y) \bmod y^{n_{\max}+1}$.

With $\rho_0$ on the number of categories present (M3), the product also tracks categories,
using a bivariate polynomial per category,

$$
H_c(y, v) = 1 + v \sum_{k \ge 1} g_{c,k}\, y^k, \qquad
Z_z = \sum_{n=1}^{n_{\max}}\sum_{m=1}^{n} e^{-\rho_0(m)}\ [y^n v^m] \prod_c H_c(y, v),
$$

an $(n_{\max}+1)\times(n_{\max}+1)$ table per trip.

**Cost.** The ESPs cost $O(|\mathcal J_t| R)$ per trip. The size program costs
$O(C_t\, n_{\max} R)$ in one dimension, or $O(C_t\, n_{\max}^2 R)$ with M3, where $C_t$ is the
number of categories offered.

**Numerics.** All coefficients are kept in log space. Utilities are centred at the trip's
largest utility $M_t$ (added back as $n M_t$). Each category polynomial is divided by its
largest coefficient. The trip's polynomials are tilted by their largest per-degree slope $\tau$,
$g_{c,k} \to g_{c,k} e^{-\tau k}$, and the tilt is restored as $n\tau$. These are exact rescalings.

### 4.3 Incidences are gradients

Since $u_{tj}$ enters $E_z$ linearly with coefficient $\mathbb 1[j \in S]$,

$$
\frac{\partial \log Z_z}{\partial u_{tj}} = P_z(j \in S),
$$

the exact purchase probability of every offered product given $z$. The backward passes of both
programs are written by hand (`v4/native/`). They propagate $h_k = e_k\, \partial L/\partial e_k$,
so every multiplier is a share in $[0,1]$ and cannot overflow.

## 5. Exact sampling given $z$

Draw $S \sim p(\cdot \mid z)$ in three backward steps over the same tables (forward filtering,
backward sampling):

1. **Size** (and, with M3, the number of categories): draw $n$ with probability
   $\propto e^{-\rho_0(n)}[y^n]F_C(y)$, or $(n, m)$ with probability $\propto e^{-\rho_0(m)}[y^n v^m]\prod_c H_c$.
2. **Count per category**, for $c = C, \dots, 1$ with $n$ (and $m$) still to fill: draw $n_c = k$
   with probability $\propto [y^{n-k}]F_{c-1}(y)\, g_{c,k}$, then reduce $n$ by $k$ (and $m$ by
   $\mathbb 1[k>0]$).
3. **Products within a category**: to choose $k$ of the category's products with probability
   $\propto \prod_{j \in A} w_j$, go through the products from last to first and include product $i$
   with probability $w_i\, e^{(i-1)}_{k'-1} / e^{(i)}_{k'}$, where $k'$ is the number still needed.

Each step is an exact conditional, so the draw is exact. The C++ sampler (`sample_slots_cpp`)
does this for a whole batch with one random seed per trip.

## 6. Generation: drawing baskets from the full model

### 6.1 The block Gibbs chain

The joint law $p(S, z)$ of section 3 has exact conditionals, so alternating them is a Gibbs
sampler whose stationary distribution is $p(S, z)$. Its $S$ draws are therefore draws from the
basket law:

$$
z^{(i)} \sim N\big(s(S^{(i-1)}),\, I\big), \qquad S^{(i)} \sim p\big(\cdot \mid z^{(i)}\big).
$$

One **sweep** is these two steps (`gibbs_sweeps_cpp`). The chain is irreducible on the support
because, given any $z$, every basket in $\mathcal S_t$ has positive probability.

### 6.2 Generating a basket for a trip

To generate the basket of a trip (`sampling.draw_from_model`):

1. build the trip's context $x_t$ (prices, promotions, household, history) and its utilities;
2. draw a starting basket exactly from the model *without* interactions ($\beta = 0$, section 5);
3. run `burn` Gibbs sweeps at $\beta = 1$ (200 in evaluation);
4. return the final basket $S$.

A **counterfactual basket** (for example after a price change) is generated the same way from
the changed context. The **expected** purchases of each product are better estimated without
sampling noise (Rao–Blackwellisation). Since $P(j \in S) = \mathbb E_{q(z)} P_z(j \in S)$
(section 3), the chain's $z$ draws are averaged against the exact incidences of section 4.3:

$$
\widehat P(j \in S) = \frac1K \sum_{k=1}^K P_{z_k}(j \in S), \qquad z_k = s(S_k) + \varepsilon_k,\ \varepsilon_k \sim N(0, I).
$$

### 6.3 Realism check

For every test trip, one basket is generated per seed. A gradient-boosted classifier is then
trained to tell real from generated baskets, using two feature sets: product indicators,
category counts and size; or category presence only. Folds are grouped by household. The
held-out AUC measures realism: 0.5 means indistinguishable.

## 7. Counterfactuals

A counterfactual is the **same law at a changed context** $x'$: changed prices, display or
mailer. Changing product $k$'s log price by $\Delta$ changes $\Delta_{tk}$, and the trip mean
$\bar m_t$ moves by $\Delta/|\mathcal J_t|$. With a discount panel the discount depth moves too
(`price.changed_price_context`).

### 7.1 Base model

With $\phi = 0$ the counterfactual incidences are exact: recompute $u$ at $x'$, then take
$\partial\log Z/\partial u$.

### 7.2 Version 4: reweighting the latent draws

The $z$-marginal changes with the context, $q'(z) \propto N(z;0,I)\,Z'_z$, so draws $z_k \sim q$
taken at the factual context are reweighted by $Z'_{z_k}/Z_{z_k}$ (self-normalised importance
weights):

$$
\widehat P'(j \in S) = \sum_k \omega_k\, P'_{z_k}(j\in S), \qquad
\omega_k = \frac{Z'_{z_k}/Z_{z_k}}{\sum_l Z'_{z_l}/Z_{z_l}} .
$$

Each term is exact given $z$. One set of chains serves every scenario, which keeps scenario
differences free of independent Monte Carlo noise.

### 7.3 One utility: the exact tilt

If only product $j$'s utility changes, by $\Delta$, then for every $z$

$$
\frac{Z'_z}{Z_z} = 1 - p_z + p_z e^{\Delta}, \qquad
P'_z(j \in S) = \frac{p_z e^{\Delta}}{1 - p_z + p_z e^{\Delta}}, \qquad p_z = P_z(j\in S).
$$

Inserting this into section 7.2, the reweighted marginal is the same tilt of the factual
marginal $\bar p = \mathbb E_q\, p_z$:

$$
P'(j \in S) = \frac{\bar p\, e^{\Delta}}{1 - \bar p + \bar p e^{\Delta}} .
$$

The discount and promotion check uses this tilt for "this product's display on or off" and
"only this product's price back to last week". For the price, it ignores the $1/|\mathcal J_t|$
shift of every other product's utility through $\bar m_t$, and, with M4c, the change of
$\iota_{tc}$.

### 7.4 Elasticity

The aggregate price elasticity is $\varepsilon = \mathbb E[\partial \mathbb E[n]/\partial \log p_{\text{all}}] / \mathbb E[n]$
over trips. A common log-price change $h$ shifts every utility by $-h\,g_{hj}$ ($\kappa$ cancels).
It is computed through the exact base-model program by the fourth-order central difference
$(4D(h) - D(2h))/3$, and the code checks that this agrees with the second-order difference.

## 8. Estimating $\log Z$ with interactions

For held-out likelihood, $\log p(S) = E(S) - \log Z$ needs $\log Z = \log \mathbb E_{N(0,I)} Z_z$, an
$r$-dimensional integral. It is estimated by importance sampling with a proposal $\pi(z)$ built
from the chain:

$$
\widehat Z = \frac1D \sum_{d=1}^D \frac{N(z_d; 0, I)\, Z_{z_d}}{\pi(z_d)}, \qquad
\pi(z) = (1-\epsilon)\frac1{K}\sum_{k=1}^{K} N(z; s_k, I) + \epsilon\, N(z; 0, \Sigma_{\text{wide}}).
$$

- The **centres** $s_k = s(S_k)$ are basket fields from four chains: one from $\beta = 0$ with a
  burn-in, and three started at $z = 0$ and at $z = \pm 1.5\,\sigma_1 v_1$ (the leading singular
  direction of $\Phi$), each annealed through $\beta = 0.25, 0.5, 0.75, 1$. Since $q(z)$ is
  exactly a mixture of $N(s(S), I)$ over the model's baskets (section 3), a mixture over baskets
  drawn from the chains approximates it, and $N(0,I)Z_z/\pi$ stays close to constant.
- The **defensive component** ($\epsilon = 0.1$) is $N(0, \Sigma)$, with standard deviation 1.35
  along the directions where $\Phi^\top\Phi$ has an eigenvalue above 4 and 1 elsewhere. It bounds
  the weights in the tails.
- Component choices are **stratified**, separately within each half of the draws. The two
  half-estimates are independent, and their difference is reported as a convergence check,
  together with the effective sample size.

$\log \widehat Z$ is consistent but biased low for finite $D$ (Jensen's inequality); the
half-split difference shows how far from converged it is.

## 9. Estimation

### 9.1 Base model: exact maximum likelihood

With $\phi = 0$, $\log p(S) = E(S) - \log Z$ and its gradient are exact. Minibatches $\mathcal B$ of
128 training trips minimise, with AdamW,

$$
\mathcal L = -\frac1{|\mathcal B|}\sum_{t\in\mathcal B}\log p(S_t \mid x_t)
 + \lambda_s\Big[-\sum_n \pi_n \log \bar p_n + \lambda_r\, \mathrm{KL}(\bar p\,\|\,\tilde\pi)\Big]
 + w_\varepsilon(\varepsilon - \varepsilon^\star)^2
 + \omega \!\!\sum_{(P,Q)}\! \frac{\operatorname{tr}(P^\top P\, Q^\top Q)}{\mathrm{rows}(P)\,\mathrm{rows}(Q)} .
$$

- $\bar p_n$ is the batch-average model size law and $\pi_n$ the training size law; $\tilde\pi$ is
  $\pi$ smoothed by $10^{-4}$. $\lambda_s = 1$, $\lambda_r = 10$.
- $\varepsilon$ is the batch's aggregate price elasticity (section 7.4), anchored at
  $\varepsilon^\star = -0.121$ with $w_\varepsilon = 20$. It sets the *level* of price response
  while the household × product slopes are learned.
- The pooling term runs over $(P,Q) \in \{(\mu, \delta), (\zeta, \xi), (\alpha, \theta)\}$ with
  $\omega = 1.45$. A product of two factor matrices is unidentified in scale, and this term
  shrinks the product rather than either factor.

### 9.2 Version 4: pseudo-likelihood warm-up

The conditional of one product given the rest of the basket is exact and cheap:

$$
\log P(x_j \mid S \setminus j) = \log\sigma\big(\pm d_j\big), \qquad d_j = E(S\cup j) - E(S\setminus j),
$$

with the sign $+$ if $j \in S$. Writing $o_j = n_{c(j)}(S\setminus j)$ for the other products of $j$'s category,

$$
d_j = b_{tj} + \Delta\mathcal P_j - \rho_{c}\Big[\tbinom{o_j+1}{2}-\tbinom{o_j}{2}\Big] - \big[f_c(o_j+1) - f_c(o_j)\big]
 - \mathbb 1[o_j=0]\,\big[\rho_0(k'+1) - \rho_0(k')\big] + \mathbb 1[o_j=0]\,\iota_{tc},
$$

where $k'$ is the number of categories present without $j$. With M2, $\Delta\mathcal P_j = v_j^\top s(S\setminus j) - \mathbb 1[o_j\ge1]\,\psi_c^\top\delta_j$,
with $v_j = \delta_j$ if $o_j \ge 1$ and $v_j = \phi_j$ otherwise. The pseudo-likelihood sums these
over every offered product of every trip whose basket without $j$ has a size in $[1, n_{\max}-1]$.

### 9.3 Version 4: stochastic maximum likelihood

For a training basket,

$$
\nabla_\theta \log p(S) = \nabla_\theta E(S) - \nabla_\theta \log Z, \qquad
\nabla_\theta \log Z = \mathbb E_{q(z)}\big[\nabla_\theta \log Z_z\big],
$$

because $\nabla \log \int N(z) Z_z\, dz = \int N(z)Z_z\, \nabla\log Z_z\, dz / Z$, with $z$ an
integration variable that does not depend on $\theta$. Each training trip keeps a **persistent
Gibbs chain** (started at its observed basket and carried across epochs). Each step:

1. runs 4 sweeps of the trip's chain and draws $z \sim N(s(S^{\text{chain}}), I)$;
2. takes a gradient step on $E(S_t) - \log Z_z$, with $\log Z_z$ from the exact program of section 4
   and its gradient from the hand-written backward pass.

Given $z \sim q$ the gradient is unbiased; the only approximation is the chain's distance from
stationarity. The parameters are averaged (Polyak) after the first third of the stage.

**Priors** (divided by the number of training trips $N$):

$$
\frac1{2N}\Big(\tau_\psi\|\psi\|^2 + \tau_\delta\|\delta\|^2 + \tau_\rho\|\rho_c\|^2 + \tau_f\|f\|^2
 + \tau_p\|\vartheta_p - \vartheta_p^{(0)}\|^2 + \tau_u\|u\|^2 + \tau_r(\|\psi\|^2 + \|\psi^{\text{item}}\|^2)\Big),
$$

where $\vartheta_p$ are the price parameters, centred at the base model's values
$\vartheta_p^{(0)}$. The pooling term of section 9.1 is added with weight 0.5.

**Moment matching.** The score of $\rho_0(k)$ is the model's minus the data's frequency of $k$
categories present; similarly for $\rho_c$ (category pairs) and $\lambda_j$ (product incidence).
Besides Adam, these parameters take steps along their gradients divided by the data
frequencies, a diagonal natural-gradient step. They are applied every update and once per
epoch, clipped, from the epoch's mean gradient. They let the size and category terms track the
chain quickly.

**Starting point.** The base model's $\rho_0(n)$ is converted for M3 by moving its average slope
over sizes 1–10 into $\lambda$, then using the rest as $\rho_0(k)$. $\psi_c$ starts small and random,
and $\delta_j$, $f_c$, $u_c$, $\psi^{\text{item}}$ start at 0.

**Stages.** A: pseudo-likelihood (section 9.2). B: SML with the price parameters unfrozen;
validation log-likelihood (section 8) on held-out training trips selects the averaged
parameters. C: calibration, SML with $\phi$ and the context factors fixed.

### 9.4 Gauges and constraints

These operations leave the law unchanged; they fix directions the likelihood cannot see.

- $\theta$, $\delta_w$, $\xi_s$ are centred and $\tilde\alpha$ is centred over products: a constant in
  a context factor equals a per-product constant, which belongs to $\lambda$. The forward pass
  uses the centred value with the mean detached, and the stored values are re-centred after
  every step.
- Base model: $\lambda \to \lambda - \bar\lambda$ together with $\rho_0(n) \to \rho_0(n) - n\bar\lambda$
  changes no basket energy. Version 4 (M3): $\rho_0(m) \to \rho_0(m) - \rho_0(1)$ shifts every
  non-empty basket equally.
- **Safety bounds** (these do restrict the parameters):
  $\rho_c \ge \max\big(-1.5,\, -1.5/\binom{m_c}{2}\big)$, with $m_c$ the largest number of products
  any store offers in $c$ (capped at $n_{\max}$), so no category can reward a basket by more than
  1.5 nats. With M1, also $\rho_c\binom{k}{2} + f_c(k) \ge -1.5$ for $k = 2, 3, 4$.

## 10. What is exact and what is approximate

| Quantity | Base model | Version 4 |
|---|---|---|
| $\log Z$, incidences, size law | exact | exact given $z$; over $z$ by importance sampling (section 8) |
| sampling a basket | exact | exact given $z$; Gibbs chain over $z$ (section 6) |
| training gradient | exact | unbiased given $z \sim q$; chain approximates $q$ |
| price and promotion counterfactuals | exact | exact given $z$; reweighted draws (section 7.2) |
| single-utility tilt | exact | exact in the reweighted estimator (section 7.3) |
| support | sizes $1..n_{\max}$, at most $R$ per category | same |

## 11. Where each formula lives

| Section | Code |
|---|---|
| 2.2 utilities | `v4/model.py` `BasketModel.utilities` |
| 2.3 energy, $s(S)$, $\iota$ | `BasketModel.energy`, `basket_field`, `iv_shift`, `presence_shift` |
| 3 $u(z)$, tempering | `BasketModel.slot_utilities_given_z` |
| 4.1 ESPs, category polynomials | `v4/exact.py` `row_esp`, `category_polynomials`; `v4/native/exact_dp.cpp` |
| 4.2 size programs | `log_mass_by_size`, `log_mass_by_size_and_categories`; `exact_dp.cpp`, `sampler.cpp` |
| 4.3 incidences | `exact.incidence` |
| 5, 6 sampling, Gibbs, generation | `v4/sampling.py`; `sampler.cpp` (`sample_slots_cpp`, `gibbs_sweeps_cpp`) |
| 7 counterfactuals | `v4/price.py`; `scripts/evaluate.py`; `checks/discount_promotion.py` |
| 8 importance-sampled $\log Z$ | `sampling.importance_log_z` |
| 9.1 base-model objective | `scripts/fit_additive.py` |
| 9.2–9.3 PL, SML, priors | `v4/training.py`; `scripts/fit_interactions.py` |
| 9.4 gauges, bounds | `BasketModel.project_context_gauges`, `v4/constraints.py`, `training.project` |

The tests in `tests/` check sections 4–6, 9.2 and 7.3 against enumeration of every basket of a
small world.
