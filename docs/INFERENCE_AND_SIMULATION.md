# Inference, sampling, counterfactuals and simulation from the Version-4 law

Status: **mathematical contract of the implemented downstream pipeline**

Date: **2026-09-01**

This document starts after the Version-4 parameters have been fitted. It derives how the
same joint basket law produces samples, incidence probabilities, recommendations, price
counterfactuals, customer summaries and a restricted retailer planning environment. It
introduces no new energy term and does not replace the partition-function theory in
[THEORY.md](THEORY.md) or its numerical analysis in [ESTIMATOR.md](ESTIMATOR.md).

The central distinction is simple:

- deterministic Smolyak rules evaluate integrals for locked likelihood claims;
- interaction-tempered sequential Monte Carlo (SMC) generates positive basket particles;
- the exact category/cardinality dynamic program is shared by likelihood and conditional
  basket draws; and
- locked add-one recommendation needs neither quadrature nor SMC because its normalizer
  is common to all candidates.

The selected pipeline generates **incidence baskets**, meaning sets of distinct SKUs. A
conditional quantity model exists in the broader codebase, but the selected fresh pipeline
does not fit or certify it. The present promotion environment is consequently a
distinct-SKU planning experiment, not a profit or unit-sales simulator.

---

## 1. One fitted law, several mathematical queries

For context $x$, let $\mathcal A_x$ be the offered assortment and

\[
\Omega_x^+=\{S\subseteq\mathcal A_x:1\le |S|\le n_{\max}\},
\qquad n_{\max}=120.
\tag{S.1}
\]

The fitted Version-4 incidence law is

\[
p(S\mid x)=\frac{\exp\{E_0(S;x)+V_\Phi(S)\}}{Z_+(x)},
\qquad S\in\Omega_x^+,
\tag{S.2}
\]

where

\[
E_0(S;x)=\sum_{j\in S}b_j(x)
-\sum_c\rho_c g(n_c(S))-\rho_0(|S|),
\tag{S.3}
\]

\[
V_\Phi(S)=\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k
=\frac12\left(\left\|\sum_{j\in S}\phi_j\right\|^2
-\sum_{j\in S}\|\phi_j\|^2\right).
\tag{S.4}
\]

Here $b_j(x)$ is complete contextual utility, $n_c(S)$ is the count in category $c$, and
$g(k)=\binom{k}{2}$ on the declared category-count support. The following are different
queries of (S.2), not different models:

1. likelihood evaluates $E(S;x)-\log Z_+(x)$;
2. generation draws $S\sim p(\cdot\mid x)$;
3. marginal incidence evaluates $P(j\in S\mid x)$;
4. recommendation ranks a missing item conditional on a revealed basket;
5. counterfactual analysis changes declared components of $x$; and
6. retailer planning places those responses inside a separately declared decision model.

Likelihood needs a scalar normalizer; sampling needs positive probability transitions;
recommendation can sometimes cancel the normalizer exactly.

---

## 2. Tempering only the existing interaction

Define the bridge

\[
p_\beta(S\mid x)\propto
\exp\{E_0(S;x)+\beta V_\Phi(S)\},
\qquad 0\le\beta\le1.
\tag{S.5}
\]

At $\beta=0$, utilities, categories, total size and complete support are retained; only
the Gram interaction is absent. At $\beta=1$, this is exactly the fitted Version-4 law.
Thus $\beta$ is an algorithmic path, not a model parameter. Let

\[
m(S)=\sum_{j\in S}\phi_j.
\tag{S.6}
\]

### Proposition 1 — exact Hubbard--Stratonovich augmentation at every bridge

For $\beta\in[0,1]$, define

\[
\widetilde p_\beta(S,z\mid x)=
\exp\left\{E_0(S;x)-\frac12\|z\|^2
+\sqrt\beta\,z^\top m(S)
-\frac\beta2\sum_{j\in S}\|\phi_j\|^2\right\}.
\tag{S.7}
\]

Integrating $z\in\mathbb R^r$ produces a basket density proportional to
$\exp\{E_0(S;x)+\beta V_\Phi(S)\}$, and

\[
z\mid S,x,\beta\sim
\mathcal N\!\left(\sqrt\beta\,m(S),I_r\right).
\tag{S.8}
\]

#### Proof

Complete the square:

\[
-\frac12\|z\|^2+\sqrt\beta\,z^\top m(S)
=-\frac12\|z-\sqrt\beta m(S)\|^2
+\frac\beta2\|m(S)\|^2.
\tag{S.9}
\]

The Gaussian integral contributes the same constant for every basket. The remaining
basket energy is

\[
E_0(S;x)+\frac\beta2\|m(S)\|^2
-\frac\beta2\sum_{j\in S}\|\phi_j\|^2
=E_0(S;x)+\beta V_\Phi(S),
\tag{S.10}
\]

using (S.4). Equation (S.9) also gives (S.8). $\square$

Conditioning on $z$ makes the interaction additive. Product $j$ has conditional log
weight

\[
\eta_j(z,\beta,x)=b_j(x)-\frac\beta2\|\phi_j\|^2
+\sqrt\beta\,z^\top\phi_j.
\tag{S.11}
\]

---

## 3. Exact conditional basket law given the latent state

Set $w_j=\exp\{\eta_j\}$. For category $c$, define

\[
e_{c,k}(z)=
\sum_{\substack{A\subseteq\mathcal A_{xc}\\|A|=k}}
\prod_{j\in A}w_j,
\qquad e_{c,0}=1,
\tag{S.12}
\]

\[
G_c(t;z)=\sum_{k=0}^{R_c}
e_{c,k}(z)e^{-\rho_cg(k)}t^k,
\qquad
A_n(z)=[t^n]\prod_cG_c(t;z).
\tag{S.13}
\]

### Proposition 2 — $A_n$ is the exact conditional mass at size $n$

For fixed $z,x,\beta$,

\[
\sum_{\substack{S\subseteq\mathcal A_x\\|S|=n}}
\exp\left\{\sum_{j\in S}\eta_j
-\sum_c\rho_cg(n_c(S))\right\}=A_n(z).
\tag{S.14}
\]

Consequently,

\[
P_\beta(N=n\mid z,x)=
\frac{A_n(z)e^{-\rho_0(n)}}
{\sum_{m=1}^{n_{\max}}A_m(z)e^{-\rho_0(m)}}.
\tag{S.15}
\]

#### Proof

Selecting degree $k_c$ from $G_c$ selects $k_c$ distinct products from category $c$,
sums their product weights through $e_{c,k_c}$ and attaches the original category factor.
Multiplication over categories enumerates every basket once. The coefficient of total
degree $n=\sum_ck_c$ retains exactly size-$n$ baskets. Multiplication by the size factor,
exclusion of degree zero and normalization prove (S.15). $\square$

The code evaluates (S.12) in stable log coordinates and multiplies (S.13) with a
degree-aware native polynomial tree over all sizes $1,\ldots,120$. Per-trip centering and
degree tilting are exact polynomial identities used to preserve float64 range.

### Proposition 3 — reverse category backtracking is exact

Suppose total size $n$ is drawn. Let $P_c(q)$ be the degree-$q$ coefficient in the product
of categories preceding $c$. Conditional on requiring $q$ products, the probability of
taking $k$ from category $c$ is proportional to

\[
G_{c,k}P_c(q-k),
\qquad G_{c,k}=e_{c,k}e^{-\rho_cg(k)}.
\tag{S.16}
\]

Repeated backward draws produce the exact category-count law given $N=n,z,x,\beta$.

#### Proof

The numerator is the total weight of partial baskets taking $k$ products from category
$c$ and $q-k$ from preceding categories. Summing over feasible $k$ gives the prefix
coefficient of degree $q$. Normalization and the chain rule prove the result. $\square$

### Proposition 4 — conditional Bernoulli gives the exact fixed-count subset

Within one category, choose any scalar shift $a$ and independent Bernoulli variables

\[
q_j=\frac{e^{\eta_j+a}}{1+e^{\eta_j+a}}.
\tag{S.17}
\]

Conditioning on exactly $k$ successes gives

\[
P(A\mid |A|=k)\propto\prod_{j\in A}e^{\eta_j},
\qquad |A|=k.
\tag{S.18}
\]

#### Proof

For a size-$k$ set $A$, its independent-Bernoulli probability is

\[
\prod_{j\in A}q_j\prod_{j\notin A}(1-q_j)
=\frac{e^{ka}\prod_{j\in A}e^{\eta_j}}
{\prod_j(1+e^{\eta_j+a})}.
\tag{S.19}
\]

The denominator and $e^{ka}$ are common to every size-$k$ set and cancel. $\square$

The implementation chooses $a$ so the Bernoulli expected count is close to $k$, then
rejects until the count is exactly $k$. The shift improves acceptance without changing
the law. If the attempt limit is exhausted, the sampler raises an error; it never
substitutes an approximate basket or skips the context. Propositions 2--4 prove that

\[
N\longrightarrow(N_c)_c\longrightarrow S
\tag{S.20}
\]

is exact conditional on $z,x,\beta$.

---

## 4. Sequential Monte Carlo for the full interaction law

Choose a fixed increasing schedule

\[
0=\beta_0<\beta_1<\cdots<\beta_L=1.
\tag{S.21}
\]

The current default uses 17 levels with

\[
\beta_\ell=1-\left(1-\frac{\ell}{L}\right)^2.
\tag{S.22}
\]

At $\beta_0=0$, draw $P$ exact baskets using one forward DP and repeated reverse
backtracking. At bridge $\ell$, calculate

\[
W_\ell^{(p)}=
\exp\{(\beta_\ell-\beta_{\ell-1})V_\Phi(S^{(p)})\},
\tag{S.23}
\]

\[
\widehat R_\ell=\frac1P\sum_{p=1}^PW_\ell^{(p)},
\qquad
\overline W_\ell^{(p)}=
\frac{W_\ell^{(p)}}{\sum_mW_\ell^{(m)}}.
\tag{S.24}
\]

Multinomially resample with probabilities $\overline W_\ell$. For each resampled basket,
draw

\[
z^{(p)}\sim\mathcal N\!\left(
\sqrt{\beta_\ell}\sum_{j\in S^{(p)}}\phi_j,I\right),
\tag{S.25}
\]

then draw $S'\mid z^{(p)},x,\beta_\ell$ exactly using Section 3.

### Proposition 5 — blocked mutation preserves the bridge law

The transition $S\to z\to S'$ in (S.25) leaves $p_{\beta_\ell}(S\mid x)$ invariant.

#### Proof

Proposition 1 supplies a joint density whose two full conditionals are exactly the
Gaussian in (S.25) and the Section 3 basket law. Successive full-conditional draws form a
Gibbs kernel, which preserves the joint law and hence its basket marginal. $\square$

Stationarity does not imply independence after one update. A short mutation can retain
resampling ancestry; the temperature schedule and ESS diagnostics remain important.

### Proposition 6 — the SMC estimator is unbiased on the $Z$ scale

Let $Z_\beta(x)$ normalize (S.5), with $Z_0(x)$ calculated exactly. Under exact
initialization, multinomial resampling and Proposition 5's invariant kernels,

\[
\widehat Z_1(x)=Z_0(x)\prod_{\ell=1}^{L}\widehat R_\ell
\tag{S.26}
\]

satisfies

\[
\mathbb E[\widehat Z_1(x)]=Z_1(x).
\tag{S.27}
\]

#### Proof

At the first bridge, exact $p_{\beta_0}$ particles make the expected average incremental
weight equal $Z_{\beta_1}/Z_{\beta_0}$. Multinomial resampling is conditionally unbiased
for the weighted empirical measure. Applying an invariant kernel does not change its
target expectation. Inductively, the expected unnormalized particle measure after bridge
$\ell$ has total mass $Z_{\beta_\ell}/Z_{\beta_0}$. Multiplication by exact $Z_0$ proves
(S.27). $\square$

This statement is deliberately about $Z$, not $\log Z$. Jensen's inequality gives

\[
\mathbb E[\log\widehat Z_1]\le\log Z_1,
\tag{S.28}
\]

so the finite-particle logarithm has downward bias.

### Proposition 7 — basket particles are consistent, not finite-sample IID exact

For a fixed finite schedule, bounded $f$, positive finite bridge weights and ergodic
invariant mutation kernels,

\[
\frac1P\sum_{p=1}^Pf(S_L^{(p)})
\xrightarrow[P\to\infty]{P}\mathbb E_{p_1}[f(S)].
\tag{S.29}
\]

#### Proof sketch

The base empirical measure obeys the law of large numbers. Importance weighting is a
continuous ratio of empirical averages when expected weight is positive. Unbiased
resampling preserves the limiting weighted measure, and invariant mutation maps
convergence at one bridge to convergence at the next. Induction proves (S.29). $\square$

At finite $P$, particles share ancestors. The code applies one additional $\beta=1$
blocked update before recording generated baskets. This improves diversity without
changing the target, but does not create IID samples.

The normalized bridge ESS is

\[
\operatorname{ESS}_\ell/P=
\frac{1}{P\sum_p(\overline W_\ell^{(p)})^2}.
\tag{S.30}
\]

A small value proves weight concentration among represented particles. A large value does
not prove that an unvisited mode is absent. Global confidence requires replicates,
alternative schedules or independent deterministic audits on manageable panels.

### 4.1 Computational cost

Let $J_x=|\mathcal A_x|$, let $C_x$ be the nonempty-category count, and let $R$, $N$,
$P$ and $L$ denote within-category degree, maximum basket size, particles and bridges.
One conditional table requires $O(J_xR)$ ESP arithmetic plus a degree-truncated category
product. A conservative direct bound for the latter is $O(C_xN^2)$; degree-aware
truncation makes realized work smaller because most categories have degree far below $N$.
The base stage pays for one table and $P$ reverse draws. Each mutated bridge evaluates
$P$ tables in a vectorized leading dimension. Interaction statistics cost

\[
O(PL\,\bar n r),
\tag{S.31}
\]

where $\bar n$ is mean size and $r$ active rank. No step enumerates $2^{J_x}$ baskets.
The implementation remains CPU-bound because its stable float64 DP kernels are native CPU
kernels.

---

## 5. Incidence probabilities and Rao--Blackwellization

Let $X_j(S)=\mathbf1\{j\in S\}$. Marginal incidence is

\[
\pi_j(x)=P(j\in S\mid x)
=\frac{\partial\log Z_+(x)}{\partial b_j}.
\tag{S.32}
\]

For a basket with coordinate $j$ removed, define

\[
r_j(S_{-j},x)=P(X_j=1\mid S_{-j},x).
\tag{S.33}
\]

It is a logistic probability whose log odds are the exact add-one energy difference,
including utility, Gram interaction, category and size increments.

### Proposition 8 — Rao--Blackwell incidence is unbiased and no noisier

For an exact target draw $S$,

\[
\mathbb E[r_j(S_{-j},x)]=\pi_j(x),
\qquad
\operatorname{Var}(r_j(S_{-j},x))
\le\operatorname{Var}(X_j(S)).
\tag{S.34}
\]

#### Proof

The first identity is the tower property. The law of total variance writes

\[
\operatorname{Var}(X_j)=
\operatorname{Var}(\mathbb E[X_j\mid S_{-j}])
+\mathbb E[\operatorname{Var}(X_j\mid S_{-j})],
\]

and the final term is nonnegative. $\square$

With finite SMC particles this inherits SMC approximation, but removes the extra
conditional Bernoulli noise from each represented basket neighborhood.

---

## 6. Exact counterfactual identity and particle reweighting

Let action $a$ produce energy $E_a(S;x)$, let subscript ${\mathrm f}$ denote the factual
context, and define

\[
\Delta_a(S;x)=E_a(S;x)-E_{\mathrm f}(S;x).
\tag{S.35}
\]

### Proposition 9 — counterfactual expectation identity

For every integrable basket functional $f$,

\[
\mathbb E_a[f(S)]=
\frac{\mathbb E_{\mathrm f}[f(S)e^{\Delta_a(S;x)}]}
{\mathbb E_{\mathrm f}[e^{\Delta_a(S;x)}]}.
\tag{S.36}
\]

#### Proof

The density ratio is

\[
\frac{p_a(S\mid x)}{p_{\mathrm f}(S\mid x)}
=e^{\Delta_a(S;x)}\frac{Z_{\mathrm f}(x)}{Z_a(x)}.
\]

Summing $f$ against it gives (S.36); the denominator is
$Z_a/Z_{\mathrm f}$. $\square$

When only contextual item utilities change,

\[
\Delta_a(S;x)=\sum_{j\in S}[b_j(x,a)-b_j(x,{\mathrm f})].
\tag{S.37}
\]

This is the exact weight used by price and promotion code. Both raw product-price
deviation and assortment-wide mean price must be updated. Factual particles use
self-normalized weights

\[
\widehat w_p(a)=
\frac{e^{\Delta_a(S^{(p)};x)}}{\sum_me^{\Delta_a(S^{(m)};x)}}.
\tag{S.38}
\]

The resulting estimator is generally biased at finite $P$, but consistent under
Proposition 7. Low action ESS means factual particles do not represent the action law;
SMC should then be rerun under the changed context rather than clipping weights.

Equation (S.36) is a probability identity, not causal identification. Calling the result
a causal elasticity additionally requires exogenous variation, randomization or another
credible causal design.

---

## 7. Recommendation without a normalizer

The locked evaluation reveals basket $R$, hides one product and asks which single
candidate $j\notin R$ completes it. Define

\[
s_j(R,x)=E(R\cup\{j\};x)-E(R;x).
\tag{S.39}
\]

### Proposition 10 — add-one energy gives the exact one-item-completion ranking

\[
P(j\text{ completes }R\mid x,|T|=1)
=\frac{e^{s_j(R,x)}}{\sum_{k\notin R}e^{s_k(R,x)}}.
\tag{S.40}
\]

Therefore ranking by $s_j$ is exact and needs no $Z_+$ or quadrature.

#### Proof

Every completion is $R\cup\{j\}$. Conditioning (S.2) on $R$ and one-item completion
cancels the original partition function and common $E(R;x)$, leaving (S.40). $\square$

Expanding gives

\[
s_j=b_j(x)+\sum_{k\in R}\phi_j^\top\phi_k
-\rho_{c(j)}[g(n_{c(j)}(R)+1)-g(n_{c(j)}(R))]
-[\rho_0(|R|+1)-\rho_0(|R|)].
\tag{S.41}
\]

The size increment is common to candidates and may be omitted from ranking. For
$g(k)=\binom{k}{2}$, the category increment is $n_{c(j)}(R)$. This protocol answers
“which one item completes this basket?”, not unconditional incidence (S.32).

---

## 8. Interaction meaning

Let $K=\Phi\Phi^\top$, so $K_{jk}=\phi_j^\top\phi_k$.

### Proposition 11 — the Gram matrix, not latent coordinates, is identified

For orthogonal $Q$, replacing $\Phi$ by $\Phi Q$ leaves all energies, probabilities,
recommendations and generated distributions unchanged.

#### Proof

\[
(\Phi Q)(\Phi Q)^\top=\Phi QQ^\top\Phi^\top=\Phi\Phi^\top.
\]

$\square$

A positive $K_{jk}$ means that, holding the rest of the basket and all other energy terms
fixed, including $j$ raises $k$'s add-one log score by $K_{jk}$. This is a model-energy
complement statement, not automatically a positive marginal cross-price effect: size
competition, category penalties, other products and context mixing can reverse marginal
response. It is not causal without an intervention design.

The embedding audit selects pairs using invariant $K_{jk}$, then checks held-out
co-incidence against a frequency-and-size null. Names are interpretations after the
invariant numerical test, not evidence used to select pairs.

---

## 9. Customer segmentation as a descriptive functional

The taste surface is $B^{\mathrm{taste}}=\Theta A^\top$. The nonnegative price surface,
before the scalar relative-price multiplier, is
$B^{\mathrm{price}}=\Gamma_+B_+^\top$.

### Proposition 12 — induced distances equal distances between fitted surfaces

For factor pair $L,R$, write $R^\top R=V\Lambda V^\top$ on its positive eigenspace and
define $Y=LV\Lambda^{1/2}$. Then

\[
\|Y_h-Y_{h'}\|^2
=\|(L_h-L_{h'})R^\top\|^2.
\tag{S.42}
\]

#### Proof

With $d=L_h-L_{h'}$,

\[
\|dV\Lambda^{1/2}\|^2
=dV\Lambda V^\top d^\top
=dR^\top Rd^\top
=\|dR^\top\|^2.
\]

$\square$

Thus the unscaled coordinates preserve distances between identified catalogue-wide taste
or price surfaces rather than arbitrary latent-axis signs. The implementation standardizes
taste and price blocks separately and gives each equal aggregate weight; that final metric
is a declared clustering choice, not a likelihood theorem.

Candidate cluster counts use no test outcomes. Three K-means repeats provide silhouette
and adjusted-Rand stability, every segment must contain at least 5% of households, and the
declared score is

\[
\text{silhouette}+0.05\times\text{stability}.
\tag{S.43}
\]

Only after labels are fixed are held-out baskets used for descriptions and predictive
checks. Segments summarize a continuous fitted household surface; they are not latent
classes added to (S.2).

---

## 10. Generation calibration

For size, category, item and category-pair projections, observed and generated counts form
smoothed distributions

\[
\widehat p_i=\frac{c_i+1/d}{\sum_kc_k+1},
\qquad
\widehat q_i=\frac{\widetilde c_i+1/d}{\sum_k\widetilde c_k+1}.
\tag{S.44}
\]

The audit reports

\[
D_{\mathrm{KL}}(\widehat p\|\widehat q)
=\sum_i\widehat p_i\log\frac{\widehat p_i}{\widehat q_i},
\tag{S.45}
\]

\[
D_{\mathrm{JS}}(\widehat p,\widehat q)
=\tfrac12D_{\mathrm{KL}}(\widehat p\|m)
+\tfrac12D_{\mathrm{KL}}(\widehat q\|m),
\quad m=\tfrac12(\widehat p+\widehat q),
\tag{S.46}
\]

and $D_{\mathrm{TV}}=\frac12\sum_i|\widehat p_i-\widehat q_i|$. These are divergences of
declared projections, not KL over all $2^{5455}$ baskets. A large projected discrepancy
proves mismatch in that projection; small reported discrepancies do not prove equality of
the full laws. Observed and generated split-half comparisons supply a sampling-noise
reference.

Generation also checks support, unavailable products, duplicates, bridge ESS, size
moments, extreme tails, fixed seeds and identical context mixtures.

---

## 11. Quantities: compatible in theory, presently uncertified

If $q_j\ge1$ is units conditional on $j\in S$, the broader model factors as

\[
P(S,(q_j)_{j\in S}\mid x)
=P(S\mid x)\prod_{j\in S}P(q_j\mid j\in S,x).
\tag{S.47}
\]

The coded family is shifted negative binomial,
$K_j=q_j-1\sim\operatorname{NB}(\mu_j,r)$:

\[
P(K_j=k)=
\frac{\Gamma(k+r)}{\Gamma(r)\Gamma(k+1)}
\left(\frac{r}{r+\mu_j}\right)^r
\left(\frac{\mu_j}{r+\mu_j}\right)^k.
\tag{S.48}
\]

### Proposition 13 — normalized conditional quantities do not alter basket $Z_+$

If every quantity factor sums to one, marginalizing quantities gives exactly $P(S\mid x)$.

#### Proof

For fixed $S$, summing (S.47) over all quantities yields

\[
P(S\mid x)\prod_{j\in S}\sum_{q_j\ge1}P(q_j\mid j\in S,x)=P(S\mid x).
\]

$\square$

Compatibility is not empirical readiness. The selected pipeline does not optimize or
certify these quantity parameters. Unit revenue, margin, replenishment and inventory need
a separately fitted and held-out-calibrated quantity block.

---

## 12. Restricted promotion-budget decision process

The planning state is $s_t=(\tau_t,B_t)$: campaign days and expected markdown budget
remaining. Actions are no promotion or one segment/bundle/discount. Responses use (S.36)
with common factual particles, subject to ESS and extreme-tail gates.

For discretized cost $\widetilde C_a$ and conservative value $L_a$,

\[
V_\tau(b)=\max_{a:\widetilde C_a\le b}
\{L_a+V_{\tau-1}(b-\widetilde C_a)\}.
\tag{S.49}
\]

### Proposition 14 — Bellman recursion is optimal for the declared planning model

If rewards and expected costs are stationary, additive and depend on history only through
remaining time and budget, (S.49) maximizes total declared reward over feasible sequences.

#### Proof

The zero-day value is correct by definition. Assuming optimality for $\tau-1$, any
$\tau$-day policy chooses a feasible first action, obtains $L_a$, then faces the optimal
continuation at reduced budget. Maximizing over the first action proves the induction
step. $\square$

### Proposition 15 — upward cost rounding prevents overspend

For grid width $h=B/M$, let

\[
\widetilde C_a=h\left\lceil C_a/h\right\rceil.
\tag{S.50}
\]

Any sequence feasible under rounded costs has actual expected cost at most $B$.

#### Proof

Each $C_a\le\widetilde C_a$. Summing over the sequence makes actual cost no larger than
rounded cost, which the dynamic program constrains to $B$. $\square$

The code tightens terminal utilization by one grid cell per day to cover accumulated
rounding slack. This proves correctness only for the discretized stationary planning
model. The current state omits visits, inventory, consumption, competitor actions and
causal treatment response; operational use requires an expanded MDP or POMDP.

---

## 13. End-to-end implementation contract

For every requested household/store/date context:

1. construct the assortment and all features used by $b_j(x)$;
2. draw exact $\beta=0$ baskets with one forward DP and repeated backtracking;
3. traverse the fixed temperature schedule using positive weights, resampling and exact
   blocked mutation;
4. record bridge ESS and the SMC estimate of $Z_1/Z_0$;
5. apply a final $\beta=1$ rejuvenation for generation;
6. map assortment-slot indices to product IDs;
7. compute incidence and size summaries with Rao--Blackwellization;
8. evaluate modest actions with exact energy-ratio weights, rerunning SMC when overlap is
   inadequate;
9. compare generated and observed baskets on identical context mixtures; and
10. permit planning only after sampling, tail and action-overlap gates pass.

| Mathematical object | Implementation |
|---|---|
| Contextual utility $b_j(x)$ | `RaggedModel.b_at` / `b_flat` in `ragged.py` |
| Conditional weights (S.11) | `conditional_log_tables_levels` |
| ESP/category/size DP | `esp_log_bucketed`, `log_poly_tree_degree_native` |
| Exact reverse draw | `_numpy_backtrack`, `_numpy_backtrack_repeated` |
| Interaction statistic | `basket_interaction` |
| Tempered SMC | `annealed_smc_logz` |
| Final invariant update | `blocked_rejuvenation` |
| Rao--Blackwell summaries | `rao_blackwell_particle_statistics` |
| Price/generation audit | `audit_particle_counterfactual_generation.py` |
| Customer segmentation | `audit_customer_segments.py` |
| Locked recommendation | `eval_smolyak_rank8_mrr.py` |
| Complement audit | `audit_interaction_embeddings.py` |
| Promotion planning | `run_segment_pricing_mdp.py` |

---

## 14. Authorized and unauthorized claims

When empirical gates pass, the implementation may claim that conditional basket draws are
exact; blocked mutation preserves each bridge; the SMC estimator is unbiased for $Z$, not
$\log Z$; finite-particle summaries are consistent; price reweighting follows an exact
density-ratio identity; add-one ranking is exact for its protocol; Gram comparisons are
rotation invariant; and the Bellman solution is optimal for its restricted planning table.

It may not claim that high ESS proves every mode was found, that projected divergences
establish full-joint equality, that observational price responses are causal, that
segments are true latent customer types, that the current generator models quantities or
visits, or that the promotion policy estimates deployable profit or ROI.
