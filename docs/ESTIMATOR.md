# Smolyak and alternative normaliser estimators for the unchanged version-4 model

Status: **current estimator theory and measured implementation**. This document does not
change the Version-4 basket law, its support, interaction energy, size response, incidence
formula, or counterfactual interpretation. Earlier sections compare possible outer
Gaussian estimators; Section 11 records the estimator actually used by the corrected
pipeline.

## 1. The exact numerical problem

For a trip context \(x\), the version-4 law is

\[
p_\theta(S\mid x)=\frac{\exp E_\theta(S;x)}{Z_\theta(x)},
\qquad
Z_\theta(x)=\sum_{\varnothing\ne S\subseteq\mathcal A_x}
\exp E_\theta(S;x),
\]

with the original energy

\[
E_\theta(S;x)
=\sum_{j\in S}b_j(x)
 +\sum_{j<k\,;\,j,k\in S}\phi_j^\top\phi_k
 -\sum_c\rho_c {n_c(S)\choose 2}
 -\rho_0(|S|).
\]

Let

\[
\mu_S=\sum_{j\in S}\phi_j,
\qquad
\alpha_S(x)=
\exp\left\{
\sum_{j\in S}\left(b_j(x)-\frac12\|\phi_j\|^2\right)
-\sum_c\rho_c{n_c(S)\choose2}-\rho_0(|S|)
\right\}.
\]

The Hubbard--Stratonovich identity gives the exact representation

\[
Z_\theta(x)=\mathbb E_{z\sim\mathcal N(0,I_d)}F_\theta(z;x),
\qquad
F_\theta(z;x)=\sum_S\alpha_S(x)e^{\mu_S^\top z}.
\]

The product/category/size recursion evaluates \(F\) without enumerating baskets. Every
estimator discussed below approximates only this outer Gaussian expectation.

### Proposition 1: the latent posterior is an exact Gaussian mixture

The normalized integration target is

\[
p_\theta(z\mid x)
=\frac{\varphi_d(z)F_\theta(z;x)}{Z_\theta(x)}
=\sum_S p_\theta(S\mid x)\,\mathcal N(z;\mu_S,I_d).
\]

**Proof.** Completing the square gives

\[
\varphi_d(z)e^{\mu_S^\top z}
=e^{\|\mu_S\|^2/2}\varphi_d(z-\mu_S).
\]

The interaction identity

\[
\frac12\left(\|\mu_S\|^2-
\sum_{j\in S}\|\phi_j\|^2\right)
=\sum_{j<k\,;\,j,k\in S}\phi_j^\top\phi_k
\]

implies \(\alpha_S e^{\|\mu_S\|^2/2}=e^{E_\theta(S;x)}\). Summing and dividing by
\(Z_\theta(x)\) proves the result. \(\square\)

This identity is central. Standard Smolyak integrates under a Gaussian centered at zero,
whereas the actual normalized mass is a mixture of unit Gaussians centered at the aggregate
basket embeddings \(\mu_S\). The estimator is easy when the important \(\mu_S\)'s occupy a
small, nearby, anisotropic region. It is hard when appreciable basket probability is split
among many distant centers.

Define

\[
\ell_x(z)=\log F_\theta(z;x)-\frac12\|z\|^2.
\]

If \(S\mid z,x\) denotes the polynomial law proportional to
\(\alpha_S(x)e^{\mu_S^\top z}\), then

\[
\nabla_z\ell_x(z)=\mathbb E[\mu_S\mid z,x]-z,
\]

\[
\nabla_z^2\ell_x(z)=\operatorname{Cov}(\mu_S\mid z,x)-I_d.
\]

Consequently, a latent mode satisfies the exact mean-field equation

\[
z^\star=\mathbb E[\mu_S\mid z^\star,x].
\]

A sufficient condition for a unique, strongly log-concave latent target is

\[
\sup_z\lambda_{\max}\!left(
\operatorname{Cov}(\mu_S\mid z,x)
\right)<1.
\]

When this inequality fails, multimodality is possible. Thus a “hard trip” is not defined by
basket size alone. It is a context whose latent basket-embedding distribution is broad,
poorly aligned with the rule, or multimodal.

## 2. What Smolyak is approximating

Let \(H_n\) be the orthonormal probabilists' Hermite polynomial and
\(H_\nu(z)=\prod_kH_{\nu_k}(z_k)\). The generating identity gives an exact Hermite
expansion.

### Proposition 2: exact Hermite coefficients of the version-4 integrand

If

\[
F(z;x)=\sum_{\nu\in\mathbb N_0^d}F_\nu(x)H_\nu(z),
\]

then

\[
F_\nu(x)
=\frac{1}{\sqrt{\nu!}}
\sum_S e^{E_\theta(S;x)}\mu_S^\nu,
\]

and therefore

\[
\boxed{
\frac{F_\nu(x)}{Z_\theta(x)}
=\frac{\mathbb E_{S\sim p_\theta(\cdot\mid x)}[\mu_S^\nu]}
{\sqrt{\nu!}}
}.
\]

**Proof.** For a standard Gaussian,

\[
\mathbb E\!\left[e^{\mu^\top z}H_\nu(z)\right]
=e^{\|\mu\|^2/2}\frac{\mu^\nu}{\sqrt{\nu!}}.
\]

Apply this term by term and use
\(\alpha_Se^{\|\mu_S\|^2/2}=e^{E_\theta(S;x)}\). \(\square\)

This is more informative than using \(\Phi^\top\Phi\) alone. Sparse-grid convergence is
controlled by mixed moments of the *selected basket sum* \(\mu_S\). Those moments depend on
utilities, prices, size/category terms, and context as well as on \(\Phi\).

For a parameter block \(\vartheta\) that does not alter the Hermite basis,

\[
\left(\partial_\vartheta F\right)_\nu
=\partial_\vartheta\left[
\frac{1}{\sqrt{\nu!}}\sum_S e^{E(S)}\mu_S^\nu
\right].
\]

The score integrand can therefore have a materially slower Hermite decay than the scalar
normaliser. Agreement in \(\log Z\) does not certify an accurate likelihood gradient.

## 3. Classical Smolyak: accuracy and node growth

For one-dimensional rules \(Q_\ell\), define

\[
\Delta_0=Q_0,
\qquad
\Delta_\ell=Q_\ell-Q_{\ell-1}.
\]

For a downward-closed multi-index set \(\Lambda\),

\[
Q_\Lambda=\sum_{\nu\in\Lambda}
\bigotimes_{k=1}^d\Delta_{\nu_k}.
\]

The historical implementation uses a classical isotropic combination rule with a
\((2i-1)\)-point Gauss--Hermite rule at one-based level \(i\). Write

\[
s=q-d
\]

for the excess Smolyak level. Increasing rank while leaving \(q\) fixed is not meaningful:
the rule requires \(q\ge d\). Matching the rank-4, \(q=7\) mixed resolution at rank 32
means \(q=35\), not \(q=7\).

The curse appears immediately in the mixed terms:

- \(s=0\) uses the origin only;
- \(s=1\) has roughly \(1+2d\) unique nodes, hence 65 at \(d=32\);
- \(s=2\) contains at least \(4{d\choose2}\) off-axis pair nodes, hence at least 1,984
  pair nodes at \(d=32\), before axis refinements; and
- \(s=3\) contains at least \(8{d\choose3}\) three-axis nodes, hence at least 39,680
  at \(d=32\), again before lower-order refinements.

The old rank-4, \(q=7\) rule has \(s=3\) and 201 merged nodes. Its rank-4, \(q=8\)
successor has 681 nodes. The same mixed resolution at rank 32 is already tens of thousands
of nodes. No implementation optimization can make that classical isotropic rule a practical
per-update estimator over the full catalogue.

### Radial reach

For a single exponential term, the exact Gaussian contribution is

\[
\mathbb E e^{\mu^\top z}=e^{\|\mu\|^2/2},
\]

and its mass is centered at \(z=\mu\). A finite origin-centered rule samples only its node
set \(\{z_i\}\). If an important \(\mu_S\) is far from every node, the rule must infer a
remote Gaussian component by polynomial extrapolation. Entire analyticity does not prevent
this failure: exponential growth in the tails controls the constant in the quadrature
error.

In one dimension, an \(m\)-point Gauss--Hermite rule has outer node of order \(\sqrt{2m}\),
whereas the term \(e^{tz}\varphi(z)\) is centered at \(z=t\). Hence the required univariate
level grows approximately quadratically with a displaced mode, \(m=O(t^2)\), if the rule is
not recentered. Sparse tensorization reduces mixed-dimensional growth; it does not remove
this radial requirement.

### Signed cancellation

Smolyak difference weights are signed. For positive \(F\), define

\[
\kappa
=\frac{\sum_i|w_i|F(z_i)}{\left|\sum_iw_iF(z_i)\right|}.
\]

Floating-point error is amplified by approximately \(\kappa\). A non-positive partition
estimate or large \(\kappa\) is a deterministic quadrature failure, not stochastic
variance, and must never be repaired by clamping.

## 4. What the Gaussian sparse-grid theorem actually guarantees

Chen's Gaussian sparse-quadrature theorem assumes:

1. univariate exactness and boundedness on Hermite polynomials; and
2. weighted mixed derivatives through sufficiently high order, with inverse directional
   weights in \(\ell^q\), \(0<q<2\).

Under those assumptions there exists a downward-closed \(N\)-index rule satisfying

\[
\|I(H)-Q_{\Lambda_N}H\|_{\mathcal H}
\le C(N+1)^{-s},
\qquad
s=\frac1q-\frac12.
\]

The nominal dimension does not enter the exponent, but the mixed-derivative envelope and
its summability enter the constant. The paper also states explicitly that the commonly used
goal-oriented posterior surplus construction does not inherit the theorem's
dimension-independent guarantee automatically.

For finite \(J\), finite support, and finite \(d\), the version-4 integrand and its score
are finite sums of exponentials times polynomials. They are in Gaussian \(L^2\), and
fixed-dimensional Gauss--Hermite quadrature eventually converges. This is only an eventual
convergence statement. A practical dimension-independent rate requires decay of the
directional basket-embedding moments in Proposition 2. A flat set of 32 important
directions has no cheap sparse-grid guarantee.

The correct a-priori directional quantity is therefore a bound on selected-basket sums,
for example

\[
a_k(x)=\max_{S\in\mathcal S_x}|(\mu_S)_k|
\le
\text{sum of the largest }R\text{ values among }\{|\phi_{jk}|\}_{j\in\mathcal A_x},
\]

not catalogue size by itself. This worst-case bound can be pessimistic. Proposition 2 gives
the sharper probability-weighted quantity, but it depends on the model distribution and
therefore changes during training.

## 5. Why 20 products were fast and 5,455 are not

The 20-product limit was an implementation cache, not a mathematical condition of
Smolyak. If only \(M\) products have nonzero \(\phi_j\), the other \(J-M\) product
polynomials are independent of \(z\) and can be computed once per minibatch. With all
5,455 rows active, every quadrature node changes every item weight.

For \(P\) nodes and batch size \(B\), merely forming all projections has cost

\[
\Omega(PBJd),
\]

and reading all node-specific product weights costs \(\Omega(PBJ)\). A direct degree-\(R\)
ESP implementation adds an upper-order term \(O(PBJR)\); category polynomial products add
their own degree-dependent work. With a sparse interaction mask, much of the \(J\)-dependent
work sits outside the node loop. With all products active, it sits inside.

Thus there are two independent explosions:

\[
\underbrace{P(d,\text{accuracy})}_{\text{quadrature nodes}}
\times
\underbrace{C_F(B,J,C,R,d)}_{\text{cost per node}}.
\]

The earlier 83.7-fold cache speedup at a 20-product mask cannot be retained unchanged when
every product has a nonzero interaction row. Parallel and fused kernels can reduce the
constant, but no exact all-product implementation can beat the \(\Omega(PBJ)\) input-work
lower bound.

## 6. Why a small quadrature error can stop learning

For a parameter block, write the true average log-likelihood gradient as

\[
g_{\mathrm{train}}
=g_{\mathrm{positive}}-g_{\log Z}.
\]

Quadrature changes this to

\[
\widehat g_{\mathrm{train}}
=g_{\mathrm{train}}-e_Z,
\qquad
e_Z=\widehat g_{\log Z}-g_{\log Z}.
\]

The positive and negative phases may each be large while their difference is small. Hence
an audit such as

\[
\frac{\|e_Z\|}{\|g_{\log Z}\|}\le0.1\%
\]

does not show that the learning direction is accurate. The meaningful ratio is

\[
\frac{\|e_Z\|}{\|g_{\mathrm{positive}}-g_{\log Z}\|}.
\]

### Proposition 3: fixed score bias creates a convergence floor

Let \(f\) be an \(L\)-smooth loss minimized with step \(1/L\), using
\(\widehat g=\nabla f+e\) and \(\|e\|\le\varepsilon\). Then

\[
\frac1T\sum_{t=0}^{T-1}\|\nabla f(\theta_t)\|^2
\le
\frac{2L(f(\theta_0)-f_\star)}{T}+\varepsilon^2.
\]

**Proof.** The smoothness inequality at
\(\theta_{t+1}=\theta_t-(\nabla f+e)/L\) yields

\[
f(\theta_{t+1})
\le f(\theta_t)-\frac{1}{2L}\|\nabla f(\theta_t)\|^2
+\frac{1}{2L}\|e_t\|^2.
\]

Sum over \(t\), use \(f(\theta_T)\ge f_\star\), and divide by \(T/(2L)\). \(\square\)

Therefore a fixed-fidelity estimator can stagnate even when its displayed \(\log Z\) is
stable. The score tolerance must tighten with the net learning signal. Final convergence
requires either increasing estimator fidelity or an absolute score-error target below the
desired optimization tolerance.

Increasing rank and activating more product rows do not imply faster optimization. They
increase statistical capacity, but they also increase quadrature effective dimension,
per-node work, and the number of weakly identified parameters. The rank-4/20-product and
higher-rank/full-product convergence rates are not estimator-only comparisons.

## 7. The strongest Smolyak design that remains theoretically defensible

The classical isotropic rank-32 rule should be rejected. A viable Smolyak continuation
would require all of the following.

### 7.1 Law-preserving coordinate alignment

For any orthogonal \(Q\), replacing quadrature nodes by \(Qy_i\) is an exact change of
Gaussian coordinates:

\[
\mathbb E_zF_\Phi(z)=\mathbb E_yF_\Phi(Qy).
\]

This changes neither \(\Phi\Phi^\top\) nor the version-4 law. The alignment should target
the basket-sum moments from Proposition 2, ideally an aggregate of

\[
M_x=\mathbb E_{p_\theta(S\mid x)}[\mu_S\mu_S^\top]
\]

over calibration contexts. Diagonalizing \(\Phi^\top\Phi\) is a cheap proxy, not a proof,
because it ignores which products the basket law selects.

If the basis is updated during training, near-degenerate eigenspaces must be aligned to the
previous basis, for example by orthogonal Procrustes, to prevent arbitrary axis swaps. The
basis should be frozen over an optimizer window. Differentiating a changing finite grid at
every step otherwise produces a moving approximate objective.

### 7.2 Dimension-adaptive, score-oriented indices

Use a downward-closed anisotropic index set rather than \(|\nu|_1\le s\). Refinement must
use a block-scaled vector containing the normaliser numerator and all negative-phase score
blocks, not scalar \(F\) alone. A candidate surplus is

\[
\eta_\nu
=\left\|\Delta_\nu
\left(F,
s_b^{-1}\partial_{\theta_b}F\right)_{b=1}^B
\right\|.
\]

However, the unobserved frontier tail still needs an a-priori envelope or a stronger
independent rule. The last selected surplus is not a proof of the remaining error.

### 7.3 Nested one-dimensional rules

Nested Genz--Keister rules reuse nodes across levels and were more accurate per point than
non-nested Gauss--Hermite in Chen's examples. They reduce construction and escalation
waste. They do not remove \(O(d^s)\) mixed-index growth, have a limited available level
sequence, and retain signed sparse-grid cancellation. They are a constant-factor
improvement, not the complete solution.

### 7.4 Posterior recentering only under a shape certificate

For a Gaussian proposal \(q_x=\mathcal N(m_x,L_xL_x^\top)\), the exact identity is

\[
Z(x)=\mathbb E_{q_x}
\left[F(z;x)\frac{\varphi_d(z)}{q_x(z)}\right].
\]

Smolyak can be applied in the standardized \(q_x\) coordinates. If the log-concavity
criterion after Proposition 1 holds with margin, choosing the unique mode and local
curvature removes radial displacement and can dramatically lower the required level. If
the target has separated modes, one Gaussian can miss mass regardless of local accuracy.

A finite mixture proposal is exact through the deterministic-mixture identity

\[
q_x(z)=\sum_{r=1}^M\alpha_rq_{xr}(z),
\qquad
Z=\sum_{r=1}^M\alpha_r
\mathbb E_{q_{xr}}
\left[F(z)\frac{\varphi(z)}{q_x(z)}\right].
\]

It costs approximately \(M\) sparse rules and is useful only when the number of material
modes is small and mode discovery itself is reliable.

### 7.5 A net-gradient accuracy contract

For every parameter block, the production rule \(L\) and stronger reference rule \(H\)
must satisfy both

\[
|\log Z_L-\log Z_H|\le\tau_Z
\]

and

\[
\|g_{L,b}-g_{H,b}\|
\le
\min\left(
\eta\|g_{\mathrm{positive},b}-g_{H,b}\|,
\varepsilon_{b,t}
\right),
\]

where \(\varepsilon_{b,t}\) decreases toward the desired final stationarity tolerance.
The score angle of the *net* gradient must also agree. A fixed percentage of the negative
phase is insufficient.

### 7.6 Separate kernel engineering from estimator claims

All-product projections should be a batched matrix multiplication, and node, trip, and
category work should be fused or parallelized subject to memory. Rule construction belongs
outside the optimizer loop. These changes can reduce latency without changing the
estimator, but the speed claim must be reported at matched score error, not matched node
count.

## 8. Other estimator families

### 8.1 Adaptive Laplace or adaptive Gauss--Hermite

Find a latent mode, use the inverse negative Hessian as scale, and integrate a residual near
that mode. The cost can be only a few mode/curvature evaluations when the target is strongly
log-concave. Laplace error is asymptotic in posterior concentration; it has no uniform
guarantee for the unrestricted mixture in Proposition 1. This is the best simple method for
certified unimodal contexts and a poor universal method.

### 8.2 Importance sampling and randomized QMC

For ordinary importance sampling from \(q\), the exact relative variance identity is

\[
\frac{\operatorname{Var}_q(\widehat Z)}{Z^2}
=\frac1N\chi^2\!\left(p_\theta(z\mid x)\,\|\,q(z)\right).
\]

Thus high variance is caused by proposal/posterior mismatch, not by randomness alone. A
single Gaussian proposal can have enormous \(\chi^2\) divergence from a separated Gaussian
mixture. Scrambled Sobol points improve integration rate only when the transformed weighted
integrand has the required mixed smoothness and tail behavior; the inverse-normal transform
is singular at the cube boundary. Randomization supplies an error diagnostic but does not
repair a missed mode. QMC remains useful after a trustworthy transport or mixture proposal,
not as the first universal estimator.

### 8.3 Annealed importance sampling and sequential Monte Carlo

Introduce bridges

\[
p_t(S\mid x)\propto
\exp\{E_0(S;x)+tE_{\mathrm{int}}(S;x)\},
\qquad 0=t_0<\cdots<t_K=1.
\]

SMC estimates the product of successive normalizer ratios. With resampling and invariant
rejuvenation kernels, the normalizer estimator is unbiased under the standard SMC
construction, and a central-limit error decreases as \(O(N^{-1/2})\) for fixed bridge
schedule under regularity. Adaptive temperatures control local weight variance. The price
is \(K\) transitions per context and a mixing problem in basket space. This is slower but
more robust to multiple modes than a single local quadrature. It is the implemented
positive basket generator and scalar-normalizer fallback, not the selected likelihood
trainer. Its exact H--S conditionals, reverse dynamic program, unbiased-normalizer result
and finite-particle limitations are proved in
[INFERENCE_AND_SIMULATION.md](INFERENCE_AND_SIMULATION.md).

### 8.4 Tempered MCMC negative phases

Persistent Gibbs, MALA, or replica exchange can estimate model expectations without
computing \(\log Z\) every update. Exact stationarity preserves the original likelihood
gradient, but finite-chain bias is governed by mixing time, which can become exponential
across separated modes. It also does not directly provide held-out likelihood. This can be
useful for long-run negative phases, but a finite-step implementation is not automatically
accurate.

### 8.5 Cumulant, Plefka, TAP, or cluster expansion

Let \(p_0\) be the same version-4 model with only the low-rank pair interaction removed,
while retaining the original utility, size, and category terms. If \(V(S)\) is the removed
interaction energy,

\[
\log Z=\log Z_0+log\mathbb E_{p_0}e^{V}
=\log Z_0+\sum_{m\ge1}\frac{\kappa_m(V)}{m!}.
\]

This avoids latent quadrature and can be fast when interaction strength lies inside a
verified high-temperature/cluster-convergence region. Truncation error becomes uncontrolled
near a phase transition or strong complementarity. It is promising as an analytic control
variate or a rigorously gated weak-interaction estimator, not as a universal replacement.

### 8.6 Spherical--radial quadrature

Using \(z=ru\), with \(r\sim\chi_d\) and \(u\) uniform on the sphere,

\[
Z=\mathbb E_u\mathbb E_r F(ru).
\]

Adaptive one-dimensional radial integration can explicitly search distant shells that an
origin-centered Cartesian rule misses. Angular cubature remains \((d-1)\)-dimensional, and
high polynomial degree again requires a number of directions growing polynomially with
large powers of \(d\). It addresses radial reach, not the full effective-dimension problem.

### 8.7 Variational, pseudolikelihood, NCE, and score matching

Variational bounds can be fast but optimize a biased normalizer. Pseudolikelihood, noise
contrastive estimation, and score matching replace the likelihood objective. They may be
valid alternative models or training criteria, but they are excluded from estimator-only
work on version 4.

### 8.8 Tensor trains and polynomial chaos compression

Low tensor rank in the Hermite coefficients could compress a high-dimensional rule. The
exact coefficient identity in Proposition 2 supplies the relevant object. There is no
general guarantee that its tensor rank stays low over the optimizer trajectory, and adaptive
cross construction still requires expensive calls to \(F\). This is a research direction,
not the simple immediate answer.

## 9. A limitation no estimator can evade

The family in Proposition 1 can contain many well-separated Gaussian components with
non-negligible weights. By adjusting utilities and embeddings, one can construct contexts
with more material modes than a fixed low-node rule visits. Analyticity alone does not rule
this out. Consequently there is no uniformly accurate, fixed-cost, low-node deterministic
quadrature for the unrestricted rank-32 version-4 parameter space.

One of the following structural facts must be established:

1. the trajectory stays in a strongly log-concave or few-mode region;
2. basket-embedding moments have sufficient anisotropic decay;
3. interaction strength stays inside a convergent cluster-expansion region; or
4. a more expensive global method such as tempered SMC is used when those facts fail.

This is not giving up on approximation. It identifies the assumption under which each
approximation is valid and prevents another long run from discovering the same violation
after the fact.

## 10. Theory-first decision

The most promising Smolyak path is not “rank-32 classical Smolyak.” It is:

1. an orthogonally aligned coordinate system based on basket-embedding moments;
2. dimension-adaptive downward-closed indices;
3. nested Genz--Keister or Gauss--Hermite differences;
4. posterior recentering/scaling only for contexts with a certified unimodal target;
5. a small deterministic mixture only when a small number of modes is certified;
6. score-oriented refinement and a tightening net-gradient error contract; and
7. fused, parallel full-catalogue kernels.

This design is worth testing only if a paper calculation on the declared parameter envelope
predicts a rule below the latency budget. Before any training, it must answer:

\[
\text{How fast do the rotated }\mu_S\text{ moments decay?}
\]

\[
\text{Is }\sup_z\lambda_{\max}\operatorname{Cov}(\mu_S\mid z,x)<1
\text{ for the intended common path?}
\]

\[
\text{How many univariate, pair, and higher mixed indices are required by the score tail?}
\]

\[
P\,C_F(B,J,C,R,d)\le\text{declared time budget?}
\]

If these bounds predict thousands of full-catalogue nodes, Smolyak must be rejected before
spending training compute. The next comparison should then be between a rigorously gated
weak-interaction expansion and tempered SMC, not another QMC node-count tweak.

## 11. Implemented resolution in the corrected pipeline

The corrected pipeline does not put a noisy Gaussian-integral estimate inside every
stochastic-gradient update. It exploits a simpler decomposition that is exact at the
parent and low-dimensional in the interaction stage.

### 11.1 Exact parent, exact draws

Set \(\Phi=0\) and fit the entire additive Version-4 parent with the exact
category/cardinality dynamic program. Let this fitted law be \(p_0(S\mid x)\). The same
recursion supports exact ancestral draws \(S_{md}\sim p_0(\cdot\mid x_m)\) over all 5,455
products and sizes \(1,\ldots,120\); there is no Markov chain and no signed quadrature in
these training draws.

The locally informative interaction object is the pair-statistic score at \(K=0\),

\[
R=
\mathbb E_{\rm data}\!\left[P(S)\right]
-
\mathbb E_{p_0}\!\left[P(S)\right],
\tag{26}
\]

where \(P(S)\) is the symmetric item-pair incidence matrix. Positive eigenvectors of \(R\)
are ascent directions in the PSD Gram kernel \(K=\Phi\Phi^\top\). Independent training
halves select a reproducible basis \(U\) before interaction magnitude is fitted.

### 11.2 Deterministic sampled likelihood ratio

In the selected basis write \(K=UCU^\top\), and combine the Gram change with two directions
inside the already-defined total-size potential:

\[
h_\eta(S)
=
\operatorname{tr}\{C F_U(S)\}
-a\frac{|S|}{10}
-c\left(\frac{|S|}{10}\right)^2,
\qquad
\eta=(C,a,c).
\tag{27}
\]

Because

\[
\frac{p_\eta(S\mid x)}{p_0(S\mid x)}
=
\frac{\exp h_\eta(S)}
{\mathbb E_{p_0(\cdot\mid x)}[\exp h_\eta(S)]},
\tag{28}
\]

fixed common draws give the objective

\[
\widehat G(\eta)
=
\frac1M\sum_{m=1}^M
\left[
h_\eta(S_m^{\rm obs})
-
\log\left\{
\frac1D\sum_{d=1}^D
\exp h_\eta(S_{md})
\right\}
\right].
\tag{29}
\]

For fixed draws, Eq. (29) is deterministic and concave in the natural coordinates:
a linear observed term minus log-sum-exp. Its score is

\[
\nabla_\eta\widehat G
=
\frac1M\sum_m
\left[
T(S_m^{\rm obs})
-
\sum_d\widetilde w_{md}T(S_{md})
\right],
\qquad
\widetilde w_{md}
=
\frac{e^{h_\eta(S_{md})}}
{\sum_{d'}e^{h_\eta(S_{md'})}}.
\tag{30}
\]

Thus the training variance is frozen into one declared Monte Carlo panel rather than
changing at every update. Effective sample size

\[
\operatorname{ESS}_m
=
\frac{1}{\sum_d\widetilde w_{md}^2}
\tag{31}
\]

directly detects when the child leaves the support region represented by its exact-parent
draws. Swapped context halves test out-of-sample gain, and the final quadrature likelihood
audit is independent of this sampled objective.

### Proposition 4: the sampled solve has no optimization-induced local minima

On the feasible set

\[
0\preceq C\preceq\sigma_{\max}^2I,
\qquad
c\ge0,
\qquad
a+(n_{\max}/10)c\ge0,
\tag{32}
\]

\(\widehat G(C,a,c)\) is concave.

**Proof.** Equation (27) is affine in \((C,a,c)\). For each context, the observed term is
affine and the log-sum-exp of affine functions is convex. Its negative is concave. Sums
preserve concavity. The PSD interval and two half-space size constraints form a convex
feasible set. Therefore every constrained local maximum is global for the fixed-draw
objective. \(\square\)

This proposition concerns optimization of the approximation. It does not assert that a
finite draw panel equals the exact likelihood, which is why cross-fit ESS and independent
quadrature certification remain mandatory.

### 11.3 Measured error and convergence

The corrected execution used 12,000 contexts, 64 exact parent draws per context and rank
5. Median ESS was \(63.878/64=0.9981\); the first percentile was \(59.446\). The full
fixed-draw gain was \(0.024517\pm0.001085\) nats/basket.

The locked child likelihood used the positive Smolyak target rule at level 7 and a level-8
audit on identical trips. The audit bounds were

\[
\epsilon_{\rm val}\le0.000510,
\qquad
\epsilon_{\rm test}\le0.000720
\quad\text{nats/basket}.
\tag{33}
\]

The paired child gains were \(0.021630\pm0.001581\) on validation and
\(0.023908\pm0.001647\) on test. Their lower confidence limits remain positive after
subtracting Eq. (33), so numerical error cannot explain the measured interaction gain.

For the population safety audit, level 6 is only a screen. Level 7 confirms the highest
risk contexts and a random 2,048-context panel estimates aggregate level bias. This
distinction matters: in the high-risk panel the mean absolute expected-size difference
between levels was \(1.95\), with maximum \(9.27\). The higher rule, not the cheap screen,
therefore controls the safety decision.

The estimator succeeds at likelihood fitting and certification, but the fitted law fails
the local extreme-tail gate. That is a model-parameter safety failure, not evidence that
Eq. (29) aborted or that the locked likelihood comparison is numerically unjustified.

## References

- P. Chen, “Sparse quadrature for high-dimensional integration with Gaussian measure,”
  *ESAIM: M2AN* 52 (2018), 631--657.
  <https://doi.org/10.1051/m2an/2018012>
- T. Gerstner and M. Griebel, “Dimension-adaptive tensor-product quadrature,”
  *Computing* 71 (2003), 65--87.
  <https://doi.org/10.1007/s00607-003-0015-5>
- A. Genz and B. D. Keister, “Fully symmetric interpolatory rules for multiple integrals
  over infinite regions with Gaussian weight,” *Journal of Computational and Applied
  Mathematics* 71 (1996), 299--309.
  <https://doi.org/10.1016/0377-0427(95)00232-4>
- P. R. Conrad and Y. M. Marzouk, “Adaptive Smolyak pseudospectral approximations,”
  *SIAM Journal on Scientific Computing* 35 (2013), A2643--A2670.
  <https://doi.org/10.1137/120890715>
- B. Bilodeau, A. Stringer, and Y. Tang, “Stochastic convergence rates and applications of
  adaptive quadrature in Bayesian inference,” *Journal of the American Statistical
  Association* 118 (2023), 243--261.
  <https://doi.org/10.1080/01621459.2021.1967164>
