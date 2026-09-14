# Probability foundations, basket generation, and price counterfactuals

Date: 2026-09-12

Scope: a reorganized foundation for the existing Version-4 incidence model.

This document defines one random experiment and one basket-valued random variable. Item
incidences, basket size, category counts, recommendations, and price-response statistics
then follow from the same probability measure. It distinguishes a valid probability law
from a numerical method for evaluating or sampling it, and a simulated price response
from an identified causal effect.

The basket energy and declared support are those of the current model. The current
branch implements the corrected price-response calculation, shared action transformations,
single-utility tilt formulas, and stronger ESS and household-uncertainty checks. Optional
model restrictions and causal extensions remain separate proposals. The associated
[synthetic pipeline](PROBABILITY_FOUNDATIONS_SYNTHETIC_PIPELINE.md) tests production code
against exact enumeration. Historical results remain in the specialist documents.

## 1. The random experiment

### 1.1 What is observed

Fix a household and shopping occasion, the store, the offered catalogue, prices, and
other declared contextual features. Observe the set of distinct modeled products in a
nonempty checkout basket. Repeated units of one product produce one incidence in that
set.

This is the experiment represented by the selected real-data implementation. Its outcome
does not include the decision to visit, an opportunity with no purchase, quantities,
purchase order, or subsequent household history. Those require additional observations
and probability specifications if they are to be generated.

Write the context as \(x=(w,a)\), where \(w\) contains background information and \(a\)
contains prices and any other explicitly varied action components. The decomposition
helps define a price experiment; it does not itself assert that historical prices were
randomized or unconfounded. For a causal analysis, which variables may be held fixed must
be justified by their position relative to the intervention.

Let \(\mathcal J=\{1,\ldots,J\}\) be the modeled catalogue, and let
\(\mathcal A_x\subseteq\mathcal J\) be the externally specified offered set. In the
current experiment, no inventory feed is available, and the declared offered set is the
complete training-defined catalogue at every modeled store. Its size is \(J=5{,}455\).

### 1.2 Sample space, events, and random variables

For fixed \(x\), define the finite, nonempty support

\[
\Omega_x
=\{s\subseteq\mathcal A_x:1\le |s|\le n_{\max}\},
\qquad n_{\max}=120,
\]

and the event sigma-algebra

\[
\mathcal F_x=2^{\Omega_x}.
\]

An outcome \(s\) is a set of products. An event \(A\in\mathcal F_x\) is a set of
possible baskets. These are different kinds of sets. For example, for product \(j\),

\[
A_j=\{s\in\Omega_x:j\in s\}
\]

is the event that the basket contains that product. Since the support is finite, every
subset of outcomes is measurable; no measure-theoretic complication is needed here.

On the canonical space \(\Omega_x\), define the basket-valued random variable by
\(B(s)=s\). Define the incidence vector, total size, and category counts by

\[
Y_j=\mathbf1\{j\in B\},\qquad
N=\sum_{j=1}^{J}Y_j,\qquad
N_c=\sum_{j:c(j)=c}Y_j.
\]

The basket and its incidence vector determine one another. Consequently,

\[
\sigma(B)=\sigma(Y_1,\ldots,Y_J)=\mathcal F_x.
\]

In contrast, \(\sigma(N)\) and \(\sigma((N_c)_c)\) are sub-sigma-algebras that usually
contain less information. Knowing the size does not identify the products. Knowing all
category counts usually does not identify them either.

If one instead begins with a richer underlying space \((\Xi,\mathcal G,P)\), the
definition is \(\sigma(B)=\{B^{-1}(A):A\subseteq\Omega_x\}\subseteq\mathcal G\).
Equality with the entire underlying sigma-algebra is then unnecessary and generally
false. Choosing the canonical basket space simply removes irrelevant underlying detail.

## 2. One probability law

### 2.1 Energy and probability measure

Let \(y(s)\) denote the incidence vector of basket \(s\). The Version-4 energy is

\[
E_\theta(s,x)
=\sum_{j\in s}b_j(x)
+\sum_{i<j;\,i,j\in s}K_{ij}
-\sum_c\rho_c {n_c(s)\choose2}
-\rho_0(|s|),
\qquad K=\Phi\Phi^\top.
\]

Higher values of \(E\) mean higher probability. Thus \(E\) is an energy score, with the
opposite sign from the physical-energy convention \(p\propto e^{-E}\).

Define

\[
Z_\theta(x)=\sum_{s\in\Omega_x}e^{E_\theta(s,x)},\qquad
p_\theta(s\mid x)=\frac{e^{E_\theta(s,x)}}{Z_\theta(x)},
\]

and, for every event \(A\in\mathcal F_x\),

\[
P_{\theta,x}(A)=\sum_{s\in A}p_\theta(s\mid x).
\]

This specifies the probability space \((\Omega_x,\mathcal F_x,P_{\theta,x})\).
Nonnegativity and total mass one are immediate; countable additivity follows from the
finite sum over disjoint outcomes.

**Existence statement.** If the support is finite and nonempty and all supported energies
are finite, then \(0<Z_\theta(x)<\infty\). No spectral bound, category-attraction bound,
or latent log-concavity assumption is required to define this probability measure.

Low rank and the PSD representation make the Gaussian reduction available. Coefficient
bounds and size regularization restrict the fitted family and help control computation
and extrapolation. They are not prerequisites for existence of a finite-support law.

### 2.2 Derived distributions need no new mechanisms

For any statistic \(T=T(B)\), its distribution is the pushforward of the basket law:

\[
P(T\in D\mid x)
=\sum_{s:T(s)\in D}p_\theta(s\mid x).
\]

In particular,

\[
\pi_j(x)=P(j\in B\mid x),\qquad
P(N=n\mid x)=\sum_{|s|=n}p_\theta(s\mid x).
\]

There is no need to posit independent category draws, independent item choices, or a
separate size distribution. Such assumptions would generally change the model.

The size potential \(\rho_0\) is not itself the size distribution. Define

\[
H_x(n)=\sum_{s\in\Omega_x:\,|s|=n}
\exp\left\{\sum_{j\in s}b_j(x)
+\sum_{i<j;\,i,j\in s}K_{ij}
-\sum_c\rho_c{n_c(s)\choose2}\right\}.
\]

Then

\[
P(N=n\mid x)=\frac{e^{-\rho_0(n)}H_x(n)}{Z_\theta(x)}.
\]

Utilities, interactions, and the combinatorial number of baskets all contribute through
\(H_x(n)\). A common \(\rho_0\) therefore does not imply a common size distribution
across contexts.

### 2.3 Random contexts and repeated observations

With random context \(X\), the model specifies a conditional probability kernel
\(x\mapsto P_{\theta,x}\), rather than a complete joint distribution of contexts and
baskets. It can be embedded in the fixed finite universe
\(\mathcal S=2^{\mathcal J}\), with zero mass outside \(\Omega_x\). Measurable utilities
and availability indicators make this a measurable kernel.

To generate a population of occasions, specify a context law \(Q\):

\[
P(X\in C,B\in A)=\int_C P_{\theta,x}(A)\,Q(dx).
\]

Alternatively, fix observed contexts \(x_1,\ldots,x_m\) and generate conditional
replicates. Taking those draws independently defines a conditional product experiment;
independence across real checkouts is an additional modeling assumption, not a
consequence of the single-occasion energy.

For a longitudinal simulator, specify the initial state, the policy selecting actions,
and the transition from a sampled basket to the next state. The basket kernel alone
does not determine any of those components.

## 3. Algebraic redundancies and identification

### 3.1 Category attraction is a structured pair term

The energy may equivalently be written

\[
E_\theta(s,x)
=b(x)^\top y(s)
+\sum_{i<j}W_{ij}y_i(s)y_j(s)
-\rho_0(\mathbf1^\top y(s)),
\]

where

\[
W_{ij}=K_{ij}-\rho_{c(i)}\mathbf1\{c(i)=c(j)\},\qquad i\ne j.
\]

Keeping the category and Gram components separate is computationally useful: category
counts permit exact polynomial calculations, and the low-rank Gram term permits a
low-dimensional Gaussian integral. They do not represent separate random experiments.
The combined off-diagonal matrix \(W\) need not be PSD.

### 3.2 Probability identifies energy differences

For supported baskets \(s,t\),

\[
\log\frac{p_\theta(s\mid x)}{p_\theta(t\mid x)}
=E_\theta(s,x)-E_\theta(t,x).
\]

Adding a context-only constant to every supported basket energy changes no probability.
An absolute energy level is therefore unidentified by a conditional basket law.

Other relevant invariances include:

- Common utility and linear size: \(b_j^+=b_j+c\) and
  \(\rho_0^+(n)=\rho_0(n)+cn\) preserve every energy.
- Interaction and quadratic size: \(K^+=K+c\mathbf1\mathbf1^\top\) and
  \(\rho_0^+(n)=\rho_0(n)+c{n\choose2}\) preserve every energy whenever the transformed
  parameters remain admissible.
- Latent rotation: \(\Phi^+=\Phi Q\), for orthogonal \(Q\), preserves \(K\).
- Kernel diagonals never enter the pair energy. Diagonal identification, when possible,
  comes from additional rank or structural restrictions, not from a direct diagonal
  sufficient statistic.

Thus rotation invariance alone does not prove that the entire Gram matrix or absolute
pair coefficients are statistically identified. State the rank, subspace, gauges, and
penalties under which a fitted representation is interpreted.

Fixing \(\rho_0(0)=0\) only sets an empty-basket energy convention outside the present
support. It does not identify an empty-basket probability from nonempty observations.
Likewise, fixing a size correction at \(n=1\) removes its constant-level redundancy;
if common utilities are also free, their linear-size gauge requires a separate choice.

For a background basket \(T\) such that all four baskets below are supported, the
invariant conditional log-odds contrast is

\[
\log\frac{p(T\cup\{i,j\}\mid x)p(T\mid x)}
{p(T\cup\{i\}\mid x)p(T\cup\{j\}\mid x)}
=W_{ij}-\big[\rho_0(t+2)-2\rho_0(t+1)+\rho_0(t)\big],
\quad t=|T|.
\]

Under nonempty, size-capped support, this probability statement requires
\(1\le t\le n_{\max}-2\). At excluded boundaries, the algebraic energy difference
is not the log odds of four positive-probability events. Even in the interior, this
conditional association is not automatically a marginal or causal cross-price effect.

## 4. Three identities organize inference

### 4.1 Log-partition derivatives give moments

Hold interactions, category coefficients, and size potential fixed, and treat utilities
\(b\in\mathbb R^J\) as free coordinates. Finite sums can be differentiated directly:

\[
\frac{\partial\log Z}{\partial b_j}=\mathbb E[Y_j],\qquad
\frac{\partial^2\log Z}{\partial b_i\partial b_j}
=\operatorname{Cov}(Y_i,Y_j).
\]

More generally, for energy \(\eta^\top T(s)+h(s)\),

\[
\nabla_\eta\log Z=\mathbb E[T(B)],\qquad
\nabla_\eta^2\log Z=\operatorname{Cov}(T(B)).
\]

The log likelihood score is observed minus expected sufficient statistics, and its
Hessian is negative semidefinite in these natural coordinates. Concavity does not
automatically extend to bilinear household/product factors or \(\Phi\) coordinates.

### 4.2 Perturbations give covariance responses

For a differentiable energy \(E_\varepsilon\) on fixed support and a statistic
\(f_\varepsilon\), differentiating the finite normalized sum gives

\[
\frac{d}{d\varepsilon}\mathbb E_\varepsilon[f_\varepsilon(B)]
=\mathbb E_\varepsilon\!\left[\frac{\partial f_\varepsilon(B)}{\partial\varepsilon}\right]
+\operatorname{Cov}_\varepsilon\!\left(
f_\varepsilon(B),\frac{\partial E_\varepsilon(B)}{\partial\varepsilon}\right).
\]

The first term is essential when the outcome itself contains the action price, as in
revenue. For a fixed statistic it vanishes. For \(E_\varepsilon=E+\varepsilon N\),

\[
\frac{d\mathbb E[N]}{d\varepsilon}=\operatorname{Var}(N).
\]

This last identity concerns a uniform utility increment. Turning it into a price
elasticity requires the actual price-to-utility map in Section 6.

### 4.3 Finite changes give exponential reweighting

For two energies on the same support, let \(\Delta(s)=E'(s)-E(s)\). Then

\[
\frac{Z'}{Z}=\mathbb E[e^{\Delta(B)}],\qquad
p'(s)=\frac{p(s)e^{\Delta(s)}}{\mathbb E[e^{\Delta(B)}]},
\]

and

\[
\mathbb E'[f(B)]
=\frac{\mathbb E[f(B)e^{\Delta(B)}]}{\mathbb E[e^{\Delta(B)}]}.
\]

**Proof.** Substitute \(p(s)=e^{E(s)}/Z\), multiply by \(e^{\Delta(s)}\), and sum.
The same identity supplies an interaction-training partition ratio and a price-action
expectation. They are different applications of one change of measure.

For changed assortment support, the ratio must also enforce the support indicator.
Reweighting cannot recover action baskets outside the factual support. Even with common
support, a finite sample may have poor overlap with the action distribution. Exact
identities do not make finite self-normalized estimates unbiased.

## 5. Scientific generation and numerical sampling

### 5.1 A generator exists without an additional behavioral story

Enumerate the finite support as \(s_1,\ldots,s_M\), and set
\(F_k(x)=\sum_{i\le k}p_\theta(s_i\mid x)\). If \(U\sim\mathrm{Uniform}(0,1)\),
return the first \(s_k\) for which \(U\le F_k(x)\). This defines

\[
B=T_\theta(x,U),\qquad B\sim p_\theta(\cdot\mid x).
\]

Enumeration establishes existence, not computational feasibility at \(J=5{,}455\).
The energy law is a complete stochastic specification for the declared conditional
outcome. Empirical fidelity, parameter uncertainty, and numerical sampling error are
separate questions. A good simulated fit is testable evidence, not a proof of the true
shopping mechanism.

### 5.2 The exact Gaussian augmentation

Write \(m(s)=\sum_{j\in s}\phi_j\) and let \(\varphi_r\) be the standard Gaussian
density. Define the augmented joint law

\[
p(s,z\mid x)=\frac{\varphi_r(z)}{Z_\theta(x)}
\exp\left\{
\sum_{j\in s}\left[b_j(x)-\tfrac12\|\phi_j\|^2+z^\top\phi_j\right]
-\sum_c\rho_c{n_c(s)\choose2}-\rho_0(|s|)
\right\}.
\]

Integrating \(z\) gives the original basket law because
\(\mathbb E_{Z\sim N(0,I)}e^{Z^\top m}=e^{\|m\|^2/2}\). Completing the square also
gives

\[
z\mid B=s,x\sim N(m(s),I_r).
\]

Let \(F_\theta(z,x)\) be the sum of the exponential term in the joint expression over
all supported baskets. The marginal distribution needed to sample \(z\) first is

\[
p(z\mid x)=\frac{\varphi_r(z)F_\theta(z,x)}{Z_\theta(x)}.
\]

It is generally not \(N(0,I_r)\). The Gaussian in the integral identity is a reference
measure. Drawing it directly and then drawing \(B\mid z,x\) generally produces a
different basket law. Nor does the augmentation identify \(z\) as a physical household
shock; its role here is computational.

### 5.3 Conditional sampling through polynomials

At fixed \(z\), let

\[
w_j(z,x)=e^{b_j(x)-\|\phi_j\|^2/2+z^\top\phi_j},\qquad
G_c(t)=\sum_{k=0}^{\min(|\mathcal A_x\cap c|,n_{\max})}
e_k((w_j)_{j\in\mathcal A_x\cap c})e^{-\rho_c{k\choose2}}t^k.
\]

If \(A_n=[t^n]\prod_cG_c(t)\), then

\[
P(N=n\mid z,x)=\frac{e^{-\rho_0(n)}A_n}
{\sum_{q=1}^{n_{\max}}e^{-\rho_0(q)}A_q}.
\]

After drawing \(N\), reverse polynomial sampling draws category counts and then the
products within each category. These are exact conditional calculations in exact
arithmetic. Their order is an application of the probability chain rule, not a claim
that customers choose size before products.

At \(\Phi=0\), the latent state is irrelevant and this gives exact conditional basket
draws directly. At nonzero \(\Phi\), the implementation uses interaction-tempered SMC
with exact conditional mutation kernels. A finite particle population approximates the
target; invariant transitions and an unbiased partition-function estimate do not imply
independent exact finite-particle basket draws.

Smolyak quadrature evaluates an integral using signed weights. Those weights are not
probabilities for drawing a latent state. A deterministic quadrature error is also not
eliminated merely because the procedure has no Monte Carlo randomness.

### 5.4 Recommendation is another conditioning operation

Given a revealed set \(R\), condition on \(R\subseteq B\) and
\(|B|=|R|+1\). For feasible candidates,

\[
P(B=R\cup\{j\}\mid R\subseteq B,|B|=|R|+1,x)
=\frac{e^{E(R\cup\{j\},x)}}{\sum_{k\notin R}e^{E(R\cup\{k\},x)}}.
\]

The full normalizer and common size potential cancel. This supports exact add-one
ranking. It is not the general probability that an item occurs in any completion with
arbitrary additional products. Interpreting an observed partial basket also requires an
appropriate reveal mechanism; the uniform hide-one evaluation has the required common
factor at fixed final size.

## 6. Price responses under the fitted law

All derivatives in this section hold background context, model parameters, and offered
support fixed, while recomputing deterministic price features as specified. They are
responses of the fitted conditional law. Their causal interpretation is addressed in
Section 7.

### 6.1 General response and aggregate elasticity

Let \(\ell_k=\log p_k\), and suppose price enters only the utilities. Put

\[
A_{jk}(x)=\frac{\partial b_j(x)}{\partial\ell_k}.
\]

For \(\pi_i=\mathbb E[Y_i]\), Section 4 gives

\[
\frac{\partial\pi_i}{\partial\ell_k}
=\sum_j\operatorname{Cov}(Y_i,Y_j)A_{jk},\qquad
\frac{\partial\pi}{\partial\ell^\top}=\operatorname{Cov}(Y)A.
\]

For an unsplit price specification \(b_j=\cdots-g_{hj}\ell_j\), with \(g_{hj}\ge0\),

\[
\frac{\partial\pi_i}{\partial\ell_k}
=-g_{hk}\operatorname{Cov}(Y_i,Y_k),\qquad
\frac{\partial\pi_k}{\partial\ell_k}
=-g_{hk}\pi_k(1-\pi_k)\le0.
\]

These statements retain all category, size, and interaction effects. Cross-price
response depends on marginal covariance; a positive fitted Gram entry alone does not
fix its sign.

For a common log-price increase \(\ell_j^+=\ell_j+\varepsilon\), define the
price-weighted basket statistic

\[
G_h(B)=\sum_jg_{hj}Y_j.
\]

Whenever the price map gives \(\partial_\varepsilon b_j=-g_{hj}\),

\[
\frac{d\mathbb E_x[N]}{d\varepsilon}=-\operatorname{Cov}_x(N,G_h(B)).
\]

For fixed context weights \(\omega_x\) summing to one, the elasticity of aggregate
expected distinct-product demand is

\[
\epsilon_{\mathrm{agg}}
=-\frac{\sum_x\omega_x\operatorname{Cov}_x(N,G_h(B))}
{\sum_x\omega_x\mathbb E_x[N]}.
\]

If \(g_{hj}=g_h\) for all products, the context-specific covariance reduces to
\(g_h\operatorname{Var}_x(N)\). Replacing it by a catalogue-average coefficient times
variance is not an identity with heterogeneous product coefficients. Averaging across
contexts requires retaining any dependence between \(g_h\) and conditional variance.

**Implemented correction.** The additive trainer now targets the covariance response
using differentiable fourth-order symmetric perturbations of exact DP size expectations.
With \(M(\varepsilon)=\mathbb E_\varepsilon[N]\), it computes
\(D_h=[M(h)-M(-h)]/(2h)\) and \(D^{(4)}=(4D_h-D_{2h})/3\). The step is capped by the
maximum supported energy perturbation, and disagreement with \(D_h\) fails closed when
it exceeds the declared absolute/relative tolerance. This is a numerical derivative of
an exact normalizer, not an exact derivative or a deterministic error certificate.
Only first native DP adjoints are required, avoiding unsupported double differentiation
through the existing custom adjoint. The synthetic oracle checks both response values
and penalty gradients. Four perturbed DP evaluations are added when the penalty is on.

### 6.2 The current common/relative price parameterization

For a fixed offered set of size \(J_x\), write

\[
b_j=\cdots-g_{hj}\{m+\kappa(d_j-m)\},\qquad
m=\frac1{J_x}\sum_{q\in\mathcal A_x}d_q,\qquad\kappa>0,
\]

where \(d_j\) is the log-price deviation with its reference centering held fixed.
For offered \(j,k\),

\[
A_{jk}=-g_{hj}\left[\kappa\mathbf1\{j=k\}
+\frac{1-\kappa}{J_x}\right].
\]

Thus a single-product price intervention also changes the other products' utilities
through \(m\). The exact incidence derivative is

\[
\frac{\partial\pi_i}{\partial\ell_k}
=-\kappa g_{hk}\operatorname{Cov}(Y_i,Y_k)
-\frac{1-\kappa}{J_x}\operatorname{Cov}(Y_i,G_h(B)).
\]

Although the direct own-utility derivative is nonpositive, the sign of the total
own-incidence derivative does not follow from coefficient positivity alone. Its second
term can offset the first. Conversely, a uniform price increment gives
\(dm/d\varepsilon=1\), leaves relative deviations unchanged, and recovers the aggregate
covariance formula in Section 6.1.

Setting \(\kappa=1\) is an optional simplification: it removes mean-induced cross-item
utility effects and restores the direct own-price guarantee. It also removes the
separately fitted common/relative price response, so it should be evaluated as a model
restriction rather than called an equivalent reparameterization.

### 6.3 Closed-form single-product utility changes

Suppose only \(b_k\) changes by \(d\). Let
\(r=e^d\), \(\pi_i=\mathbb E[Y_i]\), and
\(\pi_{ik}=\mathbb E[Y_iY_k]\), so \(\pi_{kk}=\pi_k\). Since

\[
e^{dY_k}=1+(r-1)Y_k,
\]

the finite-change identity gives

\[
\frac{Z'}{Z}=1+(r-1)\pi_k,
\]

\[
\pi_k'=\frac{r\pi_k}{1+(r-1)\pi_k},\qquad
\pi_i'=\frac{\pi_i+(r-1)\pi_{ik}}{1+(r-1)\pi_k}.
\]

For a fixed statistic \(f\), the same argument gives

\[
\mathbb E'[f]
=\frac{\mathbb E[f]+(r-1)\mathbb E[fY_k]}
{1+(r-1)\pi_k}.
\]

Given the factual moments, no new full normalizer calculation is needed for this update.
The formulas remain exact with arbitrary interactions and the declared size support.
They are exact identities, not a claim that estimated factual moments are error-free.
The stable implementation is `single_utility_incidence` in
[`price_response.py`](../scripts/version4/price_response.py); it also handles forced
absent/present items without overflowing under large finite utility changes.

For the unsplit price map, a log-price change \(\delta\) gives \(d=-g_{hk}\delta\).
For the current \(\kappa\ne1\) specification, an actual one-product price change
generally changes multiple utilities, so the single-utility shortcut does not apply;
use the general exponential reweighting formula instead.

### 6.4 Revenue has a direct price effect as well as a demand effect

For the distinct-product revenue proxy \(R_a(B)=\sum_jp_j(a)Y_j\),

\[
\frac{d\mathbb E_a[R_a]}{da}
=\sum_j\frac{dp_j}{da}\pi_j(a)
+\operatorname{Cov}_a\!\left(R_a(B),\frac{\partial E_a(B)}{\partial a}\right).
\]

This is not unit-sales revenue unless quantities equal one. Actual revenue requires
\(\sum_jp_jQ_jY_j\) and a fitted conditional quantity law. Profit also requires costs.
A lower price can increase incidence and still reduce revenue or profit.

## 7. What a price counterfactual means

### 7.1 Three distinct targets

| Target | Mathematical object | What is needed |
|---|---|---|
| Changed-price prediction | \(p_\theta(s\mid w,a')\) | Evaluate the fitted law at the declared features |
| Intervention response | \(P(B(a')=s\mid W=w)\) | A causal design or justified intervention assumptions |
| Individual counterfactual | \(P(B(a')=s\mid B(a)=b,W=w)\) | A joint model linking potential baskets across actions |

The energy law supplies the first directly. Equating it to the second requires more
than a good observational likelihood. A sufficient route, for a well-defined outcome and
action, is consistency, conditional exchangeability \(B(a)\perp A\mid W\), and
positivity/overlap, with interference handled by the experimental unit or explicitly
modeled. Under those assumptions, the conditional observational outcome distribution
identifies the corresponding intervention distribution. For continuous prices,
identification is understood over supported price regions with appropriate regularity,
not through a positive point mass at every exact price.

These are assumptions about the data-generating experiment. Retail threats include
promotion targeting, simultaneous display changes, demand-driven pricing, stockouts,
and customers switching stores. Defining a sigma-algebra or sampling synthetic baskets
does not resolve them. See the sources in Section 9 for the distinction between
observational, interventional, and counterfactual questions.

### 7.2 A random seed does not identify individual counterfactuals

The generator representation permits

\[
B(a)=T_\theta(w,a,U),\qquad
B(a')=T_\theta(w,a',U).
\]

Using the same \(U\) defines a coupling across actions. It is not uniquely determined by
the marginal distributions. For example, let a binary purchase have probability one
half under each of two actions. With \(U\) uniform, both of these constructions match
those probabilities:

\[
Y(0)=\mathbf1\{U\le1/2\},\quad Y(1)=Y(0),
\]

\[
Y(0)=\mathbf1\{U\le1/2\},\quad Y(1)=1-Y(0).
\]

They disagree completely about what happens to a particular individual. Randomizing
actions can identify each marginal distribution without identifying this cross-action
joint distribution.

Common random numbers are useful for numerical comparisons. Interpreting the shared
randomness as a shopper's persistent latent state requires a structural assumption,
an appropriate factual-conditioning step, and scientific justification. The auxiliary
H–S Gaussian does not provide that interpretation automatically.

### 7.3 Purchase opportunities, selection, and total demand

The current real-data law conditions on a nonempty modeled basket. It does not estimate
how price changes whether an opportunity produces a purchase. A full opportunity law
can be constructed with a purchase probability \(v(w,a)\):

\[
P(B=s\mid w,a)=
\begin{cases}
1-v(w,a),&s=\varnothing,\\
v(w,a)p_\theta(s\mid w,a,R=1),&s\ne\varnothing,
\end{cases}
\]

where \(R\) records a nonempty modeled purchase on a defined opportunity. The definition
must specify how visits, purchases outside the modeled catalogue, and store choice are
handled; \(R=0\) is not automatically the same as no store visit.

For distinct-product demand per opportunity,

\[
\mathbb E[N\mid w,a]
=v(w,a)\mathbb E[N\mid w,a,R=1].
\]

Its derivative contains both the change in purchase probability and the change in
conditional basket size. Conditioning on observed purchase after an intervention can
also change the composition of purchasers. Even randomized prices do not make such a
comparison the effect on the same set of shoppers. Opportunity-level observations are
needed to study total demand and selection explicitly.

A conditional quantity law can then be appended. Assuming a product-wise factorization
for quantities is an additional modeling assumption; the fact that quantities can be
marginalized out does not derive their conditional independence.

### 7.4 A controlled experiment that can test the generator

A practical validation design fixes the eligible opportunity population, randomizes
well-defined price actions at a unit that accounts for spillovers, and records offered
availability, other promotion components, non-purchases, basket incidence, quantities,
and costs where relevant. Predictions and evaluation rules are fixed before examining
the experimental outcomes.

Compare predicted and observed action differences in purchase rates, item incidence,
basket size, and revenue or margin on that population. This evaluates both the response
model and the outcome coverage required for the decision. A synthetic experiment with
a declared oracle is useful for checking recovery and numerical correctness, but does
not substitute for intervention evidence in the real population.

## 8. Implementation boundary and suggested simplifications

| Component | Current implementation | Interpretation or proposed correction |
|---|---|---|
| Basket energy and utilities | [`ragged.py`](../scripts/version4/ragged.py), `energy`, `b_at` | One conditional incidence law |
| Exact no-Gram normalizer and size masses | [`interaction_particles.py`](../scripts/version4/interaction_particles.py), `differentiable_log_size_beta0` | Exact finite DP, subject to arithmetic |
| Conditional reverse sampler | [`tempered_block_gibbs.py`](../scripts/version4/tempered_block_gibbs.py) | Exact conditional draws; not a physical purchase sequence |
| Interaction population sampler | [`tempered_ais.py`](../scripts/version4/tempered_ais.py), `annealed_smc_logz` | Finite SMC approximates the interaction law |
| Interaction fit | [`stratified_natural.py`](../scripts/version4/stratified_natural.py) | Concavity applies to the fixed-bank natural block |
| Elasticity penalty | [`fit_exact_additive.py`](../scripts/version4/fit_exact_additive.py) | Audited fourth-order response through exact DP expectations |
| Price-action audit | [`audit_particle_counterfactual_generation.py`](../scripts/version4/audit_particle_counterfactual_generation.py) | Recompute both price components and audit finite-particle overlap |
| Single-utility finite update | [`price_response.py`](../scripts/version4/price_response.py) | Exact formula from factual moments; actual split-price actions use the full utility change |
| Opportunity and quantity extension | Section 7.3 | Not fitted or certified by the selected real-data pipeline |

The theory can be maintained with one foundational definition and three reusable
identities rather than separate probability models for each query. Keep the H–S/ESP
calculation as a computational theorem, and move optimizer details and run-specific
certification into their specialist documents.

The current stratified fit uses 76 draws allocated as \((16,16,12,8,8,8,8)\), and requires
both ESS fraction at least 0.20 and absolute ESS at least 2 in every active band, for
both cross-fit directions and the final bank. These diagnostics still cannot certify
coverage of unseen compositions. Paired likelihood and interaction cross-fit reports
now use household-clustered uncertainty; adjacent-rule allowances are labeled empirical.

Possible modeling simplifications should be tested separately: \(\kappa=1\) simplifies
price interpretation; more compact size bases regularize a sparse tail; restricted joint
utility/interaction refinement may improve a staged fit. These change the estimation
domain or fitted family and are not algebraic consequences of the probability-space
construction. No such change is made here.

## 9. Further derivations and sources

- [`THEORY.md`](THEORY.md): the expanded Version-4 energy and model derivations.
- [`ESTIMATOR.md`](ESTIMATOR.md): Gaussian integration and estimator analysis.
- [`SIZE_STRATIFIED_JOINT_ESTIMATOR.md`](SIZE_STRATIFIED_JOINT_ESTIMATOR.md): fixed-bank
  stratified estimation, its concavity, and its finite-draw limitations.
- [`INFERENCE_AND_SIMULATION.md`](INFERENCE_AND_SIMULATION.md): conditional sampling,
  SMC, reweighting, recommendation, and simulation.
- [`JOINT_INTERACTION_POLISH_AUDIT.md`](JOINT_INTERACTION_POLISH_AUDIT.md): the existing
  restricted joint-refinement experiment.
- Judea Pearl, [discussion of causal questions and their required assumptions](https://causality.cs.ucla.edu/blog/index.php/category/causal-effect/):
  observational, intervention, and individual counterfactual targets require different
  information about the generating mechanism.
- Miguel A. Hernán and James M. Robins, *Causal Inference: What If*, Chapters 1–3:
  consistency, exchangeability, positivity, and the distinction between randomized and
  observational studies.

The finite-space constructions, reweighting formulas, price Jacobian, and binary coupling
example are derived directly above. They do not depend on the numerical success of a
particular fitted checkpoint.
