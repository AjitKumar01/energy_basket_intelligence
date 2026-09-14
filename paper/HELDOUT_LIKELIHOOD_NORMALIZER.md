# Held-out basket likelihood and the Version-4 normalizer

## 1. The question answered by this document

For a held-out checkout context \(x_t\), the data contain one observed nonempty basket
\(S_t\). The evaluation asks:

> Under the fitted Version-4 model, what probability was assigned to exactly this basket
> among every supported nonempty basket that could have been purchased in the same
> context?

The answer is the conditional log likelihood

\[
\ell_t(\Theta)
=\log p_\Theta(S_t\mid x_t,N_t\ge1)
=E_\Theta(S_t,x_t)-\log Z_{+,\Theta}(x_t).
\tag{1}
\]

The observed energy in the first term is evaluated directly. The difficult term is the
normalizer

\[
Z_{+,\Theta}(x_t)
=\sum_{\substack{S\subseteq\mathcal A_{x_t}\\1\le |S|\le n_{\max}}}
\exp\{E_\Theta(S,x_t)\},
\qquad n_{\max}=120.
\tag{2}
\]

This repository **does use Smolyak quadrature** when evaluating held-out likelihood for a
checkpoint with active Gram interactions. Smolyak is used only for the remaining
low-dimensional Gaussian expectation after the discrete basket sum has been reduced
exactly by the Hubbard--Stratonovich identity and polynomial dynamic programming.

For the accepted rank-five checkpoint:

| Purpose | Smolyak level | Nodes |
|---|---:|---:|
| Lower-cost comparison | \(q=6=r+1\) | 11 |
| Reported validation likelihood | \(q=7=r+2\) | 71 |
| Higher-accuracy audit | \(q=8=r+3\) | 341 |

The 71-node rule does not examine only 71 baskets. At each node, the dynamic program sums
over the complete declared catalogue and every supported basket size from 1 through 120.
The nodes belong to the five-dimensional auxiliary Gaussian integral, not to product or
basket support.

---

## 2. Construction of the held-out evaluation panel

The selected evaluation executable is
`scripts/version4/compare_rank8_parent_likelihood.py`. The historical filename contains
“rank8”, but the implementation infers and verifies the checkpoint's actual active rank.
The current checkpoint has active rank five.

For validation, the executable performs the following steps:

1. Load the exact additive-parent checkpoint and the final interaction checkpoint.
2. Verify checkpoint capability, data fingerprint, support and active rank.
3. Select baskets having split identifier 1 and observed size at most 120.
4. Apply a deterministic permutation with the declared seed.
5. Retain the first 4,096 validation trips.
6. Score the parent and child on the identical ordered trip manifest.
7. Save every trip identifier and every paired score in
   `reports/likelihood_validation_per_trip.npz`.

The test calculation uses split identifier 2 and the same procedure. Validation is used
for model acceptance; test is reporting-only.

For each selected trip, the batcher creates two synchronized representations:

- the purchased product lines used in the observed numerator; and
- the full declared assortment used in the denominator.

Both representations call the same contextual utility function. This is essential. If a
price, promotion, household or store term were present in the normalizer but absent from
the observed energy, the resulting difference would not be a probability.

---

## 3. The Version-4 basket probability

The energy of basket \(S\) in context \(x\) is

\[
\begin{aligned}
E_\Theta(S,x)
={}&\sum_{j\in S}b_j(x)
+\sum_{\substack{j<k\\j,k\in S}}\phi_j^\top\phi_k\\
&-\sum_c\rho_c{n_c(S)\choose2}
-\rho_0(|S|).
\end{aligned}
\tag{3}
\]

The symbols are:

- \(b_j(x)\): contextual utility of product \(j\), including popularity, household
  taste, price, promotion, week and store terms;
- \(\phi_j\in\mathbb R^r\): active interaction embedding of product \(j\);
- \(n_c(S)\): number of selected products in affinity group \(c\);
- \(\rho_c\): within-group pair potential; and
- \(\rho_0(n)\): total-size potential for a basket of size \(n\).

Every supported basket receives a positive unnormalized weight

\[
W_\Theta(S,x)=e^{E_\Theta(S,x)}.
\tag{4}
\]

Dividing by the sum of all such weights gives

\[
p_\Theta(S\mid x,N\ge1)
=\frac{W_\Theta(S,x)}{Z_{+,\Theta}(x)}.
\tag{5}
\]

### Proposition 1: Eq. (5) is a normalized probability law

For a finite declared support and finite energies,

\[
\sum_{\substack{S\subseteq\mathcal A_x\\1\le|S|\le n_{\max}}}
p_\Theta(S\mid x,N\ge1)=1.
\tag{6}
\]

**Proof.** Substitute Eq. (5) and use the definition in Eq. (2):

\[
\begin{aligned}
\sum_Sp_\Theta(S\mid x,N\ge1)
&=\sum_S\frac{e^{E_\Theta(S,x)}}{Z_{+,\Theta}(x)}\\
&=\frac{\sum_Se^{E_\Theta(S,x)}}{Z_{+,\Theta}(x)}\\
&=\frac{Z_{+,\Theta}(x)}{Z_{+,\Theta}(x)}=1.
\end{aligned}
\]

All weights are positive, so every supported basket has nonnegative probability.
\(\square\)

The empty basket is not part of the fitted checkout law. Its unnormalized energy would
conventionally contribute one. Therefore \(Z_+=Z-1\), but the implementation does not
calculate a large \(Z\) and numerically subtract one. It directly omits the degree-zero
coefficient and sums sizes 1 through 120.

---

## 4. Exact computation of the observed energy

For an observed basket \(S_t\), direct evaluation is inexpensive because only purchased
products appear.

### 4.1 Contextual item contribution

The implementation sums

\[
L_t=\sum_{j\in S_t}b_j(x_t).
\tag{7}
\]

The same `b_at` function calculates \(b_j(x_t)\) for observed products and for products
inside the normalizer.

### 4.2 Gram pair contribution

Let

\[
m_t=\sum_{j\in S_t}\phi_j.
\tag{8}
\]

Expanding the squared norm gives

\[
\|m_t\|^2
=\sum_{j\in S_t}\|\phi_j\|^2
+2\sum_{j<k\in S_t}\phi_j^\top\phi_k.
\tag{9}
\]

Consequently the pair sum is evaluated as

\[
V_t
=\frac12\left[
\|m_t\|^2-\sum_{j\in S_t}\|\phi_j\|^2
\right].
\tag{10}
\]

This costs a sum over observed items rather than a separate calculation for every pair.
Equation (10) is an algebraic identity, not an approximation.

### 4.3 Group-count and size contributions

The implementation counts the number \(n_{tc}\) of observed products in every affinity
group and evaluates

\[
C_t=\sum_c\rho_c{n_{tc}\choose2},
\qquad
R_t=\rho_0(|S_t|).
\tag{11}
\]

The final observed energy is

\[
E_\Theta(S_t,x_t)=L_t+V_t-C_t-R_t.
\tag{12}
\]

No Smolyak rule, Monte Carlo draw or product candidate pruning is used in Eq. (12).

---

## 5. Why the normalizer cannot be enumerated directly

If \(J_x=|\mathcal A_x|\), the number of supported alternatives is

\[
\sum_{n=1}^{n_{\max}}{J_x\choose n}.
\tag{13}
\]

For a catalogue containing thousands of products, Eq. (13) is prohibitively large. The
normalizer must nevertheless include alternatives that were not purchased: excluding
them would change the probability denominator and reward the model for ignoring difficult
choices.

The computation becomes feasible in two reductions:

1. The Hubbard--Stratonovich identity converts the rank-\(r\) Gram interaction into an
   expectation over \(z\in\mathbb R^r\).
2. Conditional on \(z\), elementary-symmetric and category polynomials sum all baskets
   exactly without enumerating them.

Only after these reductions is Smolyak applied to the remaining \(r\)-dimensional
expectation.

---

## 6. Hubbard--Stratonovich reduction of the interaction

### Lemma 1: Gaussian exponential identity

For \(z\sim\mathcal N(0,I_r)\) and any \(v\in\mathbb R^r\),

\[
\mathbb E_z[e^{z^\top v}]
=e^{\|v\|^2/2}.
\tag{14}
\]

**Proof.** Write the Gaussian expectation as an integral:

\[
\mathbb E_z[e^{z^\top v}]
=(2\pi)^{-r/2}\int_{\mathbb R^r}
\exp\left\{-\frac12\|z\|^2+z^\top v\right\}\,dz.
\tag{15}
\]

Complete the square:

\[
-\frac12\|z\|^2+z^\top v
=-\frac12\|z-v\|^2+\frac12\|v\|^2.
\tag{16}
\]

The constant factor may be removed from the integral:

\[
\mathbb E_z[e^{z^\top v}]
=e^{\|v\|^2/2}(2\pi)^{-r/2}
\int_{\mathbb R^r}e^{-\|z-v\|^2/2}\,dz.
\tag{17}
\]

The remaining normalized Gaussian integral equals one, proving Eq. (14). \(\square\)

Apply Lemma 1 to

\[
v=m(S)=\sum_{j\in S}\phi_j.
\tag{18}
\]

Using Eq. (10), define the conditional product weight

\[
w_j(z,x)=
\exp\left\{
b_j(x)-\frac12\|\phi_j\|^2+z^\top\phi_j
\right\}.
\tag{19}
\]

Then

\[
\begin{aligned}
&\exp\left\{
\sum_{j\in S}b_j(x)
+\sum_{j<k\in S}\phi_j^\top\phi_k
\right\}\\
&\qquad=
\mathbb E_z\left[\prod_{j\in S}w_j(z,x)\right].
\end{aligned}
\tag{20}
\]

The product-specific pair interaction has disappeared conditional on \(z\). The group
and total-size potentials remain, but they depend only on discrete counts. This is the
structure exploited by the dynamic program.

---

## 7. Exact discrete summation at a fixed Gaussian node

### 7.1 Elementary symmetric polynomials

For product weights \(w_1,\ldots,w_m\), define

\[
e_k(w_1,\ldots,w_m)
=\sum_{1\le i_1<\cdots<i_k\le m}
w_{i_1}\cdots w_{i_k}.
\tag{21}
\]

This is exactly the total weight of every distinct size-\(k\) subset.

### Lemma 2: ESP recursion

Let \(e_k^{(q)}\) be the degree-\(k\) polynomial after the first \(q\) products have
been processed. Then

\[
e_k^{(q)}
=e_k^{(q-1)}+w_qe_{k-1}^{(q-1)},
\qquad e_0^{(q)}=1.
\tag{22}
\]

**Proof.** Every size-\(k\) subset of the first \(q\) products either excludes product
\(q\), contributing to \(e_k^{(q-1)}\), or includes it. In the second case, the remaining
\(k-1\) products form a subset of the first \(q-1\) products and product \(q\) contributes
the factor \(w_q\). The two cases are disjoint and exhaustive, giving Eq. (22).
\(\square\)

The recursion performs work over products and degrees rather than subsets. Degrees above
120 are never needed because they are outside the declared model support.

### 7.2 Affinity-group polynomials

For offered products \(\mathcal A_{xc}\) in group \(c\), define

\[
G_c(u;z,x)
=\sum_{k=0}^{\min(|\mathcal A_{xc}|,n_{\max})}
e^{-\rho_c{k\choose2}}
e_k(\{w_j(z,x):j\in\mathcal A_{xc}\})u^k.
\tag{23}
\]

The coefficient of \(u^k\) is the total conditional weight of selecting exactly \(k\)
products from group \(c\), including the group-count energy.

Multiply the group polynomials and truncate above degree 120:

\[
\prod_cG_c(u;z,x)
=\sum_{n=0}^{n_{\max}}A_n(z,x)u^n.
\tag{24}
\]

Polynomial multiplication sums over every allocation
\((n_1,\ldots,n_C)\) satisfying \(\sum_cn_c=n\). Therefore \(A_n(z,x)\) contains the
conditional weight of every basket of total size \(n\), with every product and group
choice counted exactly once.

### Theorem 1: complete-support normalizer representation

The nonempty Version-4 normalizer is

\[
\boxed{
Z_{+,\Theta}(x)
=\mathbb E_{z\sim\mathcal N(0,I_r)}
\left[
F_+(z,x)
\right],}
\tag{25}
\]

where

\[
F_+(z,x)
=\sum_{n=1}^{n_{\max}}
e^{-\rho_0(n)}A_n(z,x).
\tag{26}
\]

For fixed \(z\), Eq. (26) is evaluated exactly by Eqs. (22)--(24).

**Proof.** Substitute Eq. (20) into Eq. (2). The basket support is finite and every
summand is nonnegative, so expectation and summation may be interchanged:

\[
\begin{aligned}
Z_+(x)
&=\mathbb E_z\sum_{\substack{S\subseteq\mathcal A_x\\1\le|S|\le n_{\max}}}
\left[\prod_{j\in S}w_j(z,x)\right]
e^{-\sum_c\rho_c{n_c(S)\choose2}}
e^{-\rho_0(|S|)}.
\end{aligned}
\tag{27}
\]

Within group \(c\), Eq. (21) sums all product subsets of each size and Eq. (23) applies
the exact group-count factor. Equation (24) sums every compatible group-count allocation.
Its degree-\(n\) coefficient is therefore the complete conditional weight at total size
\(n\). Applying \(e^{-\rho_0(n)}\), omitting \(n=0\), and summing through 120 produces
Eq. (26). Taking the Gaussian expectation gives Eq. (25). \(\square\)

### Corollary 1: the additive-parent normalizer is exact

If \(\Phi=0\), every \(\phi_j=0\), so Eq. (19) becomes \(w_j=e^{b_j(x)}\), independent of
\(z\). Hence

\[
Z_{+,0}(x)=F_+(0,x)
\tag{28}
\]

and no numerical integration is required.

This is why the parent validation likelihood is called exact. Its discrete dynamic
program still covers all declared products and sizes, but it runs once per context rather
than once per Gaussian node.

---

## 8. Smolyak approximation of the remaining expectation

The integrand \(F_+(z,x)\) in Eq. (25) is positive, but it is generally not a polynomial
of bounded degree. A finite quadrature rule therefore approximates the Gaussian
expectation; it does not make it algebraically exact.

### 8.1 One-dimensional Gaussian rules

Let \(U^i\) be the probabilists' Gauss--Hermite rule having \(2i-1\) nodes. The raw
Hermite weights are divided by \(\sqrt{2\pi}\), so \(U^i\) approximates expectation under
\(N(0,1)\), not an unnormalized integral.

A dense tensor rule in \(r\) dimensions multiplies one-dimensional node counts and becomes
expensive rapidly. Smolyak combines tensor rules so that low total multi-index levels are
represented without using the full high-order tensor product.

### 8.2 Smolyak combination rule

The implemented rank-\(r\), level-\(q\) rule is

\[
A(q,r)
=\sum_{q-r+1\le|\boldsymbol i|\le q}
(-1)^{q-|\boldsymbol i|}
{r-1\choose q-|\boldsymbol i|}
\left(U^{i_1}\otimes\cdots\otimes U^{i_r}\right),
\tag{29}
\]

where every \(i_k\ge1\) and
\(|\boldsymbol i|=i_1+\cdots+i_r\). Tensor nodes that coincide are merged and weights
whose magnitude is below \(10^{-14}\) are discarded.

Applied to Eq. (25), this gives

\[
\widehat Z_{+,q}(x)
=\sum_{p=1}^{M_q(r)}a_pF_+(z_p,x),
\tag{30}
\]

where \(z_p\in\mathbb R^r\) and \(a_p\) are the merged Smolyak nodes and signed weights.

The checkpoint stores 32 columns for a stable artifact shape, but only five interaction
directions are active. `smolyak_rule` constructs five-dimensional nodes and pads the
remaining 27 coordinates with zero. The integration dimension is therefore five, not 32.

### 8.3 Why some weights are negative

Smolyak uses inclusion and exclusion of overlapping tensor rules. Negative coefficients
in Eq. (29) are intrinsic to that combination. Consequently Eq. (30) cannot be computed
by an ordinary log-sum-exp over all nodes, because log-sum-exp assumes positive terms.

Define

\[
P=\sum_{p:a_p>0}a_pF_+(z_p,x),
\qquad
N=\sum_{p:a_p<0}|a_p|F_+(z_p,x).
\tag{31}
\]

Then

\[
\widehat Z_{+,q}=P-N.
\tag{32}
\]

Both \(P\) and \(N\) are accumulated in log coordinates. If
\(L_P=\log P\), \(L_N=\log N\), and \(L_P>L_N\), then

\[
\log\widehat Z_{+,q}
=L_P+\log\left(1-e^{L_N-L_P}\right).
\tag{33}
\]

The implementation uses `log1p` for the final subtraction. It also records the logarithmic
cancellation condition

\[
\kappa_{\log}
=\log\frac{P+N}{|P-N|}.
\tag{34}
\]

Large \(\kappa_{\log}\) means that large positive and negative contributions almost
cancel, so small floating-point errors may affect the result.

### Proposition 2: fail-closed signed accumulation

If \(P>N\), Eq. (33) equals the logarithm of the signed quadrature sum in Eq. (30). If
\(P\le N\), the rule has not produced a positive partition estimate and no real-valued
log normalizer exists for that approximation.

**Proof.** For \(P>N\), factor \(P=e^{L_P}\) from Eq. (32):

\[
P-N=e^{L_P}\left(1-e^{L_N-L_P}\right)>0.
\]

Taking logarithms gives Eq. (33). If \(P=N\), the estimate is zero; if \(P<N\), it is
negative. Neither is a valid partition function. \(\square\)

The evaluator therefore raises an error when the signed estimate is nonpositive or
nonfinite. It never clamps the value to a small positive number, because doing so would
silently manufacture a likelihood.

---

## 9. The exact per-trip evaluation path

For one held-out trip, the interaction-model evaluator performs:

1. `Batcher.make` constructs the observed purchased lines and full assortment.
2. `RaggedModel.energy` evaluates Eq. (12) exactly.
3. `smolyak_rule` installs the target nodes and signed weights.
4. `RaggedModel.log_Z(..., drop_empty=True)` selects the deterministic quadrature path.
5. `sparse_prepare` caches every quantity that does not vary with \(z\).
6. At each Smolyak node, `log_f_sparse`:
   - forms the contextual weights in Eq. (19);
   - runs the log-coordinate ESP recursion;
   - applies group-count potentials;
   - multiplies group polynomials through a balanced native tree;
   - restores exact degree rescalings used for numerical stability;
   - applies \(\rho_0(n)\); and
   - directly drops the empty size.
7. `signed_log_integral` evaluates Eqs. (31)--(34).
8. The reported trip score is

   \[
   \widehat\ell_{t,q}
   =E_\Theta(S_t,x_t)-\log\widehat Z_{+,q}(x_t).
   \tag{35}
   \]

The degree shifts used inside `log_f_sparse` are exact identities. If every weight in a
trip is divided by \(e^a\), its degree-\(n\) polynomial coefficient is divided by
\(e^{na}\); adding \(na\) back in log coordinates restores the original coefficient.
These shifts prevent overflow without changing Eq. (26).

The native C++ polynomial code changes execution speed and memory layout, not the
probability law. Tests compare it with eager recursions and verify forward values and
gradients.

---

## 10. Three fidelity levels and the numerical acceptance rule

For active rank \(r=5\), the pipeline evaluates:

\[
q_{\mathrm{low}}=6,
\qquad
q_{\mathrm{target}}=7,
\qquad
q_{\mathrm{audit}}=8.
\tag{36}
\]

The lower and target rules score all 4,096 validation trips. Because the audit rule is
more expensive, it scores the first 128 trips in the same deterministic manifest.

Let

\[
d_t=\widehat\ell_{t,7}-\widehat\ell_{t,8}
\tag{37}
\]

on the audit subset. The implementation reports

\[
\bar d
=\frac1{T_a}\sum_td_t,
\qquad
\operatorname{SE}(\bar d)
=\frac{\operatorname{sd}(d_1,\ldots,d_{T_a})}{\sqrt{T_a}},
\tag{38}
\]

and forms the empirical allowance

\[
\epsilon_{\mathrm{audit}}
=|\bar d|+1.96\operatorname{SE}(\bar d).
\tag{39}
\]

This is a conservative observed discrepancy allowance. It is not a mathematical upper
bound on the exact integral: two adjacent rules could in principle miss the same feature.
The document therefore calls it a numerical audit rather than a proof of quadrature
exactness.

For paired child-versus-parent gains

\[
g_t=\widehat\ell_{t,7}^{\mathrm{child}}
-\ell_t^{\mathrm{parent}},
\tag{40}
\]

the ordinary lower 95% bound is

\[
L_g=\bar g-1.96\operatorname{SE}(\bar g).
\tag{41}
\]

Validation accepts a positive interaction likelihood gain only when

\[
L_g-\epsilon_{\mathrm{audit}}>0.
\tag{42}
\]

Pairing is important: the parent and child score exactly the same baskets, so much of the
large trip-to-trip likelihood variability cancels in \(g_t\).

The current implementation's standard error treats trip scores as observations. It does
not cluster the likelihood standard error by household. The saved per-trip file makes a
household-cluster reanalysis possible and should be used when the scientific claim
requires dependence-robust inference.

---

## 11. Current validation calculation

The accepted checkpoint and report are:

- child: `artifacts/candidate_rank1.pt`;
- child SHA-256: `9e198024b83fa748653ab9cc5fa3ddd6f5a35a01f62830c6d2516e421475e2db`;
- report: `reports/likelihood_validation.json`; and
- per-trip values: `reports/likelihood_validation_per_trip.npz`.

The locked 4,096-trip validation result is:

| Quantity | Result in nats per basket |
|---|---:|
| Exact additive parent | \(-43.714530\) |
| Rank-5 child at \(q=7\) | \(-43.687776\) |
| Paired child gain | \(0.026754\pm0.002376\) |
| Ordinary paired 95% interval | \([0.022097,0.031411]\) |
| Mean \(q=6\) minus \(q=7\) likelihood | \(0.003747\) |
| Mean \(q=7\) minus \(q=8\) likelihood on 128 trips | \(0.000254\) |
| \(q=7/q=8\) empirical 95% allowance | \(0.000353\) |
| Gain lower bound after allowance | \(0.021744\) |

The \(\pm0.002376\) is the standard error of the paired gain, not the standard deviation
of individual basket log likelihoods.

The child improvement remains positive after subtracting the observed adjacent-rule
allowance. This supports the statement that the recorded improvement is not explained by
the measured \(q=7\) versus \(q=8\) discrepancy. It does not turn the finite Smolyak rule
into an exact analytic integral.

---

## 12. Where Smolyak is and is not used

| Pipeline operation | Smolyak used? | Numerical method |
|---|---:|---|
| Exact additive training | No | Exact category/cardinality dynamic program |
| Spectral interaction-rank audit | No | Sparse pair residuals and leading eigensolver |
| Canonical interaction fitting | No | Fixed size-stratified additive-parent draw bank |
| Post-interaction household-size screen | Yes | Rank-relative size-law quadrature |
| Final child validation likelihood | **Yes** | \(q=r+2\), audited at \(q=r+3\) |
| Final child test likelihood | **Yes** | Same rank-relative rule |
| Exact additive-parent comparison | No | Exact no-Gram dynamic program |
| Locked add-one recommendation | No | Normalizer cancels analytically |
| Full basket generation | No | Interaction-tempered SMC plus invariant rejuvenation |
| Population basket-size certification | **Yes** | Lower rule screen and higher rule confirmation |

The canonical training estimator and held-out likelihood estimator are intentionally
different. Training reuses a fixed stratified basket bank to optimize a small concave
natural-parameter block efficiently. Final evaluation uses deterministic Smolyak
quadrature so that the training bank cannot certify its own approximation.

---

## 13. What is exact, approximate and empirically checked

| Statement | Status | Reason |
|---|---|---|
| Observed energy in Eq. (12) | Exact | Direct algebra on purchased products |
| H--S identity in Eq. (14) | Exact | Gaussian completing-square proof |
| ESP and category sum at fixed \(z\) | Exact | Exhaustive polynomial identities |
| Product and size support | Complete as declared | No top-product or size-20 evaluation mask |
| Additive-parent normalizer | Exact | Integrand is independent of \(z\) |
| Rank-5 child Gaussian expectation | Approximate | Finite Smolyak rule |
| Smolyak calculation | Deterministic | No random nodes or Monte Carlo sampling |
| Signed-sum positivity | Checked for every evaluated context | Nonpositive result raises an error |
| Residual quadrature error | Empirically audited | Adjacent \(q=7/q=8\) comparison |
| Exact-error bound from adjacent rules | Not established | Both finite rules could share residual error |
| Positive validation gain after allowance | Established for the recorded manifest | Paired interval minus empirical allowance is positive |

---

## 14. Measured evaluation time

### 14.1 Benchmark contract

The current validation code path was benchmarked on 8 September 2026 using:

- an Apple M5 Pro CPU;
- PyTorch 2.9.1 in float64;
- eight CPU threads;
- the accepted rank-five checkpoint;
- all 5,455 products and basket sizes 1 through 120; and
- the same deterministic validation-trip ordering used by the likelihood report.

The model, data and native polynomial extension were loaded before timing. Each case had
one untimed warm-up call, inference ran under `torch.no_grad()`, and the reported number is
the median of repeated wall-clock measurements. Consequently these are warm inference
times, not process-startup times.

### 14.2 Time for one reported likelihood

For one basket evaluated alone with the reported (q=7), 71-node Smolyak rule:

| Component | Median wall time |
|---|---:|
| Construct full-catalogue context | 1.538 ms |
| Evaluate observed basket energy | 0.416 ms |
| Compute (log\widehat Z_{+,7}(x)) | 20.411 ms |
| Complete end-to-end likelihood call | **21.989 ms** |

The repeated end-to-end range was 21.678--22.452 ms. Thus approximately 93% of the
single-basket latency is the normalizer. The observed numerator is not the bottleneck.

The end-to-end time is not required to equal the sum of separately timed medians exactly:
the rows were measured in separate repeated experiments and medians are not additive.

### 14.3 Batched validation throughput

The validation evaluator normally processes 24 contexts together. At that batch size:

| Evaluator | Nodes | Median batch time | Effective time per basket |
|---|---:|---:|---:|
| Exact additive parent | No Gaussian rule | 34.886 ms | 1.454 ms |
| Rank-5 child, low (q=6) | 11 | 96.041 ms | 4.002 ms |
| Rank-5 child, reported (q=7) | 71 | 403.395 ms | **16.808 ms** |
| Rank-5 child, audit (q=8) | 341 | 6,819.331 ms | 284.139 ms |

The reported interaction likelihood therefore runs at approximately 59.5 held-out
baskets per second after loading when evaluated in batches of 24. Calling it one basket at
a time runs at approximately 45.5 baskets per second.

### 14.4 The (q=8) audit has a different optimal batch size

The 341-node audit creates a much larger intermediate working set. Its measured scaling
was:

| (q=8) batch size | Median batch time | Effective time per basket |
|---:|---:|---:|
| 1 | 79.947 ms | 79.947 ms |
| 2 | 158.08 ms | 79.04 ms |
| 4 | 301.75 ms | **75.44 ms** |
| 8 | 622.96 ms | 77.87 ms |
| 12 | 1,593.71 ms | 132.81 ms |
| 16 | 3,335.37 ms | 208.46 ms |
| 24 | 6,819.33 ms | 284.14 ms |

This is a batching/working-memory effect, not a change in the estimator. Splitting the
same 128 audit trips into batches of four or eight evaluates the same per-context
quadrature rule and should reduce audit wall time without dropping products, sizes or
nodes. The current evaluator inherits the target-rule chunk size of 24 for the audit, so
this is a concrete remaining latency inefficiency.

Using the measured production batch costs, the current complete 4,096-trip validation
comparison is projected to spend approximately:

- 6 seconds on the exact parent;
- 16 seconds on the (q=6) child;
- 69 seconds on the reported (q=7) child; and
- 35--36 seconds on the 128-trip (q=8) audit at its present chunk size.

That is roughly 2.1 minutes of numerical scoring, plus model/data loading and report I/O.
Changing only the (q=8) audit chunk to four would reduce its projected contribution to
about 10 seconds while preserving the quadrature definition.

These measurements are machine-specific. The scientifically relevant likelihood and
quadrature levels do not change across machines, but latency depends on CPU, thread count,
memory bandwidth, compiled native extension and batch size.

---

## 15. Plain-language summary

For one held-out basket, the model first scores the products that were actually bought.
It then asks how much total score it would assign to **every other allowed basket** in the
same household, store, week, price and promotion context.

It does not list those alternatives one by one. Low-rank interaction theory introduces a
five-dimensional Gaussian variable. Once that variable is fixed, polynomial recursions
sum every supported product combination and size exactly. Smolyak supplies 71 carefully
weighted Gaussian points for the reported rank-five validation normalizer. Some weights
are negative, so positive and negative contributions are accumulated separately and the
calculation fails if their net result is not a valid positive partition function.

The resulting held-out log likelihood is

\[
\text{score of observed basket}
-\log(\text{total score of all supported baskets}).
\]

Thus Smolyak is present, but only in the final five-dimensional Gaussian expectation. It
does not restrict the evaluation to 71 products, 71 baskets, rank four, or size 20.
