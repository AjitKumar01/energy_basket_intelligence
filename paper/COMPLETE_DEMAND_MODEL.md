# A joint complete-demand extension of the Version-4 basket model

## 1. Scope

The current real-data model describes a nonempty basket conditional on a recorded
checkout. This document specifies a research extension for data that also record
household opportunities, no-visit periods, store visits without purchase, feasible
stores, store-time assortments, and prices at chosen and unchosen stores.

The extension does not replace the Version-4 basket energy. It places that law inside one
larger probability distribution over no visit, empty store visits, and nonempty store
baskets. The basket partition function becomes an inclusive value linking product-level
utility to store visitation and purchase activation.

The synthetic implementation is in
`scripts/version4/audit_synthetic_complete_demand.py`; the replicated experiment driver
is `scripts/run_complete_demand_experiment.py`.

---

## 2. Complete outcome space

For household \(h\) and decision period \(t\), let

\[
Y_{ht}\in\{0,(s,\varnothing),(s,S)\}.
\tag{1}
\]

The alternatives mean:

- \(0\): no visit to a modeled store;
- \((s,\varnothing)\): a visit to store \(s\) with no purchase; and
- \((s,S)\): a visit to store \(s\) with nonempty basket \(S\).

Let \(\mathcal G_{ht}\) be the feasible stores for that household-period and
\(\mathcal A_{hst}\) the products offered at store \(s\). Feasibility, assortment, price,
and promotion information must be available for chosen and unchosen stores. Otherwise
the joint denominator cannot be constructed correctly.

---

## 3. The unchanged conditional basket law

At store \(s\), the Version-4 energy remains

\[
\begin{aligned}
E_\Theta(S,x_{hst})
={}&\sum_{j\in S}b_j(x_{hst})
+\sum_{j<k\in S}\phi_j^\top\phi_k\\
&-\sum_c\rho_c{n_c(S)\choose2}-\rho_0(|S|).
\end{aligned}
\tag{2}
\]

Its nonempty partition function is

\[
Z_{+,hst}
=
\sum_{\substack{S\subseteq\mathcal A_{hst}\\1\le |S|\le n_{\max}}}
e^{E_\Theta(S,x_{hst})}.
\tag{3}
\]

Conditional on store \(s\) and a nonempty purchase,

\[
P_\Theta(S\mid h,s,t,\text{purchase})
=\frac{e^{E_\Theta(S,x_{hst})}}{Z_{+,hst}}.
\tag{4}
\]

The existing Hubbard--Stratonovich and polynomial theorem continues to compute Eq. (3).
No new basket enumeration is introduced by the complete-demand extension.

---

## 4. Purchase activation within a store visit

Introduce a scalar \(\chi_{hst}\), added once to every nonempty basket. Define the
within-store energy

\[
E^*_{hst}(A)=
\begin{cases}
0,&A=\varnothing,\\
\chi_{hst}+E_\Theta(A,x_{hst}),&A\ne\varnothing.
\end{cases}
\tag{5}
\]

The store-level partition function including no purchase is

\[
\widetilde Z_{hst}=1+e^{\chi_{hst}}Z_{+,hst}.
\tag{6}
\]

It follows that

\[
P(\text{purchase}\mid\text{visit }s)
=\frac{e^{\chi_{hst}}Z_{+,hst}}
{1+e^{\chi_{hst}}Z_{+,hst}}.
\tag{7}
\]

Conditioning Eq. (5) on a nonempty outcome cancels \(\chi_{hst}\) and recovers Eq. (4)
exactly. Thus \(\chi\) estimates the extensive purchase-activation margin without
altering relative probabilities among nonempty baskets.

---

## 5. Store access and the inclusive-value link

Let \(g_{hst}\) describe the direct convenience of visiting store \(s\): distance,
opening status, household--store affinity, and other access variables. Product prices
and assortments need not be duplicated freely in \(g\), because their value is already
summarized by \(\log\widetilde Z_{hst}\).

Set no-visit utility to zero for identification and define

\[
D_{ht}
=1+\sum_{s\in\mathcal G_{ht}}e^{g_{hst}}\widetilde Z_{hst}.
\tag{8}
\]

The joint law is

\[
P(Y_{ht}=0\mid X_{ht})=\frac1{D_{ht}},
\tag{9}
\]

\[
P(Y_{ht}=(s,\varnothing)\mid X_{ht})
=\frac{e^{g_{hst}}}{D_{ht}},
\tag{10}
\]

and

\[
\boxed{
P(Y_{ht}=(s,S)\mid X_{ht})
=\frac{e^{g_{hst}+\chi_{hst}+E_\Theta(S,x_{hst})}}{D_{ht}}.}
\tag{11}
\]

Summing Eq. (11) over nonempty baskets and combining it with Eq. (10) gives

\[
P(\text{visit }s\mid X_{ht})
=\frac{e^{g_{hst}}[1+e^{\chi_{hst}}Z_{+,hst}]}{D_{ht}}.
\tag{12}
\]

Every parameter inside the basket energy therefore affects visit and purchase
probability through \(Z_+\). A price or promotion can change composition, size, purchase
activation, store choice, and total retailer visitation in one coherent distribution.

### Proposition 1: normalization

Equations (9)--(11) sum to one.

**Proof.** Sum Eq. (11) over all nonempty baskets at store \(s\). By Eq. (3), this gives
\(e^{g_{hst}+\chi_{hst}}Z_{+,hst}/D_{ht}\). Add the empty-visit probability in Eq. (10)
to obtain \(e^{g_{hst}}\widetilde Z_{hst}/D_{ht}\). Summing over stores and adding the
outside probability \(1/D_{ht}\) produces the numerator \(D_{ht}\). \(\square\)

### Proposition 2: conditional Version-4 recovery

Conditioning the joint law on store \(s\) and a nonempty purchase gives Eq. (4).

**Proof.** Divide Eq. (11) by its sum over nonempty \(S\). The common factors
\(e^{g_{hst}+\chi_{hst}}/D_{ht}\) cancel, leaving
\(e^{E_\Theta(S,x)}/Z_+\). \(\square\)

---

## 6. One joint training objective

The final objective is

\[
\mathcal L_{\mathrm{joint}}
=\sum_{h,t}\log P(Y_{ht}\mid X_{ht}).
\tag{13}
\]

The observation contributions are

\[
\ell_{ht}=
\begin{cases}
-\log D_{ht},&Y_{ht}=0,\\
g_{hst}-\log D_{ht},&Y_{ht}=(s,\varnothing),\\
g_{hst}+\chi_{hst}+E_\Theta(S,x_{hst})-\log D_{ht},
&Y_{ht}=(s,S).
\end{cases}
\tag{14}
\]

No-visit and empty-visit observations now send gradients into basket parameters through
\(D_{ht}\). This is the formal sense in which the visit model is linked rather than an
independent classifier.

When energy, store utility, and activation utility are linear in natural parameters,
Eq. (14) is a selected linear energy minus a log-sum-exp normalizer. It is concave in
those natural parameters over a convex constraint set. The raw factorization
\(K=\Phi\Phi^\top\) remains nonconvex, so the existing stable-basis parameterization
\(K=UC_{\mathrm{int}}U^\top\), \(C_{\mathrm{int}}\succeq0\), remains appropriate.

The recommended training order is:

1. construct and audit the complete household-period opportunity panel;
2. conditionally pretrain Version-4 as initialization;
3. initialize access and activation blocks;
4. jointly refine the exact additive law on Eq. (13);
5. recompute the interaction score from the joint likelihood;
6. fit the stable natural interaction block against joint likelihood;
7. recalibrate household size against joint likelihood; and
8. accept models using locked joint validation evidence.

The conditional pretraining stage is not the final objective. The accepted model must be
selected by the joint probability in Eq. (13).

---

## 7. Unconditional incidence and unit demand

The conditional incidence of product \(j\) in a nonempty basket is

\[
I_{hjst}
=P(j\in S\mid h,s,t,\text{purchase})
=\frac{\partial\log Z_{+,hst}}{\partial b_j(x_{hst})}.
\tag{15}
\]

The unconditional probability of purchasing product \(j\) in opportunity \((h,t)\) is

\[
\boxed{
P(j\text{ purchased}\mid X_{ht})
=\sum_s
P(\text{visit }s\mid X_{ht})
P(\text{purchase}\mid\text{visit }s,X_{ht})
I_{hjst}.}
\tag{16}
\]

Expected distinct-product demand over a horizon is the sum of Eq. (16) over households
and periods. Price counterfactuals automatically contain both the extensive visit margin
and the conditional basket margin.

If physical units are required, attach a positive-count distribution only after item
inclusion, for example a shifted negative binomial. Its mean may reuse household taste,
price sensitivity, promotion effects, and interaction summaries with head-specific
scales. Then

\[
\mathbb E[Q_j\mid X_{ht}]
=\sum_s P(\text{visit }s)
P(j\in S\mid\text{visit }s)
\mathbb E[Q_j\mid j\in S,s,X_{ht}].
\tag{17}
\]

The quantity likelihood is added to Eq. (13) for purchased lines. Inventory, stockouts,
cost, and causal promotion response still require their own observed variables.

---

## 8. Sampling from the complete model

For a household-period:

1. Compute \(Z_{+,hst}\) and \(\widetilde Z_{hst}\) for every feasible store.
2. Draw no visit or a store using Eqs. (9) and (12).
3. Given a store, draw empty versus nonempty using Eq. (7).
4. Given a purchase, use the existing Version-4 generator: latent interaction state,
   total size, affinity-group counts, and actual products.
5. If a quantity head is fitted, draw positive units for selected products.

Steps 1--3 add unconditional demand. Step 4 remains the original Version-4 sampling
problem and retains its exact conditional and SMC theory.

---

## 9. Scalability

The complete-demand denominator adds a sum over feasible stores, not a new sum over
baskets. In the additive case, the leading cost per opportunity is approximately

\[
O\!\left(
|\mathcal G_{ht}|
[\text{unique store contexts}]
[J_xn_{\max}+C_xn_{\max}^2]
\right).
\tag{18}
\]

Repeated household-periods can reuse a partition value when segment, store, assortment,
price, promotion, and other basket context are identical. Direct access utilities remain
cheap per household-store pair.

The scalable real-data implementation must:

- use actual feasible-store sets and actual store assortments;
- cache repeated store-time partition values where context permits;
- minibatch all opportunity periods, including sampled zero periods with correct weights;
- retain high-order Smolyak as an audit rather than an ordinary training operation; and
- parallelize across store contexts, periods, and SMC particles.

Interaction rank remains the principal integration limit. Adding the outer visit law does
not remove or worsen the intrinsic Smolyak node growth except by requiring \(Z_+\) at more
store contexts.

---

## 10. Identification and evidence requirements

The following restrictions are essential:

1. no-visit utility is fixed to zero;
2. empty-within-store energy is fixed to zero;
3. \(\chi\) is the relative activation of any nonempty purchase;
4. direct access utility avoids unrestricted duplication of basket price and assortment;
5. store and factor gauges are centered;
6. opportunity periods and feasible stores are defined before observing the choice;
7. train, validation, and test periods are chronological; and
8. an unsupported interaction child falls back exactly to its linked additive parent.

The independent-hurdle ablation removes \(Z_+\) from visit and purchase heads. It is a
necessary baseline: the linked model is justified only if it improves held-out joint
likelihood, probability calibration, and counterfactual demand recovery.

Synthetic evidence validates algebra, optimization, and recovery under known truth. It
does not prove that observational retailer prices or promotions have causal effects.
