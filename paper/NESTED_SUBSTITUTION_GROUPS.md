# Nested substitution groups: learning the partition during training

Status: design, verified on small exact worlds. **Deferred, 2026-09-17: the flat baseline
model is frozen.** A nested tree changes the model's energy and every computation that uses
it, so it would be a separate model variant, not an input option. ERIM evidence for a
category → type tree is in [ERIM_CATALOGUE_PARTITION_REFIT.md](ERIM_CATALOGUE_PARTITION_REFIT.md).
Date: 2026-09-17.

## 0. Summary

The energy basket model captures substitution through one penalty per product group,
−ρ_c·C(n_c, 2). Today the groups are fixed before training, and the model can only learn
substitution inside the groups it is given. On the known-truth synthetic world this term is
worth **0.286 nats per basket** (§1). The model captured 0.259 of it with merchandise-category
groups and only 0.012 with the default co-purchase groups.

This document extends the group term to a **tree of candidate groups**, each node with its
own penalty ρ_v ≥ 0, all learned in one training run. The tree is a generous menu built
before training. Training decides which groupings matter: nodes whose grouping does not
matter get ρ_v = 0. The extension:

- keeps the energy form and the **exact** normalizer (tree dynamic program);
- reduces exactly to today's model when the tree has one level;
- keeps complements in the φ interaction term, unchanged;
- is non-circular: the tree is built once from training data, the model is fitted once, and
  nothing is regrouped from model output.

Every mathematical claim below was checked numerically against brute-force enumeration
(§9); scripts are in `scripts/verification/nested_rho/`.

---

## 1. Why this is needed

Evidence from the synthetic capability test
([SYNTHETIC_CAPABILITY_VALIDATION.md](SYNTHETIC_CAPABILITY_VALIDATION.md)). All figures are
test log-likelihood per basket, removing one component without refitting.

| Model | Remove ρ (group penalty) | Remove φ (interactions) |
|---|---:|---:|
| True generating model | −0.286 | −0.040 |
| Fitted, merchandise-category groups | −0.259 | −0.036 |
| Fitted, co-purchase affinity groups | −0.012 | −0.041 |

- Substitution through ρ carries about 7× more likelihood than pairwise interactions.
- Whether the model obtains it depends almost entirely on the grouping.
- Co-purchase groups put substitutes in different groups, so ρ cannot act.
- φ cannot replace ρ for a whole group of substitutes: for any group G,
  Σ_{j<k∈G} φ_jᵀφ_k ≥ −½ Σ_{j∈G} ‖φ_j‖², so total repulsion is bounded by the (small) φ norms.

A fixed partition therefore has to be right before training, and there is no way to find
out from training whether it was. SHOPPER (Ruiz, Athey and Blei) learns relationships during
training, but through continuous vectors and an approximate likelihood (sequential choice,
variational bounds). The goal here is to let training choose the grouping while keeping our
exact joint law.

## 2. Requirements

1. **Same probability model.** P(S | x) ∝ exp E(S, x) over nonempty baskets 1 ≤ |S| ≤ nmax.
2. **Exact normalizer.** Z₊ computed exactly given the φ quadrature node, as today.
3. **Today's model as a special case.** A one-level tree must reproduce the current energy
   exactly.
4. **Training chooses the grouping** within a declared menu, in one fit.
5. **No circularity.** The menu is built from training data before fitting and is never
   rebuilt from model output.
6. **Complements stay in φ.** Groups express substitution only, so ρ_v ≥ 0.
7. **Numerical stability and bounded cost.**
8. **All derived quantities stay exact:** gradients, conditional completion, add-one scores,
   sampling and price responses.

## 3. The model

### 3.1 Candidate tree

A **candidate tree** 𝒯 is a laminar family of product sets:

- **Root:** all products.
- **Internal nodes:** candidate groups, e.g. category families, categories, sub-clusters.
- **Leaf groups:** the finest groups. A leaf group may be a single product.
- **Laminar:** any two nodes are either disjoint or nested.
- Every product belongs to exactly one leaf group, and hence to one chain of ancestors.

Let 𝒱 be the set of non-root nodes with at least two products (leaf groups included), and
let n_v(S) = |S ∩ v|.

### 3.2 Energy

$$
E(S,x)=\sum_{j\in S}b_j(x)\;+\;\sum_{j<k\in S}\phi_j^\top\phi_k\;-\;\sum_{v\in\mathcal V}\rho_v\binom{n_v(S)}{2}\;-\;\rho_0(|S|),
\qquad \rho_v\ge 0 .
$$

- **b_j(x):** unchanged (appeal, household taste, price, store, availability, …).
- **φ, ρ₀:** unchanged.
- **Current model:** a tree whose root's children are the leaf groups, so 𝒱 = leaf groups
  and ρ_v = ρ_c (verified, §9 check 3).
- **Pair cap:** the pair feature C(n, 2) keeps its existing cap at nmax. It never binds,
  because n_v ≤ |S| ≤ nmax.

### 3.3 The pair view (what the parameters mean)

Expanding C(n_v, 2) as a count of pairs,

$$
\sum_{v\in\mathcal V}\rho_v\binom{n_v(S)}{2}=\sum_{j<k\in S}P_{jk},\qquad
P_{jk}=\sum_{v\in\mathcal V:\,\{j,k\}\subseteq v}\rho_v .
$$

P_jk is the substitution penalty between j and k. It is the sum of ρ over all nodes
containing both, i.e. over the lowest common ancestor of j and k and every ancestor above
it (excluding the root). The effective pairwise effect is

$$
\gamma_{jk}=\phi_j^\top\phi_k-P_{jk}.
$$

**Modelling restriction.** Because ρ_v ≥ 0, P_jk can only grow as the common ancestor gets
finer: products in the same sub-cluster substitute at least as strongly as products that
only share a category. This matches retail intuition (two squeeze ketchups compete more than
a squeeze and a glass bottle). A pattern where a coarser group substitutes more strongly than
a finer one cannot be represented by ρ.

### 3.4 Normalizer

With the existing Hubbard–Stratonovich identity for φ,

$$
Z_+(x)=\mathbb E_{z\sim N(0,I)}\Big[\sum_{S\ne\emptyset}\prod_{j\in S}w_j(z)\prod_{v\in\mathcal V}e^{-\rho_v\binom{n_v(S)}{2}}e^{-\rho_0(|S|)}\Big],
\qquad w_j(z)=e^{\,b_j-\frac12\|\phi_j\|^2+z^\top\phi_j}.
$$

The expectation over z is computed with the existing quadrature, unchanged. Only the inner
sum changes, and §4 computes it exactly.

## 4. Exact computation: the tree dynamic program

### 4.1 Algorithm (for a fixed z)

For every node u, define the truncated generating polynomial F_u(t) = Σ_{k=0}^{nmax} F_u[k]·t^k:

1. **Leaf group L:** F_L[k] = e_k({w_j : j ∈ L}) · exp(−ρ_L C(k,2)).
   The first factor is the elementary symmetric polynomial (existing ESP code); the second
   factor applies only if |L| ≥ 2.
2. **Internal non-root node v with children c₁…c_m:**
   F_v = (F_{c₁} · … · F_{c_m} truncated at nmax) ⊙ exp(−ρ_v C(k,2)), where ⊙ multiplies
   coefficient by coefficient.
3. **Root:** F_root = product of the root's children's polynomials, with no ρ factor.
4. **Normalizer term:** Σ_{k=1}^{nmax} F_root[k]·e^{−ρ₀(k)}. The empty basket (k = 0) is
   excluded, as today.

### 4.2 Why it is exact

**Claim.** F_u[k] = Σ over subsets S of u with |S| = k of Π_{j∈S} w_j · Π_{v∈𝒱, v⊆u} exp(−ρ_v C(|S ∩ v|, 2)).

**Proof by induction over the tree.**
- **Leaf:** this is the definition of the elementary symmetric polynomial times the leaf's
  own factor.
- **Internal node:** children are disjoint and cover u, so a subset S ⊆ u splits uniquely
  into S_c = S ∩ c with Σ_c |S_c| = |S|.
  - Every node strictly below u lies inside exactly one child, so its factor depends only on
    that child's part. The product of the children's polynomials therefore enumerates all
    splits with the correct weights.
  - The node's own factor depends only on |S ∩ u| = k, i.e. only on the coefficient index,
    which is exactly what ⊙ applies. ∎

### 4.3 Truncation at nmax is exact

The degree of a product is the sum of the factors' degrees, so coefficients of degree
≤ nmax never depend on higher-degree coefficients. The node factor acts coefficient by
coefficient. Truncating before or after multiplying gives identical coefficients up to nmax
(§9 check 2: difference exactly 0).

### 4.4 Constant subtrees

A subtree containing no product with nonzero φ has a polynomial that does not depend on z.
The existing split into z-independent and φ-active categories (`A_const`) generalizes: such
subtrees are computed once per trip, not once per quadrature node.

### 4.5 Cost

- **Multiplications.** A node with m children needs m − 1 polynomial multiplications. Over
  the whole tree,

  $$
  \sum_{v\ \text{internal}}(m_v-1)=(\#\text{edges})-(\#\text{internal nodes})=\#\text{leaf groups}-1 .
  $$

  **Depth adds no multiplications.** The cost depends on the number of leaf groups, exactly
  as the current flat model depends on its number of groups. Each non-leaf node adds one
  coefficient-wise scaling, which is O(nmax).
- **Degrees.** A node's polynomial has degree at most min(|v|, nmax), so small nodes are
  cheap.
- **Estimates:**
  - ERIM with 8 categories split into about 60 leaf groups costs about 60 multiplications
    per trip and quadrature node, against 20 today.
  - The synthetic world with 8 categories and no sub-clusters costs the same as today's
    category run.
- **Leaf size cap.** Elementary symmetric polynomial cost grows with leaf size, so a cap on
  leaf size (as today's group cap) still applies. Internal nodes need no cap.

### 4.6 Stability

ρ_v ≥ 0 makes every node factor ≤ 1, so there is no attraction and no quadratic blow-up.
The current attraction floor (ρ_c ≥ −1.5) and category-reward cap
(`category_safety.py`) exist only because affinity groups drive ρ_c negative. They are not
needed for substitution trees.

## 5. Derived quantities (all exact)

### 5.1 Gradient and sufficient statistics

$$
\frac{\partial\log Z_+}{\partial\rho_v}=-\,\mathbb E\Big[\binom{n_v(S)}{2}\Big],\qquad
\frac{\partial\log P(S)}{\partial\rho_v}=-\binom{n_v(S)}{2}+\mathbb E\Big[\binom{n_v(S)}{2}\Big].
$$

C(n_v, 2) is a linear sufficient statistic, as C(n_c, 2) is today. The likelihood stays in
the same exponential-family form in ρ, and automatic differentiation through the tree
program returns these expectations exactly (§9 check 4).

### 5.2 Observed energy

The counts n_v(S) are sums of leaf-group counts over leaves under v, computed with one
sparse leaf-to-node membership matrix.

### 5.3 Conditional completion (revealed cart R)

For an unobserved remainder T with R ∪ T = S, the node potential becomes an offset:

$$
\rho_v\Big[\tbinom{f_v+k}{2}-\tbinom{f_v}{2}\Big],\qquad f_v=|R\cap v| .
$$

This is the tree version of the existing `_condition_cat_count` path. Degrees beyond
nmax − |R| stay impossible (§9 check 5).

### 5.4 Add-one increments (recommendation, Gibbs moves)

Adding product j to a basket R changes the energy by

$$
\Delta E=b_j+\phi_j^\top\!\sum_{k\in R}\phi_k-\sum_{v\ni j}\rho_v\,n_v(R)-\big[\rho_0(|R|+1)-\rho_0(|R|)\big].
$$

The current −ρ_c·(count in category) becomes a sum over j's ancestors (§9 check 6). The
same formula gives the single-product conditionals used in `interaction_particles.py` and
`tempered_block_gibbs.py`.

### 5.5 Exact sampling

Top-down backtracking through the tree program samples baskets exactly:
1. sample the size k at the root;
2. at each node, split its count among its children in proportion to the product of their
   coefficients, using prefix products over the children;
3. at leaves, sample products from the elementary symmetric polynomial as today.

The node's own factor is constant given its count, so it cancels in the split (§9 check 7).

### 5.6 Price responses

Price enters only through b_j, so all existing price calculations are unchanged. The
cross-price identity holds exactly and now includes nested substitution:

$$
\frac{\partial\log P(j\in S)}{\partial\log p_k}=-g_k\big[P(k\in S\mid j\in S)-P(k\in S)\big].
$$

## 6. Identifiability

1. **Nodes are separable.** The linear map from ρ = (ρ_v) to pair penalties (P_jk) is
   injective when every internal node has at least two children and the root is excluded.
   - For each node v, choose two products whose lowest common ancestor is v (possible with
     two children).
   - P at that pair equals the sum of ρ from v upward, so ρ_v = P(v) − P(parent(v)).
   - The map has full column rank (§9 check 8: rank 8 of 8).
2. **The root has no ρ.** Σ ρ_root C(|S|, 2) depends only on basket size and is absorbed
   exactly by ρ₀.
3. **Single-child chains are collapsed.** A node with one child contains exactly the same
   pairs as its child, so their ρ's can't be told apart (§9: rank 1 for a two-node chain).
   The tree builder must merge such chains.
4. **Gauges.** The ρ₀(1) = 0 convention, and the λ versus seasonal/store/taste gauges, are
   unchanged.
5. **Overlap with φ.** φ_jᵀφ_k is a low-rank positive semidefinite Gram term, and ρ gives
   block-constant nested penalties. They overlap only partly. The recovery test (§9) fits φ
   and ρ together and still separates them. Weak identification can arise in small samples
   (§7.2), which is why selection needs regularization and validation.

## 7. Estimation: training chooses the partition

### 7.1 Parameterization

- **Values.** ρ_v = softplus(r_v), or projected ≥ 0 after each step. The latter matches
  `project_rho_c`, with the floor changed from −1.5 to 0.
- **Sparsity.** An L1 penalty λ·Σ_v ρ_v, with λ chosen by **validation** likelihood.
- **Initialization.** ρ_v = 0 for all nodes, meaning "no substitution" until data says
  otherwise. Optionally a small positive start from the pre-training evidence; this is an
  initialization, not a constraint.

### 7.2 Why regularization and a relaxed refit are both needed (verified, §9 checks 9–10)

**Without the penalty,** finite samples let sub-cluster decoys absorb part of a true
category's ρ. With 150 baskets per context, a true uniform ρ = 0.9 was split into
0.59 (category) + 0.64/0.67 (sub-clusters). With 4× and 16× more data, the unpenalized fit
recovers the truth on its own (0.88, 0.90; decoys 0). The split is a small-sample effect,
not a bias.

**With the penalty,** decoys go exactly to zero at every sample size, but the true ρ's shrink:

| λ | Result (true ρ: 0.9 / 1.2 / 0.6) |
|---|---|
| 0.002 | 0.84 / 1.09 / 0.54 |
| 0.008 | 0.78 / **0.49** / 0.44 (a real group is mostly suppressed, and test fit worsens) |

**Relaxed refit, within the same training procedure:**
1. fit with the validation-selected λ;
2. collapse nodes with ρ_v = 0 into their parent (pruning, never adding or moving products);
3. refit without the penalty on the pruned tree.

This recovered 0.915 / 1.229 / 0.588 at the smallest sample size, identical to fitting the
true partition directly, with the best held-out likelihood.

Step 2 only removes groupings the fit itself rejected. The candidate menu is never rebuilt
or extended from model output.

### 7.3 Output: the effective partition

- **Active nodes:** ρ_v > 0 after the relaxed refit, each with its learned strength.
- **Effective pair penalties** P_jk, and substitute sets read off the active nodes.
- **Recorded** in the checkpoint with the candidate-tree hash, λ and validation scores.

### 7.4 Where it enters the pipeline

- **Additive stage** (`fit_exact_additive.py`): fits λ, taste, price and so on, plus ρ_v with
  the exact tree program, then selects λ, prunes and refits.
- **Interaction stage** (`fit_stratified_natural_interactions.py`): it currently adds a
  natural-parameter correction Δρ_c. That becomes Δρ_v on the pruned tree's node statistics,
  with ρ_v + Δρ_v ≥ 0 (a bound-constrained solve).
- **All later stages** use the fitted tree through the same model methods.

## 8. Building the candidate tree (before training, model-free)

The tree is a **menu, not a decision**. Training can only select groupings present in it, so
**coverage matters most**. In the recovery test, a tree missing the true groups lost about
0.04 nats per basket, almost as much as having no groups (§9 check 11).

### 8.1 Inputs

- Training baskets only.
- The availability panel: pairs are compared only on trips where both products were stocked.
- Optional catalogue categories.

### 8.2 Evidence

Both computed from training data without any model:

**(a) Within-household co-purchase shortfall.** Same households, different trips, adjusted
for trip size and availability:

$$
\text{score}_{jk}=\log\frac{O_{jk}+\kappa}{E_{jk}+\kappa},\qquad
E_{jk}=\sum_h\sum_t p^{j}_{ht}p^{k}_{ht},\quad p^{j}_{ht}=\min\!\Big(1,\;a^{j}_h\frac{s_t}{\textstyle\sum_{t'} s_{t'}}\Big).
$$

**(b) Exchangeability** (from SHOPPER, computed empirically). Each product's companion
profile is its size- and availability-adjusted lift with every other product, shrunk toward 0.
Exchangeability is a symmetrized divergence between two profiles (Jensen–Shannon), excluding
the pair itself. It uses a product's association with all other products, so it works for
products too rare for pairwise evidence.

**Link rules:**

| Exchangeable? | Co-purchase evidence | Link |
|---|---|---|
| yes | shortfall | strong can-link |
| yes | none (rare pair) | weak can-link |
| yes | excess | cannot-link (complements) |
| no | excess | cannot-link |
| no | shortfall or none | no link |

### 8.3 Tree construction

1. **Top levels:** catalogue categories, if trusted. Optionally category families above
   them. Otherwise take the top levels from the evidence dendrogram.
2. **Within each category:** agglomerative clustering on combined substitution evidence
   (average linkage), respecting cannot-links.
3. **Cut** the dendrogram at 2–3 declared evidence levels to form sub-cluster nodes. This is
   **generous**: extra levels are cheap, since training sets unneeded ones to zero.
4. **Leaf groups:** the finest sub-clusters, capped in size. Products with no evidence stay
   as single-product leaves directly under their category.
5. **Clean up:** collapse single-child chains, and give the root no ρ.
6. **Record** the tree, its settings and evidence summaries in the bundle; hash them into
   the data fingerprint.

## 9. Verification

Small worlds with 12 products and nmax = 5 (1,585 baskets) allow brute-force enumeration of
every basket.

**Exactness** (`verify_exactness.py`, `artifacts/nested_rho_verification/exactness.json`):

| # | Check | Result |
|---|---|---|
| 1 | Tree program for fixed z vs enumeration (3-level tree, rank-2 φ, an unavailable product) | 6.7e-16 |
| 1b | Full normalizer (Gauss–Hermite over z, 40×40) vs enumeration of energies | 9.8e-15 |
| 2 | Truncation at nmax commutes | 0.0 |
| 3 | One-level tree vs current flat ρ_c energy | 2.7e-15 |
| 4 | ∂log Z/∂ρ_v vs −E[C(n_v,2)] (finite differences) | 4.8e-09 |
| 5 | Conditional completion offsets vs enumeration | 3.6e-15 |
| 6 | Add-one increment formula | 2.7e-15 |
| 8 | Rank of ρ → pair-penalty map (8 non-root nodes) | 8 of 8; two-node chain: 1 |

**Exact sampling** (`verify_sampler.py`):

| # | Check | Result |
|---|---|---|
| 7 | Top-down tree sampler: total variation to the exact law over 200,000 draws | 0.0265 (pure sampling-noise reference 0.0268) |

**Recovery** (`verify_recovery.py`). Truth: three substitute categories (ρ = 0.9, 1.2, 0.6),
rank-1 cross-category complements, 240 contexts. The candidate tree contains the true
categories plus decoys: a family node above two categories and three sub-clusters inside
categories. φ, λ, ρ₀ and ρ_v are all fitted by exact maximum likelihood.

| Fit (test log-likelihood) | 150 / context | 600 / context | 2,400 / context |
|---|---|---|---|
| Oracle | −5.51511 | −5.52059 | −5.52987 |
| Tree, no penalty | −5.51571; ρ cat0 0.59, cl00 0.64, cl01 0.67 | −5.52080; decoys ≈ 0 | −5.52991; truth recovered |
| Tree, λ = 0.002 | −5.51632; decoys 0; true ρ shrunk | −5.52094 | −5.53025 |
| **Tree, λ = 0.002 then relaxed refit** | **−5.51556**; ρ 0.915 / 1.229 / 0.588 | −5.52079 | −5.52992 |
| True flat partition (reference) | −5.51556 | −5.52079 | −5.52992 |
| Tree missing the true groups (check 11) | −5.55723 | −5.55805 | −5.56444 |
| No groups (φ only) | −5.56873 | −5.56775 | −5.57472 |

**Conclusions:**
- Training on the candidate tree selects the true groups.
- The relaxed refit matches fitting the true partition directly, at every sample size.
- A tree without the true groups gives up nearly all of the benefit.

## 10. Code impact (for implementation)

| Component | Change |
|---|---|
| `prepare_model_bundle.py` / partition builder | Evidence statistics (§8.2), tree construction (§8.3), tree file + manifest, fingerprint entry |
| `data.py` (ragged index) | Leaf groups become rows (as today); add node arrays (parent, children, leaf→node membership) and per-store ordering of leaves by subtree |
| `ragged.py` `RaggedModel` | ρ_leaf (current `rho_c`) + ρ_node for internal nodes; tree-structured product replacing the flat `poly_tree` over categories; node factors; constant-subtree split; `energy()` node penalties; conditional offsets per node; projection ρ ≥ 0 |
| `fit_exact_additive.py` | L1 on ρ, λ grid on validation, pruning, relaxed refit; skip `category_safety` for repulsive trees |
| `stratified_natural.py`, `fit_stratified_natural_interactions.py` | Node sufficient statistics; bound-constrained Δρ_v |
| `interaction_particles.py`, `tempered_block_gibbs.py` | Add-one logits over ancestors (§5.4); top-down sampling (§5.5) |
| `fit.py` (`rec_eval`, size initialization), `eval_smolyak_rank8_mrr.py`, `basket_counterfactual_query.py` | Ancestor sums in place of −ρ_c·count |
| `audit_interaction_embeddings.py`, `diagnose_size_phase.py` | γ_jk = φ_jᵀφ_k − P_jk; node-level ρ comparisons |
| Checkpoint / API | Tree hash, active nodes and ρ recorded; conditional completion through the same model path |
| Tests | Port §9 checks 1–8 against the production implementation; recovery test as a slow integration test |

Backward compatibility: a bundle without a tree file is a one-level tree, so existing
checkpoints and the Dunnhumby and ERIM runs are unchanged.

## 11. Risks and open questions

1. **Coverage.** Groupings absent from the candidate tree cannot be learned. The builder
   should be generous, and its coverage should be checked on synthetic truth before real use.
2. **Monotone nesting** (§3.3). Finer groups cannot substitute less than coarser ones.
   Probably realistic, but an assumption.
3. **Uneven strength inside a leaf.** All pairs in a leaf share one penalty. Unevenness needs
   finer leaves or falls back on φ.
4. **Complements inside a node.** With ρ ≥ 0 a node cannot reward co-purchase. Cannot-links
   in the builder keep complements apart, and any that slip in are left to φ.
5. **Choice of λ and the refit.** Selection is a two-step procedure within training. The λ
   grid and pruning threshold must be declared before fitting.
6. **Scale.** For Dunnhumby-size catalogues, the number of leaf groups (not depth) drives
   cost, so leaves must stay reasonably coarse.
7. **Interaction-stage solve.** The bound-constrained natural-parameter step is new and needs
   its own verification.
8. **Mixing with attractive groups.** This design assumes substitution trees (ρ ≥ 0). The
   existing attractive affinity mode would stay a separate, flat option.

## 12. Proposed synthetic test (criteria fixed before running)

On the known-truth synthetic world, extended with (i) a second level of true nested
substitution inside some categories and (ii) rare products:

1. **Tree coverage.** All true substitute sets appear as nodes; report which are missing.
2. **One training run:**
   - decoy nodes ρ = 0;
   - true node ρ within ±0.15 of truth after the refit;
   - adjusted Rand index of the effective partition vs truth ≥ 0.9;
   - effective pair-penalty correlation with truth ≥ 0.95.
3. **Likelihood.** Held-out log-likelihood at least that of the category-partition run
   (−12.80 on the current world) and closer to the oracle.
4. **Cross-price.** Signs and magnitudes of complement and substitute cross-price effects
   vs the oracle.
5. **Promotion policy.** Action-value correlation with the oracle ≥ the category run's 0.98;
   the best action matches the oracle.

## Reproduce the verification

```bash
python scripts/verification/nested_rho/verify_exactness.py
python scripts/verification/nested_rho/verify_sampler.py
python scripts/verification/nested_rho/verify_recovery.py --per-context 150
python scripts/verification/nested_rho/verify_recovery.py --per-context 600
python scripts/verification/nested_rho/verify_recovery.py --per-context 2400
```

Outputs: `artifacts/nested_rho_verification/`.
