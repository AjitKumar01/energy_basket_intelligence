# Substitution-evidence partition: method, scalability and checks

Date: 2026-09-17

## 1. Purpose

Build product groups once, from training baskets only, so that each group holds products
that are bought *instead of* each other. The model is not changed. The partition is an
input: products in one group share the exact within-group penalty ρ_c. The option is
`affinity.partition = "substitution_evidence"` in a dataset config, implemented in
`scripts/version4/evidence_partition.py`.

Out of scope: the design's sign constraint ρ_c ≥ 0 is a model change, and the baseline
model is frozen. Negative fitted penalties are reported, not prevented.

## 2. Method

### 2.1 Pair evidence

Notation, for a household h with training trips t:
- **s_t:** basket size.
- **y_tj:** 1 if product j is bought on trip t.
- **st_tj:** 1 if j was stocked at the trip's store and week (availability panel; all ones
  without one).
- **Household counts:** a_hj = Σ_t y_tj st_tj, and S_hj = Σ_t s_t st_tj.

$$
p_{tj}=st_{tj}\,\min\!\Big(1,\;a_{hj}\,\frac{s_t}{S_{hj}}\Big),\qquad
O_{jk}=\sum_t y_{tj}y_{tk}\,st_{tj}st_{tk},\qquad
E_{jk}=\sum_t p_{tj}\,p_{tk},
$$

$$
\text{score}_{jk}=\log\frac{O_{jk}+\kappa}{E_{jk}+\kappa}.
$$

- **κ** is the Gamma–Poisson maximum-likelihood shrinkage: O ~ Poisson(E·θ), θ ~ Gamma(κ, κ),
  over all evaluated pairs with E > 0.
- **Evidence:** a pair has evidence when E_jk ≥ `minimum_expected_cooccurrence`. Pairs without
  evidence count as score 0.
- **Can-link:** score ≤ `evidence_threshold` (< 0).
- **Cannot-link:** score ≥ `complement_threshold` (> 0), with evidence.

### 2.2 Groups

The partition maximises the objective Σ_groups Σ_{j<k in group} (−score_jk) subject to:
- group size ≤ `maximum_group_size`;
- average within-group score ≤ `evidence_threshold`;
- no cannot-link pair inside a group;
- optionally, groups stay inside one catalogue category (`category_boundary`).

The procedure is:
1. **Greedy merging.** Start from single products. Repeatedly merge the feasible pair of
   groups with the most negative average between-group score. Ties break by the smallest
   product ids.
2. **Local moves.** Move single products, to another group or out on their own, while the
   objective increases and the constraints hold.
3. **Leftovers.** Products left alone stay single-product groups (`singleton`), or are pooled
   into one leftover group per category (`residual_per_category`). Leftover groups are
   marked in the manifest, and they are exempt from the size and homogeneity constraints.

## 3. Scalability

### 3.1 Why the first implementation did not scale

Sizes (training data):

| Dataset | Households H | Trips T | Products J | Categories | Largest category | Distinct products per household (mean) |
|---|---:|---:|---:|---:|---:|---:|
| ERIM | 3,862 | 86,036 | 464 | 8 | 108 | small |
| Dunnhumby | 1,920 | 160,007 | 5,455 | 187 | 236 | 305 |

The first implementation had three costs:

| Step | Computation | Cost | Dunnhumby, category boundary | Dunnhumby, no boundary |
|---|---|---|---|---|
| Evidence | dense p over **all trips × block products**, then PᵀP | Σ_blocks T·n_b² | ≈ 1e10 flops, 187 passes over all trips | T·J² ≈ 4.8e12 flops, J² per trip chunk (hours) |
| Merging | recompute every group-to-group sum each merge (M·s·Mᵀ), up to n merges | Σ_blocks n_b⁴ | ≈ 3e9 for the largest block | J⁴ ≈ 9e14 (infeasible) |
| Local moves | per product, per group: dense sums and an O(size²) feasibility slice | ≥ n²·size per pass | slow | infeasible |

### 3.2 Evidence: household factorisation (exact)

**Claim.** E_jk = Σ_h P_hᵀP_h restricted to B_h, the products household h bought. Here P_h
is the (h's trips) × B_h matrix of p_tj.

**Proof.**
- If a_hj = 0, then p_tj = st_tj·min(1, 0) = 0 for every trip of h.
- So the rows of household h contribute to E_jk only when j, k ∈ B_h.
- Summing p_tj p_tk over h's trips is exactly (P_hᵀP_h)_jk. ∎

The same holds for O, since y_tj = 1 implies j ∈ B_h. With a category boundary, B_h splits
into per-category sets B_hc.

**Cost.**

| Mode | Formula | Dunnhumby |
|---|---|---|
| Category boundary | Σ_h trips_h · Σ_c \|B_hc\|² | **6.7e8** flops |
| No boundary | Σ_h trips_h · \|B_h\|² | **3.3e10** flops (tens of seconds with BLAS) |

**Memory.**
- One dense n_b × n_b matrix per block, for E and O.
- Without a boundary that is J² = 3e7 entries for Dunnhumby (480 MB for O and E together).
- A declared guard, `maximum_dense_pairs` (default 5e7 entries per block), fails with a
  clear message beyond that. Larger catalogues would need sparse pair accumulation or a
  boundary.

### 3.3 Merging: sparse average linkage with a lazy heap (same merges)

Let G_ab = Σ_{j∈a, k∈b} s_jk, where s = score with no-evidence pairs set to 0. Let W_a be
a's within-group sum, and K_ab the number of cannot-link pairs between a and b.

1. **Sparsity.** The average G_ab/(n_a n_b) is negative only if a and b share at least one
   negative evidence edge. The candidate pairs are therefore the group pairs joined by
   negative edges, and only those enter the heap. G is kept for group pairs joined by any
   evidence edge, positive or negative. That way merged sums stay exact when a merge brings
   a negative neighbour together with a positive one.
2. **Additivity (Lance–Williams, sum form).** After merging a and b into u:
   - G_uc = G_ac + G_bc
   - K_uc = K_ac + K_bc
   - W_u = W_a + W_b + G_ab
   - n_u = n_a + n_b

   Each merge costs O(deg a + deg b) dictionary updates.
3. **Feasibility of a candidate (a, b) is O(1):**
   - n_a + n_b ≤ cap;
   - K_ab = 0;
   - (W_a + W_b + G_ab) / C(n_a + n_b, 2) ≤ threshold.

   The within-group cannot-link counts are already 0 for existing groups.
4. **Lazy heap equivalence.** The heap holds (average, min id a, min id b, versions).
   - A popped entry whose groups changed since it was pushed is stale and skipped. Current
     pairs were pushed after every merge.
   - A popped entry that is current and infeasible stays infeasible: its groups are
     unchanged, and all three conditions depend only on the two groups.
   - Changed groups get new entries.
   - So the first current, feasible entry popped is exactly the pair the first
     implementation chose: the most negative feasible average, with the same tie-break.

   Averages come from incremental sums, so round-off differs by about 1e-15. Exact ties are
   broken by ids.
5. **Cost.** O((E⁻ + Σ merge degrees) · log E⁻), where E⁻ is the number of negative
   evidence edges.

### 3.4 Local moves: sparse product-to-group sums

For each product j, keep L[j, g] = Σ_{k∈g, k≠j} s_jk (only groups holding an evidence
neighbour of j) and C[j, g], the number of cannot-link neighbours of j in g.

**Moves.** For product j in group A:
- **Into group B:** gain = L[j, A] − L[j, B]. Feasible if:
  - n_B + 1 ≤ cap;
  - C[j, B] = 0;
  - (W_B + L[j, B]) / C(n_B + 1, 2) ≤ threshold;
  - the source remainder A∖j satisfies (W_A − L[j, A]) / C(n_A − 1, 2) ≤ threshold when it
    keeps two or more products.
- **Out on its own:** gain = L[j, A]. This move is new, and fixes audit item 9.

**Update after moving j.** For every evidence neighbour k of j: L[k, A] −= s_kj,
L[k, B] += s_kj, and C likewise. Update W and n for A and B. Cost O(deg j).

**Termination.** Each accepted move raises the objective by more than 1e-9, and there are
finitely many partitions. Passes are also capped.

**Cost.** O(Σ_j deg_j) = O(E) per pass.

### 3.5 Resulting complexity

| Step | Category boundary | No boundary |
|---|---|---|
| Evidence | Σ_h trips_h Σ_c \|B_hc\|² (6.7e8, Dunnhumby) | Σ_h trips_h \|B_h\|² (3.3e10) |
| κ | O(pairs) per likelihood evaluation (2e5 pairs) | O(pairs) (1.5e7 pairs) |
| Merging and moves | O(E log E), E = evidence edges | same |
| Memory | Σ_blocks n_b² | J² (guarded) |

Evidence dominates, and it is linear in trips.

## 4. Verification plan

1. **Old versus new implementation**, on ERIM and the synthetic world:
   - O and E equal to ≤ 1e-9 relative;
   - κ equal;
   - merged partitions identical before local moves.
2. **Unit tests:** sparse merging against a brute-force greedy reference on random score
   matrices; move gains against recomputed objectives; the singleton move; the dense-pairs
   guard.
3. **Timing** on Dunnhumby, with and without the category boundary.

## 5. Results

### 5.1 Implementation matches the first version

Reference: the first implementation, run on the same inputs.

| Check | Result |
|---|---|
| O, every ERIM and synthetic category block | identical |
| E, maximum relative difference | 3e-15 (synthetic), 1e-14 (ERIM) |
| κ | identical |
| Merged partitions (no moves) on 56 cases: all ERIM and synthetic category blocks plus 40 random score matrices | 56 of 56 identical |
| Final partitions after moves | 56 of 56 identical; objective never lower |
| ERIM evidence bundles rebuilt with the new code | `items_affinity.parquet` byte-identical for both leftover rules |
| Pre-training check on ERIM (validation shortfalls, stability) | identical to the first implementation |

Unit tests (`tests/test_evidence_partition.py`):
- sparse merging equals a dense brute-force greedy reference on 30 random cases;
- after moves, partitions are feasible and no single move to another group, or out on its
  own, raises the objective;
- household evidence equals the trip-level formula, with stocking;
- the dense-pairs guard fails with a clear message.

### 5.2 Timing on Dunnhumby (5,455 products, 160,007 training trips)

| Mode | Seconds | Peak memory | Evidence pairs | Evidence groups | Single-product groups |
|---|---:|---:|---:|---:|---:|
| Category boundary (187 commodities) | 4.1 | | 216,146 evaluated; 8,845 with evidence | 304 | 4,114 |
| No boundary | 16.3 | 2.4 GB | 14,875,785 evaluated; 299,423 with evidence | 409 | 4,005 |

The first implementation would need about 1e10 operations with the boundary and more than
1e14 without it, which is hours to infeasible.

### 5.3 Pre-training checks (`scripts/verification/partition/check_evidence_partition.py`)

The criteria were written after a smoke build had shown the synthetic partition:
- synthetic same-group precision ≥ 0.9;
- no true complement pair grouped;
- stability ARI ≥ 0.8;
- on real data, validation within-group shortfall below the same-category, other-group
  shortfall.

| Dataset / settings | Groups (evidence / other) | Truth: precision, recall, complements grouped, ARI | Validation shortfall: within / same category, other group / cross-category | Stability ARI |
|---|---|---|---|---|
| Synthetic, category boundary | 8 / 0 | 0.994, 1.00, 0, 1.00 | −0.63 / no pairs / −0.14 | 0.998 |
| Synthetic, no boundary | 8 / 0 | 0.994, 1.00, 0, 1.00 | −0.63 / no pairs / −0.14 | 0.97 |
| ERIM, singleton leftovers | 18 / 240 singletons | — | −1.23 / −0.55 / −0.39 | **0.52** (fails) |
| ERIM, residual per category | 18 / 8 residual | — | −1.12 / −0.58 / −0.39 | **0.75** (fails) |

**Reading.**
- **Synthetic world:** the method recovers the true substitute sets, even without a
  catalogue boundary. That world is easy: substitutes are whole categories of 15 products,
  and baskets hold 3.5 products.
- **ERIM:** baskets hold 1.7 products, so only 982 of 17,367 within-category pairs reach
  the expected co-occurrence floor. Most products get no evidence, and the groups are
  unstable across household resamples.
- **Dunnhumby:** most products are also left without evidence (about 4,100 of 5,455), even
  though the catalogue is large and baskets are big. Evidence at the pair level is
  concentrated in popular products.

The ERIM model refits with both leftover rules were stopped before completion, so that
the scalability fixes could come first. They have not been rerun.
