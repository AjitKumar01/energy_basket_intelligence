# Nested substitution groups (parked)

**Branch:** `nested-substitution` (from `clean`). **Status:** implemented and tested; **not run in
full and not merged**, because a cheaper diagnostic showed it cannot close the gap it was built
for (§3).

## 1. What was built

Each product group (leaf) can name a parent group, and the energy gets one penalty per parent on
top of the per-leaf penalty:

\[
E(S)=\dots-\sum_{c}\rho_c\binom{n_c}{2}-\sum_{p}\rho_p\binom{n_p}{2},\qquad
n_p=\sum_{c\in p}n_c,\quad \rho_p\ge 0 .
\]

- `prepare_model_bundle.py`, partition `nested_catalogue`: leaves are the finest declared
  catalogue level (one group per (category, subcategory) path), parents are categories;
  `items_affinity.parquet` carries `parent_id`.
- One exact program for every computation (`ragged.nested_log_coefficients`): leaf log ESP times
  the leaf potential, product within the parent times the parent potential, product within the
  trip. It is used by the quadrature normaliser, the no-interaction normaliser, the conditional
  (revealed-basket) law and the size-potential initialization.
- Every sampler dispatches to one exact nested reverse sampler (size, then parent counts, then
  leaf counts within a parent, then products). Add-one scores, the refinement energy and design,
  and the embedding audit carry the parent term. \(\rho_p\) is fitted and projected to \(\ge 0\).
- A bundle without parents gives exactly the flat model (no new parameters, unchanged paths).
- `tests/test_nested_groups.py` (14 tests) checks everything against brute-force enumeration:
  normaliser, conditional law, gradients, both flat special cases, add-one log odds, refinement
  energy, and all samplers within Monte Carlo noise. The full smoke pipeline (all stages with
  refinement) runs end to end. The nested sampler is NumPy only; a compiled version was not built.

## 2. Why it was tried

On the synthetic stress world the staged model predicts own-price effects well (correlation 0.875
with the causal truth) but substitution to a product's subcategory siblings poorly (0.49), with a
predicted mean response of 0.000 against a true +0.026. The world's shoppers pick a subcategory
and then a product (nested logit), so a two-level penalty looked like the matching structure.

## 3. The diagnostic that parked it

Before a full nested run, the staged pipeline was run with the existing `finest_catalogue_level`
partition (penalty on subcategories only), against the category partition on the same 4,096 test
trips and the same 32 price scenarios:

| | Category groups | Subcategory groups |
|---|---|---|
| Test log-likelihood (paired) | — | −0.172 [−0.191, −0.153] |
| MRR | 0.237 | 0.221 |
| Own-price effect: corr / MAE | 0.875 / 0.152 | 0.875 / 0.153 |
| Same-subcategory siblings: corr / MAE | 0.488 / 0.061 | 0.201 / 0.061 |
| Rest of category: corr / MAE | 0.871 / 0.034 | −0.946 / 0.037 |
| Siblings' mean response, model vs truth | 0.000 vs +0.026 | −0.000 vs +0.026 |

## 4. Conclusion

In this model a price change shifts demand to product \(j\) only through the co-purchase
penalties: \(\partial\,\Pr(j\in S)/\partial b_k=\operatorname{Cov}(1_j,1_k)\), and the penalties are
learned from how often products are bought together. In the stress world a product's siblings
and its other category-mates are both rarely bought with it (one product per need), so co-purchase
data cannot separate them, and no grouping makes the model predict extra switching to siblings.
The truth's sibling substitution comes from choice (subcategory, then product) and is visible in
responses to price variation, not in co-purchase counts. Finer or nested groups therefore cannot
reach the target (sibling correlation \(\ge 0.7\)); closing the gap needs cross-price effects
learned from price variation, which is a change to the model's substitution mechanism.

The implementation stays on this branch because it is exact and general: on data where products
of one subcategory really are bought together less than products of different subcategories,
nested groups are the right structure.
