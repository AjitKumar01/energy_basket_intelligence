# Raw-data and preprocessing audit

## Current verdict

The corrected preprocessing pipeline passes its independent, fail-closed audit. The
model dataset is now constructed from the locked raw sources without post-cohort outcome
deletion, future-dependent cohort selection, inconsistent price reconstruction, or
promotion-boundary miscoding.

The executable audit is `scripts/data/audit_preprocessing.py`; its machine-readable output
is `basket_input/preprocessing_manifest.json`. `scripts/run_pipeline.py` executes this
audit on every run, including runs that reuse derived files, before initialization or
optimization is allowed.

Raw source digests:

| file | SHA-256 |
|---|---|
| `transaction_data.csv` | `3a685c0729cef664d634486189f774518b84f53cde7cbf701a5963238692b476` |
| `product.csv` | `7ecbcec41e0f1e5a51b43a359965cc50dc5586a2b900a378b64032750fedc949` |
| `causal_data.csv` | `60ed7021fefc209c0caf36fcf95d1e93693775acfdac01fc9ee38273da88937a` |

## Corrected cohort and split

The model retains the intended computational budget of 5,455 products, but no longer
uses weeks 83--102 to decide which products exist. Products are ordered by transaction
line frequency in training weeks 9--82, with `PRODUCT_ID` as the deterministic tie-break,
and the top 5,455 are retained. The selection boundary is 76 training lines; 72 products
tie at that frequency. Every selected product has a training observation.

Households are selected only after the product catalogue is fixed. A household must have
20--300 distinct shopping days in weeks 9--82. This gives 1,920 households, every one of
which has training support. The resulting dataset is:

| split | weeks | basket-product rows | baskets |
|---|---:|---:|---:|
| train | 9--82 | 1,223,933 | 160,007 |
| validation | 83--90 | 129,731 | 17,351 |
| test | 91--101 | 181,342 | 23,340 |
| total | 9--101 | 1,535,006 | 200,698 |

Week 102 is excluded because `causal_data.csv` ends at week 101. Weeks 1--8 are likewise
outside causal-data coverage. Within weeks 9--101, an absent sparse causal record means no
display or mailer; outside that interval it would mean unmeasured, so the pipeline refuses
to encode it as zero.

This is a selected frequent-product loyalty-card cohort, not the full 92,339-product raw
retailer universe. “All 5,455 products” means all products in this declared model catalogue.

## Outcome and support integrity

The previous support builder inferred stock from observed training sales and then removed
7,970 validation/test purchase lines that did not fit that inferred support. This edited
716 validation and 1,201 test baskets and removed 655 baskets completely.

That behavior is eliminated. The transactions contain sales, not a stock/availability
feed. The corrected likelihood therefore declares the same 5,455-product chain catalogue
at every one of the 115 modeled stores. This creates 627,325 store-product pairs and is
conservative for likelihood because every denominator contains the complete declared
choice set. A real stock feed can later replace this declared support without changing the
Version-4 law.

The index builder now fails if *any* train, validation, or test purchase is outside the
declared support. Current result: zero missing lines in all three splits. The independent
audit also reconstructs all basket-product keys from `data/tx.parquet` and verifies that
none were added or deleted after cohort definition.

## Price construction

Stage 1 estimates a deterministic modal faced price for every observed product-week and
product-store-week. Ties are broken toward the lower cent value. Stage 22 now consumes
these tables directly:

\[
\Delta\log p_{jst}
=\log p^{\mathrm{mode}}_{jsw(t)}
-\frac{1}{|\mathcal T_{\mathrm{train}}|}
  \sum_{t'\in\mathcal T_{\mathrm{train}}}\log p^{\mathrm{mode}}_{jw(t')}.
\]

When a store-week modal value is observed, the sparse store deviation makes the gathered
price equal that modal store-week value. Otherwise the feature falls back to the modal
chain-week value. Missing weeks are carried within product, while the centering constant
uses training weeks 9--82 only.

The old second reconstruction--daily medians plus store-week medians--is gone. It differed
from the declared modal table by more than one cent in 2.04% of measured cells and allowed
the realized buyer mix to move the price feature.

The default is modal loyalty-card price because every modeled household holds a loyalty
card. Regular shelf price remains a sensitivity basis via `NF_PRICE_BASIS=base`. Price-
causal claims must still acknowledge that loyalty price is not perfectly single-valued
within every store-week; promotion controls and sensitivity analysis remain necessary.

## Basket identity, quantity, and state

`BASKET_ID`, not household-day, is the observational basket. Every retained `BASKET_ID`
maps to exactly one household, store, day, and week. The old Stage-1 text claiming a
one-to-one household-day mapping was corrected: 18,469 retained household-days contain
multiple checkout baskets.

Product quantities are summed inside a checkout and clipped at 12 units. This affects 731
basket-product rows and removes 0.214% of units, protecting against bulk/random-weight
coding while retaining genuine large baskets. The largest retained basket contains 120
distinct products; large baskets were not deleted merely because they are computationally
difficult.

Recency queries are strictly before the current day, so held-out history may legitimately
be used when predicting a later held-out basket. Aggregate recency scales are estimated
from training only. Same-day ordering is unavailable at the state resolution: a later
checkout on the same day does not see an earlier same-day checkout. This is a declared
data-resolution limitation, not silently inferred order.

## Affinity partition

The affinity partition algorithm is unchanged: supported positive-lift training pairs are
joined greedily, with non-residual groups capped at 128 products, and isolated products
pooled in one residual group. On the corrected training-only cohort it deterministically
produces 300 groups, not the old cohort's 280: the residual group contains 1,724 products
and the largest connected group contains 128.

The number of groups is an empirical output, not a theorem parameter. The partition now
has its own `affinity_manifest.json` containing its parameters and SHA-256 digest.
Initialization verifies that digest and the item/group dimensions instead of relying on a
stale magic number.

## Checks executed before model fitting

The audit independently verifies:

1. all three locked raw SHA-256 digests;
2. the training-only top-5,455 catalogue and deterministic boundary tie-break;
3. the training-only 20--300-day household rule;
4. exact equality of expected and stored basket-product outcomes and clipped units;
5. unique household/store/day/week/split values within each checkout;
6. chronological split labels and complete promotion coverage;
7. all product and household parameters have training observations;
8. exact reconstruction of the modal chain-week price panel;
9. exact reconstruction of modal store-week deviations;
10. training-only price centering; and
11. sorted, unique promotion keys and checksummed derived artifacts.

The full raw-to-index command has been executed successfully, followed by fresh model
initialization and the repository test suite (28 tests). Because the cohort, prices,
partition, choice support, and test period changed, all old checkpoints and their reported
likelihood/MRR values are incompatible with this corrected dataset. Final empirical claims
require a fresh end-to-end fit.
