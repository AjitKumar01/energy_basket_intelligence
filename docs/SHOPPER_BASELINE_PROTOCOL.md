# SHOPPER baseline: locked protocol

Locked 2026-10-21, before any Dunnhumby SHOPPER run. Changes after this date need a dated
entry under "Amendments" with the reason; results are reported against this protocol.

## Implementation

`shopper/` is a PyTorch reimplementation of SHOPPER (Ruiz, Athey and Blei,
arXiv 1711.03560; the released C++/CUDA code needs an NVIDIA GPU). Deliberate differences
from the released code, stated with every result:

- MAP point estimates with Adam instead of variational inference (same prior variances,
  shapes and rates as the released defaults);
- an exact softmax over each trip's offered items instead of the one-vs-each bound;
- no look-ahead (the paper reports look-ahead improves fit, so this SHOPPER may be weaker
  than the best published variant).

Verified by `check_shopper.py` (law sums to one, sampling, unbiased ordering estimators,
gradients, padding) and `check_illustrative.py` (the paper's simulation).

## Inputs

The same model-data bundle, trips, splits, offered assortments, log-availability offsets,
price panel and weeks as the energy model, read through the energy model's own `Batcher`
(`shopper_data.BundleSource`). Offered products only. Item groups (shared price and season
factors) are subcategories.

## Settings and selection

- Fixed: price factors P = 10, season factors D = 10, price prior shape 100 / rate 1000,
  Gaussian prior variances 1 (season 0.01), Adam learning rate 0.01, batch 256, evaluation
  every 500 updates, early stopping after 6 evaluations without a 0.001 gain.
- Selected on validation trips only, by the basket log-likelihood: taste size K.
  - Stress world (done): K in {8, 16, 32, 50} -> K = 8 (validation -28.659). Loose price
    priors (shape 1 / rate 1 and rate 0.1) were also fitted; validation did not prefer them
    (-28.664, -28.687), so the default stays. No further SHOPPER variants on the stress world.
  - Dunnhumby: K in {8, 16, 32}, everything else fixed as above; the chain-only price panel
    (`chain_product_week_only`), as for the refitted energy model.

## Scoring (identical for SHOPPER and the energy model)

1. Basket log-likelihood log P(S): SHOPPER sums over orderings by importance sampling
   (256 orderings, half-split reported); the energy model uses its exact or IS normaliser.
   Paired per trip on the same trips.
2. Realism: the classifier two-sample AUCs (full basket, category presence), one generated
   basket per trip per seed (3 seeds), generation capped at the energy model's nmax.
3. Price: stress world - the 32 causal scenarios (own, same subcategory, rest of category),
   Monte Carlo with common random numbers for SHOPPER; Dunnhumby - the validation
   discount-depth slope against the empirical conditional-binomial slope.

Validation trips for selection and comparison; test weeks are opened once, for the final
models only.

## Amendments

- 2026-10-21 (measurement only; no model or selection change). `check_shopper_ll_bias.py`
  on 256 stress-world reference trips: the 256-ordering estimate of SHOPPER's log P(S) is
  0.41 +/- 0.07 nats per basket too low against 8,192 orderings (0.14 for baskets of 1-5
  items, 1.42 for 21+), and at 8,192 orderings two halves still differ by 0.41, so the true
  value is higher again. The stress-world gap to the energy model is therefore reported as
  about -3.0 (at least -2.5 allowing for the remaining bias), not -3.39. From now on SHOPPER's
  log-likelihood uses at least 8,192 orderings with the half-split and this bias check
  reported alongside.
