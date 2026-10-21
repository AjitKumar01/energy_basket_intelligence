# The Version-4 pipeline

This is the write-up of the pipeline as it stands on 2026-10-21: what the model is, the stages
that fit it, which parameters each stage learns, how models are chosen and checked, where it
works well and where it does not yet. Every command is in `run/`. The code is in `v4/` (the
model), `scripts/` (one script per stage), `checks/` (the real-data discount and promotion check),
`shopper/` (the SHOPPER baseline) and `tests/` (exact checks that need no data).

## 1. The model in one paragraph

For a shopping trip (household, store, day, prices, promotions, the products the store offers),
the model gives every possible non-empty basket a probability:

    p(S | trip) = exp E(S) / Z,       Z = sum of exp E over every possible basket

    E(S) = sum of product utilities b_j over the basket
         + sum over product pairs of phi_i . phi_j          (which products go together)
         - within-category terms rho_c, f_c                  (how many from one category)
         - size potential rho_0                              (how many categories in total)

A product's utility b_j adds its popularity, the household's taste, its price (relative to the
trip's other prices, with discounts priced separately), display and mailer, the season, the
store, and the household's purchase history. All formulas are written out at the top of
`v4/model.py`.

The normaliser Z is computed **exactly**: with phi = 0 by a dynamic program over categories and
basket sizes (`v4/exact.py`); with phi != 0 the pair term is written as an average over a
Gaussian latent z, and given z the same exact program applies (`v4/sampling.py` averages over z).
One fitted model therefore gives the basket likelihood, generated baskets, and price
counterfactuals (change a price, recompute the law).

## 2. Input: the model-data bundle

A bundle is a directory prepared by a dataset adapter (outside this repository):

| File | Content |
|---|---|
| `basket_input/meta.json` | sizes, the train / validation / test week boundaries |
| `basket_input/baskets.parquet` | one row per purchased product: basket, household, store, day, week, product, split |
| `basket_input/items.parquet` | product attributes (merchandise category, subcategory) |
| `basket_input/items_affinity.parquet` | the model's categories (stage 1) |
| `basket_input/log_price_dev.npy` | chain-wide log price minus the product's training mean, per product and day |
| `basket_input/promo.npz` | display and mailer, per product, store and week |
| `basket_input/state.npz` | purchase events for the recency features |
| `basket_input/model_data_fingerprint.json` | hashes of the above; every checkpoint records it |
| `data/price_week.parquet` | weekly chain price, regular price and promo depth (for the discount panel and check) |

Dunnhumby: 5,455 products, 1,920 households, 115 stores, 300 model categories, 200,698
baskets, basket sizes 1–120; training weeks 9–82, validation 83–90, test 91–101.
Stress world (synthetic, known truth): 476 products, 2,802 households, 15 stores, 48 categories,
sizes 1–34.

## 3. The stages

| # | Script | What it does | Output |
|---|---|---|---|
| 1 | `scripts/build_affinity_partition.py` | groups products bought together in training into categories (at most 128 products each; the rest in one residual group) | `items_affinity.parquet` |
| 2 | `scripts/build_discount_panel.py` | discount depth min(log(price / regular price), 0) per product and day, centred on training days | `log_discount.npy` |
| 2b | `scripts/initialize.py` | starting values, no fitting | `initialization.pt` |
| 3 | `scripts/fit_additive.py` | exact base model, phi = 0 | `additive/v3_pipeline_additive_best.pt` |
| 4 | `scripts/fit_interactions.py` | full Version-4 model, in three parts (below) | `v4_*.pt` |
| 5 | `scripts/evaluate.py` | held-out likelihood, realism, price scenarios | `*.eval.json` |
| check | `checks/discount_promotion.py` | real discounts, displays and mailers vs the model | `discount_check.json` |
| baseline | `shopper/fit_shopper.py` | SHOPPER, a sequential basket model | `runs/shopper/` |

**Stage 3, the additive base model.** With phi = 0 the likelihood is exact, so this is plain
maximum likelihood (AdamW, minibatches of 128 trips) plus four small terms that keep it well
behaved:

- a size-law term that keeps the model's average basket-size distribution close to the data's;
- an elasticity anchor, (aggregate price elasticity + 0.121)^2 × 20, that keeps the overall level
  of price response near the data's while the household × product price slopes are learned;
- a pooling term on the three product × context factorisations (season, store, taste);
- after every step, safety bounds on the within-category attraction rho_c.

Prices are **chain-only** (`chain_product_week_only`). The store price deviations of the old
contract exist only where a product sold, so they leaked purchases into the feature. The price
is split into a regular part and a discount part. The discount part gets its own multiplier
(`disc_kappa`) and a slope shared by all households and products (`disc_common`); both start
at values that make the model exactly the one-slope model. The learning rate halves after four
validation checks without a 0.001 gain; the fit stops at the minimum learning rate after eight
such checks.

**Stage 4, Version 4.** The base model gains:

- tied interactions phi_j = psi_c(j) + delta_j (rank 16), with psi charged per category present
  (M2);
- a free potential for 2, 3 and at least 4 products of one category (M1);
- the size potential on the number of categories present rather than products (M3);
- household taste per category (M4a);
- with `--recency`, sub-commodity recency and product purchase history;
- with `--m4c`, a per-category factor lambda_c in (0, 1) that lets a discount move choice within
  its category but raise the category's incidence only by a fraction of its inclusive-value gain
  (M4c).

It is trained in three parts:

- **A. Pseudo-likelihood**, 2 epochs: exact conditionals log P(product j | rest of basket); warms
  up phi cheaply.
- **B. Stochastic maximum likelihood**, up to 28 epochs: each training trip keeps a Gibbs chain
  across epochs; each step draws z from the chain and differentiates one exact log Z(z). The
  running average of the parameters is scored on 2,048 held-out *training* trips every 4 epochs
  and the best is kept (early stop). Price is unfrozen here, with a prior centred at the base
  model's values.
- **C. Calibration**, 2 epochs: the same objective with phi and the context factors fixed, so the
  size and category terms settle around the final interactions.

Besides Adam, rho_0, rho_c and lam also take moment-matching steps toward the data's category
presence, category pairs and product rates.

## 4. Which parameters are learned where

| Parameters | 2b initialize | 3 additive | 4A pseudo-lik. | 4B SML | 4C calibrate |
|---|---|---|---|---|---|
| popularity lam | log purchase rate | learned | learned | learned + moments | learned (½ lr) + moments |
| taste theta, alpha (last coordinate = household size shift) | SVD of household shares | learned | learned | learned | fixed |
| price slopes gamma, beta, price_kappa | small random | learned (elasticity anchor) | fixed | learned (prior at base) | learned (½ lr) |
| discount disc_kappa, disc_common | — | learned (start = one slope) | fixed | learned (prior at base) | learned (½ lr) |
| display, mailer w_dsp, w_mlr | 0 | learned | learned | learned | fixed |
| season mu, delta; store zeta, xi | small random | learned | learned | learned | fixed |
| within-category rho_c | 0 | learned (bounded) | learned | learned + moments | learned + moments |
| size rho_0 | matched to the data's size law | learned | learned (on categories, M3) | learned + moments | learned + moments |
| interactions psi_cat, delta (phi) | — | held at 0 | learned | learned | fixed |
| M1 f_extra | — | — | learned | learned | learned |
| M4a u_cat | — | — | learned | learned | learned (½ lr) |
| recency psi, psi_item (`--recency`) | — | — | learned | learned | learned (½ lr) |
| M4c iv_slope_raw (`--m4c`) | — | — | learned | learned | learned (½ lr) |

Without `--recency` the unused recency loading psi sits with the price block.

## 5. How models are chosen

- Every choice uses **validation** only: the additive fit's early stopping (validation weeks),
  Version-4's early stopping (held-out training trips), and the choice between variants (e.g.
  with or without M4c), which uses the validation log-likelihood and the validation discount check.
- **Test weeks are opened once per final model** (`run/dunnhumby.sh` stage 5, only when
  `CHOSEN` is set).
- On the stress world the rules are fixed in advance: paired held-out log-likelihood above
  the certified model (household-clustered 95% interval above 0), category-presence realism AUC
  of at most 0.80, and own-price scenario correlation of at least 0.85.
- SHOPPER follows its own locked protocol (`docs/SHOPPER_BASELINE_PROTOCOL.md`); changes need a
  dated amendment.

## 6. The checks

1. **Exact tests** (`tests/`, `python3 -m pytest -q`, about 1 second, no data). On a tiny world
   where every basket can be listed, they compare against enumeration: log Z and incidences
   given z (each Version-4 block alone and all together), the additive size law, the energy
   (Gaussian identity), the exact sampler, Gibbs chains and the importance-sampled log Z, the
   pseudo-likelihood, the discount split's starting point, and the one-product price tilt.
2. **Stress world, known truth** (`run/stress_world.sh`): likelihood, the realism classifier,
   and 32 single-product price scenarios against the generator's causal answers (own product,
   rest of its subcategory, rest of its category, basket size).
3. **Dunnhumby discounts and promotions** (`checks/discount_promotion.py`, real data, no
   truth). An event is a product whose chain discount depth changed by at least 5% between two
   weeks in a store. The *empirical* effect on its within-category share comes from a conditional
   binomial (given the two weeks' purchases; offset = the category's basket change). The *model's*
   effect comes from moving that price back, or switching the display or mailer on and off.
   Compare the depth slope with the depth coefficient, and the promotion log ratios with the
   display and mailer coefficients.
4. **SHOPPER baseline**: a sequential model (Ruiz, Athey and Blei), refitted in PyTorch on the
   same trips, scored the same way.

## 7. Where things stand (2026-10-21)

**Stress world**, model with recency (`hs_rec`; M4c in brackets):

| Measure | Version 4 | SHOPPER (K = 8) |
|---|---|---|
| Held-out log-likelihood per basket | −24.66 (−24.61) | −28.04; about −3.0 behind once its ordering estimate's bias is allowed for |
| vs the certified model, 95% interval | +4.27 [4.16, 4.37] | — |
| Realism AUC, full basket / categories (0.5 is perfect) | 0.62 / 0.73 (0.62 / 0.73) | 0.92 / 0.89 |
| Price scenarios, correlation with truth: own / same subcategory / rest of category | 0.88 / 0.51 / 0.92 (0.89 / 0.47 / 0.81) | 0.82 / ≈0 / ≈0 |

The own-price response is right in pattern but about 1.7× too strong on average (mean change
+0.086 against +0.050). Substitution to the rest of the subcategory follows the truth's
pattern, but its average size is near zero, against +0.026 in the truth.

**Dunnhumby base model**, validation:

| Model | Validation log-likelihood | Discount depth slope (single item) | Display effect (purchase-weighted) | Mailer effect |
|---|---|---|---|---|
| old parent (store prices, one slope) | −44.706 | 0.49 | 0.11 | 0.38 |
| pilot: chain-only + discount split (price block only refitted) | −44.700 | 0.87 | 0.10 | 0.38 |
| empirical (train / validation / test) | — | 1.07 / 0.64 / 1.00 | 0.43 / 0.38 / 0.41 | 0.31 / 0.35 / 0.23 |

The full refit of the base model with chain-only prices and the discount split (stage 3, run
on branch `v4-training` as Step 3) is in progress. At update 13,300 its validation log-likelihood
was −44.711, still improving.

**Known weaknesses**

- **Display is too weak.** The model's display effect is 0.10–0.11 purchase-weighted and about 0
  on average over events, against 0.38–0.43 in the data. 21% of events show a negative effect,
  and the mean display weight over the catalogue is −0.33. The likely cause is that display
  is learned separately for each product with no pooling, while in Dunnhumby display nearly
  always comes with a discount. The likely fix, like the discount one, is a common display slope
  plus per-product variation shrunk toward it. Accepted for now.
- The discount response (0.87) lies between the training and test empirical slopes (1.07, 1.00)
  but above validation (0.64). The empirical slopes themselves move by split.
- On the stress world the substitution response is too small on average (above).

## 8. Commands

```bash
python3 -m pytest -q                                     # exact checks, no data
nohup run/dunnhumby.sh > runs/dunnhumby/pipeline.out 2>&1 &
nohup run/stress_world.sh > runs/stress/pipeline.out 2>&1 &
nohup run/shopper.sh > runs/shopper/pipeline.out 2>&1 &
CHOSEN=m4c FROM=5 run/dunnhumby.sh                       # test weeks, once, for the chosen model
```

The native kernels (`v4/native/`) compile on first use into `build/native/` (clang++ and the
PyTorch C++ extension). Use at most 4 threads per job (`THREADS`).

## 9. How this branch relates to `v4-training`

`v4-pipeline` contains only the current pipeline. It was rewritten from `v4-training`, with the
same numerical results:

| Here | On `v4-training` |
|---|---|
| `v4/bundle.py`, `v4/features.py`, `v4/batches.py` | `scripts/version4/data.py`, `features.py`, `fit.py` (Batcher), `ragged.py` (RaggedIndex), `stress.py` (PurchaseHistory) |
| `v4/model.py` | `ragged.py` (RaggedModel) and the Version-4 parts of `stress.py` |
| `v4/exact.py`, `v4/kernels.py`, `v4/native/` | `interaction_particles.py`, `ragged.py`, `poly_degree_native.*`, `stress.py`, `stress_sampler.cpp` |
| `v4/sampling.py` | `stress.py` (sampler, Gibbs, importance sampling) |
| `scripts/initialize.py` | `initialize_version4.py` and the initializers in `fit.py` |
| `scripts/fit_additive.py` | `fit_exact_additive.py` |
| `scripts/fit_interactions.py`, `scripts/evaluate.py` | `stress.py train`, `stress.py eval`, `eval_3seed_auc.py` |
| `checks/discount_promotion.py` | `fit_discount_response.py` (events, effects, model check) |

**Left out**, because the current pipeline no longer uses them:

- the old staged pipeline (Smolyak quadrature, rank scoring, natural-parameter solve,
  household-size block, joint refinement);
- the frozen "supported price" option of the additive fit, which the stress world's base model
  was originally fitted with;
- M4b, the full inclusive-value term (M4c, its price-only form, is kept);
- the pure-Python reference sampler (the tests compare the C++ sampler with enumeration
  instead);
- the quantity model;
- all one-off diagnostics.

**Checked to give the same numbers** as `v4-training`:

- Utilities, energy, log Z, incidences, sampler draws, Gibbs chains and importance-sampled
  log Z: identical on a Version-4 checkpoint.
- A short Version-4 training: the same logged values, parameters within 5e-15 (rounding), and
  the new code repeats itself exactly.
- Short additive fits on the stress world and on Dunnhumby (with the discount split and the
  elasticity anchor): identical at every update.
- Evaluation of a Version-4 checkpoint: identical held-out log-likelihood, realism AUCs (3 seeds)
  and all 32 price scenarios.
- The discount events and empirical effects: identical. The model check on two base models:
  identical depth slopes and promotion effects.
- The initializer: identical except phi (now 0, which the additive stage imposed anyway) and
  rho_0, which the old code set using its random phi and the store prices.
