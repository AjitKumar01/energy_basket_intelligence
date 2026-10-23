# The Version-4 pipeline

This is the write-up of the pipeline as it stands on 2026-10-21: what the model is, the stages
that fit it, which parameters each stage learns, what is decided on validation data, how it is checked, where it
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
store, and the household's purchase history. The full theory, with every formula and derivation, is in
[`THEORY.md`](THEORY.md).

The normaliser Z is computed **exactly**: with phi = 0 by a dynamic program over categories and
basket sizes (`v4/exact.py`); with phi != 0 the pair term is written as an average over a
Gaussian latent z, and given z the same exact program applies (`v4/sampling.py` averages over z).
One fitted model therefore gives the basket likelihood, generated baskets, and price
counterfactuals (change a price, recompute the law).

## 2. Input: the model-data bundle

A bundle is a directory built by `run/prepare_dunnhumby.sh` (from the raw Dunnhumby CSVs) or by
`data_prep/canonical/prepare_bundle.py` (from any dataset's canonical directory, such as the
stress world's):

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
sizes 1–34; how it is generated is in [`STRESS_WORLD.md`](STRESS_WORLD.md).

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

## 5. What is decided on validation data

There is one model. Fitting it involves a few decisions about which fitted version to keep,
and all of them use validation data only:

- **When to stop training.** Each fit saves checkpoints along the way, and the one with the
  best held-out score is kept: validation weeks for the additive fit, 2,048 held-out training
  trips for the Version-4 fit.
- **Whether to switch on an optional block.** M4c (and recency) are parts of the same model
  that can be switched off, which makes them zero. Whether to keep M4c is decided by the
  validation log-likelihood and the validation discount check.
- **The test weeks are used once**, only for the final fitted model (`run/dunnhumby.sh` stage 5,
  only when `CHOSEN` is set), so nothing is ever picked by looking at test results.

Two related fixed rules:
- On the stress world the pass rules were fixed in advance: paired held-out log-likelihood above
  the comparison model on the same reference trips (household-clustered 95% interval above 0),
  category-presence realism AUC of at most 0.80, and own-price scenario correlation of at least
  0.85. The comparison model was the earlier certified Version-4 model. In this branch it is the
  base model, whose exact per-trip scores `scripts/reference_trips.py --base-model` stores.
- SHOPPER, the comparison model, has one setting chosen on validation (its taste size K) under
  its own locked protocol (`docs/SHOPPER_BASELINE_PROTOCOL.md`); changes need a dated amendment.

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
4. **Support cap** (`checks/support_cap.py`): the probability the fitted parameters would put on
   baskets beyond the per-category cap (and beyond 40 or 60 categories) if the cap were removed,
   exact per trip and averaged over latent draws. It shows whether the cap is harmless.
5. **SHOPPER baseline**: a sequential model (Ruiz, Athey and Blei), refitted in PyTorch on the
   same trips, scored the same way.

## 7. Where things stand (2026-10-22)

**Stress world**, model with recency (`hs_rec`; M4c in brackets):

| Measure | Version 4 | SHOPPER (K = 8) |
|---|---|---|
| Held-out log-likelihood per basket | −24.66 (−24.61) | −28.04; about −3.0 behind once its ordering estimate's bias is allowed for |
| vs the earlier certified model, 95% interval | +4.27 [4.16, 4.37] | — |
| vs its base model (the same trips) | +4.62 (+4.67) | — |
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
| **refit: chain-only + discount split, everything refitted (Step 3)** | **−44.706** | **1.30** | **0.11** | **0.27** |
| empirical (train / validation / test) | — | 1.07 / 0.64 / 1.00 | 0.43 / 0.38 / 0.41 | 0.31 / 0.35 / 0.23 |

The full refit converged at update 14,300 (best at 13,500). It fits validation no better than
the old parent, but the discount response is no longer too weak: it now overshoots, at 1.30.
In the refit the discount part has its own strong slope (about 0.97 on average) next to a
regular-price slope of about 0.19.

**Dunnhumby Version 4** (Step 4, finished 2026-10-22). Both variants were fitted from the
Step-3 refit with the update-matched schedule and at most 40 products per category, and
both converged. All likelihoods below are on the same 4,096 validation trips (the old
pipeline's validation trips; `scripts/reference_trips.py`), as log-likelihood per basket:

| Model | Log-likelihood | Version 4 (recency) gains over it |
|---|---|---|
| old base model | −43.676 | +3.27 |
| old Version 4 | −43.662 | +3.25 [3.04, 3.48] |
| new base model (Step 3 refit) | −43.689 | +3.28 |
| **Version 4, recency (chosen)** | **−40.407** | — |
| Version 4, recency + M4c | −40.462 | −0.055 ± 0.021 |

The gain grows with basket size: −0.14 for 1–3 products, +2.6 for 4–8, +6.6 for 9–20 and
+15.9 for 21–120. M4c is slightly worse, so **recency is the chosen model**.

**Test weeks** (opened once, 2026-10-22, for the recency model; the old pipeline's 4,096 test
trips, same order; `runs/dunnhumby/reference_test_trips.npz`, `v4_rec_test.per_trip.npz`):

| Model | Test log-likelihood | Version 4 (recency) gains over it, household-clustered 95% |
|---|---|---|
| old base model | −46.073 | +3.62 [3.39, 3.85] |
| old Version 4 | −46.051 | +3.60 [3.37, 3.83] |
| new base model (Step 3 refit) | −46.097 | +3.64 [3.42, 3.87] |
| **Version 4, recency** | **−42.456** | — |

By basket size against the old Version 4: −0.09 (1–3 products), +2.5 (4–8), +7.3 (9–20), +16.4
(21–120). IS check: ESS fraction 0.76, mean half-split difference 0.16. Realism on the test
weeks was stopped after two of three seeds (full-basket AUC 0.80, category AUC 0.69), because
the generated baskets are far too big (24.9 products against 7.8); see the next paragraph.

**Basket size drifts with calendar time** (`checks/size_by_period.py`). The model's expected
basket size under its own law, against the real size:

| Weeks | Products the household had ever bought before the trip | Model E[N] | Real |
|---|---|---|---|
| training, early (9–28) | 68 | 4.3 | 5.5 |
| training, late (66–82) | 358 | 11.8 | 6.4 |
| validation (83–90) | 422 | 17.8 | 7.1 |
| test (91–101) | 442 | 18.9 | 7.2 |

The sampler is not at fault: the C++ Gibbs chain matches one built from the tested pieces, and
an importance-sampling estimate whose proposal covers both small and big baskets agrees with
the standard log Z (6.279 against 6.296 on 32 test trips) and gives the same E[N]. The cause
is the purchase-history feature "ever bought this product": it only grows over time, the model
gives it a positive weight (+0.77 per category on average; right for *which* products are
bought), and in this law raising many utilities also raises basket size, the same mechanism
as the price halo. Early training trips are predicted too small and late ones too big, which
cancel in the training moment check (7.76 against 7.65). The likelihoods above stay valid (the
law is correctly normalised); generation, size predictions and counterfactuals in later weeks
are biased upward.

| Discount and promotion check (validation) | Base (Step 3) | V4 recency | V4 + M4c | Data: train / validation / test |
|---|---|---|---|---|
| Discount depth slope, single item | 1.30 | 1.42 | 1.45 | 1.07 / 0.64 / 1.00 |
| Display effect, purchase-weighted | 0.11 | 0.06 | 0.08 | 0.43 / 0.38 / 0.41 |
| Mailer effect | 0.27 | 0.22 | 0.24 | 0.31 / 0.35 / 0.23 |

Support cap: removing the 40-per-category cap would put at most 4.5e-10 (recency) and
2.6e-5 (M4c) of a trip's probability above it, so it is harmless.

**Recommendation** (`checks/recommendation.py`; hide one product of a validation basket, rank
every offered product by the exact log odds of adding it; 1,620 cases):

| Model | MRR | Recall@10 |
|---|---|---|
| popularity | 0.045 | — |
| old base / old Version 4 | 0.093 | 18.0% |
| new base | 0.094 | 17.7% |
| **Version 4, recency** | **0.118** | **21.8%** (recall@100 49.9%) |

Paired MRR gain over the old model +0.025 (s.e. 0.004).

**Speed** (Version 4; one trip / per trip in a batch of 64): likelihood with 256 latent draws
9.7 s / 3.2 s, recommendation 9 ms / 2.4 ms, a price counterfactual with 64 draws 3.4 s / 1.3 s.
The base model takes a few milliseconds for each.

**What the model learned** (`checks/embeddings.py`, `checks/segments.py`):

- *Interactions:* the first direction carries 59% of the interaction variance, and the
  category part carries 87%. Learned pair strength correlates 0.40 with real log
  co-purchase lift. The strongest pairs are picnic/BBQ, convenience and baking groups;
  fresh milk repels candy, soft drinks, water and baby food.
- *Taste:* 36% of each product's 5 nearest neighbours share its commodity (1.6% by chance).
  The household size shift is higher for large households and households with children.
- *Price:* a household's price slope correlates 0.60 with the share it really buys on
  discount. Produce, isotonic drinks and bath tissue are the most price-sensitive; magazines,
  cards, cigarettes and baby food the least. National brands are more sensitive than private
  label (0.017 against 0.014).
- *Segments:* k-means on taste, size shift and price slope gives five segments (deal-seeking
  families, convenience stock-up, fresh produce, frequent small households, cooking basics).
  Each segment's over-bought commodities in training weeks correlate 0.67–0.94 with those in
  validation weeks. There are no sharp clusters (silhouette about 0.02) and stability across
  seeds is moderate (adjusted Rand 0.58–0.77), so they are a useful cut of a continuum.

**Promotion environment** (`sim/`). A Gymnasium environment: one step is one week, the action
is a discount level (0/10/20/30%) for each of the 12 highest-revenue commodities, and the
reward is expected revenue at paid prices relative to the same week with no promotion. The
simulator is either the base model (exact) or Version 4. It works, but its answers are not yet
believable: 30% off everything triples revenue in the base model, because of the halo below.
Do not use it for policy decisions until Phase 2 is done.

**The halo** (`checks/halo.py`, stress world, 20% price cut on one product; change in items
per 1,000 baskets, mean of 16 scenarios):

| | Own product | Own category total | Other categories |
|---|---|---|---|
| truth | +7.0 | 0.0 | 0.0 |
| base model | +6.9 | +6.4 | +7.9 |
| Version 4, latent draws reweighted (the law's own counterfactual) | +8.1 | +7.0 | +5.1 |
| Version 4, each trip's latent state held fixed | +6.5 | +3.9 | +0.8 ± 0.4 |

In the truth a price cut is pure switching: the product gains, its rivals lose the same amount,
nothing else moves. Every model gets the own effect roughly right, but (a) adds an invented
gain to the rest of the basket, because in the joint law a more attractive product makes big
baskets more likely, and big baskets contain everything; and (b) takes too little from rivals.
Holding the latent state fixed removes most of (a), not (b). That fit used the true categories
as its partition, so (b) is not only a partition problem. On Dunnhumby the affinity partition
makes it worse: a commodity is split over 7 groups on average (soft drinks over 41), and
1,724 products share one mixed leftover group.

**Known weaknesses**

- **Basket size drifts with time** through the purchase-history features (above).
- **The halo and too little switching** (above). This affects every price and promotion
  counterfactual on real data.
- **Display is too weak**, and Version 4 made it weaker (0.06 purchase-weighted, against 0.38–0.43
  in the data). Likely cause: display is learned separately per product with no pooling, while
  in Dunnhumby display nearly always comes with a discount.
- **The discount response overshoots** (1.42 in Version 4, 1.30 in the base model, against
  0.64–1.07 in the data).
- On the stress world the own-price response is about 1.7× too strong on average.

**Phase 2** (agreed 2026-10-22, in this order):

0. *Make the purchase-history features neutral to basket size* (the drift above), so the
   model's E[N] follows the real size in every period; worked on in branch `v4-history-size`.
   Implemented there as `--centre-history` (v4/model.py: every utility of a trip shifted by
   the trip mean of the history shift plus a learned hist_size * log(1 + products ever
   bought); exact, tested against enumeration). Applied to the fitted v4_rec.pt *without*
   refitting it already moves validation E[N] from 16.8 to 10.3 (real 8.7) and test from
   15.6 to 9.1 (real 8.1), and raises the log-likelihood on 512 validation trips by
   +0.25 ± 0.10 (+0.5 on baskets of 1–8 products, −2.3 on the 34 of 21+). Refit:
   `DATA=... FROM=runs/dunnhumby/v4_rec.pt run/dunnhumby_history.sh`.

   *Refit result* (`v4_hist.pt`, 2026-10-23; warm start, 4 SML epochs + 1 calibration, 4.8 h).
   Held-out training trips −38.325 against −38.506 for v4_rec (same 1,024 trips). The learned
   hist_size stayed small (+0.015). Expected basket size, the same 96 trips per period:

   | Weeks | Real | v4_rec | v4_rec centred, no refit | v4_hist |
   |---|---|---|---|---|
   | training, early | 7.33 | 5.01 | 5.90 | 6.70 |
   | training, late | 8.04 | 10.21 | 7.59 | 9.60 |
   | validation | 8.69 | 16.76 | 10.34 | 11.00 |
   | test | 8.06 | 15.62 | 9.07 | 11.95 |

   Recommendation (1,620 validation cases): MRR 0.121 against 0.118, recall@100 50.4% against
   49.9%. Validation log-likelihood on the 4,096 trips: running.

   The drift is much smaller but not gone (0.91× the real size early in training, 1.48× on
   test), and the refit is worse on size than centring without refitting. Most of the excess is
   a tail: the validation median is 7.2 while the mean is 11.0. Two likely reasons: (a) the
   plain-mean centring is first order only; as history grows, more products carry the +0.77
   "ever bought" shift, the spread of h widens, and through log-sum-exp a wider spread still
   raises size; (b) hist_size is learned from training trips where early and late errors
   partly cancel. Next: centre by the history shift's effect on the trip's inclusive value,
   o_t = log sum_j exp(b0_tj + h_tj) - log sum_j exp(b0_tj), which removes the spread effect to
   first order and is still a per-trip constant (exact); check it on v4_rec without a refit
   first, as before.
1. *Within-group competition in the model:* a learned term that changes the energy of buying a
   second product from the same group, so a price change moves purchases within the group
   instead of adding items. It goes in the exact per-group DP. Test it on the stress world
   with `checks/halo.py` (switching and halo) before anything else.
2. *Hold the latent state fixed* in price and promotion counterfactuals (the environment and
   the scenario scorer), if step 1 does not make it unnecessary.
3. *A partition that keeps rivals together on Dunnhumby:* commodities, or affinity groups split
   by commodity; then refit Steps 3 and 4.
4. *Measure the real halo:* when one commodity is discounted in a store-week, do other
   commodities' purchases per trip move?
5. *Promotions:* a common display slope with per-product variation shrunk toward it, and a
   firmer prior on the promotion weights during Version-4 training.
6. Then the promotion environment, the test weeks for the chosen model, and SHOPPER on Dunnhumby.

## 8. Commands

```bash
python3 -m pytest -q                                     # exact checks, no data
RAW=/path/to/raw/csvs OUT=data/dunnhumby run/prepare_dunnhumby.sh   # Dunnhumby bundle
OUT=data/stress_world run/prepare_stress_world.sh                    # stress world and its bundle
DATA=data/dunnhumby nohup run/dunnhumby.sh > runs/dunnhumby/pipeline.out 2>&1 &
DATA=... INIT=... ADDITIVE=... nohup run/dunnhumby_step4.sh > runs/dunnhumby/step4.out 2>&1 &
WORLD=data/stress_world nohup run/stress_world.sh > runs/stress/pipeline.out 2>&1 &
DATA=... CONTRACT=... nohup run/shopper.sh > runs/shopper/pipeline.out 2>&1 &
DATA=data/dunnhumby CHOSEN=rec FROM=5 run/dunnhumby.sh       # test weeks, once, for the chosen model
```

The native kernels (`v4/native/`) compile on first use into `build/native/` (`$CXX`, else clang++,
else g++, and the PyTorch C++ extension). Setting up a new machine: see the README. Use at most 4 threads per job (`THREADS`).

## 9. History: how this branch relates to `v4-training`

`v4-training` is the research branch where this pipeline was worked out (every experiment,
diagnostic and earlier pipeline). `v4-pipeline` is self-contained and needs nothing from it.
It was rewritten from it, with the same numerical results:

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
