"""Version-4 energy basket model: one exact joint law over shopping baskets.

For a trip x (household, store, day, prices, promotions, offered products) the model gives
every non-empty basket S of offered products the probability

    p(S | x) = exp E(S, x) / Z(x),

    E(S, x) = sum_{j in S} b_j(x)                     product utilities (model.py)
            + sum_{i<j in S} phi_i . phi_j            low-rank pairwise interactions
            - sum_c rho_c * C(n_c, 2)                 within-category penalty
            - rho_0(size)                             size potential

and Z(x) sums exp E over all baskets.  With phi = 0, Z is computed exactly by a dynamic
program over categories and sizes (exact.py).  With phi != 0 the Gram term is written as a
Gaussian integral over a latent z, so given z the law is again exactly tractable; training
and scoring average over z with Gibbs chains and importance sampling (sampling.py).

Modules, in the order data flows through them:

    bundle.py      the model-data bundle: trips, baskets, assortments
    features.py    price, discount, promotion, recency and availability panels
    batches.py     one batch of trips laid out as ragged (trip, category, product) slots
    model.py       parameters, utilities and the energy of a basket
    exact.py       the exact normaliser log Z given z (and with phi = 0)
    sampling.py    exact basket sampling given z, Gibbs chains, importance-sampled log Z
    price.py       counterfactual price contexts and the aggregate price elasticity
    checkpoints.py loading initialization artifacts and fitted checkpoints

Everything runs on the CPU in float64: scripts call torch.set_default_dtype(torch.float64)
before building a model (BasketModel checks).
"""
