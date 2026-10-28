"""Version-4 energy basket model in the mixture form (docs/mixture_theory.pdf).

For a trip x (household, store, day, prices, promotions, offered products) each trip has a
taste z in R^r with prior N(0, I).  Given z, every non-empty basket S of offered products has

    p(S | z, x) = exp E_z(S, x) / Z_z(x),

    E_z(S, x) = sum_{j in S} b_j(x) - |phi_j|^2 / 2    product utilities (model.py)
              + z . s(S)                               the taste meets the basket's direction
              - sum_c rho_c * C(n_c, 2) - f_c(n_c)      within-category terms
              - rho_0(categories present)              size potential

with s(S) = sum_{j in S} phi_j (less psi_c per extra product of a category), and Z_z(x)
the sum of exp E_z over all baskets, computed exactly by a dynamic program over categories
and sizes (exact.py).  The probability of a basket is p(S | x) = E_{z ~ N(0, I)} p(S | z, x);
it is estimated by importance sampling around the one peak of the posterior of z, and a basket
is generated exactly by drawing z ~ N(0, I) and then S | z (mixture.py, sampling.py).

Modules, in the order data flows through them:

    bundle.py         the model-data bundle: trips, baskets, assortments
    features.py       price, discount, promotion, recency and availability panels
    batches.py        one batch of trips laid out as ragged (trip, category, product) slots
    model.py          parameters, utilities and the energy of a basket
    exact.py          the exact normaliser log Z_z (and with phi = 0)
    sampling.py       exact basket sampling given z
    mixture.py        the posterior of z, the likelihood estimate, generation, whitening
    training.py       the full model's blocks, priors and projections
    competition_iv.py the price instrument for within-category competition
    price.py          counterfactual price contexts and the aggregate price elasticity
    checkpoints.py    loading initialization artifacts and fitted checkpoints

Everything runs on the CPU in float64: scripts call torch.set_default_dtype(torch.float64)
before building a model (BasketModel checks).
"""
