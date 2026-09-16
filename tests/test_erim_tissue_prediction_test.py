"""Exactness tests for the tissue-only energy model used in the coupon prediction test."""
import itertools

import numpy as np
import torch

import run_erim_tissue_prediction_test as tissue


def enumerate_baskets(u, rho, K):
    J = len(u)
    subsets = [s for k in range(K + 1) for s in itertools.combinations(range(J), k)]
    energy = np.array([u[list(s)].sum() - rho[len(s)] for s in subsets])
    p = np.exp(energy - np.logaddexp.reduce(energy))
    return subsets, p


def test_basket_log_prob_matches_enumeration_including_empty_basket():
    torch.manual_seed(1)
    J, K = len(tissue.BRANDS), 3
    model = tissue.TissueModel(1, K)
    with torch.no_grad():
        model.rho_free.copy_(torch.tensor([0.7, 1.9]))
    u = torch.randn(J) - 1.0
    subsets, p = enumerate_baskets(u.numpy(), model.rho().detach().numpy(), K)
    assert np.isclose(p.sum(), 1.0)
    for index in (0, 1, 20, len(subsets) - 1):
        basket = torch.zeros(J)
        basket[list(subsets[index])] = 1.0
        assert np.isclose(float(model.basket_log_prob(u, basket)), np.log(p[index]))


def test_cottonelle_marginals_match_enumeration():
    rng = np.random.default_rng(3)
    J, K = len(tissue.BRANDS), 4
    rho = np.array([0.0, 0.0, 1.1, 2.3, 3.0])
    u = rng.normal(-1.5, 1.0, size=J)
    subsets, p = enumerate_baskets(u, rho, K)
    c = tissue.CTNL
    expected = (sum(pi for s, pi in zip(subsets, p) if c in s),
                sum(pi for s, pi in zip(subsets, p) if s),
                sum(pi for s, pi in zip(subsets, p) if set(s) - {c}))
    got = tissue.ctnl_marginals(np.delete(u, c)[None, :], np.array([u[c]]), rho, K)
    assert np.allclose([float(g[0]) for g in got], expected)


def test_forward_recursion_reduces_to_basket_likelihood_without_coupon_exposure():
    rng = np.random.default_rng(5)
    weeks = tissue.erim_weeks(198527, 198532)
    H, T, J, K = 4, len(weeks), len(tissue.BRANDS), 3
    basket = np.zeros((H, T, J), dtype=bool)
    basket[rng.random((H, T)) < 0.4, tissue.CTNL] = True
    basket[..., 0] |= rng.random((H, T)) < 0.3
    data = {"weeks": weeks, "basket_t": torch.from_numpy(basket).double(),
            "opportunity_t": torch.from_numpy(rng.random((H, T)) < 0.8),
            "redeemed_t": {tissue.FIT_COUPON: torch.zeros(H, T, dtype=torch.bool)}}
    dev = torch.from_numpy(rng.normal(0, 0.05, size=(J, T)))
    level = torch.full((T,), 1.2)
    model = tissue.TissueModel(H, K)
    with torch.no_grad():
        model.q_raw.fill_(-40.0)
        model.alpha.copy_(torch.randn(H))
        ll = tissue.household_log_likelihood(model, data, dev, level)
        base = model.basket_log_prob(model.utilities(dev), data["basket_t"])
        expected = (base * data["opportunity_t"]).sum(1)
    assert torch.allclose(ll, expected, atol=1e-8)


def test_erim_weeks_roll_over_year_end():
    assert tissue.erim_weeks(198551, 198602) == [198551, 198552, 198601, 198602]
