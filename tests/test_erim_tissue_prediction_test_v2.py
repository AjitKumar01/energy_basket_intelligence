"""Tests for the version-2 tissue model: availability, direct redemption and coupon response."""
import numpy as np
import pandas as pd
import torch

import run_erim_tissue_prediction_test as v1
import run_erim_tissue_prediction_test_v2 as v2


def test_availability_mixes_household_store_shares_over_launched_stores():
    weights = pd.DataFrame({2: [1.0, 0.25], 3: [0.0, 0.75]}, index=[10, 11])
    pooled = pd.Series({2: 0.5, 3: 0.5})
    launch = {2: 198522, 3: 198530}
    weeks = [198521, 198522, 198530]
    got = v2.availability(np.array([10, 11, 12]), weeks, weights, pooled, launch)
    assert np.allclose(got, [[0, 1, 1], [0, 0.25, 1], [0, 0.5, 1]])


def test_weeks_since_launch_is_zero_before_launch_and_crosses_year_end():
    got = v2.weeks_since_launch([198520, 198521, 198552, 198601])
    assert got.tolist() == [0, 0, 31, 32]


def test_coupon_alternative_equals_explicit_two_alternative_choice():
    """Holding a coupon = Cottonelle plus a mutually exclusive 'Cottonelle with coupon' item."""
    rng = np.random.default_rng(2)
    J, K = len(v1.BRANDS), 3
    rho = np.array([0.0, 0.0, 1.0, 2.0])
    u = rng.normal(-1.2, 0.8, size=J)
    x = 0.9
    p_c, p_any, _ = v1.ctnl_marginals(np.delete(u, v1.CTNL)[None, :], np.array([u[v1.CTNL] + np.log1p(x)]), rho, K)
    # explicit enumeration with an extra alternative that cannot co-occur with plain Cottonelle
    import itertools
    items = list(range(J)) + ["coupon"]
    util = dict(enumerate(u)); util["coupon"] = u[v1.CTNL] + np.log(x)
    energy, has_c, nonempty = [], [], []
    for k in range(K + 1):
        for s in itertools.combinations(items, k):
            if v1.CTNL in s and "coupon" in s:
                continue
            energy.append(sum(util[i] for i in s) - rho[len(s)])
            has_c.append(v1.CTNL in s or "coupon" in s)
            nonempty.append(len(s) > 0)
    p = np.exp(np.array(energy) - np.logaddexp.reduce(energy))
    assert np.isclose(float(p_c[0]), p[np.array(has_c)].sum())
    assert np.isclose(float(p_any[0]), p[np.array(nonempty)].sum())


def test_zero_value_coupon_has_no_effect_on_likelihood():
    rng = np.random.default_rng(4)
    weeks = v1.erim_weeks(198527, 198532)
    H, T, J = 3, len(weeks), len(v1.BRANDS)
    basket = np.zeros((H, T, J), dtype=bool)
    basket[..., v1.CTNL] = rng.random((H, T)) < 0.4
    data = {"weeks": weeks, "basket_t": torch.from_numpy(basket).double(),
            "opportunity_t": torch.ones(H, T, dtype=torch.bool),
            "redeemed_t": {v1.FIT_COUPON: torch.zeros(H, T, dtype=torch.bool)}}
    tensors = dict(dev=torch.zeros(J, T), level=torch.full((T,), 1.2),
                   log_avail=torch.zeros(H, T), since=torch.zeros(T))
    model = v2.TissueModelV2(H, 3)
    with torch.no_grad():
        model.kappa_raw.fill_(-60.0)  # kappa ~ 0
        ll = v2.log_likelihood(model, data, **tensors)
        base = model.basket_log_prob(model.utilities(tensors["dev"], tensors["log_avail"], tensors["since"]),
                                     data["basket_t"]).sum(1)
    assert torch.allclose(ll, base, atol=1e-6)
