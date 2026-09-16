"""Tests for the version-3 trial-and-repeat tissue model."""
import numpy as np
import torch

import run_erim_tissue_prediction_test as v1
import run_erim_tissue_prediction_test_v2 as v2
import run_erim_tissue_prediction_test_v3 as v3


def _inputs(H=3, seed=0):
    rng = np.random.default_rng(seed)
    weeks = v1.erim_weeks(198527, 198532)
    T, J = len(weeks), len(v1.BRANDS)
    basket = np.zeros((H, T, J), dtype=bool)
    basket[..., v1.CTNL] = rng.random((H, T)) < 0.4
    basket[..., 0] = rng.random((H, T)) < 0.3
    data = {"weeks": weeks, "basket_t": torch.from_numpy(basket).double(),
            "opportunity_t": torch.ones(H, T, dtype=torch.bool),
            "redeemed_t": {v1.FIT_COUPON: torch.zeros(H, T, dtype=torch.bool)}}
    tensors = dict(dev=torch.zeros(J, T), level=torch.full((T,), 1.2),
                   log_avail=torch.zeros(H, T), since=torch.zeros(T))
    history = basket[..., v1.CTNL]
    tried = torch.from_numpy(np.cumsum(history, axis=1) - history > 0).double()
    return data, tensors, tried


def test_zero_lambda_reproduces_version_two_likelihood():
    data, tensors, tried = _inputs()
    old = v2.TissueModelV2(3, 3)
    new = v3.TissueModelV3(3, 3)
    new.load_state_dict({**old.state_dict(), "lam": torch.tensor(0.0)})
    new.tried = tried
    with torch.no_grad():
        assert torch.allclose(v2.log_likelihood(old, data, **tensors), v2.log_likelihood(new, data, **tensors))


def test_history_indicator_is_strictly_lagged():
    _, _, tried = _inputs(seed=7)
    data, _, _ = _inputs(seed=7)
    bought = data["basket_t"][..., v1.CTNL].numpy().astype(bool)
    for h in range(bought.shape[0]):
        first = np.flatnonzero(bought[h])
        if first.size:
            assert tried[h, : first[0] + 1].sum() == 0
            assert (tried[h, first[0] + 1:] == 1).all()


def test_positive_lambda_raises_simulated_repeat_purchasing():
    H, T, J = 200, 12, len(v1.BRANDS)
    weeks = v1.erim_weeks(198533, 198544)
    kwargs = dict(weeks=weeks, opportunity=np.ones((H, T), bool), log_avail=np.zeros((H, T)),
                  since=np.zeros(T), dev=np.zeros((J, T)), level=np.full(T, 1.2),
                  initial_hold=np.zeros(H), initial_tried=np.zeros(H, bool), schedule=(38,),
                  replicates=50, seed=3)
    results = {}
    for lam in (0.0, 1.5):
        model = v3.TissueModelV3(H, 3)
        with torch.no_grad():
            model.b.fill_(-3.0)
        model.fixed_lam = lam
        results[lam] = v3.simulate(model, **kwargs)["ctnl_weeks"][:, -4:].sum()
    assert results[1.5] > results[0.0] * 1.2
