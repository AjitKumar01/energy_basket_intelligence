import pytest
import torch

from audit_original_probability import exact_covariance_response
from audit_probability_foundations import exact_energy, make_world


def test_original_audit_first_adjoint_matches_enumerated_covariance():
    from poly_degree_native import _extension
    if _extension is None:
        pytest.skip("native DP must be built for the covariance oracle")
    model, ix, membership, _ = make_world(
        seed=91723, products=7, contexts=3, nmax=4, strength=0, kappa=3.7)
    mean, variance, slope, coefficient = exact_covariance_response(model, ix)
    with torch.no_grad():
        probability = exact_energy(model, ix, membership).softmax(-1)
        size = membership.sum(-1)
        g = torch.zeros(ix.B, model.J).index_put((ix.item_trip, ix.item), coefficient)
        price_statistic = g @ membership.T
        exact_mean = probability @ size
        exact_variance = probability @ size.square() - exact_mean.square()
        exact_slope = -((probability * price_statistic) @ size
                        - exact_mean * (probability * price_statistic).sum(-1))
    assert torch.allclose(mean, exact_mean, atol=1e-11, rtol=1e-11)
    assert torch.allclose(variance, exact_variance, atol=1e-11, rtol=1e-11)
    assert torch.allclose(slope, exact_slope, atol=1e-11, rtol=1e-11)
    assert all(parameter.grad is None for parameter in model.parameters())
