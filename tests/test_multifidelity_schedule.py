import torch

from fit_multifidelity_rank8 import (decay_learning_rates,
                                     install_spectral_transform,
                                     spectral_transform_gradient)


def test_group_learning_rate_decay_preserves_separation_until_floor():
    first = torch.nn.Parameter(torch.zeros(()))
    second = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.AdamW([
        {"params": [first], "lr": 5e-4},
        {"params": [second], "lr": 1e-4},
    ])

    old, new = decay_learning_rates(optimizer, 0.5, 3.125e-5)
    assert old == [5e-4, 1e-4]
    assert new == [2.5e-4, 5e-5]

    _, new = decay_learning_rates(optimizer, 0.5, 3.125e-5)
    assert new == [1.25e-4, 3.125e-5]
    assert [group["lr"] for group in optimizer.param_groups] == new


def test_spectral_transform_chain_rule_and_reconstruction():
    torch.manual_seed(4)
    basis = torch.randn(11, 4, dtype=torch.float64)
    transform = torch.randn(4, 4, dtype=torch.float64, requires_grad=True)
    direction = torch.randn(11, 4, dtype=torch.float64)
    direct = ((basis @ transform) * direction).sum()
    direct.backward()

    assert torch.allclose(
        spectral_transform_gradient(direction, basis, 4), transform.grad)

    phi = torch.empty(11, 7, dtype=torch.float64)
    install_spectral_transform(phi, basis, transform.detach(), 4)
    assert torch.allclose(phi[:, :4], basis @ transform.detach())
    assert torch.count_nonzero(phi[:, 4:]) == 0
