import math

import numpy as np
import torch

from audit_synthetic_interactions import (enumerate_support, gram_pair_energy,
                                          joint_log_probability)


def test_enumerated_support_has_every_nonempty_basket_through_nmax():
    baskets, membership, sizes, *_ = enumerate_support(6, 3, 2)
    assert len(baskets) == sum(math.comb(6, n) for n in range(1, 4))
    assert membership.shape == (41, 6)
    assert set(sizes.tolist()) == {1, 2, 3}


def test_gram_identity_matches_explicit_pair_sum():
    baskets, membership, *_ = enumerate_support(5, 4, 2)
    phi = torch.arange(15, dtype=torch.float64).reshape(5, 3) / 10
    got = gram_pair_energy(membership, phi)
    expected = []
    for basket in baskets:
        expected.append(sum(float(phi[j] @ phi[k]) for position, j in enumerate(basket)
                            for k in basket[position + 1:]))
    np.testing.assert_allclose(got.numpy(), expected, atol=1e-12)


def test_zero_gram_is_exactly_the_additive_joint_law():
    _, membership, sizes, _, _, category_pairs = enumerate_support(6, 3, 2)
    b = torch.linspace(-1, 1, 12).reshape(2, 6)
    rho_size = torch.tensor([0.0, 0.2, 0.7])
    rho_category = torch.tensor([0.1, -0.05])
    zero = torch.zeros(6, 2)
    additive = joint_log_probability(
        b, rho_size, rho_category, None, membership, sizes, category_pairs)
    interaction = joint_log_probability(
        b, rho_size, rho_category, zero, membership, sizes, category_pairs)
    torch.testing.assert_close(additive, interaction, atol=1e-14, rtol=0)
