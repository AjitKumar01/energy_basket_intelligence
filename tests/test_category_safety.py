import numpy as np
import torch

from category_safety import (attractive_category_rewards,
                             project_category_reward_, support_lower_bounds)


class Model:
    def __init__(self, values):
        self.rho_c = torch.nn.Parameter(torch.tensor(values, dtype=torch.float64))


def test_support_bounds_preserve_narrow_pairs_but_scale_broad_groups():
    got = support_lower_bounds(np.array([1, 2, 4, 120]), 1.5, -1.5)
    np.testing.assert_allclose(got, [-1.5, -1.5, -0.25, -1.5 / 7140.0])


def test_projection_caps_every_category_without_changing_safe_values():
    capacities = np.array([2, 4, 120])
    model = Model([-1.5, -0.1, 0.2])
    result = project_category_reward_(model, capacities, 1.5)
    np.testing.assert_allclose(model.rho_c.detach().numpy(), [-1.5, -0.1, 0.2])
    assert result["projected_coefficients"] == 0
    assert float(attractive_category_rewards(model.rho_c, capacities).max()) <= 1.5

    model.rho_c.data[:] = torch.tensor([-2.0, -1.0, -0.1])
    result = project_category_reward_(model, capacities, 1.5)
    assert result["projected_coefficients"] == 3
    assert result["maximum_reward_after"] <= 1.5 + 1e-12


def test_projection_clears_only_outward_adam_momentum_at_bound():
    capacities = np.array([4, 120, 2])
    model = Model([-1.0, -0.1, 0.1])
    optimizer = torch.optim.AdamW([model.rho_c], lr=0.01)
    optimizer.state[model.rho_c]["exp_avg"] = torch.tensor(
        [2.0, 3.0, 4.0], dtype=torch.float64)
    optimizer.state[model.rho_c]["exp_avg_sq"] = torch.ones(3, dtype=torch.float64)
    result = project_category_reward_(model, capacities, 1.5, optimizer=optimizer)
    # First two coordinates are projected to their lower bounds; the safe third is not.
    np.testing.assert_allclose(
        optimizer.state[model.rho_c]["exp_avg"].numpy(), [0.0, 0.0, 4.0])
    assert result["cleared_outward_moments"] == 2
