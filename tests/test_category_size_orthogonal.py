import unittest

import numpy as np
import torch

from fit_multifidelity_rank8 import project_rho_c_trust_region


class _Model:
    def __init__(self, value):
        self.rho_c = torch.nn.Parameter(torch.as_tensor(value, dtype=torch.float64))


class CategorySizeOrthogonalTests(unittest.TestCase):
    def test_energy_identity(self):
        rng = np.random.default_rng(4)
        categories, nmax = 5, 9
        rho = rng.normal(size=categories)
        rho0 = rng.normal(size=nmax)
        reference = rng.normal(size=(categories, nmax))
        for n in range(1, nmax + 1):
            count = rng.multinomial(n, np.full(categories, 1 / categories))
            statistic = count * (count - 1) / 2
            original = -rho @ statistic - rho0[n - 1]
            rho0_tilde = rho0[n - 1] + rho @ reference[:, n - 1]
            centred = (-rho @ (statistic - reference[:, n - 1])
                       - rho0_tilde)
            self.assertAlmostEqual(original, centred, places=13)

    def test_compensated_update_is_centred_score(self):
        rng = np.random.default_rng(8)
        rho = rng.normal(size=4)
        delta = rng.normal(size=4) * .01
        rho0 = rng.normal(size=7)
        reference = rng.normal(size=(4, 7))
        n = 5
        count = np.asarray([2, 1, 1, 1])
        statistic = count * (count - 1) / 2
        old = -rho @ statistic - rho0[n - 1]
        new_rho0 = rho0 - delta @ reference
        new = -(rho + delta) @ statistic - new_rho0[n - 1]
        expected_change = -delta @ (statistic - reference[:, n - 1])
        self.assertAlmostEqual(new - old, expected_change, places=13)

    def test_trust_projection(self):
        model = _Model([3.0, 4.0])
        norm, changed = project_rho_c_trust_region(
            model, torch.zeros(2, dtype=torch.float64), 2.0)
        self.assertTrue(changed)
        self.assertAlmostEqual(norm, 2.0)
        self.assertLess(abs(float(model.rho_c.detach().norm()) - 2.0), 1e-14)


if __name__ == "__main__":
    unittest.main()
