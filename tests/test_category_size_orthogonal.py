import unittest

import numpy as np


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


if __name__ == "__main__":
    unittest.main()
