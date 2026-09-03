import unittest

import numpy as np

from profile_rho0_size_likelihood import (evaluate_delta, profile_delta,
                                           size_gain, tilted_probability)


class Rho0ProfileTest(unittest.TestCase):
    def test_gain_identity_at_zero(self):
        probability = np.array([[0.2, 0.3, 0.5], [0.5, 0.4, 0.1]])
        observed = np.array([3, 1])
        np.testing.assert_allclose(
            size_gain(np.zeros(3), observed, np.log(probability)), 0.0,
            atol=1e-14)

    def test_profile_improves_misspecified_size_law(self):
        rng = np.random.default_rng(7)
        old = np.tile(np.array([0.7, 0.2, 0.08, 0.02]), (2000, 1))
        truth_delta = np.array([0.0, -0.7, -1.0, -1.2])
        truth = tilted_probability(truth_delta, np.log(old))
        observed = np.array([
            rng.choice(np.arange(1, 5), p=row) for row in truth])
        fitted, solve = profile_delta(
            observed, np.log(old), prior_mass=1.0, bound=5.0)
        audit = evaluate_delta(fitted, observed, np.log(old))
        self.assertGreater(audit["mean_loglik_gain"], 0.03)
        self.assertLess(solve["maximum_absolute_free_score"], 2e-5)
        fitted_probability = tilted_probability(fitted, np.log(old)).mean(0)
        empirical = np.bincount(observed, minlength=5)[1:] / len(observed)
        np.testing.assert_allclose(fitted_probability, empirical, atol=2e-3)

    def test_size_one_is_fixed_gauge(self):
        probability = np.tile(np.array([0.4, 0.35, 0.25]), (300, 1))
        observed = np.repeat(np.arange(1, 4), [60, 90, 150])
        fitted, _ = profile_delta(observed, np.log(probability))
        self.assertEqual(float(fitted[0]), 0.0)


if __name__ == "__main__":
    unittest.main()
