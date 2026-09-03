"""Tests for identified segment representations and distribution metrics."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from audit_customer_segments import divergence, induced_representation


class CustomerSegmentAuditTest(unittest.TestCase):
    def test_induced_representation_preserves_surface_similarity_under_rebasis(self):
        rng = np.random.default_rng(41)
        left = rng.normal(size=(12, 4))
        right = rng.normal(size=(20, 4))
        transform = rng.normal(size=(4, 4)) + 3.0 * np.eye(4)
        changed_left = left @ transform
        changed_right = right @ np.linalg.inv(transform).T
        first = induced_representation(left, right)
        second = induced_representation(changed_left, changed_right)
        np.testing.assert_allclose(first @ first.T, second @ second.T,
                                   atol=2e-10, rtol=2e-10)

    def test_distribution_divergence_is_zero_for_identical_counts(self):
        count = np.array([0.0, 2.0, 8.0, 1.0])
        result = divergence(count, count)
        self.assertLess(abs(result["kl_observed_to_generated"]), 1e-14)
        self.assertLess(abs(result["jensen_shannon"]), 1e-14)
        self.assertLess(abs(result["total_variation"]), 1e-14)


if __name__ == "__main__":
    unittest.main()
