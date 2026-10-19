"""Scientific checks for the post-hoc readout references. James Edward Ball."""
import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('references', Path(__file__).resolve().parents[1] / 'check_readout_references.py')
references = importlib.util.module_from_spec(spec)
spec.loader.exec_module(references)
a = references.analysis


class ReadoutReferenceTests(unittest.TestCase):
    def test_uniform_order_matches_analytic_and_exact_tie_expectation(self):
        expected = sum(1 / rank for rank in range(1, 9)) / 8
        # Across these cyclic full-gallery orders, each label visits each rank once.
        scores = np.array([np.roll(np.arange(8), shift) for shift in range(8)])
        for category in range(8):
            ranks = a.reciprocal_rank(np.full(8, category), scores)
            self.assertAlmostEqual(ranks.mean(), expected)
        np.testing.assert_allclose(a.reciprocal_rank(np.arange(8), np.zeros((8, 8))), expected)

    def test_constant_targets_use_only_permitted_fitting_categories(self):
        permitted = np.array([False, False, True, True, True, True, True, True])
        weights = references.target_weights(np.array([2, 2, 3, 4, 5, 6, 7]), permitted)
        for value in weights.values():
            self.assertAlmostEqual(value.sum(), 1)
            np.testing.assert_array_equal(value[:2], 0)
        self.assertAlmostEqual(weights['train_frequency'][2], 2 / 7)
        self.assertAlmostEqual(weights['uniform_permitted_categories'][2], 1 / 6)
        with self.assertRaises(ValueError):
            references.target_weights(np.array([0, 2]), permitted)

    def test_ridge_text_identity_in_primal_and_dual_with_intercept(self):
        rng = np.random.default_rng(462)
        for width in (5, 40):
            with self.subTest(width=width):
                x = rng.normal(size=(24, width)) + 4
                val, test = rng.normal(size=(11, width)), rng.normal(size=(13, width))
                # Unbalanced fitting labels exercise the nonzero target mean.
                z = np.eye(8)[np.r_[np.zeros(12, dtype=int), rng.integers(0, 6, 12)]]
                t = a.unit(rng.normal(size=(8, 19)))
                for (alpha, _, p), (alpha2, _, q) in zip(a.ridge_predictions(x, z, val, test), a.ridge_predictions(x, z @ t, val, test)):
                    self.assertEqual(alpha, alpha2)
                    np.testing.assert_allclose(p @ t, q, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(p[:, 6:], 0, atol=1e-12, rtol=0)
                    np.testing.assert_allclose(a.language_scores(p, t), a.unit(q) @ t.T, atol=1e-12, rtol=0)


if __name__ == '__main__':
    unittest.main()
