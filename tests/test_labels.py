"""Checks for rating replacement and failure on ambiguous alignment."""
import unittest

import numpy as np

from cfnn.labels import corrected_labels


class LabelRepairTests(unittest.TestCase):
    def test_replaces_only_first_two_columns_and_preserves_original(self):
        expected = np.tile([7.71, 7.6, 6.9, 7.83], (40, 1))
        actual = expected.copy()
        actual[:3, :2] = 9 - actual[:3, :2]
        original = actual.copy()
        result = corrected_labels(actual, expected)
        np.testing.assert_allclose(result, expected)
        np.testing.assert_array_equal(actual, original)
        np.testing.assert_array_equal(result[:, 2:], original[:, 2:])
        np.testing.assert_array_equal(corrected_labels(result, expected), result)

    def test_refuses_unverified_alignment(self):
        expected = np.tile([7.71, 7.6, 6.9, 7.83], (40, 1))
        actual = expected.copy()
        actual[5, 2] = 1
        with self.assertRaisesRegex(ValueError, "alignment"):
            corrected_labels(actual, expected)

    def test_refuses_nonfinite_or_wrong_shape(self):
        expected = np.full((40, 4), 6.0)
        for actual in (np.zeros((40, 2)), np.full((40, 4), np.nan)):
            with self.assertRaises(ValueError):
                corrected_labels(actual, expected)


if __name__ == "__main__":
    unittest.main()
