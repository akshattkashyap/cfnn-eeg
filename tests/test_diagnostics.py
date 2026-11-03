"""Non-training checks for diagnostic isolation and artifact consistency."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from cfnn import diagnostics
from cfnn import diagnostic_cli
from cfnn.diagnostic_cli import parse_args


class DiagnosticTests(unittest.TestCase):
    def test_batch_is_balanced_training_only_and_one_row_per_trial(self):
        trial = np.repeat(np.arange(40), 3)
        y = (trial % 2).astype(np.int64)
        train = np.arange(90)
        rows = diagnostics.balanced_rows(y, trial, train, size=16)
        self.assertTrue(np.isin(rows, train).all())
        self.assertEqual(len(np.unique(trial[rows])), 16)
        np.testing.assert_array_equal(np.bincount(y[rows]), [8, 8])
        np.testing.assert_array_equal(rows, diagnostics.balanced_rows(y, trial, train, 16))
        y[90:] = 1 - y[90:]  # Held-out labels must not affect selection.
        np.testing.assert_array_equal(rows, diagnostics.balanced_rows(y, trial, train, 16))

    def test_rejects_inconsistent_trials_and_insufficient_class_trials(self):
        trial = np.repeat(np.arange(4), 3)
        y = (trial % 2).astype(np.int64)
        with self.assertRaisesRegex(ValueError, "Need at least"):
            diagnostics.balanced_rows(y, trial, np.arange(12), size=8)
        y[0] = 1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            diagnostics.balanced_rows(y, trial, np.arange(12), size=4)

    def test_detects_reordered_data_and_overlapping_partitions(self):
        ratings = np.arange(16).reshape(8, 2)
        subject, trial = np.repeat([1, 2], 4), np.arange(8)
        saved = {"ratings": ratings.copy(), "subject": subject, "trial": trial}
        split = {"train": np.arange(4), "validation": np.arange(4, 6), "test": np.arange(6, 8)}
        diagnostics.validate_saved_rows(ratings, subject, trial, saved, split)
        with self.assertRaisesRegex(ValueError, "ratings"):
            diagnostics.validate_saved_rows(ratings[::-1], subject, trial, saved, split)
        split["test"] = np.array([5, 7])
        with self.assertRaisesRegex(ValueError, "exactly once"):
            diagnostics.validate_saved_rows(ratings, subject, trial, saved, split)

    def test_reports_identical_features_with_conflicting_labels(self):
        X = np.array([[1., 0.], [1., 0.], [2., 0.], [2., 0.]])
        scaler = StandardScaler().fit(X)
        x, report = diagnostics.batch_features(X, np.arange(4), scaler, np.array([0, 1, 0, 1]))
        self.assertEqual(x.shape, (4, 2, 1))
        self.assertEqual(report["constant_training_columns"], [1])
        self.assertEqual(report["identical_feature_groups_with_conflicting_labels"], 2)

    def test_saved_scaler_must_match_current_training_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "summary.json").write_text("{}")
            (root / "config.json").write_text(json.dumps({"feature_dim": 3}))
            X = np.arange(36, dtype=np.float32).reshape(12, 3)
            R = np.tile([[3., 7.], [6., 4.]], (6, 1))
            subject, trial = np.repeat([1, 2, 3], 4), np.arange(12)
            np.savez(root / "row_metadata.npz", ratings=R, subject=subject, trial=trial)
            np.savez(root / "split_f0.npz", train=np.arange(6), validation=np.arange(6, 9), test=np.arange(9, 12))
            joblib.dump(StandardScaler().fit(X[:6]), root / "scaler_f0.joblib")
            with patch.object(diagnostics, "load_deap", return_value=(X, R, subject, trial)):
                result = diagnostics.load_training_data(root)
                np.testing.assert_array_equal(result[5], np.arange(6))
                joblib.dump(StandardScaler().fit(X), root / "scaler_f0.joblib")
                with self.assertRaisesRegex(ValueError, "scaler"):
                    diagnostics.load_training_data(root)

    def test_cli_rejects_invalid_memorization_settings(self):
        for flags in [["--batch-size", "3"], ["--steps", "0"], ["--learning-rates", "nan"]]:
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(["--run-dir", "unused", *flags])

    def test_mode_comparison_restores_moving_statistics(self):
        class Tensor:
            def __init__(self, value):
                self.value = np.asarray(value)

            def numpy(self):
                return self.value

            def assign(self, value):
                self.value = np.array(value)

        class Model:
            def __init__(self):
                self.non_trainable_variables = [Tensor([4.])]

            def __call__(self, x, training):
                if training:
                    self.non_trainable_variables[0].assign([999.])
                    return Tensor([[.9, .1], [.1, .9]])
                return Tensor([[.9, .1], [.9, .1]])

        model = Model()
        result = diagnostics.batch_scores(model, np.zeros((2, 3, 1)), np.array([0, 1]))
        self.assertEqual(result["training_mode"]["accuracy"], 1)
        self.assertEqual(result["inference_mode"]["accuracy"], .5)
        self.assertEqual(result["prediction_disagreement_fraction"], .5)
        np.testing.assert_array_equal(model.non_trainable_variables[0].numpy(), [4.])

    def test_check_only_writes_report_and_never_fits(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "diagnostic"
            saved_args = type("Config", (), {"window_samples": 32, "nfft": 0, "taper": "hann",
                "feature_dim": 3, "pad_mode": "zero", "fuzzy_agg": "mean_exp",
                "seed": 42, "clipnorm": 0., "threshold": 5., "threshold_op": "ge"})()
            X = np.random.default_rng(4).normal(size=(24, 3)).astype(np.float32)
            R = np.tile([[3., 7.], [6., 4.]], (12, 1))
            trial, subject = np.arange(24), np.repeat(np.arange(3), 8)
            train = np.arange(16)
            scaler = StandardScaler().fit(X[train])
            with patch.object(diagnostic_cli, "load_training_data", return_value=(
                    saved_args, X, R, subject, trial, train, scaler)), \
                 patch.object(diagnostic_cli, "memorize_batch") as fit:
                diagnostic_cli.main(["--run-dir", tmp, "--output-dir", str(out),
                                     "--batch-size", "8", "--target", "both", "--check-only"])
                fit.assert_not_called()
            report = json.loads((out / "diagnostics.json").read_text())
            self.assertTrue(report["uses_training_rows_only"])
            self.assertTrue(report["check_only"])
            for target in ("valence", "arousal"):
                selected = report["targets"][target]["batch_rows"]
                self.assertTrue(np.isin(selected, train).all())
                self.assertEqual(report["targets"][target]["class_counts"], [4, 4])
                self.assertEqual(report["targets"][target]["runs"], [])


if __name__ == "__main__":
    unittest.main()
