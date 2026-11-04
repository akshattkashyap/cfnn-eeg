"""DREAMER MAT-file, protocol and artifact checks; no model optimization."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.io import savemat

import collect
from cfnn import cli, data, diagnostics, dreamer, experiment, features
from cfnn.constants import DREAMER_CHANNELS, FS


def mat_fixture(subject_count=4):
    """Create the original nested MATLAB struct/cell layout, not a loader mock."""
    subjects = np.empty((1, subject_count), dtype=object)
    for si in range(subject_count):
        stimuli, baseline = np.empty((18, 1), dtype=object), np.empty((18, 1), dtype=object)
        for ti in range(18):
            time = np.arange(512 + ti * 128 + 7) / FS
            frequency = np.resize([6, 10, 14, 20, 30], 14)
            amplitude = np.linspace(1, 2, len(time))[:, None]
            stimuli[ti, 0] = amplitude * np.sin(2 * np.pi * time[:, None] * frequency)
            baseline[ti, 0] = np.full((128, 14), 1e12)
        subjects[0, si] = {
            "EEG": {"stimuli": stimuli, "baseline": baseline},
            "ScoreValence": (1 + (si + np.arange(18)) % 5).reshape(18, 1),
            "ScoreArousal": (1 + (2 * si + np.arange(18)) % 5).reshape(18, 1),
        }
    return {"Data": subjects, "EEG_SamplingRate": 128,
            "EEG_Electrodes": np.array(DREAMER_CHANNELS, dtype=object)}


def settings(path, *flags):
    return cli.parse_args(["--dataset", "dreamer", "--model", "htsk", "--data-dir", str(path),
                           "--window-samples", "128", *flags])


class DreamerTests(unittest.TestCase):
    def test_dataset_defaults_and_explicit_midpoint_policy(self):
        deap = cli.parse_args([])
        self.assertEqual((deap.dataset, deap.data_dir, deap.threshold, deap.threshold_op),
                         ("deap", "data/data_preprocessed_python", 5.0, "ge"))
        args = cli.parse_args(["--dataset", "dreamer", "--model", "htsk"])
        self.assertEqual((args.data_dir, args.threshold, args.threshold_op, args.feature_dim),
                         ("data/DREAMER.mat", 3.0, "ge", 70))
        ratings = np.array([1, 2, 3, 4, 5])
        np.testing.assert_array_equal(data.binarize(ratings, args.threshold, args.threshold_op),
                                      [0, 0, 1, 1, 1])
        args = cli.parse_args(["--dataset", "dreamer", "--threshold-op", "gt"])
        np.testing.assert_array_equal(data.binarize(ratings, args.threshold, args.threshold_op),
                                      [0, 0, 0, 1, 1])
        for flags in (["--threshold", "6"], ["--max-subjects", "24"],
                      ["--dreamer-last-seconds", "nan"], ["--dreamer-last-seconds", "-1"],
                      ["--dreamer-last-seconds", "0.1"]):
            with self.subTest(flags=flags), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    settings("unused", *flags)
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            cli.parse_args(["--dreamer-last-seconds", "60"])

    def test_variable_lengths_stimulus_only_channels_and_rating_alignment(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "DREAMER.mat"
            fixture = mat_fixture()
            savemat(path, {"DREAMER": fixture})
            args = settings(path)
            X, R, S, T = data.load_data(args)
            counts = np.array([(512 + ti * 128 + 7) // 128 for ti in range(18)])
            self.assertEqual(X.shape, (4 * counts.sum(), 70))
            np.testing.assert_array_equal(np.unique(S), [1, 2, 3, 4])
            self.assertEqual(len(np.unique(T)), 72)
            offset = 0
            for si in range(4):
                subj = fixture["Data"][0, si]
                for ti in range(18):
                    expected = features.trial_features(subj["EEG"]["stimuli"][ti, 0].T,
                        128, 128, 128, "hann", 70, "zero", features.band_masks(128))
                    stop = offset + counts[ti]
                    np.testing.assert_allclose(X[offset:stop], expected)
                    np.testing.assert_array_equal(R[offset:stop, 0], subj["ScoreValence"][ti, 0])
                    np.testing.assert_array_equal(R[offset:stop, 1], subj["ScoreArousal"][ti, 0])
                    np.testing.assert_array_equal(S[offset:stop], si + 1)
                    np.testing.assert_array_equal(T[offset:stop], si * 18 + ti)
                    offset = stop
            self.assertEqual(dreamer.source_file(path.parent), path)
            manifest = data.feature_manifest(args, data.source_files(args))
            self.assertEqual(manifest["channel_names"], list(DREAMER_CHANNELS))
            self.assertEqual(manifest["baseline_samples"], 0)
            self.assertEqual(manifest["dataset"], "dreamer")

    def test_cache_reuse_subject_limit_source_changes_crop_and_stride(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "DREAMER.mat"
            fixture = mat_fixture(5)
            savemat(path, {"DREAMER": fixture})
            args = settings(path, "--cache-dir", str(Path(tmp) / "cache"), "--max-subjects", "4")
            original = data.load_data(args)
            with patch.object(dreamer, "loadmat", side_effect=AssertionError("cache miss")):
                for actual, expected in zip(data.load_data(args), original):
                    np.testing.assert_array_equal(actual, expected)
            args.max_subjects = 5
            self.assertEqual(len(np.unique(data.load_data(args)[2])), 5)
            self.assertEqual(len(list((Path(tmp) / "cache").glob("*.npz"))), 2)
            fixture["Data"][0, 0]["ScoreValence"][0, 0] = 5
            savemat(path, {"DREAMER": fixture}, do_compression=True)
            self.assertEqual(data.load_data(args)[1][0, 0], 5)
            args.dreamer_last_seconds = 1.0
            cropped = data.load_data(args)
            self.assertEqual(cropped[0].shape, (5 * 18, 70))
            expected = features.trial_features(fixture["Data"][0, 0]["EEG"]["stimuli"][0, 0][-128:].T,
                128, 128, 128, "hann", 70, "zero", features.band_masks(128))
            np.testing.assert_allclose(cropped[0][:1], expected)
            args.dreamer_last_seconds = 2.0
            args.stride = 64
            args.feature_dim = 73
            overlap = data.load_data(args)
            self.assertEqual(overlap[0].shape, (5 * 18 * 3, 73))
            np.testing.assert_array_equal(overlap[0][:, -3:], 0)

    def test_channel_metadata_reorders_into_canonical_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "DREAMER.mat"
            fixture = mat_fixture()
            savemat(path, {"DREAMER": fixture})
            expected = data.load_data(settings(path))[0]
            fixture["EEG_Electrodes"] = fixture["EEG_Electrodes"][::-1]
            for subject in fixture["Data"][0]:
                for ti in range(18):
                    subject["EEG"]["stimuli"][ti, 0] = subject["EEG"]["stimuli"][ti, 0][:, ::-1]
            savemat(path, {"DREAMER": fixture})
            np.testing.assert_allclose(data.load_data(settings(path))[0], expected)
            del fixture["EEG_Electrodes"]
            for subject in fixture["Data"][0]:
                for ti in range(18):
                    subject["EEG"]["stimuli"][ti, 0] = subject["EEG"]["stimuli"][ti, 0][:, ::-1]
            savemat(path, {"DREAMER": fixture})
            np.testing.assert_allclose(data.load_data(settings(path))[0], expected)

    def test_rejects_invalid_mat_metadata_signals_and_ratings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "DREAMER.mat"
            for problem in ("sampling_rate", "rating", "rating_count", "channels", "signal",
                            "nan_signal", "trial_count", "missing_field", "crop"):
                with self.subTest(problem=problem):
                    fixture = mat_fixture()
                    first = fixture["Data"][0, 0]
                    flags = []
                    if problem == "sampling_rate":
                        fixture["EEG_SamplingRate"] = 256
                    elif problem == "rating":
                        first["ScoreValence"][0, 0] = 6
                    elif problem == "rating_count":
                        first["ScoreArousal"] = np.ones((17, 1))
                    elif problem == "channels":
                        fixture["EEG_Electrodes"][0] = "invalid"
                    elif problem == "signal":
                        first["EEG"]["stimuli"][0, 0] = np.ones((14, 512))
                    elif problem == "nan_signal":
                        first["EEG"]["stimuli"][0, 0][0, 0] = np.nan
                    elif problem == "trial_count":
                        first["EEG"]["stimuli"] = first["EEG"]["stimuli"][:17]
                    elif problem == "missing_field":
                        del first["ScoreValence"]
                    else:
                        flags = ["--dreamer-last-seconds", "60"]
                    savemat(path, {"DREAMER": fixture})
                    with self.assertRaises(ValueError):
                        data.load_data(settings(path, *flags))
            savemat(path, {"other": np.ones(3)})
            with self.assertRaisesRegex(ValueError, "DREAMER structure"):
                data.load_data(settings(path))
            with self.assertRaises(FileNotFoundError):
                dreamer.source_file(Path(tmp) / "missing.mat")

    def test_prepare_only_and_completed_artifacts_diagnostics_and_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            path = base / "DREAMER.mat"
            savemat(path, {"DREAMER": mat_fixture()})
            prepare = base / "prepared"
            args = settings(path, "--prepare-only", "--output-dir", str(prepare))
            with patch.object(experiment, "train_neural") as fit, \
                 patch.object(experiment.StandardScaler, "fit") as scaler:
                experiment.run(args)
                fit.assert_not_called()
                scaler.assert_not_called()
            preparation = json.loads((prepare / "preparation.json").read_text())
            self.assertEqual(preparation["dataset"], "dreamer")
            for overlaps in preparation["folds"][0]["overlap"].values():
                self.assertEqual(overlaps["subjects"], 0)
                self.assertEqual(overlaps["trials"], 0)
            out = base / "completed"
            # A trainer stub verifies I/O/scaling/dispatch without optimizing any model.
            args = settings(path, "--output-dir", str(out), "--cache-dir", str(base / "cache"))

            def fake_train(kind, settings, xtr, ytr, xva, yva, xte, ckpt):
                np.testing.assert_allclose(xtr.mean(0), 0, atol=1e-5)
                self.assertEqual(kind, "htsk")
                self.assertEqual(settings.threshold, 3.0)
                return np.full((len(xte), 2), .5), {"best_epoch": 1}

            with patch.object(experiment, "train_neural", side_effect=fake_train) as fit:
                experiment.run(args)
                self.assertEqual(fit.call_count, 2)
            restored = diagnostics.load_training_data(out)
            self.assertEqual(restored[0].dataset, "dreamer")
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(summary["dataset_reference"]["name"], "DREAMER")
            rows = collect.collect(base)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["dataset"], "dreamer")


if __name__ == "__main__":
    unittest.main()
