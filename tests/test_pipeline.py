"""Non-training checks. TensorFlow checks skip when TensorFlow is unavailable."""
import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import collect
from cfnn import cli, constants, data as dataset, experiment, features, metrics, models
from cfnn import splits as splitting


class FeatureTests(unittest.TestCase):
    def test_band_boundaries_have_no_overlap(self):
        masks = np.stack(features.band_masks(128))
        self.assertLessEqual(masks.sum(0).max(), 1)
        self.assertTrue(masks[-1, 45])
        self.assertFalse(masks[-1, 46])
        self.assertFalse(masks[0, 8])
        self.assertTrue(masks[1, 8])

    def test_sinusoids_channel_order_and_padding(self):
        freqs = np.resize([6, 10, 14, 20, 30], 14)
        time = np.arange(1024) / constants.FS
        eeg = np.sin(2 * np.pi * freqs[:, None] * time)
        masks = features.band_masks(512)
        x = features.trial_features(eeg, 512, 512, 512, "hann", 73, "zero", masks)
        self.assertEqual(x.shape, (2, 73))
        np.testing.assert_array_equal(x[:, -3:], 0)
        expected = np.resize(np.arange(5), 14)
        np.testing.assert_array_equal(x[0, :70].reshape(14, 5).argmax(1), expected)
        summary = features.trial_features(eeg, 512, 512, 512, "hann", 73, "summary", masks)
        np.testing.assert_allclose(summary[:, -3:], x[:, :70].reshape(2, 14, 5).mean(1)[:, [0, 1, 4]], atol=1e-5)

    def test_paper_window_sweep_is_finite(self):
        eeg = np.random.default_rng(2).normal(size=(14, 7680))
        for window in [4, 8, 16, 32, 64, 128, 512]:
            with self.subTest(window=window):
                nfft = max(window, 128)
                x = features.trial_features(eeg, window, window, nfft, "hann", 70, "zero", features.band_masks(nfft))
                self.assertEqual(x.shape, (7680 // window, 70))
                self.assertTrue(np.isfinite(x).all())

    def test_loader_drops_baseline_preserves_ids_and_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for sid in range(1, 5):
                (root / f"s{sid:02d}.dat").write_bytes(b"fixture")
            args = cli.parse_args(["--data-dir", tmp, "--cache-dir", str(root / "cache")])
            # Broadcast a fixture to the full shape without allocating a DEAP dataset.
            time = np.arange(8064) / 128
            signal = np.sin(2 * np.pi * 10 * time)
            signal[:384] = 1e9
            data = np.broadcast_to(signal, (40, 40, 8064))
            labels = np.full((40, 4), 6.0)
            labels[::2, 0] = 3
            labels[1, 0] = 0  # The local DEAP copy contains finite ratings below 1.
            fixture = {"data": data, "labels": labels}
            with patch.object(dataset.pickle, "load", return_value=fixture) as loader, self.assertWarns(UserWarning):
                x, ratings, subject, trial = dataset.load_deap(args)
                self.assertEqual(loader.call_count, 4)
            self.assertEqual(x.shape, (2400, 73))
            np.testing.assert_array_equal(np.unique(subject), [1, 2, 3, 4])
            self.assertEqual(len(np.unique(trial)), 160)
            np.testing.assert_array_equal(ratings[:15, 0], 3)
            np.testing.assert_array_equal(ratings[15:30, 0], 0)
            expected = features.trial_features(data[0, constants.CH0, 384:], 512, 512, 512, "hann", 73,
                                            "zero", features.band_masks(512))
            np.testing.assert_allclose(x[:15], expected)
            with patch.object(dataset.pickle, "load", side_effect=AssertionError("cache miss")):
                np.testing.assert_array_equal(dataset.load_deap(args)[0], x)
            (root / "s01.dat").write_bytes(b"changed-fixture")
            with patch.object(dataset.pickle, "load", return_value=fixture) as loader, self.assertWarns(UserWarning):
                dataset.load_deap(args)
                self.assertEqual(loader.call_count, 4)

    def test_invalid_arguments_and_threshold_equality(self):
        np.testing.assert_array_equal(dataset.binarize(np.array([4, 5, 6]), 5, "ge"), [0, 1, 1])
        np.testing.assert_array_equal(dataset.binarize(np.array([4, 5, 6]), 5, "gt"), [0, 0, 1])
        for argv in [["--window-samples", "0"], ["--nfft", "32"], ["--cv-folds", "1"],
                     ["--val-size", "1"], ["--learning-rate", "nan"]]:
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    cli.parse_args(argv)


class SplitAndMetricTests(unittest.TestCase):
    def test_all_split_modes_holdout_and_cv(self):
        subject = np.repeat(np.arange(8), 24)
        trial = np.repeat(np.arange(48), 4)
        for mode in ["random", "trial", "subject"]:
            for cv in [0, 4]:
                with self.subTest(mode=mode, cv=cv):
                    args = cli.parse_args(["--split", mode, "--cv-folds", str(cv)])
                    splits = splitting.make_splits(args, subject, trial)
                    repeated = splitting.make_splits(args, subject, trial)
                    self.assertEqual(len(splits), cv or 1)
                    for split, other in zip(splits, repeated):
                        for x, y in zip(split, other):
                            np.testing.assert_array_equal(x, y)
                        np.testing.assert_array_equal(np.sort(np.concatenate(split)), np.arange(192))
                        audit = splitting.split_audit(*split, subject, trial)
                        for overlap in audit["overlap"].values():
                            self.assertEqual(overlap["rows"], 0)
                            if mode != "random":
                                self.assertEqual(overlap["trials"], 0)
                            if mode == "subject":
                                self.assertEqual(overlap["subjects"], 0)
                    if cv:
                        np.testing.assert_array_equal(np.sort(np.concatenate([s[2] for s in splits])), np.arange(192))

    def test_trial_averages_scores_before_decision(self):
        y = np.array([0, 0, 0, 1])
        scores = np.array([[.51, .49], [.51, .49], [.01, .99], [.1, .9]])
        trial = np.array([0, 0, 0, 1])
        subject = np.array([1, 1, 1, 2])
        result = metrics.trial_level(y, scores, trial, subject)
        self.assertEqual(result["accuracy"], .5)
        self.assertEqual(result["confusion_matrix"], [[0, 1], [0, 1]])
        self.assertEqual(metrics.metrics(y, scores)["accuracy"], .75)
        with self.assertRaises(ValueError):
            metrics.trial_level(np.array([0, 1, 0, 1]), scores, trial, subject)
        with self.assertRaises(ValueError):
            metrics.metrics(y, np.full((4, 2), np.nan))

    def test_single_subject_ci_is_unavailable(self):
        self.assertIsNone(metrics.bootstrap_ci([.75]))

    def test_prepare_only_never_fits_and_writes_reproducible_splits(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            args = cli.parse_args(["--prepare-only", "--output-dir", str(out)])
            rng = np.random.default_rng(8)
            x = rng.normal(size=(192, 73)).astype(np.float32)
            ratings = np.tile([[3, 6], [7, 4]], (96, 1)).astype(np.float32)
            subject, trial = np.repeat(np.arange(8), 24), np.repeat(np.arange(48), 4)
            with patch.object(experiment, "source_files", return_value=[]), \
                 patch.object(experiment, "load_data", return_value=(x, ratings, subject, trial)), \
                 patch.object(experiment, "train_neural") as neural, \
                 patch.object(experiment, "train_sklearn") as baseline, \
                 patch.object(experiment.StandardScaler, "fit") as scaler:
                experiment.run(args)
                neural.assert_not_called()
                baseline.assert_not_called()
                scaler.assert_not_called()
            self.assertTrue((out / "preparation.json").exists())
            self.assertTrue((out / "split_f0.npz").exists())
            self.assertFalse((out / "summary.json").exists())
            self.assertEqual(collect.collect(tmp), [])
            with self.assertRaisesRegex(ValueError, "not empty"):
                experiment.run(args)

    def test_run_artifacts_scaler_and_collection_without_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "run"
            args = cli.parse_args(["--model", "knn", "--output-dir", str(out)])
            rng = np.random.default_rng(8)
            x = rng.normal(size=(192, 73)).astype(np.float32)
            subject, trial = np.repeat(np.arange(8), 24), np.repeat(np.arange(48), 4)
            x += subject[:, None] * 10  # Whole-dataset scaling would visibly leak.
            ratings = np.stack([np.where(trial % 2, 7, 3), np.where(trial % 2, 4, 6)], 1)

            def fake_train(kind, settings, xtr, ytr, xte, path):
                np.testing.assert_allclose(xtr.mean(0), 0, atol=1e-5)
                return np.full((len(xte), 2), .5), {"train_rows_used": len(xtr)}

            with patch.object(experiment, "source_files", return_value=[]), \
                 patch.object(experiment, "load_data", return_value=(x, ratings, subject, trial)), \
                 patch.object(experiment, "train_sklearn", side_effect=fake_train) as trainer:
                experiment.run(args)
                self.assertEqual(trainer.call_count, 2)
            with np.load(out / "split_f0.npz") as split:
                scaler = experiment.joblib.load(out / "scaler_f0.joblib")
                np.testing.assert_allclose(scaler.mean_, x[split["train"]].astype(np.float64).mean(0))
                with np.load(out / "knn_valence_f0_predictions.npz") as pred:
                    np.testing.assert_array_equal(pred["row"], split["test"])
                    self.assertEqual(pred["scores"].shape, (len(split["test"]), 2))
            summary = json.loads((out / "summary.json").read_text())
            self.assertEqual(set(summary["results"]), {"valence", "arousal"})
            rows = collect.collect(tmp)
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["run"], "run")


@unittest.skipIf(models.tf is None, "TensorFlow is not installed")
class ModelTests(unittest.TestCase):
    def test_all_models_gradients_counts_and_serialization(self):
        tf = models.tf
        for kind in sorted(constants.NEURAL):
            for length in [70, 73]:
                aggregations = ["mean_exp", "mean_mem", "prod"] if kind.startswith("cfnn") else ["mean_exp"]
                for agg in aggregations:
                    with self.subTest(kind=kind, length=length, agg=agg):
                        tf.keras.backend.clear_session()
                        tf.keras.utils.set_random_seed(42)
                        model = models.build_model(kind, length, agg)
                        x = np.random.default_rng(1).normal(size=(4, length, 1)).astype(np.float32)
                        with tf.GradientTape() as tape:
                            scores = model(x, training=True)
                            loss = tf.keras.losses.sparse_categorical_crossentropy(np.array([0, 1, 0, 1]), scores)
                            loss = tf.reduce_mean(loss)
                        grads = tape.gradient(loss, model.trainable_variables)
                        self.assertTrue(all(g is not None and np.isfinite(g.numpy()).all() for g in grads))
                        np.testing.assert_allclose(scores.numpy().sum(1), 1, atol=1e-6)
                        if kind == "cfnn":
                            self.assertEqual(model.count_params(), {70: 11058, 73: 11186}[length])
                            self.assertEqual(sum(int(np.prod(v.shape)) for v in model.non_trainable_weights), 4)
                        if kind == "plain_matched" and length == 73:
                            self.assertEqual(model.count_params(), 11198)
                        with tempfile.TemporaryDirectory() as tmp:
                            path = Path(tmp) / "model.keras"
                            model.save(path)
                            restored = tf.keras.models.load_model(path)
                            np.testing.assert_allclose(restored(x).numpy(), model(x).numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
