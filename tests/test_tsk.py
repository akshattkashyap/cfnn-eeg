"""HTSK math, train-only initialization and persistence checks; no model fitting."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
from scipy.special import softmax

from cfnn import cli, models, training, tsk


class HTSKInitializationTests(unittest.TestCase):
    def test_centers_are_reproducible_and_use_only_supplied_training_rows(self):
        train = np.random.default_rng(5).uniform(-.5, .5, (80, 4)).astype(np.float32)
        first, positions, info = tsk.training_rule_centers(train, 4, 40, 42)
        second, repeated, _ = tsk.training_rule_centers(train, 4, 40, 42)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_array_equal(positions, repeated)
        self.assertEqual(len(np.unique(positions)), 40)
        self.assertTrue(np.all((positions >= 0) & (positions < len(train))))
        self.assertTrue(np.all((first >= train.min(0)) & (first <= train.max(0))))
        self.assertEqual(info["initialization_rows_used"], 40)

    def test_initializer_passes_exact_sample_to_clustering(self):
        train = np.arange(400, dtype=np.float32).reshape(100, 4)
        with patch.object(tsk, "KMeans") as clustering:
            fitted = clustering.return_value.fit.return_value
            fitted.cluster_centers_ = np.stack([train[0], train[-1]])
            fitted.inertia_ = 1.0
            centers, positions, _ = tsk.training_rule_centers(train, 2, 30, 9)
            np.testing.assert_array_equal(clustering.return_value.fit.call_args.args[0], train[positions])
            self.assertEqual(centers.shape, (2, 4))

    def test_invalid_initialization_data_is_rejected(self):
        for x in (np.ones((10, 3)), np.full((10, 3), np.nan), np.ones((1, 3))):
            with self.assertRaises(ValueError):
                tsk.training_rule_centers(x, 2, 10)

    def test_cli_defaults_and_validation(self):
        self.assertEqual(cli.parse_args([]).feature_dim, 73)
        options = cli.parse_args(["--model", "htsk"])
        self.assertEqual((options.feature_dim, options.htsk_rules), (70, 16))
        self.assertEqual(cli.parse_args(["--model", "htsk", "--feature-dim", "73"]).feature_dim, 73)
        for flags in (["--htsk-rules", "1"], ["--htsk-init-sigma", "nan"],
                      ["--htsk-init-max-samples", "2"]):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                cli.parse_args(["--model", "htsk", *flags])

    def test_trainer_initializes_from_training_only_and_persists_sample_positions(self):
        # Mock fitting: verify data boundaries and artifacts without training locally.
        args = cli.parse_args(["--model", "htsk", "--htsk-rules", "2", "--epochs", "1"])
        xtr = np.full((8, 70), .25, dtype=np.float32)
        xva = np.full((4, 70), 100., dtype=np.float32)
        xte = np.full((4, 70), -100., dtype=np.float32)
        model = MagicMock()
        model.fit.return_value = SimpleNamespace(history={"loss": [.7], "val_loss": [.8]})
        model.count_params.return_value = 564
        model.predict.return_value = np.full((4, 2), .5)
        centers, positions = np.zeros((2, 70), np.float32), np.array([0, 2, 4, 6])
        info = {"rules": 2, "initialization_rows_used": 4}
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "htsk_valence_f0.weights.h5"
            with patch.object(training, "tf", MagicMock()), \
                 patch.object(training, "build_model", return_value=model), \
                 patch.object(training, "initialize_htsk", return_value=(centers, positions, info)) as init:
                scores, report = training.train_neural("htsk", args, xtr, np.arange(8) % 2,
                    xva, np.arange(4) % 2, xte, checkpoint)
            self.assertIs(init.call_args.args[1], xtr)
            np.testing.assert_array_equal(model.fit.call_args.args[0], xtr[..., None])
            np.testing.assert_array_equal(model.fit.call_args.kwargs["validation_data"][0], xva[..., None])
            np.testing.assert_array_equal(model.predict.call_args.args[0], xte[..., None])
            self.assertEqual(report["htsk_initialization"], info)
            self.assertEqual(scores.shape, (4, 2))
            with np.load(Path(tmp) / "htsk_valence_f0_initialization.npz") as saved:
                np.testing.assert_array_equal(saved["centers"], centers)
                np.testing.assert_array_equal(saved["training_positions"], positions)


@unittest.skipIf(models.tf is None, "TensorFlow is not installed")
class HTSKTensorFlowTests(unittest.TestCase):
    def test_rule_weights_and_logits_match_independent_numpy_equations(self):
        layer = tsk.HTSKClassifier(n_rules=2, init_sigma=.7)
        x = np.array([[1., -.5, 2.], [-1., 1., .3]], dtype=np.float32)
        layer(x)
        centers = np.array([[0., 0., 0.], [1., 2., 1.]], dtype=np.float32)
        kernel = np.arange(12, dtype=np.float32).reshape(2, 3, 2) / 10
        bias = np.array([[.2, -.4], [-.1, .3]], dtype=np.float32)
        layer.centers.assign(centers)
        layer.kernel.assign(kernel)
        layer.bias.assign(bias)
        z = .5 * ((x[:, None, :] - centers[None, :, :]) / .7)**2
        expected_weights = softmax(-z.mean(-1), axis=-1)
        rule_logits = np.einsum("bd,rdc->brc", x, kernel) + bias
        expected_logits = np.einsum("br,brc->bc", expected_weights, rule_logits)
        np.testing.assert_allclose(layer.rule_weights(x).numpy(), expected_weights, atol=1e-6)
        np.testing.assert_allclose(layer(x).numpy(), expected_logits, atol=1e-6)

    def test_high_dimensional_weights_are_finite_and_normalized(self):
        layer = tsk.HTSKClassifier(n_rules=8)
        x = np.random.default_rng(6).normal(size=(4, 1152)).astype(np.float32)
        layer(x)
        weights = layer.rule_weights(x).numpy()
        self.assertTrue(np.isfinite(weights).all())
        self.assertTrue((weights >= 0).all())
        np.testing.assert_allclose(weights.sum(1), 1, atol=1e-6)

    def test_feature_replication_preserves_mean_log_memberships(self):
        x = np.array([[1., 2.]], dtype=np.float32)
        small, large = tsk.HTSKClassifier(2), tsk.HTSKClassifier(2)
        small(x)
        large(np.tile(x, (1, 10)))
        centers = np.array([[0., 0.], [2., 3.]], dtype=np.float32)
        small.centers.assign(centers)
        large.centers.assign(np.tile(centers, (1, 10)))
        np.testing.assert_allclose(small.rule_weights(x).numpy(),
            large.rule_weights(np.tile(x, (1, 10))).numpy(), atol=1e-6)

    def test_data_initialization_parameter_count_and_save_roundtrip(self):
        tf = models.tf
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        train = np.random.default_rng(8).normal(size=(40, 70)).astype(np.float32)
        model = models.build_model("htsk", 70, htsk_rules=4, htsk_init_sigma=.8)
        centers, positions, _ = tsk.initialize_htsk(model, train, 30, 42)
        np.testing.assert_array_equal(model.get_layer("htsk").centers.numpy(), centers)
        self.assertEqual(len(positions), 30)
        self.assertEqual(model.count_params(), 4 * (4*70 + 2))
        x = train[:8, ..., None]
        y = np.arange(8) % 2
        with tf.GradientTape() as tape:
            scores = model(x, training=True)
            loss = tf.keras.losses.SparseCategoricalCrossentropy()(y, scores)
        gradients = tape.gradient(loss, model.trainable_variables)
        self.assertTrue(all(g is not None and np.isfinite(g.numpy()).all() for g in gradients))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "htsk.keras"
            model.save(path)
            restored = tf.keras.models.load_model(path)
            np.testing.assert_allclose(restored(x).numpy(), model(x).numpy(), atol=1e-6)
            np.testing.assert_allclose(restored.get_layer("htsk").rule_weights(train[:8]).numpy(),
                model.get_layer("htsk").rule_weights(train[:8]).numpy(), atol=1e-6)


if __name__ == "__main__":
    unittest.main()
