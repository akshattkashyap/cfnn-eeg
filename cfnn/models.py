"""CFNN custom layers, neural architectures and non-training model checks."""
from __future__ import annotations

import numpy as np
from .constants import NEURAL

try:
    import tensorflow as tf
except ImportError:  # sklearn-only models (svm/knn) still work without TensorFlow
    tf = None


if tf is not None:

    @tf.keras.utils.register_keras_serializable(package="cfnn")
    class GaussianFuzzification(tf.keras.layers.Layer):
        """RECONSTRUCTION. Gaussian membership per (input, rule); 2 rules -> 2*d*2 params.
        agg: mean_exp  = exp(-mean_i z_ij)  (geometric-mean AND, stable)
             prod      = exp(-sum_i z_ij)   (literal product; underflows at d=1152)
             mean_mem  = mean_i exp(-z_ij)  (arithmetic mean of memberships)
        """

        def __init__(self, units=2, agg="mean_exp", **kw):
            super().__init__(**kw)
            if agg not in {"mean_exp", "prod", "mean_mem"}:
                raise ValueError(f"Unknown fuzzy aggregation: {agg}")
            self.units, self.agg = units, agg

        def build(self, input_shape):
            n = int(input_shape[-1])
            self.centers = self.add_weight(
                name="centers", shape=(n, self.units),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.5))
            self.raw_scales = self.add_weight(          # softplus(0.5413) ~= 1.0
                name="raw_scales", shape=(n, self.units),
                initializer=tf.keras.initializers.Constant(0.5413))
            super().build(input_shape)

        def call(self, x):
            v = tf.expand_dims(x, -1)                                   # (B,d,1)
            sigma = tf.nn.softplus(self.raw_scales) + 1e-5
            z = 0.5 * tf.square((v - self.centers) / sigma)             # (B,d,u)
            if self.agg == "mean_exp":
                return tf.exp(-tf.reduce_mean(z, axis=1))
            if self.agg == "prod":
                return tf.exp(-tf.reduce_sum(z, axis=1))
            return tf.reduce_mean(tf.exp(-z), axis=1)

        def get_config(self):
            c = super().get_config()
            c.update(units=self.units, agg=self.agg)
            return c

    @tf.keras.utils.register_keras_serializable(package="cfnn")
    class WeightedDefuzzification(tf.keras.layers.Layer):
        """RECONSTRUCTION of Eq. 10: v = (a @ C) / sum(a), a = softplus(input), C is 2x2 (4 params)."""

        def __init__(self, units=2, **kw):
            super().__init__(**kw)
            self.units = units

        def build(self, input_shape):
            self.cores = self.add_weight(
                name="cores", shape=(int(input_shape[-1]), self.units),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.5))
            super().build(input_shape)

        def call(self, x):
            a = tf.nn.softplus(x)
            return tf.matmul(a, self.cores) / (tf.reduce_sum(a, -1, keepdims=True) + 1e-7)

        def get_config(self):
            c = super().get_config()
            c.update(units=self.units)
            return c


def build_model(kind, length, agg="mean_exp"):
    if tf is None:
        raise RuntimeError("TensorFlow is required: pip install -r requirements.txt")
    L = tf.keras.layers
    inp = L.Input(shape=(length, 1), name="features")
    if kind == "bilstm":
        x = L.Bidirectional(L.LSTM(32))(inp)
        return tf.keras.Model(inp, L.Dense(2, activation="softmax")(x), name=kind)
    x = L.Conv1D(64, 5, padding="same", activation="relu", name="conv1")(inp)
    x = L.Conv1D(32, 3, padding="same", activation="relu", name="conv2")(x)
    x = L.MaxPooling1D(2, name="pool")(x)
    x = L.Flatten(name="flatten")(x)
    if kind == "cfnn":
        x = GaussianFuzzification(2, agg, name="fuzzy")(x)
        x = L.BatchNormalization(name="bn")(x)
        x = L.Dense(2, name="dense")(x)
        x = WeightedDefuzzification(2, name="defuzzy")(x)
    elif kind == "cfnn_nodefuzz":
        x = GaussianFuzzification(2, agg, name="fuzzy")(x)
        x = L.BatchNormalization(name="bn")(x)
        x = L.Dense(2, name="dense")(x)
    elif kind == "plain":              # same layout, fuzzy block replaced by Dense(2, relu)
        x = L.Dense(2, activation="relu", name="plain2")(x)
        x = L.BatchNormalization(name="bn")(x)
        x = L.Dense(2, name="dense")(x)
    elif kind == "plain_matched":      # ~parameter-matched to the fuzzy layer (1152*4+4 = 4612)
        x = L.Dense(4, activation="relu", name="plain4")(x)
        x = L.BatchNormalization(name="bn")(x)
        x = L.Dense(2, name="dense")(x)
    elif kind == "cnn":
        x = L.Dense(2, name="logits")(x)
    else:
        raise ValueError(kind)
    return tf.keras.Model(inp, L.Softmax(name="out")(x), name=kind)


def self_test(args):
    if args.model not in NEURAL:
        raise SystemExit("self-test is for neural models")
    m = build_model(args.model, args.feature_dim, args.fuzzy_agg)
    m.summary()
    x = np.random.default_rng(args.seed).normal(size=(8, args.feature_dim, 1)).astype("float32")
    y = np.random.default_rng(1).integers(0, 2, 8)
    with tf.GradientTape() as tape:
        p = m(x, training=True)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(y, p)
    grads = tape.gradient(loss, m.trainable_variables)
    assert p.shape == (8, 2)
    np.testing.assert_allclose(p.numpy().sum(1), 1.0, atol=1e-5)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in grads), "bad gradients"
    if args.model == "cfnn":
        want = {73: 11186, 70: 11058}[args.feature_dim]
        assert m.count_params() == want, (m.count_params(), want)
    print("SELF-TEST OK | params:", m.count_params(), "| loss:", float(loss))
