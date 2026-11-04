"""Independent TensorFlow first-order HTSK classifier and train-only initialization.

References:
  Cui, Wu & Xu (IJCNN 2021), https://arxiv.org/abs/2102.04271
  PyTSK reference equations: https://pytskdocs.readthedocs.io/en/latest/models.html
"""
from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans

try:
    import tensorflow as tf
except ImportError:
    tf = None


def training_rule_centers(x_train, n_rules=16, max_samples=30000, seed=42):
    """K-means on a bounded sample of scaled TRAIN rows; positions are relative to x_train."""
    x_train = np.asarray(x_train)
    if x_train.ndim != 2 or not x_train.shape[1] or not np.isfinite(x_train).all():
        raise ValueError("Expected a finite 2D matrix of scaled training features")
    if n_rules < 2 or max_samples < n_rules or len(x_train) < n_rules:
        raise ValueError("Need >=2 rules and at least one training/init sample per rule")
    rng = np.random.default_rng(seed)
    positions = (rng.choice(len(x_train), max_samples, replace=False)
                 if len(x_train) > max_samples else np.arange(len(x_train)))
    sample = x_train[positions]
    if len(np.unique(sample, axis=0)) < n_rules:
        raise ValueError("Fewer distinct initialization samples than HTSK rules")
    clustering = KMeans(n_clusters=n_rules, n_init=5, random_state=seed).fit(sample)
    centers = clustering.cluster_centers_.astype(np.float32)
    if len(np.unique(centers, axis=0)) < n_rules:
        raise ValueError("K-means produced duplicate rule centers; reduce --htsk-rules")
    info = {"method": "kmeans", "training_rows_available": len(x_train),
            "initialization_rows_used": len(positions), "rules": n_rules,
            "seed": seed, "inertia": float(clustering.inertia_),
            "sample_positions_relative_to": "saved split_fN.npz train array"}
    return centers, positions, info


def initialize_htsk(model, x_train, max_samples=30000, seed=42):
    """Initialize a built model using only the array explicitly supplied by its trainer."""
    layer = model.get_layer("htsk")
    centers, positions, info = training_rule_centers(x_train, layer.n_rules, max_samples, seed)
    layer.centers.assign(centers)
    return centers, positions, {**info, "initial_sigma": layer.init_sigma}


if tf is not None:

    @tf.keras.utils.register_keras_serializable(package="cfnn")
    class HTSKClassifier(tf.keras.layers.Layer):
        """D inputs, R rules, C classes: softmax_r(-mean_d z) weighted linear logits.

        z[b,r,d] = 0.5 * ((x[b,d] - centers[r,d]) / sigma[r,d])**2
        y[b,c] = sum_r weight[b,r] * (sum_d x[b,d]*kernel[r,d,c] + bias[r,c])
        Class softmax is applied by the enclosing model. No conv, BN or CFNN defuzzy.
        """

        def __init__(self, n_rules=16, n_classes=2, init_sigma=1.0, min_sigma=1e-4, **kwargs):
            super().__init__(**kwargs)
            if n_rules < 2 or n_classes < 2:
                raise ValueError("HTSK needs >=2 rules and >=2 output classes")
            if not np.isfinite(min_sigma) or min_sigma <= 0:
                raise ValueError("min_sigma must be finite and positive")
            if not np.isfinite(init_sigma) or init_sigma <= min_sigma:
                raise ValueError("init_sigma must be finite and greater than min_sigma")
            self.n_rules, self.n_classes = int(n_rules), int(n_classes)
            self.init_sigma, self.min_sigma = float(init_sigma), float(min_sigma)

        def build(self, input_shape):
            if len(input_shape) != 2 or input_shape[-1] is None:
                raise ValueError("HTSK expects a fixed-length 2D feature matrix")
            dim = int(input_shape[-1])
            self.centers = self.add_weight(name="centers", shape=(self.n_rules, dim),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.5))
            # Stable inverse softplus: the resulting initial positive width is init_sigma.
            value = self.init_sigma - self.min_sigma
            raw = value + np.log(-np.expm1(-value))
            self.raw_scales = self.add_weight(name="raw_scales", shape=(self.n_rules, dim),
                initializer=tf.keras.initializers.Constant(raw))
            self.kernel = self.add_weight(name="consequent_kernel",
                shape=(self.n_rules, dim, self.n_classes),
                initializer=tf.keras.initializers.RandomNormal(stddev=np.sqrt(2.0 / dim)))
            self.bias = self.add_weight(name="consequent_bias", shape=(self.n_rules, self.n_classes),
                                        initializer="zeros")
            super().build(input_shape)

        def rule_weights(self, x):
            sigma = tf.nn.softplus(self.raw_scales) + self.min_sigma
            z = 0.5 * tf.square((x[:, None, :] - self.centers[None, :, :]) / sigma[None, :, :])
            return tf.nn.softmax(-tf.reduce_mean(z, axis=-1), axis=-1)

        def call(self, x):
            weights = self.rule_weights(x)
            rule_logits = tf.einsum("bd,rdc->brc", x, self.kernel) + self.bias[None, :, :]
            return tf.einsum("br,brc->bc", weights, rule_logits)

        def get_config(self):
            return {**super().get_config(), "n_rules": self.n_rules, "n_classes": self.n_classes,
                    "init_sigma": self.init_sigma, "min_sigma": self.min_sigma}
