"""Fixed-batch memorization checks; no validation or test scoring."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from .data import binarize, load_deap
from .models import build_model, tf


def balanced_rows(y, trial, training_rows, size=32, seed=42):
    """Select balanced training rows with at most one window from each trial."""
    if size < 4 or size % 2:
        raise ValueError("Batch size must be even and >=4")
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(training_rows)
    _, first, inverse = np.unique(trial[shuffled], return_index=True, return_inverse=True)
    candidates = shuffled[first]
    if not np.array_equal(y[shuffled], y[candidates][inverse]):
        raise ValueError("A training trial has inconsistent labels")
    selected = []
    for label in (0, 1):
        pool = candidates[y[candidates] == label]
        if len(pool) < size // 2:
            raise ValueError(f"Need at least {size // 2} training trials of class {label}")
        selected.extend(rng.choice(pool, size // 2, replace=False))
    selected = np.array(selected, dtype=np.int64)
    rng.shuffle(selected)
    return selected


def validate_saved_rows(ratings, subject, trial, saved, split):
    """Reject wrong data, row order, or split artifacts before fitting anything."""
    for key, actual in (("ratings", ratings), ("subject", subject), ("trial", trial)):
        if not np.array_equal(saved[key], actual):
            raise ValueError(f"Saved {key} differ from current data; use the same dataset as the run")
    partitions = [split[k] for k in ("train", "validation", "test")]
    if any(a.ndim != 1 or not np.issubdtype(a.dtype, np.integer) or not len(a)
           for a in partitions):
        raise ValueError("Expected nonempty integer split-index arrays")
    rows = np.concatenate(partitions)
    if not np.array_equal(np.sort(rows), np.arange(len(subject))):
        raise ValueError("Saved splits must assign every dataset row exactly once")


def load_training_data(run_dir, fold=0, data_dir=None, cache_dir=None):
    run_dir = Path(run_dir)
    if not (run_dir / "summary.json").is_file():
        raise ValueError("--run-dir must be a completed experiment containing summary.json")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if data_dir is not None:
        config["data_dir"] = str(data_dir)
    if cache_dir is not None:
        config["cache_dir"] = str(cache_dir)
    args = argparse.Namespace(**config)
    X, R, subject, trial = load_deap(args)
    if X.shape != (len(R), args.feature_dim) or not np.isfinite(X).all():
        raise ValueError("Invalid feature matrix")
    with np.load(run_dir / "row_metadata.npz", allow_pickle=False) as saved, \
         np.load(run_dir / f"split_f{fold}.npz", allow_pickle=False) as split:
        validate_saved_rows(R, subject, trial, saved, split)
        train = split["train"].copy()
    scaler = joblib.load(run_dir / f"scaler_f{fold}.joblib")
    # Check that current extraction matches the saved train-only statistics too.
    current = StandardScaler().fit(X[train])
    if not (np.allclose(current.mean_, scaler.mean_, rtol=1e-6, atol=1e-6)
            and np.allclose(current.var_, scaler.var_, rtol=1e-6, atol=1e-6)):
        raise ValueError("Current features disagree with the saved scaler; check code/cache/data")
    return args, X, R, subject, trial, train, scaler


def batch_features(X, rows, scaler, y):
    x = scaler.transform(X[rows]).astype(np.float32)
    if not np.isfinite(x).all():
        raise ValueError("Non-finite scaled features")
    _, groups = np.unique(x, axis=0, return_inverse=True)
    conflicting = sum(len(np.unique(y[rows][groups == g])) > 1 for g in np.unique(groups))
    report = {
        "shape": list(x.shape), "scaled_min": float(x.min()), "scaled_max": float(x.max()),
        "constant_training_columns": np.flatnonzero(scaler.var_ == 0).tolist(),
        "unique_batch_feature_rows": int(len(np.unique(groups))),
        "identical_feature_groups_with_conflicting_labels": int(conflicting),
    }
    return x[..., None], report


def batch_scores(model, x, y):
    """Compare modes while preserving BN moving statistics and other non-trainable state."""
    inference = model(x, training=False).numpy()
    state = [v.numpy().copy() for v in model.non_trainable_variables]
    try:
        training = model(x, training=True).numpy()
    finally:
        for variable, value in zip(model.non_trainable_variables, state):
            variable.assign(value)
    result = {}
    for name, p in (("inference_mode", inference), ("training_mode", training)):
        if p.shape != (len(y), 2) or not np.isfinite(p).all():
            raise ValueError("Invalid model probabilities")
        prediction = p.argmax(1)
        result[name] = {"accuracy": float(np.mean(prediction == y)),
                        "loss": float(-np.log(np.clip(p[np.arange(len(y)), y], 1e-7, 1)).mean()),
                        "predicted_counts": np.bincount(prediction, minlength=2).tolist()}
    result["prediction_disagreement_fraction"] = float(
        np.mean(inference.argmax(1) != training.argmax(1)))
    return result


def memorize_batch(kind, x, y, agg, learning_rate, seed, steps, clipnorm=0.0, log_every=100):
    if tf is None:
        raise RuntimeError("TensorFlow is required for memorization checks")
    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(seed)
    model = build_model(kind, x.shape[1], agg)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                         **({"clipnorm": clipnorm} if clipnorm > 0 else {}))
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
    tx, ty = tf.convert_to_tensor(x), tf.convert_to_tensor(y)

    @tf.function
    def update():
        with tf.GradientTape() as tape:
            p = model(tx, training=True)
            loss = loss_fn(ty, p)
        gradients = tape.gradient(loss, model.trainable_variables)
        if any(g is None for g in gradients):
            raise ValueError("A trainable variable has no gradient")
        tf.debugging.assert_all_finite(loss, "Non-finite loss")
        for g in gradients:
            tf.debugging.assert_all_finite(g, "Non-finite gradient")
        norm = tf.linalg.global_norm(gradients)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        return loss, norm

    initial = batch_scores(model, tx, y)
    trace = []
    for step in range(1, steps + 1):
        loss, gradient_norm = update()
        if step == 1 or step % log_every == 0 or step == steps:
            scores = batch_scores(model, tx, y)
            row = {"step": step, "update_loss": float(loss),
                   "gradient_norm": float(gradient_norm), **scores}
            trace.append(row)
            print(f"  {kind} lr={learning_rate:g} step={step}/{steps}: "
                  f"train-mode={scores['training_mode']['accuracy']:.1%}, "
                  f"inference={scores['inference_mode']['accuracy']:.1%}", flush=True)
    return {"model": kind, "learning_rate": learning_rate, "steps": steps,
            "params": model.count_params(), "initial": initial,
            "final": trace[-1], "trace": trace}
