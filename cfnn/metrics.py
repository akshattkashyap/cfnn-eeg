"""Window/trial metrics and subject-bootstrap confidence intervals."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)


def metrics(y, p):
    if p.shape != (len(y), 2) or not np.isfinite(p).all():
        raise ValueError("Expected finite class scores with shape (n, 2)")
    if np.any(p < 0) or not np.allclose(p.sum(1), 1.0, atol=1e-5):
        raise ValueError("Class scores must be nonnegative and sum to one")
    pred = p.argmax(1)
    pr, rc, f1, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    return {"accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "precision": float(pr), "recall": float(rc), "f1": float(f1),
            "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
            "positive_fraction": float(np.mean(y)), "n": int(len(y)),
            "majority_baseline": float(max(np.mean(y), 1 - np.mean(y)))}


def bootstrap_ci(vals, n=2000, seed=0):
    vals = np.asarray(vals, float)
    if len(vals) < 2:
        return None
    rng = np.random.default_rng(seed)
    means = [rng.choice(vals, len(vals), replace=True).mean() for _ in range(n)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def trial_level(y, p, trial, subj):
    # Aggregate in one pass; avoid scanning millions of windows once per trial.
    _, first, inverse, counts = np.unique(trial, return_index=True, return_inverse=True,
                                         return_counts=True)
    ys, ss = y[first], subj[first]
    if not np.array_equal(y, ys[inverse]) or not np.array_equal(subj, ss[inverse]):
        raise ValueError("Each trial must have exactly one label and one subject")
    ps = np.zeros((len(first), 2), dtype=np.float64)
    np.add.at(ps, inverse, p)
    ps /= counts[:, None]
    res = metrics(ys, ps)
    per = {int(s): float(accuracy_score(ys[ss == s], ps[ss == s].argmax(1))) for s in np.unique(ss)}
    res["per_subject_accuracy"] = per
    res["subject_accuracy_mean"] = float(np.mean(list(per.values())))
    res["subject_bootstrap_ci95"] = bootstrap_ci(list(per.values()))
    return res
