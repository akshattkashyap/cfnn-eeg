"""CLI orchestration for diagnostics on a completed run's training partition."""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np

from .artifacts import environment_info, write_json
from .constants import LABEL_COLS, NEURAL
from .data import binarize
from .diagnostics import balanced_rows, batch_features, load_training_data, memorize_batch
from .models import tf


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True, help="Completed original run directory")
    parser.add_argument("--data-dir", help="Override data path, e.g. when moving between laptop/Colab")
    parser.add_argument("--cache-dir", help="Override feature cache directory")
    parser.add_argument("--output-dir", type=Path, help="New directory; default: timestamped diagnostics in run")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--target", choices=["valence", "arousal", "both"], default="valence")
    parser.add_argument("--models", nargs="+", choices=sorted(NEURAL), default=["cnn", "cfnn"])
    parser.add_argument("--learning-rates", nargs="+", type=float, default=[0.01, 0.001])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000, help="Updates of the same batch, not full-data epochs")
    parser.add_argument("--check-only", action="store_true", help="Check artifacts/features; no model fitting")
    args = parser.parse_args(argv)
    if args.batch_size < 4 or args.batch_size % 2:
        parser.error("--batch-size must be even and >=4")
    if args.steps < 1 or args.fold < 0:
        parser.error("--steps must be positive and --fold nonnegative")
    if not all(np.isfinite(lr) and lr > 0 for lr in args.learning_rates):
        parser.error("Learning rates must be finite and positive")
    return args


def main(argv=None):
    options = parse_args(argv)
    if tf is None and not options.check_only:
        raise SystemExit("TensorFlow is required; --check-only can run without it")
    out = options.output_dir or options.run_dir / datetime.now().strftime("diagnostics_%Y%m%d_%H%M%S_%f")
    out.mkdir(parents=True, exist_ok=False)
    args, X, R, subject, trial, train, scaler = load_training_data(
        options.run_dir, options.fold, options.data_dir, options.cache_dir)
    report = {"reference_run": str(options.run_dir.resolve()), "environment": environment_info(),
              "options": {k: str(v) if isinstance(v, Path) else v for k, v in vars(options).items()},
              "feature_settings": {k: getattr(args, k) for k in
                  ("window_samples", "nfft", "taper", "feature_dim", "pad_mode", "fuzzy_agg")},
              "uses_training_rows_only": True, "check_only": options.check_only,
              "interpretation": "Memorization diagnostics, not generalization or reproduction scores",
              "targets": {}}
    targets = list(LABEL_COLS) if options.target == "both" else [options.target]
    for target in targets:
        y = binarize(R[:, LABEL_COLS[target]], args.threshold, args.threshold_op)
        rows = balanced_rows(y, trial, train, options.batch_size, args.seed)
        x, feature_report = batch_features(X, rows, scaler, y)
        result = {"batch_rows": rows.tolist(), "batch_trials": trial[rows].tolist(),
                  "batch_subjects": subject[rows].tolist(),
                  "class_counts": np.bincount(y[rows], minlength=2).tolist(),
                  "features": feature_report, "runs": []}
        report["targets"][target] = result
        print(f"{target}: {len(rows)} training windows from distinct trials; "
              f"class counts {result['class_counts']}; features {feature_report}", flush=True)
        np.savez_compressed(out / f"{target}_batch.npz", row=rows, x=x, y=y[rows])
        write_json(out / "diagnostics.json", report)
        if options.check_only:
            continue
        for kind in options.models:
            for lr in options.learning_rates:
                experiment = memorize_batch(kind, x, y[rows], args.fuzzy_agg, lr,
                                             args.seed, options.steps, args.clipnorm)
                result["runs"].append(experiment)
                write_json(out / "diagnostics.json", report)
    print("Saved", (out / "diagnostics.json").resolve(), flush=True)
