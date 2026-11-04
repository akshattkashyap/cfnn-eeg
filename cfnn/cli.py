"""CFNN reconstruction and comparison models for EEG valence/arousal on DEAP or DREAMER.

NOT the authors' code. Every choice the paper leaves open is an explicit, flagged option.
Paper: https://doi.org/10.1038/s41598-024-60977-9

Split modes
  random  : paper-style random window-level split (windows of one trial can straddle train/test)
  trial   : whole subject-trials held out (no within-trial leakage)
  subject : whole subjects held out (subject-independent)
Add --cv-folds K for grouped K-fold instead of one hold-out."""
from __future__ import annotations

import argparse
import numpy as np

from .constants import NEURAL, SKLEARN, TRIAL_LEN


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", choices=["deap", "dreamer"], default="deap")
    p.add_argument("--data-dir", default=None,
                   help="DEAP subject directory, or original DREAMER.mat file / containing directory")
    p.add_argument("--output-dir", default="runs/default")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--model", choices=sorted(NEURAL | SKLEARN), default="cfnn")
    p.add_argument("--target", choices=["both", "valence", "arousal"], default="both")
    p.add_argument("--split", choices=["random", "trial", "subject"], default="subject")
    p.add_argument("--cv-folds", type=int, default=0, help="0 = single hold-out")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--window-samples", type=int, default=512, help="paper sweep: 4..128; 512 = 4 s")
    p.add_argument("--stride", type=int, default=0,
                   help="window stride in samples; 0 = non-overlapping (= --window-samples). "
                        "Overlapping + --split random can place closely related windows in train and test.")
    p.add_argument("--nfft", type=int, default=0, help="0 = max(window, 128)")
    p.add_argument("--taper", choices=["hann", "none"], default="hann")
    p.add_argument("--feature-dim", type=int, choices=[70, 73], default=None,
                   help="default: 70 for HTSK, 73 for other models")
    p.add_argument("--pad-mode", choices=["zero", "summary"], default="zero")
    p.add_argument("--threshold", type=float, default=None,
                   help="rating threshold: default 5 for DEAP, 3 for DREAMER")
    p.add_argument("--threshold-op", choices=["ge", "gt"], default="ge",
                   help="high = rating >= threshold (ge, default), or > threshold (gt)")
    p.add_argument("--dreamer-last-seconds", type=float, default=0.0,
                   help="DREAMER stimulus crop: 0 = all stimulus samples; positive = last N seconds")
    p.add_argument("--fuzzy-agg", choices=["mean_exp", "prod", "mean_mem"], default="mean_exp")
    p.add_argument("--htsk-rules", type=int, default=16, help="number of HTSK rules")
    p.add_argument("--htsk-init-max-samples", type=int, default=30000,
                   help="maximum scaled TRAIN rows used for HTSK k-means initialization")
    p.add_argument("--htsk-init-sigma", type=float, default=1.0,
                   help="initial Gaussian width in standardized feature units")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--learning-rate", type=float, default=0.01)
    p.add_argument("--clipnorm", type=float, default=0.0)
    p.add_argument("--early-stopping", action="store_true")
    p.add_argument("--sk-max-train", type=int, default=30000)
    p.add_argument("--svm-c", type=float, default=1.0)
    p.add_argument("--knn-neighbors", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--prepare-only", action="store_true", help="Extract/cache features and audit splits; no training")
    args = p.parse_args(argv)
    if args.data_dir is None:
        args.data_dir = "data/DREAMER.mat" if args.dataset == "dreamer" else "data/data_preprocessed_python"
    if args.threshold is None:
        args.threshold = 3.0 if args.dataset == "dreamer" else 5.0
    if args.feature_dim is None:
        args.feature_dim = 70 if args.model == "htsk" else 73
    if args.htsk_rules < 2 or args.htsk_init_max_samples < args.htsk_rules:
        p.error("--htsk-rules must be >=2 and --htsk-init-max-samples must be >= rules")
    if not np.isfinite(args.htsk_init_sigma) or args.htsk_init_sigma <= 1e-4:
        p.error("--htsk-init-sigma must be finite and >0.0001")
    if not 3 <= args.window_samples <= TRIAL_LEN:
        p.error("--window-samples must be in [3, 7680]")
    if args.stride < 0 or (args.stride and args.stride > args.window_samples):
        p.error("--stride must be 0 (non-overlapping) or in [1, --window-samples]")
    if args.nfft < 0 or (args.nfft and args.nfft < args.window_samples):
        p.error("--nfft must be 0 (automatic) or >= --window-samples")
    if args.cv_folds < 0 or args.cv_folds == 1:
        p.error("--cv-folds must be 0 or >=2")
    if not 0 < args.test_size < 1 or not 0 < args.val_size < 1:
        p.error("--test-size and --val-size must be in (0, 1)")
    if args.epochs < 1 or args.batch_size < 1 or args.sk_max_train < 2 or args.knn_neighbors < 1:
        p.error("Epochs, batch size, neighbors must be positive; --sk-max-train must be >=2")
    if not np.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("--learning-rate must be finite and positive")
    if not np.isfinite(args.clipnorm) or args.clipnorm < 0:
        p.error("--clipnorm must be finite and nonnegative")
    if not np.isfinite(args.svm_c) or args.svm_c <= 0:
        p.error("--svm-c must be finite and positive")
    rating_max = 5 if args.dataset == "dreamer" else 9
    if not np.isfinite(args.threshold) or not 1 <= args.threshold <= rating_max:
        p.error(f"--threshold must be in [1, {rating_max}] for {args.dataset}")
    if not np.isfinite(args.dreamer_last_seconds) or args.dreamer_last_seconds < 0:
        p.error("--dreamer-last-seconds must be finite and nonnegative")
    if args.dreamer_last_seconds:
        if args.dataset != "dreamer":
            p.error("--dreamer-last-seconds requires --dataset dreamer")
        if round(args.dreamer_last_seconds * 128) < args.window_samples:
            p.error("DREAMER stimulus crop must contain at least one full window")
    subject_max = 23 if args.dataset == "dreamer" else 32
    if args.max_subjects is not None and not 4 <= args.max_subjects <= subject_max:
        p.error(f"--max-subjects must be in [4, {subject_max}] for {args.dataset}")
    if args.seed < 0 or args.seed >= 2**32 - 1:
        p.error("--seed must be between 0 and 2**32-2")
    if args.self_test and args.prepare_only:
        p.error("Choose either --self-test or --prepare-only")
    return args


def main(argv=None):
    args = parse_args(argv)
    from .experiment import run

    run(args)
