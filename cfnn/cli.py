"""Independent reconstruction of the 2024 Sci. Reports CFNN paper (DEAP, valence/arousal).

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
    p.add_argument("--data-dir", default="data/data_preprocessed_python")
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
                        "Overlapping + --split random replicates likely paper leakage.")
    p.add_argument("--nfft", type=int, default=0, help="0 = max(window, 128)")
    p.add_argument("--taper", choices=["hann", "none"], default="hann")
    p.add_argument("--feature-dim", type=int, choices=[70, 73], default=73)
    p.add_argument("--pad-mode", choices=["zero", "summary"], default="zero")
    p.add_argument("--threshold", type=float, default=5.0)
    p.add_argument("--threshold-op", choices=["ge", "gt"], default="ge")
    p.add_argument("--fuzzy-agg", choices=["mean_exp", "prod", "mean_mem"], default="mean_exp")
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
    if not np.isfinite(args.threshold) or not 1 <= args.threshold <= 9:
        p.error("--threshold must be in [1, 9]")
    if args.max_subjects is not None and not 4 <= args.max_subjects <= 32:
        p.error("--max-subjects must be in [4, 32]")
    if args.seed < 0 or args.seed >= 2**32 - 1:
        p.error("--seed must be between 0 and 2**32-2")
    if args.self_test and args.prepare_only:
        p.error("Choose either --self-test or --prepare-only")
    return args


def main(argv=None):
    args = parse_args(argv)
    from .experiment import run

    run(args)
