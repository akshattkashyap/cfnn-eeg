"""Coordinate data, partitions, train-only scaling, training and reports."""
from __future__ import annotations

import random
from pathlib import Path

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

from .artifacts import environment_info, write_json
from .constants import LABEL_COLS, NEURAL
from .data import binarize, feature_manifest, load_data, source_files
from .metrics import metrics, trial_level
from .models import self_test, tf
from .splits import make_splits, split_audit
from .training import train_neural, train_sklearn


def run(args):
    np.random.seed(args.seed); random.seed(args.seed)
    if tf is not None:
        tf.keras.utils.set_random_seed(args.seed)
    if args.self_test:
        return self_test(args)
    if args.model in NEURAL and tf is None and not args.prepare_only:
        raise SystemExit("TensorFlow is required for neural models")
    if not args.data_dir:
        raise SystemExit("--data-dir is required")
    out = Path(args.output_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory is not empty: {out}. Choose a new --output-dir.")
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "config.json", vars(args))
    write_json(out / "environment.json", environment_info())
    write_json(out / "feature_manifest.json", feature_manifest(args, source_files(args)))

    X, R, subj, trial = load_data(args)
    print(f"Feature matrix {X.shape}; subjects {len(np.unique(subj))}; trials {len(np.unique(trial))}")
    if args.split == "random":
        print("WARNING: random window-level split; windows from one trial can be in train AND test "
              "(window-level scores can be optimistic).")
    splits = make_splits(args, subj, trial)
    targets = list(LABEL_COLS) if args.target == "both" else [args.target]
    folds = {t: [] for t in targets}
    np.savez_compressed(out / "row_metadata.npz", ratings=R, subject=subj, trial=trial)
    rating_max = 5 if args.dataset == "dreamer" else 9
    preparation = {"dataset": args.dataset, "feature_matrix_shape": list(X.shape),
                   "rating_ranges": {t: {"min": float(R[:, col].min()),
                       "max": float(R[:, col].max()),
                       "trials_outside_nominal_range": int(len(np.unique(trial[
                           (R[:, col] < 1) | (R[:, col] > rating_max)])))} for t, col in LABEL_COLS.items()},
                   "class_counts": {t: np.bincount(binarize(R[:, LABEL_COLS[t]],
                       args.threshold, args.threshold_op), minlength=2).tolist() for t in targets},
                   "folds": []}
    for fi, (tr, va, te) in enumerate(splits):
        np.savez_compressed(out / f"split_f{fi}.npz", train=tr, validation=va, test=te)
        audit = split_audit(tr, va, te, subj, trial)
        write_json(out / f"split_f{fi}_audit.json", audit)
        preparation["folds"].append({"fold": fi, **audit})
    write_json(out / "preparation.json", preparation)
    if args.prepare_only:
        print("Prepared features and audited splits; no training performed.")
        return

    for fi, (tr, va, te) in enumerate(splits):
        sc = StandardScaler().fit(X[tr])                       # TRAIN ONLY
        joblib.dump(sc, out / f"scaler_f{fi}.joblib")
        xtr, xva, xte = (sc.transform(X[i]).astype(np.float32) for i in (tr, va, te))
        for tgt in targets:
            y = binarize(R[:, LABEL_COLS[tgt]], args.threshold, args.threshold_op)
            if len(np.unique(y[tr])) != 2:
                raise ValueError(f"Fold {fi}, {tgt}: training partition needs both classes")
            print(f"[fold {fi}] {args.model} {tgt}: train hist {np.bincount(y[tr], minlength=2)}")
            if args.model in NEURAL:
                probs, info = train_neural(args.model, args, xtr, y[tr], xva, y[va], xte,
                                           out / f"{args.model}_{tgt}_f{fi}.weights.h5")
            else:
                probs, info = train_sklearn(args.model, args, xtr, y[tr], xte,
                                            out / f"{args.model}_{tgt}_f{fi}.joblib")
            np.savez_compressed(out / f"{args.model}_{tgt}_f{fi}_predictions.npz",
                                row=te, y_true=y[te], scores=probs, subject=subj[te], trial=trial[te])
            train_majority = int(np.argmax(np.bincount(y[tr], minlength=2)))
            folds[tgt].append({
                "fold": fi,
                "test_subjects": np.unique(subj[te]).tolist(),
                "n_test_windows": int(len(te)),
                "train_class_counts": np.bincount(y[tr], minlength=2).tolist(),
                "validation_class_counts": np.bincount(y[va], minlength=2).tolist(),
                "train_majority_class": train_majority,
                "train_majority_test_accuracy": float(np.mean(y[te] == train_majority)),
                "window": metrics(y[te], probs),
                "trial": trial_level(y[te], probs, trial[te], subj[te]),
                **{k: v for k, v in info.items() if k != "history"}})
            if "history" in info:
                write_json(out / f"{args.model}_{tgt}_f{fi}_history.json", info["history"])
            write_json(out / "partial_results.json", folds)

    summary = {"paper_doi": "10.1038/s41598-024-60977-9", "reconstruction_not_original": True,
               "args": vars(args), "feature_matrix_shape": list(X.shape), "results": {}}
    if args.dataset == "dreamer":
        summary["dataset_reference"] = {"name": "DREAMER", "url": "https://zenodo.org/records/546113"}
    if args.model == "htsk":
        summary["model_reference"] = {
            "doi": "10.1109/IJCNN52387.2021.9534265", "arxiv": "2102.04271",
            "implementation": "Independent TensorFlow first-order HTSK baseline; not the CFNN architecture"}
    for tgt, fl in folds.items():
        w = [f["window"]["accuracy"] for f in fl]
        t = [f["trial"]["accuracy"] for f in fl]
        summary["results"][tgt] = {"folds": fl,
                                   "window_accuracy_mean": float(np.mean(w)),
                                   "window_accuracy_std": float(np.std(w)),
                                   "trial_accuracy_mean": float(np.mean(t)),
                                   "trial_accuracy_std": float(np.std(t))}
        print(f"== {args.model} | {args.split} | {tgt}: window acc {np.mean(w):.4f}+/-{np.std(w):.4f}"
              f" | trial acc {np.mean(t):.4f}+/-{np.std(t):.4f}")
    write_json(out / "summary.json", summary)
    print("Saved", (out / "summary.json").resolve())
