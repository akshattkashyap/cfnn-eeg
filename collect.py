#!/usr/bin/env python3
"""Collect completed experiment summaries into CSV; partial runs are excluded."""
import argparse
import csv
import json
import sys
from pathlib import Path


def collect(root):
    root = Path(root)
    rows = []
    for path in sorted(root.rglob("summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        args = summary["args"]
        for target, result in summary["results"].items():
            folds = result["folds"]
            rows.append({
                "run": str(path.parent.relative_to(root)), "model": args["model"],
                "dataset": args.get("dataset", "deap"),
                "split": args["split"], "cv_folds": args["cv_folds"],
                "window_samples": args["window_samples"], "feature_dim": args["feature_dim"],
                "pad_mode": args["pad_mode"], "fuzzy_agg": args["fuzzy_agg"],
                "stride": args.get("stride", 0) or args["window_samples"],
                "dreamer_last_seconds": args.get("dreamer_last_seconds", 0.0)
                    if args.get("dataset", "deap") == "dreamer" else None,
                "htsk_rules": args.get("htsk_rules", 16) if args["model"] == "htsk" else None,
                "htsk_init_sigma": args.get("htsk_init_sigma", 1.0) if args["model"] == "htsk" else None,
                "threshold": args["threshold"], "threshold_op": args["threshold_op"],
                "seed": args["seed"], "target": target,
                "window_acc": result["window_accuracy_mean"],
                "window_acc_std": result["window_accuracy_std"],
                "trial_acc": result["trial_accuracy_mean"],
                "trial_acc_std": result["trial_accuracy_std"],
                "window_balanced_acc": sum(f["window"]["balanced_accuracy"] for f in folds) / len(folds),
                "trial_balanced_acc": sum(f["trial"]["balanced_accuracy"] for f in folds) / len(folds),
                "test_majority_fraction": sum(f["window"]["majority_baseline"] for f in folds) / len(folds),
                "subject_ci95_by_fold": json.dumps([f["trial"]["subject_bootstrap_ci95"] for f in folds]),
            })
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="?", default="runs")
    parser.add_argument("--output", type=Path, help="CSV path; default: stdout")
    args = parser.parse_args()
    rows = collect(args.runs)
    if not rows:
        print(f"No completed summaries found in {args.runs}", file=sys.stderr)
        return
    stream = args.output.open("w", newline="", encoding="utf-8") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            stream.close()


if __name__ == "__main__":
    main()
