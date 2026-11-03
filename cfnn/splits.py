"""Holdout/CV partitions and row, trial and subject overlap audits."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit


def split_train_val(tv_idx, mode, subj, trial, seed, val_size):
    if mode == "random":
        a, b = next(ShuffleSplit(1, test_size=val_size, random_state=seed + 1).split(tv_idx))
    else:
        g = (subj if mode == "subject" else trial)[tv_idx]
        a, b = next(GroupShuffleSplit(1, test_size=val_size, random_state=seed + 1)
                    .split(tv_idx, groups=g))
    return tv_idx[a], tv_idx[b]


def make_splits(args, subj, trial):
    idx = np.arange(len(subj))
    out = []
    if args.cv_folds:
        if args.split == "random":
            it = KFold(args.cv_folds, shuffle=True, random_state=args.seed).split(idx)
        else:
            g = subj if args.split == "subject" else trial
            it = GroupKFold(args.cv_folds).split(idx, groups=g)
        for tv, te in it:
            tr, va = split_train_val(tv, args.split, subj, trial, args.seed, args.val_size)
            out.append((tr, va, te))
    else:
        if args.split == "random":
            tv, te = next(ShuffleSplit(1, test_size=args.test_size, random_state=args.seed).split(idx))
        else:
            g = subj if args.split == "subject" else trial
            tv, te = next(GroupShuffleSplit(1, test_size=args.test_size, random_state=args.seed)
                          .split(idx, groups=g))
        tr, va = split_train_val(tv, args.split, subj, trial, args.seed, args.val_size)
        out.append((tr, va, te))
    for tr, va, te in out:      # leakage audit
        if not all(len(i) for i in (tr, va, te)):
            raise ValueError("Each partition must contain at least one window")
        combined = np.concatenate([tr, va, te])
        if len(combined) != len(idx) or len(np.unique(combined)) != len(idx):
            raise ValueError("Split must assign every row exactly once")
        if args.split in ("trial", "subject"):
            for a, b in ((tr, te), (tr, va), (va, te)):
                assert not set(trial[a]) & set(trial[b]), "trial overlap"
        if args.split == "subject":
            for a, b in ((tr, te), (tr, va), (va, te)):
                assert not set(subj[a]) & set(subj[b]), "subject overlap"
    return out


def split_audit(tr, va, te, subj, trial):
    partitions = {"train": tr, "validation": va, "test": te}
    audit = {name: {"windows": len(i), "subjects": np.unique(subj[i]).tolist(),
                    "trials": np.unique(trial[i]).tolist()} for name, i in partitions.items()}
    audit["overlap"] = {}
    for a, b in (("train", "validation"), ("train", "test"), ("validation", "test")):
        ia, ib = partitions[a], partitions[b]
        audit["overlap"][f"{a}_{b}"] = {
            "rows": int(np.intersect1d(ia, ib).size),
            "subjects": int(np.intersect1d(subj[ia], subj[ib]).size),
            "trials": int(np.intersect1d(trial[ia], trial[ib]).size)}
    return audit
