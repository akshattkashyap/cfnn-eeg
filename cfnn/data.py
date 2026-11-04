"""Dataset dispatch, DEAP loading, feature caching and label thresholds."""
from __future__ import annotations

import hashlib
import json
import pickle
import warnings
from pathlib import Path

import numpy as np

from .constants import (BANDS, BASELINE, CACHE_VERSION, CHANNELS_1, CH0,
                        DREAMER_CHANNELS, FS, TRIAL_LEN)
from .features import band_masks, trial_features


def source_files(args):
    if getattr(args, "dataset", "deap") == "dreamer":
        from .dreamer import source_file
        return [source_file(args.data_dir)]
    files = sorted(Path(args.data_dir).glob("s[0-9][0-9].dat"))
    if args.max_subjects:
        files = files[:args.max_subjects]
    if len(files) < 4:
        raise ValueError(f"Need >=4 subject files in {args.data_dir}, found {len(files)}")
    return files


def feature_manifest(args, files):
    """Include source identity so changed/replaced data cannot reuse an old cache."""
    dataset = getattr(args, "dataset", "deap")
    manifest = {
        "dataset": dataset,
        "cache_version": CACHE_VERSION, "window_samples": args.window_samples,
        "stride": args.stride or args.window_samples,
        "nfft": args.nfft or max(args.window_samples, 128), "taper": args.taper,
        "feature_dim": args.feature_dim, "pad_mode": args.pad_mode,
        "channels_1based": list(CHANNELS_1), "bands": BANDS,
        "sample_rate": FS, "baseline_samples": BASELINE,
        "files": [{"path": str(f.resolve()), "size": f.stat().st_size,
                   "mtime_ns": f.stat().st_mtime_ns} for f in files],
    }
    if dataset == "dreamer":
        manifest.update({
            "channels_1based": list(range(1, 15)), "channel_names": list(DREAMER_CHANNELS),
            "baseline_samples": 0, "baseline_policy": "use separate EEG.stimuli only; no correction",
            "max_subjects": args.max_subjects,
            "last_stimulus_samples": round(getattr(args, "dreamer_last_seconds", 0.0) * FS),
        })
    return manifest


def feature_cache_path(args, manifest):
    if not args.cache_dir:
        return None
    Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
    key = json.dumps(manifest, sort_keys=True)
    return Path(args.cache_dir) / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".npz")


def read_feature_cache(cache):
    if cache is None or not cache.exists():
        return None
    with np.load(cache, allow_pickle=False) as z:
        arrays = tuple(z[k] for k in ("X", "R", "S", "T"))
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError(f"Non-finite values in cache {cache}; remove it and retry")
    print("Loaded cached features", cache, flush=True)
    return arrays


def write_feature_cache(cache, X, R, S, T):
    if cache is not None:
        tmp = cache.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, X=X, R=R, S=S, T=T)
        tmp.replace(cache)


def load_data(args):
    if getattr(args, "dataset", "deap") == "dreamer":
        from .dreamer import load_dreamer
        return load_dreamer(args)
    return load_deap(args)


def load_deap(args):
    files = source_files(args)
    manifest = feature_manifest(args, files)
    nfft = manifest["nfft"]
    masks = band_masks(nfft)
    cache = feature_cache_path(args, manifest)
    cached = read_feature_cache(cache)
    if cached is not None:
        return cached
    # Preallocate instead of retaining and concatenating millions of small windows.
    stride_ = args.stride or args.window_samples
    n_win = (TRIAL_LEN - args.window_samples) // stride_ + 1
    rows = len(files) * 40 * n_win
    X = np.empty((rows, args.feature_dim), dtype=np.float32)
    R = np.empty((rows, 2), dtype=np.float32)
    S = np.empty(rows, dtype=np.int32)
    T = np.empty(rows, dtype=np.int32)
    offset = 0
    for fp in files:
        with fp.open("rb") as h:
            obj = pickle.load(h, encoding="latin1")
        data, rat = np.asarray(obj["data"]), np.asarray(obj["labels"])
        if data.shape != (40, 40, 8064) or rat.shape != (40, 4):
            raise ValueError(f"Unexpected shapes in {fp.name}: {data.shape}, {rat.shape}")
        if not np.isfinite(rat).all():
            raise ValueError(f"Non-finite DEAP ratings in {fp.name}")
        if np.any((rat < 1) | (rat > 9)):
            warnings.warn(f"{fp.name} has ratings outside the guide's nominal [1, 9] range. "
                          "Keeping the supplied values and applying the configured threshold.")
        sid = int(fp.stem[1:])
        if not 1 <= sid <= 32:
            raise ValueError(f"Unexpected DEAP subject ID in {fp.name}")
        for t in range(40):
            f = trial_features(data[t, CH0, BASELINE:], args.window_samples, stride_, nfft,
                               args.taper, args.feature_dim, args.pad_mode, masks)
            sl = slice(offset, offset + len(f))
            X[sl], R[sl] = f, rat[t, :2]
            S[sl], T[sl] = sid, (sid - 1) * 40 + t
            offset += len(f)
        del obj, data, rat
        print("Processed", fp.name, flush=True)
    write_feature_cache(cache, X, R, S, T)
    return X, R, S, T


def binarize(ratings, thr, op):
    return (ratings >= thr if op == "ge" else ratings > thr).astype(np.int64)
