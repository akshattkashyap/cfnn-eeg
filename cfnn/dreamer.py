"""Read the original DREAMER MATLAB structure, keeping trials and subjects intact.

Schema reference (not a TorchEEG dependency):
https://torcheeg.readthedocs.io/en/v1.1.1/_modules/torcheeg/datasets/module/emotion_recognition/dreamer.html
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import loadmat

from .constants import DREAMER_CHANNELS, FS
from .data import feature_cache_path, feature_manifest, read_feature_cache, write_feature_cache
from .features import band_masks, trial_features


def source_file(path):
    """Accept either a .mat path or a directory containing DREAMER.mat."""
    path = Path(path)
    if path.is_file() and path.suffix.lower() == ".mat":
        return path
    if path.is_dir():
        matches = [p for p in path.iterdir() if p.is_file() and p.name.lower() == "dreamer.mat"]
        if len(matches) == 1:
            return matches[0]
    raise FileNotFoundError(
        f"Expected the original DREAMER.mat file or its containing directory, got {path}")


def _field(obj, name, context):
    if not hasattr(obj, name):
        raise ValueError(f"Missing {context}.{name}; use the original DREAMER.mat structure")
    return getattr(obj, name)


def _channel_order(root):
    if not hasattr(root, "EEG_Electrodes"):
        # The published dataset's standard order; never apply DEAP's channel indices.
        return np.arange(14)
    raw = np.asarray(root.EEG_Electrodes).reshape(-1)
    names = [str(n).strip().upper() for n in raw]
    if len(names) != 14 or set(names) != set(DREAMER_CHANNELS):
        raise ValueError("DREAMER.EEG_Electrodes must identify the 14 standard DREAMER channels")
    return np.array([names.index(name) for name in DREAMER_CHANNELS])


def _read_trials(path, args):
    try:
        content = loadmat(path, squeeze_me=True, struct_as_record=False, variable_names=["DREAMER"])
    except NotImplementedError as exc:
        raise ValueError("This loader expects the original SciPy-readable DREAMER.mat, "
                         "not a MATLAB v7.3/HDF5 conversion") from exc
    if "DREAMER" not in content:
        raise ValueError("MAT file must contain the original DREAMER structure")
    root = content["DREAMER"]
    if hasattr(root, "EEG_SamplingRate"):
        rate = np.asarray(root.EEG_SamplingRate).reshape(-1)
        if len(rate) != 1 or not np.isfinite(rate[0]) or rate[0] != FS:
            raise ValueError("DREAMER EEG sampling rate must be 128 Hz; resampling is not implicit")
    order = _channel_order(root)
    subjects = np.atleast_1d(_field(root, "Data", "DREAMER")).reshape(-1)
    if not 4 <= len(subjects) <= 23:
        raise ValueError(f"Expected 4..23 DREAMER subjects, found {len(subjects)}")
    subjects = subjects[:args.max_subjects] if args.max_subjects else subjects
    crop = round(getattr(args, "dreamer_last_seconds", 0.0) * FS)
    trials = []
    for sid, subject in enumerate(subjects, start=1):
        context = f"DREAMER subject {sid}"
        eeg = _field(subject, "EEG", context)
        stimuli = np.atleast_1d(_field(eeg, "stimuli", f"{context}.EEG")).reshape(-1)
        if len(stimuli) != 18:
            raise ValueError(f"{context}: expected 18 stimulus trials, found {len(stimuli)}")
        ratings = []
        for name in ("ScoreValence", "ScoreArousal"):
            values = np.asarray(_field(subject, name, context), dtype=np.float64).reshape(-1)
            if values.shape != (18,) or not np.isfinite(values).all():
                raise ValueError(f"{context}.{name}: expected 18 finite ratings")
            if np.any((values < 1) | (values > 5)):
                raise ValueError(f"{context}.{name}: ratings must be in [1, 5]; no label repair is applied")
            ratings.append(values)
        ratings = np.stack(ratings, axis=1)
        for ti, stimulus in enumerate(stimuli):
            context_trial = f"{context}, trial {ti + 1}"
            signal = np.asarray(stimulus)
            if signal.ndim != 2 or signal.shape[1] != 14:
                raise ValueError(f"{context_trial}: expected stimulus shape (samples, 14), "
                                 f"got {signal.shape}")
            if not np.issubdtype(signal.dtype, np.number) or np.iscomplexobj(signal):
                raise ValueError(f"{context_trial}: EEG must be real numeric samples")
            if not np.isfinite(signal).all():
                raise ValueError(f"{context_trial}: non-finite EEG samples")
            if crop:
                if len(signal) < crop:
                    raise ValueError(f"{context_trial}: fewer samples than --dreamer-last-seconds requests")
                signal = signal[-crop:]
            if len(signal) < args.window_samples:
                raise ValueError(f"{context_trial}: stimulus is shorter than --window-samples")
            # Baseline is a separate field. Do not drop DEAP's first 384 samples here.
            signal = signal.T
            if not np.array_equal(order, np.arange(14)):
                signal = signal[order]
            trials.append((signal, ratings[ti], sid, (sid - 1) * 18 + ti))
    return trials


def load_dreamer(args):
    path = source_file(args.data_dir)
    manifest = feature_manifest(args, [path])
    cache = feature_cache_path(args, manifest)
    cached = read_feature_cache(cache)
    if cached is not None:
        return cached
    trials = _read_trials(path, args)
    stride = args.stride or args.window_samples
    rows = sum((signal.shape[1] - args.window_samples) // stride + 1
               for signal, _, _, _ in trials)
    X = np.empty((rows, args.feature_dim), dtype=np.float32)
    R = np.empty((rows, 2), dtype=np.float32)
    S = np.empty(rows, dtype=np.int32)
    T = np.empty(rows, dtype=np.int32)
    nfft = manifest["nfft"]
    masks = band_masks(nfft)
    offset = 0
    for signal, ratings, sid, tid in trials:
        features = trial_features(signal, args.window_samples, stride, nfft, args.taper,
                                  args.feature_dim, args.pad_mode, masks)
        sl = slice(offset, offset + len(features))
        X[sl], R[sl], S[sl], T[sl] = features, ratings, sid, tid
        offset += len(features)
        if tid % 18 == 17:
            print(f"Processed DREAMER subject {sid:02d}", flush=True)
    write_feature_cache(cache, X, R, S, T)
    return X, R, S, T
