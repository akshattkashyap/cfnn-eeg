"""Windowed FFT band-power extraction; no learned preprocessing."""
from __future__ import annotations

import numpy as np
from .constants import BANDS, CH0, FS


def band_masks(nfft: int):
    freq = np.fft.rfftfreq(nfft, 1.0 / FS)
    masks = []
    for j, (name, lo, hi) in enumerate(BANDS):
        m = (freq >= lo) & ((freq <= hi) if j == len(BANDS) - 1 else (freq < hi))
        if not m.any():
            raise ValueError(f"No FFT bin for band {name}; raise --nfft")
        masks.append(m)
    return masks


def trial_features(eeg, window, stride, nfft, taper, feature_dim, pad_mode, masks):
    """eeg (14, samples) -> (n_win, feature_dim). Channel-major, band-minor.
    stride=window means non-overlapping; stride<window means overlapping.
    """
    if eeg.ndim != 2 or eeg.shape[0] != len(CH0):
        raise ValueError("Expected EEG shape (14, samples)")
    if not 3 <= window <= eeg.shape[1] or nfft < window:
        raise ValueError("Window must be >=3 and fit the signal; nfft must be >= window")
    n_win = (eeg.shape[1] - window) // stride + 1
    starts = np.arange(n_win) * stride                      # (n_win,)
    idx = starts[:, None] + np.arange(window)[None, :]      # (n_win, window)
    x = eeg[:, idx].transpose(1, 0, 2).astype(np.float64)   # (n_win, 14, window)
    w = np.hanning(window) if taper == "hann" else np.ones(window)
    power = np.abs(np.fft.rfft(x * w, n=nfft, axis=-1)) ** 2 / (np.sum(w ** 2) + 1e-12)
    bands = np.stack([power[..., m].mean(-1) for m in masks], axis=-1)   # (n_win,14,5)
    bands = np.log10(bands + 1e-12)
    feats = bands.reshape(n_win, -1)                                      # 70
    if feature_dim == 73:
        if pad_mode == "summary":   # across-channel means of theta/alpha/gamma (invented bridge)
            extra = bands[:, :, [0, 1, 4]].mean(axis=1)
        else:                       # zeros: constant column, standardizes to 0
            extra = np.zeros((n_win, 3))
        feats = np.concatenate([feats, extra], axis=1)
    if not np.isfinite(feats).all():
        raise ValueError("Non-finite features")
    return feats.astype(np.float32)
