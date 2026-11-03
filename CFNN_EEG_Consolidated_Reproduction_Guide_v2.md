# CFNN for EEG Emotion Recognition: Consolidated Reproduction Guide (v2)

**Paper:** Nobari Azar, Cavus, Esmaili, Sekeroglu & Aşır, *Detecting emotions through EEG signals based on modified convolutional fuzzy neural network*, Scientific Reports 14:10371 (2024). DOI: <https://doi.org/10.1038/s41598-024-60977-9>
**Dataset:** DEAP, <https://www.eecs.qmul.ac.uk/mmv/datasets/deap/>

This guide merges two earlier reproduction guides into one. It keeps the best parts of each:

- a single runnable script with leakage audits, validation-based checkpointing, trial-level scoring and JSON reports;
- the paper's own evaluation protocol (random window-level split) and its window-size sweep, so the headline numbers can be *measured*, not just doubted;
- all four baselines from the paper's Table 6, plus ablations that isolate what the fuzzy layers actually contribute.

> **Status of the code.** `train_cfnn.py` was tested for this guide on **synthetic DEAP-shaped data** (TensorFlow 2.21, CPU): all model self-tests pass, parameter counts match the paper (**11,186** for 73 inputs; **11,058** for 70), and every split mode, model kind and the CV path ran end to end. It has **not** been run on real DEAP, and **no real result is reported here**. Fill the results tables yourself.

---

## Contents

1. [Aim and reproduction standard](#1-aim-and-reproduction-standard)
2. [What the paper reports](#2-what-the-paper-reports)
3. [Decisions and assumptions](#3-decisions-and-assumptions)
4. [Paper discrepancy audit](#4-paper-discrepancy-audit)
5. [Data](#5-data)
6. [Environment](#6-environment)
7. [Feature pipeline](#7-feature-pipeline)
8. [Model specification](#8-model-specification)
9. [Evaluation design](#9-evaluation-design)
10. [The script](#10-the-script-train_cfnnpy)
11. [Experiment plan and commands](#11-experiment-plan-and-commands)
12. [Reading the results](#12-reading-the-results)
13. [Debugging](#13-debugging)
14. [Reporting template](#14-reporting-template)
15. [Checklist](#15-checklist)
16. [References](#16-references)

---

## 1. Aim and reproduction standard

The paper trains two **independent binary classifiers** on DEAP, one for valence and one for arousal. It does not demonstrate a multi-class emotion classifier.

| Level | Goal | Status |
|---|---|---|
| **A. Specification** | Match named layers, kernels, parameter count, optimizer, lr, batch size, epochs | **Achieved** (11,186 params; verified by `--self-test`) |
| **B. Functional** | A trainable FFT → CNN → fuzzy → defuzzy classifier on public DEAP | **Provided** (`train_cfnn.py`) |
| **C. Exact** | Match the authors' features, fuzzy math, splits and 98% | **Not possible from the paper alone** (no code, splits undocumented) |

Describe results from this guide as an *independent reconstruction inspired by the published CFNN*, never as the authors' implementation.

**Milestones**

- **M0** model builds, 11,186 params, finite gradients (`--self-test`)
- **M1** real DEAP features load, finite, with subject/trial ids kept
- **M2** paper-style random split at window 32, to see whether ~97-98% is reachable at all
- **M3** the same model under trial-grouped and subject-grouped splits
- **M4** baselines and ablations on identical splits
- **M5** explain any gap to the paper; never tune on the test split to close it

---

## 2. What the paper reports

| Item | Value | Where |
|---|---|---|
| Data | DEAP, 32 subjects x 40 one-minute music videos, 32 EEG channels, 128 Hz after resampling | p. 4 |
| Channels used | 1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31 (14) | Table 1, p. 7 |
| Bands | theta 4-8, alpha 8-12, beta-low 12-16, beta-high 16-25, gamma 25-45 Hz | Table 1 |
| Features | FFT band power; window sizes 4, 8, 16, 32, 64, 128 | pp. 6, 9 |
| Model | Conv1D(64, k5) → Conv1D(32, k3) → MaxPool → Flatten → Fuzzy → BatchNorm → Dense(2) → Defuzzy → Softmax | Fig. 3, Table 2 |
| Parameters | 11,186 total; 11,182 trainable; 4 non-trainable | Table 2 |
| Training | Adam, lr 0.01, batch 256, 100 epochs | p. 7 |
| Splits | hold-out 90:10, 80:20, 75:25, plus 5-fold CV | p. 8, Table 3 |
| Chosen setting | 75:25, window 32 | p. 9 |
| Claimed result | Valence 98.21%, arousal 98.08% accuracy | Table 5 |
| Baselines (Table 6) | SVM 55/54, KNN 61/60, Bi-LSTM 72.01/70.42, CNN 88.87/83.35 (valence/arousal) | p. 10 |
| External validation | none; the paper lists this as a limitation | p. 9 |
| Software | Python 3.10.12, TensorFlow 2.12.0, Colab | p. 7 |

These are *reported* numbers, not acceptance criteria.

---

## 3. Decisions and assumptions

Tags: **[PAPER]** stated; **[ASSUMED]** unstated, my choice; **[DEVIATION]** deliberate change. Every row maps to a script flag so you can vary it.

| Open question | Choice (default) | Tag | Flag |
|---|---|---|---|
| Channel numbering | 1-based, subtract 1 | ASSUMED | n/a |
| File version | DEAP preprocessed Python `.dat` | ASSUMED | n/a |
| Baseline period | drop first 3 s (384 samples) | ASSUMED | n/a |
| Extra filtering | none (DEAP is already 4-45 Hz, EOG-cleaned) | ASSUMED | n/a |
| Window unit | **samples** (paper: 4..128); default **512 = 4 s** for the principled arm, **32** for the paper arm | ASSUMED | `--window-samples` |
| FFT details | Hann taper, power normalized by sum of w², `log10`, zero-padded to `nfft = max(window, 128)` | ASSUMED | `--taper`, `--nfft` |
| The extra 3 inputs (70 → 73) | **zero-pad** (default) or across-channel means of theta/alpha/gamma | ASSUMED | `--pad-mode`, `--feature-dim` |
| Label threshold | `rating >= 5` is high | ASSUMED | `--threshold`, `--threshold-op ge/gt` |
| Fuzzy math | learnable Gaussian centers and softplus widths; aggregation `mean_exp` | ASSUMED | `--fuzzy-agg` |
| Defuzz | softplus on Dense output, then `(a @ C) / sum(a)` with 2x2 `C` | ASSUMED | n/a |
| Loss | sparse categorical cross-entropy | ASSUMED | n/a |
| Checkpoint | best **validation** loss | DEVIATION | `--early-stopping` optional |
| Primary evaluation | grouped splits, with the paper's random split as a separate arm | DEVIATION | `--split` |
| Optimizer/lr/batch/epochs | Adam / 0.01 / 256 / 100 | PAPER | `--learning-rate` etc. |
| Library versions | any recent TF; record versions | DEVIATION | n/a |

---

## 4. Paper discrepancy audit

Things in the paper that do not line up, and how this guide handles each.

| # | Issue | Handling |
|---|---|---|
| 1 | Table 2 output shapes are shifted/mislabeled; "5x5" and "3x3" kernels are really 1D (5, 3). The param counts fix the true shapes: `64*5*1+64 = 384`, `32*3*64+32 = 6176` | Use true shapes |
| 2 | 14 channels x 5 bands = **70**, but Table 2 implies **73** | `--feature-dim 70/73`, `--pad-mode` |
| 3 | Eq. 6 omits the square on the numerator | Standard Gaussian |
| 4 | Three different "75:25" accuracies: 97.24% (Table 3), 97.88% (Table 4, W=32), 98.21%/98.08% (Table 5) | Don't expect one number |
| 5 | Table 4, W=64: precision/F1/recall identical to W=8 while accuracy differs; likely a copy-paste error | Treat as unreliable |
| 6 | Text selects W=32, but W=128 scores higher (98.70%) | Sweep all windows, select on validation |
| 7 | Text says K=5 CV was best, yet all later experiments use 75:25 | n/a |
| 8 | 95% CIs of about +/-0.05% on hundreds of thousands of correlated windows | Don't reproduce; use subject-level CIs |
| 9 | No baseline hyperparameters; SVM 55% / KNN 61% are unusually low | Tune baselines fairly |
| 10 | Splits are not described beyond ratios, and windows share labels within a trial | Run random, trial and subject arms |
| 11 | Table 7 mixes studies with different datasets and protocols (e.g. one DEAP-SEED row) | Don't treat as like-for-like |
| 12 | "EEG cannot be tampered with" is asserted without support | Ignore |
| 13 | Xing et al. is ref 24 in the Discussion but ref 27 in the tables | Minor |

The paper does **not** prove leakage occurred. It simply does not document the splits well enough to rule it out. That is why this guide measures it instead of asserting it.

---

## 5. Data

Download the **preprocessed Python** DEAP release (EULA required) and extract `s01.dat`..`s32.dat` to `data/deap/data_preprocessed_python/`. Do not commit or redistribute the data. (Not to be confused with the unrelated `deap` evolutionary-computation package.)

| Key | Shape | Meaning |
|---|---|---|
| `data` | `(40, 40, 8064)` | 40 trials x 40 channels x 63 s at 128 Hz |
| `labels` | `(40, 4)` | valence, arousal, dominance, liking (1-9) |

Channels 0-31 are EEG (Geneva order); 32-39 are peripheral. The first 384 samples are pre-trial baseline; the next 7,680 are the 60 s stimulus.

**Channel names (Geneva order, verify against the DEAP docs for your copy):** 1 Fp1, 2 AF3, 3 F3, 4 F7, 5 FC5, 6 FC1, 7 C3, 8 T7, 9 CP5, 10 CP1, 11 P3, 12 P7, 13 PO3, 14 O1, 15 Oz, 16 Pz, 17 Fp2, 18 AF4, 19 Fz, 20 F4, 21 F8, 22 FC6, 23 FC2, 24 Cz, 25 C4, 26 T8, 27 CP6, 28 CP2, 29 P4, 30 P8, 31 PO4, 32 O2. The paper's selection is therefore Fp1, AF3, F3, F7, FC1, P3, PO3, Fp2, Fz, F4, F8, C4, P4, PO4, which is not the Emotiv EPOC set.

Inspect before training:

```python
import pickle
d = pickle.load(open("data/deap/data_preprocessed_python/s01.dat", "rb"), encoding="latin1")
print(d["data"].shape, d["labels"].shape)   # (40, 40, 8064) (40, 4)
```

Suggested layout:

```text
cfnn-eeg/
├── train_cfnn.py
├── collect.py
├── data/deap/data_preprocessed_python/s01.dat ... s32.dat
├── cache/          # cached feature arrays (--cache-dir)
├── runs/           # one folder per experiment
└── .gitignore      # data/, runs/, cache/, .venv/, *.dat, *.h5
```

---

## 6. Environment

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install tensorflow numpy scipy scikit-learn
python -m pip freeze > environment.txt
python -c "import tensorflow as tf; print(tf.__version__, tf.config.list_physical_devices('GPU'))"
```

- The script is tested on TF 2.21 (Keras 3). It needs no GPU; the model has about 11k parameters, so runtime is dominated by the **number of windows**.
- To mimic the paper's stack, use a separate Python 3.10 env with `tensorflow==2.12.0` and `numpy<2`. Record whichever you use.
- `--model svm` and `--model knn` run without TensorFlow.

**Window count and memory (1,280 trials x `7680 // W` windows; 73 float32 features is about 292 bytes/row):**

| W (samples) | Seconds | Windows/trial | Total rows | Approx. RAM | True freq. resolution |
|---:|---:|---:|---:|---:|---:|
| 4 | 0.03 | 1920 | 2,457,600 | 0.7 GB | 32 Hz |
| 8 | 0.06 | 960 | 1,228,800 | 0.36 GB | 16 Hz |
| 16 | 0.13 | 480 | 614,400 | 0.18 GB | 8 Hz |
| 32 | 0.25 | 240 | 307,200 | 0.09 GB | 4 Hz |
| 64 | 0.5 | 120 | 153,600 | 0.04 GB | 2 Hz |
| 128 | 1 | 60 | 76,800 | 0.02 GB | 1 Hz |
| 512 | 4 | 15 | 19,200 | 0.006 GB | 0.25 Hz |

Training time on the small windows is large (2.4M rows x 100 epochs). Use `--cache-dir` and start with W >= 32.

---

## 7. Feature pipeline

```text
DEAP .dat (128 Hz) -> 14 channels -> drop 3 s baseline -> 7680 samples
  -> non-overlapping windows of W samples
  -> taper (Hann) -> rFFT zero-padded to nfft -> |X|^2 / sum(w^2)
  -> mean power per band (5 bands) -> log10
  -> 14 x 5 = 70 features (channel-major) [+3 for 73]
  -> StandardScaler fit on TRAIN rows only
  -> (batch, 70|73, 1)
```

**Band masks** are half-open `[lo, hi)`, with gamma closed at 45 Hz, so no bin is counted twice. `nfft = max(W, 128)` guarantees every band contains at least one bin.

**Zero-padding the FFT interpolates; it does not add resolution.** At W = 4, 8 or 16 samples, the true resolution (32, 16, 8 Hz) is wider than the 4 Hz-wide theta band, so the five band powers are heavily smeared and largely redundant. Treat the paper's smallest windows with suspicion (the paper itself says W = 4 and 8 "might not be suitable").

**70 vs 73.** Table 2's shapes only work for length 73 (`73 → 36 → 1152`). Two bridges are offered:

- `--pad-mode zero` (default): three constant zero columns. They standardize to 0, so they inject no invented information.
- `--pad-mode summary`: across-channel means of theta, alpha, gamma. These are linear combinations of existing features and are **not** in the paper.
- `--feature-dim 70` is the principled version with no bridge: flatten becomes `35 x 32 = 1120`, fuzzy params `4,480`, total **11,058**.

---

## 8. Model specification

### 8.1 Layers (73-input mode)

| # | Layer | True output | Params | Check |
|---|---|---|---:|---|
| 1 | Conv1D 64, k5, same, ReLU | (73, 64) | 384 | 64x5x1 + 64 |
| 2 | Conv1D 32, k3, same, ReLU | (73, 32) | 6,176 | 32x3x64 + 32 |
| 3 | MaxPool1D(2) | (36, 32) | 0 | floor(73/2) = 36 |
| 4 | Flatten | (1152) | 0 | 36x32 |
| 5 | Fuzzification (2 rules) | (2) | 4,608 | 1152x2 centers + 1152x2 widths |
| 6 | BatchNorm | (2) | 8 | 4 trainable + 4 non-trainable |
| 7 | Dense(2) | (2) | 6 | 2x2 + 2 |
| 8 | Defuzzification | (2) | 4 | 2x2 matrix `C` |
| 9 | Softmax | (2) | 0 | |
| | **Total** | | **11,186** | trainable 11,182, non-trainable 4 |

### 8.2 Fuzzification (reconstruction)

For flattened features `z` (d = 1152) and rule `j`:

```text
z_ij = 0.5 * ((x_i - c_ij) / sigma_ij)^2 ,   sigma_ij = softplus(raw_ij) + 1e-5
```

Three aggregations are available via `--fuzzy-agg`:

| Value | Formula | Notes |
|---|---|---|
| `mean_exp` (default) | `exp(-mean_i z_ij)` | geometric-mean AND; numerically stable |
| `prod` | `exp(-sum_i z_ij)` | literal product of memberships; underflows toward 0 at d = 1152 |
| `mean_mem` | `mean_i exp(-z_ij)` | arithmetic mean of memberships (OR-like) |

The paper gives no executable form for this layer, so all three are plausible and **none is verified**. Comparing them is a cheap sensitivity experiment.

### 8.3 Defuzzification (reconstruction of Eq. 10)

```text
a = softplus(Dense output)           # nonnegative firing strengths
v = (a @ C) / (sum(a) + 1e-7)        # C is 2x2, 4 parameters
```

The paper's formula divides by a sum, so the inputs must be nonnegative; the softplus is my addition.

### 8.4 Structural note: the head is a one-dimensional bottleneck

Because `v` is a weighted average with weights `a0/(a0+a1)` and `a1/(a0+a1)`, the output logits are a **convex combination of the two rows of C**. Everything downstream of the Dense layer therefore reduces to a single scalar `t = a0 / (a0 + a1)`, and the class decision is a threshold on `t`. Any benefit of the fuzzy head must pass through that one scalar. This is a property of Eq. 10 as written, not of this particular code, and it is why the ablations below matter.

### 8.4b Model kinds in the script

| `--model` | Description | Purpose |
|---|---|---|
| `cfnn` | the reconstructed CFNN | the model under study |
| `cfnn_nodefuzz` | fuzzy → BN → Dense(2) → softmax | does defuzzification matter? |
| `plain` | conv stack → Dense(2, ReLU) → BN → Dense(2) | same layout, fuzzy block removed |
| `plain_matched` | conv stack → Dense(4, ReLU) → BN → Dense(2) | about the same parameter count (11,198) as CFNN |
| `cnn` | conv stack → Dense(2) | plain CNN baseline |
| `bilstm` | Bidirectional LSTM(32) → Dense(2) | paper baseline (hyperparameters assumed) |
| `svm` | RBF SVM, C=1, subsampled to `--sk-max-train` | paper baseline |
| `knn` | k = 5 | paper baseline |

---

## 9. Evaluation design

### 9.1 Why several protocols

All windows from one trial share one label, and a subject's EEG has strong person-specific structure. A split that ignores this can reward memorizing the trial or the person. So the same model is evaluated under splits of increasing strictness:

| `--split` | Held out | Measures | Leakage |
|---|---|---|---|
| `random` | random windows | **the paper's protocol** | windows of the same trial/subject in train and test |
| `trial` | whole subject-trials | new videos for known people | subject leakage only |
| `subject` | whole subjects | new people | none of these |
| any, with `--cv-folds K` | K grouped folds | variability across splits | as above |

Comparing the three at identical features and seeds attributes any drop to leakage, not to a different model.

### 9.2 Protocol rules (enforced in code)

- Features and scaler are fit on **training rows only**.
- A **validation set** is carved from the training side only (random / by trial / by subject to match the split mode); the best-validation-loss checkpoint is used and the **test split is read once**.
- Disjointness is asserted: no shared trials for `trial`/`subject`, no shared subjects for `subject`.
- Labels: one label per trial, verified when aggregating.

### 9.3 Metrics reported

For every fold and target: **window-level** and **trial-level** (class probabilities averaged over a trial's windows) accuracy, balanced accuracy, precision, recall, F1, confusion matrix `[low, high]`, positive fraction and **majority-class baseline**. Trial-level results add per-subject accuracy and a subject-bootstrap 95% CI. The trial-level number matches the unit that was actually annotated, so prefer it for conclusions.

---

## 10. The script (`train_cfnn.py`)

Save as `train_cfnn.py` in the project root. Requires Python 3.10+, numpy, scikit-learn, and TensorFlow for the neural models.

```python
#!/usr/bin/env python3
"""Independent reconstruction of the 2024 Sci. Reports CFNN paper (DEAP, valence/arousal).

NOT the authors' code. Every choice the paper leaves open is an explicit, flagged option.
Paper: https://doi.org/10.1038/s41598-024-60977-9

Split modes
  random  : paper-style random window-level split (windows of one trial can straddle train/test)
  trial   : whole subject-trials held out (no within-trial leakage)
  subject : whole subjects held out (subject-independent)
Add --cv-folds K for grouped K-fold instead of one hold-out.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold, ShuffleSplit
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
except ImportError:  # sklearn-only models (svm/knn) still work without TensorFlow
    tf = None

# ----------------------------------------------------------------------------- constants
CHANNELS_1 = (1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31)   # paper Table 1 (1-based)
CH0 = np.asarray([c - 1 for c in CHANNELS_1])
BANDS = (("theta", 4.0, 8.0), ("alpha", 8.0, 12.0), ("beta_low", 12.0, 16.0),
         ("beta_high", 16.0, 25.0), ("gamma", 25.0, 45.0))
FS = 128
BASELINE = 3 * FS            # first 3 s of every DEAP trial is pre-trial baseline
TRIAL_LEN = 60 * FS          # 7680 samples of stimulus period
LABEL_COLS = {"valence": 0, "arousal": 1}
NEURAL = {"cfnn", "cfnn_nodefuzz", "plain", "plain_matched", "cnn", "bilstm"}
SKLEARN = {"svm", "knn"}

# ----------------------------------------------------------------------------- features


def band_masks(nfft: int):
    freq = np.fft.rfftfreq(nfft, 1.0 / FS)
    masks = []
    for j, (name, lo, hi) in enumerate(BANDS):
        m = (freq >= lo) & ((freq <= hi) if j == len(BANDS) - 1 else (freq < hi))
        if not m.any():
            raise ValueError(f"No FFT bin for band {name}; raise --nfft")
        masks.append(m)
    return masks


def trial_features(eeg, window, nfft, taper, feature_dim, pad_mode, masks):
    """eeg (14, 7680) -> (n_win, feature_dim). Channel-major, band-minor."""
    n_win = eeg.shape[1] // window
    x = eeg[:, :n_win * window].reshape(14, n_win, window).transpose(1, 0, 2).astype(np.float64)
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


def load_deap(args):
    nfft = args.nfft or max(args.window_samples, 128)
    key = json.dumps([args.window_samples, nfft, args.taper, args.feature_dim,
                      args.pad_mode, args.max_subjects, str(args.data_dir)])
    cache = None
    if args.cache_dir:
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        cache = Path(args.cache_dir) / (hashlib.md5(key.encode()).hexdigest() + ".npz")
        if cache.exists():
            z = np.load(cache)
            print("Loaded cached features", cache)
            return z["X"], z["R"], z["S"], z["T"]
    files = sorted(Path(args.data_dir).glob("s[0-9][0-9].dat"))
    if args.max_subjects:
        files = files[:args.max_subjects]
    if len(files) < 4:
        raise SystemExit(f"Need >=4 subject files in {args.data_dir}, found {len(files)}")
    if not 0 < args.window_samples <= TRIAL_LEN:
        raise SystemExit("--window-samples must be in (0, 7680]")
    masks = band_masks(nfft)
    X, R, S, T = [], [], [], []
    for fp in files:
        with fp.open("rb") as h:
            obj = pickle.load(h, encoding="latin1")
        data, rat = np.asarray(obj["data"]), np.asarray(obj["labels"])
        if data.shape != (40, 40, 8064) or rat.shape != (40, 4):
            raise SystemExit(f"Unexpected shapes in {fp.name}: {data.shape}, {rat.shape}")
        sid = int(fp.stem[1:])
        for t in range(40):
            f = trial_features(data[t, CH0, BASELINE:], args.window_samples, nfft, args.taper,
                               args.feature_dim, args.pad_mode, masks)
            X.append(f)
            R.append(np.repeat(rat[t:t + 1, :2], len(f), axis=0))
            S.append(np.full(len(f), sid))
            T.append(np.full(len(f), (sid - 1) * 40 + t))
        print("Processed", fp.name, flush=True)
    X, R = np.concatenate(X), np.concatenate(R).astype(np.float32)
    S, T = np.concatenate(S), np.concatenate(T)
    if cache:
        np.savez_compressed(cache, X=X, R=R, S=S, T=T)
    return X, R, S, T


def binarize(ratings, thr, op):
    return (ratings >= thr if op == "ge" else ratings > thr).astype(np.int64)

# ----------------------------------------------------------------------------- splits


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
        if args.split in ("trial", "subject"):
            for a, b in ((tr, te), (tr, va), (va, te)):
                assert not set(trial[a]) & set(trial[b]), "trial overlap"
        if args.split == "subject":
            for a, b in ((tr, te), (tr, va), (va, te)):
                assert not set(subj[a]) & set(subj[b]), "subject overlap"
    return out

# ----------------------------------------------------------------------------- models
if tf is not None:

    class GaussianFuzzification(tf.keras.layers.Layer):
        """RECONSTRUCTION. Gaussian membership per (input, rule); 2 rules -> 2*d*2 params.
        agg: mean_exp  = exp(-mean_i z_ij)  (geometric-mean AND, stable)
             prod      = exp(-sum_i z_ij)   (literal product; underflows at d=1152)
             mean_mem  = mean_i exp(-z_ij)  (arithmetic mean of memberships)
        """

        def __init__(self, units=2, agg="mean_exp", **kw):
            super().__init__(**kw)
            self.units, self.agg = units, agg

        def build(self, input_shape):
            n = int(input_shape[-1])
            self.centers = self.add_weight(
                name="centers", shape=(n, self.units),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.5))
            self.raw_scales = self.add_weight(          # softplus(0.5413) ~= 1.0
                name="raw_scales", shape=(n, self.units),
                initializer=tf.keras.initializers.Constant(0.5413))
            super().build(input_shape)

        def call(self, x):
            v = tf.expand_dims(x, -1)                                   # (B,d,1)
            sigma = tf.nn.softplus(self.raw_scales) + 1e-5
            z = 0.5 * tf.square((v - self.centers) / sigma)             # (B,d,u)
            if self.agg == "mean_exp":
                return tf.exp(-tf.reduce_mean(z, axis=1))
            if self.agg == "prod":
                return tf.exp(-tf.reduce_sum(z, axis=1))
            return tf.reduce_mean(tf.exp(-z), axis=1)

        def get_config(self):
            c = super().get_config()
            c.update(units=self.units, agg=self.agg)
            return c

    class WeightedDefuzzification(tf.keras.layers.Layer):
        """RECONSTRUCTION of Eq. 10: v = (a @ C) / sum(a), a = softplus(input), C is 2x2 (4 params)."""

        def __init__(self, units=2, **kw):
            super().__init__(**kw)
            self.units = units

        def build(self, input_shape):
            self.cores = self.add_weight(
                name="cores", shape=(int(input_shape[-1]), self.units),
                initializer=tf.keras.initializers.RandomNormal(stddev=0.5))
            super().build(input_shape)

        def call(self, x):
            a = tf.nn.softplus(x)
            return tf.matmul(a, self.cores) / (tf.reduce_sum(a, -1, keepdims=True) + 1e-7)

        def get_config(self):
            c = super().get_config()
            c.update(units=self.units)
            return c


def build_model(kind, length, agg="mean_exp"):
    L = tf.keras.layers
    inp = L.Input(shape=(length, 1), name="features")
    if kind == "bilstm":
        x = L.Bidirectional(L.LSTM(32))(inp)
        return tf.keras.Model(inp, L.Dense(2, activation="softmax")(x), name=kind)
    x = L.Conv1D(64, 5, padding="same", activation="relu", name="conv1")(inp)
    x = L.Conv1D(32, 3, padding="same", activation="relu", name="conv2")(x)
    x = L.MaxPooling1D(2, name="pool")(x)
    x = L.Flatten(name="flatten")(x)
    if kind == "cfnn":
        x = GaussianFuzzification(2, agg, name="fuzzy")(x)
        x = L.BatchNormalization(name="bn")(x)
        x = L.Dense(2, name="dense")(x)
        x = WeightedDefuzzification(2, name="defuzzy")(x)
    elif kind == "cfnn_nodefuzz":
        x = GaussianFuzzification(2, agg, name="fuzzy")(x)
        x = L.BatchNormalization(name="bn")(x)
        x = L.Dense(2, name="dense")(x)
    elif kind == "plain":              # same layout, fuzzy block replaced by Dense(2, relu)
        x = L.Dense(2, activation="relu", name="plain2")(x)
        x = L.BatchNormalization(name="bn")(x)
        x = L.Dense(2, name="dense")(x)
    elif kind == "plain_matched":      # ~parameter-matched to the fuzzy layer (1152*4+4 = 4612)
        x = L.Dense(4, activation="relu", name="plain4")(x)
        x = L.BatchNormalization(name="bn")(x)
        x = L.Dense(2, name="dense")(x)
    elif kind == "cnn":
        x = L.Dense(2, name="logits")(x)
    else:
        raise ValueError(kind)
    return tf.keras.Model(inp, L.Softmax(name="out")(x), name=kind)

# ----------------------------------------------------------------------------- training


def train_neural(kind, args, xtr, ytr, xva, yva, xte, ckpt):
    tf.keras.backend.clear_session()
    model = build_model(kind, args.feature_dim, args.fuzzy_agg)
    opt = tf.keras.optimizers.Adam(learning_rate=args.learning_rate,
                                   **({"clipnorm": args.clipnorm} if args.clipnorm > 0 else {}))
    model.compile(opt, loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=["accuracy"])
    cbs = [tf.keras.callbacks.ModelCheckpoint(str(ckpt), monitor="val_loss", mode="min",
                                              save_best_only=True, save_weights_only=True)]
    if args.early_stopping:
        cbs.append(tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=12,
                                                    restore_best_weights=True))
    h = model.fit(xtr[..., None], ytr, validation_data=(xva[..., None], yva),
                  epochs=args.epochs, batch_size=args.batch_size, shuffle=True,
                  callbacks=cbs, verbose=2)
    model.load_weights(str(ckpt))                    # best VALIDATION-loss weights only
    probs = model.predict(xte[..., None], batch_size=1024, verbose=0)
    return probs, {"params": int(model.count_params()),
                   "epochs_run": len(h.history["loss"]),
                   "best_val_loss": float(min(h.history["val_loss"])),
                   "history": {k: [float(v) for v in vs] for k, vs in h.history.items()}}


def train_sklearn(kind, args, xtr, ytr, xte):
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    if len(xtr) > args.sk_max_train:
        sel = np.random.default_rng(args.seed).choice(len(xtr), args.sk_max_train, replace=False)
        xtr, ytr = xtr[sel], ytr[sel]
    if kind == "svm":
        d = SVC(kernel="rbf", C=1.0, gamma="scale").fit(xtr, ytr).decision_function(xte)
        p1 = 1.0 / (1.0 + np.exp(-d))
        return np.stack([1 - p1, p1], 1), {"train_rows_used": int(len(xtr))}
    p = KNeighborsClassifier(5).fit(xtr, ytr).predict_proba(xte)
    return p, {"train_rows_used": int(len(xtr))}

# ----------------------------------------------------------------------------- metrics


def metrics(y, p):
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
    ys, ps, ss = [], [], []
    for t in np.unique(trial):
        m = trial == t
        assert len(np.unique(y[m])) == 1, "one label per trial expected"
        ys.append(y[m][0]); ps.append(p[m].mean(0)); ss.append(subj[m][0])
    ys, ps, ss = np.asarray(ys), np.asarray(ps), np.asarray(ss)
    res = metrics(ys, ps)
    per = {int(s): float(accuracy_score(ys[ss == s], ps[ss == s].argmax(1))) for s in np.unique(ss)}
    res["per_subject_accuracy"] = per
    res["subject_bootstrap_ci95"] = bootstrap_ci(list(per.values()))
    return res

# ----------------------------------------------------------------------------- main


def self_test(args):
    if args.model not in NEURAL:
        raise SystemExit("self-test is for neural models")
    m = build_model(args.model, args.feature_dim, args.fuzzy_agg)
    m.summary()
    x = np.random.default_rng(args.seed).normal(size=(8, args.feature_dim, 1)).astype("float32")
    y = np.random.default_rng(1).integers(0, 2, 8)
    with tf.GradientTape() as tape:
        p = m(x, training=True)
        loss = tf.keras.losses.SparseCategoricalCrossentropy()(y, p)
    grads = tape.gradient(loss, m.trainable_variables)
    assert p.shape == (8, 2)
    np.testing.assert_allclose(p.numpy().sum(1), 1.0, atol=1e-5)
    assert all(g is not None and np.isfinite(g.numpy()).all() for g in grads), "bad gradients"
    if args.model == "cfnn":
        want = {73: 11186, 70: 11058}[args.feature_dim]
        assert m.count_params() == want, (m.count_params(), want)
    print("SELF-TEST OK | params:", m.count_params(), "| loss:", float(loss))


def run(args):
    np.random.seed(args.seed); random.seed(args.seed)
    if tf is not None:
        tf.keras.utils.set_random_seed(args.seed)
    if args.self_test:
        return self_test(args)
    if args.model in NEURAL and tf is None:
        raise SystemExit("TensorFlow is required for neural models")
    if not args.data_dir:
        raise SystemExit("--data-dir is required")
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    X, R, subj, trial = load_deap(args)
    print(f"Feature matrix {X.shape}; subjects {len(np.unique(subj))}; trials {len(np.unique(trial))}")
    if args.split == "random":
        print("WARNING: random window-level split; windows from one trial can be in train AND test "
              "(paper-style protocol, expected to be optimistic).")
    splits = make_splits(args, subj, trial)
    targets = list(LABEL_COLS) if args.target == "both" else [args.target]
    folds = {t: [] for t in targets}

    for fi, (tr, va, te) in enumerate(splits):
        sc = StandardScaler().fit(X[tr])                       # TRAIN ONLY
        xtr, xva, xte = (sc.transform(X[i]).astype(np.float32) for i in (tr, va, te))
        for tgt in targets:
            y = binarize(R[:, LABEL_COLS[tgt]], args.threshold, args.threshold_op)
            print(f"[fold {fi}] {args.model} {tgt}: train hist {np.bincount(y[tr], minlength=2)}")
            if args.model in NEURAL:
                probs, info = train_neural(args.model, args, xtr, y[tr], xva, y[va], xte,
                                           out / f"{args.model}_{tgt}_f{fi}.weights.h5")
            else:
                probs, info = train_sklearn(args.model, args, xtr, y[tr], xte)
            folds[tgt].append({
                "fold": fi,
                "test_subjects": np.unique(subj[te]).tolist(),
                "n_test_windows": int(len(te)),
                "window": metrics(y[te], probs),
                "trial": trial_level(y[te], probs, trial[te], subj[te]),
                **{k: v for k, v in info.items() if k != "history"}})
            if "history" in info:
                (out / f"{args.model}_{tgt}_f{fi}_history.json").write_text(json.dumps(info["history"]))

    summary = {"paper_doi": "10.1038/s41598-024-60977-9", "reconstruction_not_original": True,
               "args": vars(args), "feature_matrix_shape": list(X.shape), "results": {}}
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
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved", (out / "summary.json").resolve())


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir"); p.add_argument("--output-dir", default="runs/default")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--model", choices=sorted(NEURAL | SKLEARN), default="cfnn")
    p.add_argument("--target", choices=["both", "valence", "arousal"], default="both")
    p.add_argument("--split", choices=["random", "trial", "subject"], default="subject")
    p.add_argument("--cv-folds", type=int, default=0, help="0 = single hold-out")
    p.add_argument("--test-size", type=float, default=0.25)
    p.add_argument("--val-size", type=float, default=0.2)
    p.add_argument("--window-samples", type=int, default=512, help="paper sweep: 4..128; 512 = 4 s")
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
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-subjects", type=int, default=None)
    p.add_argument("--self-test", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())
```

### 10.1 `collect.py` (gather all runs into one CSV)

```python
#!/usr/bin/env python3
"""Collect runs/*/summary.json into one table (CSV to stdout)."""
import csv, json, sys
from pathlib import Path

rows = []
for f in sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "runs").glob("*/summary.json")):
    s = json.loads(f.read_text()); a = s["args"]
    for tgt, r in s["results"].items():
        ci = [x["trial"]["subject_bootstrap_ci95"] for x in r["folds"]]
        rows.append({"run": f.parent.name, "model": a["model"], "split": a["split"],
                     "cv_folds": a["cv_folds"], "window_samples": a["window_samples"],
                     "feature_dim": a["feature_dim"], "fuzzy_agg": a["fuzzy_agg"],
                     "threshold_op": a["threshold_op"], "seed": a["seed"], "target": tgt,
                     "window_acc": round(r["window_accuracy_mean"], 4),
                     "trial_acc": round(r["trial_accuracy_mean"], 4),
                     "trial_acc_std": round(r["trial_accuracy_std"], 4),
                     "majority": round(r["folds"][0]["window"]["majority_baseline"], 4),
                     "subject_ci95_fold0": ci[0]})
if rows:
    w = csv.DictWriter(sys.stdout, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
```

Usage: `python collect.py runs > results.csv`

---

## 11. Experiment plan and commands

```bash
DATA=data/deap/data_preprocessed_python
```

**E0. Model checks (no data needed)**

```bash
python train_cfnn.py --self-test --model cfnn --feature-dim 73    # expect 11186
python train_cfnn.py --self-test --model cfnn --feature-dim 70    # expect 11058
python train_cfnn.py --self-test --model plain_matched
```

**E1. Smoke test on 4 subjects (discard the numbers)**

```bash
python train_cfnn.py --data-dir $DATA --max-subjects 4 --split subject --model cfnn \
  --window-samples 512 --epochs 2 --target valence --output-dir runs/smoke
```

**E2. Paper-style arm (random window split, W = 32, 75:25)**

```bash
python train_cfnn.py --data-dir $DATA --split random --window-samples 32 \
  --model cfnn --target both --cache-dir cache --output-dir runs/cfnn_random_w32
```

**E3. Same features, stricter splits (the key comparison)**

```bash
for SP in trial subject; do
  python train_cfnn.py --data-dir $DATA --split $SP --window-samples 32 \
    --model cfnn --target both --cache-dir cache --output-dir runs/cfnn_${SP}_w32
done
```

**E4. Principled long-window arm (4 s)**

```bash
for SP in random trial subject; do
  python train_cfnn.py --data-dir $DATA --split $SP --window-samples 512 \
    --model cfnn --target both --cache-dir cache --output-dir runs/cfnn_${SP}_w512
done
```

**E5. Window sweep (paper's 4..128), random vs subject**

```bash
for W in 128 64 32 16 8 4; do        # start from the cheap end; W=4 needs ~1 GB RAM and long training
  for SP in random subject; do
    python train_cfnn.py --data-dir $DATA --split $SP --window-samples $W \
      --model cfnn --cache-dir cache --output-dir runs/cfnn_${SP}_w${W}
  done
done
```

**E6. Baselines and ablations on the same split** (shown for subject-grouped W = 32; repeat for `random`)

```bash
for M in cnn plain plain_matched cfnn_nodefuzz bilstm svm knn; do
  python train_cfnn.py --data-dir $DATA --split subject --window-samples 32 \
    --model $M --cache-dir cache --output-dir runs/${M}_subject_w32
done
```

**E7. Grouped cross-validation (variability across people)**

```bash
for M in cfnn plain_matched cnn; do
  python train_cfnn.py --data-dir $DATA --split subject --cv-folds 8 --window-samples 512 \
    --model $M --cache-dir cache --output-dir runs/${M}_cv8_subject_w512
done
```

**E8. Sensitivity to undocumented choices**

```bash
B="--data-dir $DATA --split subject --window-samples 512 --model cfnn --cache-dir cache"
python train_cfnn.py $B --feature-dim 70                     --output-dir runs/sens_dim70
python train_cfnn.py $B --pad-mode summary                   --output-dir runs/sens_padsummary
python train_cfnn.py $B --fuzzy-agg mean_mem                 --output-dir runs/sens_aggmeanmem
python train_cfnn.py $B --fuzzy-agg prod                     --output-dir runs/sens_aggprod
python train_cfnn.py $B --threshold-op gt                    --output-dir runs/sens_gt
python train_cfnn.py $B --learning-rate 0.001                --output-dir runs/sens_lr1e-3
python train_cfnn.py $B --early-stopping                     --output-dir runs/sens_earlystop
for S in 1 2 3 4; do python train_cfnn.py $B --seed $S       --output-dir runs/seed_$S; done
```

**E9. Collect**

```bash
python collect.py runs > results.csv
```

### Ablation map

| ID | Question | How |
|---|---|---|
| A0 | Does the feature extractor work? | `--model cnn` |
| A1 | Does the fuzzy head help? | `--model cfnn` vs `plain` |
| A2 | Is it the fuzzy math or just parameters? | `cfnn` vs `plain_matched` |
| A3 | Does defuzzification matter? | `cfnn` vs `cfnn_nodefuzz` |
| A4 | How much depends on my fuzzy math? | `--fuzzy-agg` variants |
| A5 | Does the undocumented 73rd-input bridge matter? | `--feature-dim 70`, `--pad-mode` |
| A6 | Window size | E5 |
| A7 | Leakage | `random` vs `trial` vs `subject` (E2-E4) |
| A8 | Label convention | `--threshold-op gt` |
| A9 | Seed variance | seed loop in E8 |

For fair comparisons, every run in a comparison must share the split, seed, features, scaler and epoch budget, and the model architecture must be fixed **before** looking at test results.

---

## 12. Reading the results

Fill this table from `results.csv`. **Do not copy the paper's numbers into it.**

| Run | Split | W | Model | Window acc | Trial acc | Majority baseline | Seed |
|---|---|---:|---|---:|---:|---:|---:|
| `cfnn_random_w32` | random | 32 | cfnn | *TBD* | *TBD* | *TBD* | 42 |
| `cfnn_trial_w32` | trial | 32 | cfnn | *TBD* | *TBD* | *TBD* | 42 |
| `cfnn_subject_w32` | subject | 32 | cfnn | *TBD* | *TBD* | *TBD* | 42 |

**Interpreting patterns**

| Pattern | Likely meaning | Next step |
|---|---|---|
| `random` high (near the paper), `trial` and `subject` much lower | The headline is driven by shared trials/subjects across train and test | Report the grouped numbers as the generalization estimate |
| `random` ≈ `trial` ≈ `subject`, all high | Leakage is not the explanation; unusual for DEAP | Audit labels, caching, duplicated windows; check seeds |
| `random` also far below 98% | Your features or fuzzy math differ from the authors' | Run E8 sensitivity; do not tune on the test split |
| Subject-grouped accuracy near the majority baseline | No cross-subject signal at these features | Expected to be common on DEAP; compare to `cnn`/`svm` |
| `cfnn` ≈ `plain_matched` | The fuzzy layers add no measurable benefit | Report this plainly |
| `cfnn` > `plain_matched` consistently across seeds and folds | Possible real effect | Repeat with more seeds; check `fuzzy-agg` sensitivity |
| Accuracy jumps toward 98% unexpectedly fast | Suspect leakage or a bug first | Re-check split mode, cache reuse, label alignment |

Accuracy is only meaningful next to the **majority baseline**; check class balance (`positive_fraction`). Differences smaller than the spread across seeds/folds are not findings.

---

## 13. Debugging

| Symptom | Likely cause | Fix |
|---|---|---|
| `Need >=4 subject files` | wrong `--data-dir`, or ZIP not extracted | point at the folder holding `s01.dat`.. |
| Shape error on load | not the preprocessed Python release | use the correct archive; don't reshape unknown arrays |
| Loss stuck near 0.693 / accuracy about 50% | lr 0.01 too high for the fuzzy layer, or label misalignment | try `--learning-rate 0.001` (record as deviation); verify `labels[:,0]` valence, `[:,1]` arousal; check class histogram |
| NaN loss | near-zero denominator or exploding gradients | `--clipnorm 1.0`; inspect scaled-feature min/max |
| Fuzzy outputs about 0 | `--fuzzy-agg prod` at d = 1152 | use `mean_exp` or `mean_mem` |
| Param count is not 11,186 | wrong input length or padding | `--feature-dim 73`; run `--self-test` |
| Parameter count 11,058 | you are in 70-feature mode | expected |
| Out of memory at small W | millions of rows | start at W >= 32; use `--cache-dir`; subset with `--max-subjects` for debugging only |
| SVM very slow | SVM scales poorly with rows | lower `--sk-max-train` |
| No FFT bin in a band | `--nfft` too small | leave `--nfft` at 0 |
| Accuracy near 98% instantly | leakage or a bug | confirm `--split`; compare against `subject` |
| GPU not used | TF cannot see a GPU (common on native Windows) | not required for this model; try WSL/Linux if you want one |

---

## 14. Reporting template

> *We developed an independent TensorFlow reconstruction of the CFNN proposed by Azar et al. (2024). It matches the paper's specified architecture and parameter count (11,186 for a 73-element input). Because the publication does not specify the feature transformation, the fuzzy-layer computation, the label threshold, the window unit or the train/test assignment, our implementation uses declared substitute choices (Section 3). We evaluate under the paper-style random window split and under trial-grouped and subject-grouped splits at identical features, and compare against SVM, KNN, 1D-CNN, Bi-LSTM and parameter-matched non-fuzzy heads.*

Always state: threshold and operator, window size and unit, `nfft`, 70 vs 73 and pad mode, fuzzy aggregation, split mode, seeds, library versions, and whether results are window- or trial-level. Use measured numbers only. Phrase concerns about the original as **reproducibility gaps and possible leakage that the paper's description cannot rule out**, not as established fault.

---

## 15. Checklist

**Data and features**
- [ ] Preprocessed Python DEAP, 32 files, shapes `(40,40,8064)` and `(40,4)` verified
- [ ] 1-based to 0-based channel translation checked
- [ ] Baseline dropped; threshold, operator and class balance recorded
- [ ] Window size/unit, `nfft`, taper, 70/73 and pad mode recorded
- [ ] No scaler or feature statistic uses validation/test rows

**Model**
- [ ] `--self-test` passes: 11,186 (73) and 11,058 (70); gradients finite
- [ ] Fuzzy aggregation recorded; sensitivity run done
- [ ] Original-vs-reconstructed choices listed

**Experiments**
- [ ] Paper-style `random` arm run (E2)
- [ ] `trial` and `subject` arms at identical features and seed (E3)
- [ ] Window sweep done where feasible (E5)
- [ ] SVM, KNN, CNN, Bi-LSTM, `plain`, `plain_matched`, `cfnn_nodefuzz` on identical splits (E6)
- [ ] Grouped CV for variability (E7); at least 3 seeds
- [ ] Majority baseline and balanced accuracy reported
- [ ] Hyperparameters chosen on validation only; test read once per run

**Interpretation**
- [ ] Paper-reported and measured numbers kept separate
- [ ] No claim of exact replication or of generalization from the random-split number
- [ ] Gaps explained, not closed by tuning on the test split
- [ ] Versions, seeds, split membership and logs preserved (`summary.json`)

---

## 16. References

- Nobari Azar, N. A., Cavus, N., Esmaili, P., Sekeroglu, B., Aşır, S. Detecting emotions through EEG signals based on modified convolutional fuzzy neural network. *Sci. Rep.* 14, 10371 (2024). <https://doi.org/10.1038/s41598-024-60977-9>
- Koelstra, S. et al. DEAP: A database for emotion analysis using physiological signals. *IEEE Trans. Affective Computing* 3(1), 18-31 (2012). <https://www.eecs.qmul.ac.uk/mmv/datasets/deap/>

**Provenance.** Everything labeled ASSUMED or DEVIATION is an implementation decision, not a statement attributed to the paper. Page references are to the 13-page PDF of the article.
