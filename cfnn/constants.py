"""DEAP acquisition settings and supported model names."""
from __future__ import annotations

import numpy as np


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
CACHE_VERSION = 1
