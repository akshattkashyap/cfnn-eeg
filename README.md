# CFNN EEG reproduction

Normal Python scripts for the supplied reproduction guide. Train in Colab after
uploading/extracting your ZIP. This is an independent reconstruction, with the
guide's assumptions; it is not the authors' implementation and does not promise
their reported accuracy.

## Files to zip

- `train_cfnn.py`: small command-line entry point.
- `repair_labels.py`: create separate dataset copies using participant-rating metadata.
- `diagnose_cfnn.py`: check saved-run consistency and tiny-batch memorization.
- `cfnn/`: include this entire package directory; it contains the implementation.
- `collect.py`: collect completed experiments into a CSV.
- `tests/test_pipeline.py`: preprocessing, split, metric and model checks.
- `requirements.txt` and this README.

Keep the DEAP data separate, or include it in your own private upload. The existing
folder is `data/data_preprocessed_python/s01.dat` through `s32.dat`. Only load
trusted DEAP pickle files. No notebook is needed in the ZIP.

## Code layout

```text
train_cfnn.py           # Entry point; existing commands still work
cfnn/
    __init__.py
    cli.py             # Command-line arguments and validation
    constants.py       # Channels, bands, sample rate, model names
    features.py        # Windowing and FFT band-power features
    data.py            # DEAP loading, caching and labels
    splits.py          # Holdout/CV splits and leakage audits
    models.py          # Fuzzy layers, model architectures and self-test
    training.py        # Neural training and sklearn baselines
    metrics.py         # Window/trial scoring and bootstrap intervals
    artifacts.py       # JSON writing and environment metadata
    experiment.py      # Connect pipeline stages and save run results
collect.py             # Combine completed runs into CSV
tests/test_pipeline.py
```

## Run in Colab

Select a GPU runtime, upload/extract your code ZIP, then change directory to the
folder containing `train_cfnn.py`. For example, these are Colab cell commands
(adjust the ZIP name and extracted folder):

```python
!unzip -q /content/cfnn-eeg.zip -d /content/project
%cd /content/project
%pip install -r requirements.txt
!python train_cfnn.py --self-test --feature-dim 73
!python train_cfnn.py --self-test --feature-dim 70
!python -m unittest discover -s tests -v
```

If your ZIP contains a top-level directory, `%cd` into that directory instead.
Restart the Colab runtime if pip asks you to after replacing installed libraries.
Use `--data-dir /your/path/data_preprocessed_python` if the data is elsewhere.
For faster reads, extract/copy data to `/content` rather than reading all subject
files from Drive on each run.

First check features and splits without training:

```bash
python train_cfnn.py --prepare-only --max-subjects 4 --cache-dir cache --output-dir runs/check
```

Then train the default CFNN: both targets independently, subject-held-out split,
512-sample (4 s) windows, 73 inputs, Adam 0.01, batch size 256, 100 epochs:

```bash
python train_cfnn.py --cache-dir cache --output-dir runs/cfnn_subject_w512
```

Prefix shell commands with `!` when running them in a Colab cell. Save run outputs
to mounted Drive if they must survive runtime disconnection, for example
`--output-dir /content/drive/MyDrive/cfnn_runs/subject_w512`. Mount Drive in Colab
before using that path. Each run requires a new or empty output directory to
prevent accidental checkpoint mixing. Interrupted runs retain checkpoints, CSV
logs and partial results; automatic training resume is not implemented.

## Experiments from the guide

Paper-style random windows, 32 samples, 75:25 outer split:

```bash
python train_cfnn.py --split random --window-samples 32 --cache-dir cache --output-dir runs/cfnn_random_w32
```

Compare the same features with whole trials or whole subjects held out:

```bash
python train_cfnn.py --split trial --window-samples 32 --cache-dir cache --output-dir runs/cfnn_trial_w32
python train_cfnn.py --split subject --window-samples 32 --cache-dir cache --output-dir runs/cfnn_subject_w32
```

Cross-validation and a baseline:

```bash
python train_cfnn.py --split subject --cv-folds 5 --cache-dir cache --output-dir runs/cfnn_subject_cv5
python train_cfnn.py --model plain_matched --cache-dir cache --output-dir runs/plain_matched_subject_w512
python collect.py runs --output results.csv
```

Available models: `cfnn`, `cfnn_nodefuzz`, `plain`, `plain_matched`, `cnn`,
`bilstm`, `svm`, `knn`. SVM and KNN work without TensorFlow (install the other
requirements). Their training rows are capped at 30,000 using stratified sampling;
change `--sk-max-train` to control cost. `--svm-c` and `--knn-neighbors` expose
baseline settings. Choose settings on validation data; no automatic search is
performed. SVM trial scores use the guide's sigmoid-transformed decision margin,
which is not a calibrated probability.

Other options include `--target valence`, `--feature-dim 70`, `--pad-mode summary`,
`--fuzzy-agg mean_mem`, `--threshold-op gt`, `--early-stopping`,
`--learning-rate 0.001`, and `--clipnorm 1`. See `python train_cfnn.py --help`.

## Protocol and output

- Select 14 EEG channels, drop 384 baseline samples, compute Hann-windowed FFT
  log band powers in five bands. Non-overlapping windows discard any remainder.
  Automatic FFT length is `max(window_samples, 128)`; padding adds no true
  frequency resolution. Defaults reproduce the guide, including three zero
  columns to extend 70 features to 73.
- Fit `StandardScaler` on training rows only. Validation is 20% of the outer
  training portion; a 75:25 outer split therefore gives about 60:15:25 overall.
  Groups stay separate in both splitting stages. Test results never choose a
  neural checkpoint; the minimum validation loss does.
- Random window splitting permits shared trials and subjects. Trial splitting
  holds out subject-trial pairs, but can share people and video IDs. Subject
  splitting holds out people. Compare these protocols separately.
- CFNN has 11,186 parameters at 73 inputs and 11,058 at 70. Gaussian aggregation
  and defuzzification follow the guide's reconstruction. `--self-test` checks
  counts, normalized outputs and finite gradients without fitting a model.

Each run saves configuration, library/GPU versions, feature/source metadata,
row metadata, exact split indices, overlap audits, fitted scalers, model files,
test predictions, training logs and `summary.json`. Neural outputs include best
`.weights.h5` and a `.keras` model; sklearn models use `.joblib`. Per-fold results
are also written to `partial_results.json` as they finish.

Reports include window and trial metrics, confusion matrices, majority baselines
and per-subject trial accuracy. The subject-bootstrap CI describes the unweighted
mean of subject accuracies (`subject_accuracy_mean`), which can differ from the
pooled trial accuracy when subjects have different test trial counts. The guide's
`majority_baseline` is the test set's majority fraction; the separate
`train_majority_test_accuracy` evaluates a class selected from training labels.

Import `cfnn.models` before loading a `.keras` model to register the custom layers:

```python
import joblib
import tensorflow as tf
import cfnn.models

model = tf.keras.models.load_model("runs/cfnn_subject_w512/cfnn_valence_f0.keras")
scaler = joblib.load("runs/cfnn_subject_w512/scaler_f0.joblib")
# features: (n_windows, 73), extracted with the saved run's feature settings
# scores = model.predict(scaler.transform(features)[..., None])
# scores[:, 0] = low; scores[:, 1] = high
```

Caches are keyed by feature settings, source paths, sizes and modification times.
Changing labels' binarization or splitting can reuse raw features safely. If you
change feature extraction code, increment `CACHE_VERSION` in `cfnn/constants.py`
or clear the cache.

The local data check found finite ratings below the guide's nominal 1–9 scale.
The loader reports this and preserves the supplied values; it does not clamp,
rescale or discard them. `preparation.json` records rating ranges and affected
trial counts. The configured threshold still applies directly to these ratings.

## Correct the supplied local labels before further experiments

Comparison with the supplied `data/metadata_xls/participant_ratings.xls` found
439 differing valence and 439 differing arousal ratings. On those trials the
`.dat` values equal 9 minus the spreadsheet values. The repair uses the spreadsheet
as an explicit replacement source; retain its provenance when reporting results.

From the project root:

```powershell
python -m pip install xlrd
python repair_labels.py
```

The script validates all 32 subjects against metadata keyed by participant ID
and experiment/video ID. Dominance and liking must match to confirm alignment.
It writes `data/corrected/data_preprocessed_python/`, preserves the source files,
and reopens every corrected file to verify labels and unchanged EEG values.
`label_repair.json` in that directory records all rating changes and file hashes.
An existing destination is refused. A partial directory without a completed audit
is not a verified repair; choose a new destination if an interrupted run needs repeating.

Train with the corrected directory explicitly (the training CLI's default still
refers to the original data). Use a fresh output folder for every run:

```powershell
python -u train_cfnn.py --data-dir data/corrected/data_preprocessed_python --cache-dir cache/corrected --output-dir "runs/corrected_$(Get-Date -Format yyyyMMdd_HHmmss)"
```

For the guide's paper-style experiment append `--split random --window-samples 32`.
Keep model settings and seed fixed when comparing the effect of the label repair.
Expected high-trial counts at threshold >=5 are 724 for valence and 754 for arousal.

To use the same corrected data in Colab, zip the corrected subject folder:

```powershell
Compress-Archive -Path data/corrected/data_preprocessed_python -DestinationPath deap_corrected.zip -Force
```

Upload that ZIP to Drive and extract it into `/content/deap_corrected`. Use
`--data-dir /content/deap_corrected/data_preprocessed_python` and a fresh cache/output
directory in the training command. Include `repair_labels.py` and `cfnn/labels.py`
when recreating the code ZIP. No training is performed by the repair script.

## Diagnose low accuracy before another full experiment

Use `diagnose_cfnn.py` with a completed corrected-label run. Replace `YOUR_RUN`
with its directory name. These are Colab terminal commands:

```bash
python -u /content/cfnn-eeg/diagnose_cfnn.py --run-dir /content/drive/MyDrive/cfnn_runs/YOUR_RUN
```

The diagnostic reuses that run's feature configuration, exact split, scaler,
label threshold and seed. It verifies row metadata and training feature statistics
before proceeding. Optional `--data-dir` and `--cache-dir` override paths if the
data/cache have moved. Use trusted run artifacts; joblib and pickle load Python objects.

By default it selects 32 distinct trials from the training partition, using one
window per trial, balanced 16 low / 16 high for valence. The same batch is reused
for CNN and CFNN at learning rates 0.01 and 0.001, with 1,000 updates each. These
are diagnostic updates on 32 examples, not 1,000 full-data epochs. No validation
or test predictions are computed and no trained diagnostic model is exported.

`--check-only` verifies artifacts and feature health without fitting models and
works without TensorFlow. `--target both` adds arousal; `--models` and
`--learning-rates` narrow the comparison if desired.

The report and the selected batch are saved in a new `diagnostics_TIMESTAMP`
subdirectory of the reference run. `diagnostics.json` is updated after each model
finishes. The original run's checkpoints and scores are preserved.

Interpret these as debugging clues, not a guaranteed pass/fail threshold:

- Near-perfect tiny-batch fit indicates basic optimization can memorize these
  examples. It does not demonstrate useful generalization or match the paper.
- Failure in both CNN and CFNN motivates inspecting features, duplicates,
  gradients and optimizer settings before another full training run.
- CNN fitting successfully while CFNN struggles points toward the reconstructed
  fuzzy head or its optimization as a place to investigate.
- High training-mode accuracy with substantially lower inference-mode accuracy
  motivates checking batch-normalization moving statistics. The diagnostic restores
  non-trainable state after each comparison so checks do not recalibrate BN.
- Differences between the two learning rates are a diagnostic observation, not
  permission to select settings based on test results. Subsequent full-data choices
  should be made using validation data only.
