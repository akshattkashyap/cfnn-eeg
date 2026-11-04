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
    data.py            # Dataset dispatch, DEAP loading, caching and labels
    dreamer.py         # Original DREAMER.mat loading with variable-length trials
    splits.py          # Holdout/CV splits and leakage audits
    models.py          # Fuzzy layers, model architectures and self-test
    tsk.py             # Standalone first-order HTSK and train-only rule initialization
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
`bilstm`, `htsk`, `svm`, `knn`. SVM and KNN work without TensorFlow (install the other
requirements). Their training rows are capped at 30,000 using stratified sampling;
change `--sk-max-train` to control cost. `--svm-c` and `--knn-neighbors` expose
baseline settings. Choose settings on validation data; no automatic search is
performed. SVM trial scores use the guide's sigmoid-transformed decision margin,
which is not a calibrated probability.

Other options include `--target valence`, `--feature-dim 70`, `--pad-mode summary`,
`--fuzzy-agg mean_mem`, `--threshold-op gt`, `--early-stopping`,
`--learning-rate 0.001`, and `--clipnorm 1`. See `python train_cfnn.py --help`.

## DEAP protocol and output

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

## Standalone HTSK baseline

`--model htsk` adds an independent TensorFlow implementation of a first-order
high-dimensional Takagi-Sugeno-Kang classifier. It takes the scaled band-power
features directly. Its Gaussian rule weights use the mean over feature dimensions
in log space, then a softmax over rules. Each rule has a separate linear two-class
consequent, and the weighted consequent logits feed the class softmax:

```text
z[b,r,d] = 0.5 * ((x[b,d] - center[r,d]) / sigma[r,d])**2
weight[b,r] = softmax_over_rules(-mean_over_features(z[b,r,:]))
logit[b,c] = sum_r weight[b,r] * (sum_d kernel[r,d,c]*x[b,d] + bias[r,c])
```

This follows the HTSK aggregation and first-order TSK equations in
[Cui, Wu and Xu (2021)](https://arxiv.org/abs/2102.04271) and the
[PyTSK reference](https://pytskdocs.readthedocs.io/en/latest/models.html).
It is a new comparison model, not the architecture from the original CFNN paper.
We use positive softplus-parameterized widths, initial width 1 in standardized
feature units, and deterministic sklearn KMeans with five initializations on at
most 30,000 scaled training rows. Those implementation choices are explicit;
we do not claim to reproduce all experimental details of the HTSK paper.
The baseline does not add BN, DropRule, or uniform regularization.

Defaults specific to HTSK are 70 features and 16 rules. All other training CLI
defaults stay the same, including learning rate 0.01, 100 epochs and seed 42.
Set `--feature-dim 73` explicitly only for a deliberate padded-input comparison.
For R rules and D features, there are R*(4*D+2) learned parameters: **4,512** for
the default HTSK. Inputs still use `(batch, D, 1)` for API consistency; the model
flattens this singleton dimension and has no convolutional layers.

Check the model without training:

```bash
python train_cfnn.py --model htsk --self-test
python -m unittest discover -s tests -p test_tsk.py -v
```

Colab terminal command for the corrected-label random-window comparison:

```bash
python -u /content/cfnn-eeg/train_cfnn.py --model htsk --data-dir /content/deap_corrected/data_preprocessed_python --split random --window-samples 32 --feature-dim 70 --htsk-rules 16 --seed 42 --cache-dir /content/cfnn_cache_corrected --output-dir "/content/drive/MyDrive/cfnn_runs/htsk_w32_$(date +%s)"
```

Leave `--stride` at zero (non-overlapping) for this comparison. The 70-feature
cache is separate from the previous 73-feature cache. Its first 70 feature columns
match the previous zero-padded inputs and row identities/split membership are
unchanged. Compare validation results under the fixed protocol; a different
architecture is not evidence that the earlier paper has been reproduced.

K-means is called inside training with `xtr` only. Neither validation nor test
features or labels enter rule-center initialization. Each target/fold initializes
a fresh model. The selected training positions and initial centers are saved in
`htsk_TARGET_fN_initialization.npz`; positions index the corresponding saved
`split_fN.npz` train array. Summary reports record initialization sample counts
and the HTSK paper reference. Standard validation checkpointing and model/scaler
saving work as for the other models.

Options: `--htsk-rules`, `--htsk-init-max-samples`, and `--htsk-init-sigma`.
Diagnostics support `--models htsk` and an optional `--htsk-rules` override; the
rule count cannot exceed the tiny batch's number of distinct feature rows.
Import `cfnn.models` before loading an HTSK `.keras` model to register its layer.

## DREAMER as a separate dataset

Use `--dataset dreamer` to train the same models on the original `DREAMER.mat`
file. `--data-dir` accepts either that file or its containing directory. Obtain
the dataset from the [authors' Zenodo record](https://zenodo.org/records/546113),
following its access instructions. Keep the data separate from your code ZIP.
No new Python dependency is needed; the loader uses SciPy, already required.
Converted MATLAB v7.3/HDF5 files and third-party preprocessed layouts are not
supported by this adapter.

The adapter reads `DREAMER.Data`, each subject's `EEG.stimuli`, `ScoreValence`
and `ScoreArousal`. It uses all 14 native EEG channels at 128 Hz and preserves
variable trial durations. Channel order is AF3, F7, F3, FC5, T7, P7, O1, O2, P8,
T8, FC6, F4, F8, AF4; electrode metadata, when present, is used to validate and
reorder channels into this order. The MATLAB structure follows the public
[TorchEEG reader](https://torcheeg.readthedocs.io/en/v1.1.1/_modules/torcheeg/datasets/module/emotion_recognition/dreamer.html);
TorchEEG itself is not installed or needed.

DREAMER-specific choices are explicit:

- Ratings are 1..5. The default threshold is **3**, with **high >= 3** and
  **low < 3**, consistent with this project's `ge` convention. Rating 3 therefore
  belongs to high. Use `--threshold-op gt` for high > 3 / low <= 3 and document
  that different convention. Labels are read directly; the DEAP label-repair
  script does not apply to DREAMER.
- Stimulus and baseline recordings are separate. Only `EEG.stimuli` is used;
  the DEAP-specific 384-sample baseline removal is not applied. This adapter
  performs no additional artifact removal, baseline subtraction or filtering.
  It computes the same five log band-power features per channel as the DEAP
  pipeline, giving 70 inputs (or 73 with the existing padding option).
- All stimulus samples are used by default. `--dreamer-last-seconds 60`
  explicitly selects the last 60 seconds of each trial. A requested crop or
  window that does not fit a trial causes an error. Incomplete trailing windows
  are discarded; windows never cross trial boundaries. With full recordings,
  longer trials contribute more windows to training and window-level metrics.
- Subject IDs and subject-trial IDs are kept in the saved row metadata. Trial
  and subject splits retain their existing grouping guarantees. `random` still
  permits windows from the same trial on both sides of a split. Dataset, subject
  limit and stimulus crop are included in cache identity. Old DEAP commands keep
  their defaults, and old DEAP run configurations remain readable by diagnostics.

After uploading the updated code ZIP, put `DREAMER.mat` at
`/content/dreamer/DREAMER.mat`. First check data and splits in a Colab cell:

```python
!python /content/cfnn-eeg/train_cfnn.py --dataset dreamer --model htsk --data-dir /content/dreamer/DREAMER.mat --window-samples 128 --max-subjects 4 --prepare-only --cache-dir /content/cfnn_cache_dreamer --output-dir /content/dreamer_check
```

The check output directory must be new or empty. Then, in the Colab terminal,
train both targets using whole-trial holdout, 128-sample windows, 16 rules and
seed 42 (100 epochs, Adam 0.01, batch size 256):

```bash
python -u /content/cfnn-eeg/train_cfnn.py --dataset dreamer --model htsk --data-dir /content/dreamer/DREAMER.mat --split trial --window-samples 128 --htsk-rules 16 --seed 42 --cache-dir /content/cfnn_cache_dreamer --output-dir "/content/drive/MyDrive/cfnn_runs/dreamer_htsk_w128_$(date +%s)"
```

Use `--split random` specifically for a window-split experiment corresponding
to your earlier DEAP runs, or `--split subject` for held-out participants. Report
these protocols separately. This integration trains fresh DREAMER models; it
does not pool datasets or evaluate a DEAP-trained checkpoint on DREAMER.
The montages and acquisition conditions differ, so a DEAP model cannot be
treated as a DREAMER model just because both produce 70 features. DREAMER
experiments provide another evaluation dataset, not a promise of higher accuracy.

`collect.py` includes a dataset column (old runs default to `deap`), and
`diagnose_cfnn.py` reloads the correct dataset from each completed run's config.
Loader checks use synthetic MATLAB fixtures, including variable durations,
label alignment, channel order, cropping, caches and artifact round-trips; the
real DREAMER file is not included in this repository.
