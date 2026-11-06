"""Render HTSK diagrams for DEAP and the separate DREAMER configuration.

Run with a Python environment containing matplotlib. No data/model is loaded.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


OUT = Path(__file__).resolve().parent / "diagrams"
COLORS = {
    "ink": "#16324f", "muted": "#52657a", "arrow": "#6b7f93",
    "blue": ("#edf5ff", "#3978b8"), "purple": ("#f3edfc", "#8452b5"),
    "teal": ("#e9f7f3", "#2c8b76"), "orange": ("#fff3e4", "#bc7a2c"),
    "gray": ("#f3f6f9", "#8493a3"),
}


def canvas(height, title, subtitle, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=.025, right=.975, top=.985, bottom=.025)
    ax.set_xlim(0, 100)
    ax.set_ylim(-3, height)
    ax.axis("off")
    ax.text(50, height - 1, title, ha="center", va="top", fontsize=23,
            fontweight="bold", color=COLORS["ink"])
    ax.text(50, height - 5, subtitle, ha="center", va="top", fontsize=12,
            color=COLORS["muted"])
    return fig, ax


def box(ax, x, y, w, h, title, lines=(), color="blue", size=12, title_size=14):
    fill, edge = COLORS[color]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.2,rounding_size=.7",
                               linewidth=1.6, edgecolor=edge, facecolor=fill, zorder=2))
    gap = min(2.45, (h - 1.4) / (len(lines) + 1))
    top = y + h / 2 + gap * len(lines) / 2
    ax.text(x + w / 2, top, title, ha="center", va="center", color=COLORS["ink"],
            fontsize=title_size, fontweight="bold", zorder=3)
    for i, line in enumerate(lines, start=1):
        ax.text(x + w / 2, top - gap * i, line, ha="center", va="center",
                color=COLORS["ink"], fontsize=size, zorder=3)


def arrow(ax, points, color=None, dashed=False):
    color = color or COLORS["arrow"]
    style = "--" if dashed else "-"
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot([start[0], end[0]], [start[1], end[1]], color=color, linewidth=1.7,
                linestyle=style, zorder=1)
    ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                 mutation_scale=14, linewidth=1.7, color=color,
                                 linestyle=style, zorder=1))


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for extension in ("svg", "png"):
        target = OUT / f"{name}.{extension}"
        fig.savefig(target, dpi=180, facecolor="white")
        print(target)
    fig.savefig(OUT / f"{name}_preview.png", dpi=85, facecolor="white")
    plt.close(fig)


def pipeline():
    fig, ax = canvas(131, "DEAP → HTSK: data, training and predictions",
                     "Last completed configuration: 128-sample windows · 32 rules · seed 42",
                     (16, 18))
    box(ax, 5, 111, 90, 9, "Corrected DEAP dataset", [
        "32 subjects × 40 trials = 1,280 trials; sampling rate 128 Hz",
        "Each recorded trial: 40 channels × 8,064 samples; labels are trial ratings"], size=12)
    box(ax, 5, 90, 59, 11, "EEG selection and baseline exclusion", [
        "Keep 14 configured EEG channels; exclude first 384 samples (3 s)",
        "Stimulus EEG per trial: [14, 7,680] = 60 seconds"], size=11.5)
    box(ax, 69, 90, 26, 11, "Trial labels", [
        "Valence / arousal separately",
        "Rating ≥5 → high (1)",
        "Rating <5 → low (0)"], color="orange", size=11, title_size=13)
    arrow(ax, [(34.5, 111), (34.5, 101.3)])
    arrow(ax, [(82, 111), (82, 101.3)])
    box(ax, 5, 77, 59, 9, "Non-overlapping 1-second windows", [
        "Window = stride = 128 samples; 60 windows per trial",
        "76,800 windows in total; each EEG window is [14, 128]"], size=11.5)
    box(ax, 69, 77, 26, 9, "Inherit trial ratings", [
        "Every window keeps its",
        "subject ID and trial ID"], color="orange", size=11, title_size=13)
    arrow(ax, [(34.5, 90), (34.5, 86.3)])
    arrow(ax, [(82, 90), (82, 86.3)])
    box(ax, 5, 62, 59, 11, "Fixed spectral feature extraction", [
        "Hann taper → 128-point real FFT → normalized squared magnitude",
        "Mean power in theta, alpha, beta-low, beta-high, gamma; then log10",
        "14 channels × 5 bands = 70 features; feature matrix [76,800, 70]"], size=10.7)
    arrow(ax, [(34.5, 77), (34.5, 73.3)])
    box(ax, 5, 48, 90, 10, "Random window split: features and labels use identical indices", [
        "Training: 46,080 (60%)    Validation: 11,520 (15%)    Test: 19,200 (25%)",
        "Rows are disjoint; windows from the same trial can occur in all three partitions"],
        color="gray", size=12)
    arrow(ax, [(34.5, 62), (34.5, 58.3)])
    arrow(ax, [(82, 77), (82, 58.3)])
    box(ax, 5, 36, 90, 8, "Feature standardization", [
        "Fit StandardScaler on training rows only; transform training, validation and test",
        "One scaler per fold, shared by the two target-specific classifiers"], size=11.5)
    arrow(ax, [(50, 48), (50, 44.3)])
    box(ax, 5, 20, 43, 11, "Valence HTSK classifier", [
        "70 inputs → 32 fuzzy rules → 2 output logits",
        "9,024 trainable parameters",
        "Separate learned centers, widths and consequents"], color="purple", size=11)
    box(ax, 52, 20, 43, 11, "Arousal HTSK classifier", [
        "70 inputs → 32 fuzzy rules → 2 output logits",
        "9,024 trainable parameters",
        "Separate learned centers, widths and consequents"], color="teal", size=11)
    arrow(ax, [(50, 36), (50, 33.5), (26.5, 33.5), (26.5, 31.3)])
    arrow(ax, [(50, 36), (50, 33.5), (73.5, 33.5), (73.5, 31.3)])
    box(ax, 5, 8, 43, 8, "Class softmax → low / high valence", [
        "Two probabilities per window; class = argmax"], color="orange", size=11, title_size=12.5)
    box(ax, 52, 8, 43, 8, "Class softmax → low / high arousal", [
        "Two probabilities per window; class = argmax"], color="orange", size=11, title_size=12.5)
    arrow(ax, [(26.5, 20), (26.5, 16.3)])
    arrow(ax, [(73.5, 20), (73.5, 16.3)])
    ax.text(50, 4.8, "Training: Adam 0.01 · batch 256 · 100 epochs · sparse categorical cross-entropy",
            ha="center", va="center", fontsize=11.5, color=COLORS["ink"])
    ax.text(50, 2.5, "Keep each target's lowest-validation-loss checkpoint; test after training. No weights are shared between targets.",
            ha="center", va="center", fontsize=10.8, color=COLORS["muted"])
    ax.text(50, .1, "Trial report: average held-out window probabilities within each trial, then argmax. This random split does not hold out whole trials.",
            ha="center", va="center", fontsize=10.2, color=COLORS["muted"])
    save(fig, "htsk_deap_pipeline")


def dataset_comparison():
    fig, ax = canvas(146, "DEAP and DREAMER → HTSK architecture",
                     "Two separate dataset pipelines · same model design · independently learned scalers and weights",
                     (20, 22))
    box(ax, 5, 123, 43, 12, "DEAP: completed experiment", [
        "32 subjects × 40 trials = 1,280 trials; 128 Hz",
        "14 configured channels selected from the recorded data",
        "Trial ratings: valence / arousal on a 1–9 scale"], size=12)
    box(ax, 52, 123, 43, 12, "DREAMER: implemented, example configuration", [
        "23 subjects × 18 trials = 414 trials; 128 Hz",
        "Original DREAMER.mat; 14 native EEG channels",
        "Trial ratings: valence / arousal on a 1–5 scale"], color="teal", size=12, title_size=13)
    box(ax, 5, 107, 43, 12, "DEAP stimulus selection", [
        "Exclude the first 384 baseline samples (3 seconds)",
        "Keep 7,680 stimulus samples (60 seconds) per trial",
        "Corrected ratings are read from the supplied DEAP files"], size=12)
    box(ax, 52, 107, 43, 12, "DREAMER stimulus selection", [
        "Read EEG.stimuli; baseline is stored separately",
        "All stimulus samples by default; trial durations vary",
        "Optional last-N-seconds crop; no DEAP baseline exclusion"], color="teal", size=12)
    box(ax, 5, 92, 43, 11, "DEAP windows and labels", [
        "128 samples = 1 second; stride 128, without overlap",
        "60 windows per trial; 76,800 windows in total",
        "Rating ≥5 → high (1); rating <5 → low (0)"], color="orange", size=12)
    box(ax, 52, 92, 43, 11, "DREAMER windows and labels", [
        "128 samples = 1 second; stride 128, without overlap",
        "Windows per trial = floor(stimulus samples / 128)",
        "Rating ≥3 → high (1); rating <3 → low (0)"], color="orange", size=12)
    for x in (26.5, 73.5):
        arrow(ax, [(x, 123), (x, 119.3)])
        arrow(ax, [(x, 107), (x, 103.3)])
    for x in (5, 52):
        box(ax, x, 77, 43, 11, "Same fixed spectral feature extraction", [
            "Hann taper → real FFT → five mean band powers → log10",
            "Theta / alpha / low beta / high beta / gamma",
            "14 channels × 5 bands = 70 features per window"], size=11.5, title_size=13)
    box(ax, 5, 62, 43, 11, "DEAP split used in the completed run", [
        "Random windows: train 46,080 / val 11,520 / test 19,200",
        "Windows of one trial can occur in multiple partitions",
        "60% / 15% / 25% of all windows"], color="gray", size=11.5, title_size=13)
    box(ax, 52, 62, 43, 11, "DREAMER split in the proposed command", [
        "Whole-trial holdout: no trial crosses partitions",
        "25% of trials for test; 20% of the rest for validation",
        "Window fractions depend on the variable trial durations"], color="gray", size=11.5, title_size=13)
    for x in (5, 52):
        box(ax, x, 49, 43, 9, "Train-only feature standardization", [
            "Fit StandardScaler on this dataset's training rows",
            "Use its own saved scaler for validation, test and inference"], size=11.5, title_size=13)
    for x in (26.5, 73.5):
        arrow(ax, [(x, 92), (x, 88.3)])
        arrow(ax, [(x, 77), (x, 73.3)])
        arrow(ax, [(x, 62), (x, 58.3)])
    box(ax, 5, 32, 43, 13, "DEAP HTSK: 32 rules", [
        "Input [B,70,1] → Flatten → HTSK → class softmax",
        "2,240 centers + 2,240 raw widths",
        "4,480 consequent weights + 64 biases",
        "9,024 trainable parameters per target classifier"], color="purple", size=12)
    box(ax, 52, 32, 43, 13, "DREAMER HTSK: 16 rules (configurable)", [
        "Input [B,70,1] → Flatten → HTSK → class softmax",
        "1,120 centers + 1,120 raw widths",
        "2,240 consequent weights + 32 biases",
        "4,512 trainable parameters per target classifier"], color="teal", size=12, title_size=13)
    for x in (5, 52):
        box(ax, x, 16, 43, 12, "Two independent target models", [
            "Valence → P(low), P(high)",
            "Arousal → P(low), P(high)",
            "Each target has its own learned fuzzy rules and consequents"], color="orange", size=12)
    for x in (26.5, 73.5):
        arrow(ax, [(x, 49), (x, 45.3)])
        arrow(ax, [(x, 32), (x, 28.3)])
    ax.text(50, 11.4, "Shared design: Gaussian rule matching → mean negative distance → rule softmax; parallel linear consequents → weighted logits → class softmax",
            ha="center", va="center", fontsize=11.5, color=COLORS["ink"])
    ax.text(50, 8.3, "Training settings shown: Adam 0.01 · batch 256 · 100 epochs · seed 42 · train-only K-means initialization · lowest-validation-loss checkpoint",
            ha="center", va="center", fontsize=11.5, color=COLORS["ink"])
    ax.text(50, 5.2, "The two lanes are separate experiments. Feature vectors are not pooled, and trained weights are not shared across datasets.",
            ha="center", va="center", fontsize=11.5, color=COLORS["muted"])
    ax.text(50, 2.1, "DREAMER training results are pending. Setting --htsk-rules 32 gives 9,024 parameters per target on either dataset.",
            ha="center", va="center", fontsize=11.5, color=COLORS["muted"])
    save(fig, "htsk_deap_dreamer_pipeline")


def unified_architecture():
    """One full schematic, with measured results separate from an arithmetic scenario."""
    observed = {
        "valence": {"confusion_matrix": [[4410, 4029], [3014, 7747]],
                    "balanced_accuracy_percent": 62.12, "best_epoch": 40},
        "arousal": {"confusion_matrix": [[4368, 3525], [2823, 8484]],
                   "balanced_accuracy_percent": 65.19, "best_epoch": 29},
    }
    for result in observed.values():
        cm = result["confusion_matrix"]
        result["window_accuracy_percent"] = 100 * (cm[0][0] + cm[1][1]) / sum(map(sum, cm))
        result["illustrative_plus_8pp_percent"] = result["window_accuracy_percent"] + 8
    v, a = observed["valence"], observed["arousal"]
    fig, ax = canvas(250, "Unified HTSK architecture: DEAP and DREAMER",
                     "Dataset-specific input adapters → one reusable model design · run each dataset and target independently",
                     (24, 36))
    box(ax, 4, 225, 43, 12, "DEAP input: completed 32-rule experiment", [
        "32 subjects × 40 trials; 128 Hz; corrected DEAP .dat files",
        "Select 14 configured EEG channels; exclude first 384 baseline samples",
        "Keep the 7,680-sample / 60-second stimulus period"], size=12.5, title_size=14)
    box(ax, 53, 225, 43, 12, "DREAMER input: implemented 16-rule example", [
        "23 subjects × 18 trials; 128 Hz; original DREAMER.mat",
        "Use all 14 native channels from EEG.stimuli; baseline is separate",
        "Variable stimulus durations; optional explicit last-N-seconds crop"],
        color="teal", size=12.5, title_size=14)
    box(ax, 4, 209, 43, 12, "DEAP window and label adapter", [
        "128-sample window / stride = 1 second, no overlap",
        "60 windows per trial → 76,800 windows",
        "Rating ≥5: high (1); rating <5: low (0)",
        "Retain each window's subject and trial identities"], color="orange", size=12.5)
    box(ax, 53, 209, 43, 12, "DREAMER window and label adapter", [
        "128-sample window / stride = 1 second, no overlap",
        "Per-trial windows = floor(stimulus samples / 128)",
        "Rating ≥3: high (1); rating <3: low (0)",
        "Retain each window's subject and trial identities"], color="orange", size=12.5)
    arrow(ax, [(25.5, 225), (25.5, 221.3)])
    arrow(ax, [(74.5, 225), (74.5, 221.3)])
    box(ax, 18, 195, 64, 10, "Apply the common pipeline to the selected dataset", [
        "One dataset per run; the two sources are not concatenated",
        "EEG window shape [B, 14, 128]; labels and metadata accompany every row"],
        color="gray", size=13)
    arrow(ax, [(25.5, 209), (25.5, 207), (35, 207), (35, 205.3)])
    arrow(ax, [(74.5, 209), (74.5, 207), (65, 207), (65, 205.3)])
    box(ax, 4, 180, 92, 11, "Fixed feature extraction: 14 channels × 5 bands = 70 inputs", [
        "Hann taper → 128-point real FFT [B,14,65] → squared magnitude / taper energy → band means → log10",
        "Theta [4,8) · alpha [8,12) · low beta [12,16) · high beta [16,25) · gamma [25,45] Hz",
        "Output X [N,70]; labels and subject/trial IDs are carried unchanged"], size=12.5)
    arrow(ax, [(50, 195), (50, 191.3)])
    box(ax, 4, 164, 92, 12, "Dataset-specific partitioning and saved row indices", [
        "DEAP completed run: random windows → training 46,080 / validation 11,520 / test 19,200",
        "DREAMER proposed protocol: whole-trial holdout → 25% trials for test; 20% of remaining trials for validation",
        "The DEAP run can share parent trials across partitions; DREAMER window fractions vary with trial duration"],
        color="gray", size=12.5)
    arrow(ax, [(50, 180), (50, 176.3)])
    box(ax, 4, 149, 92, 11, "Train-only standardization and initialization", [
        "Fit StandardScaler on training features only; transform training / validation / test with the saved scaler",
        "Initialize rule centers with K-means on up to 30,000 standardized training rows; initial Gaussian width = 1",
        "Each dataset has its own scaler; each target has separate trainable rule and consequent parameters"], size=12.5)
    arrow(ax, [(50, 164), (50, 160.3)])
    box(ax, 23, 135, 54, 10, "Keras model input → flatten", [
        "Input [B,70,1] → x [B,70]; every branch receives all 70 features",
        "D = 70; C = 2; R = 32 for DEAP / 16 for the DREAMER example"], size=12.5)
    arrow(ax, [(50, 149), (50, 145.3)])
    box(ax, 4, 113, 43, 16, "Fuzzy antecedents: Gaussian memberships", [
        "Centers c [R,70]; raw widths s [R,70]",
        r"Positive widths: $\sigma = \mathrm{softplus}(s) + 10^{-4}$",
        "z[b,r,d] = 0.5 × ((x[b,d] - c[r,d]) / sigma[r,d])²",
        "Distance tensor [B,R,70]",
        "Centers + raw widths: DEAP 4,480 / DREAMER 2,240 parameters"],
        color="purple", size=12.5, title_size=14)
    box(ax, 53, 113, 43, 16, "Linear consequents: one predictor per rule", [
        "Weights W [R,70,2]; biases b [R,2]",
        "g[b,r,c] = dot(x[b,:], W[r,:,c]) + b[r,c]",
        "Every rule produces a low and a high class score",
        "Rule logits g [B,R,2]; no per-rule class softmax",
        "Weights + biases: DEAP 4,544 / DREAMER 2,272 parameters"],
        color="teal", size=12.5, title_size=14)
    arrow(ax, [(50, 135), (50, 132), (25.5, 132), (25.5, 129.3)])
    arrow(ax, [(50, 135), (50, 132), (74.5, 132), (74.5, 129.3)])
    box(ax, 4, 99, 43, 10, "HTSK score: mean over 70 features", [
        "q[b,r] = -mean(z[b,r,:]) → [B,R]",
        "A higher score means a closer match to that rule's pattern",
        "Compute directly in log space"], color="purple", size=12.5)
    arrow(ax, [(25.5, 113), (25.5, 109.3)])
    box(ax, 4, 85, 43, 10, "Rule softmax: normalize across R rules", [
        "alpha = softmax(q, axis=rules) → [B,R]",
        "Rule strengths sum to 1; several rules can match a window",
        "Membership counts: DEAP 2,240 / DREAMER 1,120"], color="purple", size=12.5)
    arrow(ax, [(25.5, 99), (25.5, 95.3)])
    box(ax, 53, 85, 43, 20, "Supervised training of both branches", [
        "Training labels → sparse categorical cross-entropy",
        "Backpropagation + Adam update c, s, W and b",
        "Learning rate 0.01; batch size 256; 100 epochs; seed 42",
        "Select each target's lowest-validation-loss checkpoint",
        "Validation / test rows do not update model parameters",
        "DEAP selected epochs: valence 40; arousal 29"], color="gray", size=12.5)
    box(ax, 23, 69, 54, 11, "Combine the weighted rule logits", [
        "logits[b,c] = sum_r(alpha[b,r] × g[b,r,c]) → [B,2]",
        "A nonlinear mixture of R input-dependent linear predictors"], size=13)
    arrow(ax, [(25.5, 85), (25.5, 82), (39, 82), (39, 80.3)])
    arrow(ax, [(96, 121), (98, 121), (98, 82), (65, 82), (65, 80.3)],
          color=COLORS["teal"][1])
    box(ax, 23, 55, 54, 10, "Class softmax: normalize across 2 classes", [
        "p = softmax(logits, axis=classes) → [P(low), P(high)]",
        "argmax gives the window's predicted class"], color="orange", size=13)
    arrow(ax, [(50, 69), (50, 65.3)])
    box(ax, 4, 40, 92, 11, "Run the model separately for valence and arousal on each dataset", [
        "DEAP: 32 rules → 9,024 trainable parameters per target; DREAMER example: 16 rules → 4,512 per target",
        "Export each best model + its dataset/fold scaler; score held-out windows after training",
        "Trial reporting averages held-out window probabilities within each trial, then applies argmax"],
        color="gray", size=12.5)
    arrow(ax, [(50, 55), (50, 51.3)])
    box(ax, 4, 14, 57, 21, "DEAP test-window accuracy: measured and +8pp illustration", [
        f"Valence measured: {v['window_accuracy_percent']:.2f}%   →   measured +8pp: {v['illustrative_plus_8pp_percent']:.2f}%",
        f"Arousal measured: {a['window_accuracy_percent']:.2f}%   →   measured +8pp: {a['illustrative_plus_8pp_percent']:.2f}%",
        "The +8pp values are arithmetic illustrations, not measured training results",
        "Measured balanced accuracy remains: valence 62.12%; arousal 65.19%",
        "Source: supplied 32-rule DEAP confusion matrices; random window split"],
        color="orange", size=13, title_size=13)
    box(ax, 65, 14, 31, 21, "DREAMER accuracy: pending", [
        "Valence: no measured result supplied",
        "Arousal: no measured result supplied",
        "No numeric +8pp figure can be calculated",
        "until a measured baseline is available",
        "The example is architecture support, not an accuracy claim"],
        color="teal", size=11.5, title_size=13)
    arrow(ax, [(32.5, 40), (32.5, 35.3)])
    arrow(ax, [(80.5, 40), (80.5, 35.3)])
    ax.text(50, 9.5, "+8pp means adding 8 percentage points (A + 8), rather than multiplying A by 1.08. The adjusted figures are explicitly hypothetical.",
            ha="center", va="center", fontsize=12.5, color=COLORS["ink"])
    ax.text(50, 6.2, "The unified drawing describes a reusable architecture. Datasets, splits, fitted scalers and trained weights remain separate; no joint model was evaluated.",
            ha="center", va="center", fontsize=12, color=COLORS["muted"])
    ax.text(50, 2.9, "Measured DEAP results are for random window holdout; they do not establish unseen-trial or unseen-subject performance.",
            ha="center", va="center", fontsize=12, color=COLORS["muted"])
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = {
        "measured_source": "User-supplied completed DEAP HTSK 32-rule W128 run; confusion matrices",
        "dataset": "deap", "split": "random", "rules": 32, "window_samples": 128,
        "measured_results": {
            target: {key: value for key, value in result.items()
                     if key != "illustrative_plus_8pp_percent"}
            for target, result in observed.items()
        },
        "hypothetical_plus_8pp_results_percent": {
            target: result["illustrative_plus_8pp_percent"]
            for target, result in observed.items()
        },
        "adjustment": {"percentage_points": 8, "status": "hypothetical arithmetic illustration",
                       "measured_improvement": False, "training_performed_for_this_diagram": False},
        "dreamer": {"rules_in_example": 16, "measured_results": None, "status": "pending"},
    }
    (OUT / "htsk_unified_accuracy_provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    save(fig, "htsk_unified_complete_architecture")


def internals(n_rules=32, dataset="DEAP"):
    centers, weights, biases = n_rules * 70, n_rules * 70 * 2, n_rules * 2
    total = 2 * centers + weights + biases
    status = "completed run" if dataset == "DEAP" else "example configuration; training results pending"
    fig, ax = canvas(119, f"Inside one HTSK classifier: {dataset}",
                     f"{status} · D = 70 features · R = {n_rules} rules · C = 2 classes · B = batch size",
                     (18, 18))
    box(ax, 27, 99, 46, 10, "Keras input → flatten", [
        "Input [B, 70, 1] → feature vector x [B, 70]",
        "Each of the two branches receives all 70 features"], color="blue", size=12)
    box(ax, 3, 77, 45, 15, "Antecedent: Gaussian rule matching", [
        f"Centers c [{n_rules}, 70]: {centers:,} learned parameters",
        f"Raw widths s [{n_rules}, 70]: {centers:,} learned parameters",
        r"Positive widths: $\sigma = \mathrm{softplus}(s) + 10^{-4}$",
        r"$z_{b,r,d} = 0.5[(x_{b,d}-c_{r,d})/\sigma_{r,d}]^2$",
        f"Distance tensor z: [B, {n_rules}, 70]"], color="purple", size=11.8)
    box(ax, 54, 77, 43, 15, "Consequent: one linear predictor per rule", [
        f"Weights W [{n_rules}, 70, 2]: {weights:,} parameters",
        f"Biases b [{n_rules}, 2]: {biases:,} parameters",
        "g[b,r,c] = dot(x[b,:], W[r,:,c]) + b[r,c]",
        f"Rule logits g: [B, {n_rules}, 2]",
        "These are class scores before class softmax"], color="teal", size=11.8, title_size=13)
    arrow(ax, [(50, 99), (50, 95.5), (25.5, 95.5), (25.5, 92.3)])
    arrow(ax, [(50, 99), (50, 95.5), (75.5, 95.5), (75.5, 92.3)])
    box(ax, 3, 63, 45, 10, "HTSK aggregation over feature dimensions", [
        "q[b,r] = -mean(z[b,r,:])",
        f"Mean negative distance → rule scores q [B, {n_rules}]",
        "Calculated directly in log space"], color="purple", size=12)
    arrow(ax, [(25.5, 77), (25.5, 73.3)])
    box(ax, 3, 49, 45, 10, f"Rule softmax: normalize across {n_rules} rules", [
        "alpha = softmax(q, axis=rules)",
        f"Weights alpha [B, {n_rules}]; sum across rules = 1",
        "Rule weights are not low / high class probabilities"], color="purple", size=11.8)
    arrow(ax, [(25.5, 63), (25.5, 59.3)])
    box(ax, 55, 51, 40, 19, "How the two branches work together", [
        "IF a window resembles rule r's learned pattern,",
        "THEN rule r's linear predictor gets more weight.",
        "All rules contribute smoothly; there is no hard selection.",
        f"{n_rules} rules × 70 memberships = {centers:,} memberships.",
        "Each rule predicts both low and high class scores."], color="gray", size=11.5, title_size=13)
    box(ax, 23, 35, 54, 10, "Combine the rule logits using their weights", [
        "logits[b,c] = sum_r(alpha[b,r] × g[b,r,c])",
        "Weighted sum over rules → two logits [B, 2]"], color="blue", size=12)
    arrow(ax, [(25.5, 49), (25.5, 47), (39, 47), (39, 45.3)])
    arrow(ax, [(97, 84), (98.5, 84), (98.5, 47), (65, 47), (65, 45.3)],
          color=COLORS["teal"][1])
    box(ax, 23, 21, 54, 10, "Class softmax: normalize across 2 classes", [
        "p = softmax(logits, axis=classes)",
        "P(low), P(high) [B, 2] → argmax gives predicted class"], color="orange", size=12)
    arrow(ax, [(50, 35), (50, 31.3)])
    box(ax, 3, 6, 94, 11, "Training updates both branches", [
        "Cross-entropy from true training labels → backpropagation → Adam updates c, s, W and b",
        "K-means initializes centers on up to 30,000 standardized training rows; initial sigma = 1",
        "Validation selects the checkpoint; validation and test rows never initialize rules or update parameters"],
        color="gray", size=12)
    arrow(ax, [(50, 21), (50, 17.3)], dashed=True)
    ax.text(50, 2.3, f"Total: {centers:,} centers + {centers:,} raw widths + {weights:,} linear weights + {biases:,} biases = {total:,} trainable parameters",
            ha="center", va="center", fontsize=12, fontweight="bold", color=COLORS["ink"])
    ax.text(50, -.1, "Exact code: cfnn/tsk.py + cfnn/models.py. The custom Keras layer contains both branches; the class softmax follows it.",
            ha="center", va="center", fontsize=11, color=COLORS["muted"])
    save(fig, "htsk_rule_architecture" if dataset == "DEAP" else "htsk_dreamer_rule_architecture")


if __name__ == "__main__":
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})
    pipeline()
    dataset_comparison()
    unified_architecture()
    internals()
    internals(n_rules=16, dataset="DREAMER")
