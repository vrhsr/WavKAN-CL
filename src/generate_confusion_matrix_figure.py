"""
generate_confusion_matrix_figure.py -- Fig. 3 of the manuscript: the DS2 confusion
matrix of the evaluated configuration (Submission_Array/manuscript.tex,
\\label{fig:confusion}).

Redrawn 2026-09-30 (final pre-submission audit). The previous version plotted one seed
(42). Its S-recall (0.25) is well above the 20-seed mean the paper reports (0.198), so the
figure contradicted Table 4 and invited a seed-selection question, even though seed 42 was
simply the first seed of the list. It also implied that S beats go mainly to V (0.42 vs
0.30 to N), which the 20-seed mean does not support (0.40 vs 0.39).

This version shows, for every cell, the MEAN of the 20 per-seed row-normalised matrices
(each seed weighted equally, so the diagonal equals the per-class recall of Table 4) with
the across-seed standard deviation, and it labels each row with its support so that the
7-beat Q row is not over-read. It reads the saved per-seed confusion_matrix.npy files; it
never re-runs inference, so it cannot silently use the wrong architecture's predictions.
No title inside the image (the caption carries it); drawn at its printed size.

Usage:
    python src/generate_confusion_matrix_figure.py --arm results/ablation_no_rr_attn \\
        --output Submission_Array/final_confusion_matrix_main.pdf
"""
import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
})

CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def load_row_normalised(arm):
    files = sorted(glob.glob(os.path.join(arm, "seed_*", "confusion_matrix.npy")))
    if not files:
        raise FileNotFoundError(f"no seed_*/confusion_matrix.npy under {arm}")
    cms = [np.load(f) for f in files]
    support = cms[0].sum(1)
    for c in cms:
        assert np.array_equal(c.sum(1), support), "per-seed matrices disagree on class support"
    rn = np.array([c / c.sum(1, keepdims=True) for c in cms])
    return rn.mean(0), rn.std(0, ddof=1), support, len(files)


def plot_confusion_matrix(mean, sd, support, n_seeds, output_path):
    fig, ax = plt.subplots(figsize=(3.9, 3.35))
    im = ax.imshow(mean, cmap="Blues", vmin=0.0, vmax=1.0)
    k = len(CLASS_NAMES)
    ax.set_xticks(range(k))
    ax.set_yticks(range(k))
    ax.set_xticklabels(CLASS_NAMES, fontsize=8)
    ax.set_yticklabels([f"{c}  (n = {int(n):,})" for c, n in zip(CLASS_NAMES, support)], fontsize=7.4)
    ax.set_xlabel("Predicted class", fontsize=8, labelpad=3)
    ax.set_ylabel("True class", fontsize=8, labelpad=3)
    ax.tick_params(length=0, pad=3)
    for i in range(k):
        for j in range(k):
            v, e = mean[i, j], sd[i, j]
            col = "white" if v > 0.55 else "#1A1A1A"
            ax.text(j, i - 0.12, f"{v:.2f}", ha="center", va="center", fontsize=7.8, color=col,
                    fontweight="bold" if i == j else "normal")
            ax.text(j, i + 0.22, f"±{e:.2f}", ha="center", va="center", fontsize=6.0, color=col)
    ax.set_xticks(np.arange(-0.5, k, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, k, 1), minor=True)
    ax.grid(which="minor", color="white", lw=0.8)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_color("#555555")
        sp.set_linewidth(0.7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label(f"row-normalised fraction (mean of {n_seeds} seeds)", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, width=0.5, length=2)
    cbar.outline.set_linewidth(0.5)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved -> {output_path}  ({n_seeds} seeds)")
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c}: " + "  ".join(f"{m:.3f}+-{s:.3f}" for m, s in zip(mean[i], sd[i])))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="results/ablation_no_rr_attn",
                    help="directory with seed_*/confusion_matrix.npy (the evaluated configuration)")
    ap.add_argument("--output", default="Submission_Array/final_confusion_matrix_main.pdf")
    a = ap.parse_args()
    plot_confusion_matrix(*load_row_normalised(a.arm), a.output)
