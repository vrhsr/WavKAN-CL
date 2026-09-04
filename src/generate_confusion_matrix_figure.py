"""
generate_confusion_matrix_figure.py -- Fig. "Normalized Confusion Matrix"
(Submission_JBHI/ieee_manuscript_v2.tex, \\label{fig:confusion}).

Why this script exists (2026-09-02, deep check of final_confusion_matrix_main.pdf
at the project owner's request): the current, real figure embedded in the
manuscript was verified byte-for-byte against
results/ablation_no_rr_attn/seed_42/confusion_matrix.npy (values match to 3
decimals) -- the figure itself is accurate. But there was no standalone,
correctly-sourced script that could reproduce it: the one script in this repo
that plots a confusion matrix, src/generate_publication_figures.py, hardcodes
the OLD, superseded HybridWavKAN_RR model (95,189 params, imported from
src/wavkan.py), not the canonical models.wavkan_v2.WavKAN_v2 this figure's
own caption claims ("final architecture, 153,045 params") -- the same
stale-model-class bug already found and fixed in several sibling scripts
this session (AUDIT_FINDINGS.md H25 and others). Re-running that script
against a real WavKAN_v2 checkpoint would either crash (state_dict mismatch)
or silently plot the wrong architecture's confusion matrix. This script is a
fresh, minimal, correctly-sourced replacement scoped to exactly this one
figure -- it does not touch generate_publication_figures.py's other
functions (PR curves, ROC curves, comparison bar chart), which are out of
scope for this fix.

Also restyled to match the publication format established this session for
the paper's other two schematic figures (serif typography, restrained
styling) for visual consistency across the paper.

Usage:
    python src/generate_confusion_matrix_figure.py \\
        --checkpoint-dir results/ablation_no_rr_attn/seed_42 \\
        --output Submission_JBHI/final_confusion_matrix_main.pdf \\
        --label "PC-WavKAN (153,045 params), seed 42, DS2 test set"
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

CLASS_NAMES = ["N", "S", "V", "F", "Q"]


def plot_confusion_matrix(cm: np.ndarray, output_path: str, label: str):
    row_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(row_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, fontsize=12)
    ax.set_yticklabels(CLASS_NAMES, fontsize=12)
    ax.set_xlabel("Predicted", fontsize=12.5, fontweight="bold")
    ax.set_ylabel("True", fontsize=12.5, fontweight="bold")

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            val = row_norm[i, j]
            color = "white" if val > 0.55 else "#1A1A1A"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                     fontsize=11, color=color, fontweight="bold" if i == j else "normal")

    ax.set_title(f"Normalized Confusion Matrix\n{label}", fontsize=12.5, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Row-normalized fraction", fontsize=11)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#333333")
        spine.set_linewidth(0.9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved -> {output_path}")
    print("Row-normalized matrix:")
    for i, c in enumerate(CLASS_NAMES):
        print(f"  {c}: " + "  ".join(f"{v:.3f}" for v in row_norm[i]))


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description="Confusion-matrix figure from a real checkpoint's saved predictions")
    parser.add_argument("--checkpoint-dir", type=str, default="results/ablation_no_rr_attn/seed_42",
                        help="Directory containing this seed's confusion_matrix.npy")
    parser.add_argument("--output", type=str, default="Submission_JBHI/final_confusion_matrix_main.pdf")
    parser.add_argument("--label", type=str,
                        default="PC-WavKAN (153,045 params), seed 42, DS2 test set")
    args = parser.parse_args()

    cm_path = Path(args.checkpoint_dir) / "confusion_matrix.npy"
    if not cm_path.exists():
        raise FileNotFoundError(
            f"{cm_path} not found -- this script reads a real, already-saved "
            "confusion matrix, it does not run inference itself. Point "
            "--checkpoint-dir at a directory containing confusion_matrix.npy "
            "(e.g. results/ablation_no_rr_attn/seed_<N>/)."
        )
    cm = np.load(cm_path)
    plot_confusion_matrix(cm, args.output, args.label)
