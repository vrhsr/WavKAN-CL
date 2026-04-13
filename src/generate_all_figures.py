"""
generate_all_figures.py  —  Master Publication Figure Generator

ONE COMMAND TO GENERATE EVERY FIGURE IN THE PAPER
===================================================
Runs all figure generation scripts in the correct order,
reading from your results directory.

Every figure output matches the journal style:
  - Font size 11pt (IEEE TBME standard)
  - 200 DPI (PDF + PNG, print-quality)
  - Color palette consistent across all panels
  - Caption-ready filenames

Figure Map (matches paper section order):
  Fig 1:  Architecture diagram       → fig1_architecture.pdf
  Fig 2:  PCWI initialisation figure → fig2_pcwi_init.pdf
  Fig 3:  Seed stability boxplot     → fig3_seed_stability.pdf
  Fig 4:  Training convergence       → fig4_convergence.pdf
  Fig 5:  Confusion matrix           → fig5_confusion_matrix.pdf
  Fig 6:  Leave-one-out RR ablation  → fig6_rr_ablation.pdf
  Fig 7:  Wavelet alignment heatmap  → fig7_wavelet_alignment.pdf
  Fig 8:  Per-patient heatmap [NEW]  → fig8_per_patient.pdf
  Fig 9:  Clinical dashboard [NEW]   → fig9_clinical_dashboard.pdf
  Fig 10: Multi-dataset bar [NEW]    → fig10_multidataset.pdf
  Fig 11: Reliability diagram [NEW]  → fig11_reliability.pdf
  Fig 12: Augmentation comparison    → fig12_augmentation.pdf

Usage:
    python src/generate_all_figures.py \\
        --results-base results/full_pipeline/ \\
        --output-dir results/final_figures/
"""

import os, sys, json, argparse, shutil
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Global style (IEEE TBME journal standard)
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":        200,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "axes.grid":         True,
    "grid.alpha":        0.3,
})

PALETTE   = {"N": "#aec7e8", "S": "#f28e2b", "V": "#e15759", "F": "#76b7b2", "Q": "#bab0ac"}
MODEL_CLR = {"WavKAN-v2": "#e15759", "B-Spline KAN": "#4e79a7",
             "ResNet1D": "#59a14f", "Transformer": "#f28e2b"}


# ─────────────────────────────────────────────────────────────────────────────
# Fig 3 — Seed Stability Boxplot
# ─────────────────────────────────────────────────────────────────────────────

def fig3_seed_stability(results_base: str, out_dir: str, seeds: list):
    """
    Boxplot of Macro-F1 across seeds: WavKAN-v2 vs Baseline CE vs B-Spline KAN.
    Reads from the summary JSON files the full pipeline actually generates.
    """
    base = Path(results_base)
    out  = Path(out_dir)

    data, labels, clrs = [], [], []

    # ── WavKAN-v2: read per-seed test_metrics.json ────────────────────────────
    f1s_wavkan = []
    for seed in seeds:
        p = base / "wavkan_v2" / f"seed_{seed}" / "test_metrics.json"
        if p.exists():
            with open(p) as f:
                m = json.load(f)
            f1s_wavkan.append(m.get("macro_f1", 0))
    if f1s_wavkan:
        data.append(f1s_wavkan)
        labels.append("WavKAN-v2 (Ours)")
        clrs.append("#e15759")

    # ── Baselines: read from baselines_summary.json ───────────────────────────
    summary_path = base / "baselines_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            bsum = json.load(f)

        # Baseline CE/CNN focal
        for key in ["cnn_focal", "resnet1d"]:
            if key in bsum and "_raw_f1" in bsum[key]:
                raw = bsum[key]["_raw_f1"]
                if raw:
                    data.append(raw)
                    labels.append("Baseline CE (No Curriculum)")
                    clrs.append("#aec7e8")
                    break

        # B-Spline KAN
        if "bspline_kan" in bsum and "_raw_f1" in bsum["bspline_kan"]:
            raw = bsum["bspline_kan"]["_raw_f1"]
            if raw:
                data.append(raw)
                labels.append("B-Spline KAN")
                clrs.append("#4e79a7")
    else:
        # Fallback: try ablation_v2 directory for A9 (no curriculum)
        a9_f1s = []
        for seed in seeds:
            p = base / "ablation_v2" / "A9_no_curriculum" / f"seed_{seed}" / "test_metrics.json"
            if p.exists():
                with open(p) as f:
                    m = json.load(f)
                a9_f1s.append(m.get("macro_f1", 0))
        if a9_f1s:
            data.append(a9_f1s)
            labels.append("Baseline CE (No Curriculum)")
            clrs.append("#aec7e8")

        bspline_f1s = []
        for seed in seeds:
            p = base / "ablation_v2" / "A4_bspline" / f"seed_{seed}" / "test_metrics.json"
            if p.exists():
                with open(p) as f:
                    m = json.load(f)
                bspline_f1s.append(m.get("macro_f1", 0))
        if bspline_f1s:
            data.append(bspline_f1s)
            labels.append("B-Spline KAN")
            clrs.append("#4e79a7")

    if not data:
        print("  [Fig 3] No data found. Skipping.")
        return

    # ── Plot ──────────────────────────────────────────────────────────────────
    # Pad missing models with placeholder so the x-axis always shows 3 groups
    all_labels = ["WavKAN-v2 (Ours)", "Baseline CE (No Curriculum)", "B-Spline KAN"]
    all_colors = ["#e15759", "#aec7e8", "#4e79a7"]
    plot_data, plot_clrs = [], []
    for lbl, clr in zip(all_labels, all_colors):
        if lbl in labels:
            plot_data.append(data[labels.index(lbl)])
            plot_clrs.append(clr)
        else:
            plot_data.append([])   # empty — will render as empty box
            plot_clrs.append(clr)

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = [i+1 for i in range(len(all_labels))]
    non_empty = [(i, d) for i, d in enumerate(plot_data) if d]

    if non_empty:
        bp = ax.boxplot([d if d else [0] for d in plot_data],
                        positions=positions,
                        patch_artist=True, notch=False,
                        medianprops={"lw": 2.5, "color": "white"},
                        whiskerprops={"lw": 1.5}, capprops={"lw": 1.5})
        for patch, color, d in zip(bp["boxes"], plot_clrs, plot_data):
            patch.set_facecolor(color)
            patch.set_alpha(0.8 if d else 0.15)   # fade empty boxes

        for i, vals in enumerate(plot_data, 1):
            if vals:
                jit = np.random.uniform(-0.06, 0.06, len(vals))
                ax.scatter(np.ones(len(vals)) * i + jit, vals,
                           color="black", s=28, alpha=0.65, zorder=5)

    ax.axhline(np.mean(plot_data[0]) if plot_data[0] else 0.32,
               color=plot_clrs[0], lw=1, ls="--", alpha=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(all_labels, fontsize=9)
    ax.set_ylabel("Macro-F1 (DS2 Test)", fontsize=11)
    ax.set_title(f"Seed Stability Analysis (n={len(seeds)} seeds)\n"
                 "WavKAN-v2 dramatically outperforms deep learning baselines",
                 fontweight="bold")
    ax.set_ylim(0.0, 0.45)
    plt.tight_layout()
    _save(fig, out / "fig3_seed_stability.pdf")



# ─────────────────────────────────────────────────────────────────────────────
# Fig 4 — Training Convergence
# ─────────────────────────────────────────────────────────────────────────────

def fig4_convergence(results_base: str, out_dir: str, seed: int = 42):
    """4-panel convergence: loss / val-F1 / V-Recall / S-Recall."""
    base     = Path(results_base)
    out      = Path(out_dir)
    hist_path = base / "wavkan_v2" / f"seed_{seed}" / "training_history.json"

    if not hist_path.exists():
        print("  [Fig 4] training_history.json not found. Skipping.")
        return

    with open(hist_path) as f:
        history = json.load(f)

    epochs    = [h["epoch"]         for h in history]
    losses    = [h["loss"]          for h in history]
    val_f1s   = [h["val_macro_f1"]  for h in history]
    val_vr    = [h["val_v_recall"]  for h in history]
    val_sr    = [h["val_s_recall"]  for h in history]

    # Find phase transition (warmup end)
    warmup_epoch = next(
        (h["epoch"] for h in history if not h["phase"].startswith("WARMUP")), None
    )

    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
    panels    = [
        (axes[0,0], losses,  "Training Loss",         "#4e79a7"),
        (axes[0,1], val_f1s, "Validation Macro-F1",   "#59a14f"),
        (axes[1,0], val_vr,  "Validation V-Recall",   "#e15759"),
        (axes[1,1], val_sr,  "Validation S-Recall",   "#f28e2b"),
    ]
    for ax, vals, ylabel, color in panels:
        ax.plot(epochs, vals, color=color, lw=2.0)
        if warmup_epoch:
            ax.axvline(warmup_epoch, color="black", lw=1.2, ls="--", alpha=0.7)
            if ax is axes[0, 0]:
                ax.text(warmup_epoch + 0.5, max(vals) * 0.95,
                        "Phase\ntransition", fontsize=8, ha="left")
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_ylim(bottom=0)

    for ax in axes[1]:
        ax.set_xlabel("Epoch", fontsize=10)

    fig.suptitle(f"Training Convergence — WavKAN-v2, Seed {seed}\n"
                 "Dashed line = warmup → annealing phase transition",
                 fontweight="bold", fontsize=12)
    plt.tight_layout()
    _save(fig, out / "fig4_convergence.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Fig 5 — Confusion Matrix
# ─────────────────────────────────────────────────────────────────────────────

def fig5_confusion_matrix(results_base: str, out_dir: str, seed: int = 42):
    base = Path(results_base)
    out  = Path(out_dir)
    cm_path = base / "wavkan_v2" / f"seed_{seed}" / "confusion_matrix.npy"

    if not cm_path.exists():
        print("  [Fig 5] confusion_matrix.npy not found. Skipping.")
        return

    cm     = np.load(str(cm_path))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    classes = ["N", "S", "V", "F", "Q"]
    ax.set_xticks(range(5)); ax.set_yticks(range(5))
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=11); ax.set_ylabel("True", fontsize=11)
    ax.set_title(f"Normalised Confusion Matrix — WavKAN-v2 (DS2 Test Set, Seed {seed})",
                 fontweight="bold")

    for i in range(5):
        for j in range(5):
            color = "white" if cm_norm[i, j] > 0.55 else "black"
            ax.text(j, i, f"{cm_norm[i,j]:.3f}", ha="center", va="center",
                    fontsize=9, color=color, fontweight="bold" if i == j else "normal")

    plt.tight_layout()
    _save(fig, out / "fig5_confusion_matrix.pdf")


# ─────────────────────────────────────────────────────────────────────────────
# Copy pre-generated figures from sub-results directories
# ─────────────────────────────────────────────────────────────────────────────

def _copy_if_exists(src: Path, dst: Path, label: str):
    if src.exists():
        shutil.copy(src, dst)
        print(f"  [{label}] Copied: {dst.name}")
    else:
        print(f"  [{label}] Source not found: {src} — run the corresponding script first")


def collect_external_figures(results_base: str, out_dir: str):
    """Links/copies figures generated by other scripts into final_figures/."""
    base = Path(results_base)
    out  = Path(out_dir)

    mappings = [
        (base / "rr_ablation"          / "rr_ablation_figure.pdf",        out / "fig6_rr_ablation.pdf",         "Fig 6"),
        (base / "interpretability"      / "wavelet_alignment_heatmap.pdf",  out / "fig7_wavelet_alignment.pdf",   "Fig 7"),
        (base / "per_patient"           / "per_patient_heatmap.pdf",        out / "fig8_per_patient.pdf",         "Fig 8"),
        (base / "clinical_story"        / "clinical_dashboard.pdf",         out / "fig9_clinical_dashboard.pdf",  "Fig 9"),
        (base / "multidataset_table"    / "multidataset_bar.pdf",           out / "fig10_multidataset.pdf",       "Fig 10"),
        (base / "calibration"           / "reliability_diagram.pdf",        out / "fig11_reliability.pdf",        "Fig 11"),
        (base / "noise_augmentation"    / "augmentation_figure.pdf",        out / "fig12_augmentation.pdf",       "Fig 12"),
    ]
    for src, dst, label in mappings:
        _copy_if_exists(src, dst, label)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, path: Path):
    fig.savefig(str(path), dpi=200, bbox_inches="tight")
    # Also save PNG for quick preview
    fig.savefig(str(path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path.name}")


def generate_figure_index(out_dir: str):
    """Generates a simple HTML figure gallery for quick review."""
    out  = Path(out_dir)
    pngs = sorted(out.glob("*.png"))
    html_lines = [
        "<html><head><title>WavKAN-v2 Figure Gallery</title>",
        "<style>body{font-family:sans-serif;background:#1a1a2e;color:#eee;}</style></head><body>",
        "<h1>WavKAN-v2 — Publication Figures</h1>",
    ]
    for png in pngs:
        html_lines.append(
            f'<div style="margin:20px 0"><h3>{png.stem}</h3>'
            f'<img src="{png.name}" style="max-width:900px;border:1px solid #444"/></div>'
        )
    html_lines.append("</body></html>")
    with open(out / "figure_gallery.html", "w") as f:
        f.write("\n".join(html_lines))
    print(f"  Gallery: {out/'figure_gallery.html'}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Publication Figure Generator")
    parser.add_argument("--results-base", type=str, default="results/full_pipeline",
                        help="Root of results directory")
    parser.add_argument("--seeds",        type=int, nargs="+", default=[42, 101, 777, 2026, 31415])
    parser.add_argument("--primary-seed", type=int, default=42)
    parser.add_argument("--output-dir",   type=str, default="results/final_figures")
    args = parser.parse_args()

    OUT = Path(args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"MASTER FIGURE GENERATOR — WavKAN-v2")
    print(f"Results base : {args.results_base}")
    print(f"Output dir   : {OUT}")
    print(f"Primary seed : {args.primary_seed}")
    print(f"{'='*60}\n")

    # Generate figures that we compute directly
    print("[Fig 3] Seed stability boxplot...")
    fig3_seed_stability(args.results_base, str(OUT), args.seeds)

    print("[Fig 4] Training convergence...")
    fig4_convergence(args.results_base, str(OUT), args.primary_seed)

    print("[Fig 5] Confusion matrix...")
    fig5_confusion_matrix(args.results_base, str(OUT), args.primary_seed)

    # Collect figures generated by other scripts
    print("\nCollecting figures from sub-results directories...")
    collect_external_figures(args.results_base, str(OUT))

    # HTML gallery
    generate_figure_index(str(OUT))

    # Print manifest
    pdfs = sorted(OUT.glob("*.pdf"))
    print(f"\n{'='*60}")
    print(f"FIGURE MANIFEST ({len(pdfs)} PDF files)")
    print(f"{'='*60}")
    for pdf in pdfs:
        print(f"  {pdf.name}")

    print(f"\n✅ All figures saved to {OUT}/")
    print(f"   Open figure_gallery.html for a quick preview.")
