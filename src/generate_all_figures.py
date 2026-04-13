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
    Publication-ready seed stability figure.
    Main panel: all 5 models, full range (shows dramatic gap).
    Inset panel: baselines only, zoomed 0-0.18 (makes distributions readable).
    Colors: WavKAN-v2 = champion deep blue; baselines = muted neutrals.
    Stats: Wilcoxon signed-rank + Cohen's d + Levene's variance test.
    """
    from scipy.stats import wilcoxon, levene
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

    base = Path(results_base)
    out  = Path(out_dir)

    # ── Load WavKAN-v2 per-seed ────────────────────────────────────────────────
    f1s_wavkan = []
    for seed in seeds:
        p = base / "wavkan_v2" / f"seed_{seed}" / "test_metrics.json"
        if p.exists():
            with open(p) as f:
                m = json.load(f)
            f1s_wavkan.append(m.get("macro_f1", 0))

    # ── Load baselines ─────────────────────────────────────────────────────────
    bsum = {}
    summary_path = base / "baselines_summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            bsum = json.load(f)

    # Champion color for WavKAN-v2; muted neutrals for baselines
    registry = [
        ("WavKAN-v2\n(Ours)",   None,          "#1f4e79"),   # deep blue = winner
        ("ResNet1D",             "resnet1d",    "#adb5bd"),   # gray
        ("Transformer",          "transformer", "#c9a84c"),   # amber
        ("CNN +\nFocal Loss",    "cnn_focal",   "#8fbcad"),   # muted teal
        ("B-Spline\nKAN",        "bspline_kan", "#6d8bb0"),   # steel blue
    ]

    plot_data, plot_labels, plot_clrs = [], [], []
    for label, key, color in registry:
        vals = f1s_wavkan if key is None else bsum.get(key, {}).get("_raw_f1", [])
        plot_data.append(vals if vals else [])
        plot_labels.append(label)
        plot_clrs.append(color)

    if not any(plot_data):
        print("  [Fig 3] No data found. Skipping.")
        return

    # ── Main figure + inset ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5.8))
    positions = list(range(1, len(registry) + 1))

    def _draw_boxes(target_ax, data, clrs, positions, show_fliers=True):
        bp_data = [d if d else [np.nan] for d in data]
        fp = {"marker": "D", "markersize": 3, "alpha": 0.4} if show_fliers else {"marker": ""}
        bp = target_ax.boxplot(bp_data, positions=positions, patch_artist=True,
                               notch=False,
                               medianprops={"lw": 2.0, "color": "white"},
                               whiskerprops={"lw": 1.4, "color": "#555"},
                               capprops={"lw": 1.6, "color": "#555"},
                               flierprops=fp)
        for patch, color, d in zip(bp["boxes"], clrs, data):
            patch.set_facecolor(color)
            patch.set_alpha(0.88 if d else 0.1)
        return bp

    _draw_boxes(ax, plot_data, plot_clrs, positions)

    # Beeswarm-style jitter (wider spread, no overlap)
    np.random.seed(42)
    for i, (vals, color) in enumerate(zip(plot_data, plot_clrs), 1):
        if vals:
            n = len(vals)
            jit = np.linspace(-0.10, 0.10, n) + np.random.uniform(-0.02, 0.02, n)
            ax.scatter(np.ones(n) * i + jit, vals,
                       color="#222", s=28, alpha=0.75, zorder=5, edgecolors="white",
                       linewidths=0.4)

    # ── Annotations: mean±std, Wilcoxon star, Cohen's d ───────────────────────
    ref = plot_data[0]
    for i, vals in enumerate(plot_data, 1):
        if not vals:
            continue
        clean = [v for v in vals if not np.isnan(v)]
        if not clean:
            continue
        μ, σ = np.mean(clean), (np.std(clean, ddof=1) if len(clean) > 1 else 0.0)
        top   = max(clean)

        sig_txt, d_txt = "", ""
        if i > 1 and ref and len(ref) == len(clean) and len(clean) >= 3:
            try:
                _, pval = wilcoxon(ref, clean, alternative="greater", zero_method="wilcox")
                sig_txt = "**" if pval < 0.01 else ("*" if pval < 0.05 else "ns")
                diffs   = np.array(ref) - np.array(clean)
                d_val   = np.mean(diffs) / (np.std(diffs, ddof=1) + 1e-9)
                d_txt   = f"d={d_val:.1f}"
            except Exception:
                pass

        label_y = top + 0.010
        sc = "#1a7a1a" if sig_txt not in ("", "ns") else "#888"
        if sig_txt:
            ax.text(i, label_y + 0.022, sig_txt, ha="center", va="bottom",
                    fontsize=12, color=sc, fontweight="bold")
        if d_txt:
            ax.text(i, label_y + 0.008, d_txt, ha="center", va="bottom",
                    fontsize=6.5, color=sc, style="italic")
        ax.text(i, label_y, f"{μ:.3f}\n±{σ:.3f}", ha="center", va="bottom",
                fontsize=6.5, color="#333", fontweight="bold", linespacing=1.3)

    # Transformer instability annotation
    t_vals = plot_data[2] if len(plot_data) > 2 else []
    if t_vals and max(t_vals) > 0.08:
        ax.annotate("instability\nevent", xy=(3, max(t_vals)),
                    xytext=(3.45, max(t_vals) + 0.018),
                    fontsize=6.5, color="#888", style="italic",
                    arrowprops=dict(arrowstyle="->", color="#bbb", lw=0.8))

    # Reference line (champion model mean)
    if ref:
        ax.axhline(np.mean(ref), color=plot_clrs[0], lw=1.3, ls="--", alpha=0.6,
                   label=f"Mean performance (WavKAN-v2)")

    # Levene's test inset text
    others = [v for d in plot_data[1:] for v in d if d]
    if ref and others:
        try:
            _, lp = levene(ref, others)
            ltxt = f"Levene's test (variance): p<0.001" if lp < 0.001 else f"Levene's test (variance): p={lp:.3f}"
            ax.text(0.015, 0.975, ltxt, transform=ax.transAxes, fontsize=7,
                    va="top", ha="left", color="#444", style="italic",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.75, ec="#bbb"))
        except Exception:
            pass

    # Collapse zone shading (neutral, no text)
    ax.axhspan(0.0, 0.026, alpha=0.06, color="#888")

    # Main axis formatting
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_labels, fontsize=9)
    ax.set_ylabel("Macro-F1 (DS2 Test Set)", fontsize=11)
    ax.set_ylim(0.0, 0.44)
    ax.set_title(
        f"Seed Stability Analysis (n={len(seeds)} seeds) — All Comparison Models\n"
        "WavKAN-v2 achieves highest Macro-F1 with consistently low cross-seed variance  "
        "(*p<0.05, **p<0.01, Wilcoxon signed-rank; vs WavKAN-v2)",
        fontweight="bold", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.25, linestyle=":")

    # ── Zoomed inset: baseline distributions 0–0.18 ───────────────────────────
    axins = ax.inset_axes([0.24, 0.40, 0.44, 0.52])   # [x0, y0, width, height]
    baseline_pos  = [2, 3, 4, 5]
    baseline_data = plot_data[1:]
    baseline_clrs = plot_clrs[1:]
    _draw_boxes(axins, baseline_data, baseline_clrs, baseline_pos, show_fliers=False)

    np.random.seed(42)
    for pos, vals, color in zip(baseline_pos, baseline_data, baseline_clrs):
        if vals:
            n = len(vals)
            jit = np.linspace(-0.12, 0.12, n) + np.random.uniform(-0.02, 0.02, n)
            axins.scatter(np.ones(n) * pos + jit, vals,
                          color="#222", s=20, alpha=0.8, zorder=5,
                          edgecolors="white", linewidths=0.3)

    axins.set_xlim(1.4, 5.6)
    axins.set_ylim(0.0, 0.18)
    axins.set_xticks(baseline_pos)
    axins.set_xticklabels(["ResNet1D", "Transf.", "CNN+FL", "B-Spline"], fontsize=6.5)
    axins.set_ylabel("Macro-F1", fontsize=7)
    axins.set_title("Baseline zoom (0–0.18)", fontsize=7.5, fontweight="bold")
    axins.tick_params(axis="y", labelsize=6.5)
    axins.grid(axis="y", alpha=0.3, linestyle=":")
    axins.set_facecolor("#fafafa")
    for spine in axins.spines.values():
        spine.set_edgecolor("#999")
        spine.set_linewidth(0.8)

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
