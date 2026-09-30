"""
generate_workflow_diagram.py -- Fig. 1 of the manuscript: PC-WavKAN as implemented.

Redrawn 2026-09-30 (final pre-submission audit) for print legibility and fidelity to
the code:

* drawn at its printed size (7.2 in wide, 7-8 pt text) instead of a 16-in canvas that
  shrank every label to ~4 pt once placed at text width;
* every block, operation and tensor shape follows models/wavkan_v2.py, src/pwam.py and
  src/wavkan_pcwi.py (use_rr_attn=False, the evaluated configuration);
* per-block parameter counts are computed from the instantiated model at run time, so
  the figure cannot drift from the code;
* no title inside the image (the caption carries it) and no decorative labels;
* the training-only mechanisms are drawn as a separate band, because class-balanced
  sampling acts on the training data stream, not on any block of the network.

Earlier corrections kept: the beat window has the R-peak at 250 ms (sample 90), and the
side branch reads samples 80-160, i.e. -28 to +194 ms around R (QRS/ST), not the P-wave
(AUDIT_FINDINGS.md H60); every reported checkpoint was selected in the class-balanced
phase (H61).

Usage:
    python src/generate_workflow_diagram.py --output Submission_Array/final_methodology_workflow_v2.pdf
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
})

# restrained, print-safe palette: fill, edge
COL = {
    "input":  ("#F4F6F8", "#4D5B6A"),
    "morph":  ("#E3ECF6", "#2F5D8C"),
    "side":   ("#EAF1F8", "#2F5D8C"),
    "rhythm": ("#F7E8E6", "#9A3B30"),
    "fuse":   ("#EDEDED", "#555555"),
    "head":   ("#E5F2E7", "#2E7040"),
    "train":  ("#FDF6E3", "#A07800"),
}
ARROW = "#2B2B2B"


def param_counts():
    """Per-block trainable parameters of the evaluated configuration."""
    from models.wavkan_v2 import WavKAN_v2
    m = WavKAN_v2(use_rr_attn=False)
    n = lambda mod: sum(p.numel() for p in mod.parameters() if p.requires_grad)  # noqa: E731
    counts = {
        "kan": n(m.kan) + n(m.kan_norm),
        "gru": n(m.bigru),
        "side": n(m.pwam),
        "rr": n(m.rr_branch),
        "head": n(m.classifier),
    }
    counts["total"] = n(m)
    assert sum(v for k, v in counts.items() if k != "total") == counts["total"], counts
    return counts


def box(ax, x0, y0, w, h, kind, lines, title_bold=True, dashed=False, fs=7.5):
    fill, edge = COL[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.25,rounding_size=0.9",
                                fc=fill, ec=edge, lw=0.9, ls=(0, (4, 2.5)) if dashed else "-",
                                zorder=2))
    n = len(lines)
    step = min(h / (n + 0.6), 3.25)
    top = y0 + h / 2 + step * (n - 1) / 2
    for i, t in enumerate(lines):
        bold = title_bold and i == 0
        ax.text(x0 + w / 2, top - i * step, t, ha="center", va="center", fontsize=fs,
                fontweight="bold" if bold else "normal", color="#1A1A1A", zorder=3)


def note(ax, x, y, t, **kw):
    ax.text(x, y, t, ha="center", va="top", fontsize=6.6, color="#555555", style="italic", **kw)


def arrow(ax, p0, p1, color=ARROW, dashed=False, rad=0.0, lw=0.9):
    ax.annotate("", xy=p1, xytext=p0, zorder=1,
                arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", lw=lw, color=color,
                                ls=(0, (4, 2.5)) if dashed else "-",
                                connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))


def label(ax, x, y, t, color="#333333"):
    ax.text(x, y, t, ha="center", va="bottom", fontsize=6.8, color=color, zorder=3)


def generate_workflow_figure(output_path):
    pc = param_counts()
    # Taller canvas (2026-09-30, on request): more room between the lanes, around the
    # side-branch encoder and inside each box. Units stay equal in x and y.
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 67)
    ax.axis("off")

    k = lambda v: f"{v / 1000:.1f}K params"  # noqa: E731
    YM, YR, H = 38.0, 18.0, 12.5   # lane centre lines and box height
    YE0, YE1 = 51.0, 61.0           # side-branch encoder box (bottom, top)
    YROUTE = (YE0 + YE1) / 2

    ax.text(0.3, 66.5, "Morphology branch", fontsize=7.4, fontweight="bold",
            color=COL["morph"][1], va="top")
    ax.text(0.3, 29.0, "Rhythm branch", fontsize=7.4, fontweight="bold",
            color=COL["rhythm"][1], va="top")

    # ── morphology lane ─────────────────────────────────────────────────────
    box(ax, 0.5, YM - H / 2, 13.0, H, "input",
        ["Beat window", r"$\mathbf{x}\in\mathbb{R}^{360}$", "−250 to +750 ms", "z-scored"], fs=7.0)
    box(ax, 17.5, YM - H / 2, 18.5, H, "morph",
        ["Wavelet-KAN layer", "PCWI init., Mexican Hat", r"360$\rightarrow$64, + linear path",
         "LayerNorm, dropout"], fs=7.0)
    note(ax, 26.75, YM - H / 2 - 1.0, k(pc["kan"]))
    box(ax, 40.0, YM - H / 2, 11.5, H, "morph",
        ["BiGRU", "1 step, 2×32", r"$\mathbf{z}_m\in\mathbb{R}^{64}$"], fs=7.0)
    note(ax, 45.75, YM - H / 2 - 1.0, k(pc["gru"]))
    box(ax, 55.5, YM - H / 2, 15.0, H, "side",
        ["Gated fusion", r"$g=\sigma(W[\mathbf{z}_m;\mathbf{a}])$", r"$\mathbf{z}_m+g\odot\mathbf{a}$"],
        fs=7.0)
    note(ax, 63.0, YM - H / 2 - 1.0, k(pc["side"]) + " (with encoder)")

    # side-branch encoder above the lane
    box(ax, 37.0, YE0, 35.0, YE1 - YE0, "side",
        ["Side-branch encoder", r"samples 80$-$160 of $\mathbf{x}$ ($-$28 to +194 ms)",
         r"Wavelet-KAN 80$\rightarrow$32, Linear$\rightarrow$64:  $\mathbf{a}\in\mathbb{R}^{64}$"],
        fs=7.0)

    arrow(ax, (13.8, YM), (17.2, YM))
    label(ax, 15.5, YM + 0.6, "360")
    arrow(ax, (36.3, YM), (39.7, YM))
    label(ax, 38.0, YM + 0.6, "64")
    arrow(ax, (51.8, YM), (55.2, YM))
    label(ax, 53.5, YM + 0.6, "64")
    # beat window -> encoder, routed over the top
    ax.plot([7.0, 7.0, 36.0], [YM + H / 2 + 0.3, YROUTE, YROUTE], color=ARROW, lw=0.9, zorder=1)
    arrow(ax, (35.5, YROUTE), (36.7, YROUTE))
    # encoder -> gated fusion
    arrow(ax, (63.0, YE0 - 0.3), (63.0, YM + H / 2 + 0.3))
    label(ax, 64.4, (YE0 + YM + H / 2) / 2 - 0.8, r"$\mathbf{a}$")

    # ── rhythm lane ─────────────────────────────────────────────────────────
    box(ax, 0.5, YR - H / 2, 13.0, H, "input",
        ["RR intervals", r"$\mathbf{r}\in\mathbb{R}^{5}$",
         r"RR$_{-2}\ldots$RR$_{+2}$", "seconds"], fs=7.0)
    box(ax, 17.5, YR - H / 2, 18.5, H, "rhythm",
        ["Rhythm encoder", r"MLP 5$\rightarrow$64$\rightarrow$32$\rightarrow$16",
         r"ReLU; $\mathbf{z}_r\in\mathbb{R}^{16}$"], fs=7.0)
    note(ax, 26.75, YR - H / 2 - 1.0, k(pc["rr"]))
    arrow(ax, (13.8, YR), (17.2, YR))
    label(ax, 15.5, YR + 0.6, "5")

    # ── concatenation and head ──────────────────────────────────────────────
    xc0, xc1 = 74.5, 79.0
    fill, edge = COL["fuse"]
    ax.add_patch(FancyBboxPatch((xc0, YR - H / 2), xc1 - xc0, (YM + H / 2) - (YR - H / 2),
                                boxstyle="round,pad=0.25,rounding_size=0.9", fc=fill, ec=edge,
                                lw=0.9, zorder=2))
    ax.text((xc0 + xc1) / 2, (YM + YR) / 2, r"concatenate  $\mathbb{R}^{80}$", rotation=90,
            ha="center", va="center", fontsize=7.0, fontweight="bold", color="#1A1A1A", zorder=3)
    arrow(ax, (70.8, YM), (xc0 - 0.3, YM))
    label(ax, 72.6, YM + 0.6, "64")
    ax.plot([36.3, 72.5], [YR, YR], color=ARROW, lw=0.9, zorder=1)
    arrow(ax, (72.0, YR), (xc0 - 0.3, YR))
    label(ax, 55.0, YR + 0.6, "16")

    yh = (YM + YR) / 2
    box(ax, 83.0, yh - H / 2, 16.5, H, "head",
        ["Classifier head", r"80$\rightarrow$48$\rightarrow$5", "ReLU, dropout", "softmax"], fs=7.0)
    note(ax, 91.25, yh - H / 2 - 1.0, k(pc["head"]))
    arrow(ax, (xc1 + 0.3, yh), (82.7, yh))
    label(ax, 81.0, yh + 0.6, "80")
    arrow(ax, (91.25, yh + H / 2 + 0.3), (91.25, yh + H / 2 + 4.5))
    ax.text(91.25, yh + H / 2 + 4.9, "N, S, V, F, Q", ha="center", va="bottom", fontsize=7.4,
            fontweight="bold", color="#1A1A1A")

    # ── training-only band and legend ───────────────────────────────────────
    box(ax, 17.5, 0.5, 82.0, 6.6, "train",
        ["Training only: class-balanced sampling with unweighted cross-entropy (every retained checkpoint "
         "from epochs ≤ 25);",
         "augmentation of non-N beats; AdamW; checkpoint selection on DS1-validation Macro-F1."],
        title_bold=False, dashed=True, fs=6.8)
    ax.plot([0.8, 4.3], [5.4, 5.4], color=ARROW, lw=0.9)
    ax.text(5.0, 5.4, "data flow", fontsize=6.6, va="center", color="#333333")
    ax.plot([0.8, 4.3], [2.2, 2.2], color=COL["train"][1], lw=0.9, ls=(0, (4, 2.5)))
    ax.text(5.0, 2.2, "training only", fontsize=6.6, va="center", color="#333333")

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Generated workflow diagram at {output_path} (total {pc['total']:,} parameters)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="Submission_Array/final_methodology_workflow_v2.pdf")
    generate_workflow_figure(ap.parse_args().output)
