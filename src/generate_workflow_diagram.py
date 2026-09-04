"""
generate_workflow_diagram.py -- PC-WavKAN methodology/workflow figure.
Method name updated 2026-09-03 (AUDIT_FINDINGS.md H43): the paper no longer
uses "WavKAN-v2", which read as a second version of Bozorgasl & Chen's Wav-KAN.

Publication-format pass (2026-09-02): restyled for a submission-grade IEEE
figure (serif typography matching body text, a restrained/desaturated
palette, a legend distinguishing data-flow vs. training-only signal arrows,
row-group labels) on top of the content fix from the same date
(AUDIT_FINDINGS.md C21) -- this is a visual-polish pass only; no box's text,
no arrow's real source/target, and no architectural claim was changed here.
Cross-checked once more against models/wavkan_v2.py before this pass: the
WavKAN backbone is PCWI-initialized (use_pcwi=True) with a Mexican Hat
wavelet (wavelet_type="mexican_hat") by default -- now named explicitly in
the backbone box, since PCWI is one of this paper's own emphasized
contributions and was previously only named in prose, not the figure.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.linewidth": 0.8,
})

# Restrained, print-safe palette (fill / edge pairs per functional group)
COL = {
    "input":     ("#EFF5FB", "#2C5F8A"),
    "morph":     ("#D6E6F5", "#1F4E79"),
    "rhythm":    ("#F7E0DE", "#9E3A32"),
    "fusion":    ("#E7E7E7", "#595959"),
    "output":    ("#DCEEE1", "#2E7D46"),
    "train":     ("#FBF1D6", "#B8860B"),
}


def draw_box(ax, center, width, height, text, group, fontsize=10.5, zorder=3):
    """Box with a subtle drop shadow -- restrained offset/alpha for a print
    (not slide-deck) look."""
    fc, ec = COL[group]

    shadow = patches.FancyBboxPatch(
        (center[0] - width / 2 + 0.045, center[1] - height / 2 - 0.045),
        width, height,
        boxstyle="round,pad=0.045,rounding_size=0.06",
        ec="none", fc="#000000", alpha=0.10, zorder=zorder - 1,
    )
    ax.add_patch(shadow)

    box = patches.FancyBboxPatch(
        (center[0] - width / 2, center[1] - height / 2),
        width, height,
        boxstyle="round,pad=0.045,rounding_size=0.06",
        linewidth=1.3, edgecolor=ec, facecolor=fc, zorder=zorder,
    )
    ax.add_patch(box)

    ax.text(center[0], center[1], text, ha="center", va="center",
             fontsize=fontsize, fontweight="bold", color="#1A1A1A",
             zorder=zorder + 1, linespacing=1.35)
    return box


def draw_arrow(ax, start, end, text=None, color="#333333", lw=1.5,
               style="-", fontsize=8.5):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=lw, color=color, ls=style,
                         mutation_scale=13, shrinkA=1, shrinkB=1),
        zorder=2,
    )
    if text:
        mid_x, mid_y = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.10, text, ha="center", va="bottom",
                 fontsize=fontsize, fontstyle="italic", color=color, zorder=4)


def draw_curved_arrow(ax, start, end, rad=0.3, text=None, color="#1F4E79"):
    ax.annotate(
        "", xy=end, xytext=start,
        arrowprops=dict(arrowstyle="-|>", lw=1.3, color=color, ls="-",
                         mutation_scale=11,
                         connectionstyle=f"arc3,rad={rad}"),
        zorder=2,
    )
    if text:
        ax.text((start[0] + end[0]) / 2, max(start[1], end[1]) + 0.62, text,
                 ha="center", va="bottom", fontsize=8, fontstyle="italic",
                 color=color, zorder=4)


def group_label(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="center", fontsize=9.5,
             fontweight="bold", color="#4D4D4D", style="italic")


def generate_workflow_figure(output_path):
    fig, ax = plt.subplots(figsize=(15.5, 7.6))
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.1)
    ax.axis("off")
    ax.set_aspect("equal")

    TOP_Y = 5.25
    BOT_Y = 2.05

    # ── Row group labels ─────────────────────────────────────────────────
    group_label(ax, 0.55, TOP_Y + 1.05, "Morphology Branch (beat waveform)")
    group_label(ax, 0.55, BOT_Y + 0.85, "Rhythm Branch (RR-interval history)")

    # ── 1. Inputs ────────────────────────────────────────────────────────
    draw_box(ax, (1.3, TOP_Y), 1.85, 0.62, "Raw ECG Signal\n(360 samples, 1 s)", "input", fontsize=9.5)
    draw_box(ax, (1.3, BOT_Y), 1.85, 0.68, "RR Intervals\n(2 Preceding + Current\n+ 2 Following)", "input", fontsize=9)

    # ── 2. Morphology branch: WavKAN -> BiGRU -> PWAM ───────────────────
    draw_box(ax, (3.75, TOP_Y), 2.05, 0.8, "WavKAN Backbone\n(PCWI-init., Mexican Hat)", "morph", fontsize=9.5)
    draw_box(ax, (6.15, TOP_Y), 1.85, 0.8, "BiGRU\n(Temporal Context)", "morph", fontsize=9.5)
    draw_box(ax, (8.55, TOP_Y), 2.0, 0.8, "PWAM\n(P-Wave Attention)", "morph", fontsize=9.5)

    draw_arrow(ax, (2.23, TOP_Y), (2.72, TOP_Y))
    draw_arrow(ax, (4.78, TOP_Y), (5.22, TOP_Y))
    draw_arrow(ax, (7.08, TOP_Y), (7.55, TOP_Y))

    draw_curved_arrow(ax, (1.75, TOP_Y + 0.33), (8.15, TOP_Y + 0.58), rad=-0.22,
                       text="raw signal (P-wave detail)")

    # ── 3. Rhythm branch ─────────────────────────────────────────────────
    draw_box(ax, (3.75, BOT_Y), 2.05, 0.8, "RR-Timing Encoder\n(MLP on RR-history)", "rhythm", fontsize=9.5)
    draw_arrow(ax, (2.23, BOT_Y), (2.72, BOT_Y))

    # ── 4. Fusion ─────────────────────────────────────────────────────────
    draw_box(ax, (11.0, 3.65), 1.0, 3.5, "Feature\nFusion", "fusion", fontsize=10.5)
    draw_arrow(ax, (9.55, TOP_Y), (10.5, 3.95), "morphology", color="#1F4E79")
    draw_arrow(ax, (4.78, BOT_Y), (10.5, 3.35), "rhythm", color="#9E3A32")

    # ── 5. Curriculum scheduler (training-only signal) ─────────────────
    draw_box(ax, (6.15, 0.62), 3.0, 0.62, "Curriculum Scheduler\n(training-only)", "train", fontsize=9.5)
    draw_arrow(ax, (7.65, 0.62), (12.75, 2.75), color="#B8860B", lw=1.6, style="--")
    ax.text(10.15, 1.45, "loss / sample re-weighting\n(no effect on inference)",
             color="#8A6900", fontsize=8, fontweight="bold", ha="center",
             bbox=dict(facecolor="white", alpha=0.92, edgecolor="none", pad=1.6),
             zorder=5)

    # ── 6. Output ─────────────────────────────────────────────────────────
    draw_box(ax, (12.75, 3.65), 1.55, 0.8, "Classifier\n(Softmax)", "output", fontsize=10)
    draw_arrow(ax, (11.55, 3.65), (11.98, 3.65))
    ax.text(14.55, 3.65, "Prediction\n(N, S, V, F, Q)", ha="left", va="center",
             fontsize=11.5, fontweight="bold", color="#1A1A1A")
    draw_arrow(ax, (13.55, 3.65), (14.25, 3.65))

    # ── Legend ────────────────────────────────────────────────────────────
    lx, ly = 0.55, 0.65
    ax.annotate("", xy=(lx + 0.55, ly), xytext=(lx, ly),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#333333"))
    ax.text(lx + 0.68, ly, "data flow (train + inference)", fontsize=8.3, va="center", color="#333333")
    ax.annotate("", xy=(lx + 0.55, ly - 0.32), xytext=(lx, ly - 0.32),
                arrowprops=dict(arrowstyle="-|>", lw=1.5, color="#B8860B", ls="--"))
    ax.text(lx + 0.68, ly - 0.32, "training-only signal", fontsize=8.3, va="center", color="#333333")

    # ── Title & framing box ─────────────────────────────────────────────
    fig.suptitle("PC-WavKAN Architecture  (153,045 params, MLP rhythm encoder)",
                 fontsize=15.5, y=0.975, fontweight="bold")

    rect = patches.FancyBboxPatch(
        (0.35, 0.12), 14.55, 6.75,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=1.4, edgecolor="#8C8C8C", facecolor="none", linestyle=(0, (5, 3)),
    )
    ax.add_patch(rect)
    ax.text(7.5, 0.30, "End-to-End Trainable Framework", ha="center",
             fontsize=9.5, color="#666666", fontstyle="italic")

    plt.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated workflow diagram at {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default="results/figures/final_methodology_workflow_v2.pdf",
                        help="Fixed 2026-09-02 (AUDIT_FINDINGS.md C21): added the "
                             "missing BiGRU + PWAM stages and corrected the RR-window "
                             "label from the false '5 Preceding, Causal' to the real "
                             "'2 Preceding + Current + 2 Following' (matching "
                             "src/process_data.py). Restyled the same day for "
                             "publication format: serif typography, a restrained "
                             "palette, a data-flow/training-signal legend, and an "
                             "explicit PCWI/Mexican-Hat label on the WavKAN backbone "
                             "box (previously named only in prose, not the figure).")
    args = parser.parse_args()
    generate_workflow_figure(args.output)
