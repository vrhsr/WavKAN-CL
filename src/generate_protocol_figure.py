"""
generate_protocol_figure.py -- evaluation-protocol flowchart for the manuscript
(Submission_Array/manuscript.tex, \\label{fig:protocol}).

Added 2026-09-30. Shows which data each step uses: training records for fitting,
validation records for every checkpoint and configuration decision, DS2 for a single
evaluation, and the external databases for forward passes only.

Every count is read from a file, not typed: MIT-BIH partitions from
configs/mitbih_split_counts.json (derived from the PhysioNet annotations), external
beat counts from results/external_matched/*/report.json. Drawn at print size.

Usage:
    python src/generate_protocol_figure.py --output Submission_Array/evaluation_protocol.pdf
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
})

COL = {
    "src":   ("#F4F6F8", "#4D5B6A"),
    "train": ("#E3ECF6", "#2F5D8C"),
    "val":   ("#FDF3E1", "#A07000"),
    "test":  ("#E5F2E7", "#2E7040"),
    "ext":   ("#F3ECF7", "#6A4C93"),
    "step":  ("#FFFFFF", "#555555"),
}
ARROW = "#2B2B2B"


def box(ax, x0, y0, w, h, kind, lines, fs=7.2, dashed=False):
    fc, ec = COL[kind]
    ax.add_patch(FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                                fc=fc, ec=ec, lw=0.9, ls=(0, (4, 2.5)) if dashed else "-", zorder=2))
    n = len(lines)
    step = min(h / (n + 0.5), 3.1)
    top = y0 + h / 2 + step * (n - 1) / 2
    for i, t in enumerate(lines):
        ax.text(x0 + w / 2, top - i * step, t, ha="center", va="center", fontsize=fs,
                fontweight="bold" if i == 0 else "normal", color="#1A1A1A", zorder=3)


def arrow(ax, p0, p1, dashed=False, color=ARROW):
    ax.annotate("", xy=p1, xytext=p0, zorder=1,
                arrowprops=dict(arrowstyle="-|>,head_length=0.45,head_width=0.22", lw=0.9, color=color,
                                ls=(0, (4, 2.5)) if dashed else "-", shrinkA=0, shrinkB=0))


def generate(output_path):
    gt = json.load(open("configs/mitbih_split_counts.json", encoding="utf-8"))
    n_rec = {s: len(gt["records"][s]) for s in ("train", "val", "test")}
    n_beat = gt["totals"]
    ext = {d: json.load(open(f"results/external_matched/{d}/report.json"))["n_beats"] for d in ("incart", "svdb")}

    fig, ax = plt.subplots(figsize=(7.2, 4.1))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 57)
    ax.axis("off")

    XS0, XS1 = 0.5, 17.0          # source
    XP0, XP1 = 22.0, 41.5         # partitions
    XT0, XT1 = 47.0, 75.0         # steps
    XR0, XR1 = 80.0, 99.5         # statistics
    H = 9.0
    YT, YV, YD = 46.0, 33.0, 20.0  # training, validation, DS2 rows (box bottoms)

    # ── source ──────────────────────────────────────────────────────────────
    box(ax, XS0, 30.0, XS1 - XS0, 15.0, "src",
        ["MIT-BIH", "Arrhythmia Database", f"{sum(n_rec.values())} non-paced records", "channel 0, 360 Hz"])

    # ── partitions ──────────────────────────────────────────────────────────
    for key, y, name in [("train", YT, "DS1 training"), ("val", YV, "DS1 validation"),
                         ("test", YD, "DS2 test (held out)")]:
        box(ax, XP0, y, XP1 - XP0, H, key, [name, f"{n_rec[key]} records · {n_beat[key]:,} beats"])
        arrow(ax, (XS1 + 0.3, 37.5), (XP0 - 0.3, y + H / 2))
        arrow(ax, (XP1 + 0.3, y + H / 2), (XT0 - 0.3, y + H / 2))

    # ── steps ───────────────────────────────────────────────────────────────
    box(ax, XT0, YT, XT1 - XT0, H, "step", ["Training", "5 models × 20 seeds, one protocol"])
    box(ax, XT0, YV, XT1 - XT0, H, "step", ["Model selection", "best validation Macro-F1"])
    box(ax, XT0, YD, XT1 - XT0, H, "step", ["Single evaluation on DS2", "primary outcome: Macro-F1"])
    xm = (XT0 + XT1) / 2
    arrow(ax, (xm, YT - 0.3), (xm, YV + H + 0.3))
    arrow(ax, (xm, YV - 0.3), (xm, YD + H + 0.3))
    ax.text(xm + 1.0, (YV + YD + H) / 2, "selected models", fontsize=6.6, color="#444444",
            va="center", style="italic")

    # ── statistics ──────────────────────────────────────────────────────────
    box(ax, XR0, YD, XR1 - XR0, 22.0, "step",
        ["Paired statistics", "paired by seed", "two-sided Wilcoxon", "Holm correction",
         r"effect size $d_z$", "95% confidence intervals"], fs=7.0)
    arrow(ax, (XT1 + 0.3, YD + H / 2), (XR0 - 0.3, YD + H / 2))

    # ── external and sensitivity (forward pass only) ────────────────────────
    box(ax, XS0, 1.0, XP1 - XS0, 14.0, "ext",
        ["External databases", "never used for training or selection",
         f"INCART: 75 records · {ext['incart']:,} beats (lead II)",
         f"SVDB: 78 records · {ext['svdb']:,} beats",
         "resampled to 360 Hz; same extraction code"], fs=6.9)
    box(ax, XT0, 1.0, XT1 - XT0, 14.0, "step",
        ["Forward pass only", "cross-database evaluation",
         "RR-feature sensitivity on DS2"], fs=7.0, dashed=True)
    arrow(ax, (XP1 + 0.3, 8.0), (XT0 - 0.3, 8.0), dashed=True)
    ax.plot([xm, xm], [YD - 0.3, 15.4], color=ARROW, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
    xr = (XR0 + XR1) / 2
    ax.plot([XT1 + 0.3, xr], [8.0, 8.0], color=ARROW, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
    arrow(ax, (xr, 8.0), (xr, YD - 0.3), dashed=True)

    # ── legend ──────────────────────────────────────────────────────────────
    ax.plot([XR0, XR0 + 3.5], [53.5, 53.5], color=ARROW, lw=0.9)
    ax.text(XR0 + 4.3, 53.5, "training, selection,\nevaluation", fontsize=6.6, va="center")
    ax.plot([XR0, XR0 + 3.5], [47.5, 47.5], color=ARROW, lw=0.9, ls=(0, (4, 2.5)))
    ax.text(XR0 + 4.3, 47.5, "forward pass of\nselected models", fontsize=6.6, va="center")

    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved -> {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="Submission_Array/evaluation_protocol.pdf")
    generate(ap.parse_args().output)
