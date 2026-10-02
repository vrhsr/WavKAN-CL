"""
generate_graphical_abstract.py -- graphical abstract for the Array revision (ARRAY-D-26-02633).

Added 2026-10-01; the handling editor requires a graphical abstract. Every number drawn is read
or computed from the released result files at generation time, never typed in:
  - Macro-F1 per model and database: DS2 from each arm's per-seed test_metrics.json, INCART and
    SVDB from results/external_matched/*/per_seed.json (matched-pipeline evaluation);
  - prior retention (PPR, Eq. 3 of the manuscript): computed from the 20 PC-WavKAN checkpoints,
    the 20 checkpoints of the no-PCWI ablation arm, and 20 untrained initialisations of each kind,
    with the same definition as src/verify_manuscript_numbers.py;
  - supraventricular recall: from the PC-WavKAN per-seed test_metrics.json.
The panel statements repeat conclusions stated in the manuscript.

Elsevier's minimum is 531 x 1328 px (h x w), readable at 5 x 13 cm; the output is 2.5:1 at 300 dpi.

Usage:
    python src/generate_graphical_abstract.py --output Submission_Array/graphical_abstract.pdf
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.linewidth": 0.6,
    "pdf.fonttype": 42,
})

ARMS = {"PC-WavKAN": "results/ablation_no_rr_attn", "ResNet1D": "results/baseline_resnet1d",
        "CNN+Focal": "results/baseline_cnn_focal", "Transformer": "results/baseline_transformer",
        "B-Spline KAN": "results/baseline_bspline_kan"}
# Okabe-Ito palette (colour-blind safe), with distinct markers as a second cue.
STYLE = {"PC-WavKAN": ("#D55E00", "o", 4.6), "ResNet1D": ("#0072B2", "s", 3.4),
         "CNN+Focal": ("#56B4E9", "D", 3.2), "Transformer": ("#009E73", "^", 3.6),
         "B-Spline KAN": ("#E69F00", "v", 3.6)}
DBS = [("ds2", "MIT-BIH\nDS2"), ("incart", "INCART"), ("svdb", "SVDB")]
MU_R, GA_R = 0.40, 0.20


def ds2_scores(arm, key="macro_f1"):
    return [json.load(open(f))[key] for f in sorted(glob.glob(os.path.join(arm, "seed_*", "test_metrics.json")))]


def external_scores(db):
    per_seed = json.load(open(f"results/external_matched/{db}/per_seed.json"))
    return {m: [v["macro_f1"] for v in d.values()] for m, d in per_seed.items()}


def eppr(kan):
    from src.wavkan_pcwi import ECG_PRIORS
    vals = []
    for p_ in ECG_PRIORS.values():
        c0, c1 = p_["channels"]
        emu = (kan.translation.data[c0:c1] - p_["mu_center"]).abs().mean().item()
        ega = (kan.scale.data[c0:c1].abs() - p_["gamma"]).abs().mean().item()
        vals.append(max(0.0, 1 - (emu / MU_R + ega / GA_R) / 2))
    return float(np.mean(vals))


def ppr_values():
    import torch
    from models.wavkan_v2 import WavKAN_v2

    def trained(pattern, **kw):
        out = []
        for ck in sorted(glob.glob(pattern)):
            m = WavKAN_v2(**kw)
            m.load_state_dict(torch.load(ck, map_location="cpu"))
            out.append(eppr(m.kan))
        return float(np.mean(out))

    def untrained(pcwi):
        out = []
        for sd in range(1000, 1020):
            torch.manual_seed(sd)
            out.append(eppr(WavKAN_v2(use_pcwi=pcwi, use_pwam=True, use_rr_attn=False).kan))
        return float(np.mean(out))

    return [("Untrained, prior-centred", untrained(True)),
            ("Trained, prior-centred", trained("results/ablation_no_rr_attn/seed_*/best_model.pth",
                                               use_pcwi=True, use_pwam=True, use_rr_attn=False)),
            ("Untrained, isotropic", untrained(False)),
            ("Trained, isotropic", trained("results/ablation_no_pcwi/seed_*/best_model.pth",
                                           use_pcwi=False, use_pwam=True, use_rr_attn=True))]


def card(ax, title):
    ax.set_axis_off()
    ax.add_patch(FancyBboxPatch((0.0, 0.0), 1.0, 1.0, boxstyle="round,pad=0.0,rounding_size=0.04",
                                transform=ax.transAxes, fc="#F4F6F8", ec="#9AA5B1", lw=0.7, zorder=0))
    ax.text(0.5, 0.93, title, transform=ax.transAxes, ha="center", va="top", fontsize=8.6,
            fontweight="bold", color="#1F2A36")


def generate(output):
    scores = {"ds2": {m: ds2_scores(a) for m, a in ARMS.items()},
              "incart": external_scores("incart"), "svdb": external_scores("svdb")}
    ppr = ppr_values()
    s_recall = float(np.mean(ds2_scores(ARMS["PC-WavKAN"], "s_recall")))

    fig = plt.figure(figsize=(6.6, 2.64))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.7, 1.05], wspace=0.2,
                          left=0.01, right=0.99, top=0.91, bottom=0.27)

    # Panel 1: what was done
    ax0 = fig.add_subplot(gs[0])
    card(ax0, "A controlled evaluation")
    lines = ["Wavelet KAN (153K parameters)", "vs. four baselines", "",
             "MIT-BIH, inter-patient", "(de Chazal DS1/DS2)", "",
             "Same protocol and budget", "20 paired seeds per model", "",
             "Validation-based selection", "Holm-corrected paired tests"]
    y = 0.80
    for ln in lines:
        if ln:
            ax0.text(0.5, y, ln, transform=ax0.transAxes, ha="center", va="center", fontsize=7.2,
                     color="#26323F")
        y -= 0.062 if ln else 0.035

    # Panel 2: Macro-F1 in and out of distribution
    ax1 = fig.add_subplot(gs[1])
    offs = np.linspace(-0.26, 0.26, len(ARMS))
    for k, (db, _) in enumerate(DBS):
        for off, m in zip(offs, ARMS):
            v = np.array(scores[db][m])
            col, mk, ms = STYLE[m]
            ax1.errorbar(k + off, v.mean(), yerr=v.std(ddof=1), fmt=mk, ms=ms, color=col, mec=col,
                         mfc=col, elinewidth=0.8, capsize=1.4, capthick=0.6, zorder=3 if m == "PC-WavKAN" else 2)
    ax1.set_xticks(range(len(DBS)))
    ax1.set_xticklabels([lbl for _, lbl in DBS], fontsize=7.2)
    ax1.set_xlim(-0.5, len(DBS) - 0.5)
    lo = min(np.mean(v) - np.std(v, ddof=1) for d in scores.values() for v in d.values())
    hi = max(np.mean(v) + np.std(v, ddof=1) for d in scores.values() for v in d.values())
    ax1.set_ylim(lo - 0.012, hi + 0.06)
    ax1.set_ylabel("Macro-F1 (mean $\\pm$ SD, 20 seeds)", fontsize=7.0, labelpad=2)
    ax1.tick_params(axis="both", labelsize=6.8, width=0.6, length=2.2)
    for sp in ("top", "right"):
        ax1.spines[sp].set_visible(False)
    ax1.axvline(0.5, color="#B8C0C8", lw=0.6, ls=(0, (3, 2)))
    tr = ax1.get_xaxis_transform()
    ax1.text(0.0, 0.985, "no detectable\ndifference", transform=tr, ha="center", va="top", fontsize=6.5,
             color="#26323F")
    ax1.text(1.5, 0.985, "below both\nCNN baselines", transform=tr, ha="center", va="top", fontsize=6.5,
             color="#26323F")
    ax1.set_title("Macro-F1", fontsize=8.6, fontweight="bold", color="#1F2A36", pad=3)
    handles = [plt.Line2D([], [], marker=STYLE[m][1], color=STYLE[m][0], ls="", ms=STYLE[m][2]) for m in ARMS]
    p1 = ax1.get_position()
    fig.legend(handles, list(ARMS), loc="lower center", ncol=3, fontsize=6.6, frameon=False,
               bbox_to_anchor=(0.5 * (p1.x0 + p1.x1), -0.015), handletextpad=0.15, columnspacing=0.8)

    # Panel 3: interpretability
    ax2 = fig.add_subplot(gs[2])
    labels = [l for l, _ in ppr][::-1]
    vals = [v for _, v in ppr][::-1]
    cols = ["#9AA5B1", "#C9D0D6", "#D55E00", "#F2B48A"]
    ax2.barh(range(4), vals, color=cols, height=0.7, edgecolor="none")
    for i, (v, lbl) in enumerate(zip(vals, labels)):
        ax2.text(v + 0.012, i, f"{v:.2f}", va="center", fontsize=6.8, color="#26323F")
        dark = cols[i] in ("#D55E00", "#9AA5B1")
        ax2.text(0.02, i, lbl, va="center", ha="left", fontsize=6.4, color="white" if dark else "#26323F")
    ax2.set_yticks([])
    ax2.set_xlim(0, 1.12)
    ax2.set_xticks([0, 0.5, 1.0])
    ax2.tick_params(axis="both", labelsize=6.6, width=0.6, length=2.0)
    ax2.set_xlabel("Retention of the prior values (PPR)", fontsize=6.8, labelpad=1.5)
    for sp in ("top", "right", "left"):
        ax2.spines[sp].set_visible(False)
    ax2.set_title("Interpretability: not supported", fontsize=8.6, fontweight="bold", color="#1F2A36", pad=3)
    p0, p2 = ax0.get_position(), ax2.get_position()
    fig.text(0.5 * (p2.x0 + p2.x1), 0.015, "Training moves randomly initialised\nwavelets away from the priors",
             ha="center", va="bottom", fontsize=6.4, color="#26323F")
    fig.text(0.5 * (p0.x0 + p0.x1), 0.015, f"Supraventricular recall is low\nfor all models (PC-WavKAN {s_recall:.2f})",
             ha="center", va="bottom", fontsize=6.4, color="#26323F")

    base, _ = os.path.splitext(output)
    fig.savefig(output)
    # TIFF as RGB without an alpha channel (some submission converters mishandle RGBA);
    # the rendered figure is fully opaque, so dropping alpha changes no pixel.
    import io
    from PIL import Image
    _buf = io.BytesIO()
    fig.savefig(_buf, format="png", dpi=300)
    _buf.seek(0)
    Image.open(_buf).convert("RGB").save(base + ".tiff", compression="tiff_lzw", dpi=(300, 300))
    fig.savefig(base + ".png", dpi=300)
    plt.close(fig)
    print(f"Saved -> {output}, {base}.tiff, {base}.png")
    for db in scores:
        print("  " + db + ": " + ", ".join(f"{m} {np.mean(v):.3f}" for m, v in scores[db].items()))
    print("  PPR: " + ", ".join(f"{l} {v:.3f}" for l, v in ppr))
    print(f"  PC-WavKAN S-recall {s_recall:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="Submission_Array/graphical_abstract.pdf")
    generate(ap.parse_args().output)
