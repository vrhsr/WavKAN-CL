"""
generate_forest_plot.py -- paired Macro-F1 differences, PC-WavKAN minus each baseline, on
MIT-BIH DS2 and the two external databases (Submission_Array/manuscript.tex,
\\label{fig:forest}).

Added 2026-09-30. Puts the primary comparison (Table tab:baseline_comparison) and the
cross-database comparison (Table tab:multidataset) on one scale. Statistics come from the
canonical src/paired_stats.py with the same families as those tables: seed-paired
two-sided Wilcoxon, Holm across the four baselines within each database, and t-based 95%
confidence intervals of the paired mean difference.

Data: DS2 from each arm's per-seed test_metrics.json (the published values); INCART and
SVDB from results/external_matched/*/per_seed.json (matched-pipeline evaluation).

Usage:
    python src/generate_forest_plot.py --output Submission_Array/macro_f1_forest.pdf
"""
import argparse
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import paired_stats as ps  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.linewidth": 0.6,
})

ARMS = {"PC-WavKAN": "results/ablation_no_rr_attn", "ResNet1D": "results/baseline_resnet1d",
        "Transformer": "results/baseline_transformer", "CNN+Focal": "results/baseline_cnn_focal",
        "B-Spline KAN": "results/baseline_bspline_kan"}
BASELINES = ["ResNet1D", "Transformer", "CNN+Focal", "B-Spline KAN"]
DATABASES = [("ds2", "MIT-BIH DS2\n(in distribution)", "#2F5D8C"),
             ("incart", "INCART\n(external)", "#6A4C93"),
             ("svdb", "SVDB\n(external)", "#9A3B30")]


def ds2_scores(arm):
    out = {}
    for f in glob.glob(os.path.join(arm, "seed_*", "test_metrics.json")):
        out[os.path.basename(os.path.dirname(f)).split("_")[1]] = json.load(open(f))["macro_f1"]
    return out


def external_scores(db):
    per_seed = json.load(open(f"results/external_matched/{db}/per_seed.json"))
    return {m: {s: v["macro_f1"] for s, v in d.items()} for m, d in per_seed.items()}


def families():
    fams = {}
    ds2 = {m: ds2_scores(a) for m, a in ARMS.items()}
    for db, _, _ in DATABASES:
        sc = ds2 if db == "ds2" else external_scores(db)
        fam = {b: ps.paired_compare(sc["PC-WavKAN"], sc[b], "macro_f1") for b in BASELINES}
        ps.holm_family(fam)
        fams[db] = fam
    return fams


def generate(output_path):
    fams = families()
    fig, ax = plt.subplots(figsize=(5.2, 3.7))
    y, ticks, labels, group_mid = 0, [], [], []
    for db, gname, col in DATABASES:
        ys = []
        for b in BASELINES:
            r = fams[db][b]
            sig = r["holm_p"] < 0.05
            ax.plot([r["ci_low"], r["ci_high"]], [y, y], color=col, lw=1.2, solid_capstyle="butt", zorder=2)
            ax.plot(r["mean_diff"], y, marker="o", ms=5.0, mec=col, mew=1.1,
                    mfc=col if sig else "white", zorder=3)
            ticks.append(y)
            labels.append(b)
            ys.append(y)
            y -= 1
        group_mid.append((sum(ys) / len(ys), gname, col))
        y -= 0.8
    ax.axvline(0, color="#555555", lw=0.8, ls=(0, (4, 2)), zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=7.2)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.tick_params(axis="x", labelsize=7, width=0.6, length=2.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.set_xlabel("Macro-F1 difference, PC-WavKAN minus baseline (95% CI)", fontsize=7.6, labelpad=3)
    lo = min(r["ci_low"] for f in fams.values() for r in f.values())
    hi = max(r["ci_high"] for f in fams.values() for r in f.values())
    pad = 0.012
    ax.set_xlim(lo - pad, hi + pad)
    for ym, gname, col in group_mid:
        ax.text(1.02, ym, gname, transform=ax.get_yaxis_transform(), fontsize=7.2, color=col,
                ha="left", va="center", fontweight="bold")
    ax.text(0.0, ticks[0] + 1.05, "← baseline better", transform=ax.get_xaxis_transform() if False else ax.transData,
            fontsize=6.6, ha="right", va="center", color="#555555")
    ax.text(0.0, ticks[0] + 1.05, "  PC-WavKAN better →", fontsize=6.6, ha="left", va="center", color="#555555")
    ax.set_ylim(ticks[-1] - 0.8, ticks[0] + 1.6)
    h_sig = plt.Line2D([], [], marker="o", ms=5, mfc="#333333", mec="#333333", lw=0)
    h_ns = plt.Line2D([], [], marker="o", ms=5, mfc="white", mec="#333333", lw=0)
    ax.legend([h_sig, h_ns], ["Holm-adjusted p < 0.05", "not significant"], loc="upper left",
              fontsize=6.6, frameon=False, handletextpad=0.3, borderaxespad=0.1)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved -> {output_path}")
    for db, fam in fams.items():
        for b, r in fam.items():
            print(f"  {db:6s} {b:12s} diff {r['mean_diff']:+.4f} CI [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}] "
                  f"Holm p {r['holm_p']:.3g}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="Submission_Array/macro_f1_forest.pdf")
    generate(ap.parse_args().output)
