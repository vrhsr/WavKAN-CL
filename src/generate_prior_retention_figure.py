"""
generate_prior_retention_figure.py -- Fig. 4 of the manuscript: per-edge wavelet parameters
after training, against the PCWI prior values and a trained null
(Submission_Array/manuscript.tex, \\label{fig:wavelets}).

Redrawn 2026-09-30 (final pre-submission audit). The previous version drew, for one seed,
one curve per channel from that channel's MEAN translation and MEAN |dilation| over its 360
input edges, and labelled those curves "trained edges". They were not edges. Averaging over
edges is exactly the aggregation the manuscript shows to be degenerate: per-edge translation
drift largely cancels in the mean (mean per-edge |dmu| 0.064 against 0.004 for block means),
so the curves hugged the priors and visually understated the movement Table 8 reports.

This version shows the actual per-edge parameters:
* rows are the three channel blocks (named for the prior value each was initialised at, not
  for a function; the channels are exchangeable, AUDIT_FINDINGS.md H62), columns are the
  translation mu and the dilation |gamma|;
* coloured histograms pool every edge of the block over all 20 seeds of the evaluated
  configuration (results/ablation_no_rr_attn);
* the grey outline is the trained isotropic-initialisation arm (results/ablation_no_pcwi),
  all 64 channels, i.e. the "trained, isotropic" null of Table 8;
* the dashed line is the block's prior value and the shaded band its initial PCWI range.
Parameters are read from the saved state_dicts; nothing is simulated. Drawn at print size.

Usage:
    python src/generate_prior_retention_figure.py --output Submission_Array/final_learned_wavelets.pdf
"""
import argparse
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.wavkan_pcwi import ECG_PRIORS  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
    "axes.linewidth": 0.6,
})

# Blocks are named for the prior value they were initialised with, not for a function: the
# 64 output channels are exchangeable downstream (AUDIT_FINDINGS.md H62).
BLOCKS = [("QRS", "QRS-prior block\n(ch. 1–32)", "#B2182B"),
          ("P", "P-prior block\n(ch. 33–48)", "#2166AC"),
          ("T", "T-prior block\n(ch. 49–64)", "#1A7A3E")]


def load_params(arm):
    files = sorted(glob.glob(os.path.join(arm, "seed_*", "best_model.pth")))
    if not files:
        raise FileNotFoundError(f"no seed_*/best_model.pth under {arm}")
    mu, ga = [], []
    for f in files:
        sd = torch.load(f, map_location="cpu")
        mu.append(sd["kan.translation"].numpy())
        ga.append(np.abs(sd["kan.scale"].numpy()))
    return np.array(mu), np.array(ga), len(files)


def generate(arm, null_arm, output_path):
    M, G, n = load_params(arm)
    Mn, Gn, n_null = load_params(null_arm)
    fig, axes = plt.subplots(3, 2, figsize=(6.3, 4.4), sharex="col")
    bins_mu = np.linspace(-0.45, 0.45, 91)
    bins_ga = np.linspace(0.0, 0.45, 91)
    for r, (key, name, col) in enumerate(BLOCKS):
        p = ECG_PRIORS[key]
        c0, c1 = p["channels"]
        for c, (vals, null, bins, centre, half, xlab) in enumerate([
                (M[:, c0:c1].ravel(), Mn.ravel(), bins_mu, p["mu_center"], p["mu_noise"],
                 r"translation $\mu_{jk}$"),
                (G[:, c0:c1].ravel(), Gn.ravel(), bins_ga, p["gamma"], p["gamma_jitter"],
                 r"dilation $|\gamma_{jk}|$")]):
            ax = axes[r, c]
            ax.axvspan(centre - half, centre + half, color="#DDDDDD", zorder=0, lw=0)
            ax.hist(null, bins=bins, density=True, histtype="step", color="#7F7F7F", lw=0.9, zorder=2)
            ax.hist(vals, bins=bins, density=True, color=col, alpha=0.55, lw=0, zorder=1)
            ax.axvline(centre, color="#1A1A1A", lw=0.9, ls=(0, (4, 2)), zorder=3)
            ax.set_yticks([])
            for sp in ("top", "right", "left"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(labelsize=6.8, width=0.6, length=2.5, pad=2)
            if r == 2:
                ax.set_xlabel(xlab, fontsize=7.8, labelpad=2)
        axes[r, 0].set_ylabel(name, fontsize=7.4, rotation=0, ha="right", va="center", labelpad=4, color=col)

    block_patches = tuple(plt.Rectangle((0, 0), 1, 1, fc=b[2], alpha=0.55, lw=0) for b in BLOCKS)
    handles = [block_patches,
               plt.Line2D([], [], color="#7F7F7F", lw=0.9),
               plt.Line2D([], [], color="#1A1A1A", lw=0.9, ls=(0, (4, 2))),
               plt.Rectangle((0, 0), 1, 1, fc="#DDDDDD", lw=0)]
    labels = [f"trained PC-WavKAN, every edge of the block ({n} seeds)",
              f"trained with isotropic initialisation, all channels ({n_null} seeds)",
              "prior value of the block", "initial PCWI range"]
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, fontsize=6.9,
               bbox_to_anchor=(0.56, 1.02), handlelength=2.4, columnspacing=1.5,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.6, w_pad=1.2)
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Saved -> {output_path} ({n} seeds; null {n_null} seeds)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="results/ablation_no_rr_attn")
    ap.add_argument("--null-arm", default="results/ablation_no_pcwi")
    ap.add_argument("--output", default="Submission_Array/final_learned_wavelets.pdf")
    a = ap.parse_args()
    generate(a.arm, a.null_arm, a.output)
