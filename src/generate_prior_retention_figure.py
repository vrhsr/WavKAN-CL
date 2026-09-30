"""
generate_prior_retention_figure.py -- single-column figure of the trained wavelet
bases against their PCWI priors.

Reads a real trained checkpoint's wavelet parameters and plots, per ECG channel
group, a sample of the learned Mexican Hat basis functions against the prior the
group was initialised from (see wavkan_pcwi.ECG_PRIORS). Nothing is simulated:
every curve is drawn from parameters read out of the checkpoint's state_dict.

This replaces the wide 1x3 layout produced by wavelet_alignment_score.py's
plot_learned_wavelets(), which is illegible when scaled into a single IEEE
column. The layout here is 3x1 (stacked) and sized for \\columnwidth, and it
additionally annotates each panel with that group's measured Physiological
Prior Retention so the figure and Table VIII cannot drift apart.

Usage:
    python src/generate_prior_retention_figure.py \
        --checkpoint results/ablation_no_rr_attn/seed_42/best_model.pth \
        --no-rr-attn \
        --output Submission_Array/final_learned_wavelets.pdf
"""

import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.wavkan_pcwi import ECG_PRIORS  # noqa: E402
from models.wavkan_v2 import WavKAN_v2  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Normalising spans of the PPR definition -- must match
# wavelet_alignment_score.py so the annotation agrees with the reported table.
MU_RANGE, GAMMA_RANGE = 0.40, 0.20

PALETTE = {"QRS": "#b2182b", "P": "#2166ac", "T": "#1a7a3e"}
# Blocks are named for the prior value they were initialised with, not for a
# function: the 64 output channels are exchangeable downstream, so no block can be
# said to compute a waveform component (AUDIT_FINDINGS.md H62).
LABEL = {"QRS": "QRS-prior block (ch. 1-32)",
         "P": "P-prior block (ch. 33-48)",
         "T": "T-prior block (ch. 49-64)"}


def mexican_hat(x):
    return (1.0 - x ** 2) * np.exp(-0.5 * x ** 2)


def group_ppr(kan, comp):
    """Per-edge Parameter-space Prior Retention (PPR) for one channel block."""
    prior = ECG_PRIORS[comp]
    c0, c1 = prior["channels"]
    mu = kan.translation.data[c0:c1]
    gamma = kan.scale.data[c0:c1].abs()
    e_mu = (mu - prior["mu_center"]).abs().mean().item()
    e_ga = (gamma - prior["gamma"]).abs().mean().item()
    return max(0.0, 1.0 - (e_mu / MU_RANGE + e_ga / GAMMA_RANGE) / 2.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--output", default="Submission_Array/final_learned_wavelets.pdf")
    ap.add_argument("--no-pcwi", action="store_true")
    ap.add_argument("--no-pwam", action="store_true")
    ap.add_argument("--no-rr-attn", action="store_true",
                    help="Match a checkpoint trained with use_rr_attn=False.")
    ap.add_argument("--n-curves", type=int, default=6,
                    help="Learned edges sampled per group (kept low for legibility).")
    args = ap.parse_args()

    model = WavKAN_v2(use_pcwi=not args.no_pcwi,
                      use_pwam=not args.no_pwam,
                      use_rr_attn=not args.no_rr_attn)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    model.eval()
    kan = model.kan

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "dejavuserif",
        "axes.linewidth": 0.6,
    })

    x = np.linspace(-0.45, 0.45, 800)
    fig, axes = plt.subplots(3, 1, figsize=(3.4, 3.95), sharex=True)

    for ax, comp in zip(axes, ("QRS", "P", "T")):
        prior = ECG_PRIORS[comp]
        c0, c1 = prior["channels"]
        colour = PALETTE[comp]

        # Prior wavelet for this group
        phi_prior = mexican_hat((x - prior["mu_center"]) / prior["gamma"])
        ax.plot(x, phi_prior, color="0.15", lw=1.5, ls="--", zorder=3,
                label="initial prior")

        # A sample of the trained edges, one curve per output channel
        step = max(1, (c1 - c0) // args.n_curves)
        for ch in range(c0, c1, step):
            mu = kan.translation.data[ch].mean().item()
            gamma = kan.scale.data[ch].abs().mean().item() + 1e-6
            ax.plot(x, mexican_hat((x - mu) / gamma), color=colour, lw=0.9,
                    alpha=0.75, zorder=2)
        ax.plot([], [], color=colour, lw=1.2, label="trained edges")

        ax.axvline(prior["mu_center"], color="0.55", lw=0.6, ls=":", zorder=1)
        ax.set_ylabel(r"$\psi(x)$", fontsize=8.5)
        ax.set_ylim(-0.55, 1.18)
        ax.tick_params(labelsize=7.5, width=0.6, length=2.5)
        ax.grid(alpha=0.18, lw=0.5)
        ax.text(0.015, 0.955,
                "%s   PPR $=%.3f$" % (LABEL[comp], group_ppr(kan, comp)),
                transform=ax.transAxes, fontsize=7.8, va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="0.75", lw=0.5))
        ax.legend(fontsize=6.8, loc="upper right", frameon=False,
                  handlelength=1.5, borderaxespad=0.2)

    axes[-1].set_xlabel("Normalised signal amplitude $x$", fontsize=8.5)
    fig.align_ylabels(axes)
    fig.tight_layout(pad=0.35, h_pad=0.5)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print("wrote %s" % args.output)
    for comp in ("QRS", "P", "T"):
        print("  %-4s PPR = %.4f" % (comp, group_ppr(kan, comp)))


if __name__ == "__main__":
    main()
