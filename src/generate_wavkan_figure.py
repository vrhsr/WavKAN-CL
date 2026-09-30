"""
generate_wavkan_figure.py -- Fig. 2 of the manuscript: the wavelet-KAN edge
parameterisation (Submission_Array/manuscript.tex, \\label{fig:wavkan_micro}).

Redrawn 2026-09-30 (final pre-submission audit):

* the formula is exactly Eq. (2) of the manuscript and PCWIWavKANLinear.forward in
  src/wavkan_pcwi.py, including |gamma| and the lambda-weighted linear path, which the
  previous version omitted;
* the layer is drawn as a crossbar (inputs as rows, outputs collected down columns
  into summation nodes, one edge function at every crossing), so every edge (j, k) can
  be traced; previously the output wires ran through the other cells;
* a second panel plots one edge function against the input VALUE, marking mu and
  +/- gamma, because psi acts on the amplitude of a single input sample, not on a
  time window -- the unlabelled glyphs of the previous version invited the time-domain
  reading the manuscript explicitly rules out;
* drawn at its printed size (7-8 pt text at text width) instead of an 11-in canvas;
* the (mu, gamma) values are ILLUSTRATIVE and chosen only to make the edges visibly
  different; the caption says so. Trained dilations are about 0.05-0.12 (Fig. 4).

Usage:
    python src/generate_wavkan_figure.py --output Submission_Array/wavkan_micro_architecture_v2.pdf
"""
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, ConnectionPatch, FancyBboxPatch  # noqa: E402

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 7.5,
})

C_IN = ("#F4F6F8", "#4D5B6A")
C_OUT = ("#E5F2E7", "#2E7040")
C_CELL = ("#E3ECF6", "#2F5D8C")
C_WAVE = "#1F4E79"
C_WIRE = "#9A9A9A"

# Illustrative, visibly distinct parameters for the 3 x 3 schematic (rows j, columns k).
MU = np.array([[-0.35, 0.10, 0.30], [0.05, -0.30, 0.20], [0.25, -0.05, -0.35]])
GAMMA = np.array([[0.75, 1.10, 0.85], [1.25, 0.65, 0.80], [0.80, 1.15, 0.60]])
W = np.array([[1.0, -0.7, 0.8], [0.6, 1.0, 1.0], [-0.8, 0.7, 1.0]])   # sign/scale of w_{j,k}


def mexican_hat(u):
    return (1.0 - u ** 2) * np.exp(-0.5 * u ** 2)


def crossbar(ax):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7.8)
    ax.axis("off")
    rows = [5.55, 4.05, 2.55]           # input j
    cols = [3.55, 5.55, 7.55]           # output k
    cw, ch = 1.45, 1.0
    y_sum = 0.95

    # wires first (behind everything): input rows and output columns
    for y in rows:
        ax.plot([1.25, cols[-1]], [y, y], color=C_WIRE, lw=0.8, zorder=1)
    for x in cols:
        ax.plot([x, x], [rows[0], y_sum + 0.36], color=C_WIRE, lw=0.8, zorder=1)
        ax.annotate("", xy=(x, y_sum + 0.36), xytext=(x, y_sum + 0.9), zorder=1,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.35,head_width=0.18",
                                    color=C_WIRE, lw=0.8, shrinkA=0, shrinkB=0))

    # input nodes
    for j, y in enumerate(rows):
        ax.add_patch(Circle((0.9, y), 0.34, fc=C_IN[0], ec=C_IN[1], lw=0.9, zorder=3))
        ax.text(0.9, y, f"$x_{j + 1}$", ha="center", va="center", fontsize=8, zorder=4)

    # edge-function cells
    t = np.linspace(-2.4, 2.4, 140)
    for j, y in enumerate(rows):
        for k, x in enumerate(cols):
            hl = (j, k) == (1, 2)
            ax.add_patch(FancyBboxPatch((x - cw / 2, y - ch / 2), cw, ch,
                                        boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=C_CELL[0], ec=C_CELL[1], lw=1.6 if hl else 0.8, zorder=2))
            psi = W[j, k] * mexican_hat((t - MU[j, k]) / GAMMA[j, k])
            ax.plot(x + t * (cw * 0.19), y - 0.05 + psi * (ch * 0.30), color=C_WAVE, lw=1.1, zorder=3)
            ax.text(x - cw / 2 + 0.08, y + ch / 2 - 0.05, fr"$\phi_{{{j + 1}{k + 1}}}$",
                    ha="left", va="top", fontsize=6.6, color="#34495E", zorder=4)

    # summation nodes
    for k, x in enumerate(cols):
        ax.add_patch(Circle((x, y_sum), 0.34, fc=C_OUT[0], ec=C_OUT[1], lw=0.9, zorder=3))
        ax.text(x, y_sum, r"$\Sigma$", ha="center", va="center", fontsize=8.5, color=C_OUT[1], zorder=4)
        ax.text(x + 0.45, y_sum, fr"$z^{{({k + 1})}}$", ha="left", va="center", fontsize=8, zorder=4)

    ax.text(0.9, 6.05, "inputs $j$", ha="center", va="bottom", fontsize=7.2, color="#333333")
    ax.text(cols[1], 6.35, "edge functions  "
            r"$\phi_{jk}(x_j)=w_{jk}\,\psi((x_j-\mu_{jk})/|\gamma_{jk}|)$",
            ha="center", va="bottom", fontsize=7.8, color="#1A1A1A")
    ax.text(0.2, y_sum, "outputs $k$", ha="left", va="center", fontsize=7.2, color="#333333")
    ax.text(cols[1], 0.02, r"each output also adds the linear path $\lambda\,\Sigma_j\, v_{jk}x_j$",
            ha="center", va="bottom", fontsize=6.8, color="#555555", style="italic")
    return cols, rows, cw, ch


def anatomy(ax, j=1, k=2):
    """The highlighted crossbar cell, enlarged: the same mu, gamma and w, same x scale."""
    mu, gamma, w = MU[j, k], GAMMA[j, k], W[j, k]
    x = np.linspace(-2.4, 2.4, 400)
    ax.plot(x, w * mexican_hat((x - mu) / gamma), color=C_WAVE, lw=1.3)
    ax.axhline(0, color="#BBBBBB", lw=0.6, zorder=0)
    ax.axvline(mu, color="#555555", lw=0.7, ls=(0, (3, 2)))
    ax.text(mu + 0.08, 1.08, r"$\mu_{%d%d}$" % (j + 1, k + 1), fontsize=8, ha="left", va="center")
    for sgn in (-1, 1):
        ax.annotate("", xy=(mu + sgn * gamma, -0.62), xytext=(mu, -0.62),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.3,head_width=0.15", lw=0.7,
                                    color="#555555", shrinkA=0, shrinkB=0))
    ax.text(mu, -0.56, r"$\pm|\gamma_{%d%d}|$" % (j + 1, k + 1), fontsize=7.5, ha="center", va="bottom")
    ax.set_xlim(-2.4, 2.4)
    ax.set_ylim(-0.8, 1.25)
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ax.set_xlabel(r"input value $x_%d$ (signal amplitude)" % (j + 1), fontsize=7.2, labelpad=2)
    ax.tick_params(labelsize=6.5, width=0.6, length=2.5, pad=2)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_linewidth(0.6)
    ax.set_title(r"edge function $\phi_{%d%d}(x_%d)$, enlarged" % (j + 1, k + 1, j + 1), fontsize=7.6, pad=4)


def generate_wavkan_micro_figure(output_path):
    fig = plt.figure(figsize=(6.6, 3.2))
    ax = fig.add_axes([0.0, 0.0, 0.63, 1.0])
    cols, rows, cw, ch = crossbar(ax)
    ax2 = fig.add_axes([0.715, 0.25, 0.275, 0.55])
    anatomy(ax2)
    # dashed connector from the highlighted cell (row 2, column 3) to the panel
    fig.add_artist(ConnectionPatch(xyA=(cols[2] + cw / 2 + 0.05, rows[1]), coordsA=ax.transData,
                                   xyB=(0.0, 0.62), coordsB=ax2.transAxes,
                                   color="#999999", lw=0.7, ls=(0, (3, 2))))
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="Submission_Array/wavkan_micro_architecture_v2.pdf")
    generate_wavkan_micro_figure(ap.parse_args().output)
