"""
generate_wavkan_figure.py -- WavKAN micro-architecture figure (Fig. "Micro-
Architecture of WavKAN" in Submission_JBHI/ieee_manuscript_v2.tex).

Fixed 2026-09-02 (deep check + publication-format pass, project owner
request), two real content bugs found against the manuscript's own
Methodology equation (z_kan^(k) = sum_j w_{j,k} . psi((x_j - mu_{j,k}) /
gamma_{j,k}))):
  1. The formula box used sigma as the dilation symbol; the manuscript uses
     gamma throughout (Eq. 1, Fig. "wavelet bases" caption, the WAS section).
     Fixed to match.
  2. The old figure only drew a wavelet glyph on the 3 "diagonal" edges
     (x1->y1, x2->y2, x3->y3) and left the other 6 edges of this 3x3 example
     as plain background lines with no function shown at all -- this
     misrepresents a KAN layer, where EVERY edge (all j,k pairs) has its own
     independently-parameterized wavelet, not just a diagonal subset.
     Redrawn as an explicit 3x3 grid of distinct wavelet glyphs (one per
     (input j, output k) pair, each with a visibly different mu/gamma so the
     "independently learned per edge" claim is actually visible), matching
     the equation's own j,k indexing -- rows = inputs (j), columns = outputs
     (k). This is illustrative/generic (small 3x3 example, not the real
     360x64 layer), same as before -- only the per-edge claim was wrong, not
     the choice to illustrate with a small example.

Also fixed the old broken output path (a stray e:\\The\\... path that could
never have produced the real file, same bug class as
generate_workflow_diagram.py's pre-2026-08-27 default) and restyled to match
that figure's publication format (serif typography, restrained palette).
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
})

C_INPUT  = ("#EFF5FB", "#2C5F8A")
C_OUTPUT = ("#DCEEE1", "#2E7D46")
C_CELL   = ("#FDF3E7", "#B5651D")
C_EDGE   = "#B8B8B8"
C_WAVE   = "#C1651A"


def mexican_hat(t, mu, gamma):
    u = (t - mu) / gamma
    return (1 - u ** 2) * np.exp(-0.5 * u ** 2)


def draw_node(ax, xy, r, colors, label, label_side="left", inner=None):
    fc, ec = colors
    shadow = plt.Circle((xy[0] + 0.035, xy[1] - 0.035), r, color="#000000", alpha=0.10, zorder=2)
    ax.add_patch(shadow)
    circ = plt.Circle(xy, r, facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3)
    ax.add_patch(circ)
    if inner:
        ax.text(xy[0], xy[1], inner, ha="center", va="center", fontsize=13,
                 fontweight="bold", color=ec, zorder=4)
    dx = -(r + 0.18) if label_side == "left" else (r + 0.18)
    ha = "right" if label_side == "left" else "left"
    ax.text(xy[0] + dx, xy[1], label, ha=ha, va="center", fontsize=11.5,
             fontweight="bold", color="#1A1A1A")


def draw_wavelet_cell(ax, center, w, h, mu, gamma, cell_label):
    cx, cy = center
    shadow = patches.FancyBboxPatch(
        (cx - w / 2 + 0.03, cy - h / 2 - 0.03), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        ec="none", fc="#000000", alpha=0.08, zorder=2,
    )
    ax.add_patch(shadow)
    box = patches.FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.1, edgecolor=C_CELL[1], facecolor=C_CELL[0], zorder=3,
    )
    ax.add_patch(box)
    t = np.linspace(-2.6, 2.6, 120)
    y = mexican_hat(t, mu, gamma)
    t_scaled = cx + t * (w * 0.30)
    y_scaled = cy - h * 0.06 + y * (h * 0.34)
    ax.plot(t_scaled, y_scaled, color=C_WAVE, lw=1.7, zorder=4)
    ax.text(cx, cy + h / 2 - 0.10, cell_label, ha="center", va="top",
             fontsize=7.3, color="#7A4413", zorder=5, style="italic")


def generate_wavkan_micro_figure(output_path):
    fig, ax = plt.subplots(figsize=(11.5, 7.1))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.3)
    ax.axis("off")
    ax.set_aspect("equal")

    in_y = [4.6, 3.3, 2.0]
    out_y = [4.6, 3.3, 2.0]
    in_x, grid_x0, grid_x1, out_x = 1.1, 3.6, 8.0, 10.9

    cell_w, cell_h = 1.35, 0.95
    col_x = np.linspace(grid_x0, grid_x1, 3)

    # Illustrative, visibly-distinct (mu, gamma) per (input j, output k) cell
    # -- generic values, not real trained parameters (same convention as
    # before this fix; see module docstring).
    MU = [[-0.35, 0.10, 0.30], [0.05, -0.30, 0.40], [0.25, -0.05, -0.35]]
    GAMMA = [[0.75, 1.10, 0.85], [1.25, 0.65, 0.95], [0.80, 1.15, 0.60]]

    # ── Input nodes ──────────────────────────────────────────────────────
    for j, y in enumerate(in_y):
        draw_node(ax, (in_x, y), 0.32, C_INPUT, f"Input $x_{j+1}$", "left")

    # ── Output (summation) nodes ────────────────────────────────────────
    for k, y in enumerate(out_y):
        draw_node(ax, (out_x, y), 0.32, C_OUTPUT, f"$y_{k+1}$", "right", inner=r"$\Sigma$")

    # ── Edges: input j -> cell (j,k) -> output k, every pair, not just j=k ──
    for j, jy in enumerate(in_y):
        for k, kx in enumerate(col_x):
            ky = out_y[k]
            cell_center = (kx, jy)
            ax.plot([in_x + 0.32, kx - cell_w / 2], [jy, jy], color=C_EDGE, lw=1.0, zorder=1)
            ax.plot([kx + cell_w / 2, out_x - 0.32], [jy, ky], color=C_EDGE, lw=0.9, zorder=1)
            draw_wavelet_cell(ax, cell_center, cell_w, cell_h, MU[j][k], GAMMA[j][k],
                                fr"$\psi_{{{j+1},{k+1}}}$")

    # ── Column titles (drawn first, at a fixed row well clear of the grid) ──
    TITLE_Y = 5.55
    ax.text(in_x, TITLE_Y, "Input Layer", ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text((grid_x0 + grid_x1) / 2, TITLE_Y, "WavKAN Layer\n(per-edge wavelets, summed)",
             ha="center", va="center", fontsize=13, fontweight="bold")
    ax.text(out_x, TITLE_Y, "Output", ha="center", va="center", fontsize=13, fontweight="bold")

    # ── Formula callout (own row, clear of the titles above and the grid below) ──
    ax.text(5.8, 6.75,
             r"$\psi_{j,k}(x_j) = \mathrm{MexicanHat}\!\left(\dfrac{x_j - \mu_{j,k}}{\gamma_{j,k}}\right)$"
             r"$,\quad y_k = \sum_j w_{j,k}\,\psi_{j,k}(x_j)$",
             ha="center", va="center", fontsize=11.5, fontweight="bold",
             bbox=dict(facecolor="#FFF8E1", edgecolor="#B8860B", boxstyle="round,pad=0.4"))

    ax.text(5.8, 0.55,
             "Unlike an MLP, the learnable function sits on every edge $(j,k)$, not on the node --- "
             "each edge learns its own $(\\mu_{j,k},\\gamma_{j,k})$.",
             ha="center", va="center", fontsize=10.8, style="italic", color="#4D4D4D")

    plt.tight_layout()
    fig.savefig(output_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str,
                        default="Submission_Array/wavkan_micro_architecture_v2.pdf",
                        help="Fixed 2026-09-02: the old default was a broken path "
                             "(e:\\The\\...) that could never have produced the real "
                             "file, same bug class already fixed in "
                             "generate_workflow_diagram.py.")
    args = parser.parse_args()
    generate_wavkan_micro_figure(args.output)
