"""
wavelet_alignment_score.py  —  Quantitative Interpretability Metric

Wavelet-ECG Alignment Score (WAS)
===================================
Measures how much the learned wavelet parameters (μ, γ) align with known
ECG morphological priors after training.

Formula:
    WAS_c = 1 - (|μ_learned - μ_prior| / μ_range + |γ_learned - γ_prior| / γ_range) / 2

  where:
    μ_range = expected amplitude range for component c
    γ_range = expected scale range for component c

  WAS_c = 1.0 → perfect alignment after training
  WAS_c = 0.0 → complete drift from prior

This converts "we initialise with ECG priors" into a QUANTITATIVE, publishable
interpretability metric — something completely absent in all competing KAN/ECG papers.

Usage:
    python src/wavelet_alignment_score.py \\
        --checkpoint results/pca_model/best_model.pth \\
        --output-dir results/interpretability/

Outputs:
    was_scores.json           — WAS per component (pre- and post-training)
    pcwi_drift.json           — μ/γ drift per component
    wavelet_vis.pdf           — learned wavelet basis functions per group
    wavelet_heatmap.pdf       — edge weight heatmap (interpretability figure)
"""

import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.wavkan_pcwi import PCWIWavKANLinear, ECG_PRIORS
from models.wavkan_v2 import WavKAN_v2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────────────────────────────────────
# WAS Computation
# ─────────────────────────────────────────────────────────────────────────────

# Expected ranges for normalisation
_MU_RANGE    = 0.40   # typical normalised amplitude span [-0.2, +0.2]
_GAMMA_RANGE = 0.20   # typical scale span [0.03, 0.23]


def compute_was(kan_layer: PCWIWavKANLinear) -> dict:
    """
    Computes the Wavelet-ECG Alignment Score for each ECG component.

    Args:  kan_layer — the PCWIWavKANLinear instance from a trained model
    Returns: dict with per-component WAS [0,1] and drift values
    """
    if kan_layer.out_features != 64:
        return {"error": "WAS requires 64-output PCWI layer"}

    drift   = kan_layer.pcwi_drift()
    mu_dict = kan_layer.get_component_translations()
    ga_dict = kan_layer.get_component_scales()
    result  = {}

    for comp, prior in ECG_PRIORS.items():
        mu_drift    = drift[comp]["mu_drift"]
        gamma_drift = drift[comp]["gamma_drift"]

        mu_err    = mu_drift    / (_MU_RANGE    + 1e-8)
        gamma_err = gamma_drift / (_GAMMA_RANGE + 1e-8)

        was = max(0.0, 1.0 - (mu_err + gamma_err) / 2.0)

        result[comp] = {
            "was":            round(float(was), 4),
            "mu_prior":       prior["mu_center"],
            "mu_learned":     round(float(mu_dict[comp]), 4),
            "mu_drift":       round(float(mu_drift), 4),
            "gamma_prior":    prior["gamma"],
            "gamma_learned":  round(float(ga_dict[comp]), 4),
            "gamma_drift":    round(float(gamma_drift), 4),
        }

    # Macro WAS
    result["macro_was"] = round(float(np.mean([v["was"] for v in result.values()
                                               if isinstance(v, dict) and "was" in v])), 4)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation: Learned Wavelet Bases
# ─────────────────────────────────────────────────────────────────────────────

def plot_learned_wavelets(kan_layer: PCWIWavKANLinear, save_path: str = None):
    """
    Plots example learned wavelet functions per ECG component group.
    Shows first 4 channels from each group, compared to prior.
    """
    x   = torch.linspace(-3, 3, 300)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    component_colors = {"QRS": "#e15759", "P": "#4e79a7", "T": "#59a14f"}

    for ax, (comp, prior) in zip(axes, ECG_PRIORS.items()):
        c0, c1 = prior["channels"]
        color  = component_colors[comp]
        mu_pr  = prior["mu_center"]
        gm_pr  = prior["gamma"]

        # Prior wavelet (dashed)
        x_norm_prior = (x - mu_pr) / gm_pr
        phi_prior = (1.0 - x_norm_prior ** 2) * torch.exp(-0.5 * x_norm_prior ** 2)
        ax.plot(x.numpy(), phi_prior.numpy(), "k--", lw=1.5, alpha=0.6, label="Prior")

        # Learned wavelets (sample channels)
        with torch.no_grad():
            for ch_idx in list(range(c0, min(c0 + 5, c1))):
                # Use mean μ/γ for this channel across features
                mu_ch = kan_layer.translation.data[ch_idx].mean().item()
                gm_ch = kan_layer.scale.data[ch_idx].abs().mean().item() + 1e-6
                x_norm = (x - mu_ch) / gm_ch
                phi = (1.0 - x_norm ** 2) * torch.exp(-0.5 * x_norm ** 2)
                ax.plot(x.numpy(), phi.detach().numpy(), color=color, lw=1.0, alpha=0.55)

        ax.axvline(mu_pr, color="gray", lw=0.8, ls=":")
        ax.set_title(f"{comp} Component", fontsize=11, fontweight="bold")
        ax.set_xlabel("Normalised Amplitude"); ax.set_ylabel("φ(x)")
        ax.legend(fontsize=8); ax.grid(alpha=0.25)

    fig.suptitle("Learned Wavelet Basis Functions per ECG Component (PCWI)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_edge_weight_heatmap(kan_layer: PCWIWavKANLinear, save_path: str = None):
    """
    Visualises the edge weight matrix W (out_features × in_features).
    Rows grouped by ECG component for interpretability.
    """
    with torch.no_grad():
        W = kan_layer.weights.data.cpu().numpy()   # (64, 360)

    fig, ax = plt.subplots(figsize=(14, 4))
    im = ax.imshow(W, aspect="auto", cmap="RdBu_r",
                   vmin=-np.percentile(np.abs(W), 98),
                   vmax=np.percentile(np.abs(W), 98))
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)

    # Mark component boundaries
    ax.axhline(31.5, color="white", lw=1.5, ls="--")
    ax.axhline(47.5, color="white", lw=1.5, ls="--")
    ax.text(5,  16, "QRS",  color="white", fontsize=9, fontweight="bold")
    ax.text(5,  40, "P",   color="white", fontsize=9, fontweight="bold")
    ax.text(5,  56, "T",   color="white", fontsize=9, fontweight="bold")

    # Mark approximate ECG regions on x-axis (samples)
    for sample, label in [(80, "P-on"), (120, "P-off"), (165, "Q"), (180, "R"), (195, "S"), (237, "T-peak")]:
        ax.axvline(sample, color="white", lw=0.7, alpha=0.5)
        ax.text(sample, 63.5, label, color="white", fontsize=6, ha="center", rotation=90, va="top")

    ax.set_xlabel("ECG Sample (360 Hz, R-peak at ~180)", fontsize=11)
    ax.set_ylabel("Output Channel (grouped by ECG component)", fontsize=11)
    ax.set_title("WavKAN Edge Weight Heatmap  (Structural Interpretability)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────

def analyse_interpretability(
    checkpoint: str,
    output_dir: str = "results/interpretability",
    use_pcwi:   bool = True,
    use_pwam:   bool = True,
    use_rr_attn:bool = True,
) -> dict:
    """
    Full interpretability analysis of a trained WavKAN-v2 checkpoint.
    """
    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR = Path(output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Interpretability Analysis: {checkpoint}")
    print(f"{'='*60}")

    # Load model
    model = WavKAN_v2(use_pcwi=use_pcwi, use_pwam=use_pwam,
                      use_rr_attn=use_rr_attn).to(DEVICE)
    try:
        state = torch.load(checkpoint, map_location=DEVICE)
        model.load_state_dict(state)
    except Exception as e:
        # Fixed 2026-08-27: the emoji below crashed with UnicodeEncodeError on the
        # Windows cp1252 console, which raised a SECOND exception that masked the
        # real one above (a state_dict mismatch, e.g. wrong use_rr_attn flag) --
        # same class of bug already found and fixed once in export_quantize.py
        # (CHANGELOG.md 2026-08-14). A load failure here must be loud, never
        # silently swallowed into "randomly initialised model for demonstration"
        # -- reporting WAS for an untrained model as if it were the real result
        # would be exactly the kind of fabrication rule 1 forbids. Re-raise instead.
        print(f"Could not load checkpoint: {e!r}")
        raise

    model.eval()
    kan_layer = model.kan

    # Compute WAS
    was_scores = compute_was(kan_layer)
    print(f"\n  Wavelet-ECG Alignment Scores:")
    for comp in ("QRS", "P", "T"):
        s = was_scores[comp]
        print(f"    {comp:4s}: WAS={s['was']:.4f}  "
              f"μ: {s['mu_prior']} → {s['mu_learned']} (drift={s['mu_drift']:.4f})  "
              f"γ: {s['gamma_prior']} → {s['gamma_learned']} (drift={s['gamma_drift']:.4f})")
    print(f"  Macro WAS: {was_scores['macro_was']:.4f}")

    # Save scores
    with open(OUT_DIR / "was_scores.json", "w") as f:
        json.dump(was_scores, f, indent=2)
    print(f"\n  Saved was_scores.json")

    # Figures
    plot_learned_wavelets(kan_layer, save_path=str(OUT_DIR / "learned_wavelets.pdf"))
    plot_edge_weight_heatmap(kan_layer, save_path=str(OUT_DIR / "edge_weight_heatmap.pdf"))

    print(f"\n✅ Interpretability analysis complete → {OUT_DIR}/")
    return was_scores


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",  type=str, required=True)
    parser.add_argument("--output-dir",  type=str, default="results/interpretability")
    parser.add_argument("--no-pcwi",     action="store_true")
    parser.add_argument("--no-pwam",     action="store_true")
    parser.add_argument("--no-rr-attn",  action="store_true",
                        help="Match a checkpoint trained with use_rr_attn=False "
                             "(e.g. results/ablation_no_rr_attn/*) -- added 2026-08-27, "
                             "this flag was previously missing so this script could only "
                             "ever load a use_rr_attn=True checkpoint, silently failing "
                             "(and, due to the encoding bug fixed below, silently masked) "
                             "on any other config.")
    args = parser.parse_args()

    analyse_interpretability(
        checkpoint  = args.checkpoint,
        output_dir  = args.output_dir,
        use_pcwi    = not args.no_pcwi,
        use_pwam    = not args.no_pwam,
        use_rr_attn = not args.no_rr_attn,
    )
