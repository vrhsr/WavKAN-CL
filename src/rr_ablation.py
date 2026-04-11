"""
rr_ablation.py  —  Leave-One-Out RR-Interval Feature Ablation

Reproduces and EXTENDS Figure 6 from the original JBHI manuscript.
====================================================================
Original finding (old model): masking the t-3 interval causes the
largest drop in S-Recall (Δ = -0.062, p < 0.05).

This script re-validates that finding on WavKAN-v2 with PWAM.

Expected new finding:
  - PWAM adds P-wave morphological context → RR dependence may SHIFT
  - If PWAM reduces the t-3 effect, that proves PWAM genuinely provides
    complementary information (not just redundant with RR)
  - If t-3 effect persists, it validates the architectural necessity of
    both branches working in tandem

Output:
  results/rr_ablation/
    rr_ablation_report.json          Per-position delta metrics
    rr_ablation_figure.pdf           Publication figure (matches Fig 6 style)
    rr_ablation_table.tex            LaTeX table for paper
    rr_ablation_significance.txt     Wilcoxon test results

Usage:
    python src/rr_ablation.py \\
        --model-dir results/full_pipeline/wavkan_v2/ \\
        --seeds 42 101 777 2026 31415 \\
        --output-dir results/rr_ablation/
"""

import os, sys, json, argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy.stats import wilcoxon
from sklearn.metrics import recall_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.wavkan_v2 import WavKAN_v2

CLASS_NAMES  = ["N", "S", "V", "F", "Q"]
RR_POSITIONS = ["t-4", "t-3", "t-2", "t-1", "t"]   # indices 0-4 in the 5-beat window
TARGET_CLASS = 1   # S-class (supraventricular)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset (identical to train_pca.py for compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class ECGDatasetRR(Dataset):
    def __init__(self, split: str, data_dir: str):
        base     = Path(data_dir)
        self.X   = torch.tensor(np.load(base / f"X_{split}.npy"),    dtype=torch.float32)
        self.Xrr = torch.tensor(np.load(base / f"X_rr_{split}.npy"), dtype=torch.float32)
        self.y   = torch.tensor(np.load(base / f"y_{split}.npy"),    dtype=torch.long)
    def __len__(self):           return len(self.y)
    def __getitem__(self, i):    return self.X[i], self.Xrr[i], self.y[i]


# ─────────────────────────────────────────────────────────────────────────────
# Inference with one RR position masked
# ─────────────────────────────────────────────────────────────────────────────

def inference_with_mask(
    model:    nn.Module,
    loader:   DataLoader,
    device:   torch.device,
    mask_pos: int = -1,          # -1 = no masking (baseline)
    mask_val: float = 0.8,       # replace with mean RR (neutral)
) -> dict:
    """
    Runs inference. If mask_pos >= 0, sets that RR position to mask_val for all beats.
    Returns per-class recalls.
    """
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for X, Xrr, y in loader:
            X, Xrr = X.to(device), Xrr.to(device)
            if mask_pos >= 0:
                Xrr = Xrr.clone()
                Xrr[:, mask_pos] = mask_val     # mask this position
            logits = model(X, Xrr)
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(y.numpy())

    y_true = np.array(trues)
    y_pred = np.array(preds)

    return {
        "s_recall": float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)),
        "v_recall": float(recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)),
        "macro_f1": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single-seed ablation
# ─────────────────────────────────────────────────────────────────────────────

def ablate_one_seed(
    ckpt_path: str,
    data_dir:  str,
    device:    torch.device,
) -> dict:
    """Runs leave-one-out ablation on one trained model checkpoint."""
    model = WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=True).to(device)
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)

    ds     = ECGDatasetRR("test", data_dir)
    loader = DataLoader(ds, batch_size=512, shuffle=False)

    # Baseline (no masking)
    baseline = inference_with_mask(model, loader, device, mask_pos=-1)

    results = {"baseline": baseline, "per_position": {}}

    for pos_idx, pos_name in enumerate(RR_POSITIONS):
        masked = inference_with_mask(model, loader, device, mask_pos=pos_idx)
        delta  = {k: masked[k] - baseline[k] for k in baseline}
        results["per_position"][pos_name] = {
            "absolute": masked,
            "delta":    delta,
        }
        print(f"    {pos_name}: S-Recall {baseline['s_recall']:.3f} → {masked['s_recall']:.3f} "
              f"(Δ={delta['s_recall']:+.3f})")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Multi-seed aggregation
# ─────────────────────────────────────────────────────────────────────────────

def run_multiseed_ablation(
    model_dir: str,
    seeds:     list,
    data_dir:  str,
    device:    torch.device,
) -> dict:
    """Runs ablation across all seeds, aggregates deltas with mean±std."""
    base          = Path(model_dir)
    all_seed_data = []

    for seed in seeds:
        ckpt = base / f"seed_{seed}" / "best_model.pth"
        if not ckpt.exists():
            print(f"  ⚠️  Checkpoint not found: {ckpt}")
            continue
        print(f"\n  Seed {seed}:")
        result = ablate_one_seed(str(ckpt), data_dir, device)
        all_seed_data.append(result)

    if not all_seed_data:
        print("  ⚠️  No checkpoints found.")
        return {}

    # Aggregate: mean ± std of deltas across seeds
    aggregated = {
        "n_seeds":  len(all_seed_data),
        "baseline": {
            k: {
                "mean": float(np.mean([d["baseline"][k] for d in all_seed_data])),
                "std":  float(np.std([d["baseline"][k] for d in all_seed_data], ddof=1)),
            }
            for k in ["s_recall", "v_recall", "macro_f1"]
        },
        "per_position": {},
    }

    for pos_name in RR_POSITIONS:
        s_deltas = [d["per_position"][pos_name]["delta"]["s_recall"] for d in all_seed_data]
        v_deltas = [d["per_position"][pos_name]["delta"]["v_recall"] for d in all_seed_data]

        # Wilcoxon signed-rank test  (delta vs 0)
        try:
            if len(s_deltas) >= 5 and np.std(s_deltas) > 1e-8:
                stat_s, p_s = wilcoxon(s_deltas, alternative="less")
            else:
                p_s = 1.0
        except Exception:
            p_s = 1.0

        aggregated["per_position"][pos_name] = {
            "s_recall_delta_mean": float(np.mean(s_deltas)),
            "s_recall_delta_std":  float(np.std(s_deltas, ddof=1) if len(s_deltas) > 1 else 0),
            "v_recall_delta_mean": float(np.mean(v_deltas)),
            "v_recall_delta_std":  float(np.std(v_deltas, ddof=1) if len(v_deltas) > 1 else 0),
            "p_value_s":           float(p_s),
            "significant":         bool(p_s < 0.05),
            "per_seed_s_delta":    s_deltas,
        }

    return aggregated


# ─────────────────────────────────────────────────────────────────────────────
# Figure (matches original Fig 6 style + adds V-Recall panel)
# ─────────────────────────────────────────────────────────────────────────────

def plot_rr_ablation(agg: dict, save_path: str, model_name: str = "WavKAN-v2"):
    positions  = RR_POSITIONS
    s_means    = [agg["per_position"][p]["s_recall_delta_mean"] for p in positions]
    s_stds     = [agg["per_position"][p]["s_recall_delta_std"]  for p in positions]
    v_means    = [agg["per_position"][p]["v_recall_delta_mean"] for p in positions]
    v_stds     = [agg["per_position"][p]["v_recall_delta_std"]  for p in positions]
    sig_flags  = [agg["per_position"][p]["significant"]         for p in positions]
    p_vals     = [agg["per_position"][p]["p_value_s"]           for p in positions]

    x       = np.arange(len(positions))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, means, stds, ylabel, color in [
        (axes[0], s_means, s_stds, "ΔS-Recall (Supraventricular)", "#e15759"),
        (axes[1], v_means, v_stds, "ΔV-Recall (Ventricular)",      "#4e79a7"),
    ]:
        bars = ax.bar(x, means, color=color, alpha=0.75, width=0.55,
                      yerr=stds, capsize=5, ecolor="black", error_kw={"linewidth": 1.5})
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xticks(x)
        ax.set_xticklabels(positions, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_xlabel("Masked RR-Interval Position", fontsize=11)
        ax.grid(axis="y", alpha=0.35)

        # Significance annotations
        for i, (sig, p) in enumerate(zip(sig_flags, p_vals)):
            if sig:
                ax.text(i, means[i] - 0.008 if means[i] < 0 else means[i] + 0.003,
                        f"p={p:.3f}*", ha="center", va="top" if means[i] < 0 else "bottom",
                        fontsize=8.5, color="black", fontweight="bold")

        # Shade most impactful bar
        min_idx = int(np.argmin(means))
        bars[min_idx].set_edgecolor("black")
        bars[min_idx].set_linewidth(2.5)

    axes[0].set_title("S-Class  (Supraventricular): which RR position matters most?",
                       fontweight="bold", fontsize=11)
    axes[1].set_title("V-Class  (Ventricular): sensitivity to rhythm context",
                       fontweight="bold", fontsize=11)

    fig.suptitle(
        f"Leave-One-Out RR-Interval Ablation — {model_name}  "
        f"(n={agg['n_seeds']} seeds, baseline S-Recall="
        f"{agg['baseline']['s_recall']['mean']:.3f}±"
        f"{agg['baseline']['s_recall']['std']:.3f})",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  Figure: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def generate_latex_table(agg: dict) -> str:
    baseline_s = agg["baseline"]["s_recall"]["mean"]
    rows       = []
    for pos in RR_POSITIONS:
        d   = agg["per_position"][pos]
        dm  = d["s_recall_delta_mean"]
        ds  = d["s_recall_delta_std"]
        pv  = d["p_value_s"]
        sig = "$^{*}$" if d["significant"] else ""
        bold_open  = r"\textbf{" if d["significant"] else ""
        bold_close = r"}" if d["significant"] else ""
        rows.append(
            f"  {pos} & {bold_open}{dm:+.3f} ± {ds:.3f}{bold_close}{sig} "
            f"& {pv:.3f} \\\\"
        )

    return "\n".join([
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{Leave-one-out RR-interval ablation on WavKAN-v2 (DS2 test set, "
        rf"$n$={agg['n_seeds']} seeds). Baseline S-Recall = {baseline_s:.3f}. "
        rf"$\Delta$ = masked $-$ baseline. $^{{*}}p<0.05$, Wilcoxon signed-rank test.}}",
        r"\label{tab:rr_ablation}",
        r"\begin{tabular}{@{}lcc@{}}",
        r"\toprule",
        r"  \textbf{Masked Position} & \textbf{$\Delta$S-Recall (mean±std)} & \textbf{p-value} \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Leave-One-Out RR-Interval Ablation")
    parser.add_argument("--model-dir",  type=str, default="results/full_pipeline/wavkan_v2",
                        help="Directory containing seed_*/best_model.pth")
    parser.add_argument("--seeds",      type=int, nargs="+", default=[42, 101, 777, 2026, 31415])
    parser.add_argument("--data-dir",   type=str, default="data/processed_rr_history")
    parser.add_argument("--output-dir", type=str, default="results/rr_ablation")
    parser.add_argument("--device",     type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    OUT    = Path(args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"Leave-One-Out RR Ablation  (device={device})")
    print(f"{'='*60}")

    agg = run_multiseed_ablation(args.model_dir, args.seeds, args.data_dir, device)

    if not agg:
        print("Exiting: no data.")
    else:
        with open(OUT / "rr_ablation_report.json", "w") as f:
            json.dump(agg, f, indent=2)

        latex = generate_latex_table(agg)
        with open(OUT / "rr_ablation_table.tex", "w") as f:
            f.write(latex)

        plot_rr_ablation(agg, str(OUT / "rr_ablation_figure.pdf"))

        # Print summary
        print(f"\n{'='*60}")
        print(f"RR Ablation Results (S-Recall deltas):")
        print(f"  Baseline S-Recall: {agg['baseline']['s_recall']['mean']:.4f} "
              f"± {agg['baseline']['s_recall']['std']:.4f}")
        for pos in RR_POSITIONS:
            d = agg["per_position"][pos]
            sig_marker = " ← *" if d["significant"] else ""
            print(f"  Mask {pos}: Δ={d['s_recall_delta_mean']:+.4f}±{d['s_recall_delta_std']:.4f} "
                  f"(p={d['p_value_s']:.3f}){sig_marker}")

        most_important = min(RR_POSITIONS, key=lambda p: agg["per_position"][p]["s_recall_delta_mean"])
        print(f"\n  Most impactful position: {most_important} "
              f"(Δ={agg['per_position'][most_important]['s_recall_delta_mean']:+.4f})")
        print(f"\n✅ Saved to {OUT}/")
