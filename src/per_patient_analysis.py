"""
per_patient_analysis.py  —  Per-Patient Consistency Analysis

WHY THIS IS CRITICAL FOR TBME ACCEPTANCE
==========================================
A model with good aggregate metrics can still be bad in practice if
it performs excellently on 18 easy DS2 patients and disastrously on 4.
Reviewers increasingly ask: "Is your model consistent across patients?"

This script answers that definitively by computing V-Recall and S-Recall
for EACH of the 22 DS2 test patients individually, then:

  1. Plots a per-patient heatmap (visual proof of consistency)
  2. Computes the Coefficient of Variation (CV) across patients
  3. Compares WavKAN-v2 CV vs B-Spline KAN CV
     → If WavKAN-v2 has lower CV → "more consistent deployment"

Key addition over original paper:
  Original paper only shows aggregate metrics.
  This adds patient-level granularity that TBME reviewers demand.

Output:
  results/per_patient/
    per_patient_metrics.json       Per-patient V-Recall, S-Recall, Macro-F1
    per_patient_heatmap.pdf        Color heatmap across all patients vs metrics
    per_patient_boxplot.pdf        Distribution of per-patient V-Recall
    patient_consistency_table.tex  LaTeX table

Usage:
    python src/per_patient_analysis.py \\
        --model-dir results/full_pipeline/wavkan_v2/ \\
        --baseline-dir results/full_pipeline/baseline_bspline_kan/ \\
        --seeds 42 101 777 \\
        --output-dir results/per_patient/
"""

import os, sys, json, argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import recall_score, f1_score, confusion_matrix

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.wavkan_v2 import WavKAN_v2
# We'll also load baselines dynamically

# DS2 test records (De Chazal Protocol — must exactly match process_data.py)
DS2_RECORDS = [
    '100','103','105','111','113','117','121','123',
    '200','202','210','212','213','214','219','221',
    '222','228','231','232','233','234'
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset with record IDs
# ─────────────────────────────────────────────────────────────────────────────

class ECGDatasetWithIDs(Dataset):
    def __init__(self, data_dir: str):
        base      = Path(data_dir)
        self.X    = torch.tensor(np.load(base / "X_test.npy"),    dtype=torch.float32)
        self.Xrr  = torch.tensor(np.load(base / "X_rr_test.npy"), dtype=torch.float32)
        self.y    = torch.tensor(np.load(base / "y_test.npy"),    dtype=torch.long)
        ids_raw   = np.load(base / "ids_test.npy", allow_pickle=True)
        self.ids  = np.array([str(i) for i in ids_raw])

    def __len__(self):        return len(self.y)
    def __getitem__(self, i): return self.X[i], self.Xrr[i], self.y[i], self.ids[i]


def collate_with_ids(batch):
    X    = torch.stack([b[0] for b in batch])
    Xrr  = torch.stack([b[1] for b in batch])
    y    = torch.tensor([b[2] for b in batch])
    ids  = [b[3] for b in batch]
    return X, Xrr, y, ids


# ─────────────────────────────────────────────────────────────────────────────
# Per-patient inference
# ─────────────────────────────────────────────────────────────────────────────

def run_per_patient_inference(
    model:    torch.nn.Module,
    data_dir: str,
    device:   torch.device,
) -> dict:
    """Returns dict: record_id → {v_recall, s_recall, macro_f1, n_beats, class_counts}"""
    try:
        ds = ECGDatasetWithIDs(data_dir)
    except FileNotFoundError as e:
        print(f"  ⚠️  ids_test.npy not found. Re-run process_data.py to save record IDs.")
        return {}

    loader = DataLoader(ds, batch_size=512, shuffle=False, collate_fn=collate_with_ids)

    model.eval()
    # Accumulate per-patient
    patient_preds = defaultdict(list)
    patient_trues = defaultdict(list)

    with torch.no_grad():
        for X, Xrr, y, ids in loader:
            logits = model(X.to(device), Xrr.to(device))
            preds  = logits.argmax(1).cpu().numpy()
            for i, rec_id in enumerate(ids):
                patient_preds[rec_id].append(preds[i])
                patient_trues[rec_id].append(y[i].item())

    results = {}
    for rec_id in sorted(patient_preds.keys()):
        y_true = np.array(patient_trues[rec_id])
        y_pred = np.array(patient_preds[rec_id])

        v_recall = float(recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0))
        s_recall = float(recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        from collections import Counter
        counts   = Counter(y_true.tolist())

        results[rec_id] = {
            "v_recall":     v_recall,
            "s_recall":     s_recall,
            "macro_f1":     macro_f1,
            "n_beats":      int(len(y_true)),
            "n_v_beats":    int(counts.get(2, 0)),
            "n_s_beats":    int(counts.get(1, 0)),
        }

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Aggregation across seeds
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_seeds(per_seed_results: list) -> dict:
    """Average per-patient metrics across seeds."""
    all_records = set()
    for seed_result in per_seed_results:
        all_records.update(seed_result.keys())

    aggregated = {}
    for rec in sorted(all_records):
        vrs = [s[rec]["v_recall"] for s in per_seed_results if rec in s]
        srs = [s[rec]["s_recall"] for s in per_seed_results if rec in s]
        f1s = [s[rec]["macro_f1"] for s in per_seed_results if rec in s]

        aggregated[rec] = {
            "v_recall_mean": float(np.mean(vrs)),
            "v_recall_std":  float(np.std(vrs, ddof=1) if len(vrs) > 1 else 0),
            "s_recall_mean": float(np.mean(srs)),
            "s_recall_std":  float(np.std(srs, ddof=1) if len(srs) > 1 else 0),
            "macro_f1_mean": float(np.mean(f1s)),
            "macro_f1_std":  float(np.std(f1s, ddof=1) if len(f1s) > 1 else 0),
            "n_beats":       per_seed_results[0][rec]["n_beats"] if rec in per_seed_results[0] else 0,
            "n_v_beats":     per_seed_results[0][rec]["n_v_beats"] if rec in per_seed_results[0] else 0,
            "n_s_beats":     per_seed_results[0][rec]["n_s_beats"] if rec in per_seed_results[0] else 0,
        }

    return aggregated


def compute_consistency_stats(agg: dict) -> dict:
    v_recalls = [agg[r]["v_recall_mean"] for r in agg]
    s_recalls = [agg[r]["s_recall_mean"] for r in agg if agg[r]["n_s_beats"] > 0]

    def cv(arr):  # Coefficient of Variation
        m = np.mean(arr)
        return float(np.std(arr, ddof=1) / (m + 1e-8)) if len(arr) > 1 else 0.0

    return {
        "v_recall": {
            "mean":   float(np.mean(v_recalls)),
            "std":    float(np.std(v_recalls, ddof=1)),
            "min":    float(np.min(v_recalls)),
            "max":    float(np.max(v_recalls)),
            "cv":     cv(v_recalls),
            "n_patients_above_0.8": int(sum(v >= 0.80 for v in v_recalls)),
            "n_patients_above_0.9": int(sum(v >= 0.90 for v in v_recalls)),
        },
        "s_recall": {
            "mean":   float(np.mean(s_recalls)) if s_recalls else 0,
            "std":    float(np.std(s_recalls, ddof=1)) if len(s_recalls) > 1 else 0,
            "cv":     cv(s_recalls) if s_recalls else 0,
        },
        "n_patients": len(agg),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_patient_heatmap(
    our_agg:      dict,
    baseline_agg: dict,
    save_path:    str,
):
    """Side-by-side heatmap: WavKAN-v2 vs B-Spline KAN, per patient."""
    records  = sorted(set(list(our_agg.keys()) + list(baseline_agg.keys())))
    metrics  = ["v_recall_mean", "s_recall_mean", "macro_f1_mean"]
    m_labels = ["V-Recall", "S-Recall", "Macro-F1"]

    fig, axes = plt.subplots(1, 2, figsize=(16, max(8, len(records) * 0.38 + 2)))

    for ax, agg, title in [
        (axes[0], our_agg,      "WavKAN-v2 (Ours)"),
        (axes[1], baseline_agg, "B-Spline KAN (Baseline)"),
    ]:
        if not agg:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            continue

        data = np.array([[agg.get(r, {}).get(m, 0.0) for m in metrics] for r in records])
        im   = ax.imshow(data, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
        ax.set_xticks(range(len(m_labels)))
        ax.set_xticklabels(m_labels, rotation=30, ha="right", fontsize=10)
        ax.set_yticks(range(len(records)))
        ax.set_yticklabels([f"Rec {r}" for r in records], fontsize=8.5)
        ax.set_title(title, fontweight="bold", fontsize=12)

        # Add text annotations
        for i in range(len(records)):
            for j in range(len(metrics)):
                val = data[i, j]
                color = "black" if 0.25 < val < 0.75 else "white"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=color)

        plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle("Per-Patient Performance Heatmap — DS2 Test Set (22 Patients)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Heatmap: {save_path}")


def plot_consistency_boxplot(
    our_agg:      dict,
    baseline_agg: dict,
    save_path:    str,
):
    """Boxplot of per-patient V-Recall: WavKAN-v2 vs B-Spline (visual consistency proof)."""
    our_vr = [our_agg[r]["v_recall_mean"] for r in our_agg]
    bs_vr  = [baseline_agg[r]["v_recall_mean"] for r in baseline_agg] if baseline_agg else []

    fig, ax = plt.subplots(figsize=(7, 5))

    data   = [d for d in [our_vr, bs_vr] if d]
    labels = ["WavKAN-v2\n(Ours)"]
    colors = ["#e15759"]
    if bs_vr:
        labels.append("B-Spline KAN\n(Baseline)")
        colors.append("#4e79a7")

    bp = ax.boxplot(data, patch_artist=True, notch=False, vert=True,
                    medianprops={"linewidth": 2.5, "color": "white"},
                    whiskerprops={"linewidth": 1.5},
                    capprops={"linewidth": 1.5})
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    # Overlay scatter
    for i, vals in enumerate(data, start=1):
        jitter = np.random.uniform(-0.08, 0.08, size=len(vals))
        ax.scatter(np.ones(len(vals)) * i + jitter, vals,
                   color="black", s=25, alpha=0.6, zorder=5)

    ax.axhline(0.80, color="grey", lw=1, ls="--", alpha=0.7, label="0.80 threshold")
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Per-Patient V-Recall", fontsize=12)
    ax.set_title("Per-Patient V-Recall Distribution\n(22 DS2 Patients)", fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.35)
    ax.set_ylim(0, 1.05)

    # CV annotation
    if our_vr:
        cv_our  = np.std(our_vr, ddof=1) / (np.mean(our_vr) + 1e-8)
        ax.text(1, 0.03, f"CV={cv_our:.3f}", ha="center", fontsize=9,
                color="#e15759", fontweight="bold")
    if bs_vr:
        cv_bs = np.std(bs_vr, ddof=1) / (np.mean(bs_vr) + 1e-8)
        ax.text(2, 0.03, f"CV={cv_bs:.3f}", ha="center", fontsize=9,
                color="#4e79a7", fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"  Boxplot: {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# LaTeX table
# ─────────────────────────────────────────────────────────────────────────────

def generate_latex_table(
    our_agg:  dict,
    our_stats: dict,
    our_name: str = "WavKAN-v2",
) -> str:
    rows = []
    for rec in sorted(our_agg.keys()):
        d = our_agg[rec]
        v = d["v_recall_mean"]; s = d["s_recall_mean"]; f = d["macro_f1_mean"]
        n_v = d["n_v_beats"]; n_s = d["n_s_beats"]
        v_str = f"\\textbf{{{v:.3f}}}" if v >= 0.90 else f"{v:.3f}"
        rows.append(
            f"  {rec} & {d['n_beats']:,} & {n_v} & {n_s} & "
            f"{v_str} & {s:.3f} & {f:.3f} \\\\"
        )

    stats = our_stats["v_recall"]
    return "\n".join([
        r"\begin{table*}[!htbp]",
        r"\centering",
        rf"\caption{{Per-patient performance of {our_name} on all 22 DS2 test records. "
        r"Bold V-Recall $\geq$ 0.90. CV = Coefficient of Variation across patients.}}",
        r"\label{tab:per_patient}",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"  \textbf{Record} & \textbf{Total Beats} & \textbf{V Beats} & "
        r"\textbf{S Beats} & \textbf{V-Recall} & \textbf{S-Recall} & "
        r"\textbf{Macro-F1} \\",
        r"\midrule",
        *rows,
        r"\midrule",
        rf"  \textbf{{Summary}} & & & & "
        rf"{stats['mean']:.3f}$\pm${stats['std']:.3f} & & \\",
        rf"  \textbf{{CV}} & & & & {stats['cv']:.3f} & & \\",
        rf"  \textbf{{$\geq$0.90}} & & & & {stats['n_patients_above_0.9']}/22 & & \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ])


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-Patient Consistency Analysis")
    parser.add_argument("--model-dir",    type=str, default="results/full_pipeline/wavkan_v2")
    parser.add_argument("--baseline-dir", type=str,
                        default="results/full_pipeline/baseline_bspline_kan",
                        help="Optional: baseline for comparison heatmap")
    parser.add_argument("--seeds",        type=int, nargs="+", default=[42, 101, 777])
    parser.add_argument("--data-dir",     type=str, default="data/processed_rr_history")
    parser.add_argument("--output-dir",   type=str, default="results/per_patient")
    parser.add_argument("--device",       type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-rr-attn", action="store_true",
                        help="Match a checkpoint trained with use_rr_attn=False, e.g. "
                             "results/ablation_no_rr_attn/*/best_model.pth.")
    args = parser.parse_args()

    OUT    = Path(args.output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"Per-Patient Consistency Analysis  (device={device})")
    print(f"{'='*60}")

    # ── Run WavKAN-v2 across seeds ───────────────────────────────────────────
    our_per_seed = []
    for seed in args.seeds:
        ckpt = Path(args.model_dir) / f"seed_{seed}" / "best_model.pth"
        if not ckpt.exists():
            print(f"  ⚠️  {ckpt} not found, skipping.")
            continue
        # use_rr_attn hardcoded True until 2026-08-27 -- same recurring bug fixed the
        # same day in eval_ptbxl.py/wavelet_alignment_score.py/rr_ablation.py/
        # temperature_scaling.py (AUDIT_FINDINGS.md C19).
        model = WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=not args.no_rr_attn).to(device)
        model.load_state_dict(torch.load(str(ckpt), map_location=device))
        print(f"\n  Seed {seed} — WavKAN-v2")
        res = run_per_patient_inference(model, args.data_dir, device)
        if res:
            our_per_seed.append(res)

    if not our_per_seed:
        print("No WavKAN-v2 results. Exiting.")
        exit(1)

    our_agg   = aggregate_seeds(our_per_seed)
    our_stats = compute_consistency_stats(our_agg)

    print(f"\n  WavKAN-v2 Per-Patient V-Recall:")
    print(f"    Mean   : {our_stats['v_recall']['mean']:.4f}")
    print(f"    Std    : {our_stats['v_recall']['std']:.4f}")
    print(f"    CV     : {our_stats['v_recall']['cv']:.4f}  (lower = more consistent)")
    print(f"    ≥0.90  : {our_stats['v_recall']['n_patients_above_0.9']}/22 patients")

    # ── Optional: B-Spline baseline ─────────────────────────────────────────
    bs_agg = {}
    bs_dir = Path(args.baseline_dir)
    if bs_dir.exists():
        from src.baselines_extended import BSplineKAN_RR
        bs_per_seed = []
        for seed in args.seeds:
            ckpt = bs_dir / f"seed_{seed}" / "best_model.pth"
            if not ckpt.exists():
                continue
            model = BSplineKAN_RR().to(device)
            model.load_state_dict(torch.load(str(ckpt), map_location=device))
            print(f"\n  Seed {seed} — B-Spline KAN")
            res = run_per_patient_inference(model, args.data_dir, device)
            if res:
                bs_per_seed.append(res)
        if bs_per_seed:
            bs_agg   = aggregate_seeds(bs_per_seed)
            bs_stats  = compute_consistency_stats(bs_agg)
            print(f"\n  B-Spline KAN CV: {bs_stats['v_recall']['cv']:.4f}")

    # ── Save & plot ───────────────────────────────────────────────────────────
    report = {"wavkan_v2": our_agg, "bspline_kan": bs_agg,
              "stats": {"wavkan_v2": our_stats}}
    with open(OUT / "per_patient_metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    plot_per_patient_heatmap(our_agg, bs_agg, str(OUT / "per_patient_heatmap.pdf"))
    plot_consistency_boxplot(our_agg, bs_agg, str(OUT / "per_patient_boxplot.pdf"))

    latex = generate_latex_table(our_agg, our_stats)
    with open(OUT / "patient_consistency_table.tex", "w") as f:
        f.write(latex)

    print(f"\n✅ Per-patient analysis saved to {OUT}/")
