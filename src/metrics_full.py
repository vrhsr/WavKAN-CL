"""
metrics_full.py  —  Complete Statistical & Clinical Metrics Suite

Implements everything required by TBME/TNSRE reviewers:
  1. Macro-F1, per-class Precision/Recall/F1 (standard)
  2. AUPRC  (Area Under Precision-Recall Curve, per class + macro)
  3. MCC    (Matthews Correlation Coefficient)
  4. Balanced Accuracy
  5. 95% Bootstrap Confidence Intervals
  6. Paired Wilcoxon tests with effect size (Cohen's d)
  7. Multiple comparison correction (Holm-Bonferroni)
  8. ECE    (Expected Calibration Error)
  9. Reliability diagrams (calibration plots)
  10. ROC / PR curves with threshold analysis
  11. Clinical operating point: at 90% V-sensitivity
"""

import os, json, warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats

from sklearn.metrics import (
    confusion_matrix, classification_report,
    f1_score, recall_score, precision_score,
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    matthews_corrcoef, balanced_accuracy_score,
)
from sklearn.calibration import calibration_curve

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

CLASS_NAMES = ["N", "S", "V", "F", "Q"]
N_BOOTSTRAP = 1000
RNG         = np.random.default_rng(42)
FIG_DPI     = 200


# ─────────────────────────────────────────────────────────────────────────────
# 1. Core Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(
    y_true:  np.ndarray,
    y_pred:  np.ndarray,
    y_probs: np.ndarray,          # (N, 5) predicted probabilities
    class_names: List[str] = CLASS_NAMES,
) -> Dict:
    """
    Compute the full metrics dictionary for one model/seed evaluation.

    Returns a flat dict of key → float value, ready to be saved as JSON.
    """
    n_cls = len(class_names)
    metrics = {}

    # ── Per-class recall / precision / F1 ────────────────────────────────────
    report = classification_report(
        y_true, y_pred, labels=range(n_cls),
        target_names=class_names, output_dict=True, zero_division=0,
    )
    for c in class_names:
        if c in report:
            metrics[f"{c}_recall"]    = report[c]["recall"]
            metrics[f"{c}_precision"] = report[c]["precision"]
            metrics[f"{c}_f1"]        = report[c]["f1-score"]

    # ── Aggregate ─────────────────────────────────────────────────────────────
    metrics["macro_f1"]         = report["macro avg"]["f1-score"]
    metrics["macro_precision"]  = report["macro avg"]["precision"]
    metrics["macro_recall"]     = report["macro avg"]["recall"]

    # ── MCC & Balanced Accuracy ───────────────────────────────────────────────
    metrics["mcc"]              = float(matthews_corrcoef(y_true, y_pred))
    metrics["balanced_accuracy"]= float(balanced_accuracy_score(y_true, y_pred))

    # ── AUPRC (per class, one-vs-rest) ────────────────────────────────────────
    for i, c in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0:
            metrics[f"{c}_auprc"] = float("nan")
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ap = average_precision_score(y_bin, y_probs[:, i])
        metrics[f"{c}_auprc"] = float(ap)
    valid_auprcs = [metrics[f"{c}_auprc"] for c in class_names
                    if not np.isnan(metrics[f"{c}_auprc"])]
    metrics["macro_auprc"] = float(np.mean(valid_auprcs)) if valid_auprcs else float("nan")

    # ── AUROC (per class, one-vs-rest) ────────────────────────────────────────
    for i, c in enumerate(class_names):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            metrics[f"{c}_auroc"] = float("nan")
            continue
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        metrics[f"{c}_auroc"] = float(auc(fpr, tpr))

    # ── False Negative Rate (FNR) for safety-critical V class ────────────────
    cm       = confusion_matrix(y_true, y_pred, labels=range(n_cls))
    v_idx    = class_names.index("V")
    fn_v     = cm[v_idx].sum() - cm[v_idx, v_idx]
    tp_v     = cm[v_idx, v_idx]
    metrics["V_fnr"] = float(fn_v / (tp_v + fn_v + 1e-8))   # False Negative Rate

    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# 2. Bootstrap Confidence Intervals
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci(
    y_true:   np.ndarray,
    y_pred:   np.ndarray,
    y_probs:  np.ndarray,
    n_iter:   int = N_BOOTSTRAP,
    alpha:    float = 0.05,
) -> Dict:
    """
    Compute 95% bootstrap CI for key metrics.
    Returns dict: metric → (mean, lower, upper)
    """
    n = len(y_true)
    keys = ["macro_f1", "V_recall", "S_recall", "mcc", "balanced_accuracy",
            "macro_auprc", "V_auprc"]

    records = {k: [] for k in keys}

    for _ in range(n_iter):
        idx     = RNG.integers(0, n, size=n)
        yt, yp  = y_true[idx], y_pred[idx]
        ypr     = y_probs[idx]

        try:
            m = compute_all_metrics(yt, yp, ypr)
            for k in keys:
                records[k].append(m.get(k, float("nan")))
        except Exception:
            continue

    ci = {}
    for k, vals in records.items():
        arr = np.array([v for v in vals if not np.isnan(v)])
        if len(arr) == 0:
            ci[k] = (float("nan"), float("nan"), float("nan"))
        else:
            lo, hi = np.percentile(arr, [100 * alpha / 2, 100 * (1 - alpha / 2)])
            ci[k]  = (float(arr.mean()), float(lo), float(hi))
    return ci


# ─────────────────────────────────────────────────────────────────────────────
# 3. Statistical Comparison (two models, N seeds each)
# ─────────────────────────────────────────────────────────────────────────────

def statistical_comparison(
    scores_a: List[float],
    scores_b: List[float],
    metric_name: str = "macro_f1",
) -> Dict:
    """
    Paired Wilcoxon signed-rank test + Cohen's d effect size.
    Returns dict with p-value, effect_size, interpretation.
    """
    a, b = np.array(scores_a), np.array(scores_b)
    diff = a - b

    # Wilcoxon signed-rank
    if len(diff) >= 6 and np.any(diff != 0):
        stat, p_val = stats.wilcoxon(a, b, alternative="two-sided")
    else:
        stat, p_val = float("nan"), float("nan")

    # Cohen's d  (paired, using SD of differences)
    d = diff.mean() / (diff.std(ddof=1) + 1e-10)

    return {
        "metric":       metric_name,
        "mean_a":       float(a.mean()),
        "std_a":        float(a.std(ddof=1)),
        "mean_b":       float(b.mean()),
        "std_b":        float(b.std(ddof=1)),
        "mean_diff":    float(diff.mean()),
        "wilcoxon_stat":float(stat),
        "p_value":      float(p_val),
        "cohens_d":     float(d),
        "significant":  bool(p_val < 0.05),
        "effect_size":  (
            "large"  if abs(d) >= 0.8 else
            "medium" if abs(d) >= 0.5 else
            "small"  if abs(d) >= 0.2 else "negligible"
        ),
    }


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Applies Holm-Bonferroni step-down correction to a list of p-values.
    Returns adjusted p-values in the same order as input.
    """
    n       = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [None] * n
    max_adj  = 0.0
    for rank, (orig_idx, p) in enumerate(indexed):
        adj        = p * (n - rank)
        adj        = max(adj, max_adj)   # monotone step-down
        max_adj    = adj
        adjusted[orig_idx] = min(adj, 1.0)
    return adjusted


# ─────────────────────────────────────────────────────────────────────────────
# 4. Expected Calibration Error
# ─────────────────────────────────────────────────────────────────────────────

def expected_calibration_error(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes ECE = Σ_b |B_b| / N * |acc(B_b) - conf(B_b)|

    Returns:
        ece     : scalar ECE
        bin_acc : per-bin accuracy
        bin_conf: per-bin confidence
    """
    # Use max probability as confidence
    confidences = y_probs.max(axis=1)
    predictions = y_probs.argmax(axis=1)
    correct     = (predictions == y_true).astype(float)

    bins    = np.linspace(0, 1, n_bins + 1)
    bin_acc  = np.zeros(n_bins)
    bin_conf = np.zeros(n_bins)
    bin_cnt  = np.zeros(n_bins)

    for b in range(n_bins):
        mask = (confidences >= bins[b]) & (confidences < bins[b + 1])
        if mask.sum() > 0:
            bin_acc[b]  = correct[mask].mean()
            bin_conf[b] = confidences[mask].mean()
            bin_cnt[b]  = mask.sum()

    ece = (bin_cnt / len(y_true) * np.abs(bin_acc - bin_conf)).sum()
    return float(ece), bin_acc, bin_conf


# ─────────────────────────────────────────────────────────────────────────────
# 5. Clinical Operating Point
# ─────────────────────────────────────────────────────────────────────────────

def clinical_operating_point(
    y_true:   np.ndarray,
    y_probs:  np.ndarray,
    v_class:  int          = 2,
    target_sensitivity: float = 0.90,
) -> Dict:
    """
    At target V-sensitivity (recall), compute the corresponding specificity,
    FNR, and threshold. Critical for clinical deployment discussion.
    """
    y_bin = (y_true == v_class).astype(int)
    fpr, tpr, thresholds = roc_curve(y_bin, y_probs[:, v_class])

    # Find threshold closest to target sensitivity
    idx = np.argmin(np.abs(tpr - target_sensitivity))
    t   = thresholds[idx]

    preds_at_t = (y_probs[:, v_class] >= t).astype(int)
    tp = ((preds_at_t == 1) & (y_bin == 1)).sum()
    tn = ((preds_at_t == 0) & (y_bin == 0)).sum()
    fp = ((preds_at_t == 1) & (y_bin == 0)).sum()
    fn = ((preds_at_t == 0) & (y_bin == 1)).sum()

    return {
        "target_sensitivity":  target_sensitivity,
        "achieved_sensitivity":float(tpr[idx]),
        "specificity":         float(tn / (tn + fp + 1e-8)),
        "fnr":                 float(fn / (fn + tp + 1e-8)),
        "threshold":           float(t),
        "auroc_v":             float(auc(fpr, tpr)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 6. Publication-Quality Figures
# ─────────────────────────────────────────────────────────────────────────────

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram",
) -> None:
    """
    Plots calibration (reliability) diagram.
    A perfectly calibrated model lies on the diagonal.
    """
    ece, bin_acc, bin_conf = expected_calibration_error(y_true, y_probs)
    n_bins    = len(bin_acc)
    bin_mids  = np.linspace(0.05, 0.95, n_bins)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Perfect calibration")
    ax.bar(bin_mids, bin_acc, width=0.09, alpha=0.6, color="steelblue",
           label=f"Model  (ECE={ece:.4f})")
    ax.plot(bin_mids, bin_conf, "r.", markersize=6, label="Confidence")
    ax.set_xlabel("Mean Predicted Confidence", fontsize=12)
    ax.set_ylabel("Fraction Correct", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_roc_pr_curves(
    y_true:   np.ndarray,
    y_probs:  np.ndarray,
    save_path: Optional[str] = None,
    title: str = "ROC & PR Curves",
    highlight: str = "V",
) -> None:
    """
    Side-by-side ROC and PR curves for all classes.
    V-class highlighted as the primary safety metric.
    """
    colors = {"N": "#4e79a7", "S": "#f28e2b", "V": "#e15759",
              "F": "#76b7b2", "Q": "#59a14f"}
    n_cls = len(CLASS_NAMES)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for i, c in enumerate(CLASS_NAMES):
        y_bin = (y_true == i).astype(int)
        if y_bin.sum() < 5:
            continue
        col = colors.get(c, "gray")
        lw  = 2.5 if c == highlight else 1.2

        # ROC
        fpr, tpr, _ = roc_curve(y_bin, y_probs[:, i])
        auroc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=col, lw=lw,
                     label=f"{c} (AUC={auroc:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_bin, y_probs[:, i])
        ap = average_precision_score(y_bin, y_probs[:, i])
        axes[1].plot(rec, prec, color=col, lw=lw,
                     label=f"{c} (AP={ap:.3f})")

    # Formatting
    axes[0].plot([0,1],[0,1],"k--",lw=0.8)
    axes[0].set_xlabel("False Positive Rate"); axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curves", fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curves", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_confusion_matrix_styled(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    save_path: Optional[str] = None,
    title:     str = "Normalised Confusion Matrix (DS2 Test Set)",
) -> None:
    """Publication-quality normalised confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES, fontsize=11)
    ax.set_yticks(range(len(CLASS_NAMES))); ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12, fontweight="bold")
    ax.set_ylabel("True",      fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=12, fontweight="bold", pad=12)

    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            val  = cm_norm[i, j]
            cnt  = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n({cnt})", ha="center", va="center",
                    fontsize=8, color=color)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=FIG_DPI, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Full Report Generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_full_report(
    y_true:    np.ndarray,
    y_pred:    np.ndarray,
    y_probs:   np.ndarray,
    output_dir: str = "results/full_report",
    model_name: str = "WavKAN-v2",
    run_bootstrap: bool = True,
) -> Dict:
    """
    Generates a complete metrics report + all paper-ready figures.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = out / "figures"
    figs.mkdir(exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Generating full report: {model_name}")
    print(f"{'='*60}")

    # Core metrics
    metrics = compute_all_metrics(y_true, y_pred, y_probs)
    print(f"  Macro-F1          : {metrics['macro_f1']:.4f}")
    print(f"  MCC               : {metrics['mcc']:.4f}")
    print(f"  Balanced Accuracy : {metrics['balanced_accuracy']:.4f}")
    print(f"  Macro AUPRC       : {metrics['macro_auprc']:.4f}")
    print(f"  V-Recall          : {metrics['V_recall']:.4f}  (FNR={metrics['V_fnr']:.4f})")
    print(f"  S-Recall          : {metrics['S_recall']:.4f}")
    print(f"  F-Recall          : {metrics['F_recall']:.4f}")

    # ECE
    ece, _, _ = expected_calibration_error(y_true, y_probs)
    metrics["ece"] = ece
    print(f"  ECE               : {ece:.4f}")

    # Clinical operating point
    cop = clinical_operating_point(y_true, y_probs, target_sensitivity=0.90)
    metrics["clinical_op_90"] = cop
    print(f"  @ 90% V-sensitivity: specificity={cop['specificity']:.4f}  "
          f"FNR={cop['fnr']:.4f}  threshold={cop['threshold']:.3f}")

    # Bootstrap CI
    if run_bootstrap:
        print(f"\n  Computing {N_BOOTSTRAP}-sample bootstrap CI...")
        ci = bootstrap_ci(y_true, y_pred, y_probs)
        metrics["bootstrap_ci"] = ci
        for k, (mean, lo, hi) in ci.items():
            print(f"    {k:20s}: {mean:.4f}  [{lo:.4f}, {hi:.4f}]")

    # Save metrics
    with open(out / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Figures
    print(f"\n  Generating figures → {figs}/")
    plot_reliability_diagram(y_true, y_probs,
                             save_path=str(figs / "reliability_diagram.pdf"),
                             title=f"{model_name} — Reliability Diagram")
    plot_roc_pr_curves(y_true, y_probs,
                       save_path=str(figs / "roc_pr_curves.pdf"),
                       title=f"{model_name} — ROC & PR Curves")
    plot_confusion_matrix_styled(y_true, y_pred,
                                 save_path=str(figs / "confusion_matrix.pdf"),
                                 title=f"{model_name} — Confusion Matrix (DS2)")

    print(f"\n✅ Full report saved to {out}/")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate full metrics report from saved predictions")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing test_predictions.npy, test_true.npy, test_probs.npy")
    parser.add_argument("--output-dir",  type=str, default=None)
    parser.add_argument("--model-name",  type=str, default="WavKAN-v2")
    parser.add_argument("--no-bootstrap", action="store_true")
    args = parser.parse_args()

    r = Path(args.results_dir)
    y_true  = np.load(r / "test_true.npy")
    y_pred  = np.load(r / "test_predictions.npy")
    y_probs = np.load(r / "test_probs.npy")

    generate_full_report(
        y_true     = y_true,
        y_pred     = y_pred,
        y_probs    = y_probs,
        output_dir = args.output_dir or str(r / "full_report"),
        model_name = args.model_name,
        run_bootstrap = not args.no_bootstrap,
    )
