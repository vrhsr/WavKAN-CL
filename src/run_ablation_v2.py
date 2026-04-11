"""
run_ablation_v2.py  —  Comprehensive Causal Ablation Study

Runs all ablation variants required by TBME/TNSRE reviewers.
Every variant uses the same training code (train_pca.py) so results
are directly comparable — no confounds.

Ablation Grid
=============
  Group 1 — PCWI (primary novelty)
    [A1] Full model         (PCWI=✓  PWAM=✓  RR-Attn=✓  Wavelet=mexican_hat)
    [A2] No PCWI            (PCWI=✗  PWAM=✓  RR-Attn=✓  → random init baseline)
  
  Group 2 — PWAM (secondary novelty)
    [A3] No PWAM            (PCWI=✓  PWAM=✗  RR-Attn=✓)
  
  Group 3 — Wavelet basis
    [A4] B-Spline           (PCWI=✗  PWAM=✓  Wavelet=b_spline)
    [A5] Morlet             (PCWI=✗  PWAM=✓  Wavelet=morlet)
    [A6] DOG                (PCWI=✗  PWAM=✓  Wavelet=dog)
  
  Group 4 — RR-Interval design
    [A7] No RR-Attn         (PCWI=✓  PWAM=✓  RR-Attn=✗ → plain MLP)
    [A8] Morphology only    (No RR branch at all — requires model variant)
  
  Group 5 — Curriculum strategy
    [A9] Standard CE        (no curriculum, no weighting)
    [A10] Focal Loss only   (focal without PCA schedule)
    [A11] Minority-first    (original broken curriculum from v1)

Usage:
    python src/run_ablation_v2.py --seeds 42 101 777 --epochs 100
    python src/run_ablation_v2.py --group pcwi --seeds 42 101 777
"""

import os, sys, json, argparse, time
from pathlib import Path
from typing import List, Dict

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.train_pca import train_pca


# ─────────────────────────────────────────────────────────────────────────────
# Ablation definitions
# ─────────────────────────────────────────────────────────────────────────────

ABLATIONS = {
    # ── Group 1: PCWI ────────────────────────────────────────────────────────
    "A1_full": {
        "label": "Full Model (PCWI + PWAM + MexHat)",
        "group": "pcwi",
        "kwargs": dict(use_pcwi=True, use_pwam=True, use_rr_attn=True,
                       wavelet_type="mexican_hat"),
    },
    "A2_no_pcwi": {
        "label": "No PCWI (random init)",
        "group": "pcwi",
        "kwargs": dict(use_pcwi=False, use_pwam=True, use_rr_attn=True,
                       wavelet_type="mexican_hat"),
    },

    # ── Group 2: PWAM ────────────────────────────────────────────────────────
    "A3_no_pwam": {
        "label": "No PWAM (no P-wave attention)",
        "group": "pwam",
        "kwargs": dict(use_pcwi=True, use_pwam=False, use_rr_attn=True,
                       wavelet_type="mexican_hat"),
    },

    # ── Group 3: Wavelet basis ───────────────────────────────────────────────
    "A4_bspline": {
        "label": "B-Spline KAN (no PCWI)",
        "group": "wavelet",
        "kwargs": dict(use_pcwi=False, use_pwam=True, use_rr_attn=True,
                       wavelet_type="b_spline"),
    },
    "A5_morlet": {
        "label": "Morlet wavelet",
        "group": "wavelet",
        "kwargs": dict(use_pcwi=False, use_pwam=True, use_rr_attn=True,
                       wavelet_type="morlet"),
    },
    "A6_dog": {
        "label": "DOG wavelet",
        "group": "wavelet",
        "kwargs": dict(use_pcwi=False, use_pwam=True, use_rr_attn=True,
                       wavelet_type="dog"),
    },

    # ── Group 4: RR-Interval design ──────────────────────────────────────────
    "A7_no_rr_attn": {
        "label": "RR Plain MLP (no attention)",
        "group": "rr",
        "kwargs": dict(use_pcwi=True, use_pwam=True, use_rr_attn=False,
                       wavelet_type="mexican_hat"),
    },

    # ── Group 5: Curriculum strategy ─────────────────────────────────────────
    "A9_no_curriculum": {
        "label": "No Curriculum (standard CE)",
        "group": "curriculum",
        "kwargs": dict(use_pcwi=True, use_pwam=True, use_rr_attn=True,
                       wavelet_type="mexican_hat",
                       warmup=0.0, focal_gamma=0.0, s_weight=1.0),
    },
    "A10_focal_only": {
        "label": "Focal Loss only (no PCA schedule)",
        "group": "curriculum",
        "kwargs": dict(use_pcwi=True, use_pwam=True, use_rr_attn=True,
                       wavelet_type="mexican_hat",
                       warmup=0.0, focal_gamma=2.0, s_weight=8.0),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Run ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_ablation(
    ablation_ids: List[str] = None,
    seeds: List[int]        = [42, 101, 777],
    epochs: int             = 100,
    data_dir: str           = "data/processed_rr_history",
    output_dir: str         = "results/ablation_v2",
) -> Dict:

    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)

    ids_to_run = ablation_ids or list(ABLATIONS.keys())
    all_results = {}

    for abl_id in ids_to_run:
        if abl_id not in ABLATIONS:
            print(f"⚠️  Unknown ablation ID: {abl_id}, skipping.")
            continue

        cfg    = ABLATIONS[abl_id]
        label  = cfg["label"]
        kwargs = cfg["kwargs"].copy()

        print(f"\n{'='*65}")
        print(f"ABLATION: {abl_id}  —  {label}")
        print(f"{'='*65}")

        seed_metrics = []
        for seed in seeds:
            t0 = time.time()
            try:
                m = train_pca(
                    seed       = seed,
                    epochs     = epochs,
                    data_dir   = data_dir,
                    output_dir = str(OUT / abl_id / f"seed_{seed}"),
                    **kwargs,
                )
                m["time_s"] = time.time() - t0
                seed_metrics.append(m)
            except Exception as e:
                print(f"  ⚠️  Seed {seed} failed: {e}")

        if not seed_metrics:
            continue

        # Aggregate across seeds
        def _agg(key):
            vals = [m[key] for m in seed_metrics if key in m]
            return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0)

        macro_f1_m, macro_f1_s = _agg("macro_f1")
        v_recall_m, v_recall_s = _agg("v_recall")
        s_recall_m, s_recall_s = _agg("s_recall")
        f_recall_m, f_recall_s = _agg("f_recall")

        all_results[abl_id] = {
            "label":         label,
            "group":         cfg["group"],
            "n_seeds":       len(seed_metrics),
            "macro_f1":      f"{macro_f1_m:.4f} ± {macro_f1_s:.4f}",
            "v_recall":      f"{v_recall_m:.4f} ± {v_recall_s:.4f}",
            "s_recall":      f"{s_recall_m:.4f} ± {s_recall_s:.4f}",
            "f_recall":      f"{f_recall_m:.4f} ± {f_recall_s:.4f}",
            "_raw": {"macro_f1": [m.get("macro_f1") for m in seed_metrics],
                     "v_recall":  [m.get("v_recall")  for m in seed_metrics],
                     "s_recall":  [m.get("s_recall")  for m in seed_metrics]},
        }

        print(f"\n  ✅ {abl_id} ({len(seed_metrics)} seeds):")
        print(f"     Macro-F1 : {macro_f1_m:.4f} ± {macro_f1_s:.4f}")
        print(f"     V-Recall : {v_recall_m:.4f} ± {v_recall_s:.4f}")
        print(f"     S-Recall : {s_recall_m:.4f} ± {s_recall_s:.4f}")

    # Save consolidated results
    with open(OUT / "ablation_summary.json", "w") as f:
        json.dump(all_results, f, indent=2,
                  default=lambda x: str(x) if not isinstance(x, (int, float, str, list, dict, bool, type(None))) else x)

    _print_comparison_table(all_results)
    _plot_ablation_summary(all_results, str(OUT / "ablation_summary.pdf"))

    return all_results


def _print_comparison_table(results: Dict):
    """Print a LaTeX-ready comparison table."""
    print(f"\n{'='*80}")
    print(f"{'ABLATION COMPARISON TABLE':^80}")
    print(f"{'='*80}")
    print(f"{'ID':<20} {'Label':<40} {'Macro-F1':>10} {'V-Rec':>8} {'S-Rec':>8}")
    print(f"{'-'*80}")

    # Group by ablation group
    groups = {}
    for k, v in results.items():
        g = v.get("group", "other")
        groups.setdefault(g, []).append((k, v))

    for gname, items in groups.items():
        print(f"\n  [Group: {gname.upper()}]")
        for abl_id, v in items:
            print(f"  {abl_id:<20} {v['label'][:38]:<40} "
                  f"{v['macro_f1']:>10} {v['v_recall']:>8} {v['s_recall']:>8}")
    print(f"{'='*80}")


def _plot_ablation_summary(results: Dict, save_path: str):
    """Bar chart of Macro-F1, V-Recall, S-Recall across ablations."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        ids    = list(results.keys())
        labels = [results[i]["label"][:30] for i in ids]
        raw    = {k: results[k]["_raw"] for k in ids}

        metrics = ["macro_f1", "v_recall", "s_recall"]
        colors  = ["#4e79a7", "#e15759", "#f28e2b"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

        for ax, metric, color in zip(axes, metrics, colors):
            means = [np.mean(raw[i][metric]) for i in ids]
            stds  = [np.std(raw[i][metric], ddof=1) if len(raw[i][metric]) > 1 else 0 for i in ids]
            bars  = ax.bar(range(len(ids)), means, yerr=stds, color=color, alpha=0.75,
                           capsize=4, error_kw=dict(lw=1.5))
            # Highlight full model
            if "A1_full" in ids:
                bars[ids.index("A1_full")].set_edgecolor("black")
                bars[ids.index("A1_full")].set_linewidth(2)
            ax.set_xticks(range(len(ids)))
            ax.set_xticklabels(ids, rotation=40, ha="right", fontsize=7)
            ax.set_title(metric.replace("_", " ").title(), fontweight="bold")
            ax.set_ylim(0, 1)
            ax.grid(axis="y", alpha=0.3)

        fig.suptitle("Ablation Study — WavKAN-v2", fontsize=14, fontweight="bold")
        plt.tight_layout()
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\n  Ablation figure saved: {save_path}")
        plt.close(fig)
    except Exception as e:
        print(f"  ⚠️  Could not save ablation figure: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WavKAN-v2 Comprehensive Ablation Study")
    parser.add_argument("--ids",    type=str, nargs="+", default=None,
                        help="Specific ablation IDs to run (default: all)")
    parser.add_argument("--group",  type=str, default=None,
                        choices=["pcwi", "pwam", "wavelet", "rr", "curriculum"],
                        help="Run only one ablation group")
    parser.add_argument("--seeds",  type=int, nargs="+", default=[42, 101, 777])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data-dir",   type=str, default="data/processed_rr_history")
    parser.add_argument("--output-dir", type=str, default="results/ablation_v2")
    args = parser.parse_args()

    ids_to_run = args.ids
    if args.group and not ids_to_run:
        ids_to_run = [k for k, v in ABLATIONS.items() if v["group"] == args.group]
        # Always include full model as reference
        if "A1_full" not in ids_to_run:
            ids_to_run = ["A1_full"] + ids_to_run

    run_ablation(
        ablation_ids = ids_to_run,
        seeds        = args.seeds,
        epochs       = args.epochs,
        data_dir     = args.data_dir,
        output_dir   = args.output_dir,
    )
