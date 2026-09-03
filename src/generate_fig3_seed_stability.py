"""
generate_fig3_seed_stability.py -- Real seed-stability figure (5-model comparison)

Written 2026-08-13 (AUDIT_FINDINGS.md C1). The old fig3_seed_stability() in
generate_all_figures.py had correct plotting/stats *logic* but read from paths that
never existed in this checkout (results/full_pipeline/wavkan_v2/seed_*/test_metrics.json
and a baselines_summary.json that was never produced) -- that mismatch, not a logic
bug, is why C1 found it silently rendering with only one model's data populated. This
script reads directly from the paths this repo's real training scripts actually write:
  - results/wavkan_v2_curriculum/seed_*/test_metrics.json      (train_pca.py)
  - results/baseline_{resnet1d,transformer,cnn_focal,bspline_kan}/seed_*/test_metrics.json
    (baselines_extended.py)

Also fixes the H8 Wilcoxon-pairing bug present in the old function: pairs seeds by
IDENTITY (only seeds present for BOTH WavKAN-v2 and a given baseline are compared),
never by list length/position.

Updated 2026-08-13 to use src/metrics_full.statistical_comparison (Cohen's d) and
Holm-Bonferroni correction across the 4 baseline comparisons, same upgrade as
aggregate_seed_results.py -- reports 4 p-values from one figure without correction
otherwise inflates the family-wise Type I error rate. Default seed count raised from
the original manuscript's n=5 to the full 20-seed list (unlimited compute available,
no reason to under-power this comparison relative to the main one).

Usage:
    python src/generate_fig3_seed_stability.py \\
        --wavkan-dir results/wavkan_v2_curriculum \\
        --baselines-root results \\
        --seeds 42 101 777 2026 9999 1234 2024 31415 27182 7 11 13 888 555 333 99 1001 5050 8080 1998 \\
        --output results/figures/baseline_comparison_seed_stability.pdf
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from metrics_full import statistical_comparison, holm_bonferroni_correction

BASELINE_REGISTRY = [
    ("ResNet1D",          "resnet1d",    "#adb5bd"),
    ("Transformer",       "transformer", "#c9a84c"),
    ("CNN + Focal Loss",  "cnn_focal",   "#8fbcad"),
    ("B-Spline KAN",      "bspline_kan", "#6d8bb0"),
]
WAVKAN_COLOR = "#1f4e79"


def load_macro_f1_by_seed(run_dir: str, seeds: list) -> dict:
    """Returns {seed: macro_f1} for every seed with a real test_metrics.json --
    seeds without a result are simply absent from the dict, never filled with a
    placeholder, so pairing downstream is always by real seed identity."""
    base = Path(run_dir)
    out = {}
    for seed in seeds:
        p = base / f"seed_{seed}" / "test_metrics.json"
        if p.exists():
            with open(p) as f:
                out[seed] = json.load(f)["macro_f1"]
    return out


def paired_values(a: dict, b: dict, seeds: list):
    """Returns (a_vals, b_vals) restricted to seeds present in BOTH dicts, in a
    fixed, identical order -- the exact thing AUDIT_FINDINGS.md H8 found missing
    elsewhere (pairing by list length/position instead of seed identity)."""
    common = [s for s in seeds if s in a and s in b]
    return [a[s] for s in common], [b[s] for s in common], common


def build_plot_data(wavkan_dir: str, baselines_root: str, seeds: list) -> dict:
    wavkan_f1 = load_macro_f1_by_seed(wavkan_dir, seeds)

    series = [("WavKAN-v2\n(Ours)", list(wavkan_f1.values()), WAVKAN_COLOR, None)]
    stats = {}
    for label, key, color in BASELINE_REGISTRY:
        baseline_f1 = load_macro_f1_by_seed(f"{baselines_root}/baseline_{key}", seeds)
        wavkan_vals, baseline_vals, common_seeds = paired_values(wavkan_f1, baseline_f1, seeds)
        series.append((label, list(baseline_f1.values()), color, key))
        stats[key] = {
            "n_paired": len(common_seeds),
            "paired_seeds": common_seeds,
            "wavkan_vals": wavkan_vals,
            "baseline_vals": baseline_vals,
        }
    return {"wavkan_f1": wavkan_f1, "series": series, "pairwise_stats": stats}


def compute_pairwise_tests(data: dict) -> dict:
    """One comparison per baseline (WavKAN-v2 vs. that baseline), each via
    metrics_full.statistical_comparison (adds Cohen's d, guards n>=6/non-zero variance
    the same way aggregate_seed_results.py now does), then Holm-Bonferroni-corrected
    across all baselines actually compared here -- 4 p-values from one figure without
    correction otherwise inflates the family-wise Type I error rate."""
    results = {}
    keys_with_p = []
    for key, s in data["pairwise_stats"].items():
        n = s["n_paired"]
        if n >= 6 and not _all_equal(s["wavkan_vals"], s["baseline_vals"]):
            cmp = statistical_comparison(s["wavkan_vals"], s["baseline_vals"], metric_name=key)
            results[key] = {
                "n": n, "p_two_sided": cmp["p_value"],
                "cohens_d": cmp["cohens_d"], "effect_size": cmp["effect_size"],
            }
            keys_with_p.append(key)
        else:
            results[key] = {
                "n": n, "p_two_sided": None,
                "skipped_reason": f"only {n} paired seed(s), or identical values -- "
                                   "need >=6 for a meaningful Wilcoxon test",
            }

    if keys_with_p:
        raw_ps = [results[k]["p_two_sided"] for k in keys_with_p]
        adjusted = holm_bonferroni_correction(raw_ps)
        for k, adj_p in zip(keys_with_p, adjusted):
            results[k]["holm_adjusted_p"] = float(adj_p)
            results[k]["significant_after_holm_correction"] = bool(adj_p < 0.05)

    return results


def _all_equal(a, b):
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def plot_fig3(data: dict, test_results: dict, save_path: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Publication-format pass (2026-09-02, project owner request): serif
    # typography and bolded title/axis labels to match the paper's other
    # figures restyled this session. Styling only -- no change to any
    # statistic computed above this function.
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
    })

    fig, ax = plt.subplots(figsize=(10, 6.4))
    labels = [s[0] for s in data["series"]]
    values = [s[1] for s in data["series"]]
    colors = [s[2] for s in data["series"]]
    positions = list(range(1, len(labels) + 1))

    bp = ax.boxplot([v if v else [float("nan")] for v in values], positions=positions,
                     patch_artist=True, medianprops={"lw": 2.0, "color": "white"})
    for patch, color, v in zip(bp["boxes"], colors, values):
        patch.set_facecolor(color)
        patch.set_alpha(0.88 if v else 0.15)

    rng = np.random.default_rng(0)  # fixed seed for jitter reproducibility, not results
    for i, (vals, color) in enumerate(zip(values, colors), 1):
        if vals:
            jit = rng.uniform(-0.08, 0.08, len(vals))
            ax.scatter(np.full(len(vals), i) + jit, vals, color="#222", s=26,
                       alpha=0.75, zorder=5, edgecolors="white", linewidths=0.4)

    for i, (label, key, color) in enumerate([(l, s[3], s[2]) for l, s in zip(labels, data["series"])], 1):
        if key is None or key not in test_results:
            continue
        tr = test_results[key]
        if tr["p_two_sided"] is not None:
            # Marker reflects the Holm-corrected significance, not the raw p-value --
            # showing raw-significant stars that don't survive correction for testing
            # 4 baselines from one figure would overstate the result.
            sig = tr.get("significant_after_holm_correction", False)
            marker = "*" if sig else "ns"
            ax.text(i, max(values[i - 1]) + 0.01 if values[i - 1] else 0,
                    f"{marker} (d={tr['cohens_d']:+.2f}, n={tr['n']})", ha="center", fontsize=8)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Macro-F1 (DS2 Test)", fontsize=11.5, fontweight="bold")
    ax.set_title("Seed Stability Analysis --- WavKAN-v2 vs. 4 Baselines\n"
                  "(real per-seed data; * = Holm-significant, ns = not significant after correction, "
                  "two-sided Wilcoxon vs. WavKAN-v2, paired by seed identity)",
                  fontsize=10.5, fontweight="bold", pad=16)
    ax.grid(axis="y", alpha=0.3)
    for spine in ax.spines.values():
        spine.set_color("#333333")
        spine.set_linewidth(0.9)

    # Explicit headroom above the highest significance annotation so the
    # (now taller, bolded, two-line) title never overlaps plot content --
    # was a real visual collision in an earlier pass of this restyling,
    # caught by the project owner from a rendered figure.
    all_vals = [v for series_vals in values if series_vals for v in series_vals]
    y_top = max(all_vals) + 0.045 if all_vals else 1.0
    y_bottom = min(all_vals) - 0.01 if all_vals else 0.0
    ax.set_ylim(y_bottom, y_top)

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavkan-dir", type=str, default="results/wavkan_v2_curriculum")
    parser.add_argument("--baselines-root", type=str, default="results")
    parser.add_argument("--seeds", type=int, nargs="+",
                        default=[42, 101, 777, 2026, 9999, 1234, 2024, 31415, 27182,
                                 7, 11, 13, 888, 555, 333, 99, 1001, 5050, 8080, 1998])
    parser.add_argument("--output", type=str, default="results/figures/baseline_comparison_seed_stability.pdf")
    parser.add_argument("--report", type=str, default="results/fig3_seed_stability_report.json")
    args = parser.parse_args()

    data = build_plot_data(args.wavkan_dir, args.baselines_root, args.seeds)
    test_results = compute_pairwise_tests(data)

    print(f"WavKAN-v2: {len(data['wavkan_f1'])}/{len(args.seeds)} seeds found")
    for key, tr in test_results.items():
        d_str = f"{tr['cohens_d']:+.2f} ({tr['effect_size']})" if "cohens_d" in tr else "N/A"
        holm_str = f"{tr['holm_adjusted_p']:.4f}" if "holm_adjusted_p" in tr else "N/A"
        print(f"  vs {key}: n_paired={tr['n']}  p={tr.get('p_two_sided')}  "
              f"holm_p={holm_str}  d={d_str}")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump({"wavkan_f1": data["wavkan_f1"], "tests": test_results}, f, indent=2)

    plot_fig3(data, test_results, args.output)
