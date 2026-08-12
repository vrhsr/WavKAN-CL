"""
aggregate_seed_results.py — Honest multi-seed aggregation for the Phase 5 re-run.

Written fresh for the Phase 5 WavKAN_v2 baseline-vs-curriculum re-run rather than reused
from an existing script, specifically to avoid the statistical bugs AUDIT_FINDINGS.md
found in the codebase's other aggregation paths (H8):
  - generate_all_figures.py's Wilcoxon call checks equal LIST LENGTH between arms, not
    equal SEED IDENTITY -- if each arm independently drops a different seed, both lists
    can end up the same (shorter) length and get silently paired wrong.
  - statistical_validation.py truncates to min(len) and never aligns by seed ID at all.
  - Two live call sites use inconsistent one-sided vs two-sided tests for nominally the
    same comparison, one of them hardcoded to favor "ours > baseline".

This script pairs results by SEED VALUE explicitly (an inner join on seed, not a
truncate-by-position), reports which seeds are missing from either arm instead of
silently dropping them, and always uses a two-sided Wilcoxon signed-rank test (no
directional assumption about which arm should win). No test-set peeking anywhere: this
only ever reads each seed's already-computed test_metrics.json -- it does not select,
rank, or filter seeds by any metric. Every seed that has a result gets averaged in.

Usage:
    python src/aggregate_seed_results.py \\
        --baseline-dir results/wavkan_v2_baseline \\
        --curriculum-dir results/wavkan_v2_curriculum \\
        --output results/wavkan_v2_20seed_comparison.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

METRICS = ["macro_f1", "v_recall", "s_recall", "n_recall", "f_recall"]


def load_seed_metrics(run_dir: str) -> dict:
    """Returns {seed_int: {metric_name: value}} for every seed_* subdir with a real
    test_metrics.json. Missing/unreadable seeds are reported, not silently skipped."""
    base = Path(run_dir)
    results = {}
    missing = []
    if not base.exists():
        return results, missing
    for seed_dir in sorted(base.glob("seed_*")):
        metrics_path = seed_dir / "test_metrics.json"
        if not metrics_path.exists():
            missing.append(seed_dir.name)
            continue
        with open(metrics_path) as f:
            m = json.load(f)
        seed = m.get("seed")
        if seed is None:
            missing.append(f"{seed_dir.name} (no 'seed' field in test_metrics.json)")
            continue
        results[int(seed)] = {k: m[k] for k in METRICS if k in m}
    return results, missing


def aggregate(baseline_dir: str, curriculum_dir: str) -> dict:
    baseline, baseline_missing = load_seed_metrics(baseline_dir)
    curriculum, curriculum_missing = load_seed_metrics(curriculum_dir)

    baseline_seeds = set(baseline.keys())
    curriculum_seeds = set(curriculum.keys())
    paired_seeds = sorted(baseline_seeds & curriculum_seeds)
    baseline_only = sorted(baseline_seeds - curriculum_seeds)
    curriculum_only = sorted(curriculum_seeds - baseline_seeds)

    report = {
        "n_baseline_seeds_found": len(baseline_seeds),
        "n_curriculum_seeds_found": len(curriculum_seeds),
        "n_paired_seeds": len(paired_seeds),
        "paired_seeds": paired_seeds,
        "baseline_seeds_missing_from_curriculum": baseline_only,
        "curriculum_seeds_missing_from_baseline": curriculum_only,
        "baseline_dirs_with_no_result_file": baseline_missing,
        "curriculum_dirs_with_no_result_file": curriculum_missing,
        "metrics": {},
    }

    if len(paired_seeds) < len(baseline_seeds) or len(paired_seeds) < len(curriculum_seeds):
        print(
            f"WARNING: only {len(paired_seeds)} seeds are present in BOTH arms "
            f"({len(baseline_only)} baseline-only, {len(curriculum_only)} curriculum-only "
            f"seed(s) excluded from the paired statistics below). Re-run the missing "
            f"seeds before treating this as the final 20-seed comparison."
        )

    for metric in METRICS:
        if not paired_seeds:
            continue  # no seeds present in both arms at all -- nothing to aggregate
        b_vals = [baseline[s][metric] for s in paired_seeds if metric in baseline[s]]
        c_vals = [curriculum[s][metric] for s in paired_seeds if metric in curriculum[s]]
        if len(b_vals) != len(paired_seeds) or len(c_vals) != len(paired_seeds):
            continue  # metric missing for some paired seed; skip rather than misreport

        entry = {
            "baseline_mean": float(np.mean(b_vals)),
            "baseline_std": float(np.std(b_vals, ddof=1)) if len(b_vals) > 1 else 0.0,
            "curriculum_mean": float(np.mean(c_vals)),
            "curriculum_std": float(np.std(c_vals, ddof=1)) if len(c_vals) > 1 else 0.0,
            "n": len(paired_seeds),
        }

        # Two-sided paired Wilcoxon signed-rank test, seed-aligned by construction
        # (b_vals[i] and c_vals[i] both come from paired_seeds[i]). Requires n>=6 for a
        # meaningful exact test at alpha=0.05 (matching the one clean positive-finding
        # pattern found in metrics_full.statistical_comparison during Phase 2).
        if len(paired_seeds) >= 6 and not np.allclose(b_vals, c_vals):
            stat, p = wilcoxon(b_vals, c_vals, alternative="two-sided")
            entry["wilcoxon_p_two_sided"] = float(p)
        else:
            entry["wilcoxon_p_two_sided"] = None
            entry["wilcoxon_skipped_reason"] = (
                "fewer than 6 paired seeds, or values identical" if len(paired_seeds) < 6
                else "baseline and curriculum values identical for every paired seed"
            )

        report["metrics"][metric] = entry

    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=str, default="results/wavkan_v2_baseline")
    parser.add_argument("--curriculum-dir", type=str, default="results/wavkan_v2_curriculum")
    parser.add_argument("--output", type=str, default="results/wavkan_v2_20seed_comparison.json")
    args = parser.parse_args()

    report = aggregate(args.baseline_dir, args.curriculum_dir)

    print(f"\n{'='*60}")
    print(f"Paired seeds: {report['n_paired_seeds']} "
          f"(baseline found: {report['n_baseline_seeds_found']}, "
          f"curriculum found: {report['n_curriculum_seeds_found']})")
    if report["baseline_seeds_missing_from_curriculum"]:
        print(f"Baseline-only seeds (excluded): {report['baseline_seeds_missing_from_curriculum']}")
    if report["curriculum_seeds_missing_from_baseline"]:
        print(f"Curriculum-only seeds (excluded): {report['curriculum_seeds_missing_from_baseline']}")
    print(f"{'='*60}")
    for metric, entry in report["metrics"].items():
        p_str = f"{entry['wilcoxon_p_two_sided']:.4f}" if entry["wilcoxon_p_two_sided"] is not None else "N/A"
        print(f"  {metric:10s}  baseline={entry['baseline_mean']:.4f}±{entry['baseline_std']:.4f}  "
              f"curriculum={entry['curriculum_mean']:.4f}±{entry['curriculum_std']:.4f}  "
              f"p={p_str}  (n={entry['n']})")
    print(f"{'='*60}\n")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Full report saved to {args.output}")
