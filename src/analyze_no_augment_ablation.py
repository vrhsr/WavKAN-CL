"""
analyze_no_augment_ablation.py -- H36: does removing default augmentation
improve the paper's own final-configuration headline Macro-F1?

Background (AUDIT_FINDINGS.md H36): the paper's adopted final-configuration
headline number is Table tab:family_ablation's "No RR self-attention" row,
Macro-F1=0.357, from results/ablation_no_rr_attn/ -- trained with
train_pca.py's DEFAULT use_augment=True (Combined-family augmentation:
Gaussian noise + baseline wander + amplitude scaling). Separately, the
completed Phase 7 augmentation study (results/noise_augmentation_final/)
found this same augmentation family costs real Macro-F1 under the final
configuration with no significant S-Recall benefit anymore -- but that
study used its own isolated recipe (60 epochs, no curriculum), never the
actual headline training recipe. This script compares the REAL headline
recipe's own with-augment vs. no-augment arms directly, once both exist.

This is a real, GPU-produced comparison -- this script only aggregates and
tests significance on already-real per-seed test_metrics.json files. It
never trains anything and never touches which seed to report (all seeds
present in both arms are used, always).

Inputs (all real, produced by run_gpu_pipeline_phase9.sh):
  results/ablation_no_rr_attn/seed_*/test_metrics.json                (augment=True, curriculum=True -- existing headline)
  results/ablation_no_rr_attn_no_augment/seed_*/test_metrics.json     (augment=False, curriculum=True -- new)
  results/ablation_no_rr_attn_no_curriculum_no_augment/seed_*/test_metrics.json (augment=False, curriculum=False -- new, for completeness)

Output:
  results/h36_no_augment_ablation_report.json

Usage:
    python src/analyze_no_augment_ablation.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from metrics_full import statistical_comparison, holm_bonferroni_correction

METRICS = ["macro_f1", "v_recall", "s_recall", "f_recall"]

ARMS = {
    "final_curriculum_augment":    "results/ablation_no_rr_attn",
    "final_curriculum_no_augment": "results/ablation_no_rr_attn_no_augment",
    "final_no_curriculum_no_augment": "results/ablation_no_rr_attn_no_curriculum_no_augment",
}


def load_by_seed(run_dir: str) -> dict:
    """{seed: test_metrics dict} for every seed with a real, already-produced
    test_metrics.json -- seeds without a result are simply absent, never
    filled with a placeholder."""
    base = Path(run_dir)
    out = {}
    if not base.exists():
        return out
    for seed_dir in base.glob("seed_*"):
        p = seed_dir / "test_metrics.json"
        if p.exists():
            with open(p) as f:
                m = json.load(f)
            out[m["seed"]] = m
    return out


def paired(a: dict, b: dict, metric: str):
    """Values for `metric` restricted to seeds present in BOTH dicts, in a
    fixed order determined by seed identity -- never by list position."""
    common = sorted(set(a) & set(b))
    return [a[s][metric] for s in common], [b[s][metric] for s in common], common


def compare(a: dict, b: dict, label_a: str, label_b: str) -> dict:
    result = {"n_a": len(a), "n_b": len(b), "seeds_a": sorted(a.keys()), "seeds_b": sorted(b.keys())}
    per_metric = {}
    raw_ps = []
    metric_order = []
    for metric in METRICS:
        a_vals, b_vals, common = paired(a, b, metric)
        entry = {
            "n_paired": len(common),
            "paired_seeds": common,
            f"{label_a}_mean": float(sum(a_vals) / len(a_vals)) if a_vals else None,
            f"{label_b}_mean": float(sum(b_vals) / len(b_vals)) if b_vals else None,
        }
        if len(common) >= 6 and a_vals != b_vals:
            cmp = statistical_comparison(b_vals, a_vals, metric_name=metric)
            entry["p_value"] = cmp["p_value"]
            entry["cohens_d"] = cmp["cohens_d"]
            entry["effect_size"] = cmp["effect_size"]
            raw_ps.append(cmp["p_value"])
            metric_order.append(metric)
        else:
            entry["skipped_reason"] = f"only {len(common)} paired seed(s), or identical values"
        per_metric[metric] = entry

    if raw_ps:
        adjusted = holm_bonferroni_correction(raw_ps)
        for metric, adj_p in zip(metric_order, adjusted):
            per_metric[metric]["holm_adjusted_p"] = float(adj_p)
            per_metric[metric]["significant_after_holm_correction"] = bool(adj_p < 0.05)

    result["per_metric"] = per_metric
    return result


def main():
    arms = {name: load_by_seed(path) for name, path in ARMS.items()}
    for name, seeds in arms.items():
        print(f"{name}: {len(seeds)} real seeds found ({ARMS[name]})")

    report = {"arms_found": {name: sorted(seeds.keys()) for name, seeds in arms.items()}}

    if arms["final_curriculum_augment"] and arms["final_curriculum_no_augment"]:
        print("\n=== H36 primary question: does removing augmentation change the headline "
              "(curriculum ON, final config) Macro-F1? ===")
        cmp = compare(arms["final_curriculum_augment"], arms["final_curriculum_no_augment"],
                      "with_augment", "no_augment")
        report["headline_curriculum_augment_vs_no_augment"] = cmp
        m = cmp["per_metric"]["macro_f1"]
        print(f"  Macro-F1: with_augment={m.get('with_augment_mean')}  "
              f"no_augment={m.get('no_augment_mean')}  "
              f"holm_p={m.get('holm_adjusted_p')}  d={m.get('cohens_d')} ({m.get('effect_size')})")
        s = cmp["per_metric"]["s_recall"]
        print(f"  S-Recall: with_augment={s.get('with_augment_mean')}  "
              f"no_augment={s.get('no_augment_mean')}  "
              f"holm_p={s.get('holm_adjusted_p')}  d={s.get('cohens_d')} ({s.get('effect_size')})")
    else:
        print("\n[SKIPPED] headline curriculum with-vs-without-augment comparison -- "
              "one or both arms not yet complete.")

    if arms["final_no_curriculum_no_augment"] and arms["final_curriculum_no_augment"]:
        print("\n=== Secondary: within no-augment, does curriculum still help "
              "(mirrors tab:curriculum_stats, but on the final config)? ===")
        cmp2 = compare(arms["final_no_curriculum_no_augment"], arms["final_curriculum_no_augment"],
                       "no_curriculum", "curriculum")
        report["no_augment_curriculum_effect"] = cmp2
        s = cmp2["per_metric"]["s_recall"]
        print(f"  S-Recall: no_curriculum={s.get('no_curriculum_mean')}  "
              f"curriculum={s.get('curriculum_mean')}  "
              f"holm_p={s.get('holm_adjusted_p')}  d={s.get('cohens_d')} ({s.get('effect_size')})")
    else:
        print("\n[SKIPPED] no-augment curriculum-effect comparison -- "
              "one or both arms not yet complete.")

    out_path = Path("results/h36_no_augment_ablation_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
