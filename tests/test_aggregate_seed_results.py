"""Regression tests for src/aggregate_seed_results.py.

The whole point of writing this script fresh (rather than reusing an existing
aggregator) was to avoid AUDIT_FINDINGS.md H8's seed-pairing bugs -- these tests prove
the specific failure mode H8 described (mismatched seeds silently paired by list
position/length instead of by seed identity) does NOT happen here, using synthetic
`test_metrics.json` fixtures instead of real training results.
"""
import json

import pytest

from src.aggregate_seed_results import aggregate, load_seed_metrics


def _write_seed_result(base_dir, seed, macro_f1, v_recall=0.5, s_recall=0.3):
    seed_dir = base_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    with open(seed_dir / "test_metrics.json", "w") as f:
        json.dump({
            "seed": seed, "macro_f1": macro_f1, "v_recall": v_recall,
            "s_recall": s_recall, "n_recall": 0.9, "f_recall": 0.1,
        }, f)


def test_load_seed_metrics_reports_missing_result_files(tmp_path):
    run_dir = tmp_path / "run"
    _write_seed_result(run_dir, 42, macro_f1=0.36)
    (run_dir / "seed_101").mkdir(parents=True)  # dir exists, no test_metrics.json inside

    results, missing = load_seed_metrics(str(run_dir))

    assert results == {42: {"macro_f1": 0.36, "v_recall": 0.5, "s_recall": 0.3,
                             "n_recall": 0.9, "f_recall": 0.1}}
    assert missing == ["seed_101"]


def test_aggregate_pairs_by_seed_identity_not_by_list_position(tmp_path):
    """The exact scenario H8 got wrong: baseline drops seed 42, curriculum drops seed 7
    -- both arms end up with the same COUNT of seeds, but they must NOT be paired by
    list position (which would silently mismatch seed 101's baseline result against
    seed 999's curriculum result, say). Only seeds present in BOTH arms should be used.
    """
    baseline_dir = tmp_path / "baseline"
    curriculum_dir = tmp_path / "curriculum"

    # Baseline has seeds 7, 101, 999. Curriculum has seeds 42, 101, 999.
    # Only 101 and 999 are common -- seed 42 (curriculum-only) and seed 7
    # (baseline-only) must be excluded from the paired statistics.
    _write_seed_result(baseline_dir, 7,   macro_f1=0.10)  # baseline-only
    _write_seed_result(baseline_dir, 101, macro_f1=0.30)
    _write_seed_result(baseline_dir, 999, macro_f1=0.32)
    _write_seed_result(curriculum_dir, 42,  macro_f1=0.90)  # curriculum-only
    _write_seed_result(curriculum_dir, 101, macro_f1=0.34)
    _write_seed_result(curriculum_dir, 999, macro_f1=0.36)

    report = aggregate(str(baseline_dir), str(curriculum_dir))

    assert report["n_paired_seeds"] == 2
    assert report["paired_seeds"] == [101, 999]
    assert report["baseline_seeds_missing_from_curriculum"] == [7]
    assert report["curriculum_seeds_missing_from_baseline"] == [42]

    # If this had wrongly paired-by-position, macro_f1 baseline_mean would include 0.10
    # (seed 7) and curriculum_mean would include 0.90 (seed 42) -- it must not.
    entry = report["metrics"]["macro_f1"]
    assert entry["baseline_mean"] == pytest.approx((0.30 + 0.32) / 2)
    assert entry["curriculum_mean"] == pytest.approx((0.34 + 0.36) / 2)
    assert 0.10 not in (entry["baseline_mean"],)
    assert 0.90 not in (entry["curriculum_mean"],)


def test_aggregate_skips_wilcoxon_below_n6_instead_of_reporting_a_fake_pvalue(tmp_path):
    baseline_dir = tmp_path / "baseline"
    curriculum_dir = tmp_path / "curriculum"
    for seed in [1, 2, 3]:
        _write_seed_result(baseline_dir, seed, macro_f1=0.30 + seed * 0.01)
        _write_seed_result(curriculum_dir, seed, macro_f1=0.35 + seed * 0.01)

    report = aggregate(str(baseline_dir), str(curriculum_dir))

    assert report["n_paired_seeds"] == 3
    assert report["metrics"]["macro_f1"]["wilcoxon_p_two_sided"] is None
    assert "wilcoxon_skipped_reason" in report["metrics"]["macro_f1"]


def test_aggregate_runs_two_sided_wilcoxon_with_enough_paired_seeds(tmp_path):
    baseline_dir = tmp_path / "baseline"
    curriculum_dir = tmp_path / "curriculum"
    for seed in range(20):
        _write_seed_result(baseline_dir, seed, macro_f1=0.30, v_recall=0.87)
        _write_seed_result(curriculum_dir, seed, macro_f1=0.29, v_recall=0.90)

    report = aggregate(str(baseline_dir), str(curriculum_dir))

    assert report["n_paired_seeds"] == 20
    v_entry = report["metrics"]["v_recall"]
    assert v_entry["wilcoxon_p_two_sided"] is not None
    assert v_entry["baseline_mean"] == pytest.approx(0.87)
    assert v_entry["curriculum_mean"] == pytest.approx(0.90)


def test_aggregate_handles_entirely_missing_directory_gracefully(tmp_path):
    report = aggregate(str(tmp_path / "does_not_exist"), str(tmp_path / "also_missing"))
    assert report["n_paired_seeds"] == 0
    assert report["metrics"] == {}
