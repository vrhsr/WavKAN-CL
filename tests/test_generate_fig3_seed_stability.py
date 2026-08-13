"""Regression tests for src/generate_fig3_seed_stability.py's data-pairing logic
(the plotting itself needs real baseline-model data, not yet trained -- these tests
use synthetic test_metrics.json files to verify the pairing/stats logic in isolation,
the same way tests/test_aggregate_seed_results.py did for the main 20-seed comparison).
"""
import json

import pytest

from src.generate_fig3_seed_stability import (
    build_plot_data, compute_pairwise_tests, load_macro_f1_by_seed, paired_values,
)


def _write_metrics(base_dir, seed, macro_f1):
    seed_dir = base_dir / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    (seed_dir / "test_metrics.json").write_text(json.dumps({"macro_f1": macro_f1}))


def test_load_macro_f1_by_seed_skips_missing_seeds_silently(tmp_path):
    _write_metrics(tmp_path, 42, 0.32)
    result = load_macro_f1_by_seed(str(tmp_path), seeds=[42, 101, 777])
    assert result == {42: 0.32}  # 101, 777 simply absent, not filled with a placeholder


def test_paired_values_only_uses_seeds_present_in_both():
    a = {1: 0.1, 2: 0.2, 3: 0.3}
    b = {2: 0.25, 3: 0.35, 4: 0.45}
    a_vals, b_vals, common = paired_values(a, b, seeds=[1, 2, 3, 4])
    assert common == [2, 3]           # 1 (a-only) and 4 (b-only) excluded
    assert a_vals == [0.2, 0.3]
    assert b_vals == [0.25, 0.35]


def test_build_plot_data_pairs_by_identity_not_position(tmp_path):
    """The H8-class bug this script is specifically designed to avoid: WavKAN missing
    seed 42, a baseline missing a DIFFERENT seed (101) -- must not silently pair
    WavKAN's seed 101 value against the baseline's seed 42 value just because both
    lists end up the same length."""
    wavkan_dir = tmp_path / "wavkan"
    baselines_root = tmp_path / "baselines_root"
    _write_metrics(wavkan_dir, 101, 0.40)   # WavKAN missing seed 42
    _write_metrics(wavkan_dir, 777, 0.41)
    _write_metrics(baselines_root / "baseline_resnet1d", 42, 0.10)  # resnet1d missing 101
    _write_metrics(baselines_root / "baseline_resnet1d", 777, 0.11)

    data = build_plot_data(str(wavkan_dir), str(baselines_root), seeds=[42, 101, 777])

    resnet_stats = data["pairwise_stats"]["resnet1d"]
    assert resnet_stats["paired_seeds"] == [777]  # only the seed present in BOTH
    assert resnet_stats["wavkan_vals"] == [0.41]
    assert resnet_stats["baseline_vals"] == [0.11]
    # If this had wrongly paired by position, seed 101 (WavKAN=0.40) would have been
    # paired against resnet1d's seed 42 (0.10) -- confirm that never appears together.
    assert 0.40 not in resnet_stats["wavkan_vals"]


def test_compute_pairwise_tests_skips_below_n6():
    data = {"pairwise_stats": {
        "resnet1d": {"n_paired": 3, "wavkan_vals": [0.3, 0.31, 0.32], "baseline_vals": [0.1, 0.11, 0.12]}
    }}
    results = compute_pairwise_tests(data)
    assert results["resnet1d"]["p_two_sided"] is None
    assert "skipped_reason" in results["resnet1d"]


def test_compute_pairwise_tests_runs_wilcoxon_with_enough_seeds():
    wavkan_vals = [0.40, 0.41, 0.42, 0.39, 0.43, 0.44]
    baseline_vals = [0.10, 0.11, 0.09, 0.12, 0.10, 0.11]
    data = {"pairwise_stats": {
        "resnet1d": {"n_paired": 6, "wavkan_vals": wavkan_vals, "baseline_vals": baseline_vals}
    }}
    results = compute_pairwise_tests(data)
    assert results["resnet1d"]["p_two_sided"] is not None
    assert 0.0 <= results["resnet1d"]["p_two_sided"] <= 1.0
    assert results["resnet1d"]["effect_size"] == "large"  # 0.40 vs 0.10-ish is a huge gap
    assert results["resnet1d"]["cohens_d"] > 0.8


def test_compute_pairwise_tests_applies_holm_correction_across_baselines():
    """Added 2026-08-13 for top-tier rigor: 4 baseline comparisons from one figure need
    correcting for multiple comparisons, same upgrade as aggregate_seed_results.py."""
    wavkan_vals = [0.40, 0.41, 0.42, 0.39, 0.43, 0.44]
    strong_baseline = [0.10, 0.11, 0.09, 0.12, 0.10, 0.11]   # huge, clear gap
    close_baseline  = [0.39, 0.40, 0.41, 0.38, 0.42, 0.43]   # barely different from WavKAN
    data = {"pairwise_stats": {
        "resnet1d":    {"n_paired": 6, "wavkan_vals": wavkan_vals, "baseline_vals": strong_baseline},
        "transformer": {"n_paired": 6, "wavkan_vals": wavkan_vals, "baseline_vals": close_baseline},
    }}
    results = compute_pairwise_tests(data)

    for key in ("resnet1d", "transformer"):
        assert "holm_adjusted_p" in results[key]
        assert results[key]["holm_adjusted_p"] >= results[key]["p_two_sided"] - 1e-12
        assert "significant_after_holm_correction" in results[key]
