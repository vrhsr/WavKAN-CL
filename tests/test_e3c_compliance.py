"""Regression tests for src/e3c_compliance.py.

Fixed per AUDIT_FINDINGS.md C6: `check_criterion_1`, `check_criterion_3`, and
`check_criterion_4` used to set `status="PASS", score=1.0` unconditionally at the end of
the function, regardless of what evidence (if any) was actually gathered above -- they
could not fail. `check_criterion_1` in particular only ever compared two hardcoded
literal DS1/DS2 sets to each other, never a real generated split artifact.

These tests assert the specific behavior that was missing: each criterion must report
something other than a flat PASS when the real evidence it needs is absent, and
`check_criterion_1`/`check_criterion_4` must be able to genuinely FAIL given bad input.

Fixed again 2026-08-12, found live on the real GPU box: `check_criterion_1`/`_3`
originally hardcoded the data path (`data/processed_rr_history`) instead of taking it
as a parameter, so passing a nonexistent `results_dir` here did NOT actually simulate
"no real evidence" on any machine that had *already run the data pipeline* -- it just
found the real `data/processed_rr_history/ids_train.npy` etc. that genuinely existed
there and correctly returned PASS. The bug was in this test's isolation, not in the
function: two tests below now pass an explicit, guaranteed-empty `data_dir` (via
`tmp_path`) instead of relying on a real directory elsewhere in the repo happening not
to exist.
"""
from pathlib import Path

from src.e3c_compliance import check_criterion_1, check_criterion_3, check_criterion_4

NONEXISTENT_DIR = "results/__no_such_dir_for_tests__"


def test_criterion_1_partial_without_real_split_artifacts(tmp_path):
    """Without real ids_train.npy/ids_test.npy on disk, criterion 1 can only compare
    hardcoded literals to each other -- it must report PARTIAL, not PASS."""
    empty_data_dir = tmp_path / "no_such_data_dir"
    c1 = check_criterion_1(NONEXISTENT_DIR, "wavkan_v2", data_dir=str(empty_data_dir))
    assert c1.status == "PARTIAL"
    assert c1.score == 0.5


def test_criterion_1_passes_with_real_disjoint_split_artifacts(tmp_path):
    """With real, disjoint ids_train.npy/ids_test.npy present, criterion 1 must PASS --
    covers the case AUDIT_FINDINGS.md M17 asked for (a real check that CAN pass, not
    just one that can now fail)."""
    import numpy as np
    data_dir = tmp_path / "real_data_dir"
    data_dir.mkdir()
    np.save(data_dir / "ids_train.npy", np.array(["101_0", "101_1", "106_0"]))
    np.save(data_dir / "ids_test.npy", np.array(["100_0", "100_1"]))
    c1 = check_criterion_1(NONEXISTENT_DIR, "wavkan_v2", data_dir=str(data_dir))
    assert c1.status == "PASS"
    assert c1.score == 1.0


def test_criterion_1_fails_with_real_overlapping_split_artifacts(tmp_path):
    """If the real generated ids_train/ids_test arrays actually overlap, criterion 1
    must FAIL -- the scenario the hardcoded-literal check could never catch at all."""
    import numpy as np
    data_dir = tmp_path / "leaky_data_dir"
    data_dir.mkdir()
    np.save(data_dir / "ids_train.npy", np.array(["101_0", "101_1", "999_0"]))
    np.save(data_dir / "ids_test.npy", np.array(["999_0", "100_1"]))  # "999_0" leaks
    c1 = check_criterion_1(NONEXISTENT_DIR, "wavkan_v2", data_dir=str(data_dir))
    assert c1.status == "FAIL"
    assert c1.score == 0.0


def test_criterion_3_partial_without_real_label_file(tmp_path):
    """Without a real y_test.npy on disk, criterion 3 has not actually checked any real
    annotations against the AAMI mapping -- it must report PARTIAL, not PASS."""
    empty_data_dir = tmp_path / "no_such_data_dir"
    c3 = check_criterion_3(NONEXISTENT_DIR, "wavkan_v2", data_dir=str(empty_data_dir))
    assert c3.status == "PARTIAL"
    assert c3.score == 0.5


def test_criterion_4_fails_when_both_thresholds_are_exceeded():
    """A model with far more parameters and latency than the stated thresholds must FAIL,
    not PASS -- this is the exact scenario the original hardcoded PASS could never catch."""
    c4 = check_criterion_4(
        NONEXISTENT_DIR, "wavkan_v2",
        n_params=999_999_999, latency_ms=999.0,
        max_params=200_000, max_latency=5.0,
    )
    assert c4.status == "FAIL"
    assert c4.score == 0.0


def test_criterion_4_partial_when_only_one_threshold_exceeded():
    c4 = check_criterion_4(
        NONEXISTENT_DIR, "wavkan_v2",
        n_params=100, latency_ms=999.0,  # params fine, latency way over
        max_params=200_000, max_latency=5.0,
    )
    assert c4.status == "PARTIAL"
    assert c4.score == 0.5


def test_criterion_4_passes_when_both_thresholds_met():
    c4 = check_criterion_4(
        NONEXISTENT_DIR, "wavkan_v2",
        n_params=95_189, latency_ms=0.78,
        max_params=200_000, max_latency=5.0,
    )
    assert c4.status == "PASS"
    assert c4.score == 1.0
