"""Regression tests for src/e3c_compliance.py.

Fixed per AUDIT_FINDINGS.md C6: `check_criterion_1`, `check_criterion_3`, and
`check_criterion_4` used to set `status="PASS", score=1.0` unconditionally at the end of
the function, regardless of what evidence (if any) was actually gathered above -- they
could not fail. `check_criterion_1` in particular only ever compared two hardcoded
literal DS1/DS2 sets to each other, never a real generated split artifact.

These tests assert the specific behavior that was missing: each criterion must report
something other than a flat PASS when the real evidence it needs is absent, and
`check_criterion_1`/`check_criterion_4` must be able to genuinely FAIL given bad input.
"""
from src.e3c_compliance import check_criterion_1, check_criterion_3, check_criterion_4

NONEXISTENT_DIR = "results/__no_such_dir_for_tests__"


def test_criterion_1_partial_without_real_split_artifacts():
    """Without real ids_train.npy/ids_test.npy on disk, criterion 1 can only compare
    hardcoded literals to each other -- it must report PARTIAL, not PASS."""
    c1 = check_criterion_1(NONEXISTENT_DIR, "wavkan_v2")
    assert c1.status == "PARTIAL"
    assert c1.score == 0.5


def test_criterion_3_partial_without_real_label_file():
    """Without a real y_test.npy on disk, criterion 3 has not actually checked any real
    annotations against the AAMI mapping -- it must report PARTIAL, not PASS."""
    c3 = check_criterion_3(NONEXISTENT_DIR, "wavkan_v2")
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
