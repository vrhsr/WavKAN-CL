"""Regression tests for src/paired_stats.py and the H8 defects it closes.

AUDIT_FINDINGS.md H8 recorded four statistical-implementation defects spread
across three scripts. These tests pin the behaviour that each of them violated:

  H8(a) pairing gated on equal list LENGTH rather than seed identity
  H8(b) truncation to min(len) with no seed alignment at all
  H8(c) a "p<0.05" claim asserted at n=5, where the exact two-sided Wilcoxon
        minimum is 0.0625 and so the claim is mathematically unreachable
  H8(d) a one-sided test hardcoded to the alternative "ours > baseline"
"""
import math

import numpy as np
import pytest
from scipy import stats as st

from src import paired_stats as ps


# ----------------------------------------------------------------- H8(a)/H8(b)
def test_pairs_by_seed_identity_not_by_position():
    """Two arms that dropped different seeds must pair on the intersection."""
    a = {"s1": 1.0, "s2": 2.0, "s3": 3.0, "s4": 4.0}
    b = {"s2": 9.0, "s3": 8.0, "s4": 7.0, "s5": 6.0}   # s1 missing, s5 extra
    seeds, va, vb = ps.align_by_seed(a, b)
    assert seeds == ["s2", "s3", "s4"]
    assert list(va) == [2.0, 3.0, 4.0]
    assert list(vb) == [9.0, 8.0, 7.0]


def test_equal_length_but_disjoint_seeds_would_have_mispaired():
    """The exact H8(a) failure mode: equal lengths, different seed sets.

    A length-only check would have paired these positionally and reported a
    'paired' result over unrelated runs. Alignment by identity must instead use
    only the two seeds actually in common.
    """
    a = {"s1": 1.0, "s2": 2.0, "s3": 3.0}
    b = {"s2": 2.5, "s3": 3.5, "s9": 9.9}
    assert len(a) == len(b)                      # a length check would pass
    seeds, va, vb = ps.align_by_seed(a, b)
    assert seeds == ["s2", "s3"]                 # identity check disagrees
    assert len(seeds) < len(a)


def test_no_common_seeds_raises_rather_than_returning_a_number():
    with pytest.raises(ValueError):
        ps.align_by_seed({"a": 1.0}, {"b": 2.0})


def test_paired_compare_reports_the_seeds_it_used():
    a = {f"s{i}": float(i) + 0.5 for i in range(10)}
    b = {f"s{i}": float(i) for i in range(10)}
    r = ps.paired_compare(a, b)
    assert r["n_paired"] == 10
    assert r["paired_seeds"] == sorted(a)
    assert len(r["paired_seeds"]) == r["n_paired"]


# ----------------------------------------------------------------------- H8(c)
def test_refuses_to_report_p_below_min_n():
    """At n=5 the exact two-sided minimum is 0.0625, so no p<0.05 is possible."""
    a = {f"s{i}": float(i) + 1.0 for i in range(5)}
    b = {f"s{i}": float(i) for i in range(5)}
    r = ps.paired_compare(a, b)
    assert r["n_paired"] == 5
    assert math.isnan(r["p_value"]), "a p-value at n=5 cannot support p<0.05 (H8c)"
    # the effect size is still well defined and is still reported
    assert r["cohens_d"] == r["cohens_d"]


def test_reports_p_at_min_n_and_above():
    a = {f"s{i}": float(i) + 1.0 for i in range(ps.MIN_N)}
    b = {f"s{i}": float(i) for i in range(ps.MIN_N)}
    r = ps.paired_compare(a, b)
    assert not math.isnan(r["p_value"])


# ----------------------------------------------------------------------- H8(d)
def test_test_is_two_sided():
    """A one-sided 'greater' test would report a materially smaller p when the
    reference arm happens to be higher. Pin two-sided explicitly."""
    rng = np.random.default_rng(7)
    va = rng.normal(0.36, 0.02, 20)
    vb = va - 0.004
    a = {f"s{i}": float(x) for i, x in enumerate(va)}
    b = {f"s{i}": float(x) for i, x in enumerate(vb)}
    r = ps.paired_compare(a, b)
    assert r["alternative"] == "two-sided"
    expect_two = st.wilcoxon(va, vb, alternative="two-sided")[1]
    expect_one = st.wilcoxon(va, vb, alternative="greater")[1]
    assert r["p_value"] == pytest.approx(expect_two, rel=1e-12)
    assert expect_one < expect_two, "fixture should distinguish the two alternatives"


def test_sign_convention_positive_favours_arm_a():
    a = {f"s{i}": 1.0 + 0.01 * i for i in range(10)}
    b = {f"s{i}": 0.5 + 0.01 * i for i in range(10)}
    assert ps.paired_compare(a, b)["cohens_d"] > 0
    assert ps.paired_compare(b, a)["cohens_d"] < 0


# ------------------------------------------------------------------------ Holm
def test_holm_matches_hand_computation_and_is_monotone():
    p = [0.01, 0.04, 0.03]
    adj = ps.holm(p)
    # sorted: 0.01*3=0.03, 0.03*2=0.06, 0.04*1=0.04 -> monotone: 0.03, 0.06, 0.06
    assert adj[0] == pytest.approx(0.03)
    assert adj[2] == pytest.approx(0.06)
    assert adj[1] == pytest.approx(0.06)
    srt = sorted(range(3), key=lambda i: p[i])
    vals = [adj[i] for i in srt]
    assert vals == sorted(vals), "Holm must be non-decreasing in sorted p order"


def test_holm_never_exceeds_one_and_never_decreases_a_p_value():
    p = [0.4, 0.5, 0.9]
    adj = ps.holm(p)
    assert all(x <= 1.0 for x in adj)
    assert all(a >= b - 1e-12 for a, b in zip(adj, p))


def test_holm_excludes_untestable_arms_from_the_family_size():
    """A NaN p (an arm we refused to test) must not inflate the correction."""
    with_nan = ps.holm([0.01, float("nan"), 0.02])
    without = ps.holm([0.01, 0.02])
    assert math.isnan(with_nan[1])
    assert with_nan[0] == pytest.approx(without[0])
    assert with_nan[2] == pytest.approx(without[1])


def test_holm_family_records_the_family_it_corrected_over():
    fam = {
        "x": ps.paired_compare({f"s{i}": 1.0 + 0.1 * i for i in range(10)},
                               {f"s{i}": 0.2 + 0.1 * i for i in range(10)}),
        "y": ps.paired_compare({f"s{i}": 1.0 + 0.1 * i for i in range(10)},
                               {f"s{i}": 0.9 + 0.1 * i for i in range(10)}),
    }
    out = ps.holm_family(fam)
    for k in ("x", "y"):
        assert out[k]["family_size"] == 2
        assert "holm_p" in out[k] and "significant_after_holm" in out[k]


# ------------------------------------------------------- agreement with metrics_full
def test_agrees_with_metrics_full_on_a_correctly_paired_input():
    """The canonical helper must reproduce the pipeline's existing function
    where that function is used correctly, so consolidating on paired_stats
    cannot silently change any already-published number."""
    mf = pytest.importorskip("src.metrics_full")
    rng = np.random.default_rng(11)
    va = rng.normal(0.36, 0.02, 20)
    vb = rng.normal(0.35, 0.02, 20)
    a = {f"s{i}": float(x) for i, x in enumerate(va)}
    b = {f"s{i}": float(x) for i, x in enumerate(vb)}
    mine = ps.paired_compare(a, b)
    theirs = mf.statistical_comparison(list(va), list(vb))
    assert mine["p_value"] == pytest.approx(theirs["p_value"], rel=1e-9)
    assert mine["cohens_d"] == pytest.approx(theirs["cohens_d"], rel=1e-6)
    assert mine["mean_diff"] == pytest.approx(theirs["mean_diff"], rel=1e-12)
    assert mine["effect_size"] == theirs["effect_size"]


def test_holm_agrees_with_metrics_full_correction():
    mf = pytest.importorskip("src.metrics_full")
    p = [0.001, 0.02, 0.30, 0.04, 0.9]
    assert ps.holm(p) == pytest.approx(mf.holm_bonferroni_correction(p))


# ------------------------------------------------------------------ CI sanity
def test_confidence_interval_brackets_the_mean_difference():
    rng = np.random.default_rng(3)
    va = rng.normal(0.5, 0.05, 20)
    vb = rng.normal(0.4, 0.05, 20)
    a = {f"s{i}": float(x) for i, x in enumerate(va)}
    b = {f"s{i}": float(x) for i, x in enumerate(vb)}
    r = ps.paired_compare(a, b)
    assert r["ci_low"] < r["mean_diff"] < r["ci_high"]
    n = r["n_paired"]
    diff = va - vb
    half = st.t.ppf(0.975, n - 1) * diff.std(ddof=1) / math.sqrt(n)
    assert r["ci_low"] == pytest.approx(diff.mean() - half)
    assert r["ci_high"] == pytest.approx(diff.mean() + half)
