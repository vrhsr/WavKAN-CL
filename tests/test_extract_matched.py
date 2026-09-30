"""Regression tests for src/extract_matched.py (AUDIT_FINDINGS.md H58/H59, 2026-09-30).

Synthetic tests always run. The equivalence test against process_data.py runs only
when raw MIT-BIH records are present in data/raw, and otherwise skips.
"""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import extract_matched as em  # noqa: E402


def _synthetic(n=3600):
    rng = np.random.default_rng(0)
    return rng.normal(size=n)


def test_window_puts_r_peak_at_sample_90():
    ecg = np.zeros(2000)
    ecg[1000] = 50.0                      # a spike at the annotated R-peak
    X, R, y = em.extract_beats(ecg + 1e-3 * _synthetic(2000), [1000], ["N"])
    assert X.shape == (1, 360)
    assert int(np.argmax(X[0])) == 90     # 250 ms before, 750 ms after -- not centred


def test_all_annotations_mode_reproduces_published_indexing():
    # beats at 1000, 1360, 1720; a rhythm-change '+' 10 samples before the third beat
    samples = [1000, 1360, 1710, 1720, 2080]
    symbols = ["N", "N", "+", "V", "N"]
    ecg = _synthetic(3000)
    _, R, y = em.extract_beats(ecg, samples, symbols, rr_mode="all_annotations")
    # the V beat is the third extracted beat; its RR_0 is measured from the '+'
    v = list(y).index(2)
    assert R[v, 2] == pytest.approx(0.2)           # 10/360 s, clipped up to 0.2
    _, Rb, yb = em.extract_beats(ecg, samples, symbols, rr_mode="beats_only")
    assert Rb[v, 2] == pytest.approx(1.0)          # true beat-to-beat interval, 360/360
    assert np.array_equal(y, yb)                   # same beats, same order


def test_non_beat_annotations_are_never_extracted():
    samples = [1000, 1200, 1360]
    symbols = ["N", "~", "N"]
    _, _, y = em.extract_beats(_synthetic(3000), samples, symbols)
    assert len(y) == 2


def test_record_edges_use_0_8_default():
    samples = [400, 760, 1120]
    _, R, _ = em.extract_beats(_synthetic(2000), samples, ["N", "N", "N"])
    assert R[0, 0] == pytest.approx(0.8) and R[0, 1] == pytest.approx(0.8)
    assert R[-1, 4] == pytest.approx(0.8)


def test_unknown_rr_mode_rejected():
    with pytest.raises(ValueError):
        em.extract_beats(_synthetic(2000), [1000], ["N"], rr_mode="oops")


def test_aami_map_matches_process_data():
    src = open(os.path.join(ROOT, "src", "process_data.py"), encoding="utf-8").read()
    for sym, cls in em.AAMI_MAP.items():
        assert f"'{sym}': {cls}" in src


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "data", "raw", "100.dat")),
                    reason="raw MIT-BIH not present")
def test_identical_to_process_data_on_real_records(tmp_path, monkeypatch):
    """The matched extractor must reproduce process_data.py byte for byte."""
    import importlib
    monkeypatch.chdir(ROOT)
    pd = importlib.import_module("process_data")
    monkeypatch.setattr(pd, "OUT_DIR", str(tmp_path))
    records = ["100", "208", "232"]
    pd.process_records(records, "chk")
    Xp = np.load(tmp_path / "X_chk.npy")
    Rp = np.load(tmp_path / "X_rr_chk.npy")
    yp = np.load(tmp_path / "y_chk.npy")
    Xs, Rs, ys = [], [], []
    for r in records:
        ecg, samples, symbols, _, _ = em.load_record(os.path.join("data", "raw", r), 0)
        X, R, y = em.extract_beats(em.clean_record(ecg), samples, symbols)
        Xs.append(X); Rs.append(R); ys.append(y)
    assert np.array_equal(np.concatenate(Xs), Xp)
    assert np.array_equal(np.concatenate(Rs), Rp)
    assert np.array_equal(np.concatenate(ys), yp)
