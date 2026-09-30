"""
extract_matched.py -- one beat-extraction path for MIT-BIH and the external databases.

Added 2026-09-30 (final pre-submission audit, AUDIT_FINDINGS.md H58/H59).

Why this exists
---------------
The manuscript said INCART and SVDB were "processed by the identical pipeline" as
the MIT-BIH training data. They were not. `process_incart.py` and `process_svdb.py`
resample correctly to 360 Hz, but they apply no filter (MIT-BIH is cleaned with
`nk.ecg_clean`), and INCART used lead I. This module makes the claim true by
construction. `extract_beats()` is a line-for-line mirror of the per-record loop in
`process_data.py`, the writer of every array the published checkpoints were trained
on. `tests/test_extract_matched.py` asserts byte-identical output against
`process_data.py` on real MIT-BIH records whenever the raw data is present.

RR modes
--------
`process_data.py` computes each RR interval between consecutive entries of
`ann.sample`. That array includes NON-beat annotations (rhythm-change '+', noise
'~', artefact '|', blocked P 'x', flutter wave '!', comments), so a small fraction
of intervals are annotation-to-beat gaps rather than beat-to-beat intervals.

  rr_mode="all_annotations"  what the published models were trained and tested on.
                             This is the default, so published arrays reproduce.
  rr_mode="beats_only"       intervals between consecutive BEAT annotations only.
                             Used only for the inference-time sensitivity analysis.

Nothing here changes `process_data.py`, and nothing here retrains anything.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

FS = 360
PRE_SAMPLES = int(0.25 * FS)    # 90  -> the R-peak sits at sample 90 of the window
POST_SAMPLES = int(0.75 * FS)   # 270 -> window is 250 ms before, 750 ms after R

# Identical to process_data.AAMI_MAP.
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    'Q': 4, '/': 4, 'f': 4,
}

# WFDB beat-annotation codes. Everything else ('+', '~', '|', 'x', '!', '"', '[', ']',
# ...) is a non-beat annotation.
BEAT_SYMBOLS = frozenset("NLRBAaJSVrFejnE/fQ?")


def clean_record(ecg: np.ndarray, fs: int = FS) -> np.ndarray:
    """Record-level cleaning, identical to process_data.py (including its fallback)."""
    import neurokit2 as nk
    try:
        return nk.ecg_clean(ecg, sampling_rate=fs, method="neurokit")
    except Exception:  # process_data.py uses a bare except with the same fallback
        return ecg


def extract_beats(ecg_clean: np.ndarray, r_peaks, labels, rr_mode: str = "all_annotations"):
    """Mirror of process_data.process_records' inner loop for one record.

    Returns (X float32 (n,360), X_rr float32 (n,5), y int64 (n,)).
    """
    if rr_mode not in ("all_annotations", "beats_only"):
        raise ValueError(f"unknown rr_mode {rr_mode!r}")
    r_peaks = np.asarray(r_peaks)
    labels = list(labels)

    if rr_mode == "beats_only":
        beat_idx = [k for k, s in enumerate(labels) if s in BEAT_SYMBOLS]
        pos = {k: j for j, k in enumerate(beat_idx)}
        seq = r_peaks[beat_idx]
    else:
        pos = None
        seq = r_peaks

    def get_rr(idx_current, idx_prev):
        if idx_prev < 0 or idx_current >= len(seq):
            return 0.8
        return (seq[idx_current] - seq[idx_prev]) / FS

    X, X_rr, y = [], [], []
    for i, (r, lbl) in enumerate(zip(r_peaks, labels)):
        if lbl not in AAMI_MAP:
            continue
        if r - PRE_SAMPLES < 0 or r + POST_SAMPLES > len(ecg_clean):
            continue
        beat = ecg_clean[r - PRE_SAMPLES: r + POST_SAMPLES]
        if np.std(beat) < 1e-7:
            continue
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)

        j = i if pos is None else pos[i]
        window = [get_rr(j - 2, j - 3), get_rr(j - 1, j - 2), get_rr(j, j - 1),
                  get_rr(j + 1, j), get_rr(j + 2, j + 1)]
        window = [np.clip(v, 0.2, 3.0) for v in window]

        X.append(beat)
        X_rr.append(window)
        y.append(AAMI_MAP[lbl])

    return (np.array(X, dtype=np.float32).reshape(-1, PRE_SAMPLES + POST_SAMPLES),
            np.array(X_rr, dtype=np.float32).reshape(-1, 5),
            np.array(y, dtype=np.int64))


def load_record(path: str, lead: int):
    """Reads one WFDB record, resampled to 360 Hz exactly as process_incart/svdb do."""
    import wfdb
    from scipy import signal as scipy_signal
    sig, fields = wfdb.rdsamp(path)
    ann = wfdb.rdann(path, "atr")
    fs = int(round(fields["fs"]))
    ecg = sig[:, lead].astype(np.float64) if fs != FS else sig[:, lead]
    samples = np.asarray(ann.sample)
    if fs != FS:
        ecg = scipy_signal.resample(ecg, int(len(ecg) * FS / fs))
        samples = np.round(samples * FS / fs).astype(int)
    return ecg, samples, list(ann.symbol), fields["sig_name"][lead], fs


DATASETS = {
    # name: (raw dir, lead index, expected lead name or None, record list source)
    "incart": ("data/raw_incart", 1, "II", "incartdb"),
    "svdb": ("data/raw_svdb", 0, None, "svdb"),       # SVDB leads are unlabelled (ECG1/ECG2)
    "mitdb_test": ("data/raw", 0, None, "mitdb"),     # channel 0, as process_data.py
}


def process_dataset(name: str, out_dir: str, rr_mode: str):
    raw_dir, lead, expect, db = DATASETS[name]
    if name == "mitdb_test":
        sys.path.insert(0, os.path.dirname(__file__))
        from split import TEST_RECORDS
        records = list(TEST_RECORDS)
    else:
        records = sorted(p.stem for p in Path(raw_dir).glob("*.hea"))
    Xs, Rs, Ys, ids, leads = [], [], [], [], set()
    for rec in records:
        ecg, samples, symbols, lead_name, fs = load_record(os.path.join(raw_dir, rec), lead)
        if expect is not None and lead_name != expect:
            raise RuntimeError(f"{name} {rec}: channel {lead} is {lead_name!r}, expected {expect!r}")
        leads.add(lead_name)
        X, R, Y = extract_beats(clean_record(ecg), samples, symbols, rr_mode=rr_mode)
        Xs.append(X); Rs.append(R); Ys.append(Y); ids += [rec] * len(Y)
    os.makedirs(out_dir, exist_ok=True)
    X, R, Y = np.concatenate(Xs), np.concatenate(Rs), np.concatenate(Ys)
    np.save(os.path.join(out_dir, "X_test.npy"), X)
    np.save(os.path.join(out_dir, "X_rr_test.npy"), R)
    np.save(os.path.join(out_dir, "y_test.npy"), Y)
    np.save(os.path.join(out_dir, "ids_test.npy"), np.array(ids))
    counts = np.bincount(Y, minlength=5).tolist()
    print(f"{name}: {len(records)} records, leads {sorted(leads)}, rr_mode={rr_mode}, "
          f"beats={len(Y)}, class counts N/S/V/F/Q={counts} -> {out_dir}")
    return counts


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    ap.add_argument("--out", required=True)
    ap.add_argument("--rr-mode", default="all_annotations", choices=["all_annotations", "beats_only"])
    a = ap.parse_args()
    process_dataset(a.dataset, a.out, a.rr_mode)
