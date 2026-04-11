"""
process_svdb.py  —  SVDB (MIT-BIH Supraventricular Arrhythmia Database) Preprocessor

MIT-BIH Supraventricular Arrhythmia Database
=============================================
  - 78 records (~30 min each), 2-lead, 128 Hz
  - Heavily enriched with supraventricular beats (S-class)
  - Used as the S-CLASS STRESS TEST in the multi-dataset evaluation
  - Source: PhysioNet (free download)

Download:
    wfdb.dl_database('svdb', 'data/raw/svdb/')

Why SVDB matters for this paper:
  The original paper's S-Recall of 0.29 on MIT-BIH DS2 is hard to evaluate
  because MIT-BIH DS2 has only 1,837 S-beats (2.8% of test set).
  SVDB provides a MUCH richer S-class environment to stress-test S-Recall.

  A model that achieves:
    - MIT-BIH V-Recall: 0.88  (primary endpoint)
    - SVDB S-Recall:    >0.60  (secondary, on stress test)
  has a compelling multi-dataset story.

Output: data/svdb_processed/
"""

import os, sys
from pathlib import Path
from collections import Counter

import numpy as np
import wfdb
from scipy import signal as scipy_signal
from tqdm import tqdm

OUT_DIR  = Path("data/svdb_processed")
RAW_DIR  = Path("data/raw/svdb")
OUT_FS   = 360
ORIG_FS  = 128   # SVDB native sampling rate

PRE_OUT  = int(0.25 * OUT_FS)   # 90
POST_OUT = int(0.75 * OUT_FS)   # 270
WIN_LEN  = PRE_OUT + POST_OUT   # 360

AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    'Q': 4, '/': 4, 'f': 4,
}


def resample_signal(sig: np.ndarray, orig_fs: int = ORIG_FS, target_fs: int = OUT_FS) -> np.ndarray:
    """Resamples from orig_fs to target_fs."""
    if orig_fs == target_fs:
        return sig
    n_out = int(len(sig) * target_fs / orig_fs)
    return scipy_signal.resample(sig, n_out)


def resample_ann(samples: np.ndarray, orig_fs: int = ORIG_FS, target_fs: int = OUT_FS) -> np.ndarray:
    """Scales annotation indices from orig_fs to target_fs."""
    return np.round(samples * target_fs / orig_fs).astype(int)


def process_record(rec_id: str, raw_dir: Path) -> tuple:
    rec_path = str(raw_dir / rec_id)
    try:
        sig_raw, fields = wfdb.rdsamp(rec_path)
        ann             = wfdb.rdann(rec_path, 'atr')
    except Exception as e:
        print(f"  ⚠️  {rec_id}: {e}")
        return np.array([]), np.array([]), np.array([])

    orig_fs   = fields['fs']
    ecg_raw   = sig_raw[:, 0].astype(np.float64)
    ecg       = resample_signal(ecg_raw, orig_fs)
    r_samples = resample_ann(ann.sample, orig_fs)
    symbols   = ann.symbol

    X, X_rr, y = [], [], []

    for i, (r, sym) in enumerate(zip(r_samples, symbols)):
        if sym not in AAMI_MAP:
            continue
        if r - PRE_OUT < 0 or r + POST_OUT > len(ecg):
            continue

        beat = ecg[r - PRE_OUT: r + POST_OUT].copy()
        std  = beat.std()
        if std < 1e-7:
            continue
        beat = (beat - beat.mean()) / (std + 1e-8)

        def get_rr(i_cur, i_prev):
            if i_prev < 0 or i_cur >= len(r_samples):
                return 0.8
            return float(np.clip((r_samples[i_cur] - r_samples[i_prev]) / OUT_FS, 0.2, 3.0))

        rr = [get_rr(i-2, i-3), get_rr(i-1, i-2),
              get_rr(i,   i-1), get_rr(i+1, i), get_rr(i+2, i+1)]

        X.append(beat.astype(np.float32))
        X_rr.append(rr)
        y.append(AAMI_MAP[sym])

    return (
        np.array(X,    dtype=np.float32),
        np.array(X_rr, dtype=np.float32),
        np.array(y,    dtype=np.int64),
    )


def get_svdb_records(raw_dir: Path) -> list:
    """Returns sorted list of record IDs found in raw_dir."""
    return sorted(set(
        p.stem for p in raw_dir.glob("*.hea")
        if not p.stem.startswith('.')
    ))


def process_all(raw_dir: Path, out_dir: Path):
    """
    SVDB has no standard train/test split.
    We use ALL records as a held-out test set
    (zero-shot evaluation: MIT-BIH-trained model, SVDB test only).
    """
    records = get_svdb_records(raw_dir)
    if not records:
        print(f"  ⚠️  No SVDB records found in {raw_dir}")
        print("   Download with: python src/process_svdb.py --download")
        return

    print(f"\n  Processing SVDB ({len(records)} records, S-class stress test)...")
    all_X, all_Xrr, all_y = [], [], []

    for rec_id in tqdm(records):
        X, Xrr, y = process_record(rec_id, raw_dir)
        if len(X) == 0:
            continue
        all_X.append(X)
        all_Xrr.append(Xrr)
        all_y.append(y)

    if not all_X:
        return

    X   = np.concatenate(all_X)
    Xrr = np.concatenate(all_Xrr)
    y   = np.concatenate(all_y)

    dist = dict(Counter(y.tolist()))
    s_ratio = dist.get(1, 0) / (y.shape[0] + 1e-8) * 100
    print(f"  → SVDB: {X.shape[0]:,} beats  |  dist: {dist}")
    print(f"  → S-class ratio: {s_ratio:.1f}%  (MIT-BIH DS2: 3.7% — this is the stress test)")

    out_dir.mkdir(parents=True, exist_ok=True)
    # Save as "test" split (evaluation-only database)
    np.save(out_dir / "X_test.npy",    X)
    np.save(out_dir / "X_rr_test.npy", Xrr)
    np.save(out_dir / "y_test.npy",    y)
    print(f"\n✅ SVDB saved to {out_dir}/")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)

    if args.download:
        print("Downloading SVDB via wfdb...")
        wfdb.dl_database('svdb', str(raw))

    if not raw.exists() or not any(raw.glob("*.hea")):
        print(f"\n⚠️  SVDB not found at {raw}")
        print("   Run: python src/process_svdb.py --download")
        sys.exit(1)

    process_all(raw, out)
