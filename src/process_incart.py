"""
process_incart.py  —  INCART Database Preprocessor

St. Petersburg INCART 12-Lead Arrhythmia Database
===================================================
  - 75 long-term ECG records, 12-lead, 30 minutes each
  - Sampling rate: 257 Hz (resampled to 360 Hz to match MIT-BIH pipeline)
  - Expert annotations, AAMI-mappable labels
  - Source: PhysioNet / wfdb (free download)

Download first:
    wfdb.dl_database('incartdb', 'data/raw/incart/')
    OR: python -m wfdb.io.dl_database incartdb data/raw/incart

This script uses the SAME preprocessing pipeline as process_data.py
(z-score normalisation, 360-sample window, RR-interval history)
to ensure direct model comparability — no domain adaptation needed.

Output: data/incart_processed/ (same format as processed_rr_history/)
"""

import os, sys
from pathlib import Path
from collections import Counter

import numpy as np
import wfdb
from scipy import signal as scipy_signal
from tqdm import tqdm

OUT_DIR    = Path("data/incart_processed")
RAW_DIR    = Path("data/raw/incart")
OUT_FS     = 360
PRE_MS     = 250   # ms before R-peak  (= 90 samples at 360 Hz)
POST_MS    = 750   # ms after  R-peak  (= 270 samples at 360 Hz)
PRE_OUT    = int(PRE_MS  * OUT_FS / 1000)    # 90
POST_OUT   = int(POST_MS * OUT_FS / 1000)    # 270
WIN_LEN    = PRE_OUT + POST_OUT              # 360

# AAMI mapping (same as MIT-BIH)
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    'Q': 4, '/': 4, 'f': 4,
}

# INCART record list (I01 – I75)
INCART_ALL = [f"I{i:02d}" for i in range(1, 76)]

# Train/test split: use first 45 for training, last 30 for testing
# (Mirrors the 60/40 ratio used in MIT-BIH DS1/DS2)
INCART_TRAIN = INCART_ALL[:45]
INCART_TEST  = INCART_ALL[45:]


def resample_signal(sig: np.ndarray, orig_fs: int, target_fs: int = OUT_FS) -> np.ndarray:
    """Resamples a 1-D signal from orig_fs to target_fs using scipy's resample."""
    if orig_fs == target_fs:
        return sig
    n_samples = int(len(sig) * target_fs / orig_fs)
    return scipy_signal.resample(sig, n_samples)


def resample_annotations(ann_samples: np.ndarray, orig_fs: int, target_fs: int = OUT_FS) -> np.ndarray:
    """Scales annotation sample indices from orig_fs to target_fs."""
    if orig_fs == target_fs:
        return ann_samples
    return np.round(ann_samples * target_fs / orig_fs).astype(int)


def process_record(rec_id: str, raw_dir: Path) -> tuple:
    """
    Processes one INCART record. Returns (beats, rr_feats, labels) arrays.
    Returns empty arrays if record is missing.
    """
    rec_path = raw_dir / rec_id
    try:
        sig_raw, fields = wfdb.rdsamp(str(rec_path))
        ann             = wfdb.rdann(str(rec_path), 'atr')
    except Exception as e:
        print(f"  ⚠️  {rec_id}: {e}")
        return np.array([]), np.array([]), np.array([])

    orig_fs = fields['fs']
    # Use Lead I (channel 0); INCART is 12-lead so indices 0–11 are available
    ecg_raw = sig_raw[:, 0].astype(np.float64)

    # Resample to 360 Hz
    ecg        = resample_signal(ecg_raw, orig_fs)
    r_samples  = resample_annotations(ann.sample, orig_fs)
    symbols    = ann.symbol

    X, X_rr, y = [], [], []

    for i, (r, sym) in enumerate(zip(r_samples, symbols)):
        if sym not in AAMI_MAP:
            continue
        if r - PRE_OUT < 0 or r + POST_OUT > len(ecg):
            continue

        beat = ecg[r - PRE_OUT: r + POST_OUT].copy()

        # Z-score normalisation (identical to MIT-BIH pipeline)
        std = beat.std()
        if std < 1e-7:
            continue
        beat = (beat - beat.mean()) / (std + 1e-8)

        # 5-beat RR history (identical formula)
        def get_rr(i_cur, i_prev):
            if i_prev < 0 or i_cur >= len(r_samples):
                return 0.8
            return float(np.clip((r_samples[i_cur] - r_samples[i_prev]) / OUT_FS, 0.2, 3.0))

        rr = [
            get_rr(i-2, i-3), get_rr(i-1, i-2),
            get_rr(i,   i-1), get_rr(i+1, i), get_rr(i+2, i+1)
        ]

        X.append(beat.astype(np.float32))
        X_rr.append(rr)
        y.append(AAMI_MAP[sym])

    return (
        np.array(X,    dtype=np.float32),
        np.array(X_rr, dtype=np.float32),
        np.array(y,    dtype=np.int64),
    )


def process_split(records: list, split_name: str, raw_dir: Path, out_dir: Path):
    all_X, all_Xrr, all_y = [], [], []

    print(f"\n  Processing INCART {split_name} ({len(records)} records)...")
    for rec_id in tqdm(records):
        X, Xrr, y = process_record(rec_id, raw_dir)
        if len(X) == 0:
            continue
        all_X.append(X)
        all_Xrr.append(Xrr)
        all_y.append(y)

    if not all_X:
        print(f"  ⚠️  No data found for {split_name}. Check {raw_dir}.")
        return

    X   = np.concatenate(all_X)
    Xrr = np.concatenate(all_Xrr)
    y   = np.concatenate(all_y)

    print(f"  → {split_name}: {X.shape[0]} beats  |  dist: {dict(Counter(y.tolist()))}")

    np.save(out_dir / f"X_{split_name}.npy",    X)
    np.save(out_dir / f"X_rr_{split_name}.npy", Xrr)
    np.save(out_dir / f"y_{split_name}.npy",     y)

    # Class weights (same formula as MIT-BIH)
    if split_name == "train":
        unique, counts = np.unique(y, return_counts=True)
        total          = counts.sum()
        weights        = np.ones(5, dtype=np.float32)
        for u, c in zip(unique, counts):
            weights[u] = total / (5 * c)
        np.save(out_dir / "class_weights.npy", weights)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=str, default=str(RAW_DIR))
    parser.add_argument("--out-dir", type=str, default=str(OUT_DIR))
    parser.add_argument("--download", action="store_true",
                        help="Auto-download INCART via wfdb (requires internet)")
    args = parser.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.download:
        print("Downloading INCART database via wfdb...")
        wfdb.dl_database('incartdb', str(raw))
        print("Download complete.")

    if not raw.exists():
        print(f"\n⚠️  INCART data not found at: {raw}")
        print("   Run with --download flag OR manually download from PhysioNet:")
        print("   https://physionet.org/content/incartdb/1.0.0/")
        sys.exit(1)

    process_split(INCART_TRAIN, "train", raw, out)
    process_split(INCART_TEST,  "test",  raw, out)

    # Copy test as val too (INCART has no separate val split)
    import shutil
    for fname in ["X_test.npy", "X_rr_test.npy", "y_test.npy"]:
        src = out / fname
        dst = out / fname.replace("test", "val")
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)

    print(f"\n✅ INCART data saved to {out}/")
