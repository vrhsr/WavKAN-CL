"""
PTB-XL Lead-II Preprocessing → AAMI 5-class superclasses
==========================================================
Extracts single-beat windows from PTB-XL Lead-II, maps rhythm codes to
AAMI superclasses (N, S, V, F, Q), and saves .npy arrays compatible with
the existing WavKAN-CL data pipeline.

Usage:
    python src/process_ptbxl.py --data-dir data/ptbxl --out-dir data/processed_ptbxl

PTB-XL download (run once, ~2GB):
    pip install wfdb
    cd data && wget https://physionet.org/static/published-projects/ptb-xl/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip
    unzip ptb-xl-*.zip -d ptbxl

AAMI superclass map (AAMI EC57 standard):
    N  → NSR, LBBB, RBBB, IRBBB, CRBBB, LAD, RAD, NORM, IMI, ASMI, ...
    S  → SVPB, SVT, AFIB, AFLT, PSVT, ...
    V  → PVC, BIGU, TRIGU, VFIB, VT, ...
    F  → FUSION (rare, very few records)
    Q  → PACE, SNUS, SARRH, ... (unclassifiable/paced)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from scipy.signal import resample
from tqdm import tqdm

# ---------------------------------------------------------------------------
# AAMI Superclass Mapping (SCP code → integer label)
# Source: PTB-XL paper (Wagner et al. 2020) + AAMI EC57 standard
# ---------------------------------------------------------------------------
SCP_TO_AAMI = {
    # ------- N  (Normal / mostly morphologically normal) -------
    'NORM': 0, 'NSR': 0, 'SR': 0,
    'LBBB': 0, 'RBBB': 0, 'IRBBB': 0, 'CRBBB': 0,
    'LAD': 0, 'RAD': 0, 'LPR': 0, 'LPFB': 0,
    'IMI': 0, 'ASMI': 0, 'ILMI': 0, 'AMI': 0, 'ALMI': 0,
    'INJAS': 0, 'LMI': 0, 'INJAL': 0, 'IPLMI': 0, 'IPMI': 0,
    'INJIN': 0, 'INJLA': 0, 'RVH': 0, 'ANEUR': 0,
    'NDT': 0, 'DIG': 0, 'LNGQT': 0, 'ABQRS': 0,
    'STD': 0, 'STE': 0,

    # ------- S  (Supraventricular ectopic / SVT rhythms) -------
    'SVPB': 1, 'SVT': 1, 'AFIB': 1, 'AFLT': 1,
    'PSVT': 1, 'AVNRT': 1, 'AVRT': 1, 'SVARR': 1,
    'JTACHYC': 1, 'WPW': 1, 'BIGU': 1,

    # ------- V  (Ventricular ectopic / VT rhythms) -------
    'PVC': 2, 'VFIB': 2, 'VT': 2, 'TRIGU': 2, 'VCLVH': 2,

    # ------- F  (Fusion beats) -------
    'FUSION': 3,

    # ------- Q  (Unclassifiable / paced / artifact) -------
    'PACE': 4, 'SNUS': 4, 'SARRH': 4, 'STACH': 4, 'SBRAD': 4,
    'PACEDEF': 4, 'ECTOPIC': 4, 'NODATA': 4,
}

TARGET_SAMPLES = 360   # Match MIT-BIH window size
LEAD_IDX = 1           # Lead II index in PTB-XL (0=I, 1=II, ...)
HALF_WIN = TARGET_SAMPLES // 2


def load_ptbxl_metadata(data_dir: str) -> pd.DataFrame:
    """Load and return the PTB-XL database CSV with SCP code columns."""
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"ptbxl_database.csv not found in {data_dir}.\n"
            "Download PTB-XL from https://physionet.org/content/ptb-xl/1.0.3/"
        )
    df = pd.read_csv(csv_path, index_col="ecg_id")
    import ast
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    return df


def map_record_to_aami(scp_codes: dict) -> int:
    """
    Map a record's SCP codes to AAMI superclass.
    Priority: V > S > F > Q > N  (safety-critical first)
    Returns -1 if no matching code found.
    """
    labels_found = set()
    for code in scp_codes:
        code_upper = code.upper()
        if code_upper in SCP_TO_AAMI:
            labels_found.add(SCP_TO_AAMI[code_upper])

    if not labels_found:
        return -1  # unmappable

    # Priority ordering: V(2) > S(1) > F(3) > Q(4) > N(0)
    priority = [2, 1, 3, 4, 0]
    for p in priority:
        if p in labels_found:
            return p
    return -1


def extract_beats_from_record(
    record_path: str,
    aami_label: int,
    sampling_rate: int,
) -> tuple:
    """
    Read a WFDB record, resample Lead-II to 360Hz equivalent,
    extract beat windows around each annotation, compute RR history.
    Returns (beats, rr_feats, labels) as numpy arrays.
    """
    try:
        # Load signal
        record = wfdb.rdrecord(record_path, channels=[LEAD_IDX])
        signal = record.p_signal[:, 0]  # shape: (n_samples,)

        # Resample to 360 samples per second
        orig_fs = record.fs
        if orig_fs != 360:
            n_target = int(len(signal) * 360 / orig_fs)
            signal = resample(signal, n_target)

        # Use wfdb annotation if available, else R-peak detect
        ann_path = record_path
        try:
            ann = wfdb.rdann(ann_path, "atr")
            r_peaks = ann.sample
            # Rescale r_peak positions if we resampled
            if orig_fs != 360:
                r_peaks = (r_peaks * 360 / orig_fs).astype(int)
        except Exception:
            # Fallback: no annotations → detect R-peaks using scipy
            from scipy.signal import find_peaks
            # 0.35s distance ~ 170 BPM max, using 90th percentile for height threshold
            height_thresh = np.percentile(signal, 90) * 0.5
            r_peaks, _ = find_peaks(signal, distance=int(0.35 * 360), height=height_thresh)
            
            if len(r_peaks) < 2:
                return np.array([]), np.array([]), np.array([])

        # Extract beat windows
        beats = []
        rr_feats = []

        for i, rp in enumerate(r_peaks):
            start = rp - HALF_WIN
            end = rp + HALF_WIN
            if start < 0 or end > len(signal):
                continue

            beat = signal[start:end]
            if len(beat) != TARGET_SAMPLES:
                continue

            # RR interval computation (5-element window: t-2 to t+2)
            rr_indices = [max(0, i + k) for k in range(-2, 3)]
            rr_intervals = []
            for idx in rr_indices:
                if idx > 0 and idx < len(r_peaks):
                    rr = r_peaks[idx] - r_peaks[idx - 1]
                else:
                    rr = r_peaks[1] - r_peaks[0] if len(r_peaks) > 1 else 360
                rr_intervals.append(rr / 360.0)  # normalize to seconds

            # Dynamic normalization: divide by local mean of preceding 10 beats
            local_start = max(0, i - 10)
            local_rrs = []
            for k in range(local_start, i):
                if k > 0 and k < len(r_peaks):
                    local_rrs.append(r_peaks[k] - r_peaks[k - 1])
            if local_rrs:
                rr_local_mean = np.mean(local_rrs) / 360.0
            else:
                rr_local_mean = rr_intervals[2]

            if rr_local_mean > 0:
                rr_norm = np.array(rr_intervals) / rr_local_mean
            else:
                rr_norm = np.array(rr_intervals)

            beats.append(beat)
            rr_feats.append(rr_norm)

        if not beats:
            return np.array([]), np.array([]), np.array([])

        beats = np.array(beats, dtype=np.float32)
        rr_feats = np.array(rr_feats, dtype=np.float32)
        labels = np.full(len(beats), aami_label, dtype=np.int64)

        return beats, rr_feats, labels

    except Exception as e:
        print(f"Error processing {record_path}: {e}")
        return np.array([]), np.array([]), np.array([])


def process_ptbxl(data_dir: str, out_dir: str, sampling_rate: int = 500):
    """Main preprocessing pipeline."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("Loading PTB-XL metadata...")
    df = load_ptbxl_metadata(data_dir)

    # Inter-patient split: patient_id <= 16000 → DS1 (train/val), > 16000 → DS2 (test)
    # PTB-XL has 18,869 unique patients
    df["split"] = df["patient_id"].apply(lambda p: "train" if p <= 16000 else "test")

    # Signal file path: use 100Hz for faster processing (still resampled to 360)
    # Set to 500 for higher fidelity
    if sampling_rate == 100:
        signal_col = "filename_lr"
    else:
        signal_col = "filename_hr"

    print(f"\nUsing {sampling_rate}Hz signals. Split distribution:")
    print(df["split"].value_counts())

    all_splits = {"train": [], "test": []}

    # Group by split and process
    for split in ["train", "test"]:
        split_df = df[df["split"] == split]
        print(f"\nProcessing {split} split ({len(split_df)} records)...")

        X_list, Xr_list, y_list, patient_id_list = [], [], [], []

        for ecg_id, row in tqdm(split_df.iterrows(), total=len(split_df)):
            aami_label = map_record_to_aami(row["scp_codes"])
            if aami_label == -1:
                continue

            record_path = os.path.join(data_dir, row[signal_col])
            # Strip extension if present
            record_path = record_path.replace(".hea", "")

            beats, rr_feats, labels = extract_beats_from_record(
                record_path, aami_label, sampling_rate
            )

            if len(beats) == 0:
                continue

            X_list.append(beats)
            Xr_list.append(rr_feats)
            y_list.append(labels)
            # patient_id repeated once per beat -- needed for a downstream
            # patient-disjoint split (e.g. few-shot adaptation) without
            # re-reading the raw record.
            patient_id_list.append(np.full(len(beats), row["patient_id"]))

        if not X_list:
            print(f"  ⚠️  No valid samples found for {split}.")
            continue

        X = np.concatenate(X_list, axis=0)
        Xr = np.concatenate(Xr_list, axis=0)
        y = np.concatenate(y_list, axis=0)
        patient_ids = np.concatenate(patient_id_list, axis=0)

        # Normalize each beat: zero-mean, unit-std
        X = (X - X.mean(axis=1, keepdims=True)) / (X.std(axis=1, keepdims=True) + 1e-8)

        # Save
        np.save(os.path.join(out_dir, f"X_{split}.npy"), X)
        np.save(os.path.join(out_dir, f"X_rr_{split}.npy"), Xr)
        np.save(os.path.join(out_dir, f"y_{split}.npy"), y)
        np.save(os.path.join(out_dir, f"patient_ids_{split}.npy"), patient_ids)

        print(f"  ✅ {split}: {len(X)} beats | Shape: {X.shape}")
        unique, counts = np.unique(y, return_counts=True)
        class_names = ["N", "S", "V", "F", "Q"]
        for u, c in zip(unique, counts):
            print(f"     Class {class_names[u]}: {c} ({100*c/len(y):.1f}%)")

    print(f"\n✅ Done. Files saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PTB-XL Lead-II → AAMI preprocessing")
    parser.add_argument("--data-dir", type=str, default="data/ptbxl",
                        help="Path to the PTB-XL root directory")
    parser.add_argument("--out-dir", type=str, default="data/processed_ptbxl",
                        help="Output directory for .npy arrays")
    parser.add_argument("--sampling-rate", type=int, default=500, choices=[100, 500],
                        help="PTB-XL sampling rate to use (100=LR, 500=HR)")
    args = parser.parse_args()
    process_ptbxl(args.data_dir, args.out_dir, args.sampling_rate)
