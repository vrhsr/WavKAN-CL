import os
import wfdb
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
from collections import Counter

# --- CONFIGURATION ---
DATA_DIR = "data/raw" # User said "data/raw/mitdb" in text but code moved files to "data/raw". Adjusting to where files actually are.
# Checking actual location: earlier I ran `move mit-bih-arrhythmia-database-1.0.0\* data\raw`
# So the .dat files are directly in data/raw
# But the user's script says `DATA_DIR = "data/raw/mitdb"`
# I will adhere to safe coding: I will check if data/raw/mitdb exists relative to where I put files.
# If files are in data/raw, I'll update the variable or move them.
# PREVIOUS ACTION: `move mit-bih-arrhythmia-database-1.0.0\* data\raw`
# So files are in `data/raw/`. 
# To match user's structure expectation, I should probably move them to `data/raw/mitdb/` OR adjust this script.
# Adjusting script is safer than moving files around again and risking loss.
DATA_DIR = "data/raw" 

OUT_DIR = "data/processed_rr_history"
os.makedirs(OUT_DIR, exist_ok=True)

FS = 360
# Standard 1-second window (centered on R-peak somewhat)
# 250ms before, 750ms after (captures P-wave and full T-wave)
PRE_SAMPLES = int(0.25 * FS)  # 90
POST_SAMPLES = int(0.75 * FS) # 270 
# Total = 360 samples

# --- SPLIT DEFINITIONS (De Chazal Protocol) ---
# DS1: Training (Split into Train and Val)
DS1_RECORDS = [
    '101','106','108','109','112','114','115','116','118','119',
    '122','124','201','203','205','207','208','209','215','220',
    '223','230'
]

# We carve out the last 4 records of DS1 for Validation (Inter-Patient Safe)
VAL_RECORDS = ['215', '220', '223', '230']
TRAIN_RECORDS = [r for r in DS1_RECORDS if r not in VAL_RECORDS]

# DS2: Testing (Strictly held out)
TEST_RECORDS = [
    '100','103','105','111','113','117','121','123','200','202',
    '210','212','213','214','219','221','222','228','231','232',
    '233','234'
]

# AAMI MAPPING
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,     # Normal
    'A': 1, 'a': 1, 'J': 1, 'S': 1,             # Supraventricular (S)
    'V': 2, 'E': 2,                             # Ventricular (V)
    'F': 3,                                     # Fusion (F)
    'Q': 4, '/': 4, 'f': 4                      # Unknown (Q)
}

def process_records(record_list, split_name):
    X, X_rr, y, record_ids = [], [], [], []

    print(f"Processing {split_name} ({len(record_list)} records)...")
    
    for rec in tqdm(record_list):
        record_path = os.path.join(DATA_DIR, rec)
        try:
            signal, fields = wfdb.rdsamp(record_path)
            ann = wfdb.rdann(record_path, 'atr')
        except FileNotFoundError:
            # Try appending 'mitdb' if user intended structure
            try:
                record_path_alt = os.path.join(DATA_DIR, 'mitdb', rec)
                signal, fields = wfdb.rdsamp(record_path_alt)
                ann = wfdb.rdann(record_path_alt, 'atr')
            except:
                print(f"Warning: Record {rec} not found in {DATA_DIR}. Skipping.")
                continue

        # Use Lead II (Index 0)
        ecg = signal[:, 0] 
        
        # Denoise
        try:
            ecg_clean = nk.ecg_clean(ecg, sampling_rate=FS, method="neurokit")
        except:
             # Fallback if cleaner fails (rare)
             ecg_clean = ecg

        r_peaks = ann.sample
        labels = ann.symbol

        for i, (r, lbl) in enumerate(zip(r_peaks, labels)):
            if lbl not in AAMI_MAP:
                continue

            # Skip if r-peak is too close to start/end
            if r - PRE_SAMPLES < 0 or r + POST_SAMPLES > len(ecg_clean):
                continue

            # Extract beat
            beat = ecg_clean[r - PRE_SAMPLES : r + POST_SAMPLES]
            
            # Z-score Normalization (Critical for WavKAN)
            if np.std(beat) < 1e-7:
                continue
            beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)

            # --- RR INTERVAL HISTORY (DIAMOND TIP #2) ---
            # Goal: Capture [RR_-2, RR_-1, RR_0 (Pre), RR_+1 (Post), RR_+2]
            # Context allows distinguishing 'Premature' from 'Fast Rhythm'
            
            rr_history = []
            
            # Offsets for -2, -1, 0, +1, +2
            # Note: r_peaks indices. i is current index.
            # We need intervals.
            # Interval k is between peak[k] and peak[k-1]
            
            # Let's define:
            # RR_0 = Time from i-1 to i (Pre-RR)
            # RR_-1 = Time from i-2 to i-1
            # RR_-2 = Time from i-3 to i-2
            # RR_+1 = Time from i to i+1 (Post-RR)
            # RR_+2 = Time from i+1 to i+2
            
            # Helper to safely get RR (defaults to 0.8 if out of bounds)
            def get_rr(idx_current, idx_prev):
                if idx_prev < 0 or idx_current >= len(r_peaks):
                    return 0.8
                return (r_peaks[idx_current] - r_peaks[idx_prev]) / FS

            # Local Window
            rr_0  = get_rr(i, i-1)      # Current Pre
            rr_m1 = get_rr(i-1, i-2)    # Prev
            rr_m2 = get_rr(i-2, i-3)    # Prev-Prev
            rr_p1 = get_rr(i+1, i)      # Post
            rr_p2 = get_rr(i+2, i+1)    # Post-Post
            
            # Normalize/Clip (Safety)
            window = [rr_m2, rr_m1, rr_0, rr_p1, rr_p2]
            window = [np.clip(val, 0.2, 3.0) for val in window]
            
            X_rr.append(window)
            # ------------------------------------------------

            X.append(beat)
            y.append(AAMI_MAP[lbl])
            record_ids.append(rec)

    # Convert to Numpy
    X = np.array(X, dtype=np.float32)
    X_rr = np.array(X_rr, dtype=np.float32) # Shape: (N, 2)
    y = np.array(y, dtype=np.int64)
    ids = np.array(record_ids)
    
    print(f"--> {split_name} shape: {X.shape}")
    print(f"--> RR Features shape: {X_rr.shape}")
    print(f"--> Class distribution: {Counter(y)}")
    
    # Save
    np.save(os.path.join(OUT_DIR, f"X_{split_name}.npy"), X)
    np.save(os.path.join(OUT_DIR, f"X_rr_{split_name}.npy"), X_rr) # NEW SAVE
    np.save(os.path.join(OUT_DIR, f"y_{split_name}.npy"), y)
    np.save(os.path.join(OUT_DIR, f"ids_{split_name}.npy"), ids)
    
    return y

# --- EXECUTION ---
if __name__ == "__main__":
    y_train = process_records(TRAIN_RECORDS, "train")
    y_val = process_records(VAL_RECORDS, "val")
    y_test = process_records(TEST_RECORDS, "test")

    # Compute and save Class Weights for Loss Function (handling imbalance)
    if y_train is not None and len(y_train) > 0:
        counts = Counter(y_train)
        total = sum(counts.values())
        # Formula: total / (num_classes * count)
        weights = np.zeros(5)
        for cls_idx in range(5):
            if counts[cls_idx] > 0:
                weights[cls_idx] = total / (5 * counts[cls_idx])
            else:
                weights[cls_idx] = 1.0 # default
        
        print(f"Computed Class Weights: {weights}")
        np.save(os.path.join(OUT_DIR, "class_weights.npy"), weights)
        print("DONE. Dataset ready.")
    else:
        print("ERROR: No training data processed.")
