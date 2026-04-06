"""
Extended RR-Interval Feature Extraction (10-beat window)

Expands from 5-beat to 10-beat RR history for richer temporal context.
This captures longer compensatory pauses and pre-ectopic patterns.

Output: data/processed_rr_extended/
"""

import os
import wfdb
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
from collections import Counter

# --- CONFIGURATION ---
DATA_DIR = "data/raw" 
OUT_DIR = "data/processed_rr_extended"  # New directory
os.makedirs(OUT_DIR, exist_ok=True)

FS = 360
PRE_SAMPLES = int(0.25 * FS)
POST_SAMPLES = int(0.75 * FS)

DS1_RECORDS = [
    '101','106','108','109','112','114','115','116','118','119',
    '122','124','201','203','205','207','208','209','215','220',
    '223','230'
]
VAL_RECORDS = ['215', '220', '223', '230']
TRAIN_RECORDS = [r for r in DS1_RECORDS if r not in VAL_RECORDS]

TEST_RECORDS = [
    '100','103','105','111','113','117','121','123','200','202',
    '210','212','213','214','219','221','222','228','231','232',
    '233','234'
]

AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,
    'A': 1, 'a': 1, 'J': 1, 'S': 1,
    'V': 2, 'E': 2,
    'F': 3,
    'Q': 4, '/': 4, 'f': 4
}

def process_extended_rr(record_list, split_name):
    print(f"Processing {split_name} with 10-beat RR window...")
    
    X = []
    X_rr = []  # 10 RR intervals
    y = []
    
    for rec in tqdm(record_list):
        record_path = os.path.join(DATA_DIR, rec)
        try:
            signal, fields = wfdb.rdsamp(record_path)
            ann = wfdb.rdann(record_path, 'atr')
        except:
            try:
                record_path = os.path.join(DATA_DIR, 'mitdb', rec)
                signal, fields = wfdb.rdsamp(record_path)
                ann = wfdb.rdann(record_path, 'atr')
            except:
                continue

        ecg = signal[:, 0]
        try: ecg = nk.ecg_clean(ecg, sampling_rate=FS, method="neurokit")
        except: pass
        
        r_peaks = ann.sample
        labels = ann.symbol
        
        # Extract RR intervals (in seconds)
        rr_intervals = np.diff(r_peaks) / FS
        
        for i, lbl in enumerate(labels):
            if lbl not in AAMI_MAP: continue
            
            r = r_peaks[i]
            if r - PRE_SAMPLES < 0 or r + POST_SAMPLES > len(ecg): continue
            
            # Extract beat waveform
            beat = ecg[r - PRE_SAMPLES : r + POST_SAMPLES]
            if np.std(beat) < 1e-7: continue
            
            # Z-score normalization
            beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
            
            # --- EXTENDED RR HISTORY (10-beat window) ---
            # RR indices: i-4, i-3, i-2, i-1, i (current), i+1, i+2, i+3, i+4
            rr_features = []
            for offset in range(-4, 6):  # -4 to +5 (10 intervals)
                rr_idx = i + offset - 1  # RR interval index
                if 0 <= rr_idx < len(rr_intervals):
                    rr_val = rr_intervals[rr_idx]
                    # Clip outliers
                    rr_val = np.clip(rr_val, 0.3, 2.0)
                else:
                    rr_val = 0.8  # Default (normal RR)
                rr_features.append(rr_val)
            
            X.append(beat)
            X_rr.append(rr_features)
            y.append(AAMI_MAP[lbl])
    
    X = np.array(X, dtype=np.float32)
    X_rr = np.array(X_rr, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    print(f"--> {split_name}: {X.shape[0]} beats, RR shape: {X_rr.shape}")
    
    np.save(os.path.join(OUT_DIR, f"X_{split_name}.npy"), X)
    np.save(os.path.join(OUT_DIR, f"X_rr_{split_name}.npy"), X_rr)
    np.save(os.path.join(OUT_DIR, f"y_{split_name}.npy"), y)
    
    return y

if __name__ == "__main__":
    print("="*60)
    print("EXTENDED RR FEATURE EXTRACTION (10-beat window)")
    print("="*60)
    
    y_train = process_extended_rr(TRAIN_RECORDS, "train")
    process_extended_rr(VAL_RECORDS, "val")
    process_extended_rr(TEST_RECORDS, "test")
    
    # Class Weights
    counts = Counter(y_train)
    weights = np.zeros(5)
    total = sum(counts.values())
    for i in range(5):
        if counts[i] > 0: weights[i] = total / (5 * counts[i])
        else: weights[i] = 1.0
    
    np.save(os.path.join(OUT_DIR, "class_weights.npy"), weights)
    
    print(f"\n✅ Extended RR features saved to {OUT_DIR}/")
    print(f"RR window: 10 beats (vs previous 5 beats)")
