import numpy as np
import wfdb
import neurokit2 as nk
import os
from split import TRAIN_RECORDS, VAL_RECORDS, TEST_RECORDS
from collections import Counter
import json

# === Configuration ===
SAMPLING_RATE = 360
TARGET_SAMPLES = 256  # Power of 2 for easy pooling

# AAMI Mapping
AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,  # Normal (N)
    'A': 1, 'a': 1, 'J': 1, 'S': 1,          # UPVE (S)
    'V': 2, 'E': 2,                          # VEB (V)
    'F': 3,                                  # Fusion (F)
    '/': 4, 'f': 4, 'Q': 4                   # Unknown (Q) or Paced
}
# Note: Paced beats ('/') are often excluded in some papers, but mapping to Q (4) here.
# Standard 5 classes: N, S, V, F, Q

def clean_ecg_signal(signal, fs=360):
    """Clean ECG using NeuroKit2."""
    # Using 'neurokit' method which includes highpass, lowpass, and powerline filtering
    try:
        cleaned = nk.ecg_clean(signal, sampling_rate=fs, method="neurokit")
        return cleaned
    except Exception as e:
        print(f"Warning: Neurokit clean failed, returning raw. Error: {e}")
        return signal

def extract_beats(signal, record_name, fs=360):
    """Segment beats based on annotations."""
    annotation = wfdb.rdann(f"data/raw/{record_name}", 'atr')
    ann_samples = annotation.sample
    ann_symbols = annotation.symbol
    
    beats = []
    labels = []
    
    # Window calculation for 256 samples
    # We want to center the R-peak usually. 
    # Let's say pre=90 samples, post=166 samples (Total 256)
    # 90 samples @ 360Hz = 0.25s
    # 166 samples @ 360Hz = 0.46s
    pre_samples = 90
    post_samples = 166
    
    for idx, sym in zip(ann_samples, ann_symbols):
        if sym not in AAMI_MAP:
            continue
            
        label = AAMI_MAP[sym]
        
        # Check boundaries
        if idx - pre_samples < 0 or idx + post_samples > len(signal):
            continue
            
        beat = signal[idx - pre_samples : idx + post_samples]
        
        # Z-score normalization per beat (standard practice)
        if np.std(beat) < 1e-6:
            continue # Skip flatline
            
        beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-6)
        
        beats.append(beat)
        labels.append(label)
        
    return np.array(beats), np.array(labels)

def process_dataset(records, name):
    print(f"\nProcessing {name} set...")
    str_recs = [str(r) for r in records]
    
    all_X = []
    all_y = []
    
    for rec in str_recs:
        path = f"data/raw/{rec}"
        if not os.path.exists(path + ".dat"):
            print(f"Skipping {rec}, file not found.")
            continue
            
        record = wfdb.rdrecord(path)
        # Using Lead II usually (Index 0 if available, check leads)
        # MIT-BIH usually has MLII as channel 0
        signal = record.p_signal[:, 0] 
        
        # 1. Clean
        cleaned_signal = clean_ecg_signal(signal, fs=SAMPLING_RATE)
        
        # 2. Extract
        X, y = extract_beats(cleaned_signal, rec, fs=SAMPLING_RATE)
        
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            
        print(f"Record {rec}: {len(X)} beats")

    if len(all_X) == 0:
        print(f"No data found for {name}!")
        return
        
    X_arr = np.vstack(all_X)
    y_arr = np.concatenate(all_y)
    
    # Save
    np.save(f"data/processed/X_{name}.npy", X_arr)
    np.save(f"data/processed/y_{name}.npy", y_arr)
    
    print(f"Saved data/processed/X_{name}.npy: {X_arr.shape}")
    print(f"Class distribution: {Counter(y_arr)}")
    
    return y_arr

if __name__ == "__main__":
    # Ensure processed dir exists (already checked but good practice)
    os.makedirs("data/processed", exist_ok=True)
    
    # Run pipeline
    y_train = process_dataset(TRAIN_RECORDS, "train")
    process_dataset(VAL_RECORDS, "val")
    process_dataset(TEST_RECORDS, "test")
    
    # Save Class Counts for Weighted Sampling
    if y_train is not None:
        counts = dict(Counter(y_train))
        # Convert keys to int (json serializable)
        counts = {int(k): int(v) for k, v in counts.items()}
        with open("data/processed/class_counts.json", "w") as f:
            json.dump(counts, f)
        print("\nClass counts saved to data/processed/class_counts.json")
