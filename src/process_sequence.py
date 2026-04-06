import os
import wfdb
import numpy as np
import neurokit2 as nk
from tqdm import tqdm
from collections import Counter

# --- CONFIGURATION ---
DATA_DIR = "data/raw" 
OUT_DIR = "data/processed_sequence"
os.makedirs(OUT_DIR, exist_ok=True)

FS = 360
# Standard 1-second window
PRE_SAMPLES = int(0.25 * FS)  # 90
POST_SAMPLES = int(0.75 * FS) # 270 

# DS1: Training + Val
DS1_RECORDS = [
    '101','106','108','109','112','114','115','116','118','119',
    '122','124','201','203','205','207','208','209','215','220',
    '223','230'
]
VAL_RECORDS = ['215', '220', '223', '230']
TRAIN_RECORDS = [r for r in DS1_RECORDS if r not in VAL_RECORDS]

# DS2: Testing
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

def process_sequence_records(record_list, split_name):
    print(f"Processing {split_name} (Sequence Mode)...")
    
    X_seq = [] # Shape: (N, 5, 360)
    y = []     # Label of the CENTER beat (t=0)
    
    for rec in tqdm(record_list):
        record_path = os.path.join(DATA_DIR, rec)
        try:
            signal, fields = wfdb.rdsamp(record_path)
            ann = wfdb.rdann(record_path, 'atr')
        except:
             try:
                 # Backup path check
                 record_path = os.path.join(DATA_DIR, 'mitdb', rec)
                 signal, fields = wfdb.rdsamp(record_path)
                 ann = wfdb.rdann(record_path, 'atr')
             except:
                 continue

        ecg = signal[:, 0]
        # Denoise
        try: ecg = nk.ecg_clean(ecg, sampling_rate=FS, method="neurokit")
        except: pass
        
        r_peaks = ann.sample
        labels = ann.symbol
        
        # We need to map *every* annotation to an index first, so we can access neighbors
        # Filter valid beats first? No, we need context even if context is 'unknown'
        # Strategy: Iterate valid beats, then grab neighbors from r_peaks/labels arrays.
        
        valid_indices = [i for i, lbl in enumerate(labels) if lbl in AAMI_MAP]
        
        for i in valid_indices:
            r = r_peaks[i]
            lbl = labels[i]
            
            # Boundary check for CENTER beat
            if r - PRE_SAMPLES < 0 or r + POST_SAMPLES > len(ecg): 
                continue
                
            center_beat = ecg[r - PRE_SAMPLES : r + POST_SAMPLES]
            
            # --- CONTEXT EXTRACTION ---
            # Indices: i-2, i-1, i, i+1, i+2
            context_beats = []
            
            for offset in [-2, -1, 0, 1, 2]:
                target_idx = i + offset
                
                # Verify neighbor exists
                if 0 <= target_idx < len(r_peaks):
                    r_neighbor = r_peaks[target_idx]
                    
                    # Verify neighbor bounds
                    if r_neighbor - PRE_SAMPLES >= 0 and r_neighbor + POST_SAMPLES <= len(ecg):
                        beat = ecg[r_neighbor - PRE_SAMPLES : r_neighbor + POST_SAMPLES]
                        
                        # Z-Score
                        if np.std(beat) > 1e-7:
                            beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
                        else:
                            beat = np.zeros(360) 
                    else:
                        beat = np.zeros(360) # Padding
                else:
                    beat = np.zeros(360) # Padding
                
                context_beats.append(beat)
            
            # End Context Loop
            
            # Z-Score Center Beat (for label quality)
            # If center beat is junk, skip training on it
            if np.std(center_beat) < 1e-7: continue

            # Stack: (5, 360)
            seq_block = np.stack(context_beats, axis=0) # (5, 360)
            
            X_seq.append(seq_block)
            y.append(AAMI_MAP[lbl])
            
    # Convert & Save
    X_seq = np.array(X_seq, dtype=np.float32)
    y = np.array(y, dtype=np.int64)
    
    print(f"--> {split_name} shape: {X_seq.shape}")
    np.save(os.path.join(OUT_DIR, f"X_seq_{split_name}.npy"), X_seq)
    np.save(os.path.join(OUT_DIR, f"y_{split_name}.npy"), y)
    
    return y

if __name__ == "__main__":
    y_train = process_sequence_records(TRAIN_RECORDS, "train")
    process_sequence_records(VAL_RECORDS, "val")
    process_sequence_records(TEST_RECORDS, "test")
    
    # Class Weights
    counts = Counter(y_train)
    weights = np.zeros(5)
    total = sum(counts.values())
    for i in range(5):
        if counts[i] > 0: weights[i] = total / (5 * counts[i])
        else: weights[i] = 1.0
    
    # Boost S
    weights[1] *= 5.0 # Moderate boost, let context do the work
    np.save(os.path.join(OUT_DIR, "class_weights.npy"), weights)
