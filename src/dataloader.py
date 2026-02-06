import wfdb
import os
import numpy as np

def verify_data(record_path):
    print(f"Checking record: {record_path}")
    try:
        # Read the record
        record = wfdb.rdrecord(record_path)
        # Read annotations
        annotation = wfdb.rdann(record_path, 'atr')
        
        print(f"Signal Shape: {record.p_signal.shape}")
        print(f"Sampling Frequency: {record.fs}")
        print(f"Number of Annotations: {len(annotation.symbol)}")
        
        if record.fs != 360:
            print(f"[ERROR] Expected 360Hz, got {record.fs}Hz")
        else:
            print("[OK] Sampling Rate verified.")
            
        return True
    except Exception as e:
        print(f"[FAIL] Could not read record: {e}")
        return False

if __name__ == "__main__":
    # Test with record 100
    base_path = "data/raw/100" 
    if os.path.exists("data/raw/100.dat"):
        verify_data(base_path)
    else:
        print("Record 100 not found in data/raw/")
