"""Pre-flight check for the publication pipeline."""
import os
import sys
import glob
import pandas as pd
import ast

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

data_dir = 'E:/rpr/data/ptb-xl-1.0.3'

# 1. Check CSV
csv_path = os.path.join(data_dir, 'ptbxl_database.csv')
print(f'[1] CSV exists: {os.path.exists(csv_path)}')
df = pd.read_csv(csv_path, index_col='ecg_id')
print(f'    Total records in CSV: {len(df)}')

# 2. Check SCP statements
scp_path = os.path.join(data_dir, 'scp_statements.csv')
print(f'[2] SCP statements exists: {os.path.exists(scp_path)}')

# 3. Count 100Hz .dat files
dat_100 = glob.glob(os.path.join(data_dir, 'records100', '**', '*.dat'), recursive=True)
hea_100 = glob.glob(os.path.join(data_dir, 'records100', '**', '*.hea'), recursive=True)
print(f'[3] 100Hz .dat files: {len(dat_100)}')
print(f'    100Hz .hea files: {len(hea_100)}')

# 4. Check 500Hz
dat_500 = glob.glob(os.path.join(data_dir, 'records500', '**', '*.dat'), recursive=True)
print(f'[4] 500Hz .dat files: {len(dat_500)}')

# 5. Check filename_lr column points to real files  
sample_row = df.iloc[0]
lr_path = os.path.join(data_dir, sample_row['filename_lr'])
file_exists = os.path.exists(lr_path + '.dat')
print(f'[5] Sample filename_lr: {sample_row["filename_lr"]}')
print(f'    Full path exists: {file_exists}')

# 6. Check patient_id distribution for split
pid_min = df['patient_id'].min()
pid_max = df['patient_id'].max()
print(f'[6] Patient ID range: {pid_min} - {pid_max}')
train_count = int((df['patient_id'] <= 16000).sum())
test_count = int((df['patient_id'] > 16000).sum())
print(f'    Train split (<=16000): {train_count}')
print(f'    Test split (>16000): {test_count}')

# 7. Test beat extraction on train AND test records
from src.process_ptbxl import extract_beats_from_record, map_record_to_aami
scp_codes = ast.literal_eval(df.iloc[0]['scp_codes'])
aami = map_record_to_aami(scp_codes)
b, r, l = extract_beats_from_record(lr_path, aami, 100)
print(f'[7] Beat extraction test (record 1): beats={b.shape}, rr={r.shape}, labels={l.shape}')

# Test a record from the test split (patient_id > 16000)
test_df = df[df['patient_id'] > 16000]
test_row = test_df.iloc[0]
test_lr = os.path.join(data_dir, test_row['filename_lr'])
test_scp = ast.literal_eval(test_row['scp_codes'])
test_aami = map_record_to_aami(test_scp)
bt, rt, lt = extract_beats_from_record(test_lr, test_aami, 100)
print(f'    Beat extraction test (test split): beats={bt.shape}, rr={rt.shape}, labels={lt.shape}')

# 8. Check existing MIT-BIH data for SMOTE + ablation
mitbih_files = ['data/processed_rr_history/X_train.npy', 
                'data/processed_rr_history/X_rr_train.npy',
                'data/processed_rr_history/y_train.npy',
                'data/processed_rr_history/class_weights.npy']
all_exist = all(os.path.exists(f) for f in mitbih_files)
print(f'[8] MIT-BIH training data exists: {all_exist}')

# 9. Check model checkpoint for PTB-XL eval
ckpt = 'results/hybrid_rr_history_20_seeds/seed_42/best_hybrid_rr.pth'
print(f'[9] Model checkpoint exists: {os.path.exists(ckpt)}')

# 10. Check ablation script imports
try:
    from src.run_ablation_study import FullModel, NoGRU, NoRR, PureWavKAN, FocalLoss
    m = FullModel('b_spline')
    print(f'[10] Ablation models + B-spline + FocalLoss: OK')
except Exception as e:
    print(f'[10] Ablation import ERROR: {e}')

print()
if len(dat_100) >= 21000 and all_exist and file_exists:
    print('=' * 50)
    print('  ALL PRE-FLIGHT CHECKS PASSED')
    print('=' * 50)
else:
    print('=' * 50)
    print('  SOME CHECKS FAILED — review above')
    print('=' * 50)
