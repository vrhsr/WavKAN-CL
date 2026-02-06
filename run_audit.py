
import json
import numpy as np
import glob
import os

def check_metrics():
    # 1. Baseline
    baseline_files = glob.glob('results/multiseed/baseline/seed_*/test_metrics.json')
    curr_files = glob.glob('results/multiseed/curriculum/seed_*/test_metrics.json')
    
    def get_stats(files):
        v_recalls = []
        s_recalls = []
        f1s = []
        for f in files:
            with open(f, 'r') as fp:
                d = json.load(fp)
                v_recalls.append(d['v_recall'])
                s_recalls.append(d['s_recall'])
                f1s.append(d['f1'])
                if 'n_recall' in d:
                     # Some old files might miss it, but my train_curriculum code saves it
                     # Logic check:
                     pass

        # Since I didn't save n_recalls_list in snippet, let me fix properly.
        n_recalls = [json.load(open(f))['n_recall'] for f in files]
        
        return {
            'v_mean': np.mean(v_recalls),
            'v_std': np.std(v_recalls),
            's_mean': np.mean(s_recalls),
            's_std': np.std(s_recalls),
            'f1_mean': np.mean(f1s),
            'f1_std': np.std(f1s),
            'n_mean': np.mean(n_recalls),
            'n_std': np.std(n_recalls),
            'count': len(files)
        }

    b_stats = get_stats(baseline_files)
    c_stats = get_stats(curr_files)

    print("--- METRICS AUDIT ---")
    print(f"Baseline (n={b_stats['count']})")
    print(f"  V-Recall: {b_stats['v_mean']:.3f} +/- {b_stats['v_std']:.3f}")
    print(f"  S-Recall: {b_stats['s_mean']:.3f} +/- {b_stats['s_std']:.3f}")
    print(f"  Macro-F1: {b_stats['f1_mean']:.3f} +/- {b_stats['f1_std']:.3f}")
    
    print(f"\nWavKAN-CL (n={c_stats['count']})")
    print(f"  V-Recall: {c_stats['v_mean']:.3f} +/- {c_stats['v_std']:.3f}")
    print(f"  N-Recall: {c_stats['n_mean']:.3f} +/- {c_stats['n_std']:.3f}")
    print(f"  S-Recall: {c_stats['s_mean']:.3f} +/- {c_stats['s_std']:.3f}")
    print(f"  Macro-F1: {c_stats['f1_mean']:.3f} +/- {c_stats['f1_std']:.3f}")

    print("\n--- MANUSCRIPT CLAIMS CHECK ---")
    claim_v = 0.898
    print(f"Claimed V-Recall: {claim_v}")
    print(f"Actual V-Recall:  {c_stats['v_mean']:.3f}")
    
    if abs(c_stats['v_mean'] - claim_v) < 0.005:
        print("✅ V-Recall matches claim.")
    else:
        print("❌ V-Recall DISCREPANCY.")

if __name__ == "__main__":
    check_metrics()
