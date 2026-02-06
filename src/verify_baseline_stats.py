
import json
import numpy as np
from pathlib import Path

def main():
    print("Verifying 10-Seed Baseline Statistics...")
    
    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 5678, 8192, 9999]
    metrics = {
        'f1': [],
        's_recall': [],
        'v_recall': []
    }
    
    missing_seeds = []
    
    for seed in seeds:
        path = Path(f"results/multiseed/baseline/seed_{seed}/test_metrics.json")
        if not path.exists():
            missing_seeds.append(seed)
            continue
            
        with open(path, 'r') as f:
            data = json.load(f)
            metrics['f1'].append(data['f1'])
            metrics['s_recall'].append(data['s_recall'])
            metrics['v_recall'].append(data['v_recall'])
    
    if missing_seeds:
        print(f"WARNING: Missing results for seeds: {missing_seeds}")
        
    print(f"\nFailed Seeds: {len(missing_seeds)}")
    print(f"Valid Seeds: {len(metrics['f1'])}")
    
    print("\n--- STATISTICS ---")
    for key, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        print(f"{key.upper()}: {mean:.4f} ± {std:.4f}")

if __name__ == "__main__":
    main()
