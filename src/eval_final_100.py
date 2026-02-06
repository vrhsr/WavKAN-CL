import torch
import numpy as np
import os
import sys
import json
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.wavkan import WavKANClassifier
from src.train_hybrid import HybridWavKAN, ECGDataset
from models.cnn_baseline import CNN1D

# CONFIG - Matches experiment_final_100.py
SEEDS = [
    42, 101, 777, 2026, 9999, 1234, 2024, 31415, 27182, 7, 
    11, 13, 888, 555, 333, 99, 1001, 5050, 8080, 1998
]
DEVICE = torch.device("cpu")
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
OUT_DIR_PREFIX = "results/final_run_100_epochs"

def evaluate_seeds(model_name, model_class, model_kwargs):
    model_dir = f"{OUT_DIR_PREFIX}/{model_name}"
    print(f"\nEvaluating {model_name.upper()} (20 Seeds)...")
    
    test_loader = DataLoader(ECGDataset("test"), batch_size=512, shuffle=False)
    
    # Store metrics for each seed
    accuracies = []
    macro_f1s = []
    v_recalls = []
    s_recalls = []
    
    true_labels = np.concatenate([y.numpy() for _, y in test_loader], axis=0)
    
    for i, seed in enumerate(SEEDS):
        model_path = f"{model_dir}/model_seed_{seed}.pth"
        if not os.path.exists(model_path):
            print(f"  Warning: Seed {seed} missing.")
            continue
            
        model = model_class(**model_kwargs)
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        
        all_preds = []
        with torch.no_grad():
            for X, y in test_loader:
                out = model(X)
                preds = torch.argmax(out, dim=1)
                all_preds.append(preds.numpy())
        
        final_preds = np.concatenate(all_preds, axis=0)
        
        # Metrics
        report = classification_report(true_labels, final_preds, target_names=CLASS_NAMES, output_dict=True)
        acc = report['accuracy']
        f1 = report['macro avg']['f1-score']
        v_rec = report['V']['recall']
        s_rec = report['S']['recall']
        
        accuracies.append(acc)
        macro_f1s.append(f1)
        v_recalls.append(v_rec)
        s_recalls.append(s_rec)
        
        print(f"  Seed {seed}: F1={f1:.4f}, V-Rec={v_rec:.4f}, S-Rec={s_rec:.4f}")

    if not accuracies:
        print("No models evaluated.")
        return None

    # Compute Statistics
    stats = {
        'model': model_name,
        'acc_mean': np.mean(accuracies),
        'acc_std': np.std(accuracies),
        'f1_mean': np.mean(macro_f1s),
        'f1_std': np.std(macro_f1s),
        'v_recall_mean': np.mean(v_recalls),
        'v_recall_std': np.std(v_recalls),
        's_recall_mean': np.mean(s_recalls),
        's_recall_std': np.std(s_recalls)
    }
    
    return stats

def main():
    print("="*60)
    print("📊 FINAL RUN 100-EPOCH RESULTS (Mean ± Std)")
    print("="*60)
    
    all_stats = []
    
    # 1. CNN
    stats_cnn = evaluate_seeds('cnn', CNN1D, {'num_classes': 5, 'input_len': 360})
    if stats_cnn: all_stats.append(stats_cnn)
    
    # 2. WavKAN
    stats_kan = evaluate_seeds('wavkan', WavKANClassifier, {'input_size': 360, 'num_classes': 5})
    if stats_kan: all_stats.append(stats_kan)
    
    # 3. Hybrid
    stats_hyb = evaluate_seeds('hybrid', HybridWavKAN, {'input_size': 360})
    if stats_hyb: all_stats.append(stats_hyb)
    
    # Final Table
    print("\n" + "="*80)
    print(f"{'Model':<10} {'Macro F1 (Mean ± Std)':<25} {'V-Recall (Mean ± Std)':<25} {'S-Recall (Mean ± Std)':<25}")
    print("-" * 80)
    for s in all_stats:
        f1_str = f"{s['f1_mean']:.4f} ± {s['f1_std']:.4f}"
        v_str = f"{s['v_recall_mean']:.4f} ± {s['v_recall_std']:.4f}"
        s_str = f"{s['s_recall_mean']:.4f} ± {s['s_recall_std']:.4f}"
        print(f"{s['model'].upper():<10} {f1_str:<25} {v_str:<25} {s_str:<25}")
    print("="*80)
    
    # Save to CSV
    df = pd.DataFrame(all_stats)
    df.to_csv(f"{OUT_DIR_PREFIX}/final_stats.csv", index=False)
    print(f"\nSaved stats to {OUT_DIR_PREFIX}/final_stats.csv")

if __name__ == "__main__":
    main()
