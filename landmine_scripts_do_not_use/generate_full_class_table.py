
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
import os

def main():
    # Load best curriculum seed predictions (Seed 42 is usually a good candidate or 123)
    # Based on previous analysis, Seed 42 had F1 ~0.36. Let's start there.
    # Actually, let's find the best seed from the json files or just use 42 as representative.
    # The user mentioned "Best" in Table 2 has S-RECALL 0.45. I need to find WHICH seed that was.
    # Based on multiseed_results_analysis.md, Seed 789 had S-Recall 0.30, Seed 123 had S-Recall 0.25... 
    # Wait, the "Peak Performance" table claims S-Recall 0.45.
    # I need to find the seed that gave that result to generate the correct table.
    
    seeds = [42, 123, 456, 789, 1024, 2048, 3141, 5678, 8192, 9999]
    best_s_recall = -1
    best_seed = -1
    
    print("Searching for best seed...")
    for seed in seeds:
        pred_path = f"results/multiseed/curriculum/seed_{seed}/predictions.npy"
        true_path = f"results/multiseed/curriculum/seed_{seed}/true_labels.npy"
        
        if os.path.exists(pred_path) and os.path.exists(true_path):
            y_pred = np.load(pred_path)
            y_true = np.load(true_path)
            
            # S is class 1
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
            s_recall = report['1']['recall']
            
            if s_recall > best_s_recall:
                best_s_recall = s_recall
                best_seed = seed
                
    print(f"Best Seed Found: {best_seed} (S-Recall: {best_s_recall:.4f})")
    
    # Generate table for best seed
    y_pred = np.load(f"results/multiseed/curriculum/seed_{best_seed}/predictions.npy")
    y_true = np.load(f"results/multiseed/curriculum/seed_{best_seed}/true_labels.npy")
    
    target_names = ['N', 'S', 'V', 'F', 'Q']
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True, zero_division=0)
    
    print("\n## Table 6: Detailed Per-Class Performance (Best Model)")
    print("| Class | Precision | Recall | F1-Score | Support |")
    print("| :--- | :--- | :--- | :--- | :--- |")
    
    for cls in target_names:
        r = report[cls]
        print(f"| **{cls}** | {r['precision']:.3f} | {r['recall']:.3f} | {r['f1-score']:.3f} | {r['support']} |")
        
if __name__ == "__main__":
    main()
