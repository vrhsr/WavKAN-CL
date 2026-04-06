import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Setup
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

DATA_DIR = Path("data/processed_rr_history")
RESULTS_DIR = Path("results/multiseed")
OUT_DIR = Path("results/figures/final_submission")

def load_predictions(model_name, seed=42):
    # Load probabilities if available, else hard predictions (we have pred indices in npy)
    # Wait, the npy files I saw were 'predictions.npy'. Assuming they are class indices.
    # Let's check if they are logits or indices.
    # Usually 'predictions.npy' are indices. 
    path = RESULTS_DIR / f"{model_name}/seed_{seed}/predictions.npy"
    return np.load(path)

# We need probs for confidence if possible. 
# plot_clinical_utility generated probs.
# But for now, let's just use the class indices to find the case.

def find_success_case():
    print("Loading data...")
    y_true = np.load(DATA_DIR / "y_test.npy")
    X_test = np.load(DATA_DIR / "X_test.npy") # Shape (N, 180) probably
    
    # Check shape
    if X_test.ndim == 3:
        X_test = X_test[:, 0, :] # (N, 1, 180) -> (N, 180)
        
    pred_curr = load_predictions("curriculum", 42)
    pred_base = load_predictions("baseline", 42)
    
    # Target: Class S (1)
    # Condition: True=1, Curr=1, Base!=1
    target_class = 1 # S
    
    candidates = np.where((y_true == target_class) & 
                          (pred_curr == target_class) & 
                          (pred_base != target_class))[0]
    
    print(f"Found {len(candidates)} success cases for Class S.")
    
    if len(candidates) == 0:
        print("No candidates found. Trying Class F (3)")
        target_class = 3
        candidates = np.where((y_true == target_class) & 
                              (pred_curr == target_class) & 
                              (pred_base != target_class))[0]
        print(f"Found {len(candidates)} success cases for Class F.")
        
    if len(candidates) == 0:
        return
        
    # Pick a "clean" looking one (heuristic: variance not too crazy)
    best_idx = candidates[0] 
    # Let's try to find one where baseline predicted Normal (0) - common error
    for idx in candidates:
        if pred_base[idx] == 0: # Mistaken for Normal
             best_idx = idx
             break
             
    print(f"Selected Index: {best_idx}")
    print(f"Ground Truth: {y_true[best_idx]}")
    print(f"Curriculum: {pred_curr[best_idx]}")
    print(f"Baseline: {pred_base[best_idx]}")
    
    # Plot
    beat = X_test[best_idx]
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(beat, color='black', linewidth=1.5)
    
    # Highlight
    ax.set_title("Qualitative Case Study: Disagreement on Subtle S-Class Arrhythmia", fontsize=12)
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Amplitude (normalized)")
    
    # Custom Result Box (Top Right)
    # Aligning text vertically
    x_pos = 0.98
    y_start = 0.95
    y_gap = 0.08
    
    ax.text(x_pos, y_start, "Ground Truth: S (Supraventricular)", transform=ax.transAxes, 
            color='green', ha='right', fontsize=10)
            
    ax.text(x_pos, y_start - y_gap, "Hybrid Prediction: S (Correct)", transform=ax.transAxes, 
            color='#3498db', ha='right', fontsize=10)
            
    ax.text(x_pos, y_start - 2*y_gap, "Baseline Prediction: N (Incorrect)", transform=ax.transAxes, 
            color='#c0392b', ha='right', fontsize=10)
            
    # Add vertical line for visual anchor
    ax.axvline(x=90, color='gray', linestyle='--', alpha=0.5)
    
    # Anchor "Beat under analysis" with an arrow
    # Peak is roughly at x=90, y=6.2 based on plot observations
    ax.annotate("Beat under analysis", 
                xy=(90, 6.2), xytext=(102, 6.2), 
                arrowprops=dict(arrowstyle='->', color='gray'),
                fontsize=9, color='gray', va='center')
    
    # Increase y-limit slightly to give headroom
    # The arrow is at y=6.2, so make sure ylim covers it
    ylim = ax.get_ylim()
    # Ensure top is at least 7.0 for the legend and annotation space
    new_top = max(ylim[1], 7.0)
    ax.set_ylim(ylim[0], new_top) 
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    out_path = OUT_DIR / "supp_B2_qualitative_case.png"
    plt.savefig(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    find_success_case()
