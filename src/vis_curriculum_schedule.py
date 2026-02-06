
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

def plot_curriculum_schedule():
    """
    Analytically plots the Curriculum Learning strategy.
    
    Logic from train_curriculum.py:
    - Phase 1 (Epochs 0-15): Minority classes only (S, V, F, Q) -> 0% Majority (N)
    - Phase 2 (Epochs 16-50): Full dataset -> ~89% Majority (N) in natural distribution
    """
    
    epochs = 50
    phase1_len = 15
    
    # Create x-axis
    x = np.arange(1, epochs + 1)
    
    # Create y-axis (Percentage of Majority Class Samples)
    # Natural prevalence of N is approx 89% in MIT-BIH
    natural_prevalence = 89.0 
    
    y = np.zeros_like(x, dtype=float)
    y[phase1_len:] = natural_prevalence # Phase 2 jumps to full dataset
    
    # Setup aesthetic
    sns.set_style("whitegrid")
    plt.figure(figsize=(10, 6))
    
    # Plot line
    plt.step(x, y, where='post', linewidth=3, color='#2c3e50', label='Majority Class Exposure')
    
    # Add colored regions
    plt.axvspan(0, phase1_len, alpha=0.2, color='#e74c3c', label='Phase 1: Minority Focus')
    plt.axvspan(phase1_len, epochs, alpha=0.2, color='#27ae60', label='Phase 2: Global Optimization')
    
    # Annotations
    plt.text(7.5, 45, "Phase 1\n(Minority Only)", ha='center', va='center', fontsize=12, fontweight='bold', color='#c0392b')
    plt.text(32.5, 45, "Phase 2\n(Full Dataset)", ha='center', va='center', fontsize=12, fontweight='bold', color='#2ecc71')
    
    # Labels
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("% of Majority (N) Samples in Batch", fontsize=12)
    plt.title("Curriculum Learning Schedule", fontsize=14, fontweight='bold')
    plt.xlim(0, 50)
    plt.ylim(-5, 100)
    
    # Save
    out_dir = Path("results/figures/publication")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(out_dir / "curriculum_schedule.png", dpi=300, bbox_inches='tight')
    plt.savefig(out_dir / "curriculum_schedule.pdf", bbox_inches='tight')
    print(f"Saved schedule plot to {out_dir}/curriculum_schedule.png")
    
if __name__ == "__main__":
    plot_curriculum_schedule()
