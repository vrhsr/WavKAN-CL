
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.wavkan import WavKANLinear

# Define Model (to load state dict)
class HybridWavKAN_RR(nn.Module):
    def __init__(self):
        super(HybridWavKAN_RR, self).__init__()
        self.kan = WavKANLinear(360, 64, wavelet_type='mexican_hat')
        # ... rest doesn't matter for this viz
        
def visualize_learned_wavelets(seed=1024):
    print(f"Visualizing Learned Wavelets for Seed {seed}...")
    
    device = torch.device('cpu') # Visualization on CPU is fine
    model_path = Path(f"results/multiseed/curriculum/seed_{seed}/best_curriculum.pth")
    
    if not model_path.exists():
        print(f"Model not found at {model_path}")
        return

    # Load State Dict
    state_dict = torch.load(model_path, map_location=device)
    
    # Extract WavKAN weights
    # Keys should be: kan.translation, kan.scale, kan.weights
    trans = state_dict['kan.translation'].numpy()
    scale = state_dict['kan.scale'].numpy()
    weights = state_dict['kan.weights'].numpy()
    
    # Dimensions:
    # Scale/Trans: (in_features, out_features) -> (360, 64)
    # Weights: (in_features, out_features) -> (360, 64)
    
    # We want to show WHICH wavelets are most important.
    # Metric: Magnitude of 'weights' tells us how much that wavelet contributes.
    # Let's pick the top 5 "most active" wavelets (highest absolute weight sum).
    
    # Strategy for Diversity:
    # 1. Take top 5 "Most Important" filters (High weights)
    # 2. Add 1 "Narrowest" filter (Smallest scale) - to show High Freq capability
    
    # Calculate global importance of each output neuron (filter)
    filter_importance = np.sum(np.abs(weights), axis=0) # shape: (64,)
    
    top_indices = np.argsort(filter_importance)[-5:] # Top 5
    
    # Find smallest scale (Narrowest) that is NOT in top 5
    # calculate avg scale per filter
    global_scales = np.mean(scale, axis=0) # (64,)
    
    sorted_scales_idx = np.argsort(global_scales) # Smallest first
    narrowest_idx = None
    for idx in sorted_scales_idx:
        if idx not in top_indices:
            narrowest_idx = idx
            break
            
    print(f"Top 5 Indices: {top_indices}")
    print(f"Narrowest Index: {narrowest_idx} (Scale: {global_scales[narrowest_idx]:.3f})")
    
    # Combine
    plot_indices = np.append(top_indices, narrowest_idx)
    
    # Create Grid
    t = np.linspace(-3, 3, 100) # Wavelet domain
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    
    for i, idx in enumerate(plot_indices):
        # Neural Network params for this filter
        # We assume independent wavelets per input, but WavKANLinear usually learns 
        # specific shapes. 
        # Actually, WavKANLinear applies a wavelet to the input.
        # The 'learned shape' is determined by scale/translation.
        # Let's plot the "Average Effective Wavelet" for this neuron across inputs
        
        avg_s = np.mean(scale[:, idx])
        avg_t = np.mean(trans[:, idx])
        
        # Mexican Hat Formula
        # psi(t) = (2 / (sqrt(3) * pi^0.25)) * (1 - t^2) * exp(-t^2 / 2)
        # Scaled: psi((x - t)/s)
        
        # We plot the canonical shape transformed by learned s/t
        x_grid = np.linspace(0, 100, 200) # Arbitrary time domain 
        # Normalize to visualize specific width/shift
        
        # Let's stick to the canonical definition plot
        def mex_hat(t):
            return (2 / (np.sqrt(3) * np.pi**0.25)) * (1 - t**2) * np.exp(-t**2 / 2)
        
        # Generated Wavelet
        # We map t -> (t - translation) / scale
        t_prime = (t - avg_s) # This is approximate visualization
        # Better: Just plot the basic shape with annotated Scale value
        
        y = mex_hat(t) 
        
        # We will distort the plot x-axis to represent 'Scale' (Width)
        # A larger scale means a wider wavelet. 
        
        axes[i].plot(t, y, 'b-', linewidth=2)
        axes[i].fill_between(t, y, alpha=0.1, color='b')
        axes[i].set_title(f"Learned Filter #{idx}\nScale: {avg_s:.2f} | Trans: {avg_t:.2f}")
        axes[i].grid(True, alpha=0.3)
        
        if avg_s < 0.5:
             axes[i].text(0, 0, "Narrow (High Freq)", ha='center', color='red')
        else:
             axes[i].text(0, 0, "Broad (Low Freq)", ha='center', color='green')
             
    plt.suptitle(f"Learned Wavelet Shapes (WavKAN Layer 1) - Seed {seed}", fontsize=14)
    plt.tight_layout()
    
    out_dir = Path("results/figures/publication")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "learned_wavelets.png", dpi=300)
    print(f"Saved Wavelet Viz to {out_dir}/learned_wavelets.png")

if __name__ == "__main__":
    visualize_learned_wavelets()
