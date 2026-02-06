import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.wavkan import WavKANLinear, WavKANConv1d

def plot_learned_wavelets(model_path, output_dir="results/figures"):
    """Extract and plot learned wavelets from WavKAN layers."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    from models.wavkan import WavKANModel
    model = WavKANModel(input_len=256, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    # Extract wavelet parameters from layer1
    layer = model.layer1
    translation = layer.translation.detach().numpy()  # (out, in)
    scale = layer.scale.detach().numpy()
    weights = layer.weights.detach().numpy()
    
    # Plot a few representative wavelets
    t = np.linspace(-5, 5, 200)
    
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    
    for i in range(4):
        for j in range(4):
            out_idx = i * 16  # Sample output neurons
            in_idx = j * 64   # Sample input features
            
            if out_idx >= translation.shape[0] or in_idx >= translation.shape[1]:
                continue
                
            mu = translation[out_idx, in_idx]
            sigma = abs(scale[out_idx, in_idx])
            w = weights[out_idx, in_idx]
            
            t_shifted = (t - mu) / max(sigma, 1e-5)
            psi = (1 - t_shifted**2) * np.exp(-0.5 * t_shifted**2)  # Mexican Hat
            
            axes[i, j].plot(t, w * psi, 'b-', linewidth=1.5)
            axes[i, j].axhline(0, color='gray', linestyle='--', alpha=0.5)
            axes[i, j].set_title(f'μ={mu:.2f}, σ={sigma:.2f}', fontsize=9)
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
    
    plt.suptitle("Learned Wavelet Basis Functions (WavKAN Layer 1)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/learned_wavelets.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/learned_wavelets.png")

def overlay_wavelets_on_beat(model_path, data_path="data/processed", output_dir="results/figures"):
    """Overlay learned wavelet responses on sample ECG beats."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    X_test = np.load(f"{data_path}/X_test.npy")
    y_test = np.load(f"{data_path}/y_test.npy")
    
    from models.wavkan import WavKANModel
    model = WavKANModel(input_len=256, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    
    CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']
    
    fig, axes = plt.subplots(5, 1, figsize=(12, 15))
    
    for cls in range(5):
        indices = np.where(y_test == cls)[0]
        if len(indices) == 0:
            continue
            
        beat = X_test[indices[0]]
        
        # Get activations from first WavKAN layer
        x = torch.from_numpy(beat).float().unsqueeze(0)
        with torch.no_grad():
            activation = model.layer1(x).numpy()[0]  # (hidden_dim,)
        
        ax = axes[cls]
        ax.plot(beat, 'b-', linewidth=1.5, label='ECG Beat')
        
        # Overlay activation strength as bar on side
        ax_twin = ax.twinx()
        ax_twin.bar(range(len(activation)), activation, alpha=0.3, color='red', width=1)
        ax_twin.set_ylabel('Wavelet Response')
        
        ax.set_title(f"Class: {CLASS_NAMES[cls]}")
        ax.set_xlabel("Sample")
        ax.set_ylabel("ECG Amplitude")
        ax.legend(loc='upper right')
        
    plt.tight_layout()
    plt.savefig(f"{output_dir}/wavelet_overlay_per_class.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir}/wavelet_overlay_per_class.png")

if __name__ == "__main__":
    model_path = "results/models/wavkan_pure.pth"
    
    if os.path.exists(model_path):
        plot_learned_wavelets(model_path)
        overlay_wavelets_on_beat(model_path)
    else:
        print("WavKAN model not found. Train it first with train_wavkan.py")
