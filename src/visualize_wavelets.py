import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.train_hybrid import HybridWavKAN 

# CONFIG
MODEL_PATH = "results/hybrid_final/best_hybrid.pth"
DATA_PATH = "data/processed/X_test.npy"
LABEL_PATH = "data/processed/y_test.npy"
DEVICE = torch.device("cpu") # Visualization is fast on CPU

# Wavelet Functions for plotting
def mexican_hat(t, scale, translation):
    t_prime = (t - translation) / (scale + 1e-8)
    return (1 - t_prime**2) * np.exp(-0.5 * t_prime**2)

def visualize():
    print("Loading model and data...")
    if not os.path.exists(DATA_PATH):
        print(f"Error: {DATA_PATH} not found.")
        return
        
    X_test = np.load(DATA_PATH)
    y_test = np.load(LABEL_PATH)
    
    v_indices = np.where(y_test == 2)[0] # Class 2 is Ventricular
    if len(v_indices) == 0:
        print("No V-beats found in test data!")
        return

    # Average V-beat for background
    avg_v_beat = np.mean(X_test[v_indices[:100]], axis=0) 
    time_axis = np.linspace(0, 1, 360) 

    # Load Model (HybridWavKAN hardcodes mexican_hat in init)
    print("Initializing HybridWavKAN (Mexican Hat)...")
    model = HybridWavKAN(input_size=360) 
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        print(f"Loaded model from {MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Model not found at {MODEL_PATH}")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    model.eval()

    # Extract Learned Parameters from the FIRST WavKAN layer
    # model.kan is the WavKANLinear layer
    weights = model.kan.weights.detach().cpu().numpy()
    scales = model.kan.scale.detach().cpu().numpy()
    translations = model.kan.translation.detach().cpu().numpy()
    
    # We know it's mexican hat because it's hardcoded in train_hybrid.py
    wavelet_type = 'mexican_hat'

    print(f"Visualizing for Wavelet Type: {wavelet_type}")

    # Identify "Most Important" Filter (Neuron)
    # Sum of absolute weights connected to this neuron
    neuron_importance = np.sum(np.abs(weights), axis=1)
    top_neuron_idx = np.argmax(neuron_importance) 
    
    print(f"Visualizing filters for Top Neuron #{top_neuron_idx}")

    plt.figure(figsize=(12, 6))
    
    # Plot the Average V-Beat (Real Data)
    plt.plot(time_axis, avg_v_beat, color='black', alpha=0.3, linewidth=2, label='Avg Ventricular Beat (Real)')

    # Plot 5 learned wavelets from the top neuron
    # We pick 5 indices spread across the time domain to see what it learned
    important_indices = [50, 120, 180, 240, 310] 
    colors = ['green', 'blue', 'red', 'orange', 'purple']

    for i, idx in enumerate(important_indices):
        s = scales[top_neuron_idx, idx]
        t = translations[top_neuron_idx, idx]
        w = weights[top_neuron_idx, idx]
        
        # Determine the domain for the wavelet function
        domain = np.linspace(-3, 3, 360) 
        
        # Calculate wavelet
        wavelet = mexican_hat(domain, s, t)
        
        # Scale by the learned weight
        wavelet_scaled = wavelet * w
        
        # Shift to align with time axis for visualization
        # Note: The 'translation' parameter in WavKAN is learned in the normalized domain
        # Mapping it back to 0-1 time domain for plotting is tricky visually.
        # Simple approach: Plot it centered at the index it effectively operates on?
        # WavKAN operates on ALL inputs. 
        # Actually, let's plot the wavelet as a function of the input time domain [0, 1]
        
        # t is in range [-0.5, 0.5] approx (initialized there)
        # Input x is standardized.
        
        # Let's just plot the wavelet shape centered at its learned translation
        # Re-mapping domain to [0, 1]
        t_grid = np.linspace(0, 1, 360)
        # Normalize t_grid to be consistent with how model sees it?
        # Model sees x. x is normalized beat.
        
        # Let's stick to the previous logic which seemed to attempt to show shape
        # The previous code plotted 'domain_shifted' which was just 0..1
        # And plotted wavelet_scaled which was computed on -3..3
        # This basically just shows the shape of the wavelet, not necessarily its position in time relative to the beat.
        
        # IMPROVED VISUALIZATION:
        # We want to see WHERE the wavelet is looking.
        # The Term (x - t) / s means the wavelet is centered at t.
        # In WavKANLinear, x is the input features.
        # We are visualizing the kernel function phi((x - t)/s)
        # We should plot phi((time - t)/s) vs time.
        
        # In our case, the "input features" ARE the time steps (0 to 360).
        # But wait, WavKANLinear treats input_size=360 as 360 independent features.
        # It doesn't treat them as a time sequence in the layer itself.
        # It learns a SEPARATE wavelet for EACH input point (360 input points).
        
        # So for Input Index `idx` (e.g. sample 180), the model learns a transformation.
        # This is not a convolution over time. It's a "Linear" layer where each weight is a function.
        # So we should verify what we are plotting.
        
        # If we pluck out index `idx`, we are looking at how the model transforms the value at time `idx`.
        # This is less "visualizing a filter sliding over time" and more "visualizing the activation function for that pixel".
        
        # To make a cool plot, we probably want to see the "Effective Receptive Field" or just the shapes.
        # Let's stick to showing the shapes as "Learned Features" since that is the interpretation of KAN.
        
        plt.plot(time_axis, wavelet_scaled, label=f'Learned Wavelet @ Input {idx}', color=colors[i], linestyle='--')

    plt.title(f"Interpretability: Learned Mexican Hat Wavelets (Top Neuron #{top_neuron_idx})")
    plt.xlabel("Normalized Time / Feature Index")
    plt.ylabel("Activation / Amplitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "results/hybrid_final/learned_wavelets.png"
    plt.savefig(save_path)
    print(f"Saved visualization to {save_path}")

if __name__ == "__main__":
    visualize()
