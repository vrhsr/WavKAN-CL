
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import networkx as nx

def plot_mexican_hat(ax, x_center, y_center, width, height, color='blue'):
    """Draws a mini Mexican Hat wavelet at the specified location."""
    t = np.linspace(-4, 4, 50)
    # Mexican Hat: (1 - t^2) * exp(-t^2/2)
    y = (1 - t**2) * np.exp(-t**2 / 2)
    
    # Normalize to fit box
    y_norm = (y - y.min()) / (y.max() - y.min()) # 0 to 1
    y_scaled = y_norm * height + (y_center - height/2)
    x_scaled = np.linspace(x_center - width/2, x_center + width/2, 50)
    
    ax.plot(x_scaled, y_scaled, color=color, linewidth=1.5, alpha=0.9)
    # ax.add_patch(Rectangle((x_center-width/2, y_center-height/2), width, height, fill=False, edgecolor='gray', alpha=0.3, lw=0.5))

def draw_wavkan_structure():
    """
    Generates a schematic of the WavKAN Layer.
    Structure: Input (3 nodes) -> [WavKAN Layer: Wavelets on Edges] -> Hidden (3 nodes) -> Sum -> Output
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Coordinates
    layer_1_x = 2
    layer_2_x = 6
    layer_3_x = 8 # Post-summation / LayerNorm?
    
    nodes_y = [1.5, 3.0, 4.5] # 3 Nodes per layer for simplicity
    
    # --- DRAW INPUT LAYER ---
    for y in nodes_y:
        # Node
        circle = Circle((layer_1_x, y), 0.2, color='#2c3e50', zorder=10)
        ax.add_patch(circle)
        # Label
        ax.text(layer_1_x - 0.5, y, "Input", ha='right', va='center', fontsize=10, fontweight='bold', color='#34495e')
    
    ax.text(layer_1_x, 5.2, "Input Nodes\n(ECG Time Series Samples)", ha='center', va='center', fontsize=12, fontweight='bold')
    
    # --- DRAW HIDDEN LAYER ---
    for y in nodes_y:
        # Node
        circle = Circle((layer_2_x, y), 0.2, color='#27ae60', zorder=10)
        ax.add_patch(circle)
        # Summation symbol
        ax.text(layer_2_x, y, "$\Sigma$", ha='center', va='center', fontsize=10, color='white', fontweight='bold', zorder=11)

    ax.text(layer_2_x, 5.2, "WavKAN Layer\n(Wavelet Features)", ha='center', va='center', fontsize=12, fontweight='bold')

    # --- DRAW CONNECTIONS (EDGES WITH WAVELETS) ---
    # Fully connected 3x3
    
    # We draw lines with a "break" in the middle for the wavelet box
    
    colors = ['#e74c3c', '#e67e22', '#f39c12']
    
    for i, y1 in enumerate(nodes_y):
        for j, y2 in enumerate(nodes_y):
            # Line coords
            start = (layer_1_x + 0.2, y1)
            end = (layer_2_x - 0.2, y2)
            
            # Midpoint
            mid_x = (start[0] + end[0]) / 2
            mid_y = (start[1] + end[1]) / 2
            
            # Offset midpoints slightly to avoid overlap if needed, but for 3x3 specific ones are clearer
            # Let's clean it up: only show connections from Middle Input to All Outputs, and All Inputs to Middle Output
            # to avoid visual clutter? 
            # No, let's draw all but with high transparency for background ones
            
            alpha = 0.2
            lw = 1
            is_main = False
            
            # Highlight a few paths clearly
            if i == 1: # Middle input to all outputs
                alpha = 0.8
                lw = 2
                is_main = True
            
            # Draw line segments
            # Seg 1
            ax.plot([start[0], mid_x - 0.4], [start[1], mid_y - (y2-y1)*0.1], color='gray', alpha=alpha, lw=lw, zorder=1)
            # Seg 2
            ax.plot([mid_x + 0.4, end[0]], [mid_y + (y2-y1)*0.1, end[1]], color='gray', alpha=alpha, lw=lw, zorder=1)
            
            # Draw Wavelet Box at Midpoint
            if is_main:
                # White box background
                rect = Rectangle((mid_x - 0.4, mid_y - 0.3), 0.8, 0.6, facecolor='white', edgecolor='#bdc3c7', lw=1, zorder=2)
                ax.add_patch(rect)
                
                # Plot Wavelet inside
                plot_mexican_hat(ax, mid_x, mid_y, 0.7, 0.5, color=colors[j])
                
                # Annotation for one of them
                if i == 1 and j == 2: # Top one
                    # UPDATED FORMULA
                    ax.text(mid_x, mid_y + 0.5, r"$\phi(\frac{t-\mu}{\sigma})$", ha='center', va='bottom', fontsize=10, color='#e74c3c', fontweight='bold')

    # --- EXPLANATORY TEXT ---
    ax.text(5, 0.5, "WavKAN Structure: Learnable Wavelet Activation Functions on Edges", ha='center', va='center', fontsize=12, fontstyle='italic', bbox=dict(facecolor='#ecf0f1', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.5'))

    # Save
    import os
    os.makedirs("results/figures/publication", exist_ok=True)
    plt.savefig("results/figures/publication/wavkan_micro_architecture.png", dpi=300, bbox_inches='tight')
    print("Saved custom WavKAN diagram to results/figures/publication/wavkan_micro_architecture.png")

if __name__ == "__main__":
    draw_wavkan_structure()
