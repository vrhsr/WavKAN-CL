"""
Generate polished WavKAN Micro-Architecture Diagram for publication
Improvements:
1. Correct terminology (Summation Nodes vs Input Nodes)
2. Mathematical precision (x_i, y_j, phi)
3. Explicit Mexican Hat wavelet visualization
4. Professional styling matching other figures
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def mexican_hat(t):
    return (1 - t**2) * np.exp(-0.5 * t**2)

def create_wavkan_micro_figure():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Style constants
    NODE_RADIUS = 0.4
    WAVELET_BOX_SIZE = 1.0
    
    # Colors
    c_input = '#2E4053'  # Dark Bule
    c_sum = '#27AE60'    # Green
    c_wavelet_line = '#E67E22' # Orange
    c_edge_active = '#999999'
    c_edge_faint = '#E0E0E0'

    # --- Coordinates ---
    # Inputs (Left)
    input_coords = [(2, 4.5), (2, 3.0), (2, 1.5)]
    # Outputs (Right)
    output_coords = [(8, 4.5), (8, 3.0), (8, 1.5)]
    
    # Draw faint background connections (all-to-all)
    for ix, iy in input_coords:
        for ox, oy in output_coords:
            ax.plot([ix, ox], [iy, oy], color=c_edge_faint, lw=1, zorder=1)

    # --- Valid Active Connections (Logic: just show 3 central paths clearly) ---
    
    # Path 1: Top-Top
    ax.plot([2, 8], [4.5, 4.5], color=c_edge_active, lw=2, zorder=2)
    # Path 2: Mid-Mid
    ax.plot([2, 8], [3.0, 3.0], color=c_edge_active, lw=2, zorder=2)
    # Path 3: Bot-Bot
    ax.plot([2, 8], [1.5, 1.5], color=c_edge_active, lw=2, zorder=2)

    # --- Draw Nodes ---
    
    # Input Nodes
    for i, (x, y) in enumerate(input_coords):
        circle = plt.Circle((x, y), NODE_RADIUS, color=c_input, zorder=10)
        ax.add_artist(circle)
        ax.text(x-0.8, y, f'Input $x_{i+1}$', ha='right', va='center', fontsize=12, fontweight='bold')

    # Output (Summation) Nodes
    for i, (x, y) in enumerate(output_coords):
        circle = plt.Circle((x, y), NODE_RADIUS, color=c_sum, zorder=10)
        ax.add_artist(circle)
        ax.text(x, y, '$\Sigma$', ha='center', va='center', fontsize=16, color='white', fontweight='bold', zorder=11)
        ax.text(x+0.6, y, f'$y_{i+1}$', ha='left', va='center', fontsize=14, fontweight='bold')

    # --- Draw Wavelet Boxes on Active Edges ---
    
    mid_x = 5
    y_pos = [4.5, 3.0, 1.5]
    
    t = np.linspace(-2.5, 2.5, 100)
    w = mexican_hat(t)
    
    for y in y_pos:
        # White box background
        rect = patches.Rectangle((mid_x - 0.6, y - 0.5), 1.2, 1.0, 
                                 linewidth=1, edgecolor='#999999', facecolor='white', zorder=5)
        ax.add_patch(rect)
        
        # Wavelet inside
        # Scale wavelet to fit box
        w_scaled = (w * 0.4) + y - 0.1
        t_scaled = (t * 0.2) + mid_x
        ax.plot(t_scaled, w_scaled, color=c_wavelet_line, lw=2, zorder=6)

    # --- Annotations ---
    
    # Main equation
    ax.text(5, 5.5, r"$\phi(x) = \text{MexicanHat}(\frac{x - \mu}{\sigma})$", 
            ha='center', va='center', fontsize=14, fontweight='bold', 
            bbox=dict(facecolor='#FFF8E1', edgecolor='#FBC02D', boxstyle='round,pad=0.5'))

    ax.text(5, 0.5, "Unlike MLPs, activation functions are on edges (learnable), not nodes.", 
            ha='center', va='center', fontsize=12, style='italic', color='#555555')


    # Column Titles
    ax.text(2, 5.5, "Input Layer", ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(8, 5.5, "WavKAN Layer\n(Summation)", ha='center', va='center', fontsize=14, fontweight='bold')


    plt.tight_layout()
    output_path = r'e:\The\wavkan_micro_architecture_v2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")

if __name__ == "__main__":
    create_wavkan_micro_figure()
