
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def draw_box(ax, center, width, height, text, color='lightblue', edgecolor='black', fontsize=10):
    """Helper to draw a box with text."""
    # Shadow
    shadow = patches.FancyBboxPatch(
        (center[0] - width/2 + 0.01, center[1] - height/2 - 0.01),
        width, height,
        boxstyle="round,pad=0.05",
        ec="none",
        fc='gray',
        alpha=0.3,
        mutation_scale=0.5
    )
    ax.add_patch(shadow)
    
    # Main Box
    box = patches.FancyBboxPatch(
        (center[0] - width/2, center[1] - height/2),
        width, height,
        boxstyle="round,pad=0.05",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=color
    )
    ax.add_patch(box)
    
    ax.text(center[0], center[1], text, ha='center', va='center', fontsize=fontsize, fontweight='bold', color='black')
    return box

def draw_arrow(ax, start, end, text=None):
    """Helper to draw a fancy arrow."""
    ax.annotate(
        '', xy=end, xytext=start,
        arrowprops=dict(arrowstyle='->', lw=2, color='black', mutation_scale=15)
    )
    if text:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        ax.text(mid_x, mid_y + 0.03, text, ha='center', va='bottom', fontsize=9, fontstyle='italic')

def generate_workflow_figure(output_path):
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # --- 1. Inputs ---
    # ECG Signal
    draw_box(ax, (1.5, 4.5), 1.8, 0.6, "Raw ECG Signal\n(5s Window)", color='#E6F3FF')
    # RR Interval
    draw_box(ax, (1.5, 2.5), 1.8, 0.6, "RR Intervals\n(Pre/Post Beat)", color='#FFE6E6')
    
    # --- 2. Feature Extraction ---
    # WavKAN Backbone
    draw_box(ax, (4.0, 4.5), 2.2, 0.8, "WavKAN Backbone\n(Learnable Wavelets)", color='#CCE5FF', edgecolor='#0066CC')
    # RR Embedding
    draw_box(ax, (4.0, 2.5), 2.2, 0.8, "RR-Timing Encoder\n(MLP on RR-history)", color='#FFCCCC', edgecolor='#CC0000')
    
    # Arrows Inputs -> Features
    draw_arrow(ax, (2.4, 4.5), (2.9, 4.5))
    draw_arrow(ax, (2.4, 2.5), (2.9, 2.5))
    
    # --- 3. Fusion ---
    # Concatenation
    draw_box(ax, (6.5, 3.5), 1.0, 3.0, "Feature\nFusion", color='#E0E0E0', edgecolor='#666666')
    
    # Arrows Features -> Fusion
    draw_arrow(ax, (5.1, 4.5), (6.0, 4.5), "Morphology")
    draw_arrow(ax, (5.1, 2.5), (6.0, 2.5), "Rhythm")
    
    # --- 4. Curriculum Strategy (The "Brain") ---
    # Difficulty Assessment
    draw_box(ax, (4.0, 1.0), 3.0, 0.6, "Curriculum Scheduler\n(Training-only)", color='#FFF2CC', edgecolor='#E6AC00')
    
    # Arrow Scheduler -> Training (Conceptual loop)
    # Redirecting arrow to point towards the Classifier/Loss calculation area distinct from Feature Fusion
    # Using a dashed line to indicate "Training Signal" not "Data Flow"
    style = "Simple, tail_width=0.5, head_width=4, head_length=8"
    
    # Pointing to the right side (Classifier/Loss)
    ax.annotate('', xy=(8.5, 2.9), xytext=(5.5, 1.0), arrowprops=dict(arrowstyle='->', lw=2, color='#E6AC00', ls='--'))
    
    # Labeling the arrow
    ax.text(7.2, 1.9, "Loss / Sample Weighting\n(Does not affect Inference)", color='#E6AC00', fontsize=9, fontweight='bold', ha='center',
            bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', pad=2))

    
    # --- 5. Output ---
    # Classifier
    draw_box(ax, (8.5, 3.5), 1.5, 0.8, "Classifier\n(Softmax)", color='#D4EDDA', edgecolor='#28A745')
    
    # Arrow Fusion -> Classifier
    draw_arrow(ax, (7.0, 3.5), (7.75, 3.5))
    
    # Final Output
    ax.text(9.5, 3.5, "Prediction\n(N, S, V, F, Q)", ha='left', va='center', fontsize=12, fontweight='bold')
    draw_arrow(ax, (9.25, 3.5), (9.5, 3.5))
    
    # --- Titles & Labels ---
    plt.suptitle("Proposed Hybrid WavKAN Architecture with Curriculum Learning", fontsize=16, y=0.96, fontweight='bold')
    
    # Add a bounding box for the whole system
    # Expanded y-range to prevent overlap with bottom text
    rect = patches.Rectangle((0.5, 0.3), 9.5, 5.4, linewidth=2, edgecolor='gray', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    # Moved text up slightly to clarify it belongs inside the box
    ax.text(5.0, 0.5, "End-to-End Trainable Framework", ha='center', fontsize=10, color='gray', fontstyle='italic')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Generated workflow diagram at {output_path}")

if __name__ == "__main__":
    generate_workflow_figure(r"e:\The\results\figures\final_submission\final_methodology_workflow.png")
