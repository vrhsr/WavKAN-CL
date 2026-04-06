"""
Generate polished Methodology Workflow Diagram for publication
Improvements:
1. Removed outer dotted box
2. Increased font sizes
3. Improved alignment and spacing
4. Professional color scheme matching the original design
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects

def create_workflow_figure():
    # Set up figure
    fig = plt.figure(figsize=(14, 8))
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis('off')

    # Font sizes
    TITLE_SIZE = 16
    TEXT_SIZE = 12
    SMALL_TEXT_SIZE = 10

    # Modern high-contrast colors
    c_ecg_bg = '#E3F2FD'    # Light Blue
    c_ecg_edge = '#1E88E5'  # Blue
    
    c_rr_bg = '#FFEBEE'     # Light Pink
    c_rr_edge = '#E53935'   # Red

    c_fusion_bg = '#E0E0E0' # Grey
    c_fusion_edge = '#757575'

    c_cls_bg = '#E8F5E9'    # Light Green
    c_cls_edge = '#43A047'  # Green

    c_curr_bg = '#FFF8E1'   # Light Yellow
    c_curr_edge = '#FBC02D' # Yellow/Orange

    # --- Draw Nodes ---

    # 1. Raw ECG Signal
    box_ecg = patches.FancyBboxPatch((1, 5.5), 2.5, 1.2, boxstyle="round,pad=0.1", 
                                     fc=c_ecg_bg, ec='black', lw=1.5)
    ax.add_patch(box_ecg)
    ax.text(2.25, 6.1, "Raw ECG Signal\n(5s Window)", ha='center', va='center', fontsize=TEXT_SIZE, fontweight='bold')

    # 2. WavKAN Backbone
    box_wavkan = patches.FancyBboxPatch((4.5, 5.5), 2.5, 1.2, boxstyle="round,pad=0.1", 
                                        fc='#BBDEFB', ec=c_ecg_edge, lw=2)
    ax.add_patch(box_wavkan)
    ax.text(5.75, 6.1, "WavKAN Backbone\n(Learnable Wavelets)", ha='center', va='center', fontsize=TEXT_SIZE, fontweight='bold')

    # 3. RR Intervals
    box_rr = patches.FancyBboxPatch((1, 2.5), 2.5, 1.2, boxstyle="round,pad=0.1", 
                                    fc=c_rr_bg, ec='black', lw=1.5)
    ax.add_patch(box_rr)
    ax.text(2.25, 3.1, "RR Intervals\n(Pre/Post Beat)", ha='center', va='center', fontsize=TEXT_SIZE, fontweight='bold')

    # 4. RR Encoder
    box_rrenc = patches.FancyBboxPatch((4.5, 2.5), 2.5, 1.2, boxstyle="round,pad=0.1", 
                                       fc='#FFCDD2', ec=c_rr_edge, lw=2)
    ax.add_patch(box_rrenc)
    ax.text(5.75, 3.1, "RR-Timing Encoder\n(MLP on RR-history)", ha='center', va='center', fontsize=TEXT_SIZE, fontweight='bold')

    # 5. Feature Fusion
    # Tall vertical box
    box_fusion = patches.FancyBboxPatch((8, 2.5), 1.5, 4.2, boxstyle="round,pad=0.1", 
                                        fc=c_fusion_bg, ec=c_fusion_edge, lw=1.5)
    ax.add_patch(box_fusion)
    ax.text(8.75, 4.6, "Feature\nFusion", ha='center', va='center', fontsize=TEXT_SIZE+2, fontweight='bold')

    # 6. Classifier
    box_cls = patches.FancyBboxPatch((10.5, 4.0), 2.5, 1.2, boxstyle="round,pad=0.1", 
                                     fc=c_cls_bg, ec=c_cls_edge, lw=2)
    ax.add_patch(box_cls)
    ax.text(11.75, 4.6, "Classifier\n(Softmax)", ha='center', va='center', fontsize=TEXT_SIZE, fontweight='bold')

    # 7. Curriculum Scheduler
    box_curr = patches.FancyBboxPatch((4.0, 0.5), 3.5, 1.2, boxstyle="round,pad=0.1", 
                                      fc=c_curr_bg, ec=c_curr_edge, lw=2)
    ax.add_patch(box_curr)
    ax.text(5.75, 1.1, "Curriculum Scheduler\n(Training-only)", ha='center', va='center', fontsize=TEXT_SIZE, fontweight='bold')


    # --- Arrows & Connections ---

    props = dict(boxstyle='round', fc='white', ec='white', alpha=0)
    arrow_props = dict(arrowstyle='->', lw=2, color='black')

    # ECG -> WavKAN
    ax.annotate('', xy=(4.5, 6.1), xytext=(3.6, 6.1), arrowprops=arrow_props)
    
    # RR -> Encoder
    ax.annotate('', xy=(4.5, 3.1), xytext=(3.6, 3.1), arrowprops=arrow_props)

    # WavKAN -> Fusion
    ax.annotate('', xy=(8.0, 6.1), xytext=(7.1, 6.1), arrowprops=arrow_props)
    ax.text(7.55, 6.3, "Morphology", ha='center', va='bottom', fontsize=SMALL_TEXT_SIZE, style='italic')

    # Encoder -> Fusion
    ax.annotate('', xy=(8.0, 3.1), xytext=(7.1, 3.1), arrowprops=arrow_props)
    ax.text(7.55, 3.3, "Rhythm", ha='center', va='bottom', fontsize=SMALL_TEXT_SIZE, style='italic')

    # Fusion -> Classifier
    ax.annotate('', xy=(10.5, 4.6), xytext=(9.6, 4.6), arrowprops=arrow_props)

    # Classifier -> Output
    ax.annotate('', xy=(13.8, 4.6), xytext=(13.1, 4.6), arrowprops=arrow_props)
    ax.text(14.0, 4.6, "Prediction\n(N, S, V, F, Q)", ha='left', va='center', fontsize=TEXT_SIZE+1, fontweight='bold')


    # Curriculum -> Classifier (Dashed Arrow)
    # Drawing a custom dashed arrow requires a bit more care or annotate with linestyle
    ax.annotate('', xy=(11.0, 4.0), xytext=(7.6, 1.1), 
                arrowprops=dict(arrowstyle='->', lw=2, color='#F9A825', linestyle='dashed'))
    
    # Text on dashed arrow
    ax.text(9.5, 2.5, "Loss / Sample Weighting\n(Does not affect Inference)", 
            ha='center', va='center', fontsize=SMALL_TEXT_SIZE, color='#F57F17', fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # --- Title ---
    ax.text(7, 7.5, "Proposed Hybrid WavKAN Architecture with Curriculum Learning", 
            ha='center', va='center', fontsize=18, fontweight='bold')

    # Save
    plt.tight_layout()
    output_path = r'e:\The\final_methodology_workflow_v2.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to: {output_path}")

if __name__ == "__main__":
    create_workflow_figure()
