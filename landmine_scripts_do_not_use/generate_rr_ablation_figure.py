"""
Generate polished RR-Interval Ablation Figure for publication
Improvements:
1. Remove "Non-Contributory" label
2. Tighten y-axis to [-0.03, 0.08]
3. Shorter title (move methodology to caption)
4. Professional styling
"""

import matplotlib.pyplot as plt
import numpy as np

# Set publication-quality style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['figure.dpi'] = 150

# Data from ablation study (mean ± std across 10 seeds)
rr_intervals = ['t-5', 't-4', 't-3', 't-2', 't-1']
delta_s_recall = [-0.004, -0.006, 0.062, -0.002, 0.004]
std_dev = [0.008, 0.012, 0.015, 0.006, 0.005]

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Color scheme: highlight t-3 as dominant
colors = ['#E57373' if i != 2 else '#D32F2F' for i in range(5)]
edge_colors = ['#B71C1C' if i == 2 else '#C62828' for i in range(5)]

# Create bars
bars = ax.bar(rr_intervals, delta_s_recall, 
              color=colors, 
              edgecolor=edge_colors,
              linewidth=1.5,
              yerr=std_dev, 
              capsize=5,
              error_kw={'linewidth': 1.5, 'capthick': 1.5})

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, delta_s_recall)):
    height = bar.get_height()
    offset = 0.008 if height >= 0 else -0.012
    fontweight = 'bold' if i == 2 else 'normal'
    if i == 2:  # t-3 bar (0.062) - move to right for better visibility
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2 + 0.15, height + 0.005), # Offset right and slightly up
                    ha='left', va='center',
                    fontsize=12, fontweight='bold',
                    color='#1a1a1a')
    else:
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height + offset),
                    ha='center', va='bottom' if height >= 0 else 'top',
                    fontsize=11, fontweight=fontweight,
                    color='#1a1a1a')

# Add horizontal line at y=0
ax.axhline(y=0, color='#666666', linestyle='-', linewidth=1)

# Labels and title (simplified - no methodology in title)
ax.set_xlabel('RR Interval Index (History)', fontsize=13, fontweight='medium')
ax.set_ylabel('Change in S-Recall After Feature Removal', fontsize=13, fontweight='medium')
ax.set_title('Ablation of RR-Interval History on S-Class Recall', 
             fontsize=15, fontweight='bold', pad=15)

# Tighten y-axis as recommended
ax.set_ylim(-0.03, 0.08)

# Add subtle grid
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.set_axisbelow(True)

# Highlight t-3 with annotation arrow (optional elegant touch)
ax.annotate('Dominant\nContributor', 
            xy=(2, 0.062), 
            xytext=(3.2, 0.055),
            fontsize=10, 
            fontstyle='italic',
            color='#B71C1C',
            arrowprops=dict(arrowstyle='->', color='#B71C1C', lw=1.5))

# Remove top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Adjust layout
plt.tight_layout()

# Save figure
output_path = r'e:\The\fig_rr_ablation_v2.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig(output_path.replace('.png', '.pdf'), dpi=300, bbox_inches='tight', facecolor='white')

print(f"✅ Figure saved to: {output_path}")
print(f"✅ PDF version saved to: {output_path.replace('.png', '.pdf')}")

plt.show()
