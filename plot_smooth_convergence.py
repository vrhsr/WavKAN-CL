"""
Re-plot convergence curves with EMA smoothing over real training data.
Raw curves shown as faded background; smoothed curves in bold foreground.
"""
import json
import numpy as np
import matplotlib.pyplot as plt

def ema(data, alpha=0.3):
    """Exponential moving average smoothing."""
    smoothed = [data[0]]
    for val in data[1:]:
        smoothed.append(alpha * val + (1 - alpha) * smoothed[-1])
    return smoothed

with open('results/convergence_history.json') as f:
    hist = json.load(f)

baseline = hist['baseline']
curriculum = hist['curriculum']
epochs = np.arange(1, 51)
phase_transition = 15

fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))

panels = [
    ('(a) Training Loss', 'loss', 'Loss'),
    ('(b) Val Macro-F1', 'f1_macro', 'Macro-F1'),
    ('(c) V-Recall', 'v_recall', 'V-Recall'),
    ('(d) S-Recall', 's_recall', 'S-Recall'),
]

for ax, (title, key, ylabel) in zip(axes, panels):
    # Raw curves (faded)
    ax.plot(epochs, baseline[key], color='#2196F3', alpha=0.15, linewidth=1)
    ax.plot(epochs, curriculum[key], color='#FF5722', alpha=0.15, linewidth=1)
    
    # Smoothed curves (bold)
    ax.plot(epochs, ema(baseline[key]), color='#2196F3', alpha=0.9,
            label='Baseline', linewidth=2.2)
    ax.plot(epochs, ema(curriculum[key]), color='#FF5722', alpha=0.9,
            label='Curriculum', linewidth=2.2)
    
    # Phase transition line
    ax.axvline(x=phase_transition, color='#666666', linestyle='--', alpha=0.5,
               label='Phase Transition' if key == 'loss' else '')
    
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, alpha=0.2)
    if key == 'loss':
        ax.legend(fontsize=9, loc='upper right')

plt.tight_layout()
plt.savefig('fig_convergence_curves.png', dpi=300, bbox_inches='tight')
print("✅ Smoothed figure saved to fig_convergence_curves.png")
