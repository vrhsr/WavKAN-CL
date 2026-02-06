import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Setup
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['figure.dpi'] = 300

RESULTS_DIR = Path("results")
SEEDS = [42, 123, 456, 789, 1024, 2048, 3141, 5678, 8192, 9999]

def get_metrics(model_type, seeds):
    f1s, s_recalls, v_recalls = [], [], []
    
    for seed in seeds:
        if model_type == 'pure_wavkan':
            path = RESULTS_DIR / f"pure_wavkan/seed_{seed}/test_metrics.json"
        else:
            path = RESULTS_DIR / f"multiseed/curriculum/seed_{seed}/test_metrics.json"
            
        if path.exists():
            with open(path) as f:
                d = json.load(f)
                f1s.append(d['f1'])
                s_recalls.append(d['s_recall'])
                v_recalls.append(d.get('v_recall', 0)) # v_recall might be missing in some old runs?
    
    return {
        'f1': (np.mean(f1s), np.std(f1s)),
        's_recall': (np.mean(s_recalls), np.std(s_recalls)),
        'v_recall': (np.mean(v_recalls), np.std(v_recalls))
    }

def plot_ablation():
    print("Calculating metrics...")
    pure_stats = get_metrics('pure_wavkan', SEEDS)
    hybrid_stats = get_metrics('hybrid', SEEDS)
    
    metrics = ['Macro F1', 'S-Recall', 'V-Recall']
    keys = ['f1', 's_recall', 'v_recall']
    
    # Data Prep
    pure_means = [pure_stats[k][0] for k in keys]
    pure_stds = [pure_stats[k][1] for k in keys]
    
    hybrid_means = [hybrid_stats[k][0] for k in keys]
    hybrid_stds = [hybrid_stats[k][1] for k in keys]
    
    # Plotting
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    rects1 = ax.bar(x - width/2, pure_means, width, yerr=pure_stds, label='Pure WavKAN', 
                    color='#95a5a6', alpha=0.8, capsize=5)
    rects2 = ax.bar(x + width/2, hybrid_means, width, yerr=hybrid_stds, label='Hybrid WavKAN (Ours)', 
                    color='#3498db', alpha=0.8, capsize=5) # Ours is Blue
    
    ax.set_ylabel('Score')
    ax.set_title('Ablation Study: Impact of Hybrid Architecture (n=10 Seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    # Add annotations
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.2f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)

    autolabel(rects1)
    autolabel(rects2)
    
    # Annotation for S-Recall Variance
    # S-Recall is at index 1. Pure WavKAN is rects1[1].
    # We want to point to the top of the error bar.
    s_recall_pure_mean = pure_means[1]
    s_recall_pure_std = pure_stds[1]
    
    ax.annotate('High Variance\n(Unstable)',
                xy=(x[1] - width/2, s_recall_pure_mean + s_recall_pure_std),
                xytext=(x[1] - width/2 - 0.2, s_recall_pure_mean + s_recall_pure_std + 0.1),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                fontsize=9, color='red', ha='center')
    
    # Caption
    caption = (
        "Comparison of Pure WavKAN vs. Hybrid WavKAN (Curriculum) across 10 seeds. "
        "While Pure WavKAN shows comparable S-Recall, it exhibits extreme variance (instability). "
        "The Hybrid model achieves superior Macro F1 with significantly improved reliability."
    )
    plt.figtext(0.5, 0.01, caption, wrap=True, horizontalalignment='center', fontsize=10, style='italic', color='gray')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15) # More space for caption
    
    out_path = Path("results/figures/final_submission/supp_C3_ablation.png")
    plt.savefig(out_path)
    print(f"Saved to {out_path}")

if __name__ == "__main__":
    plot_ablation()
