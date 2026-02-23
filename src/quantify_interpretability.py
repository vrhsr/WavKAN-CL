"""
Wavelet Interpretability Quantification (F1, F2, F3)
=====================================================
Extracts learned μ (translation) and γ (scale) parameters from the trained
WavKAN layer. Computes center frequencies and compares them to the known
spectral range of the QRS complex (5–40 Hz at 360 Hz sampling rate).

Usage:
    python src/quantify_interpretability.py
    python src/quantify_interpretability.py --model-path results/hybrid_rr_history_20_seeds/seed_42/best_hybrid_rr.pth

Outputs:
    - results/interpretability_report.json
    - results/wavelet_center_frequencies.png
    - results/wavelet_qrs_overlay.png
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.wavkan import WavKANLinear


# ---------------------------------------------------------------------------
# Model (must match training code)
# ---------------------------------------------------------------------------
class HybridWavKAN_RR(nn.Module):
    def __init__(self):
        super().__init__()
        self.kan = WavKANLinear(360, 64, wavelet_type='mexican_hat')
        self.ln = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.2)
        self.bigru = nn.GRU(64, 32, 1, batch_first=True, bidirectional=True)
        self.rr_mlp = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, 5)

    def forward(self, x, xr):
        x = self.kan(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = x.unsqueeze(1)
        x, _ = self.bigru(x)
        x = x.squeeze(1)
        xr = self.rr_mlp(xr)
        x = torch.cat((x, xr), dim=1)
        return self.fc2(torch.relu(self.fc1(x)))


def extract_wavelet_params(model_path: str):
    """Extract μ (translation) and γ (scale) from the WavKAN layer."""
    model = HybridWavKAN_RR()
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()

    # WavKAN layer parameters
    # kan.translation: (64, 360) — center position per channel per input
    # kan.scale:       (64, 360) — dilation per channel per input
    # kan.weights:     (64, 360) — amplitude weights
    mu = model.kan.translation.data.numpy()    # (64, 360)
    gamma = model.kan.scale.data.numpy()        # (64, 360)
    weights = model.kan.weights.data.numpy()    # (64, 360)

    return mu, gamma, weights


def compute_center_frequencies(gamma, fs=360.0):
    """
    Compute pseudo-frequency for each wavelet channel.
    
    For Mexican Hat wavelet the center (peak) frequency is:
        f_c = sqrt(5/2) / (2π · a)
    where a is the scale in seconds.
    
    In WavKAN, γ is in normalized input-sample space.
    Physical scale: a_phys = |γ| / fs  (convert to seconds)
    So: f_c = sqrt(5/2) · fs / (2π · |γ|)
    """
    gamma_mean = np.mean(np.abs(gamma), axis=1)  # (64,) — mean |scale| per channel
    # Mexican Hat center frequency formula
    f_c = np.sqrt(2.5) * fs / (2 * np.pi * (gamma_mean + 1e-8))
    return f_c, gamma_mean


def compute_temporal_centers(mu):
    """
    Compute average temporal center (in ms) for each channel.
    
    Each μ_jk shifts the Mexican Hat along the input axis.
    The weighted average position shows where each channel 'looks' in the beat.
    """
    # Average over input dim to find dominant temporal position
    # Weight by absolute value (channels attending strongly to a position)
    mu_weighted_mean = np.mean(mu, axis=1)  # (64,)
    # Convert from sample index offset to ms (360 samples = 1000ms)
    mu_ms = mu_weighted_mean * (1000.0 / 360.0)
    return mu_ms


def analyze_multi_seed(model_dir: str, seeds: list, fs=360.0):
    """Analyze μ, γ stability across multiple seeds."""
    all_freqs = []
    all_mu_ms = []
    
    for seed in seeds:
        path = os.path.join(model_dir, f"seed_{seed}", "best_hybrid_rr.pth")
        if not os.path.exists(path):
            continue
        mu, gamma, _ = extract_wavelet_params(path)
        freq, _ = compute_center_frequencies(gamma, fs)
        mu_ms = compute_temporal_centers(mu)
        all_freqs.append(freq)
        all_mu_ms.append(mu_ms)
    
    if not all_freqs:
        return None, None
    
    freqs = np.array(all_freqs)    # (n_seeds, 64)
    mus = np.array(all_mu_ms)      # (n_seeds, 64)
    
    return freqs, mus


def plot_center_frequencies(center_freq, out_path, qrs_range=(5, 40)):
    """Plot histogram of learned center frequencies vs QRS spectral range."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    ax.hist(center_freq, bins=30, color='#2196F3', alpha=0.7, edgecolor='white', linewidth=0.5)
    ax.axvspan(qrs_range[0], qrs_range[1], alpha=0.15, color='red',
               label=f'QRS spectral range ({qrs_range[0]}–{qrs_range[1]} Hz)')
    ax.axvline(x=qrs_range[0], color='red', linestyle='--', alpha=0.5)
    ax.axvline(x=qrs_range[1], color='red', linestyle='--', alpha=0.5)
    
    in_range = np.sum((center_freq >= qrs_range[0]) & (center_freq <= qrs_range[1]))
    total = len(center_freq)
    
    ax.set_xlabel('Center Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Number of Channels', fontsize=12)
    ax.set_title(f'Learned Wavelet Center Frequencies (WavKAN Layer)\n'
                 f'{in_range}/{total} channels ({100*in_range/total:.0f}%) within QRS range',
                 fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xlim(0, max(80, center_freq.max() * 1.1))
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {out_path}")
    return in_range, total


def plot_temporal_heatmap(mu, weights, out_path):
    """Plot where each WavKAN channel attends in the beat window."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # Top: Weight × Translation heatmap (what each channel looks at)
    attention = np.abs(weights) * np.abs(mu)  # (64, 360)
    im = ax1.imshow(attention, aspect='auto', cmap='hot', interpolation='bilinear')
    ax1.set_ylabel('WavKAN Channel (k)', fontsize=11)
    ax1.set_xlabel('Input Sample Position', fontsize=11)
    ax1.set_title('WavKAN Attention Map: Channel × Input Position', fontsize=13)
    plt.colorbar(im, ax=ax1, label='|weight| × |μ|')
    
    # Mark ECG landmark regions
    # P-wave: ~50-100, QRS: ~140-220, T-wave: ~260-340 (in 360-sample window)
    for region, label, color in [
        ((50, 100), 'P', '#4CAF50'),
        ((140, 220), 'QRS', '#F44336'),
        ((260, 340), 'T', '#FF9800'),
    ]:
        ax1.axvspan(region[0], region[1], alpha=0.15, color=color)
        ax1.text(np.mean(region), -2, label, ha='center', fontsize=10, fontweight='bold', color=color)
    
    # Bottom: Summed attention across all channels
    summed = attention.sum(axis=0)
    ax2.fill_between(range(360), summed, alpha=0.3, color='#2196F3')
    ax2.plot(summed, color='#1565C0', linewidth=1.5)
    ax2.set_xlabel('Input Sample Position (0=R-180, 180=R-peak, 360=R+180)', fontsize=11)
    ax2.set_ylabel('Summed Attention', fontsize=11)
    for region, color in [((50, 100), '#4CAF50'), ((140, 220), '#F44336'), ((260, 340), '#FF9800')]:
        ax2.axvspan(region[0], region[1], alpha=0.1, color=color)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {out_path}")


def plot_seed_stability(freqs_multi, out_path):
    """Plot center frequency stability across seeds."""
    mean_freq = freqs_multi.mean(axis=0)  # (64,)
    std_freq = freqs_multi.std(axis=0)
    
    sorted_idx = np.argsort(mean_freq)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.errorbar(range(64), mean_freq[sorted_idx], yerr=std_freq[sorted_idx],
                fmt='o', markersize=4, capsize=2, color='#1565C0', ecolor='#90CAF9')
    ax.axhspan(5, 40, alpha=0.1, color='red', label='QRS range (5–40 Hz)')
    ax.set_xlabel('Channel (sorted by frequency)', fontsize=11)
    ax.set_ylabel('Center Frequency (Hz)', fontsize=11)
    ax.set_title(f'Wavelet Center Frequency Stability ({freqs_multi.shape[0]} seeds)\n'
                 f'Mean σ per channel: {std_freq.mean():.2f} Hz', fontsize=13)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"  ✅ Saved: {out_path}")


def main(model_path, out_dir, multi_seed_dir=None, seeds=None):
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("WAVELET INTERPRETABILITY QUANTIFICATION")
    print("="*60)

    # Single-model analysis
    print(f"\n📊 Extracting from: {model_path}")
    mu, gamma, weights = extract_wavelet_params(model_path)
    center_freq, gamma_mean = compute_center_frequencies(gamma)
    mu_ms = compute_temporal_centers(mu)

    print(f"   μ shape: {mu.shape}")
    print(f"   γ shape: {gamma.shape}")
    print(f"   Center freq range: {center_freq.min():.1f} – {center_freq.max():.1f} Hz")
    print(f"   Mean center freq:  {center_freq.mean():.1f} ± {center_freq.std():.1f} Hz")

    # Plot 1: Center frequency histogram
    in_range, total = plot_center_frequencies(
        center_freq, os.path.join(out_dir, "wavelet_center_frequencies.png"))

    # Plot 2: Temporal attention heatmap
    plot_temporal_heatmap(mu, weights, os.path.join(out_dir, "wavelet_attention_map.png"))

    # Multi-seed stability analysis
    if multi_seed_dir and seeds:
        print(f"\n📊 Multi-seed stability analysis ({len(seeds)} seeds)...")
        freqs_multi, mus_multi = analyze_multi_seed(multi_seed_dir, seeds)
        if freqs_multi is not None:
            plot_seed_stability(freqs_multi, os.path.join(out_dir, "wavelet_seed_stability.png"))
            cross_seed_std = freqs_multi.std(axis=0).mean()
        else:
            cross_seed_std = None
    else:
        cross_seed_std = None

    # Save report
    report = {
        "model_path": model_path,
        "n_channels": int(mu.shape[0]),
        "n_input_dims": int(mu.shape[1]),
        "center_freq_mean_hz": float(center_freq.mean()),
        "center_freq_std_hz": float(center_freq.std()),
        "center_freq_min_hz": float(center_freq.min()),
        "center_freq_max_hz": float(center_freq.max()),
        "channels_in_qrs_range_5_40hz": int(in_range),
        "pct_in_qrs_range": float(100 * in_range / total),
        "scale_gamma_mean": float(gamma_mean.mean()),
        "scale_gamma_std": float(gamma_mean.std()),
        "temporal_center_mean_ms": float(np.mean(mu_ms)),
        "temporal_center_std_ms": float(np.std(mu_ms)),
    }
    if cross_seed_std is not None:
        report["cross_seed_freq_std_hz"] = float(cross_seed_std)

    with open(os.path.join(out_dir, "interpretability_report.json"), "w") as f:
        json.dump(report, f, indent=4)

    print(f"\n✅ Report saved to: {out_dir}/interpretability_report.json")
    print(f"   {in_range}/{total} channels ({100*in_range/total:.0f}%) in QRS range (5–40 Hz)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str,
                        default="results/hybrid_rr_history_20_seeds/seed_42/best_hybrid_rr.pth")
    parser.add_argument("--out-dir", type=str, default="results/interpretability")
    parser.add_argument("--multi-seed-dir", type=str,
                        default="results/hybrid_rr_history_20_seeds")
    parser.add_argument("--seeds", type=str,
                        default="42,101,777,2026,9999,1234,2024,31415,27182,7")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")] if args.seeds else None

    main(args.model_path, args.out_dir, args.multi_seed_dir, seeds)
