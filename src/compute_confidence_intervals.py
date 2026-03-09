"""
95% Confidence Intervals & Statistical Summary (E1, E4)
========================================================
Loads all 20 seed checkpoints from results/hybrid_rr_history_20_seeds/,
evaluates each on the DS2 test set, and computes 95% CIs.

Usage:
    python src/compute_confidence_intervals.py
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from scipy import stats
from sklearn.metrics import f1_score, confusion_matrix
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.wavkan import WavKANLinear


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


class ECGDatasetRR(Dataset):
    def __init__(self, split):
        self.X = torch.tensor(np.load(f"data/processed_rr_history/X_{split}.npy"), dtype=torch.float32)
        self.Xr = torch.tensor(np.load(f"data/processed_rr_history/X_rr_{split}.npy"), dtype=torch.float32)
        self.y = torch.tensor(np.load(f"data/processed_rr_history/y_{split}.npy"), dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.Xr[i], self.y[i]


def evaluate_checkpoint(model_path, test_loader, device):
    """Evaluate a single checkpoint on the test set."""
    model = HybridWavKAN_RR().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, Xr, y in test_loader:
            out = model(X.to(device), Xr.to(device))
            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_labels.extend(y.numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    cm = confusion_matrix(labels, preds, labels=list(range(5)))
    class_recalls = []
    for i in range(5):
        if cm[i].sum() > 0:
            class_recalls.append(cm[i, i] / cm[i].sum())
        else:
            class_recalls.append(0.0)

    return {
        'f1_macro': float(f1_score(labels, preds, average='macro')),
        'n_recall': float(class_recalls[0]),
        's_recall': float(class_recalls[1]),
        'v_recall': float(class_recalls[2]),
        'f_recall': float(class_recalls[3]),
    }


def compute_t_ci(values, confidence=0.95):
    n = len(values)
    mean = np.mean(values)
    se = stats.sem(values)
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, float(np.std(values)), float(se), float(h), (float(mean - h), float(mean + h))


def main(results_dir, out_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find all seed directories
    seed_dirs = sorted([
        d for d in Path(results_dir).iterdir()
        if d.is_dir() and d.name.startswith("seed_")
    ])

    if len(seed_dirs) < 2:
        print(f"❌ Need at least 2 seed dirs, found {len(seed_dirs)} in {results_dir}")
        return

    # Load test data once
    test_ds = ECGDatasetRR("test")
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)

    print(f"{'='*65}")
    print(f"  Evaluating {len(seed_dirs)} seeds on DS2 test set...")
    print(f"{'='*65}")

    all_results = []
    for sd in seed_dirs:
        ckpt_path = sd / "best_hybrid_rr.pth"
        if not ckpt_path.exists():
            continue
        seed_name = sd.name
        metrics = evaluate_checkpoint(str(ckpt_path), test_loader, device)
        metrics['seed'] = seed_name
        all_results.append(metrics)
        print(f"  {seed_name}: F1={metrics['f1_macro']:.4f}  V={metrics['v_recall']:.4f}  S={metrics['s_recall']:.4f}")

    n = len(all_results)
    if n < 2:
        print(f"❌ Only {n} valid checkpoints found")
        return

    # Compute CIs
    report = {"n_seeds": n, "seeds": [r['seed'] for r in all_results], "metrics": {}}

    print(f"\n{'='*65}")
    print(f"  95% CONFIDENCE INTERVALS ({n} seeds)")
    print(f"{'='*65}")

    for metric in ['f1_macro', 'v_recall', 's_recall', 'n_recall', 'f_recall']:
        values = [r[metric] for r in all_results]
        mean, std, se, h, (ci_lo, ci_hi) = compute_t_ci(values)
        report["metrics"][metric] = {
            "mean": mean, "std": std, "se": se,
            "ci_95": [ci_lo, ci_hi],
            "min": float(min(values)), "max": float(max(values)),
        }
        label = metric.replace('_', '-').upper()
        print(f"  {label:>10}: {mean:.4f} ± {std:.4f}  CI=[{ci_lo:.4f}, {ci_hi:.4f}]")

    os.makedirs(os.path.dirname(out_path) if os.path.dirname(out_path) else ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n✅ Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=str,
                        default="results/hybrid_rr_history_20_seeds")
    parser.add_argument("--out", type=str,
                        default="results/confidence_intervals.json")
    args = parser.parse_args()
    main(args.results_dir, args.out)
