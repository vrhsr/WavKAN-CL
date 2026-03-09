"""
Comprehensive Wavelet Ablation Study (D1–D7)
==============================================
Compares Mexican Hat vs Morlet vs DOG wavelets, and tests
architectural ablations (no GRU, no RR branch).

Usage:
    python src/run_ablation_study.py --seeds 42,101,777,2026,9999 --epochs 50
    python src/run_ablation_study.py --epochs 100 --output-dir results/ablation_comprehensive

Ablations:
    1. mexican_hat + GRU + RR  (full model — baseline)
    2. morlet + GRU + RR       (wavelet variant)
    3. dog + GRU + RR          (wavelet variant)
    4. mexican_hat + NO GRU + RR  (direct concat, no temporal modeling)
    5. mexican_hat + GRU + NO RR  (morphology-only, no rhythm features)
    6. mexican_hat + NO GRU + NO RR  (pure WavKAN classifier)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
from sklearn.metrics import f1_score, confusion_matrix, classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.wavkan import WavKANLinear


# ---------------------------------------------------------------------------
# Model Variants
# ---------------------------------------------------------------------------

class FullModel(nn.Module):
    """Full WavKAN-CL: WavKAN + GRU + RR branch."""
    def __init__(self, wavelet_type='mexican_hat'):
        super().__init__()
        self.name = f"WavKAN({wavelet_type})+GRU+RR"
        self.kan = WavKANLinear(360, 64, wavelet_type=wavelet_type)
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
        self.has_rr = True

    def forward(self, x, xr=None):
        x = self.kan(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = x.unsqueeze(1)
        x, _ = self.bigru(x)
        x = x.squeeze(1)
        xr = self.rr_mlp(xr)
        x = torch.cat((x, xr), dim=1)
        return self.fc2(torch.relu(self.fc1(x)))


class NoGRU(nn.Module):
    """WavKAN + RR, without BiGRU (direct concat)."""
    def __init__(self, wavelet_type='mexican_hat'):
        super().__init__()
        self.name = f"WavKAN({wavelet_type})+RR (no GRU)"
        self.kan = WavKANLinear(360, 64, wavelet_type=wavelet_type)
        self.ln = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.2)
        self.rr_mlp = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.fc1 = nn.Linear(80, 48)
        self.fc2 = nn.Linear(48, 5)
        self.has_rr = True

    def forward(self, x, xr=None):
        x = self.kan(x)
        x = self.ln(x)
        x = self.dropout(x)
        xr = self.rr_mlp(xr)
        x = torch.cat((x, xr), dim=1)
        return self.fc2(torch.relu(self.fc1(x)))


class NoRR(nn.Module):
    """WavKAN + GRU, no RR branch (morphology-only)."""
    def __init__(self, wavelet_type='mexican_hat'):
        super().__init__()
        self.name = f"WavKAN({wavelet_type})+GRU (no RR)"
        self.kan = WavKANLinear(360, 64, wavelet_type=wavelet_type)
        self.ln = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.2)
        self.bigru = nn.GRU(64, 32, 1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(64, 48)
        self.fc2 = nn.Linear(48, 5)
        self.has_rr = False

    def forward(self, x, xr=None):
        x = self.kan(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = x.unsqueeze(1)
        x, _ = self.bigru(x)
        x = x.squeeze(1)
        return self.fc2(torch.relu(self.fc1(x)))


class PureWavKAN(nn.Module):
    """Pure WavKAN classifier: no GRU, no RR (simplest variant)."""
    def __init__(self, wavelet_type='mexican_hat'):
        super().__init__()
        self.name = f"WavKAN({wavelet_type}) pure (no GRU, no RR)"
        self.kan1 = WavKANLinear(360, 64, wavelet_type=wavelet_type)
        self.ln1 = nn.LayerNorm(64)
        self.kan2 = WavKANLinear(64, 32, wavelet_type=wavelet_type)
        self.ln2 = nn.LayerNorm(32)
        self.head = nn.Linear(32, 5)
        self.has_rr = False

    def forward(self, x, xr=None):
        x = self.kan1(x)
        x = self.ln1(x)
        x = self.kan2(x)
        x = self.ln2(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ECGDatasetRR(Dataset):
    def __init__(self, split):
        self.X = torch.tensor(np.load(f"data/processed_rr_history/X_{split}.npy"), dtype=torch.float32)
        self.Xr = torch.tensor(np.load(f"data/processed_rr_history/X_rr_{split}.npy"), dtype=torch.float32)
        self.y = torch.tensor(np.load(f"data/processed_rr_history/y_{split}.npy"), dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.Xr[i], self.y[i]


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Training one configuration
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

def train_one_config(model, seed, epochs, device, use_focal_loss=False):
    """Train a single model configuration with curriculum learning."""
    set_seed(seed)
    
    train_ds = ECGDatasetRR("train")
    val_ds = ECGDatasetRR("val")
    test_ds = ECGDatasetRR("test")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Phase 1: Minority-only
    train_labels = train_ds.y.numpy()
    minority_idx = np.where(train_labels > 0)[0]
    minority_subset = Subset(train_ds, minority_idx)
    
    phase1_epochs = max(5, min(15, int(epochs * 0.3)))
    loader_p1 = DataLoader(minority_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    for epoch in range(phase1_epochs):
        model.train()
        for X, Xr, y in loader_p1:
            X, Xr, y = X.to(device), Xr.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X, Xr) if model.has_rr else model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
    
    # Phase 2: Full dataset
    weights = np.load("data/processed_rr_history/class_weights.npy")
    weights[1] *= 10.0
    weight_tensor = torch.tensor(weights, dtype=torch.float32).to(device)
    
    if use_focal_loss:
        criterion_p2 = FocalLoss(weight=weight_tensor, gamma=2.0)
    else:
        criterion_p2 = nn.CrossEntropyLoss(weight=weight_tensor)
        
    loader_p2 = DataLoader(train_ds, batch_size=64, shuffle=True)
    
    best_f1 = 0.0
    best_state = None
    
    for epoch in range(epochs - phase1_epochs):
        model.train()
        for X, Xr, y in loader_p2:
            X, Xr, y = X.to(device), Xr.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X, Xr) if model.has_rr else model(X)
            loss = criterion_p2(out, y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X, Xr, y in val_loader:
                out = model(X.to(device), Xr.to(device)) if model.has_rr else model(X.to(device))
                preds.extend(torch.argmax(out, 1).cpu().numpy())
                labels.extend(y.numpy())
        f1 = f1_score(labels, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    # Test with best model
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    
    preds, labels = [], []
    with torch.no_grad():
        for X, Xr, y in test_loader:
            out = model(X.to(device), Xr.to(device)) if model.has_rr else model(X.to(device))
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            labels.extend(y.numpy())
    
    preds = np.array(preds)
    labels = np.array(labels)
    cm = confusion_matrix(labels, preds)
    class_recalls = cm.diagonal() / cm.sum(axis=1)
    
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'f1_macro': float(f1_score(labels, preds, average='macro')),
        'n_recall': float(class_recalls[0]),
        's_recall': float(class_recalls[1]),
        'v_recall': float(class_recalls[2]),
        'f_recall': float(class_recalls[3]) if len(class_recalls) > 3 else 0.0,
        'val_best_f1': float(best_f1),
        'n_params': n_params,
    }


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------
def run_ablation(seeds_str, epochs, output_dir):
    seeds = [int(s) for s in seeds_str.split(",")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT = Path(output_dir)
    OUT.mkdir(parents=True, exist_ok=True)
    
    # Define all configurations
    configs = [
        ("Full (Mexican Hat)", lambda: FullModel('mexican_hat'), False),
        ("Morlet variant",     lambda: FullModel('morlet'),      False),
        ("DOG variant",        lambda: FullModel('dog'),         False),
        ("B-spline variant",   lambda: FullModel('b_spline'),    False),
        ("Focal Loss (gamma=2)",lambda: FullModel('mexican_hat'),True),
        ("No GRU",             lambda: NoGRU('mexican_hat'),     False),
        ("No RR (morph-only)", lambda: NoRR('mexican_hat'),      False),
        ("Pure WavKAN",        lambda: PureWavKAN('mexican_hat'),False),
    ]
    
    all_results = {}
    
    for config_name, model_fn, use_fl in configs:
        print(f"\n{'='*60}")
        print(f"  Config: {config_name}")
        print(f"{'='*60}")
        
        seed_results = []
        for seed in seeds:
            print(f"  Seed {seed}...", end=" ", flush=True)
            model = model_fn().to(device)
            result = train_one_config(model, seed, epochs, device, use_focal_loss=use_fl)
            result['seed'] = seed
            seed_results.append(result)
            print(f"F1={result['f1_macro']:.3f} V={result['v_recall']:.3f} S={result['s_recall']:.3f}")
        
        # Aggregate
        f1s = [r['f1_macro'] for r in seed_results]
        v_recs = [r['v_recall'] for r in seed_results]
        s_recs = [r['s_recall'] for r in seed_results]
        
        agg = {
            'config': config_name,
            'n_params': seed_results[0]['n_params'],
            'f1_macro_mean': float(np.mean(f1s)),
            'f1_macro_std': float(np.std(f1s)),
            'v_recall_mean': float(np.mean(v_recs)),
            'v_recall_std': float(np.std(v_recs)),
            's_recall_mean': float(np.mean(s_recs)),
            's_recall_std': float(np.std(s_recs)),
            'seeds': seed_results,
        }
        all_results[config_name] = agg
        
        print(f"\n  📊 {config_name}: F1={agg['f1_macro_mean']:.3f}±{agg['f1_macro_std']:.3f}  "
              f"V={agg['v_recall_mean']:.3f}±{agg['v_recall_std']:.3f}  "
              f"S={agg['s_recall_mean']:.3f}±{agg['s_recall_std']:.3f}  "
              f"Params={agg['n_params']:,}")
    
    # Save results
    with open(OUT / "ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=4)
    
    # Print summary table
    print(f"\n{'='*90}")
    print(f"ABLATION SUMMARY TABLE")
    print(f"{'='*90}")
    print(f"{'Config':<25} {'Params':>8} {'Macro-F1':>12} {'V-Recall':>12} {'S-Recall':>12}")
    print(f"{'-'*25} {'-'*8} {'-'*12} {'-'*12} {'-'*12}")
    for name, agg in all_results.items():
        print(f"{name:<25} {agg['n_params']:>8,} "
              f"{agg['f1_macro_mean']:>5.3f}±{agg['f1_macro_std']:.3f} "
              f"{agg['v_recall_mean']:>5.3f}±{agg['v_recall_std']:.3f} "
              f"{agg['s_recall_mean']:>5.3f}±{agg['s_recall_std']:.3f}")
    print(f"{'='*90}")
    print(f"\n✅ Full results saved to: {OUT / 'ablation_results.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seeds', type=str, default='42,101,777,2026,9999')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='results/ablation_comprehensive')
    args = parser.parse_args()
    
    run_ablation(args.seeds, args.epochs, args.output_dir)
