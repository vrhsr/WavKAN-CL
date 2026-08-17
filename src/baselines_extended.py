"""
baselines_extended.py  —  Extended Baseline Models

Implements all baselines required for a top-tier comparison table:
  1. ResNet1D         — 1D residual CNN (standard ECG baseline)
  2. ECGTransformer   — lightweight transformer for 1D ECG
  3. CNN_FocalLoss    — simple 1D CNN trained with focal loss
  4. CNN_RR_WeightedCE— CNN + RR MLP + weighted cross-entropy (closest to v1)
  5. BSplineKAN       — original B-spline KAN (internal ablation)

All models share the same training loop interface:
    train_baseline(model_name, seed, epochs, data_dir, output_dir)

Usage:
    python src/baselines_extended.py --model resnet1d --seed 42
    python src/baselines_extended.py --model transformer --seed 42
    python src/baselines_extended.py --all --seeds 42 101 777

Fixed 2026-08-17 (real 20-seed x 4-model Phase 5b run surfaced this): train_baseline()
previously paired a fully class-balanced WeightedRandomSampler with an
inverse-frequency-weighted Focal Loss -- a double correction that, given this
dataset's real N:Q class ratio (~6300:1), drove N's effective per-sample training
weight to near-zero while also capping N's per-batch sampling frequency down to
match the rarest class. Net effect: zero real gradient signal for N, so all 4
baselines collapsed to never predicting it (v_recall=0.0 for 3 of 4 models,
macro_f1 in the 0.02-0.10 range vs. WavKAN-v2's ~0.32 on the same test set). Now
uses natural-distribution sampling only (shuffle=True), matching train_pca.py's own
proven fair-baseline arm, which pairs the identical weight formula with natural
sampling and does not collapse. Also added the n_recall field to test_metrics.json
(previously omitted entirely, unlike every other reported class). See
AUDIT_FINDINGS.md and tests/test_baselines_extended.py.
"""

import os, sys, json, argparse, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    f1_score, recall_score, classification_report, confusion_matrix
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.wavkan_pcwi import PCWIWavKANLinear

# ─────────────────────────────────────────────────────────────────────────────
# Dataset (shared with train_pca.py)
# ─────────────────────────────────────────────────────────────────────────────

class ECGDatasetRR(Dataset):
    def __init__(self, split, data_dir):
        base = Path(data_dir)
        self.X   = torch.tensor(np.load(base / f"X_{split}.npy"),    dtype=torch.float32)
        self.Xrr = torch.tensor(np.load(base / f"X_rr_{split}.npy"), dtype=torch.float32)
        self.y   = torch.tensor(np.load(base / f"y_{split}.npy"),    dtype=torch.long)
    def __len__(self):              return len(self.y)
    def __getitem__(self, i):  return self.X[i], self.Xrr[i], self.y[i]


# ─────────────────────────────────────────────────────────────────────────────
# 1. ResNet1D  (He et al., 2016 style for 1D signals)
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv1  = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn1    = nn.BatchNorm1d(channels)
        self.conv2  = nn.Conv1d(channels, channels, kernel_size, padding=pad)
        self.bn2    = nn.BatchNorm1d(channels)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x):
        res = x
        x   = F.relu(self.bn1(self.conv1(x)))
        x   = self.drop(x)
        x   = self.bn2(self.conv2(x))
        return F.relu(x + res)


class ResNet1D(nn.Module):
    """
    1D-ResNet for ECG classification.
    Input: (B, 360) flat beat + (B, 5) RR history
    """
    def __init__(self, num_classes: int = 5, n_blocks: int = 4, channels: int = 32):
        super().__init__()
        # ECG signal branch
        self.stem = nn.Sequential(
            nn.Conv1d(1, channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )
        self.blocks = nn.Sequential(*[
            ResidualBlock1D(channels) for _ in range(n_blocks)
        ])
        self.pool = nn.AdaptiveAvgPool1d(1)

        # RR branch
        self.rr_mlp = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        # Classifier
        self.head = nn.Sequential(
            nn.Linear(channels + 16, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, x_rr):
        z = self.stem(x.unsqueeze(1))      # (B, C, 360)
        z = self.blocks(z)
        z = self.pool(z).squeeze(-1)       # (B, C)
        z_rr = self.rr_mlp(x_rr)          # (B, 16)
        return self.head(torch.cat([z, z_rr], dim=1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ECG Transformer  (lightweight, 1D positional encoding)
# ─────────────────────────────────────────────────────────────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 360):
        super().__init__()
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, T, d)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ECGTransformer(nn.Module):
    """
    Lightweight Transformer for 1D ECG beat classification.
    Each sample in the 360-sample beat is projected to d_model, then
    processed by a small Transformer encoder.
    """
    def __init__(
        self, num_classes: int = 5, d_model: int = 32,
        n_heads: int = 4, n_layers: int = 2, dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc    = PositionalEncoding(d_model)
        encoder_layer   = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pool        = nn.AdaptiveAvgPool1d(1)

        # RR branch
        self.rr_mlp = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(d_model + 16, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, x_rr):
        # x: (B, 360) → (B, 360, 1) → project → (B, 360, d_model)
        z = self.input_proj(x.unsqueeze(-1))
        z = self.pos_enc(z)
        z = self.transformer(z)              # (B, 360, d_model)
        z = self.pool(z.permute(0,2,1)).squeeze(-1)  # (B, d_model)
        z_rr = self.rr_mlp(x_rr)
        return self.head(torch.cat([z, z_rr], dim=1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 3. CNN + Focal Loss (simple 1D CNN — re-usable training baseline)
# ─────────────────────────────────────────────────────────────────────────────

class SimpleCNN1D(nn.Module):
    """Simple 4-layer CNN + RR MLP. Trained with focal loss."""
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 32, 15, padding=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.rr_mlp = nn.Sequential(
            nn.Linear(5, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + 16, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x, x_rr):
        z    = self.cnn(x.unsqueeze(1)).squeeze(-1)
        z_rr = self.rr_mlp(x_rr)
        return self.head(torch.cat([z, z_rr], dim=1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# 4. B-Spline KAN (internal ablation baseline from original paper)
# ─────────────────────────────────────────────────────────────────────────────

class BSplineKAN_RR(nn.Module):
    """WavKAN-CL with B-spline basis (the baseline that beat Mexican Hat in v1)."""
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.kan = PCWIWavKANLinear(360, 64, wavelet_type="b_spline", use_pcwi=False)
        self.ln  = nn.LayerNorm(64)
        self.gru = nn.GRU(64, 32, 1, batch_first=True, bidirectional=True)
        self.rr_mlp = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(80, 48), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(48, num_classes),
        )

    def forward(self, x, x_rr):
        z = self.ln(self.kan(x)).unsqueeze(1)
        z, _ = self.gru(z)
        z    = z.squeeze(1)
        z_rr = self.rr_mlp(x_rr)
        return self.head(torch.cat([z, z_rr], dim=1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss (same as train_pca.py)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma  = gamma
        self.weight = weight

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt  = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


# ─────────────────────────────────────────────────────────────────────────────
# Generic Training Loop
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "resnet1d":    ResNet1D,
    "transformer": ECGTransformer,
    "cnn_focal":   SimpleCNN1D,
    "bspline_kan": BSplineKAN_RR,
}


def run_eval(model, loader, device):
    model.eval()
    preds, trues, probs = [], [], []
    with torch.no_grad():
        for X, Xrr, y in loader:
            logits = model(X.to(device), Xrr.to(device))
            probs.append(torch.softmax(logits, 1).cpu())
            preds.extend(logits.argmax(1).cpu().numpy())
            trues.extend(y.numpy())
    return np.array(trues), np.array(preds), torch.cat(probs).numpy()


def train_baseline(
    model_name: str     = "resnet1d",
    seed: int           = 42,
    epochs: int         = 100,
    lr: float           = 1e-3,
    batch_size: int     = 64,
    patience: int       = 15,
    data_dir: str       = "data/processed_rr_history",
    output_dir: str     = None,
    focal_gamma: float  = 2.0,
    s_weight: float     = 8.0,
) -> dict:

    torch.manual_seed(seed)
    np.random.seed(seed)

    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(output_dir or f"results/baseline_{model_name}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training baseline: {model_name} | seed={seed} | {DEVICE}")
    print(f"{'='*60}")

    train_ds = ECGDatasetRR("train", data_dir)
    val_ds   = ECGDatasetRR("val",   data_dir)
    test_ds  = ECGDatasetRR("test",  data_dir)

    train_labels = train_ds.y.numpy()
    unique, counts = np.unique(train_labels, return_counts=True)

    # Class-weighted focal loss. Natural-distribution sampling only (shuffle=True,
    # no WeightedRandomSampler) -- found 2026-08-17 (real Phase 5b run) that pairing
    # a fully class-balanced sampler with this same inverse-frequency loss weighting
    # is a double-correction: with the real N:Q ratio (~6300:1), N's effective
    # per-sample weight collapses to ~0.003 while the balanced sampler simultaneously
    # caps N's per-batch frequency down to match the rarest class. Combined, that
    # leaves ~zero net gradient signal for N across the whole run -- confirmed via a
    # real 20-seed run where all 4 baselines never predicted class N (v_recall=0.0
    # for 3 of 4, macro_f1 in the 0.02-0.10 range vs WavKAN-v2's ~0.32). This mirrors
    # train_pca.py's own proven fair-baseline arm, which pairs natural sampling with
    # this exact weight formula and does NOT collapse (see criterion_baseline there) --
    # apply one imbalance-correction mechanism, not two stacked at full strength.
    raw_w    = (counts.sum()) / (len(unique) * counts.astype(float))
    raw_w[1] *= s_weight
    weights  = torch.tensor(raw_w / raw_w.sum() * len(unique), dtype=torch.float32).to(DEVICE)
    criterion = FocalLoss(gamma=focal_gamma, weight=weights)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False)

    model_cls = MODEL_REGISTRY[model_name]
    model     = model_cls().to(DEVICE)
    print(f"Parameters: {model.count_parameters():,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_f1, no_improve = 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for X, Xrr, y in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(X.to(DEVICE), Xrr.to(DEVICE)), y.to(DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        scheduler.step()

        y_true, y_pred, _ = run_eval(model, val_loader, DEVICE)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
        if f1 > best_f1:
            best_f1    = f1
            no_improve = 0
            torch.save(model.state_dict(), out_dir / "best_model.pth")
        else:
            no_improve += 1

        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d}  val_F1={f1:.4f}  best={best_f1:.4f}")
        if no_improve >= patience:
            print(f"Early stop at epoch {epoch}")
            break

    # Test
    model.load_state_dict(torch.load(out_dir / "best_model.pth", map_location=DEVICE))
    y_true, y_pred, y_probs = run_eval(model, test_loader, DEVICE)

    report = classification_report(y_true, y_pred, target_names=["N","S","V","F","Q"],
                                   digits=4, zero_division=0, output_dict=True)
    metrics = {
        "model":      model_name,
        "seed":       seed,
        "macro_f1":   float(report["macro avg"]["f1-score"]),
        "v_recall":   float(report["V"]["recall"]),
        "s_recall":   float(report["S"]["recall"]),
        "n_recall":   float(report["N"]["recall"]),
        "f_recall":   float(report["F"]["recall"]),
        "n_params":   model.count_parameters(),
    }
    print(f"\nTEST {model_name}: Macro-F1={metrics['macro_f1']:.4f}  "
          f"V={metrics['v_recall']:.4f}  S={metrics['s_recall']:.4f}")

    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    np.save(out_dir / "predictions.npy", y_pred)
    np.save(out_dir / "true.npy",        y_true)
    np.save(out_dir / "probs.npy",       y_probs)
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  type=str, default="resnet1d",
                        choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--all",    action="store_true", help="Run all baselines")
    parser.add_argument("--seeds",  type=int, nargs="+", default=[42])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--data-dir", type=str, default="data/processed_rr_history")
    args = parser.parse_args()

    models_to_run = list(MODEL_REGISTRY.keys()) if args.all else [args.model]
    for mname in models_to_run:
        all_metrics = []
        for seed in args.seeds:
            m = train_baseline(
                model_name = mname,
                seed       = seed,
                epochs     = args.epochs,
                data_dir   = args.data_dir,
                output_dir = f"results/baseline_{mname}/seed_{seed}",
            )
            all_metrics.append(m)

        if len(args.seeds) > 1:
            f1s = [m["macro_f1"] for m in all_metrics]
            vrs = [m["v_recall"]  for m in all_metrics]
            srs = [m["s_recall"]  for m in all_metrics]
            nrs = [m["n_recall"]  for m in all_metrics]
            print(f"\n{mname} ({len(args.seeds)} seeds):")
            print(f"  Macro-F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
            print(f"  V-Recall : {np.mean(vrs):.4f} ± {np.std(vrs):.4f}")
            print(f"  S-Recall : {np.mean(srs):.4f} ± {np.std(srs):.4f}")
            print(f"  N-Recall : {np.mean(nrs):.4f} ± {np.std(nrs):.4f}")
