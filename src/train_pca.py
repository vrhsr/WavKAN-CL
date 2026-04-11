"""
train_pca.py  —  Progressive Curriculum Anchoring (PCA)

FIXES THE BROKEN MINORITY-FIRST CURRICULUM
==========================================
The original minority-first strategy (Phase 1 = S,V,F,Q only, zero N beats)
destroys the N↔S decision boundary in Phase 1, which never fully recovers
in Phase 2. The paper's own Table 4 confirms this: "Baseline CE (No Curriculum)"
outperforms WavKAN-CL on both Macro-F1 (0.371 > 0.362) and S-Recall (0.285 > 0.260).

PCA Strategy (replaces "minority-first")
-----------------------------------------
Instead of hard phase switches, PCA uses a smooth temperature schedule to
gradually shift the sampling distribution from balanced → natural:

  Warm-up (epochs 1–E_w):
    Class-balanced sampler  [all classes equally represented]
    Loss: Standard CE  (no weighting — prevents early majority collapse)

  Annealing (epochs E_w+1 → E):
    Weighted sampler: p_c ∝ n_c^(–schedule(e))  where schedule(e) → 0 over time
    Loss: Focal Loss (γ=2) + S-class emphasis weight
    The temperature schedule gradually transitions:
      e=E_w   → fully balanced (exponent=1.0)
      e=E     → natural distribution (exponent→0.0)

  This NEVER excludes the Normal class, anchoring the decision boundary
  throughout training (hence "Progressive Curriculum Anchoring").

Arguments
---------
  --seed      : random seed
  --epochs    : total epochs (default 100)
  --warmup    : warm-up fraction (default 0.25 → 25 epochs of 100)
  --focal-gamma: Focal Loss γ parameter (default 2.0)
  --s-weight  : additional S-class weight multiplier (default 8.0)
  --output-dir: where to save checkpoints and metrics
  --data-dir  : path to processed_rr_history/ data directory
  --use-pcwi  : enable PCWI initialisation (default True)
  --use-pwam  : enable P-Wave Attention (default True)
  --augment   : enable minority augmentation (default True)
"""

import os, sys, json, argparse, math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import (
    f1_score, recall_score, precision_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from models.wavkan_v2 import WavKAN_v2

# ─────────────────────────────────────────────────────────────────────────────
# Focal Loss (better than plain CE for imbalanced inter-patient data)
# ─────────────────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal Loss: FL(p_t) = –α_t (1–p_t)^γ log(p_t)
    Down-weights easy N-class examples; focuses learning on hard minority beats.
    Reference: Lin et al. (2017), Bae & Shin (2025)
    """
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor = None, reduction: str = "mean"):
        super().__init__()
        self.gamma     = gamma
        self.weight    = weight
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = nn.functional.cross_entropy(logits, targets, weight=self.weight,
                                            reduction="none")
        p_t  = torch.exp(-ce)
        loss = (1.0 - p_t) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        return loss.sum()


# ─────────────────────────────────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────────────────────────────────

class ECGDatasetRR(Dataset):
    def __init__(self, split: str, data_dir: str):
        base = Path(data_dir)
        self.X   = torch.tensor(np.load(base / f"X_{split}.npy"),    dtype=torch.float32)
        self.Xrr = torch.tensor(np.load(base / f"X_rr_{split}.npy"), dtype=torch.float32)
        self.y   = torch.tensor(np.load(base / f"y_{split}.npy"),    dtype=torch.long)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.Xrr[i], self.y[i]


# ─────────────────────────────────────────────────────────────────────────────
# Minority Augmentation (inline, no external dependency)
# ─────────────────────────────────────────────────────────────────────────────

def augment_minority_batch(X: torch.Tensor, y: torch.Tensor, snr_db: float = 25.0):
    """
    Apply signal-preserving augmentation to minority-class beats in a batch.
    Augmentations: Gaussian noise + baseline wander + amplitude scaling.
    Normal beats are untouched.
    """
    X_aug = X.clone()
    for i in range(len(y)):
        if y[i].item() == 0:     # Skip Normal class
            continue
        sig = X_aug[i]
        # Gaussian noise (SNR ~25 dB)
        sig_power = sig.pow(2).mean()
        noise_power = sig_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(sig) * noise_power.sqrt()
        sig = sig + noise
        # Baseline wander: low-freq sinusoid
        t      = torch.linspace(0, 2 * math.pi, sig.size(0))
        freq   = torch.FloatTensor(1).uniform_(0.5, 2.5).item()  # 0.5–2.5 Hz
        amp    = torch.FloatTensor(1).uniform_(0.01, 0.05).item()
        wander = amp * torch.sin(freq * t)
        sig = sig + wander
        # Amplitude scaling ±10 %
        scale = torch.FloatTensor(1).uniform_(0.90, 1.10).item()
        sig = sig * scale
        X_aug[i] = sig
    return X_aug


# ─────────────────────────────────────────────────────────────────────────────
# Balanced Weighted Sampler (for warm-up phase)
# ─────────────────────────────────────────────────────────────────────────────

def make_balanced_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Creates a WeightedRandomSampler that equalises class frequency."""
    unique, counts = np.unique(labels, return_counts=True)
    class_weight   = 1.0 / counts
    sample_weights = np.array([class_weight[np.where(unique == l)[0][0]] for l in labels])
    return WeightedRandomSampler(
        weights     = torch.DoubleTensor(sample_weights),
        num_samples = len(sample_weights),
        replacement = True,
    )


def make_annealed_sampler(labels: np.ndarray, exponent: float) -> WeightedRandomSampler:
    """
    Annealed sampler: p_c ∝ n_c^(–exponent).
    exponent=1.0 → fully balanced
    exponent=0.0 → natural class distribution
    """
    unique, counts = np.unique(labels, return_counts=True)
    freq        = counts / counts.sum()
    class_prob  = freq ** (1.0 - exponent)             # n_c^(–exponent) = (1/n_c)^exponent × n_c
    # Use inverse-frequency weighting with exponent control
    class_weight_raw = 1.0 / (counts ** exponent)
    class_weight_raw = class_weight_raw / class_weight_raw.sum()
    sample_weights = np.array([
        class_weight_raw[np.where(unique == l)[0][0]] for l in labels
    ])
    return WeightedRandomSampler(
        weights     = torch.DoubleTensor(sample_weights),
        num_samples = len(sample_weights),
        replacement = True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device):
    model.eval()
    preds, trues, probs_list = [], [], []
    with torch.no_grad():
        for X, Xrr, y in loader:
            logits = model(X.to(device), Xrr.to(device))
            prob   = torch.softmax(logits, dim=1).cpu()
            pred   = logits.argmax(dim=1).cpu()
            preds.extend(pred.numpy())
            trues.extend(y.numpy())
            probs_list.append(prob)
    probs = torch.cat(probs_list).numpy()
    return np.array(trues), np.array(preds), probs


def per_class_metrics(y_true, y_pred, class_names=("N","S","V","F","Q")):
    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)))
    recalls    = cm.diagonal() / (cm.sum(axis=1) + 1e-8)
    precisions = cm.diagonal() / (cm.sum(axis=0) + 1e-8)
    f1s = 2 * recalls * precisions / (recalls + precisions + 1e-8)
    return {f"{c}_recall": recalls[i] for i, c in enumerate(class_names)} | \
           {f"{c}_f1":     f1s[i]     for i, c in enumerate(class_names)}


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Function
# ─────────────────────────────────────────────────────────────────────────────

def train_pca(
    seed: int       = 42,
    epochs: int     = 100,
    warmup: float   = 0.25,
    focal_gamma: float = 2.0,
    s_weight: float = 8.0,
    lr: float       = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 64,
    data_dir: str   = "data/processed_rr_history",
    output_dir: str = "results/pca_model",
    use_pcwi: bool  = True,
    use_pwam: bool  = True,
    use_rr_attn: bool = True,
    use_augment: bool = True,
    wavelet_type: str = "mexican_hat",
    patience: int   = 15,
) -> dict:

    # ── Reproducibility ───────────────────────────────────────────────────────
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR = Path(output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Progressive Curriculum Anchoring | seed={seed} | {DEVICE}")
    print(f"  PCWI={use_pcwi}  PWAM={use_pwam}  RR-Attn={use_rr_attn}")
    print(f"  Wavelet={wavelet_type}  Epochs={epochs}  Warmup={warmup:.0%}")
    print(f"{'='*60}\n")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = ECGDatasetRR("train", data_dir)
    val_ds   = ECGDatasetRR("val",   data_dir)
    test_ds  = ECGDatasetRR("test",  data_dir)

    train_labels = train_ds.y.numpy()
    unique, counts = np.unique(train_labels, return_counts=True)
    print(f"Class distribution: { {int(k): int(v) for k,v in zip(unique, counts)} }")

    # ── Loss functions ────────────────────────────────────────────────────────
    # Class weights for Focal Loss (inverse-frequency)
    total = counts.sum()
    n_cls = len(unique)
    raw_w = total / (n_cls * counts)
    raw_w[1] *= s_weight          # Extra S-class emphasis
    class_weights = torch.tensor(raw_w / raw_w.sum() * n_cls, dtype=torch.float32).to(DEVICE)

    # Warm-up: plain CE (no weighting) — anchor all class boundaries
    criterion_warmup = nn.CrossEntropyLoss()
    # Annealing: Focal Loss with class weights — focus on hard minority beats
    criterion_focal  = FocalLoss(gamma=focal_gamma, weight=class_weights)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = WavKAN_v2(
        wavelet_type = wavelet_type,
        use_pcwi     = use_pcwi,
        use_pwam     = use_pwam,
        use_rr_attn  = use_rr_attn,
    ).to(DEVICE)
    print(f"Model parameters: {model.count_parameters():,}")

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    val_loader  = DataLoader(val_ds,  batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # ── Training loop ─────────────────────────────────────────────────────────
    warmup_epochs   = max(5, int(epochs * warmup))
    best_val_f1     = 0.0
    best_val_vrecall = 0.0
    no_improve      = 0
    history         = []

    for epoch in range(1, epochs + 1):
        model.train()

        # ── Select sampler & loss for this epoch ──────────────────────────
        if epoch <= warmup_epochs:
            # Warm-up: balanced sampling, plain CE
            sampler  = make_balanced_sampler(train_labels)
            loader   = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
            criterion = criterion_warmup
            phase_tag = "WARMUP"
        else:
            # Annealing: smoothly decay from balanced → natural
            t          = (epoch - warmup_epochs) / (epochs - warmup_epochs)
            exponent   = 1.0 - t           # 1.0 → 0.0
            sampler    = make_annealed_sampler(train_labels, exponent)
            loader     = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
            criterion  = criterion_focal
            phase_tag  = f"ANNEAL(e={exponent:.2f})"

        # ── Epoch training ─────────────────────────────────────────────────
        epoch_loss = 0.0
        n_batches  = 0
        for X, Xrr, y in loader:
            X, Xrr, y = X.to(DEVICE), Xrr.to(DEVICE), y.to(DEVICE)

            if use_augment:
                X = augment_minority_batch(X, y)

            optimizer.zero_grad()
            logits = model(X, Xrr)
            loss   = criterion(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # ── Validation ────────────────────────────────────────────────────
        y_true, y_pred, _ = evaluate(model, val_loader, DEVICE)
        val_f1       = f1_score(y_true, y_pred, average="macro", zero_division=0)
        val_v_recall = recall_score(y_true, y_pred, labels=[2], average="macro", zero_division=0)
        val_s_recall = recall_score(y_true, y_pred, labels=[1], average="macro", zero_division=0)

        history.append({
            "epoch": epoch, "phase": phase_tag,
            "loss":  epoch_loss / n_batches,
            "val_macro_f1": val_f1,
            "val_v_recall": val_v_recall,
            "val_s_recall": val_s_recall,
        })

        # Primary criterion: V-recall (safety-critical); secondary: Macro-F1
        improved = val_f1 > best_val_f1
        if improved:
            best_val_f1     = val_f1
            best_val_vrecall = val_v_recall
            no_improve       = 0
            torch.save(model.state_dict(), OUT_DIR / "best_model.pth")
        else:
            no_improve += 1

        if (epoch % 10 == 0) or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} [{phase_tag:20s}] "
                  f"loss={epoch_loss/n_batches:.4f}  "
                  f"val_F1={val_f1:.4f}  V-rec={val_v_recall:.3f}  "
                  f"S-rec={val_s_recall:.3f}"
                  + (" ✓" if improved else ""))

        if no_improve >= patience:
            print(f"\n⏹  Early stop at epoch {epoch} (patience={patience})")
            break

    # ── Test Evaluation ───────────────────────────────────────────────────────
    model.load_state_dict(torch.load(OUT_DIR / "best_model.pth", map_location=DEVICE))
    y_true, y_pred, y_probs = evaluate(model, test_loader, DEVICE)

    report = classification_report(
        y_true, y_pred,
        target_names=["N", "S", "V", "F", "Q"],
        digits=4, zero_division=0, output_dict=True,
    )
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "seed":         seed,
        "epochs_run":   epoch,
        "macro_f1":     float(report["macro avg"]["f1-score"]),
        "v_recall":     float(report["V"]["recall"]),
        "s_recall":     float(report["S"]["recall"]),
        "f_recall":     float(report["F"]["recall"]),
        "n_recall":     float(report["N"]["recall"]),
        "model_params": model.count_parameters(),
        "config": {
            "use_pcwi":    use_pcwi,
            "use_pwam":    use_pwam,
            "use_rr_attn": use_rr_attn,
            "wavelet_type": wavelet_type,
        },
    }

    print(f"\n{'='*60}")
    print(f"TEST RESULTS (best val checkpoint)")
    print(f"  Macro-F1 : {metrics['macro_f1']:.4f}")
    print(f"  V-Recall : {metrics['v_recall']:.4f}  ← primary safety metric")
    print(f"  S-Recall : {metrics['s_recall']:.4f}")
    print(f"  F-Recall : {metrics['f_recall']:.4f}")
    print(f"{'='*60}")
    print(classification_report(y_true, y_pred, target_names=["N","S","V","F","Q"],
                                digits=4, zero_division=0))

    # Save outputs
    with open(OUT_DIR / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    with open(OUT_DIR / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    np.save(OUT_DIR / "test_predictions.npy", y_pred)
    np.save(OUT_DIR / "test_true.npy",        y_true)
    np.save(OUT_DIR / "test_probs.npy",       y_probs)
    np.save(OUT_DIR / "confusion_matrix.npy", cm)

    print(f"\n✅ Outputs saved to {OUT_DIR}/")
    return metrics


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Progressive Curriculum Anchoring training")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--epochs",      type=int,   default=100)
    parser.add_argument("--warmup",      type=float, default=0.25,
                        help="Fraction of epochs for balanced warm-up phase")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--s-weight",    type=float, default=8.0,
                        help="Extra S-class emphasis multiplier")
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--batch-size",  type=int,   default=64)
    parser.add_argument("--patience",    type=int,   default=15)
    parser.add_argument("--data-dir",    type=str,   default="data/processed_rr_history")
    parser.add_argument("--output-dir",  type=str,   default="results/pca_model")
    parser.add_argument("--wavelet",     type=str,   default="mexican_hat",
                        choices=["mexican_hat", "morlet", "dog", "b_spline"])
    parser.add_argument("--no-pcwi",     action="store_true")
    parser.add_argument("--no-pwam",     action="store_true")
    parser.add_argument("--no-rr-attn",  action="store_true")
    parser.add_argument("--no-augment",  action="store_true")
    args = parser.parse_args()

    train_pca(
        seed         = args.seed,
        epochs       = args.epochs,
        warmup       = args.warmup,
        focal_gamma  = args.focal_gamma,
        s_weight     = args.s_weight,
        lr           = args.lr,
        batch_size   = args.batch_size,
        patience     = args.patience,
        data_dir     = args.data_dir,
        output_dir   = args.output_dir,
        wavelet_type = args.wavelet,
        use_pcwi     = not args.no_pcwi,
        use_pwam     = not args.no_pwam,
        use_rr_attn  = not args.no_rr_attn,
        use_augment  = not args.no_augment,
    )
