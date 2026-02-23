"""
Curriculum Learning with SMOTE Augmentation
=============================================
Adds SMOTE oversampling for minority classes (S, V, F) in the training set
BEFORE curriculum training. SMOTE is applied in feature space (raw beats).

Usage:
    python src/train_curriculum_smote.py --seed 42 --epochs 50
    python src/train_curriculum_smote.py --seed 42 --epochs 100 --output-dir results/curriculum_smote

Target: Raise S-Recall from 0.28 → 0.40+
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import numpy as np
import os
import sys
import json
import argparse
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix, classification_report

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.wavkan import WavKANLinear


# ---------------------------------------------------------------------------
# Model (same as train_curriculum.py)
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


# ---------------------------------------------------------------------------
# SMOTE in ECG feature space
# ---------------------------------------------------------------------------
def smote_oversample(X, X_rr, y, target_ratio=0.5, k_neighbors=5, random_state=42):
    """
    Apply SMOTE-like oversampling to minority classes.
    
    For each minority class, generates synthetic beats by interpolating
    between existing beats and their k nearest neighbors in feature space.
    
    Args:
        X:  (N, 360) numpy array of beat waveforms
        X_rr: (N, 5) numpy array of RR-interval features
        y:  (N,) numpy array of labels
        target_ratio: target minority/majority ratio (0.5 = half as many)
        k_neighbors: number of nearest neighbors for interpolation
        random_state: random seed
    
    Returns:
        X_aug, X_rr_aug, y_aug: augmented arrays
    """
    rng = np.random.RandomState(random_state)
    
    unique, counts = np.unique(y, return_counts=True)
    max_count = counts.max()
    target_count = int(max_count * target_ratio)
    
    X_parts = [X]
    Xr_parts = [X_rr]
    y_parts = [y]
    
    print(f"\n📊 SMOTE Augmentation (target_ratio={target_ratio}):")
    print(f"   Original distribution: {dict(zip(unique, counts))}")
    
    for cls, cnt in zip(unique, counts):
        if cnt >= target_count:
            continue  # skip majority class
        
        n_synthetic = target_count - cnt
        cls_mask = (y == cls)
        X_cls = X[cls_mask]
        Xr_cls = X_rr[cls_mask]
        
        if len(X_cls) < 2:
            continue
        
        k = min(k_neighbors, len(X_cls) - 1)
        
        # Compute pairwise distances (L2 on raw beats)
        from sklearn.neighbors import NearestNeighbors
        nn_model = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
        nn_model.fit(X_cls)
        distances, indices = nn_model.kneighbors(X_cls)
        
        synthetic_X = []
        synthetic_Xr = []
        
        for _ in range(n_synthetic):
            # Pick a random minority sample
            idx = rng.randint(0, len(X_cls))
            # Pick a random neighbor (skip self at index 0)
            nn_idx = indices[idx, rng.randint(1, k + 1)]
            
            # Interpolation factor
            lam = rng.uniform(0.0, 1.0)
            
            # Generate synthetic beat (linear interpolation)
            new_beat = X_cls[idx] + lam * (X_cls[nn_idx] - X_cls[idx])
            new_rr = Xr_cls[idx] + lam * (Xr_cls[nn_idx] - Xr_cls[idx])
            
            synthetic_X.append(new_beat)
            synthetic_Xr.append(new_rr)
        
        if synthetic_X:
            synth_X = np.array(synthetic_X, dtype=np.float32)
            synth_Xr = np.array(synthetic_Xr, dtype=np.float32)
            synth_y = np.full(len(synthetic_X), cls, dtype=np.int64)
            
            X_parts.append(synth_X)
            Xr_parts.append(synth_Xr)
            y_parts.append(synth_y)
            
            class_names = ['N', 'S', 'V', 'F', 'Q']
            print(f"   Class {class_names[cls]}: {cnt} → {cnt + n_synthetic} (+{n_synthetic} synthetic)")
    
    X_aug = np.concatenate(X_parts, axis=0)
    Xr_aug = np.concatenate(Xr_parts, axis=0)
    y_aug = np.concatenate(y_parts, axis=0)
    
    # Shuffle
    perm = rng.permutation(len(y_aug))
    return X_aug[perm], Xr_aug[perm], y_aug[perm]


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
# Training
# ---------------------------------------------------------------------------
def train_curriculum_smote(seed=42, epochs=50, output_dir="results/curriculum_smote",
                           smote_ratio=0.5, smote_k=5):
    set_seed(seed)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR = Path(output_dir) / f"seed_{seed}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load raw numpy arrays for SMOTE
    X_train = np.load("data/processed_rr_history/X_train.npy")
    Xr_train = np.load("data/processed_rr_history/X_rr_train.npy")
    y_train = np.load("data/processed_rr_history/y_train.npy")
    
    # Apply SMOTE to training set ONLY
    X_aug, Xr_aug, y_aug = smote_oversample(
        X_train, Xr_train, y_train,
        target_ratio=smote_ratio, k_neighbors=smote_k, random_state=seed
    )
    
    # Create augmented dataset
    train_ds = TensorDataset(
        torch.tensor(X_aug, dtype=torch.float32),
        torch.tensor(Xr_aug, dtype=torch.float32),
        torch.tensor(y_aug, dtype=torch.long)
    )
    val_ds = ECGDatasetRR("val")
    test_ds = ECGDatasetRR("test")
    
    # Phase 1: Minority only (from augmented set)
    minority_idx = np.where(y_aug > 0)[0]
    minority_ds = TensorDataset(
        torch.tensor(X_aug[minority_idx], dtype=torch.float32),
        torch.tensor(Xr_aug[minority_idx], dtype=torch.float32),
        torch.tensor(y_aug[minority_idx], dtype=torch.long)
    )
    
    model = HybridWavKAN_RR().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    phase1_epochs = max(5, min(15, int(epochs * 0.3)))
    loader_p1 = DataLoader(minority_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    print(f"\n{'='*60}")
    print(f"PHASE 1: SMOTE-Augmented Minority Training ({phase1_epochs} epochs)")
    print(f"{'='*60}")
    
    for epoch in range(phase1_epochs):
        model.train()
        for X, Xr, y in loader_p1:
            X, Xr, y = X.to(DEVICE), Xr.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X, Xr), y)
            loss.backward()
            optimizer.step()
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            preds, labels = [], []
            with torch.no_grad():
                for X, Xr, y in val_loader:
                    out = model(X.to(DEVICE), Xr.to(DEVICE))
                    preds.extend(torch.argmax(out, 1).cpu().numpy())
                    labels.extend(y.numpy())
            f1 = f1_score(labels, preds, average='macro')
            print(f"  Epoch {epoch+1}/{phase1_epochs} | Val Macro-F1: {f1:.4f}")
    
    # Phase 2: Full augmented dataset with class weights
    phase2_epochs = epochs - phase1_epochs
    print(f"\n{'='*60}")
    print(f"PHASE 2: Full SMOTE Dataset ({phase2_epochs} epochs)")
    print(f"{'='*60}")
    
    weights = np.load("data/processed_rr_history/class_weights.npy")
    weights[1] *= 5.0   # Reduced S-boost since SMOTE already helps
    weights[3] *= 3.0   # Boost F-class
    criterion_p2 = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE))
    
    loader_p2 = DataLoader(train_ds, batch_size=64, shuffle=True)
    best_f1 = 0.0
    
    for epoch in range(phase2_epochs):
        model.train()
        for X, Xr, y in loader_p2:
            X, Xr, y = X.to(DEVICE), Xr.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion_p2(model(X, Xr), y)
            loss.backward()
            optimizer.step()
        
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X, Xr, y in val_loader:
                out = model(X.to(DEVICE), Xr.to(DEVICE))
                preds.extend(torch.argmax(out, 1).cpu().numpy())
                labels.extend(y.numpy())
        
        f1 = f1_score(labels, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), OUT_DIR / "best_curriculum_smote.pth")
        
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}/{phase2_epochs} | Val F1: {f1:.4f} (Best: {best_f1:.4f})")
    
    # Final evaluation on test set
    model.load_state_dict(torch.load(OUT_DIR / "best_curriculum_smote.pth"))
    model.eval()
    
    test_preds, test_true = [], []
    with torch.no_grad():
        for X, Xr, y in test_loader:
            out = model(X.to(DEVICE), Xr.to(DEVICE))
            test_preds.extend(torch.argmax(out, 1).cpu().numpy())
            test_true.extend(y.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_true = np.array(test_true)
    
    cm = confusion_matrix(test_true, test_preds)
    class_recalls = cm.diagonal() / cm.sum(axis=1)
    
    print(f"\n{'='*60}")
    print("TEST SET RESULTS (with SMOTE)")
    print(f"{'='*60}")
    print(classification_report(test_true, test_preds,
                                target_names=['N', 'S', 'V', 'F', 'Q'], digits=4))
    
    metrics = {
        'seed': seed,
        'smote_ratio': smote_ratio,
        'f1_macro': float(f1_score(test_true, test_preds, average='macro')),
        'n_recall': float(class_recalls[0]),
        's_recall': float(class_recalls[1]),
        'v_recall': float(class_recalls[2]),
        'f_recall': float(class_recalls[3]) if len(class_recalls) > 3 else 0.0,
        'n_train_original': int(len(y_train)),
        'n_train_augmented': int(len(y_aug)),
    }
    
    with open(OUT_DIR / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    np.save(OUT_DIR / 'predictions.npy', test_preds)
    np.save(OUT_DIR / 'true_labels.npy', test_true)
    
    print(f"\n✅ Results saved to {OUT_DIR}/")
    print(f"   Macro-F1: {metrics['f1_macro']:.4f}")
    print(f"   S-Recall: {metrics['s_recall']:.4f}")
    print(f"   V-Recall: {metrics['v_recall']:.4f}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='results/curriculum_smote')
    parser.add_argument('--smote-ratio', type=float, default=0.5)
    parser.add_argument('--smote-k', type=int, default=5)
    args = parser.parse_args()
    
    print("="*60)
    print("CURRICULUM LEARNING + SMOTE AUGMENTATION")
    print(f"Seed: {args.seed} | Epochs: {args.epochs} | SMOTE ratio: {args.smote_ratio}")
    print("="*60)
    
    train_curriculum_smote(
        seed=args.seed, epochs=args.epochs,
        output_dir=args.output_dir,
        smote_ratio=args.smote_ratio, smote_k=args.smote_k
    )
