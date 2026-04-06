"""
Curriculum Learning: Minority-First Training Strategy

Phase 1 (15 epochs): Train only on S, V, F samples (minorities)
Phase 2 (35 epochs): Gradually introduce N samples

Scientific Justification:
"We employ curriculum learning to prevent majority-class dominance during 
early training, presenting rare arrhythmia patterns before common beats."

Expected Gain: +0.05 to +0.10 S-Recall
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
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.wavkan import WavKANLinear

class HybridWavKAN_RR(nn.Module):
    def __init__(self):
        super(HybridWavKAN_RR, self).__init__()
        self.kan = WavKANLinear(360, 64, wavelet_type='mexican_hat')
        self.ln = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.2)
        self.bigru = nn.GRU(64, 32, 1, batch_first=True, bidirectional=True)
        self.rr_mlp = nn.Sequential(
            nn.Linear(5, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU()
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
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

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

def train_curriculum(seed=42, epochs=50, output_dir="results/curriculum_model"):
    set_seed(seed)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR = Path(output_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    train_ds = ECGDatasetRR("train")
    val_ds = ECGDatasetRR("val")
    test_ds = ECGDatasetRR("test")
    
    # Phase 1: Minority only
    train_labels = train_ds.y.numpy()
    minority_idx = np.where(train_labels > 0)[0]  # S, V, F, Q
    
    print(f"Phase 1: Training on {len(minority_idx)} minority samples")
    
    model = HybridWavKAN_RR().to(DEVICE)
    
    # Focal loss for minorities
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # PHASE 1: Minority-first (30% of epochs, minimum 5)
    phase1_epochs = max(5, min(15, int(epochs * 0.3)))
    minority_subset = Subset(train_ds, minority_idx)
    loader_p1 = DataLoader(minority_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    
    print("\n" + "="*60)
    print(f"PHASE 1: Learning from Minorities ({phase1_epochs} epochs)")
    print("="*60)
    
    for epoch in range(phase1_epochs):
        model.train()
        for X, Xr, y in loader_p1:
            X, Xr, y = X.to(DEVICE), Xr.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X, Xr)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
        
        # Val
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X, Xr, y in val_loader:
                out = model(X.to(DEVICE), Xr.to(DEVICE))
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                labels.extend(y.numpy())
        
        f1 = f1_score(labels, preds, average='macro')
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{phase1_epochs} | Val F1: {f1:.4f}")
    
    # PHASE 2: Full dataset with reweighting
    phase2_epochs = epochs - phase1_epochs
    print("\n" + "="*60)
    print(f"PHASE 2: Full Dataset with Class Weights ({phase2_epochs} epochs)")
    print("="*60)
    
    # Load weights
    weights = np.load("data/processed_rr_history/class_weights.npy")
    weights[1] *= 10.0  # Boost S-class
    criterion_p2 = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32).to(DEVICE))
    
    loader_p2 = DataLoader(train_ds, batch_size=64, shuffle=True)
    best_f1 = 0.0
    
    for epoch in range(phase2_epochs):
        model.train()
        for X, Xr, y in loader_p2:
            X, Xr, y = X.to(DEVICE), Xr.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(X, Xr)
            loss = criterion_p2(out, y)
            loss.backward()
            optimizer.step()
        
        # Val
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X, Xr, y in val_loader:
                out = model(X.to(DEVICE), Xr.to(DEVICE))
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                labels.extend(y.numpy())
        
        f1 = f1_score(labels, preds, average='macro')
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), OUT_DIR / "best_curriculum.pth")
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{phase2_epochs} | Val F1: {f1:.4f} (Best: {best_f1:.4f})")
    
    # Load best model and evaluate on test
    model.load_state_dict(torch.load(OUT_DIR / "best_curriculum.pth"))
    model.eval()
    
    test_preds = []
    test_true = []
    with torch.no_grad():
        for X, Xr, y in test_loader:
            X, Xr, y = X.to(DEVICE), Xr.to(DEVICE), y.to(DEVICE)
            out = model(X, Xr)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            test_preds.extend(preds)
            test_true.extend(y.cpu().numpy())
    
    # Compute metrics
    test_f1 = f1_score(test_true, test_preds, average='macro')
    test_precision = precision_score(test_true, test_preds, average='macro')
    
    # Per-class recall
    cm = confusion_matrix(test_true, test_preds)
    class_recalls = cm.diagonal() / cm.sum(axis=1)
    
    metrics = {
        'seed': seed,
        'f1': float(test_f1),
        'precision': float(test_precision),
        's_recall': float(class_recalls[1]),  # S is index 1
        'v_recall': float(class_recalls[2]),  # V is index 2
        'n_recall': float(class_recalls[0]),  # N is index 0
        'specificity': float(class_recalls[0])
    }
    
    # Save metrics
    with open(OUT_DIR / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save predictions
    np.save(OUT_DIR / 'predictions.npy', test_preds)
    np.save(OUT_DIR / 'true_labels.npy', test_true)
    
    print(f"\n✅ Curriculum model saved to {OUT_DIR}/")
    print(f"Test F1: {test_f1:.4f} | S-Recall: {class_recalls[1]:.4f}")
    
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--output-dir', type=str, default='results/curriculum_model')
    
    args = parser.parse_args()
    
    print("="*60)
    print("CURRICULUM LEARNING: Minority-First Strategy")
    print(f"Seed: {args.seed} | Epochs: {args.epochs}")
    print("="*60)
    
    train_curriculum(seed=args.seed, epochs=args.epochs, output_dir=args.output_dir)
