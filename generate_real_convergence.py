"""
Generate REAL convergence curves for the manuscript figure.
Trains both Baseline (standard CE) and Curriculum Learning for 50 epochs,
logging per-epoch metrics, then plots the 4-panel figure.
Expected runtime: ~30 minutes on CPU.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import json
import os
import sys
from sklearn.metrics import f1_score, confusion_matrix

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.wavkan import WavKANLinear


# --- Model (same as main paper) ---
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


def evaluate(model, loader, device):
    """Compute val metrics."""
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for X, Xr, y in loader:
            out = model(X.to(device), Xr.to(device))
            preds.extend(torch.argmax(out, 1).cpu().numpy())
            labels.extend(y.numpy())
    preds = np.array(preds)
    labels = np.array(labels)
    cm = confusion_matrix(labels, preds, labels=[0,1,2,3,4])
    row_sums = cm.sum(axis=1)
    row_sums[row_sums == 0] = 1
    class_recalls = cm.diagonal() / row_sums
    return {
        'f1_macro': float(f1_score(labels, preds, average='macro', zero_division=0)),
        'v_recall': float(class_recalls[2]),
        's_recall': float(class_recalls[1]),
    }


def train_with_logging(mode='baseline', seed=42, epochs=50):
    """
    Train model and return per-epoch metrics.
    mode: 'baseline' (standard CE) or 'curriculum' (minority-first)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_ds = ECGDatasetRR("train")
    val_ds = ECGDatasetRR("val")

    model = HybridWavKAN_RR().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    # Prepare minority subset for curriculum
    train_labels = train_ds.y.numpy()
    minority_idx = np.where(train_labels > 0)[0]
    minority_subset = Subset(train_ds, minority_idx)
    minority_loader = DataLoader(minority_subset, batch_size=32, shuffle=True)
    full_loader = DataLoader(train_ds, batch_size=64, shuffle=True)

    # Class weights for phase 2
    weights = np.load("data/processed_rr_history/class_weights.npy")
    weights[1] *= 10.0
    weighted_criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32).to(device)
    )

    phase_transition = max(5, min(15, int(epochs * 0.3)))

    history = {'loss': [], 'f1_macro': [], 'v_recall': [], 's_recall': []}

    print(f"\n{'='*50}")
    print(f"  Training: {mode.upper()} (seed={seed}, epochs={epochs})")
    print(f"{'='*50}")

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0

        if mode == 'curriculum' and epoch < phase_transition:
            # Phase 1: Minority only
            loader = minority_loader
            loss_fn = criterion
        else:
            # Phase 2 (or baseline): Full dataset
            loader = full_loader
            loss_fn = weighted_criterion if mode == 'curriculum' else criterion

        for X, Xr, y in loader:
            X, Xr, y = X.to(device), Xr.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(X, Xr)
            loss = loss_fn(out, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        # Evaluate on validation set
        metrics = evaluate(model, val_loader, device)

        history['loss'].append(avg_loss)
        history['f1_macro'].append(metrics['f1_macro'])
        history['v_recall'].append(metrics['v_recall'])
        history['s_recall'].append(metrics['s_recall'])

        if (epoch + 1) % 5 == 0:
            phase = "P1-MIN" if (mode == 'curriculum' and epoch < phase_transition) else "P2-FULL"
            print(f"  Epoch {epoch+1:3d}/{epochs} [{phase}] | "
                  f"Loss: {avg_loss:.4f} | F1: {metrics['f1_macro']:.4f} | "
                  f"V: {metrics['v_recall']:.3f} | S: {metrics['s_recall']:.3f}")

    return history


def plot_convergence(baseline_hist, curriculum_hist, output_path='fig_convergence_curves.png'):
    """Generate the 4-panel convergence figure."""
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(baseline_hist['loss']) + 1)
    phase_transition = 15

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    panels = [
        ('(a) Training Loss', 'loss', 'Loss'),
        ('(b) Val Macro-F1', 'f1_macro', 'Macro-F1'),
        ('(c) V-Recall', 'v_recall', 'V-Recall'),
        ('(d) S-Recall', 's_recall', 'S-Recall'),
    ]

    for ax, (title, key, ylabel) in zip(axes, panels):
        ax.plot(epochs, baseline_hist[key], color='#2196F3', alpha=0.85,
                label='Baseline', linewidth=1.5)
        ax.plot(epochs, curriculum_hist[key], color='#FF5722', alpha=0.85,
                label='Curriculum', linewidth=1.5)
        ax.axvline(x=phase_transition, color='gray', linestyle='--', alpha=0.6,
                   label='Phase Transition' if key == 'loss' else '')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        if key == 'loss':
            ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Figure saved to: {output_path}")


if __name__ == '__main__':
    # Train both models
    baseline_hist = train_with_logging(mode='baseline', seed=42, epochs=50)
    curriculum_hist = train_with_logging(mode='curriculum', seed=42, epochs=50)

    # Save raw history for reproducibility
    os.makedirs('results', exist_ok=True)
    with open('results/convergence_history.json', 'w') as f:
        json.dump({'baseline': baseline_hist, 'curriculum': curriculum_hist}, f, indent=2)
    print("\n📊 Raw history saved to results/convergence_history.json")

    # Plot
    plot_convergence(baseline_hist, curriculum_hist)
