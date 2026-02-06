import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from sklearn.metrics import classification_report, f1_score
import json
import sys

# Ensure src is in path to import wavkan
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Reuse your working WavKAN
from src.wavkan import WavKANLinear 

# --- CONFIG ---
BATCH_SIZE = 64
LEARNING_RATE = 0.0005 
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR = "results/hybrid_final"
os.makedirs(OUT_DIR, exist_ok=True)

class ECGDataset(Dataset):
    def __init__(self, split):
        data_dir = "data/processed"
        self.X = np.load(os.path.join(data_dir, f"X_{split}.npy"))
        self.y = np.load(os.path.join(data_dir, f"y_{split}.npy"))
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]

class HybridWavKAN(nn.Module):
    def __init__(self, input_size=360, num_classes=5):
        super(HybridWavKAN, self).__init__()
        # MEXICAN HAT (The Winner)
        self.kan = WavKANLinear(input_size, 64, wavelet_type='mexican_hat')
        self.ln = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.2)
        # BiGRU
        self.bigru = nn.GRU(input_size=64, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.kan(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = x.unsqueeze(1)
        x, _ = self.bigru(x)
        x = x.squeeze(1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

def train():
    print("Re-running the WINNING configuration (Mexican Hat)...")
    train_loader = DataLoader(ECGDataset("train"), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(ECGDataset("val"), batch_size=BATCH_SIZE, shuffle=False)
    
    # Simple Class Weights (The strategy that worked)
    weights = np.load("data/processed/class_weights.npy")
    class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
    
    model = HybridWavKAN(input_size=360).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)

    best_val_f1 = 0.0
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            out = model(X_batch)
            loss = criterion(out, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(DEVICE)
                out = model(X_batch)
                preds = torch.argmax(out, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.numpy())
        
        val_f1 = f1_score(all_labels, all_preds, average='macro')
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {running_loss/len(train_loader):.4f} | Val F1: {val_f1:.4f}")
        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), f"{OUT_DIR}/best_hybrid.pth")

    # Final Test
    print("\n--- FINAL VERIFICATION ---")
    model.load_state_dict(torch.load(f"{OUT_DIR}/best_hybrid.pth"))
    model.eval()
    test_loader = DataLoader(ECGDataset("test"), batch_size=BATCH_SIZE, shuffle=False)
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(DEVICE)
            out = model(X_batch)
            preds = torch.argmax(out, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.numpy())
            
    print(classification_report(all_labels, all_preds, target_names=['N', 'S', 'V', 'F', 'Q']))
    
    # Save Metrics
    report = classification_report(all_labels, all_preds, target_names=['N', 'S', 'V', 'F', 'Q'], output_dict=True)
    with open(f"{OUT_DIR}/metrics.json", "w") as f:
        json.dump(report, f, indent=4)

if __name__ == "__main__":
    train()
