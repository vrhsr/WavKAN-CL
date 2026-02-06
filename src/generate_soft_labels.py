"""
Generate Soft Labels from Ensemble for Knowledge Distillation

Uses the 25-model ensemble (20 Diamond + 5 Grandmaster) to generate
soft probability distributions for knowledge distillation training.

Time: ~30 minutes
Output: data/soft_labels/
"""

import torch
import numpy as np
import os
import sys
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.wavkan import WavKANLinear
import torch.nn as nn

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

class SequenceWavKAN(nn.Module):
    def __init__(self):
        super(SequenceWavKAN, self).__init__()
        self.encoder = WavKANLinear(360, 64, wavelet_type='mexican_hat')
        self.ln = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(64, 64, 1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 5)
        )

    def forward(self, x):
        b, s, f = x.size()
        x = x.view(b*s, f)
        x = self.encoder(x)
        x = self.ln(x)
        x = self.dropout(x)
        x = x.view(b, s, -1)
        _, h = self.gru(x)
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.classifier(h)

class ECGDatasetRR(Dataset):
    def __init__(self, split):
        self.X = torch.tensor(np.load(f"data/processed_rr_history/X_{split}.npy"), dtype=torch.float32)
        self.Xr = torch.tensor(np.load(f"data/processed_rr_history/X_rr_{split}.npy"), dtype=torch.float32)
        self.y = torch.tensor(np.load(f"data/processed_rr_history/y_{split}.npy"), dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.Xr[i], self.y[i]

class SequenceDataset(Dataset):
    def __init__(self, split):
        self.X = torch.tensor(np.load(f"data/processed_sequence/X_seq_{split}.npy"), dtype=torch.float32)
        self.y = torch.tensor(np.load(f"data/processed_sequence/y_{split}.npy"), dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

def generate_soft_labels():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    OUT_DIR = "data/soft_labels"
    os.makedirs(OUT_DIR, exist_ok=True)
    
    diamond_seeds = [42, 101, 777, 2026, 9999, 1234, 2024, 31415, 27182, 7, 11, 13, 888, 555, 333, 99, 1001, 5050, 8080, 1998]
    gm_seeds = [42, 101, 777, 2026, 9999]
    
    for split in ['train', 'val', 'test']:
        print(f"\nGenerating soft labels for {split}...")
        
        # Load data
        ds_diamond = ECGDatasetRR(split)
        ds_gm = SequenceDataset(split)
        loader_d = DataLoader(ds_diamond, batch_size=64, shuffle=False)
        loader_g = DataLoader(ds_gm, batch_size=64, shuffle=False)
        
        # Collect predictions from all models
        all_probs = []
        
        # Diamond models
        print("Loading Diamond models...")
        for seed in tqdm(diamond_seeds):
            path = f"results/hybrid_rr_history_20_seeds/seed_{seed}/best_hybrid_rr.pth"
            if not os.path.exists(path): continue
            
            model = HybridWavKAN_RR().to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            
            probs = []
            with torch.no_grad():
                for X, Xr, _ in loader_d:
                    logits = model(X.to(DEVICE), Xr.to(DEVICE))
                    probs.append(torch.softmax(logits, dim=1).cpu())
            all_probs.append(torch.cat(probs).numpy())
        
        # GM models
        print("Loading Grandmaster models...")
        for seed in tqdm(gm_seeds):
            path = f"results/sequence_model/seed_{seed}/best_model.pth"
            if not os.path.exists(path): continue
            
            model = SequenceWavKAN().to(DEVICE)
            model.load_state_dict(torch.load(path, map_location=DEVICE))
            model.eval()
            
            probs = []
            with torch.no_grad():
                for X, _ in loader_g:
                    logits = model(X.to(DEVICE))
                    probs.append(torch.softmax(logits, dim=1).cpu())
            all_probs.append(torch.cat(probs).numpy())
        
        # Average probabilities (soft labels)
        soft_labels = np.mean(all_probs, axis=0)
        
        # Save
        np.save(f"{OUT_DIR}/soft_labels_{split}.npy", soft_labels)
        print(f"Saved {split} soft labels: {soft_labels.shape}")
    
    print(f"\n✅ Soft labels saved to {OUT_DIR}/")

if __name__ == "__main__":
    print("="*60)
    print("GENERATING SOFT LABELS FROM ENSEMBLE")
    print("="*60)
    generate_soft_labels()
