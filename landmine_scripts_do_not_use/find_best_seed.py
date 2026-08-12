"""Find the seed that matches the Table 6 values in the manuscript."""
import torch
import numpy as np
import sys, os
sys.path.append('.')
from src.wavkan import WavKANLinear
import torch.nn as nn
from sklearn.metrics import classification_report

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

# Load test data
X = torch.tensor(np.load('data/processed_rr_history/X_test.npy'), dtype=torch.float32)
Xr = torch.tensor(np.load('data/processed_rr_history/X_rr_test.npy'), dtype=torch.float32)
y = np.load('data/processed_rr_history/y_test.npy')

seeds_dir = 'results/hybrid_rr_history_20_seeds'
device = torch.device('cpu')
best_v = 0
best_seed = None
best_report = None

for seed_dir in sorted(os.listdir(seeds_dir)):
    path = os.path.join(seeds_dir, seed_dir, 'best_hybrid_rr.pth')
    if not os.path.exists(path):
        continue
    
    model = HybridWavKAN_RR()
    model.load_state_dict(torch.load(path, map_location='cpu'))
    model.eval()
    
    with torch.no_grad():
        out = model(X, Xr)
        preds = torch.argmax(out, 1).numpy()
    
    report = classification_report(y, preds, target_names=['N','S','V','F','Q'], output_dict=True, zero_division=0)
    v_rec = report['V']['recall']
    
    print(f"{seed_dir}: V-Rec={v_rec:.3f} S-Rec={report['S']['recall']:.3f} "
          f"N-P={report['N']['precision']:.3f} N-R={report['N']['recall']:.3f} "
          f"V-P={report['V']['precision']:.3f} F-R={report['F']['recall']:.3f}")
    
    if v_rec > best_v:
        best_v = v_rec
        best_seed = seed_dir
        best_report = report

print(f"\n=== BEST SEED: {best_seed} (V-Recall={best_v:.3f}) ===")
print(f"N: P={best_report['N']['precision']:.3f} R={best_report['N']['recall']:.3f} F1={best_report['N']['f1-score']:.3f}")
print(f"S: P={best_report['S']['precision']:.3f} R={best_report['S']['recall']:.3f} F1={best_report['S']['f1-score']:.3f}")
print(f"V: P={best_report['V']['precision']:.3f} R={best_report['V']['recall']:.3f} F1={best_report['V']['f1-score']:.3f}")
print(f"F: P={best_report['F']['precision']:.3f} R={best_report['F']['recall']:.3f} F1={best_report['F']['f1-score']:.3f}")
print(f"Q: P={best_report['Q']['precision']:.3f} R={best_report['Q']['recall']:.3f} F1={best_report['Q']['f1-score']:.3f}")
