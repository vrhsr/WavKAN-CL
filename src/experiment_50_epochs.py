import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import os
import random
import sys
import time
from sklearn.metrics import f1_score
import torch.nn.functional as F

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# --- IMPORT YOUR MODELS ---
from src.wavkan import WavKANClassifier 
from src.train_hybrid import HybridWavKAN, ECGDataset
from models.cnn_baseline import CNN1D 

# --- CONFIGURATION ---
# "Train ALL models" - 20 seeds for maximum rigor (Aligned with Fair Fight)
SEEDS = [
    42, 101, 777, 2026, 9999, 1234, 2024, 31415, 27182, 7, 
    11, 13, 888, 555, 333, 99, 1001, 5050, 8080, 1998
] 
BATCH_SIZE = 64
EPOCHS = 50
MIXUP_ALPHA = 0.2
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OUT_DIR_PREFIX = "results/ablation_50_epochs"
os.makedirs(OUT_DIR_PREFIX, exist_ok=True)

# --- FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# --- MIXUP UTILS ---
def mixup_data(x, y, alpha=0.2):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    index = torch.randperm(x.size(0)).to(DEVICE)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

# --- UNIVERSAL TRAINER ---
def train_model(model_name, seed):
    set_seed(seed)
    print(f"\n🏃 TRAINING {model_name.upper()} | SEED {seed} | 50 EPOCHS")
    
    # 1. Setup Output
    out_dir = f"{OUT_DIR_PREFIX}/{model_name}"
    os.makedirs(out_dir, exist_ok=True)
    save_path = f"{out_dir}/model_seed_{seed}.pth"
    
    # Skip if already done
    if os.path.exists(save_path):
        print(f"   -> Seed {seed} already exists. Skipping.")
        return

    start_time = time.time()

    # 2. Load Data
    train_loader = DataLoader(ECGDataset("train"), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(ECGDataset("val"), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # 3. Initialize Model
    if model_name == "cnn":
        model = CNN1D(num_classes=5, input_len=360).to(DEVICE)
    elif model_name == "wavkan":
        model = WavKANClassifier(input_size=360, num_classes=5).to(DEVICE)
    elif model_name == "hybrid":
        model = HybridWavKAN(input_size=360).to(DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # 4. Setup Loss & Optimizer
    try:
        raw_weights = np.load("data/processed/class_weights.npy")
        class_weights = torch.tensor(raw_weights, dtype=torch.float32).to(DEVICE)
    except:
        class_weights = torch.tensor([0.2, 10.0, 2.5, 20.0, 1000.0]).to(DEVICE)
        
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    # Patience increased because we have more epochs
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=7)

    best_val_f1 = 0.0
    
    for epoch in range(EPOCHS):
        model.train()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            inputs, ta, tb, lam = mixup_data(X, y, MIXUP_ALPHA)
            
            optimizer.zero_grad()
            out = model(inputs)
            loss = mixup_criterion(criterion, out, ta, tb, lam)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
        # Validation
        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                out = model(X)
                preds.extend(torch.argmax(out, dim=1).cpu().numpy())
                labels.extend(y.numpy())
        
        val_f1 = f1_score(labels, preds, average='macro')
        scheduler.step(val_f1)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), save_path)
            epochs_no_improve = 0
            best_epoch = epoch + 1
        else:
            epochs_no_improve += 1
            
        # Log every 5 epochs or if improved
        if (epoch + 1) % 5 == 0 or val_f1 == best_val_f1:
             elapsed = (time.time() - start_time) / 60
             print(f"   Ep {epoch+1}/{EPOCHS} | Val F1: {val_f1:.4f} (Best: {best_val_f1:.4f} @ Ep {best_epoch}) | Time: {elapsed:.1f} min")

        # Early Stopping
        if epochs_no_improve >= 10: # Patience of 10 epochs
            print(f"   ⏹️ Early stopping triggered at epoch {epoch+1}!")
            break
            
    total_time = (time.time() - start_time) / 60
    print(f"   -> Finished {model_name} Seed {seed}. Best F1: {best_val_f1:.4f} ({total_time:.1f} min)")

if __name__ == "__main__":
    print("="*60)
    print("🏁 50-EPOCH ABLATION STUDY (5 Seeds per Model) 🏁")
    print("="*60)
    print(f"Device: {DEVICE}")
    print(f"Epochs: {EPOCHS}, MixUp Alpha: {MIXUP_ALPHA}")
    print("="*60)
    
    total_start = time.time()
    
    # 1. Train 5 Baseline CNNs
    print("\n--- PHASE 1: CNN BASELINE ---")
    for seed in SEEDS: train_model("cnn", seed)
    
    # 2. Train 5 Pure WavKANs
    print("\n--- PHASE 2: PURE WAVKAN ---")
    for seed in SEEDS: train_model("wavkan", seed)
    
    # 3. Train 5 Hybrid Models
    print("\n--- PHASE 3: HYBRID WAVKAN+BiGRU ---")
    for seed in SEEDS: train_model("hybrid", seed)
    
    total_time = (time.time() - total_start) / 3600
    print("\n" + "="*60)
    print(f"✅ 50-EPOCH TRAINING COMPLETE. Total Time: {total_time:.2f} hours")
    print("Run 'python src/eval_50_epochs.py' for comparison.")
    print("="*60)
