import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import json
import os
import sys
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import WavKAN
from src.wavkan import WavKANClassifier

# Config
BATCH_SIZE = 64
EPOCHS = 30
LR = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['N', 'S', 'V', 'F', 'Q']

def load_data():
    print("Loading data...")
    X_train = np.load("data/processed/X_train.npy").astype(np.float32)
    y_train = np.load("data/processed/y_train.npy").astype(np.longlong)
    X_val = np.load("data/processed/X_val.npy").astype(np.float32)
    y_val = np.load("data/processed/y_val.npy").astype(np.longlong)
    X_test = np.load("data/processed/X_test.npy").astype(np.float32)
    y_test = np.load("data/processed/y_test.npy").astype(np.longlong)
    
    # Load weights
    class_weights = None
    if os.path.exists("data/processed/class_weights.npy"):
        weights = np.load("data/processed/class_weights.npy")
        class_weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)
        print(f"Loaded class weights: {weights}")

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights

def create_dataloaders(train_data, val_data):
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    train_ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    val_ds = TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val))
    
    # Standard Shuffle Loader (Matching Hybrid Config)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    return train_loader, val_loader

def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()
            
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = running_loss / len(loader)
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    return avg_loss, macro_f1, all_preds, all_targets

def train_wavkan():
    (X_train, y_train), (X_val, y_val), (X_test, y_test), class_weights = load_data()
    train_loader, val_loader = create_dataloaders((X_train, y_train), (X_val, y_val))
    
    # WAVKAN SETUP
    input_dim = X_train.shape[-1]
    print(f"Initializing WavKANClassifier with input dim: {input_dim}")
    model = WavKANClassifier(input_size=input_dim, num_classes=5).to(DEVICE)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"WavKAN Total Parameters: {total_params:,}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"Starting Pure WavKAN training on {DEVICE}...")
    best_f1 = 0.0
    
    os.makedirs("results/models", exist_ok=True)
    os.makedirs("experiments/wavkan", exist_ok=True)
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
        train_loss = running_loss / len(train_loader)
        val_loss, val_f1, _, _ = evaluate(model, val_loader, criterion)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), "results/models/wavkan_pure.pth")

    print(f"\nTraining Complete. Best Val F1: {best_f1:.4f}")
    
    # --- TEST EVALUATION ---
    print("\nEvaluating WavKAN on Test Set...")
    model.load_state_dict(torch.load("results/models/wavkan_pure.pth"))
    test_ds = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    _, test_f1, preds, targets = evaluate(model, test_loader, criterion)
    
    print(f"Test F1 (Macro): {test_f1:.4f}")
    report = classification_report(targets, preds, labels=[0,1,2,3,4], target_names=CLASS_NAMES, output_dict=True)
    print("\nClassification Report:")
    print(classification_report(targets, preds, labels=[0,1,2,3,4], target_names=CLASS_NAMES))
    
    with open("experiments/wavkan/wavkan_metrics.json", "w") as f:
        json.dump(report, f, indent=2)
    
    cm = confusion_matrix(targets, preds, labels=[0,1,2,3,4])
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('WavKAN Pure Confusion Matrix')
    plt.savefig("experiments/wavkan/wavkan_confusion_matrix.png")

if __name__ == "__main__":
    train_wavkan()
