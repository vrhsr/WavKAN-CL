import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
import json
from sklearn.metrics import classification_report, f1_score, log_loss
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Ensure src is in path to import wavkan
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.wavkan import WavKANLinear 

# --- RE-DEFINE MODEL CLASSES SAFELY ---
# (To avoid importing messy scripts that might trigger training)

class HybridWavKAN_RR(nn.Module):
    def __init__(self, input_size=360, num_classes=5):
        super(HybridWavKAN_RR, self).__init__()
        self.kan = WavKANLinear(input_size, 64, wavelet_type='mexican_hat')
        self.ln = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.2)
        self.bigru = nn.GRU(input_size=64, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        # 10x Run uses 64->32->16 MLP
        self.rr_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        self.fc1 = nn.Linear(80, 48)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(48, num_classes)

    def forward(self, x_signal, x_rr):
        x = self.kan(x_signal)
        x = self.ln(x)
        x = self.dropout(x)
        x = x.unsqueeze(1)
        x, _ = self.bigru(x)
        x_morph = x.squeeze(1)
        x_rhythm = self.rr_mlp(x_rr)
        combined = torch.cat((x_morph, x_rhythm), dim=1)
        x = self.fc1(combined)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class SequenceWavKAN(nn.Module):
    def __init__(self, input_size=360, num_classes=5):
        super(SequenceWavKAN, self).__init__()
        self.encoder = WavKANLinear(input_size, 64, wavelet_type='mexican_hat')
        self.ln = nn.LayerNorm(64)
        self.dropout = nn.Dropout(0.2)
        self.gru = nn.GRU(input_size=64, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x_seq):
        batch_size, seq_len, feat_dim = x_seq.size()
        x_flat = x_seq.view(batch_size * seq_len, feat_dim)
        x_emb = self.encoder(x_flat)
        x_emb = self.ln(x_emb)
        x_emb = self.dropout(x_emb)
        x_emb = x_emb.view(batch_size, seq_len, -1)
        _, hidden = self.gru(x_emb)
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        logits = self.classifier(hidden_cat)
        return logits

# --- DATASETS ---

class ECGDatasetRR(Dataset):
    def __init__(self, split, base_dir):
        self.X = np.load(os.path.join(base_dir, f"X_{split}.npy"))
        self.X_rr = np.load(os.path.join(base_dir, f"X_rr_{split}.npy"))
        self.y = np.load(os.path.join(base_dir, f"y_{split}.npy"))
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.X_rr = torch.tensor(self.X_rr, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.X_rr[idx], self.y[idx]

class SequenceDataset(Dataset):
    def __init__(self, split, base_dir):
        self.X = np.load(os.path.join(base_dir, f"X_seq_{split}.npy"))
        self.y = np.load(os.path.join(base_dir, f"y_{split}.npy"))
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


# --- FUSION ENGINE ---

class FusionEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.diamond_dir = "results/hybrid_rr_history_20_seeds"
        self.grandmaster_dir = "results/sequence_model"
        self.diamond_data_dir = "data/processed_rr_history"
        self.grandmaster_data_dir = "data/processed_sequence"
        
        self.selected_models = [] # List of {'type':, 'path':, 'f1':, 'temp':}

    def load_and_evaluate_seeds(self, model_type, seeds, base_dir, data_dir):
        print(f"\nEvaluating {model_type} seeds...")
        
        # Load Val Data
        if model_type == 'diamond':
            val_ds = ECGDatasetRR("val", data_dir)
        else:
            val_ds = SequenceDataset("val", data_dir)
            
        val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
        
        results = []
        
        for seed in tqdm(seeds):
            seed_path = os.path.join(base_dir, f"seed_{seed}", "best_hybrid_rr.pth" if model_type == 'diamond' else "best_model.pth")
            
            if not os.path.exists(seed_path):
                continue
                
            # Init Model
            if model_type == 'diamond':
                model = HybridWavKAN_RR().to(self.device)
            else:
                model = SequenceWavKAN().to(self.device)
                
            model.load_state_dict(torch.load(seed_path, map_location=self.device))
            model.eval()
            
            # Inference (Get Logits and Labels for Calibration/Eval)
            all_logits = []
            all_labels = []
            
            with torch.no_grad():
                for batch in val_loader:
                    if model_type == 'diamond':
                        x, x_rr, y = batch
                        logits = model(x.to(self.device), x_rr.to(self.device))
                    else:
                        x, y = batch
                        logits = model(x.to(self.device))
                        
                    all_logits.append(logits.cpu())
                    all_labels.append(y)
            
            logits_tensor = torch.cat(all_logits)
            labels_tensor = torch.cat(all_labels)
            
            # Calc F1
            preds = torch.argmax(logits_tensor, dim=1).numpy()
            f1 = f1_score(labels_tensor.numpy(), preds, average='macro')
            
            # --- TEMPERATURE SCALING ---
            # Optimize T to minimize NLL
            T = nn.Parameter(torch.ones(1))
            optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=50)
            nll_criterion = nn.CrossEntropyLoss()
            
            def eval_nll():
                optimizer.zero_grad()
                loss = nll_criterion(logits_tensor / T, labels_tensor)
                loss.backward()
                return loss
            
            optimizer.step(eval_nll)
            optimal_T = T.item()
            
            results.append({
                'type': model_type,
                'path': seed_path,
                'f1': f1,
                'T': optimal_T
            })
            
        # Sort by F1
        results.sort(key=lambda x: x['f1'], reverse=True)
        return results

    def run_fusion(self):
        # 1. Evaluate & Select Top K
        diamond_seeds = [42, 101, 777, 2026, 9999, 1234, 2024, 31415, 27182, 7, 11, 13, 888, 555, 333, 99, 1001, 5050, 8080, 1998]
        gm_seeds = [42, 101, 777, 2026, 9999]
        
        diamond_results = self.load_and_evaluate_seeds('diamond', diamond_seeds, self.diamond_dir, self.diamond_data_dir)
        gm_results = self.load_and_evaluate_seeds('grandmaster', gm_seeds, self.grandmaster_dir, self.grandmaster_data_dir)
        
        if not diamond_results or not gm_results:
            print("❌ Models not found. Wait for training to finish.")
            return

        # Top K Selection (Anti-Pollution)
        # Keep Top 40%
        top_k_diamond = diamond_results[:max(1, int(len(diamond_results)*0.4))]
        top_k_gm = gm_results[:max(1, int(len(gm_results)*0.6))] # Keep more GM if few seeds
        
        print(f"\n✅ Selected {len(top_k_diamond)} Diamond models (Best F1: {top_k_diamond[0]['f1']:.4f})")
        print(f"✅ Selected {len(top_k_gm)} Grandmaster models (Best F1: {top_k_gm[0]['f1']:.4f})")
        
        self.selected_models = top_k_diamond + top_k_gm
        
        # 2. Inference on TEST SET
        print("\nRunning Fusion on TEST SET...")
        
        # We need two loaders because inputs are different
        test_ds_diamond = ECGDatasetRR("test", self.diamond_data_dir)
        test_ds_gm = SequenceDataset("test", self.grandmaster_data_dir)
        
        loader_d = DataLoader(test_ds_diamond, batch_size=64, shuffle=False)
        loader_g = DataLoader(test_ds_gm, batch_size=64, shuffle=False)
        
        # Accumulate PROBABILITIES (Softmaxed & Calibrated)
        # Shape: (N, 5)
        fused_probs = torch.zeros((len(test_ds_diamond), 5)).to(self.device)
        
        # Helper to run inference
        def get_probs(model_info, loader):
            model_path = model_info['path']
            model_type = model_info['type']
            T = model_info['T']
            
            if model_type == 'diamond':
                model = HybridWavKAN_RR().to(self.device)
            else:
                model = SequenceWavKAN().to(self.device)
                
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            
            probs_list = []
            with torch.no_grad():
                for batch in loader:
                    if model_type == 'diamond':
                        x, x_rr, y = batch
                        logits = model(x.to(self.device), x_rr.to(self.device))
                    else:
                        x, y = batch
                        logits = model(x.to(self.device))
                    
                    # Apply Temp Scaling & Softmax
                    probs = F.softmax(logits / T, dim=1)
                    probs_list.append(probs)
            
            return torch.cat(probs_list)

        # Run Inference for all selected models
        for model_info in tqdm(self.selected_models):
            if model_info['type'] == 'diamond':
                probs = get_probs(model_info, loader_d)
                # Diamond Weighting (Stronger on V)
                weight_vector = torch.tensor([1.0, 0.3, 0.7, 1.0, 1.0]).to(self.device)
            else:
                probs = get_probs(model_info, loader_g)
                # Grandmaster Weighting (Stronger on S)
                weight_vector = torch.tensor([1.0, 0.7, 0.3, 1.0, 1.0]).to(self.device)
            
            # Weighted Sum
            fused_probs += (probs * weight_vector)

        # Final Preds
        final_preds = torch.argmax(fused_probs, dim=1).cpu().numpy()
        
        # Get Targets (Same for both)
        targets = np.load(os.path.join(self.diamond_data_dir, "y_test.npy"))
        
        # Report
        print("\n🏆 THESIS FUSION RESULTS 🏆")
        print(classification_report(targets, final_preds, target_names=['N', 'S', 'V', 'F', 'Q'], digits=4))
        
        report = classification_report(targets, final_preds, target_names=['N', 'S', 'V', 'F', 'Q'], output_dict=True)
        with open("results/thesis_fusion_metrics.json", "w") as f:
            json.dump(report, f, indent=4)

if __name__ == "__main__":
    engine = FusionEngine()
    engine.run_fusion()
