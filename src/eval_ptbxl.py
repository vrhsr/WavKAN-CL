"""
PTB-XL Zero-Shot Evaluation
============================
Loads the best trained WavKAN-v2 checkpoint (trained on MIT-BIH) and
evaluates it zero-shot on the PTB-XL test set (Lead-II, AAMI superclasses).

Usage:
    python src/eval_ptbxl.py [--model-path results/wavkan_v2_curriculum/seed_42/best_model.pth]
                              [--data-dir data/processed_ptbxl]
                              [--out results/ptbxl_zero_shot_metrics.json]

Fixed 2026-08-13 (AUDIT_FINDINGS.md, blocking the cross-dataset Phase 5 re-run): this
script used to hardcode the old HybridWavKAN_RR (95K, "WavKAN-CL") architecture, copy-
pasted from fusion_engine.py/train_curriculum.py. Since the model identity decision
(C3) settled on WavKAN_v2 (154K) as canonical, loading the new real checkpoints into
the old class here would have failed outright (mismatched state dict keys/shapes) --
now imports the real model class, matching the pattern already correct in
eval_multidataset.py.
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, f1_score, confusion_matrix

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from models.wavkan_v2 import WavKAN_v2


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class PTBXLDataset(Dataset):
    def __init__(self, data_dir: str, split: str = "test"):
        self.X  = torch.tensor(
            np.load(os.path.join(data_dir, f"X_{split}.npy")), dtype=torch.float32)
        self.Xr = torch.tensor(
            np.load(os.path.join(data_dir, f"X_rr_{split}.npy")), dtype=torch.float32)
        self.y  = torch.tensor(
            np.load(os.path.join(data_dir, f"y_{split}.npy")), dtype=torch.long)

    def __len__(self):  return len(self.y)
    def __getitem__(self, i): return self.X[i], self.Xr[i], self.y[i]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(model_path: str, data_dir: str, out_path: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load model
    model = WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=True).to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {model_path}\n"
            "Run training first, or specify --model-path."
        )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"✅ Loaded model: {model_path}")

    # Load PTB-XL test set
    if not os.path.exists(os.path.join(data_dir, "X_test.npy")):
        raise FileNotFoundError(
            f"PTB-XL test data not found in {data_dir}.\n"
            "Run: python src/process_ptbxl.py --data-dir data/ptbxl --out-dir data/processed_ptbxl"
        )

    dataset = PTBXLDataset(data_dir, split="test")
    loader  = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=0)
    print(f"✅ PTB-XL test set: {len(dataset)} beats")

    # Inference
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, Xr, y in loader:
            logits = model(X.to(device), Xr.to(device))
            preds  = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Class distribution in test set
    class_names = ["N", "S", "V", "F", "Q"]
    unique, counts = np.unique(all_labels, return_counts=True)
    print("\n📊 PTB-XL Test Class Distribution:")
    for u, c in zip(unique, counts):
        print(f"   {class_names[u]}: {c} ({100*c/len(all_labels):.1f}%)")

    # Report
    # labels=list(range(5)) (not list(unique)) -- a class absent from the target dataset
    # (PTB-XL has no Fusion/F beats) must still occupy its own fixed position matching
    # class_names, or sklearn positionally reassigns target_names to whichever labels
    # ARE present, silently relabeling the next real class (Q) as the missing one's name
    # (F). Matches the pattern confusion_matrix already used correctly below.
    print("\n📋 Zero-Shot Classification Report (MIT-BIH → PTB-XL):")
    print(classification_report(
        all_labels, all_preds,
        target_names=class_names,
        labels=list(range(5)),
        digits=3,
        zero_division=0
    ))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(5)))
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Per-class recall
    class_recall = {}
    for i, name in enumerate(class_names):
        if cm[i].sum() > 0:
            class_recall[name] = float(cm[i, i] / cm[i].sum())
        else:
            class_recall[name] = None

    report_dict = classification_report(
        all_labels, all_preds,
        target_names=class_names,
        labels=list(range(5)),
        output_dict=True,
        zero_division=0
    )
    report_dict["macro_f1"] = float(macro_f1)
    report_dict["per_class_recall"] = class_recall
    report_dict["model_path"] = model_path
    report_dict["n_test_beats"] = int(len(dataset))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report_dict, f, indent=4)

    print(f"\n✅ Metrics saved to: {out_path}")
    print(f"   Macro-F1: {macro_f1:.4f}")
    print(f"   V-Recall: {class_recall.get('V', 'N/A')}")
    print(f"   S-Recall: {class_recall.get('S', 'N/A')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Zero-shot PTB-XL evaluation")
    parser.add_argument(
        "--model-path", type=str,
        default="results/wavkan_v2_curriculum/seed_42/best_model.pth"
    )
    parser.add_argument("--data-dir", type=str, default="data/processed_ptbxl")
    parser.add_argument("--out",  type=str, default="results/ptbxl_zero_shot_metrics.json")
    args = parser.parse_args()
    evaluate(args.model_path, args.data_dir, args.out)
