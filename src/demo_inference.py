"""
Minimal Inference Demo (I5)
===========================
Demonstrates how to run the WavKAN-CL model on a single ECG beat
and an RR-interval history vector to get a clinical prediction.

Usage:
    python src/demo_inference.py
"""

import torch
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.wavkan import WavKANLinear
from src.run_ablation_study import FullModel

def run_demo():
    print("="*60)
    print(" WAVKAN-CL MULTI-MODAL INFERENCE DEMO")
    print("="*60)

    # 1. Initialize Model
    model = FullModel(wavelet_type='mexican_hat')
    model.eval()
    
    # 2. Suppose we have a 360-sample beat (1 second at 360Hz)
    # (Here we use random noise, but in reality this is the raw ECG voltage)
    dummy_beat = torch.randn(1, 360)
    
    # 3. Suppose we have a 5-element RR-interval normalized history
    # Example: [t-4, t-3, t-2, t-1, current]
    dummy_rr = torch.tensor([[0.95, 0.98, 1.02, 1.01, 0.65]]) # Premature beat signature
    
    # 4. Run inference
    with torch.no_grad():
        logits = model(dummy_beat, dummy_rr)
        probs = torch.softmax(logits, dim=1)[0].numpy()
        pred_class = np.argmax(probs)
    
    classes = ['N (Normal)', 'S (Supraventricular)', 'V (Ventricular)', 'F (Fusion)', 'Q (Unknown)']
    
    print("\n[Input 1] Morphology:  1x360 tensor (normalized ECG beat)")
    print("[Input 2] RR History:  5-element tensor (local rhythm context)")
    print(f"\n[Model Prediction]:    {classes[pred_class]}")
    print("\n[Class Probabilities]:")
    for c, p in zip(classes, probs):
        print(f"   {c:22} {p*100:5.1f}%")
    print("="*60)

if __name__ == "__main__":
    run_demo()
