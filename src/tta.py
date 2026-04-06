"""
Test-Time Augmentation (TTA) for ECG Classification
====================================================
Instead of a single prediction, we augment the test beat 3 ways,
predict on all, and average the probabilities.

This typically adds +1-2% Macro F1 for free (no retraining).
"""
import torch
import numpy as np

def tta_predict(model, x, device='cpu'):
    """
    Perform Test-Time Augmentation on a batch of ECG beats.
    
    Args:
        model: Trained PyTorch model (in eval mode)
        x: Tensor of shape (Batch, 360) - raw ECG beats
        device: 'cpu' or 'cuda'
    
    Returns:
        avg_probs: Tensor of shape (Batch, 5) - averaged probabilities
    """
    model.eval()
    x = x.to(device)
    
    with torch.no_grad():
        # 1. Original prediction
        p1 = torch.softmax(model(x), dim=1)
        
        # 2. Shifted prediction (roll by 5 samples ~ 14ms @ 360Hz)
        x_shift = torch.roll(x, shifts=5, dims=1)
        p2 = torch.softmax(model(x_shift), dim=1)
        
        # 3. Scaled prediction (slight amplitude change)
        x_scale = x * 1.05
        p3 = torch.softmax(model(x_scale), dim=1)
        
        # 4. Flipped prediction (inverted polarity - sometimes helps)
        # x_flip = -x
        # p4 = torch.softmax(model(x_flip), dim=1)
        
        # Average probabilities
        avg_probs = (p1 + p2 + p3) / 3.0
    
    return avg_probs

def tta_evaluate(model, test_loader, device='cpu'):
    """
    Evaluate model with TTA on entire test set.
    
    Args:
        model: Trained model
        test_loader: DataLoader for test set
        device: 'cpu' or 'cuda'
    
    Returns:
        all_preds: numpy array of predictions
        all_labels: numpy array of true labels
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    for X_batch, y_batch in test_loader:
        probs = tta_predict(model, X_batch, device)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.numpy())
    
    return np.array(all_preds), np.array(all_labels)


if __name__ == "__main__":
    # Quick test
    print("TTA Module loaded successfully.")
    print("Usage: from src.tta import tta_evaluate")
