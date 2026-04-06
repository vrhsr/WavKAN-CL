import torch
import torch.nn as nn
from models.wavkan import WavKANConv1d

class WavKAN_BiGRU(nn.Module):
    def __init__(self, input_channels=1, num_classes=5, gru_hidden=64):
        super(WavKAN_BiGRU, self).__init__()
        
        # 1. WavKAN Feature Extractor
        # Input: (Batch, 1, 256)
        self.wav1 = WavKANConv1d(1, 32, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(2) # 256 -> 128
        
        self.wav2 = WavKANConv1d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(2) # 128 -> 64
        
        # Output shape: (Batch, 64, 64)
        # i.e. 64 channels, sequence length 64
        
        # 2. BiGRU
        self.gru = nn.GRU(
            input_size=64, # Matches Channel dimension if we permute? Or Input size per step?
            # Standard: (Batch, Seq, Features)
            # Here: Seq=64 (Time), Features=64 (Channels)
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.fc = nn.Linear(gru_hidden * 2, num_classes)
        
    def forward(self, x):
        # x: (Batch, 256) -> (Batch, 1, 256)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.wav1(x)
        x = self.bn1(x)
        x = self.pool1(x)
        
        x = self.wav2(x)
        x = self.bn2(x)
        x = self.pool2(x)
        
        # Prepare for GRU
        # Current: (Batch, Channels, Time) -> (Batch, 64, 64)
        # GRU expects (Batch, Time, Features)
        x = x.permute(0, 2, 1) # (Batch, 64, 64)
        
        out, _ = self.gru(x) 
        # out: (Batch, Time, Hidden * 2)
        
        # Global Average Pooling or Last Step? 
        # BiGRU last step logic is tricky (fwd last + bwd last).
        # Usually easier to pool or take mean.
        x = torch.mean(out, dim=1) # (Batch, Hidden * 2)
        
        x = self.fc(x)
        return x

if __name__ == "__main__":
    model = WavKAN_BiGRU()
    dummy = torch.randn(10, 256)
    out = model(dummy)
    print(f"Hybrid Output: {out.shape}")
