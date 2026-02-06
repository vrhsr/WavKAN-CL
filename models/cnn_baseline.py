import torch
import torch.nn as nn

class CNN1D(nn.Module):
    def __init__(self, num_classes=5, input_len=256):
        super(CNN1D, self).__init__()
        
        # Input shape: (Batch, 1, 256)
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2) 
        )
        # Output: (Batch, 32, 128)
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        # Output: (Batch, 64, 64)
        
        self.flatten_dim = 64 * (input_len // 4) # 64 * 64 = 4096
        
        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Length) -> (Batch, 1, Length)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Test shape
    dummy = torch.randn(10, 256)
    model = CNN1D()
    out = model(dummy)
    print(f"Output shape: {out.shape}") # Should be (10, 5)
