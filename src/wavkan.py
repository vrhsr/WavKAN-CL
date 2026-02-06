import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WavKANLinear(nn.Module):
    def __init__(self, in_features, out_features, wavelet_type='mexican_hat'):
        super(WavKANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type

        # --- Learnable Parameters ---
        self.weights = nn.Parameter(torch.Tensor(out_features, in_features))
        self.translation = nn.Parameter(torch.Tensor(out_features, in_features))
        self.scale = nn.Parameter(torch.Tensor(out_features, in_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        nn.init.uniform_(self.translation, -0.5, 0.5)
        nn.init.uniform_(self.scale, 0.5, 1.5)

    def mexican_hat(self, x):
        # Mexican Hat (2nd Derivative of Gaussian)
        return (1 - x**2) * torch.exp(-0.5 * x**2)

    def morlet(self, x):
        # Morlet Wavelet (approx: cos(5x) * Gaussian)
        # 5.0 is a typical frequency constant for Morlet
        return torch.cos(5.0 * x) * torch.exp(-0.5 * x**2)

    def dog(self, x):
        # DOG: Derivative of Gaussian (1st Derivative)
        # -x * exp(-x^2/2)
        return -x * torch.exp(-0.5 * x**2)

    def forward(self, x):
        x_expanded = x.unsqueeze(1)
        x_norm = (x_expanded - self.translation) / (self.scale + 1e-8)
        
        if self.wavelet_type == 'mexican_hat':
            basis_func = self.mexican_hat(x_norm)
        elif self.wavelet_type == 'morlet':
            basis_func = self.morlet(x_norm)
        elif self.wavelet_type == 'dog':
            basis_func = self.dog(x_norm)
        else:
            # Default to Mexican Hat
            basis_func = self.mexican_hat(x_norm)

        y = torch.sum(basis_func * self.weights, dim=2)
        return y

class WavKANClassifier(nn.Module):
    def __init__(self, input_size=360, num_classes=5, wavelet_type='mexican_hat'):
        super(WavKANClassifier, self).__init__()
        
        self.kan1 = WavKANLinear(input_size, 64, wavelet_type=wavelet_type)
        self.ln1 = nn.LayerNorm(64)
        
        self.kan2 = WavKANLinear(64, 32, wavelet_type=wavelet_type)
        self.ln2 = nn.LayerNorm(32)
        
        self.head = nn.Linear(32, num_classes)

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)
            
        x = self.kan1(x)
        x = self.ln1(x)
        x = self.kan2(x)
        x = self.ln2(x)
        x = self.head(x)
        return x
