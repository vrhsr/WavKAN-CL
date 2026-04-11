"""Quick smoke test for WavKAN-v2 architecture."""
import sys
sys.path.insert(0, '.')
import torch
from models.wavkan_v2 import WavKAN_v2

# Full model
m = WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=True)
print(f"Full WavKAN-v2 params: {m.count_parameters():,}")

# Ablation variants
variants = [
    ("no_pcwi",    {"use_pcwi": False}),
    ("no_pwam",    {"use_pwam": False}),
    ("b_spline",   {"wavelet_type": "b_spline"}),
    ("no_rr_attn", {"use_rr_attn": False}),
]
for name, kw in variants:
    mv = WavKAN_v2(**kw)
    print(f"  [{name}]: {mv.count_parameters():,} params")

# Forward pass smoke test
x   = torch.randn(4, 360)
xrr = torch.randn(4, 5)
out = m(x, xrr)
print(f"Forward pass: input={tuple(x.shape)} -> output={tuple(out.shape)}")
assert out.shape == (4, 5), "Wrong output shape!"

# PCWI drift check
drift = m.get_wavelet_alignment()
print("PCWI drift (post-init):")
for comp, vals in drift.items():
    print(f"  {comp}: mu_drift={vals['mu_drift']:.4f}  gamma_drift={vals['gamma_drift']:.4f}")

# Component scales
scales = m.get_component_scales()
print("Learned component gamma scales:")
for comp, val in scales.items():
    print(f"  {comp}: {val:.4f}")

# PWAM gate activation
z_kan = m.kan(x)
z_kan = m.kan_norm(z_kan)
z_kan = m.kan_drop(z_kan)
z_gru, _ = m.bigru(z_kan.unsqueeze(1))
z_m = z_gru.squeeze(1)
gate_val = m.pwam.mean_gate_activation(x, z_m)
print(f"PWAM gate activation (random input): {gate_val:.4f}")

print("\nAll checks PASSED.")
