"""
wavkan_v2.py  —  Complete Upgraded WavKAN-CL Architecture

Architecture Stack (in order)
==============================
  1. WavKAN Backbone       — PCWIWavKANLinear (360 → 64, PCWI-initialised)
  2. LayerNorm + Dropout
  3. BiGRU Temporal Context — (64 → 64 bidirectional) morphological embedding z_m
  4. PWAM                  — P-Wave Attention Module (enhances S-class sensitivity)
  5. RR-Interval Branch    — 5-beat history with lightweight self-attention
  6. Fusion Classifier     — concatenated (z_m=64, z_rr=16) → (80 → 48 → 5)

Parameter Budget (approx.)
==========================
  WavKAN layer             : 360×64×3 params  = 69,120
  BiGRU (bidirectional)    : 2×(64×32×3+32×32×3+32×3) ≈ 25,440
  PWAM                     : P_encoder+proj+attn+gate  ≈ 8,500
  RR branch                : 5→32→16 MLP + attn       ≈ 2,200
  Classifier               : 80→48→5                  ≈ 4,085
  ───────────────────────────────────────────────────────────
  Total                    ≈ 109,345  (still < 120 K, Green AI compliant)

Ablation Flags
==============
  use_pcwi    : bool → enables PCWI vs random init
  use_pwam    : bool → enables P-Wave Attention Module
  use_rr_attn : bool → enables attention-weighted RR fusion vs plain MLP
  wavelet_type: str  → 'mexican_hat' | 'morlet' | 'dog' | 'b_spline'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.wavkan_pcwi import PCWIWavKANLinear
from src.pwam import PWaveAttentionModule


# ─────────────────────────────────────────────────────────────────────────────
# RR-Interval Branch with optional self-attention over beat history
# ─────────────────────────────────────────────────────────────────────────────

class RRBranch(nn.Module):
    """
    Encodes 5-beat RR-interval history into a compact rhythm embedding.

    Args:
        use_attention : If True, adds a single-head self-attention over the
                        5 RR scalars before the MLP, enabling the model to
                        dynamically weight earlier vs. later intervals.
                        This is particularly useful for capturing compensatory
                        pause patterns in PACs (S-class discrimination).
    """

    def __init__(self, rr_len: int = 5, out_dim: int = 16, use_attention: bool = True):
        super().__init__()
        self.use_attention = use_attention

        if use_attention:
            # Lightweight self-attention: project scalars to 16-D tokens
            self.rr_proj    = nn.Linear(1, 16)
            self.self_attn  = nn.MultiheadAttention(embed_dim=16, num_heads=2,
                                                     batch_first=True, dropout=0.0)
            self.attn_norm  = nn.LayerNorm(16)
            # Flatten attended tokens (5×16=80) → compact hidden
            self.mlp = nn.Sequential(
                nn.Linear(rr_len * 16, 32), nn.ReLU(),
                nn.Linear(32, out_dim),     nn.ReLU(),
            )
        else:
            # Original MLP-only branch (ablation baseline)
            self.mlp = nn.Sequential(
                nn.Linear(rr_len, 64), nn.ReLU(),
                nn.Linear(64, 32),     nn.ReLU(),
                nn.Linear(32, out_dim), nn.ReLU(),
            )

    def forward(self, x_rr: torch.Tensor) -> torch.Tensor:
        """
        Args:  x_rr : (B, 5)  normalised RR intervals
        Returns:      (B, out_dim)
        """
        if self.use_attention:
            # (B, 5) → (B, 5, 1) → (B, 5, 16)
            tokens = self.rr_proj(x_rr.unsqueeze(-1))
            attn_out, _ = self.self_attn(tokens, tokens, tokens)
            tokens = self.attn_norm(attn_out)
            flat   = tokens.reshape(tokens.size(0), -1)  # (B, 80)
            return self.mlp(flat)
        else:
            return self.mlp(x_rr)


# ─────────────────────────────────────────────────────────────────────────────
# Full WavKAN-v2 Model
# ─────────────────────────────────────────────────────────────────────────────

class WavKAN_v2(nn.Module):
    """
    Upgraded WavKAN-CL model with PCWI + PWAM + RR-Attention.

    Usage:
        model = WavKAN_v2()                    # Full model (paper submission)
        model = WavKAN_v2(use_pcwi=False)      # Ablation: random init
        model = WavKAN_v2(use_pwam=False)      # Ablation: no P-wave attention
        model = WavKAN_v2(wavelet_type='b_spline')  # B-spline baseline
    """

    def __init__(
        self,
        input_size:   int   = 360,
        num_classes:  int   = 5,
        wavelet_type: str   = "mexican_hat",
        use_pcwi:     bool  = True,
        use_pwam:     bool  = True,
        use_rr_attn:  bool  = True,
        rr_len:       int   = 5,
        gru_hidden:   int   = 32,
        dropout:      float = 0.2,
        prior_assignment: str = "physiological",
    ):
        super().__init__()
        self.use_pwam      = use_pwam
        self.wavelet_type  = wavelet_type
        self.use_pcwi      = use_pcwi
        self.prior_assignment = prior_assignment

        # ── 1. WavKAN Backbone ────────────────────────────────────────────────
        self.kan = PCWIWavKANLinear(
            in_features  = input_size,
            out_features = 64,
            wavelet_type = wavelet_type,
            use_pcwi     = use_pcwi,
            residual_w   = 0.1,
            prior_assignment = prior_assignment,
        )
        self.kan_norm = nn.LayerNorm(64)
        self.kan_drop = nn.Dropout(dropout)

        # ── 2. Bidirectional GRU  ─────────────────────────────────────────────
        self.bigru = nn.GRU(
            input_size   = 64,
            hidden_size  = gru_hidden,
            num_layers   = 1,
            batch_first  = True,
            bidirectional= True,
        )
        # Output: 2 * gru_hidden = 64

        # ── 3. P-Wave Attention Module (optional) ─────────────────────────────
        if use_pwam:
            self.pwam = PWaveAttentionModule(
                main_dim   = gru_hidden * 2,  # 64
                p_hidden   = 32,
                attn_heads = 4,
                dropout    = dropout,
            )

        # ── 4. RR-Interval Branch ──────────────────────────────────────────────
        self.rr_branch = RRBranch(
            rr_len       = rr_len,
            out_dim      = 16,
            use_attention= use_rr_attn,
        )

        # ── 5. Fusion Classifier ───────────────────────────────────────────────
        # z_m (64) + z_rr (16) = 80
        self.classifier = nn.Sequential(
            nn.Linear(80, 48),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(48, num_classes),
        )

    # ─────────────────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,       # (B, 360)  raw beat waveform
        x_rr: torch.Tensor,    # (B, 5)    normalised RR history
    ) -> torch.Tensor:
        """Returns raw logits (B, num_classes)."""

        # Morphological branch
        z = self.kan(x)                          # (B, 64)
        z = self.kan_norm(z)
        z = self.kan_drop(z)

        z = z.unsqueeze(1)                       # (B, 1, 64)
        z, _ = self.bigru(z)
        z_m = z.squeeze(1)                       # (B, 64)

        # P-wave attention enhancement
        if self.use_pwam:
            z_m = self.pwam(x, z_m)             # (B, 64) — P-enhanced

        # Rhythm branch
        z_rr = self.rr_branch(x_rr)             # (B, 16)

        # Fusion & classify
        fused  = torch.cat([z_m, z_rr], dim=1)  # (B, 80)
        logits = self.classifier(fused)          # (B, 5)

        return logits

    # ─────────────────────────────────────────────────────────────────────────
    # Utility methods
    # ─────────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def get_wavelet_alignment(self) -> dict:
        """Returns PCWI drift statistics for the Wavelet-ECG Alignment Score."""
        return self.kan.pcwi_drift()

    @torch.no_grad()
    def get_component_scales(self) -> dict:
        """Returns mean learned |γ| per ECG component."""
        return self.kan.get_component_scales()

    def freeze_backbone(self):
        """Freeze WavKAN + BiGRU for few-shot fine-tuning. Only trains classifier."""
        for param in self.kan.parameters():
            param.requires_grad = False
        for param in self.bigru.parameters():
            param.requires_grad = False
        if self.use_pwam:
            for param in self.pwam.parameters():
                param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = WavKAN_v2(use_pcwi=True, use_pwam=True, use_rr_attn=True)
    print(f"WavKAN-v2 parameters : {model.count_parameters():,}")

    x    = torch.randn(8, 360)
    x_rr = torch.randn(8, 5)
    out  = model(x, x_rr)
    print(f"Input  : {x.shape}")
    print(f"Output : {out.shape}")
    print(f"PCWI drift: {model.get_wavelet_alignment()}")
    print(f"Component scales: {model.get_component_scales()}")

    # Ablation variants
    for variant, kwargs in [
        ("no_pcwi",    {"use_pcwi": False}),
        ("no_pwam",    {"use_pwam": False}),
        ("b_spline",   {"wavelet_type": "b_spline"}),
        ("no_rr_attn", {"use_rr_attn": False}),
    ]:
        m = WavKAN_v2(**kwargs)
        print(f"  Variant [{variant}]: {m.count_parameters():,} params")
