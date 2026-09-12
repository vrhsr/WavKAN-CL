"""
wavkan_pcwi.py  —  WavKAN with Physiology-Constrained Wavelet Initialization

PRIMARY NOVEL CONTRIBUTION
==========================
Instead of random uniform initialization of wavelet parameters (μ, γ), we
exploit known ECG morphological priors to define a physiologically structured
function space before any gradient updates.

Theoretical Framing
-------------------
KAN ≈ learnable function approximation over input features.
Each output channel k learns a function φ_k(x) = Σ_j w_{k,j} ψ((x_j - μ_{k,j})/γ_{k,j}).
The initialization of (μ, γ) defines the *coverage* of the function space at t=0.

  Random init  → isotropic coverage, no physiological alignment
  PCWI         → structured coverage: QRS-resolving channels initialized
                 with sharp (small γ) wavelets; P/T channels with smooth (large γ)

ECG Morphological Priors (360 Hz, 360-sample beat centred at R-peak)
---------------------------------------------------------------------
  Component   Amplitude μ   Scale γ    Output Channels
  QRS         ~0 (baseline) 0.05 (sharp)   0 – 31
  P-wave      ~ –0.10 mV    0.12 (smooth)  32 – 47
  T-wave      ~ +0.10 mV    0.10 (smooth)  48 – 63

μ is in units of normalised signal amplitude (z-scored ~N(0,1) input).
γ controls wavelet sharpness; smaller γ = narrower, more transient-sensitive.

Reference: Addison (2005), Bozorgasl & Chen (2024), Liu et al. (2024)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# ECG Morphological Priors
# ---------------------------------------------------------------------------
ECG_PRIORS = {
    "QRS": {
        "channels": (0, 32),
        "mu_center": 0.0,   # QRS crosses baseline
        "mu_noise":  0.02,
        "gamma":     0.05,  # Sharp transient → narrow wavelet
        "gamma_jitter": 0.005,
    },
    "P": {
        "channels": (32, 48),
        "mu_center": -0.10, # P-wave slightly below zero on normalised signal
        "mu_noise":  0.03,
        "gamma":     0.12,  # Slower deflection → wider wavelet
        "gamma_jitter": 0.01,
    },
    "T": {
        "channels": (48, 64),
        "mu_center": 0.10,  # T-wave positive deflection
        "mu_noise":  0.03,
        "gamma":     0.10,  # Medium-width
        "gamma_jitter": 0.01,
    },
}


# ---------------------------------------------------------------------------
# Prior-assignment controls
# ---------------------------------------------------------------------------
# Removing PCWI entirely (use_pcwi=False) compares a structured, three-group,
# multi-scale initialisation against a single isotropic one.  That comparison
# cannot separate two explanations of any benefit:
#
#   (A) the priors carry genuine ECG morphology information, or
#   (B) three groups initialised at three *different* scales simply beat one
#       isotropic scale, for which any three distinct (mu, gamma) pairs would do.
#
# These controls isolate (A).  Each keeps the channel partition, the exact set
# of prior values, and every other training detail identical, and changes only
# *which* group receives *which* prior.  If the physiological assignment is not
# better than a permuted one, the benefit is structural, not morphological, and
# the wording in the manuscript must say so.
#
#   physiological : QRS->QRS, P->P, T->T          (the proposed assignment)
#   swap_pt       : QRS->QRS, P->T,   T->P        (cleanest: P and T blocks are
#                                                  both 16 channels, so block
#                                                  size is exactly preserved)
#   cyclic        : QRS->P,   P->T,   T->QRS      (permutes all three; note the
#                                                  32/16/16 block sizes mean the
#                                                  number of channels receiving
#                                                  each prior changes)
#
# swap_pt is the primary control because it is size-preserving and therefore
# varies nothing but the morphology-to-group mapping.
PRIOR_ASSIGNMENTS = {
    "physiological": {"QRS": "QRS", "P": "P",   "T": "T"},
    "swap_pt":       {"QRS": "QRS", "P": "T",   "T": "P"},
    "cyclic":        {"QRS": "P",   "P": "T",   "T": "QRS"},
}


# ---------------------------------------------------------------------------
# Core Layer
# ---------------------------------------------------------------------------

class PCWIWavKANLinear(nn.Module):
    """
    WavKAN linear layer with Physiology-Constrained Wavelet Initialization.

    Supports:
      - mexican_hat  (primary basis — physiologically grounded for QRS)
      - morlet       (ablation baseline)
      - dog          (ablation baseline)
      - b_spline     (AAMI KAN baseline — no physiological basis)

    Args:
        in_features  : Input dim  (360 for raw ECG beat)
        out_features : Output dim (must be 64 when use_pcwi=True)
        wavelet_type : Basis function string
        use_pcwi     : Enable PCWI (False = random init for ablation)
        residual_w   : Weight for linear residual path (improves gradient flow;
                       set 0 to disable — matches original WavKAN design)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        wavelet_type: str = "mexican_hat",
        use_pcwi: bool = True,
        residual_w: float = 0.1,
        prior_assignment: str = "physiological",
    ):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.wavelet_type = wavelet_type
        self.use_pcwi     = use_pcwi
        self.residual_w   = residual_w
        if prior_assignment not in PRIOR_ASSIGNMENTS:
            raise ValueError(
                f"prior_assignment must be one of {sorted(PRIOR_ASSIGNMENTS)}, "
                f"got {prior_assignment!r}")
        self.prior_assignment = prior_assignment

        # Wavelet edge parameters  (out_features, in_features)
        self.weights     = nn.Parameter(torch.empty(out_features, in_features))
        self.translation = nn.Parameter(torch.empty(out_features, in_features))
        self.scale       = nn.Parameter(torch.empty(out_features, in_features))

        # Optional linear residual for gradient flow (learnable scalar gate)
        if residual_w > 0:
            self.linear_w = nn.Parameter(torch.empty(out_features, in_features))
        else:
            self.linear_w = None

        self._reset_parameters()

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.linear_w is not None:
            nn.init.kaiming_uniform_(self.linear_w, a=math.sqrt(5))

        if self.use_pcwi and self.out_features == 64:
            self._apply_pcwi()
        else:
            # Random init — ablation / small layers
            nn.init.uniform_(self.translation, -0.3, 0.3)
            nn.init.uniform_(self.scale, 0.05, 0.20)

    def _apply_pcwi(self):
        """
        Physiology-Constrained Wavelet Initialization.

        For each ECG-component group:
          1. μ  initialised around the expected normalised signal amplitude,
             with small per-channel noise to break channel symmetry.
          2. γ  initialised to the physiologically appropriate sharpness;
             jitter breaks edge symmetry without losing the prior.
        """
        mapping = PRIOR_ASSIGNMENTS[self.prior_assignment]
        with torch.no_grad():
            for group, _ in ECG_PRIORS.items():
                # Channel block always comes from the group; the prior *values*
                # come from whichever group the assignment maps it to.  With
                # "physiological" these are the same, so behaviour is unchanged.
                c0, c1 = ECG_PRIORS[group]["channels"]
                prior  = ECG_PRIORS[mapping[group]]
                nc     = c1 - c0

                # Translation (μ): centre + noise
                mu_noise = prior["mu_noise"]
                mu_init  = torch.empty(nc, self.in_features).uniform_(
                    prior["mu_center"] - mu_noise,
                    prior["mu_center"] + mu_noise,
                )
                self.translation.data[c0:c1, :] = mu_init

                # Scale (γ): prior + small jitter, clamped positive
                gj          = prior["gamma_jitter"]
                jitter      = torch.empty(nc, self.in_features).uniform_(-gj, gj)
                gamma_init  = (prior["gamma"] + jitter).clamp(min=0.01)
                self.scale.data[c0:c1, :] = gamma_init

    # ------------------------------------------------------------------
    # Basis Functions
    # ------------------------------------------------------------------

    @staticmethod
    def _mexican_hat(x: torch.Tensor) -> torch.Tensor:
        """(1 − x²) exp(−x²/2)  — 2nd derivative of Gaussian."""
        return (1.0 - x.pow(2)) * torch.exp(-0.5 * x.pow(2))

    @staticmethod
    def _morlet(x: torch.Tensor) -> torch.Tensor:
        """cos(5x) exp(−x²/2)."""
        return torch.cos(5.0 * x) * torch.exp(-0.5 * x.pow(2))

    @staticmethod
    def _dog(x: torch.Tensor) -> torch.Tensor:
        """−x exp(−x²/2)  — 1st derivative of Gaussian."""
        return -x * torch.exp(-0.5 * x.pow(2))

    @staticmethod
    def _b_spline(x: torch.Tensor) -> torch.Tensor:
        """Cubic B-spline (piecewise polynomial KAN baseline)."""
        a   = x.abs()
        m1  = (a < 1).float()
        m2  = ((a >= 1) & (a < 2)).float()
        v1  = (2.0 / 3.0) - a.pow(2) + 0.5 * a.pow(3)
        v2  = (1.0 / 6.0) * (2.0 - a).pow(3)
        return m1 * v1 + m2 * v2

    def _apply_basis(self, x_norm: torch.Tensor) -> torch.Tensor:
        wt = self.wavelet_type
        if wt == "mexican_hat":
            return self._mexican_hat(x_norm)
        elif wt == "morlet":
            return self._morlet(x_norm)
        elif wt == "dog":
            return self._dog(x_norm)
        elif wt == "b_spline":
            return self._b_spline(x_norm)
        else:
            return self._mexican_hat(x_norm)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:  x : (B, in_features)
        Returns: y : (B, out_features)
        """
        # (B, 1, F) − (out, F) → broadcast → (B, out, F)
        x_exp   = x.unsqueeze(1)
        gamma   = self.scale.abs() + 1e-6
        x_norm  = (x_exp - self.translation) / gamma

        phi = self._apply_basis(x_norm)                  # (B, out, F)
        y   = (phi * self.weights).sum(dim=2)            # (B, out)

        # Residual linear path (improves gradient flow through deep stacks)
        if self.linear_w is not None and self.residual_w > 0:
            y = y + self.residual_w * F.linear(x, self.linear_w)

        return y

    # ------------------------------------------------------------------
    # Interpretability helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def get_component_scales(self) -> dict:
        """Returns mean learned |γ| per ECG-component group."""
        if self.out_features != 64:
            return {}
        return {
            comp: self.scale.data[p["channels"][0]: p["channels"][1]].abs().mean().item()
            for comp, p in ECG_PRIORS.items()
        }

    @torch.no_grad()
    def get_component_translations(self) -> dict:
        """Returns mean learned μ per ECG-component group."""
        if self.out_features != 64:
            return {}
        return {
            comp: self.translation.data[p["channels"][0]: p["channels"][1]].mean().item()
            for comp, p in ECG_PRIORS.items()
        }

    @torch.no_grad()
    def pcwi_drift(self) -> dict:
        """
        Measures drift of learned (μ, γ) from physiological priors.
        Used in the Wavelet-ECG Alignment Score (WAS).
        Returns absolute drift for each component.
        """
        if self.out_features != 64:
            return {}
        result = {}
        for comp, prior in ECG_PRIORS.items():
            c0, c1 = prior["channels"]
            mu_learned    = self.translation.data[c0:c1].mean().item()
            gamma_learned = self.scale.data[c0:c1].abs().mean().item()
            result[comp] = {
                "mu_drift":    abs(mu_learned    - prior["mu_center"]),
                "gamma_drift": abs(gamma_learned - prior["gamma"]),
            }
        return result
