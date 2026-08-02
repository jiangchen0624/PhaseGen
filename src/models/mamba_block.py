"""Mamba-inspired State Space Model block for protein sequence encoding.

Based on Mamba (Gu & Dao, 2023) — selective structured state space models.
Simplified pure-PyTorch implementation suitable for moderate sequence lengths.
Replaces Performer/Transformer with O(T) linear-time sequence mixing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class S6Scan(nn.Module):
    """Selective diagonal state space scan (S6 core of Mamba).

    Discretizes a continuous-time SSM:
        dx/dt = A x + B u
        y = C x

    with input-dependent Δ (delta) and B, C.
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: int = None):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank or d_model // 16

        # A: diagonal state transition (learned per channel)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()).repeat(d_model, 1))

        # Delta projection: input → per-channel timestep
        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, self.dt_rank), nn.ReLU(),
            nn.Linear(self.dt_rank, d_model),
        )

        # B and C: input → state/emission projections
        self.B_proj = nn.Linear(d_model, d_state, bias=False)
        self.C_proj = nn.Linear(d_model, d_state, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Memory-efficient sequential S6 scan.

        Recurrence: h_t = A_t * h_{t-1} + B_t * u_t  (element-wise over D,N dims).

        Computes A_t, B_t on-the-fly per timestep to keep memory at O(B*D*N)
        instead of O(B*T*D*N). The for-loop overhead is acceptable for T ≤ 1024
        because each iteration is fully vectorized over (B, D, N).
        """
        B, T, D = x.shape
        N = self.d_state

        A = -torch.exp(self.A_log)  # (D, N) — negative eigenvalues for stability
        dt = F.softplus(self.dt_proj(x))  # (B, T, D)
        B_in = self.B_proj(x)  # (B, T, N)
        C = self.C_proj(x)  # (B, T, N)

        h = torch.zeros(B, D, N, device=x.device, dtype=x.dtype)
        outputs = []
        for t in range(T):
            # Compute per-timestep discretization on the fly
            dt_t = dt[:, t]  # (B, D)
            A_t = torch.exp(dt_t.unsqueeze(-1) * A.unsqueeze(0))  # (B, D, N)
            B_t = dt_t.unsqueeze(-1) * B_in[:, t].unsqueeze(1)  # (B, D, N)
            u_t = x[:, t]  # (B, D)

            h = A_t * h + B_t * u_t.unsqueeze(-1)  # (B, D, N)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, D)
            outputs.append(y_t)

        return torch.stack(outputs, dim=1)  # (B, T, D)


class MambaLayer(nn.Module):
    """Single Mamba block: Conv → SSM → Gating → FFN.

    Follows the Mamba architecture:
    1. Input projection (expand)
    2. Conv1d → SiLU (local context)
    3. S6 SSM (long-range dependencies)
    4. SiLU gating with residual
    5. Output projection (contract)
    6. Residual + LayerNorm
    """

    def __init__(self, d_model: int = 512, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        inner_dim = d_model * expand

        self.norm = nn.LayerNorm(d_model)
        self.in_proj = nn.Linear(d_model, inner_dim * 2)

        # Causal conv1d (left-padded for online compatibility)
        self.conv = nn.Conv1d(inner_dim, inner_dim, d_conv,
                              groups=inner_dim, padding=d_conv - 1)

        self.ssm = S6Scan(inner_dim, d_state)
        self.act = nn.SiLU()

        self.out_proj = nn.Linear(inner_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            mask: (B, T) optional valid mask
        Returns:
            (B, T, d_model)
        """
        residual = x
        x = self.norm(x)

        # Input projection + split for gating
        proj = self.in_proj(x)  # (B, T, 2 * inner)
        x_ssm, x_gate = proj.chunk(2, dim=-1)  # (B, T, inner) each

        # Local convolution
        x_ssm = x_ssm.transpose(1, 2)  # (B, C, T)
        x_ssm = self.conv(x_ssm)[:, :, :x_ssm.shape[-1]]  # Trim causal padding
        x_ssm = x_ssm.transpose(1, 2)  # (B, T, C)

        # SSM scan
        x_ssm = self.ssm(x_ssm)  # (B, T, inner_dim)

        # Gating
        x = x_ssm * self.act(x_gate)

        # Output projection
        x = self.out_proj(x)
        x = self.dropout(x)
        return residual + x


class MambaEncoder(nn.Module):
    """Stacked Mamba layers for protein sequence encoding.

    Drop-in replacement for PerformerEncoder — same forward signature.
    """

    def __init__(self, d_model: int = 512, n_layers: int = 2,
                 d_state: int = 16, d_conv: int = 4, expand: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaLayer(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.output_norm(x)
