import torch
import torch.nn as nn
import math


class PerformerSelfAttention(nn.Module):
    """FAVOR+ (Fast Attention Via Orthogonal Random features) self-attention.

    Approximates softmax attention in O(T) time/space instead of O(T^2).
    """

    def __init__(self, d_model: int = 1024, n_heads: int = 8,
                 nb_features: int = 256, kernel_type: str = "relu",
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.nb_features = nb_features
        self.kernel_type = kernel_type
        self.dropout = dropout

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.register_buffer("projection_matrix", None, persistent=False)
        self.attention_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for m in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)

    def _get_projection_matrix(self, dim: int, device: torch.device) -> torch.Tensor:
        if self.projection_matrix is None or self.projection_matrix.shape[0] != self.nb_features:
            if self.kernel_type == "relu":
                # Orthogonal random features
                q = torch.randn(dim, self.nb_features, device=device)
                q, _ = torch.linalg.qr(q)
                self.projection_matrix = q.T
            else:
                self.projection_matrix = torch.randn(self.nb_features, dim, device=device)
        return self.projection_matrix.to(device)

    def _kernel_fn(self, x: torch.Tensor, projection: torch.Tensor) -> torch.Tensor:
        """Apply kernel feature map: phi(x) = 1/sqrt(m) * relu(x @ W.T)."""
        raw = torch.matmul(x, projection.T)
        if self.kernel_type == "relu":
            raw = torch.relu(raw)
        return raw / math.sqrt(self.nb_features)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: (B, T, d_model)
            mask: (B, T) float, 1 for valid tokens, 0 for padding

        Returns:
            (B, T, d_model)
        """
        B, T, D = x.shape

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, D_h)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        projection = self._get_projection_matrix(self.head_dim, x.device)

        q_prime = self._kernel_fn(q, projection)  # (B, H, T, M)
        k_prime = self._kernel_fn(k, projection)  # (B, H, T, M)

        if mask is not None:
            mask_expanded = mask[:, None, :, None]  # (B, 1, T, 1)
            k_prime = k_prime * mask_expanded
            v = v * mask_expanded

        # D = phi(Q) @ (phi(K)^T @ V)
        kv = torch.matmul(k_prime.transpose(-2, -1), v)  # (B, H, M, D_h)
        z = torch.matmul(q_prime, kv)  # (B, H, T, D_h)

        # Normalize: D_hat = phi(Q) @ (phi(K)^T @ 1)
        k_sum = k_prime.sum(dim=2, keepdim=True)  # (B, H, 1, M)
        norm = torch.matmul(q_prime, k_sum.transpose(-2, -1))  # (B, H, T, 1)
        norm = norm.clamp(min=1e-6)

        out = z / norm
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


class PerformerLayer(nn.Module):
    """Single Performer encoder layer: attention + residual + LN + FFN + residual + LN."""

    def __init__(self, d_model: int = 1024, n_heads: int = 8,
                 nb_features: int = 256, kernel_type: str = "relu",
                 ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.self_attn = PerformerSelfAttention(d_model, n_heads, nb_features,
                                                kernel_type, dropout)
        self.attn_norm = nn.LayerNorm(d_model)
        self.attn_dropout = nn.Dropout(dropout)

        ffn_dim = d_model * ffn_ratio
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        attn_out = self.self_attn(x, mask)
        x = x + self.attn_dropout(attn_out)
        x = self.attn_norm(x)

        # FFN with residual
        ffn_out = self.ffn(x)
        x = x + self.ffn_dropout(ffn_out)
        x = self.ffn_norm(x)
        return x


class PerformerEncoder(nn.Module):
    """Stacked Performer encoder with optional length-based standard attention fallback."""

    def __init__(self, d_model: int = 1024, n_heads: int = 8, n_layers: int = 2,
                 nb_features: int = 256, kernel_type: str = "relu",
                 ffn_ratio: int = 4, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            PerformerLayer(d_model, n_heads, nb_features, kernel_type, ffn_ratio, dropout)
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return self.output_norm(x)
