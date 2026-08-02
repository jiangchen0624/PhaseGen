"""Sticker-aware attention pooling for protein phase separation prediction.

Based on the sticker-spacer framework (Martin et al. 2020, Science):
aromatic residues (F, Y, W) are the primary "sticker" residues that drive
phase separation via pi-pi and cation-pi interactions.

This module learns to attend to sticker-like residues directly from ProtT5
embeddings, producing both a weighted protein representation and per-residue
importance scores for interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StickerAttentionPooling(nn.Module):
    """Learn per-residue stickerness from embeddings and pool via weighted mean.

    Architecture:
        embedding (d_model) → MLP → stickerness score (1) → sigmoid
        pooled = sum(stickerness * embedding) / sum(stickerness)

    This is intentionally simpler than cross-attention pooling:
    - Biologically motivated: sticker residues (F/Y/W) have distinct embedding patterns
    - Directly interpretable: stickerness scores = residue importance map
    - Fewer parameters: single MLP instead of n_queries × cross-attention
    """

    def __init__(self, d_model: int, hidden_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """Args:
            x: (B, T, d_model) residue features
            mask: (B, T) float mask (1=valid, 0=pad)

        Returns:
            pooled: (B, d_model) stickerness-weighted mean of residue features
            scores: (B, T) per-residue stickerness scores (for interpretability)
        """
        raw_scores = self.scorer(x).squeeze(-1)  # (B, T)
        if mask is not None:
            raw_scores = raw_scores + (1.0 - mask) * -1e9
        scores = torch.softmax(raw_scores, dim=-1)  # (B, T), sums to 1 per protein
        pooled = torch.sum(x * scores.unsqueeze(-1), dim=1)  # (B, d_model)
        return pooled, scores


class SeqFeatureEmbedder(nn.Module):
    """Embed 6 scalar biophysical features into a dense vector.

    Embeds each scalar independently then fuses with a small MLP,
    similar to IDREmbedder but for global sequence-level features.
    """

    def __init__(self, n_features: int = 6, embed_dim: int = 16,
                 output_dim: int = 96, dropout: float = 0.2):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Sequential(nn.Linear(1, embed_dim), nn.GELU())
            for _ in range(n_features)
        ])
        self.fusion = nn.Sequential(
            nn.Linear(n_features * embed_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, features: torch.Tensor):
        """Args:
            features: (B, n_features) — biophysical scalar features
        Returns:
            (B, output_dim) embedded feature vector
        """
        embedded = []
        for i, emb in enumerate(self.embeddings):
            embedded.append(emb(features[:, i:i + 1]))
        concat = torch.cat(embedded, dim=-1)  # (B, n_features * embed_dim)
        return self.fusion(concat)
