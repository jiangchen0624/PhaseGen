import torch
import torch.nn as nn


class IDREmbedder(nn.Module):
    """Embeds 5 ALBATROSS-predicted IDR conformational properties.

    Supports two modes:
    - per_residue (legacy): embeds each residue's (5,) vector → (L, out_dim) — wasteful
      when ALBATROSS predicts global (per-protein) properties expanded to all residues.
    - global (recommended): pools (B, T, 5) → (B, 5), embeds → (B, out_dim).
      Injects after CTN pooling since IDR properties are global protein features.
    """

    def __init__(self, property_dim: int = 66, dropout: float = 0.1,
                 mode: str = "global", global_out_dim: int = 128):
        super().__init__()
        self.mode = mode
        self.property_dim = property_dim

        if mode == "per_residue":
            self.out_dim = 5 * property_dim
            self.embeddings = nn.ModuleList([
                nn.Sequential(nn.Linear(1, property_dim), nn.LayerNorm(property_dim), nn.ReLU())
                for _ in range(5)
            ])
        else:
            self.out_dim = global_out_dim
            self.global_mlp = nn.Sequential(
                nn.Linear(5, global_out_dim),
                nn.LayerNorm(global_out_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )

        self.dropout = nn.Dropout(dropout)

    def forward(self, albatross_features: torch.Tensor) -> torch.Tensor:
        """Args:
            albatross_features: (B, T, 5) or (B, 5) — IDR properties
        Returns:
            Per-residue mode: (B, T, out_dim). Global mode: (B, out_dim).
        """
        if self.mode == "global":
            if albatross_features.dim() == 3:
                x = albatross_features.mean(dim=1)  # (B, 5)
            else:
                x = albatross_features
            return self.global_mlp(x)

        embedded = []
        for i, embedder in enumerate(self.embeddings):
            prop = albatross_features[..., i:i + 1]
            embedded.append(embedder(prop))
        out = torch.cat(embedded, dim=-1)
        return self.dropout(out)
