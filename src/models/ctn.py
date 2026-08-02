import torch
import torch.nn as nn
from .performer import PerformerEncoder
from .mamba_block import MambaEncoder
from .sticker_attention import StickerAttentionPooling


class MultiHeadAttentionPooling(nn.Module):
    """Learnable query tokens attend to residue features via cross-attention.

    Produces both a fixed-size protein representation and per-residue
    importance scores for interpretability.
    """

    def __init__(self, d_model: int, n_queries: int = 4, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.n_queries = n_queries
        self.queries = nn.Parameter(torch.randn(1, n_queries, d_model))
        self.cross_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        nn.init.xavier_uniform_(self.queries)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """Args:
            x: (B, T, d_model) residue features
            mask: (B, T) float attention mask (1=valid, 0=pad)
        Returns:
            pooled: (B, n_queries * d_model) flattened attended features
            attn_weights: (B, n_queries, T) per-residue importance per query
        """
        B = x.shape[0]
        queries = self.queries.expand(B, -1, -1)  # (B, n_queries, d_model)
        key_mask = None if mask is None else (mask == 0)
        attn_out, attn_weights = self.cross_attn(
            queries, x, x, key_padding_mask=key_mask
        )
        return attn_out.flatten(1, 2), attn_weights


class MultiKernelConv1D(nn.Module):
    """Parallel 1D convolutions with multiple kernel sizes for multi-scale motif extraction."""

    def __init__(self, in_channels: int = 1280, filters: int = 256,
                 kernel_sizes: list = None, dropout: float = 0.2):
        super().__init__()
        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7, 9]

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels, filters, k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.bn = nn.BatchNorm1d(filters * len(kernel_sizes))
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.out_channels = filters * len(kernel_sizes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, in_channels) — ESM-2 embeddings
        Returns:
            (B, T, out_channels) — multi-scale conv features
        """
        x = x.transpose(1, 2)  # (B, C_in, T) for Conv1D
        features = [conv(x) for conv in self.convs]  # list of (B, C_f, T)
        x = torch.cat(features, dim=1)  # (B, C_out, T)
        x = self.bn(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)  # (B, T, C_out)
        return x


class CTN(nn.Module):
    """Convolutional-Transformer Network: Conv1D → Performer → Pooling."""

    def __init__(self, in_channels: int = 1280, conv_filters: int = 256,
                 conv_kernel_sizes: list = None, conv_dropout: float = 0.2,
                 performer_d_model: int = 1024, performer_n_heads: int = 8,
                 performer_n_layers: int = 2, performer_nb_features: int = 256,
                 performer_kernel_type: str = "relu", performer_ffn_ratio: int = 4,
                 performer_dropout: float = 0.1, output_dim: int = 512,
                 use_performer: bool = True, use_mamba: bool = False,
                 mamba_d_state: int = 16, mamba_d_conv: int = 4,
                 mamba_expand: int = 2,
                 use_attention_pooling: bool = False,
                 pooling_n_queries: int = 4, pooling_n_heads: int = 8,
                 use_sticker_pooling: bool = False):
        super().__init__()

        if conv_kernel_sizes is None:
            conv_kernel_sizes = [3, 5, 7, 9]

        self.use_performer = use_performer
        self.use_mamba = use_mamba
        self.use_attention_pooling = use_attention_pooling
        self.use_sticker_pooling = use_sticker_pooling
        self.performer_d_model = performer_d_model

        self.conv = MultiKernelConv1D(
            in_channels=in_channels, filters=conv_filters,
            kernel_sizes=conv_kernel_sizes, dropout=conv_dropout,
        )

        # Project conv output to encoder input dim if needed
        seq_dim = performer_d_model
        conv_out = conv_filters * len(conv_kernel_sizes)
        if conv_out != seq_dim:
            self.input_proj = nn.Linear(conv_out, seq_dim)
        else:
            self.input_proj = nn.Identity()

        if use_mamba:
            self.encoder = MambaEncoder(
                d_model=seq_dim, n_layers=performer_n_layers,
                d_state=mamba_d_state, d_conv=mamba_d_conv,
                expand=mamba_expand, dropout=performer_dropout,
            )
        elif use_performer:
            self.encoder = PerformerEncoder(
                d_model=seq_dim, n_heads=performer_n_heads,
                n_layers=performer_n_layers, nb_features=performer_nb_features,
                kernel_type=performer_kernel_type, ffn_ratio=performer_ffn_ratio,
                dropout=performer_dropout,
            )
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=seq_dim, nhead=performer_n_heads,
                dim_feedforward=seq_dim * performer_ffn_ratio,
                dropout=performer_dropout, activation="gelu",
                batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer, num_layers=performer_n_layers,
            )

        if use_sticker_pooling:
            self.sticker_pool = StickerAttentionPooling(
                d_model=performer_d_model, dropout=performer_dropout,
            )
            pool_out_dim = performer_d_model
            self.attention_pool = None
        elif use_attention_pooling:
            self.attention_pool = MultiHeadAttentionPooling(
                d_model=performer_d_model, n_queries=pooling_n_queries,
                n_heads=pooling_n_heads, dropout=performer_dropout,
            )
            pool_out_dim = performer_d_model * pooling_n_queries
            self.sticker_pool = None
        else:
            self.attention_pool = None
            self.sticker_pool = None
            pool_out_dim = performer_d_model

        self.output_proj = nn.Linear(pool_out_dim, output_dim)
        self.output_norm = nn.LayerNorm(output_dim)
        self.output_dropout = nn.Dropout(conv_dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None):
        """
        Args:
            x: (B, T, in_channels) — residue-level embeddings
            mask: (B, T) float attention mask
        Returns:
            features: (B, output_dim) — sequence-level representation
            attn_weights: (B, n_queries, T) or None — per-residue importance
        """
        x = self.conv(x)  # (B, T, conv_out)
        x = self.input_proj(x)  # (B, T, performer_d_model)

        if self.use_mamba or self.use_performer:
            x = self.encoder(x, mask)
        else:
            key_mask = None if mask is None else (mask == 0)
            x = self.encoder(x, src_key_padding_mask=key_mask)

        attn_weights = None
        if self.use_sticker_pooling:
            x, attn_weights = self.sticker_pool(x, mask)
        elif self.use_attention_pooling:
            x, attn_weights = self.attention_pool(x, mask)
        elif mask is not None:
            x = x * mask.unsqueeze(-1)
            x = x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
        else:
            x = x.mean(dim=1)

        x = self.output_proj(x)
        x = self.output_norm(x)
        x = self.output_dropout(x)
        return x, attn_weights
