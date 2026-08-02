import torch
import torch.nn as nn
import math


class KANLayer(nn.Module):
    """Kolmogorov-Arnold Network layer with B-spline activations.

    Learns activation functions via B-spline basis instead of fixed activations.
    Output = SiLU(W_base * x) + sum(c_i * B_i(x))
    """

    def __init__(self, in_dim: int, out_dim: int, grid_size: int = 5,
                 spline_order: int = 3, base_activation: nn.Module = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.grid_size = grid_size
        self.spline_order = spline_order

        # Base weight with SiLU residual connection
        self.base_linear = nn.Linear(in_dim, out_dim)
        self.base_activation = base_activation or nn.SiLU()

        # Spline coefficients: (out_dim, in_dim, grid_size + spline_order)
        self.spline_coeffs = nn.Parameter(
            torch.zeros(out_dim, in_dim, grid_size + spline_order)
        )

        # Grid points uniformly initialized on [-1, 1]
        grid = torch.linspace(-1, 1, grid_size + spline_order)
        self.register_buffer("grid", grid.repeat(in_dim, out_dim, 1))

        nn.init.kaiming_uniform_(self.spline_coeffs, a=math.sqrt(5) * 0.1)
        self._init_base()

    def _init_base(self):
        nn.init.kaiming_uniform_(self.base_linear.weight, a=math.sqrt(5))
        nn.init.zeros_(self.base_linear.bias)

    def _b_splines(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis functions of given order.

        Args:
            x: (B, in_dim) input values in [-1, 1]
        Returns:
            (B, out_dim) evaluated spline values
        """
        B, _ = x.shape

        # Rescale x from [-1, 1] to match grid range
        x = x.unsqueeze(-1)  # (B, in_dim, 1)

        # Initialize order-0 B-splines (piecewise constant)
        grid = self.grid  # (in_dim, out_dim, G+S)
        bases = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).float()
        # For x == upper bound
        bases[:, :, -1] = bases[:, :, -1] + (x.squeeze(-1) >= grid[:, :, -1]).float()

        # Cox-de Boor recursion for higher orders
        for k in range(1, self.spline_order + 1):
            left = grid[:, :, :grid.size(-1) - k]
            right = grid[:, :, k:]

            denom_left = grid[:, :, k:] - left
            denom_right = right - grid[:, :, :-k]

            term_left = torch.zeros_like(denom_left)
            term_right = torch.zeros_like(denom_right)

            mask_left = denom_left > 1e-8
            mask_right = denom_right > 1e-8

            term_left[mask_left] = ((x - grid[:, :, :-k].unsqueeze(0)) * bases[:, :, :-1])[
                mask_left.unsqueeze(0)
            ].reshape(-1)[:mask_left.sum()] / denom_left[mask_left]

            term_right[mask_right] = ((grid[:, :, k:].unsqueeze(0) - x) * bases[:, :, 1:])[
                mask_right.unsqueeze(0)
            ].reshape(-1)[:mask_right.sum()] / denom_right[mask_right]

            bases = term_left + term_right

        # Weighted sum over spline coefficients
        # bases: (B, in_dim, out_dim, G) roughly
        # coeffs: (out_dim, in_dim, G+S)
        # Need to align dimensions properly
        spline_out = torch.einsum("bij,oij->bo", bases.permute(0, 2, 1), self.spline_coeffs)
        return spline_out

    def _b_splines_simple(self, x: torch.Tensor) -> torch.Tensor:
        """Simplified B-spline evaluation using tensor operations.

        For training stability, approximate B-spline basis as a learnable
        nonlinear transform of the input, inspired by the original KAN paper.
        """
        B, in_dim = x.shape

        # Evaluate basis functions per input dimension
        grid_start = self.grid[0, 0]  # (G+S,)
        device = x.device

        # Normalize x to grid range
        x_clipped = torch.clamp(x, -0.99, 0.99)

        # Compute distance from each grid point
        x_expand = x_clipped.unsqueeze(-1)  # (B, in_dim, 1)
        grid_expand = grid_start.view(1, 1, -1)  # (1, 1, G+S)

        diff = x_expand - grid_expand  # (B, in_dim, G+S)

        # Quadratic B-spline-like basis
        if self.spline_order == 3:
            basis = torch.exp(-2.0 * diff ** 2)
        else:
            basis = torch.relu(1.0 - torch.abs(diff) * (self.grid_size + 1) / 2)

        # Contract over input dimensions using spline coefficients
        # basis: (B, in_dim, G+S), coeffs: (out_dim, in_dim, G+S)
        spline_out = torch.einsum("big,oig->bo", basis, self.spline_coeffs)
        return spline_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x) * 5.0
        base_out = self.base_activation(self.base_linear(x))
        spline_out = self._b_splines_simple(x)
        return base_out + spline_out


class KANHead(nn.Module):
    """Two-layer KAN classification head."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 grid_size: int = 5, spline_order: int = 3, dropout: float = 0.1):
        super().__init__()
        self.layer1 = KANLayer(in_dim, hidden_dim, grid_size, spline_order)
        self.dropout = nn.Dropout(dropout)
        self.layer2 = KANLayer(hidden_dim, out_dim, grid_size, spline_order)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x
