import torch
import torch.nn as nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Pulls embeddings of same-class samples closer and pushes different-class
    samples apart in the normalized embedding space.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) L2-normalized embeddings
            labels: (B,) integer class labels (binary: 0 or 1)

        Returns:
            scalar loss
        """
        device = features.device
        batch_size = features.shape[0]

        if batch_size < 2:
            return torch.tensor(0.0, device=device)

        # Similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature  # (B, B)

        # Mask: same label = positive pair
        label_mask = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B)

        # Remove self-comparisons from positive mask
        identity_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        pos_mask = label_mask & ~identity_mask

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # For numerical stability: subtract max per row
        sim_max = sim.max(dim=1, keepdim=True)[0]
        sim = sim - sim_max.detach()

        # exp of similarities
        exp_sim = torch.exp(sim)

        # Numerator: sum of exp for positive pairs
        pos_exp = exp_sim * pos_mask.float()
        pos_sum = pos_exp.sum(dim=1)  # (B,)

        # Denominator: sum of exp for all pairs except self
        denom_sum = exp_sim.sum(dim=1) - torch.exp(sim.diag())
        denom_sum = denom_sum.clamp(min=1e-8)

        # Loss per anchor
        loss = -torch.log(pos_sum.clamp(min=1e-8) / denom_sum)
        # Only compute loss for anchors that have at least one positive pair
        valid_mask = pos_mask.sum(dim=1) > 0

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        return loss[valid_mask].mean().clamp(min=0.0)
