import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class FocalLoss(nn.Module):
    """Focal Loss: -alpha_t * (1 - p_t)^gamma * log(p_t).

    Down-weights easy examples so the model focuses on hard ones.
    For binary: alpha balances positive/negative class weight.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce)
        # alpha_t: weight for positive class (PS) = alpha, negative = 1-alpha
        # For imbalanced data, set alpha > 0.5 to upweight minority (PS) class
        alpha_t = torch.where(targets == 1, self.alpha, 1.0 - self.alpha)
        focal = alpha_t * (1.0 - pt) ** self.gamma * ce
        return focal.mean()


class MultiTaskLoss(nn.Module):
    """Combined multi-task loss with configurable weights.

    L_total = w_bin * L_binary + w_ter * L_ternary + w_con * L_contrastive
    """

    def __init__(self, binary_weight: float = 1.0, ternary_weight: float = 0.5,
                 contrastive_weight: float = 0.1, contrastive_temperature: float = 0.1,
                 focal_alpha: float = 0.25, focal_gamma: float = 2.0,
                 use_focal: bool = True, contrastive_ramp_epochs: int = 5,
                 label_smoothing: float = 0.0):
        super().__init__()
        self.binary_weight = binary_weight
        self.ternary_weight = ternary_weight
        self.contrastive_weight = contrastive_weight
        self.use_focal = use_focal
        self.contrastive_ramp_epochs = contrastive_ramp_epochs

        if use_focal:
            self.binary_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
            self.ternary_loss = FocalLoss(alpha=0.33, gamma=focal_gamma)
        else:
            self.binary_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
            self.ternary_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        from .supcon import SupConLoss
        self.contrastive_loss = SupConLoss(temperature=contrastive_temperature)
        self.current_epoch = 0

    def _get_contrastive_weight(self) -> float:
        if self.current_epoch >= self.contrastive_ramp_epochs:
            return self.contrastive_weight
        return self.contrastive_weight * (self.current_epoch / self.contrastive_ramp_epochs)

    def forward(self, outputs: Dict[str, torch.Tensor],
                binary_labels: torch.Tensor, ternary_labels: torch.Tensor
                ) -> Dict[str, torch.Tensor]:
        binary_logits = outputs["binary_logits"]
        ternary_logits = outputs["ternary_logits"]
        projection = outputs["projection"]

        loss_binary = self.binary_loss(binary_logits, binary_labels)
        loss_ternary = self.ternary_loss(ternary_logits, ternary_labels)

        con_weight = self._get_contrastive_weight()
        loss_contrastive = self.contrastive_loss(projection, binary_labels)

        loss_total = (
            self.binary_weight * loss_binary
            + self.ternary_weight * loss_ternary
            + con_weight * loss_contrastive
        )

        return {
            "total": loss_total,
            "binary": loss_binary,
            "ternary": loss_ternary,
            "contrastive": loss_contrastive,
        }

    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
