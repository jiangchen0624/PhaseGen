"""Domain-adversarial module for dataset-invariant feature learning.

Gradient Reversal Layer (Ganin & Lempitsky, 2015) + Domain Discriminator.
Placed after CTN pooling, before classification head concatenation,
to encourage ProtT5 features that are invariant to dataset origin
(PPMC vs PhaSePred).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    """Gradient reversal autograd function.

    Forward:  f(x) = x  (identity)
    Backward: grad = -lambda * grad_input
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


class GradientReversalLayer(nn.Module):
    """Gradient Reversal Layer with sigmoid warmup schedule.

    The lambda coefficient controls the strength of domain adversarial
    training. It ramps from lambda_min to lambda_max over warmup_steps
    following the DANN schedule:

        p = min(step / warmup_steps, 1.0)
        lambda = lambda_max * (2 / (1 + exp(-10 * p)) - 1)
    """

    def __init__(self, lambda_max: float = 1.0, warmup_steps: int = 1000):
        super().__init__()
        self.lambda_max = lambda_max
        self.warmup_steps = warmup_steps
        self.current_lambda = 0.0
        self.current_step = 0

    def set_lambda(self, current_step: int):
        """Update lambda based on the sigmoid warmup schedule."""
        self.current_step = current_step
        p = min(current_step / max(self.warmup_steps, 1), 1.0)
        # DANN sigmoid schedule
        self.current_lambda = self.lambda_max * (2.0 / (1.0 + torch.exp(
            torch.tensor(-10.0 * p)).item()) - 1.0)

    def get_lambda(self) -> float:
        return self.current_lambda

    def forward(self, x):
        return GradReverse.apply(x, self.current_lambda)


class DomainDiscriminator(nn.Module):
    """MLP that predicts dataset origin (PPMC=0 vs PhaSePred=1) from features.

    Input: (B, input_dim) — CTN-pooled features before IDR/SeqFeat concat
    Output: (B, 2) — domain logits
    """

    def __init__(self, input_dim: int = 512, hidden_dims=None,
                 dropout: float = 0.2):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DomainAdversarialModule(nn.Module):
    """Wraps GradientReversalLayer + DomainDiscriminator.

    Usage:
        dam = DomainAdversarialModule(input_dim=512)
        ...
        # In training loop:
        dam.grl.set_lambda(global_step)
        loss, acc = dam(ctn_features, domain_labels)
    """

    def __init__(self, input_dim: int = 512, lambda_max: float = 1.0,
                 warmup_steps: int = 1000, discriminator_dropout: float = 0.2):
        super().__init__()
        self.grl = GradientReversalLayer(lambda_max=lambda_max,
                                         warmup_steps=warmup_steps)
        self.discriminator = DomainDiscriminator(
            input_dim=input_dim, dropout=discriminator_dropout)

    def forward(self, features, domain_labels):
        """Compute domain adversarial loss and accuracy.

        Args:
            features: (B, D) CTN-pooled features
            domain_labels: (B,) 0=PPMC, 1=PhaSePred

        Returns:
            dict with "loss" (scalar) and "accuracy" (float)
        """
        reversed_features = self.grl(features)
        domain_logits = self.discriminator(reversed_features)
        domain_loss = F.cross_entropy(domain_logits, domain_labels)
        domain_acc = (domain_logits.argmax(dim=-1) == domain_labels).float().mean()
        return {"loss": domain_loss, "accuracy": domain_acc.item()}
