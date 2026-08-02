import torch
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class CosineWarmupScheduler:
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer: Optimizer, warmup_steps: int, total_steps: int,
                 min_lr_factor: float = 0.01):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_factor = min_lr_factor
        self.current_step = 0

        # Store base LRs for each param group
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

        # Single LambdaLR scheduler for all groups
        def lr_lambda(step):
            return self._get_factor(step)

        self.scheduler = LambdaLR(optimizer, lr_lambda)

    def _get_factor(self, step: int) -> float:
        if step < self.warmup_steps:
            return float(step + 1) / float(max(1, self.warmup_steps))

        progress = float(step - self.warmup_steps) / float(
            max(1, self.total_steps - self.warmup_steps)
        )
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return max(self.min_lr_factor, cosine_decay)

    def step(self):
        self.scheduler.step()
        self.current_step += 1

    def state_dict(self) -> dict:
        return {
            "scheduler": self.scheduler.state_dict(),
            "current_step": self.current_step,
            "base_lrs": self.base_lrs,
        }

    def load_state_dict(self, state_dict: dict):
        self.scheduler.load_state_dict(state_dict["scheduler"])
        self.current_step = state_dict["current_step"]
