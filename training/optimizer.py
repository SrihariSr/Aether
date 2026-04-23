import math
import torch
import torch.nn as nn

def configure_optimizer(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float]=(0.9, 0.95)
) -> torch.optim.Optimizer:
    
    decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() >= 2]
    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.dim() < 2]

    param_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas)

    n_decay = sum(p.numel() for p in decay_params)
    n_no_decay = sum(p.numel() for p in no_decay_params)

    return optimizer

def get_lr(
    step: int,
    warmup_steps: int,
    max_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:

    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    
    if step > max_steps:
        return min_lr
    
    # Cosine decay
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    cos_decay = 0.5 * (1 + math.cos(math.pi * progress))
    
    return min_lr + (max_lr - min_lr) * cos_decay
