import os
import time
from typing import Optional
import torch
from dataclasses import asdict

from model.config import GPTConfig
from model.transformer import GPT

def save_checkpoint(
    path: str,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    config: GPTConfig,
    step: int,
    val_loss: float,
) -> None:

    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model":     model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config":    asdict(config),
            "step":      step,
            "val_loss":  val_loss,
        },
        path,
    )

def load_checkpoint(
    path: str,
    model: GPT,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cpu",
) -> dict:

    ckpt = torch.load(path, map_location=device)

    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt

class Timer:

    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time() - self.start