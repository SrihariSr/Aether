import os
import torch
import numpy as np

def get_batch(
    split: str,
    batch_size: int,
    block_size: int,
    device: str,
    data_dir: str = "data/prepared",):

    bin_path = os.path.join(data_dir, f"{split}.bin")
    data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    # Picking random starting position
    ix = np.random.randint(0, len(data) - block_size - 1, size=(batch_size,))
    
    # Input window
    x = torch.stack([
        torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix
    ])
    
    # Target window (shifted the Input window by 1)
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + block_size + 1].astype(np.int64)) for i in ix
    ])

    if device == "cuda": # Pin memory for NVIDIA GPUs
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y
