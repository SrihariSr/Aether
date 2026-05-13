import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from model.config import aether_medium
from model.transformer import GPT
from data.dataloader import get_batch
from training.optimizer import configure_optimizer, get_lr
from training.utilities import save_checkpoint, Timer

# Training hyperparameters

MAX_LR       = 3e-4
MIN_LR       = 3e-5
WARMUP_STEPS = 600
MAX_STEPS    = 40000
WEIGHT_DECAY = 0.1
GRAD_CLIP    = 1.0

BATCH_SIZE   = 16       # sequences per step
BLOCK_SIZE   = 512      # tokens per sequence

EVAL_INTERVAL = 1000     # validation frequency
EVAL_ITERS    = 50      # validation iterations
LOG_INTERVAL  = 10      # logging frequency

CKPT_DIR      = "checkpoints/medium"

# Device setup
def get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


@torch.no_grad()
def estimate_val_loss(model, device: str, iters: int) -> float:
    # Average loss over 'iters' validation batches
    model.eval()
    losses = torch.zeros(iters)
    for k in range(iters):
        x, y = get_batch("val", BATCH_SIZE, BLOCK_SIZE, device)
        _, loss = model(x, y)
        losses[k] = loss.item()
    model.train()
    return losses.mean().item()

# Main training loop
def main():
    device = get_device()
    print(f"Training Aether-Medium on {device}")

    torch.manual_seed(3407) # The 'all you need' seed
    if device == "cuda":
        torch.cuda.manual_seed(3407)

    # Build the model
    config = aether_medium()
    assert config.block_size == BLOCK_SIZE, \
        "BLOCK_SIZE must match config.block_size"

    model = GPT(config).to(device)
    n_params = model.num_parameters()
    print(f"Model: Aether-Medium ({n_params/1e6:.2f}M non-embedding parameters)")

    # Build the optimiser
    optimizer = configure_optimizer(
        model,
        learning_rate=MAX_LR,
        weight_decay=WEIGHT_DECAY,
    )

    # Resume from checkpoint if available
    resume_path = os.path.join(CKPT_DIR, "best.pt")
    start_step = 0
    best_val_loss = float("inf")

    if os.path.exists(resume_path):
        print(f"Resuming from {resume_path}...")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"] + 1
        best_val_loss = ckpt["val_loss"]
        print(f"Resumed at step {start_step}, val_loss {best_val_loss:.4f}")
    else:
        print("No checkpoint found, starting from scratch")

    model.train()
    t_start = time.time()

    for step in range(start_step, MAX_STEPS + 1):
        # Update learning rate for this step
        if step <= 30000:
            lr = get_lr(step, WARMUP_STEPS, 30000, MAX_LR, MIN_LR)
        else:
            # Fresh cosine decay from 5e-5 down to 1e-5 over the extension steps
            lr = get_lr(step - 30000, 50, 10000, 5e-5, 1e-5)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Periodic evaluation
        if step % EVAL_INTERVAL == 0:
            val_loss = estimate_val_loss(model, device, EVAL_ITERS)
            elapsed = time.time() - t_start
            print(f"[step {step:>5d}] val_loss = {val_loss:.4f}  "
                  f"lr = {lr:.2e}  elapsed = {elapsed/60:.1f}m")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    os.path.join(CKPT_DIR, "best.pt"),
                    model, optimizer, config, step, val_loss,
                )

        if step == MAX_STEPS:
            break

        # One training step
        with Timer() as t:
            x, y = get_batch("train", BATCH_SIZE, BLOCK_SIZE, device)
            _, loss = model(x, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        if step % LOG_INTERVAL == 0:
            tokens_per_sec = BATCH_SIZE * BLOCK_SIZE / t.elapsed
            print(f"  step {step:>5d}  "
                  f"loss {loss.item():.4f}  "
                  f"lr {lr:.2e}  "
                  f"{tokens_per_sec:,.0f} tok/s  "
                  f"({t.elapsed*1000:.0f} ms/step)")

    # Final checkpoint
    save_checkpoint(
        os.path.join(CKPT_DIR, "final.pt"),
        model, optimizer, config, MAX_STEPS, best_val_loss,
    )
    print(f"\nDone!\nBest val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()