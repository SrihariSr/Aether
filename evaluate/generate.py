import argparse
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as f
import tiktoken

from model.config import GPTConfig
from model.transformer import GPT

@torch.no_grad()
def generate(
    model: GPT,
    idx: torch.Tensor,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    top_p: float | None = None,
    repetition_penalty: float = 1.0,
) -> torch.Tensor:
    for _ in range(max_new_tokens):
        # Crop context to block_size most recent tokens
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]

        # Forward pass — we only need logits, no targets
        logits, _ = model(idx_cond)

        # Focus on the logits at the last position only
        logits = logits[:, -1, :] / temperature

        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(f.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs - f.softmax(sorted_logits, dim=-1) > top_p
            sorted_logits[sorted_indices_to_remove] = float("-inf")
            logits.scatter_(1, sorted_indices, sorted_logits)

        if repetition_penalty > 1.0:
            for token_id in set(idx[0].tolist()[-64:]):    # last 64 tokens
                logits[:, token_id] /= repetition_penalty

        # Softmax to probabilities, then sample
        probs = f.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append and continue
        idx = torch.cat((idx, next_token), dim=1)

    return idx

def load_model(ckpt_path: str, device: str) -> GPT:
    # Load a checkpoint and rebuild the model from the stored config
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # Rebuild config from the saved dict
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    print(f"Loaded checkpoint from {ckpt_path}")
    print(f"step:     {ckpt['step']}")
    print(f"val_loss: {ckpt['val_loss']:.4f}")
    print(f"params:   {model.num_parameters()/1e6:.2f}M")
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt",        default="checkpoints/best.pt")
    parser.add_argument("--prompt",      default="Once upon a time")
    parser.add_argument("--max_tokens",  type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k",             type=int,   default=40)
    parser.add_argument("--top_p",             type=float, default=None)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_samples",       type=int,   default=3)
    parser.add_argument("--seed",              type=int,   default=3407)
    args = parser.parse_args()

    # Device
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}\n")

    # Reproducible sampling
    torch.manual_seed(args.seed)

    # Load model + tokeniser
    model = load_model(args.ckpt, device)
    enc = tiktoken.get_encoding("gpt2")

    # Encode the prompt
    prompt_ids = enc.encode(args.prompt)
    idx = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0)

    print(f"\nPrompt: {args.prompt!r}")
    print(f"Settings: temperature={args.temperature}, top_k={args.top_k}, "
          f"top_p={args.top_p}, repetition_penalty={args.repetition_penalty}, "
          f"max_tokens={args.max_tokens}\n")
    print("=" * 70)

    for i in range(args.num_samples):
        out = generate(
            model, idx,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty
        )
        text = enc.decode(out[0].tolist())
        print(f"\nSample {i+1}: ")
        print(text)
        print("-" * 70)

if __name__ == "__main__":
    main()