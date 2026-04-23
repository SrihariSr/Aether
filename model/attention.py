import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from .config import GPTConfig

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Q, K, V are computed together in a single Linear for efficiency
        self.qkv_proj = nn.Linear(
            config.n_embd, 3 * config.n_embd, bias=config.bias
        )
        # Final projection after concatenating heads
        self.out_proj = nn.Linear(
            config.n_embd, config.n_embd, bias=config.bias
        )

        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)

        self.n_head   = config.n_head
        self.n_embd   = config.n_embd
        self.head_dim = config.head_dim
        self.dropout  = config.dropout

        self.flash = hasattr(f, "scaled_dot_product_attention")

        if not self.flash:
            mask = torch.tril(torch.ones(config.block_size, config.block_size))
            self.register_buffer("causal_mask", mask.view(
                1, 1, config.block_size, config.block_size
            ))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        if self.flash:
            y = f.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Manual implementation for reference / fallback
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            att = att.masked_fill(
                self.causal_mask[:, :, :T, :T] == 0, float("-inf")
            )
            att = f.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.out_proj(y)
        y = self.resid_drop(y)
        return y


if __name__ == "__main__":
    from config import andromeda_small

    config = andromeda_small()
    attn = CausalSelfAttention(config)

    # Fake input: batch of 2, 16 tokens, 384-dim embeddings
    x = torch.randn(2, 16, config.n_embd)
    y = attn(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Flash attention available: {attn.flash}")

    n_params = sum(p.numel() for p in attn.parameters())
    print(f"\nAttention parameters: {n_params:,}")

    # Manual check: two Linears of size (384, 3*384) and (384, 384), no bias
    expected = config.n_embd * 3 * config.n_embd + config.n_embd * config.n_embd
    print(f"Expected:             {expected:,}")