from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as f
from config import GPTConfig
from attention import CausalSelfAttention
from layers import LayerNorm, FeedForward, TokenPositionEmbedding

class TransformerBlock(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1  = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln2  = LayerNorm(config.n_embd, bias=config.bias)
        self.ffn  = FeedForward(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class GPT(nn.Module):
    """
    The full Andromeda GPT language model.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Embeddings
        self.embed = TokenPositionEmbedding(config)

        # Stack of transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layer)
        ])

        # Final layer norm before the output projection
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

        # Output projection to vocabulary
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between token embedding and output projection
        self.embed.token_emb.weight = self.lm_head.weight

        # Apply custom weight initialisation
        self.apply(self._init_weights)

        # Scale init of residual projections by 1/sqrt(2 * n_layer)
        # GPT-2 trick: keeps variance stable through deep networks
        for pn, p in self.named_parameters():
            if pn.endswith("out_proj.weight") or pn.endswith("proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self, non_embedding: bool = True) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.embed.pos_emb.weight.numel()
        return n

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        # Embed tokens + positions
        x = self.embed(idx) # (B, T, n_embd)

        # Run through all transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm and project to vocabulary logits
        x = self.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = f.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )

        return logits, loss

if __name__ == "__main__":
    from config import andromeda_small

    config = andromeda_small()
    model = GPT(config)

    print(f"Total parameters:         {sum(p.numel() for p in model.parameters()):,}")
    print(f"Non-embedding parameters: {model.num_parameters():,}")
    print()

    # Fake batch: 2 sequences of 16 tokens
    idx = torch.randint(0, config.vocab_size, (2, 16))
    targets = torch.randint(0, config.vocab_size, (2, 16))

    # Inference mode: just logits
    logits, loss = model(idx)
    print(f"Inference:  logits {tuple(logits.shape)}, loss {loss}")

    # Training mode: with targets, we get a loss
    logits, loss = model(idx, targets)
    print(f"Training:   logits {tuple(logits.shape)}, loss {loss.item():.4f}")

    # Sanity check: untrained loss should be ~ln(vocab_size)
    expected = math.log(config.vocab_size)
    print(f"Expected random loss ~ {expected:.4f}")