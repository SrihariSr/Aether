import torch
import torch.nn as nn
import torch.nn.functional as f
from config import GPTConfig

class LayerNorm(nn.Module):
    def __init__(self, n_embd: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_embd))
        self.bias = nn.Parameter(torch.zeros(n_embd)) if bias else None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return f.layer_norm(x, self.weight.shape, self.weight, self.bias, eps=1e-6)

class FeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias) # i.e. fully connected
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias) # i.e. projection
        self.drop = nn.Dropout(config.dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = f.gelu(x)
        x = self.proj(x)
        x = self.drop(x)
        return x

class TokenPositionEmbedding(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.block_size = config.block_size

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size, (
            f"Sequence length {T} exceeds block_size {self.block_size}"
        )

        # Token embeddings: (B, T) -> (B, T, n_embd)
        tok = self.token_emb(idx)

        # Position embeddings: (T,) -> (T, n_embd)
        pos = torch.arange(T, device=idx.device)
        pos = self.pos_emb(pos)

        # Broadcasting adds the position vector to every sample in the batch
        x = tok + pos
        return self.drop(x)
