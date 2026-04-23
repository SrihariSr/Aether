from dataclasses import dataclass

@dataclass
class GPTConfig:
    # Tokenizer
    vocab_size: int = 50257 # Using same as OpenAI's GPT-2 tokenizer

    block_size: int = 256
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

    dropout: float = 0.1

    bias: bool = False

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head # Dimension of each attention head
  
def andromeda_small() -> GPTConfig:
    # Around 15M parameters
    return GPTConfig(
        block_size=256,
        n_layer=6,
        n_head=6,
        n_embd=384,
        dropout=0.1,
    )
def andromeda_medium() -> GPTConfig:
    # Around 50M parameters
    return GPTConfig(
        block_size=512,
        n_layer=12,
        n_head=8,
        n_embd=512,
        dropout=0.1,
    )
