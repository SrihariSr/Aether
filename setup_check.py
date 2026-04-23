import torch
import tiktoken
import datasets
import tqdm

print(f"PyTorch version: {torch.__version__}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print(f"Using Apple Silicon GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using Nvidia GPU (CUDA)")
else:
    device = torch.device("cpu")
    print("Using CPU\nWarning: Training will be slow.")

# Quick GPU test
x = torch.randn(1000, 1000, device=device)
y = x @ x.T
print(f"GPU matmul test passed: {y.shape}")

# Check tokenizer
enc = tiktoken.get_encoding("gpt2")
tokens = enc.encode("Hello, world!")
print(f"Tokenizer test: 'Hello, world!' -> {tokens}")
print(f"Decoded back: '{enc.decode(tokens)}'")
print(f"GPT-2 vocabulary size: {enc.n_vocab}")
