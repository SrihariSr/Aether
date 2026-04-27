"""
Download and tokenize training data for Aether.
Parallelised across CPU cores for fast processing of large datasets.
"""
import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


# Global tokeniser reference, initialised once per worker process
enc = None


def init_worker():
    # Called once when each worker process starts. Loads the tokeniser.
    global enc
    enc = tiktoken.get_encoding("gpt2")


def tokenize_document(text: str) -> list[int]:
    # Tokenise a single document. Runs inside a worker process.
    if len(text) < 50:
        return []
    # for raw training text where we don't want special tokens inserted
    return enc.encode_ordinary(text)


def main():
    # Configuration
    data_dir = "data/prepared"
    os.makedirs(data_dir, exist_ok=True)
    num_documents = 500_000
    num_workers = max(1, cpu_count() - 2)

    print(f"Using {num_workers} worker processes")

    # Download dataset
    dataset = load_dataset("openwebtext", split="train", trust_remote_code=True)
    print(f"Total documents in dataset: {len(dataset):,}")
    print(f"Using first {num_documents:,}")

    # Extract texts as a list so we can pool.imap them
    texts = (dataset[i]["text"] for i in range(num_documents))

    # Parallel tokenisation
    all_tokens: list[int] = []

    with Pool(num_workers, initializer=init_worker) as pool:
        for tokens in tqdm(
            pool.imap(tokenize_document, texts, chunksize=64),
            total=num_documents,
        ):
            all_tokens.extend(tokens)

    total_tokens = len(all_tokens)
    print(f"\nTotal tokens: {total_tokens:,}")

    # Convert to numpy array
    all_tokens_np = np.array(all_tokens, dtype=np.uint16)

    # Split into train and val
    split_point = int(len(all_tokens_np) * 0.9)
    train_tokens = all_tokens_np[:split_point]
    val_tokens = all_tokens_np[split_point:]

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens:   {len(val_tokens):,}")

    # Save to disk
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(f"\nSaved:")
    print(f"{train_path} ({os.path.getsize(train_path) / 1e6:.1f} MB)")
    print(f"{val_path} ({os.path.getsize(val_path) / 1e6:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()