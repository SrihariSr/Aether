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

    # Parallel tokenisation, streamed to disk.
    # The old version held every token as a Python int in one list (~36 bytes each,
    # ~9 GB at 246M tokens). We buffer a few million at a time, convert to uint16,
    # and append to a flat file. Peak RAM stays around 20 MB.
    all_path = os.path.join(data_dir, "all.bin")
    total_tokens = 0
    buf: list[int] = []
    FLUSH_EVERY = 10_000_000   # ~20 MB per flush

    with open(all_path, "wb") as fout:
        with Pool(num_workers, initializer=init_worker) as pool:
            for tokens in tqdm(
                pool.imap(tokenize_document, texts, chunksize=64),
                total=num_documents,
            ):
                buf.extend(tokens)
                if len(buf) >= FLUSH_EVERY:
                    np.array(buf, dtype=np.uint16).tofile(fout)
                    total_tokens += len(buf)
                    buf.clear()
        if buf:
            np.array(buf, dtype=np.uint16).tofile(fout)
            total_tokens += len(buf)
            buf.clear()

    print(f"\nTotal tokens: {total_tokens:,}")

    # Split 90/10 by memory-mapping the flat file, so it is never fully loaded.
    data = np.memmap(all_path, dtype=np.uint16, mode="r")
    assert len(data) == total_tokens, f"file has {len(data):,}, counted {total_tokens:,}"
    split_point = int(len(data) * 0.9)

    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    CHUNK = 50_000_000
    for path, lo, hi in [(train_path, 0, split_point), (val_path, split_point, len(data))]:
        with open(path, "wb") as fout:
            for start in range(lo, hi, CHUNK):
                np.asarray(data[start:min(start + CHUNK, hi)]).tofile(fout)

    print(f"Train tokens: {split_point:,}")
    print(f"Val tokens:   {len(data) - split_point:,}")

    del data
    os.remove(all_path)

    print(f"\nSaved:")
    print(f"{train_path} ({os.path.getsize(train_path) / 1e6:.1f} MB)")
    print(f"{val_path} ({os.path.getsize(val_path) / 1e6:.1f} MB)")
    print("Done!")


if __name__ == "__main__":
    main()