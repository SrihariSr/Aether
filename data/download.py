import os
import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm

def main():
    data_dir = "data/prepared"
    os.makedirs(data_dir, exist_ok=True)

    print("Loading tokenizer...")
    enc = tiktoken.get_encoding("gpt2")
    print("Loaded tokenizer!")

    print("Downloading OpenWebText... ")
    dataset = load_dataset("openwebtext", split="train", trust_remote_code=True)
    print("Downloaded OpenWebText!")

    num_documents = 50000
    print(f"Training with {num_documents} documents...")

    print("Tokenizing...")
    all_tokens = []
    
    for i in tqdm(range(num_documents)):
        text = dataset[i]["text"]
    
        # Skipping small documents as they don't contribute much to training
        if len(text) < 50:
            continue
    
        all_tokens.extend(enc.encode(text))
    
    print(f"Total tokens: {len(all_tokens)}")

    # Converting to numpy array for efficient storage
    all_tokens = np.array(all_tokens, dtype=np.uint16)

    split_point = int(len(all_tokens) * 0.9)
    train_tokens = all_tokens[:split_point]
    val_tokens = all_tokens[split_point:]

    print(f"Train tokens: {len(train_tokens):,}")
    print(f"Val tokens:   {len(val_tokens):,}")

    # Save to disk
    train_path = os.path.join(data_dir, "train.bin")
    val_path = os.path.join(data_dir, "val.bin")

    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(f"\nSaved to:")
    print(f"{train_path} ({os.path.getsize(train_path) / 1e6:.1f} MB)")
    print(f"{val_path} ({os.path.getsize(val_path) / 1e6:.1f} MB)")
    print("Done!")

if __name__ == "__main__":
    main()




    


    