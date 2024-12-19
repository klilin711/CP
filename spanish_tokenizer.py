import argparse
import os
from datasets import load_dataset
from transformers import AutoTokenizer

from data_common import write_datafile  # Assumes write_datafile exists for saving

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "escorpius")

def download():
    """Downloads a very small fixed subset of the Escorpius dataset using streaming to DATA_CACHE_DIR."""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    data_filename = os.path.join(DATA_CACHE_DIR, "escorpius_subset.txt")
    if not os.path.exists(data_filename):
        print("Downloading a very small fixed subset (1000 samples) of the Escorpius dataset with streaming...")
        # Use streaming mode to fetch only 1000 samples
        dataset = load_dataset("LHF/escorpius", split="train", streaming=True)
        text_list = []
        for i, sample in enumerate(dataset):
            if i >= 1000:  # Limit to 1000 samples
                break
            text_list.append(sample["text"])
        
        text = "\n".join(text_list)
        
        # Save the subset locally
        with open(data_filename, "w", encoding="utf-8") as f:
            f.write(text)
        print(f"Subset of dataset saved to {data_filename}.")
    else:
        print(f"Dataset already exists at {data_filename}, skipping download.")

def tokenize():
    """Tokenizes and decodes the Escorpius dataset using the DeepESP GPT-2 Spanish tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained("DeepESP/gpt2-spanish")
    encode = lambda s: tokenizer.encode(s, add_special_tokens=False)
    decode = lambda t: tokenizer.decode(t)  # Decoding function
    eot = tokenizer.eos_token_id  # End-of-text token

    data_filename = os.path.join(DATA_CACHE_DIR, "escorpius_subset.txt")
    with open(data_filename, 'r', encoding="utf-8") as f:
        text = f.read()

    # Split text into sections (paragraphs)
    sections = text.split("\n\n")
    tokens = []
    for i, s in enumerate(sections):
        tokens.append(eot)
        spad = s + "\n\n" if i != len(sections) - 1 else s
        encoded_section = encode(spad)
        tokens.extend(encoded_section)

        # Print tokenized and decoded text for debugging
        print(f"Original text: {repr(spad[:100])}...")  # Show first 100 chars
        print(f"Tokenized IDs: {encoded_section[:100]}...")  # Show first 20 token IDs
        print(f"Decoded text: {repr(decode(encoded_section[:100]))}...\n")  # Decode first 20 tokens

    # Split into validation and training sets
    val_tokens = tokens[:32768]
    train_tokens = tokens[32768:]

    # Save to binary files
    val_filename = os.path.join(DATA_CACHE_DIR, "escorpius_val.bin")
    train_filename = os.path.join(DATA_CACHE_DIR, "escorpius_train.bin")
    write_datafile(val_filename, val_tokens, "gpt-2")  # Default to gpt-2 format for compatibility
    write_datafile(train_filename, train_tokens, "gpt-2")

    print(f"Validation tokens written to {val_filename}.")
    print(f"Training tokens written to {train_filename}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Escorpius dataset preprocessing")
    args = parser.parse_args()
    download()
    tokenize()
