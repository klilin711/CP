import argparse
import os

from transformers import AutoTokenizer
from datasets import load_dataset

from data_common import write_datafile

# -----------------------------------------------------------------------------
DATA_CACHE_DIR = os.path.join(os.path.dirname(__file__), "escorpius")

def download_and_load():
    """Loads the Escorpius dataset from Hugging Face"""
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)
    print("Downloading Escorpius dataset from Hugging Face...")
    dataset = load_dataset("LHF/escorpius")
    train_text = "\n".join(dataset["train"]["text"])
    val_text = "\n".join(dataset["validation"]["text"])
    # Save the raw text locally for reference
    with open(os.path.join(DATA_CACHE_DIR, "escorpius_train.txt"), "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(os.path.join(DATA_CACHE_DIR, "escorpius_val.txt"), "w", encoding="utf-8") as f:
        f.write(val_text)
    return train_text, val_text

def tokenize(model_desc):
    if model_desc == "gpt-2":
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
    elif model_desc == "llama-3":
        tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
    else:
        raise ValueError(f"Unknown model descriptor {model_desc}")
    
    train_text, val_text = download_and_load()

    def tokenize_text(text):
        tokens = tokenizer.encode(text, add_special_tokens=True)
        return tokens

    # Tokenize the training and validation datasets
    print("Tokenizing data...")
    train_tokens = tokenize_text(train_text)
    val_tokens = tokenize_text(val_text)

    # Save the tokenized data to binary files
    train_filename = os.path.join(DATA_CACHE_DIR, "escorpius_train.bin")
    val_filename = os.path.join(DATA_CACHE_DIR, "escorpius_val.bin")
    write_datafile(train_filename, train_tokens, model_desc)
    write_datafile(val_filename, val_tokens, model_desc)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Escorpius dataset preprocessing")
    parser.add_argument("-m", "--model_desc", type=str, default="gpt-2", choices=["gpt-2", "llama-3"], help="Model type, gpt-2|llama-3")
    args = parser.parse_args()
    tokenize(args.model_desc)
