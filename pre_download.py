print("--- Starting Pre-Download for Offline Training ---")

# 1. Setup Environment
import os
# Ensure we use the Scratch cache
os.environ["HF_HOME"] = os.environ.get("SCRATCH") + "/huggingface_cache"

from unsloth import FastLanguageModel
from datasets import load_dataset

# 2. Download Model (Llama-3 8B 4bit)
print("Downloading Model: unsloth/llama-3-8b-bnb-4bit...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-bnb-4bit",
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
print("Model Downloaded Successfully!")

# 3. Download Dataset (Mahesh2841/Agriculture)
print("Downloading Dataset: Mahesh2841/Agriculture...")
dataset = load_dataset("Mahesh2841/Agriculture", split = "train")
print("Dataset Downloaded Successfully!")

print("--- All files cached in $SCRATCH/huggingface_cache ---")
