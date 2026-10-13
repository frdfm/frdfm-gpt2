import os

import numpy as np
import tiktoken
from datasets import load_dataset

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TOTAL_TOKENS = 300_000_000
VAL_TOKENS = 1_000_000

os.makedirs(DATA_DIR, exist_ok=True)

encoding = tiktoken.get_encoding("gpt2")
stream = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)

total = 0
documents = 0
with open(os.path.join(DATA_DIR, "train.bin"), "wb") as train_file, open(os.path.join(DATA_DIR, "val.bin"), "wb") as val_file:
    for example in stream:
        text = example["text"]
        if not text:
            continue
        ids = np.array(encoding.encode_ordinary(text) + [encoding.eot_token], dtype=np.uint16)
        count = min(ids.size, TOTAL_TOKENS - total)
        val_count = min(count, max(VAL_TOKENS - total, 0))
        val_file.write(ids[:val_count].tobytes())
        train_file.write(ids[val_count:count].tobytes())
        total += count
        documents += 1
        if documents % 5000 == 0:
            print(f"{total:,} tokens")
        if total == TOTAL_TOKENS:
            break

print(f"train {TOTAL_TOKENS - VAL_TOKENS:,} tokens")
print(f"val   {VAL_TOKENS:,} tokens")
print(f"saved {DATA_DIR}")
