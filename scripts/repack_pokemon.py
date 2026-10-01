import os
import subprocess
from pathlib import Path
from datasets import load_dataset

RAW_DIR = Path("/Users/stanislav/Desktop/NAP/nra/.benchmark_data/raw/pokemon")
os.makedirs(RAW_DIR, exist_ok=True)

print("Recovering Pokemon dataset properly...")
ds = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train")

for i, item in enumerate(ds):
    if 'image' in item:
        item['image'].save(RAW_DIR / f"{i}.png")
    if 'text' in item:
        with open(RAW_DIR / f"{i}.txt", "w", encoding="utf-8") as f:
            f.write(item['text'])

print(f"Saved {len(ds)} multimodal pairs. Packing...")

NRA_CLI = Path("/Users/stanislav/Desktop/NAP/nra/target/release/nra-cli")
OUT_FILE = Path("/Users/stanislav/Desktop/NAP/nra/.benchmark_data/hf_archives/pokemon.nra")

subprocess.run([str(NRA_CLI), "pack-beta", "--input", str(RAW_DIR), "--output", str(OUT_FILE)], check=True)
print("Done packing pokemon.nra!")
