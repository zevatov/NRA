import os
import shutil
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download

RAW_DIR = Path("/Users/stanislav/Desktop/NAP/nra/.benchmark_data/raw")
os.makedirs(RAW_DIR, exist_ok=True)

# 1. Wikitext
print("Recovering Wikitext...")
wiki_dir = RAW_DIR / "wikitext"
os.makedirs(wiki_dir, exist_ok=True)
ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
for i, item in enumerate(ds):
    if item['text'].strip():
        with open(wiki_dir / f"{i}.txt", "w", encoding="utf-8") as f:
            f.write(item['text'])

# 2. Minds14
print("Recovering Minds14...")
audio_dir = RAW_DIR / "minds14"
os.makedirs(audio_dir, exist_ok=True)
import soundfile as sf
ds = load_dataset("PolyAI/minds14", "en-US", split="train")
for i, item in enumerate(ds):
    audio = item['audio']
    sf.write(str(audio_dir / f"{i}.wav"), audio['array'], audio['sampling_rate'])

# 3. Pokemon
print("Recovering Pokemon...")
poke_dir = RAW_DIR / "pokemon"
os.makedirs(poke_dir, exist_ok=True)
ds = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train")
for i, item in enumerate(ds):
    # Depending on dataset columns: usually 'image' and 'text'
    if 'image' in item:
        item['image'].save(poke_dir / f"{i}.png")
    if 'text' in item:
        with open(poke_dir / f"{i}.txt", "w", encoding="utf-8") as f:
            f.write(item['text'])

# 4. GPT-2
print("Recovering GPT-2...")
tensors_dir = RAW_DIR / "gpt2"
os.makedirs(tensors_dir, exist_ok=True)
path = hf_hub_download(repo_id="openai-community/gpt2", filename="model.safetensors")
shutil.copy(path, tensors_dir / "model.safetensors")

print("All raw files recovered into .benchmark_data/raw/")
