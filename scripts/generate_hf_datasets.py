import os
import shutil
import subprocess
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import hf_hub_download

WORKSPACE = Path("/Users/stanislav/Desktop/NAP/nra/huggingface_ready_nra")
RAW_DIR = WORKSPACE / "raw"
OUTPUT_DIR = WORKSPACE / "nra_archives"

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

NRA_CLI = Path("/Users/stanislav/Desktop/NAP/nra/target/release/nra-cli")
if not NRA_CLI.exists():
    print("Building nra-cli...")
    subprocess.run(["cargo", "build", "--release", "--bin", "nra-cli"], cwd="/Users/stanislav/Desktop/NAP/nra")

def pack_folder(name, src_dir):
    out_file = OUTPUT_DIR / f"{name}.nra"
    if out_file.exists():
        print(f"[{name}] NRA archive already exists, skipping pack.")
        return
    print(f"[{name}] Packing {src_dir} to {out_file}...")
    subprocess.run([str(NRA_CLI), "pack-beta", "--input", str(src_dir), "--output", str(out_file)], check=True)
    print(f"[{name}] Packed successfully: {out_file.stat().st_size / 1024 / 1024:.2f} MB")

# 1. Wikitext (Text)
print("Processing Wikitext...")
wiki_dir = RAW_DIR / "wikitext"
if not wiki_dir.exists():
    os.makedirs(wiki_dir)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    for i, item in enumerate(ds):
        if item['text'].strip():
            with open(wiki_dir / f"{i}.txt", "w", encoding="utf-8") as f:
                f.write(item['text'])
pack_folder("wikitext", wiki_dir)

# 2. Minds14 (Audio)
print("Processing Minds14...")
audio_dir = RAW_DIR / "minds14"
if not audio_dir.exists():
    os.makedirs(audio_dir)
    import soundfile as sf
    ds = load_dataset("PolyAI/minds14", "en-US", split="train")
    for i, item in enumerate(ds):
        audio = item['audio']
        sf.write(str(audio_dir / f"{i}.wav"), audio['array'], audio['sampling_rate'])
pack_folder("minds14", audio_dir)

# 3. Pokemon (Multimodal)
print("Processing Pokemon...")
poke_dir = RAW_DIR / "pokemon"
if not poke_dir.exists():
    os.makedirs(poke_dir)
    ds = load_dataset("svjack/pokemon-blip-captions-en-zh", split="train")
    for i, item in enumerate(ds):
        item['image'].save(poke_dir / f"{i}.png")
        with open(poke_dir / f"{i}.txt", "w", encoding="utf-8") as f:
            f.write(item['text'])
pack_folder("pokemon", poke_dir)

# 4. GPT-2 (Tensors)
print("Processing GPT-2...")
tensors_dir = RAW_DIR / "gpt2"
if not tensors_dir.exists():
    os.makedirs(tensors_dir)
    path = hf_hub_download(repo_id="openai-community/gpt2", filename="model.safetensors")
    shutil.copy(path, tensors_dir / "model.safetensors")
pack_folder("gpt2-weights", tensors_dir)

# 5. Food-101 (Vision)
print("Processing Food-101...")
food_dir = RAW_DIR / "food101"
if not food_dir.exists():
    os.makedirs(food_dir)
    ds = load_dataset("ethz/food101", split="train")
    for i, item in enumerate(ds):
        # some images might be RGBA, convert to RGB
        img = item['image']
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.save(food_dir / f"{i}.jpg")
pack_folder("food-101", food_dir)

print(f"\nAll NRA archives are ready for HuggingFace upload in: {OUTPUT_DIR}")
