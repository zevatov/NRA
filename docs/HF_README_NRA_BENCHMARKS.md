---
license: mit
task_categories:
  - image-classification
  - text-generation
  - automatic-speech-recognition
language:
  - en
tags:
  - nra
  - neural-ready-archive
  - streaming
  - zero-download
  - deduplication
  - benchmark
size_categories:
  - 100K<n<1M
---

# 🧬 NRA Benchmark Datasets

**All benchmark datasets for [Neural Ready Archive (NRA)](https://github.com/zevatov/NRA) — the Rust-native streaming format for ML training.**

Train on gigabytes of real data **without downloading a single byte**. NRA replaces `tar.gz` and `zip` for the AI era.

## 📦 Available Datasets

| File | Domain | Source | Files | Size |
|------|--------|--------|-------|------|
| `food-101.nra` | 🖼️ Vision | [ethz/food101](https://huggingface.co/datasets/ethz/food101) | 101,000 images | 4.7 GB |
| `wikitext.nra` | 📝 Text | [Salesforce/wikitext](https://huggingface.co/datasets/Salesforce/wikitext) | 23,767 text files | 7.6 MB |
| `pokemon.nra` | 🎨 Multimodal | [svjack/pokemon-blip-captions-en-zh](https://huggingface.co/datasets/svjack/pokemon-blip-captions-en-zh) | 833 image+text pairs | 329 MB |
| `minds14.nra` | 🎵 Audio | [PolyAI/minds14](https://huggingface.co/datasets/PolyAI/minds14) | 563 WAV files | 37 MB |
| `gpt2-weights.nra` | 🧠 Tensors | [openai-community/gpt2](https://huggingface.co/openai-community/gpt2) | Model weights | 448 MB |
| `synthetic.nra` | 🧪 Synthetic | Generated locally | 100K mixed files | 449 MB |

## 🚀 Quick Start (Zero-Download Training)

```bash
pip install nra torch
```

```python
import nra
from torch.utils.data import Dataset, DataLoader

class NRAStreamDataset(Dataset):
    def __init__(self, url):
        self.archive = nra.CloudArchive(url)
        self.file_ids = self.archive.file_ids()

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        return self.archive.read_file(self.file_ids[idx])

# Stream Food-101 (4.7 GB) directly from HuggingFace — 0 bytes on your SSD
dataset = NRAStreamDataset(
    "https://huggingface.co/datasets/zevatov/nra-benchmarks/resolve/main/food-101.nra"
)
loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

for batch in loader:
    pass  # Training starts instantly!
```

## 📊 Why NRA?

| | **tar.gz** | **ZIP** | **NRA** |
|---|:---:|:---:|:---:|
| Stream from cloud | ❌ | ❌ | ✅ Any file |
| Random access O(1) | ❌ | ⚠️ Slow | ✅ |
| Deduplication | ❌ | ❌ | ✅ 4-8x savings |
| Encryption (AES-256) | ❌ | ⚠️ Weak | ✅ Per-block |
| Time to first batch (5 GB) | ~7 min | ~7 min | **0.6 sec** |

## 🔍 Verify Integrity

```bash
cargo install nra-cli
nra-cli verify-beta --input food-101.nra
# ✅ VERIFICATION PASSED — 101,000 files OK (CRC32 + BLAKE3)
```

## 📄 Full Documentation

- [Technical Whitepaper](https://github.com/zevatov/NRA/blob/main/docs/nra_whitepaper.md)
- [GitHub Repository](https://github.com/zevatov/NRA)

## License

MIT
