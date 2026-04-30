---
license: mit
task_categories:
- image-classification
language:
- en
tags:
- nra
- neural-ready-archive
- streaming
- zero-download
- cifar10
- rust
- pytorch
size_categories:
- 10K<n<100K
---

# 🧬 CIFAR-10 in NRA Format — Zero-Download Cloud Training

<div align="center">

[![PyPI](https://img.shields.io/badge/pip_install_nra-1.0.3-blue)](https://pypi.org/project/nra/1.0.3/)
[![GitHub](https://img.shields.io/badge/GitHub-zevatov%2FNRA-black?logo=github)](https://github.com/zevatov/NRA)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

</div>

This dataset contains **CIFAR-10** (60,000 images, ~170 MB) packaged in the **NRA (Neural Ready Archive)** format — a next-generation binary format built in Rust for the AI era.

> 💡 **Looking for a larger dataset?** Try our [**Food-101 (5 GB)**](https://huggingface.co/datasets/zevatov/nra-food101) — 101,000 high-resolution food images in NRA format.

## 🚀 Why This Matters

**You DO NOT need to download this dataset.** NRA streams data directly into your PyTorch `DataLoader` via HTTP Range requests. Only the exact 4MB blocks your model needs are fetched on-the-fly.

| Metric | Traditional (tar.gz) | NRA (this dataset) |
|--------|---------------------|-------------------|
| Time to first batch | ~30 sec (download + unpack) | **150 ms** |
| Local disk space | 170 MB | **0 bytes** |
| Random file access | Impossible | **O(1) instant** |

---

## ⚡ Quick Start

```bash
pip install nra torch
```

```python
import nra

# Connect to this archive — nothing is downloaded!
archive = nra.BetaArchive(
    "https://huggingface.co/datasets/zevatov/nra-cifar10/resolve/main/cifar10.nra"
)

# Instantly fetch any file via HTTP Range (O(1))
image_bytes = archive.read_file("train/00499_truck.png")
print(f"Got {len(image_bytes)} bytes — streamed from Hugging Face!")
```

### Full PyTorch DataLoader Example

```python
import nra
import torch
from torch.utils.data import Dataset, DataLoader

class NraStreamDataset(Dataset):
    def __init__(self, url):
        self.archive = nra.BetaArchive(url)
        self.file_ids = self.archive.file_ids()
        
    def __len__(self):
        return len(self.file_ids)
        
    def __getitem__(self, idx):
        raw_bytes = self.archive.read_file(self.file_ids[idx])
        return torch.tensor([len(raw_bytes)], dtype=torch.float32)

dataset = NraStreamDataset(
    "https://huggingface.co/datasets/zevatov/nra-cifar10/resolve/main/cifar10.nra"
)
loader = DataLoader(dataset, batch_size=256, num_workers=4)

print(f"✅ {len(dataset)} files ready. Training starts NOW — zero bytes on your SSD!")
for batch in loader:
    pass  # Your model trains here
```

---

## 📊 Dataset Details

| Field | Value |
|-------|-------|
| **Source** | CIFAR-10 (Krizhevsky, 2009) |
| **Format** | `.nra` (Neural Ready Archive v4.5) |
| **Images** | 60,000 (32×32 RGB) |
| **Classes** | 10 |
| **Compression** | Zstd (level 15) + CDC deduplication |
| **NRA SDK** | `pip install nra==1.0.3` |

---

## 📚 Learn More

- 🏠 **[GitHub Repository](https://github.com/zevatov/NRA)** — Full source code, benchmarks, whitepapers
- 📦 **[PyPI Package](https://pypi.org/project/nra/)** — `pip install nra`
- 🍕 **[Food-101 NRA (5 GB)](https://huggingface.co/datasets/zevatov/nra-food101)** — Larger dataset for serious benchmarking
- 📄 **[Technical Whitepaper](https://github.com/zevatov/NRA/blob/main/docs/nra_whitepaper.md)** — Architecture deep-dive

## License

This dataset and the NRA format are released under the **MIT License**.
