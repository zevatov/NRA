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
- food-101
- rust
- pytorch
- benchmark
size_categories:
- 100K<n<1M
---

# 🍕 Food-101 in NRA Format — 5 GB Zero-Download Streaming Benchmark

<div align="center">

[![PyPI](https://img.shields.io/badge/pip_install_nra-1.0.3-blue)](https://pypi.org/project/nra/1.0.3/)
[![GitHub](https://img.shields.io/badge/GitHub-zevatov%2FNRA-black?logo=github)](https://github.com/zevatov/NRA)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

**5 GB · 101,000 images · 101 food categories · Streamed directly into PyTorch**

</div>

This dataset contains the full **Food-101** dataset (101,000 high-resolution food images across 101 categories) packaged in the **NRA (Neural Ready Archive)** format.

This is our **production-scale benchmark** — proving that NRA can stream real 5 GB datasets directly from cloud storage into your model with zero local disk usage.

## 🚀 The Problem This Solves

Traditional workflow with a 5 GB dataset:
1. ⏳ Download 5 GB archive (5-15 min on 100 Mbps)
2. ⏳ Unpack 101,000 files to disk (2-5 min)
3. ⏳ Wait for disk I/O during training
4. 💾 5 GB of SSD space consumed

**NRA workflow:**
1. ✅ `archive = nra.BetaArchive(url)` — manifest loads in **0.6 sec**
2. ✅ Training starts **immediately** — data streams via HTTP Range
3. ✅ **Zero bytes** on your SSD

| Metric | tar.gz (traditional) | NRA (this dataset) |
|--------|---------------------|-------------------|
| Time to first batch | **~7 min** (download + unpack) | **0.6 sec** |
| Local disk space | 5 GB | **0 bytes** |
| Files to manage | 101,000 loose files | **1 file (remote)** |
| Random file access | O(n) scan | **O(1) instant** |

---

## ⚡ Quick Start: Stream 5 GB in One Line

```bash
pip install nra torch torchvision Pillow
```

```python
import nra

# Connect to the 5 GB archive — only the manifest is downloaded (0.6 sec)!
archive = nra.BetaArchive(
    "https://huggingface.co/datasets/zevatov/nra-food101/resolve/main/food-101.nra"
)

# Fetch a pizza image directly from Hugging Face CDN
image_bytes = archive.read_file("images/pizza/1001116.jpg")
print(f"🍕 Got {len(image_bytes)} bytes — streamed from the cloud!")
```

### Full PyTorch Training Example

```python
import nra
import torch
import io
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class Food101Stream(Dataset):
    """Stream Food-101 images directly from Hugging Face — no download needed."""
    
    def __init__(self, url):
        self.archive = nra.BetaArchive(url)
        self.file_ids = [f for f in self.archive.file_ids() if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225]),
        ])
    
    def __len__(self):
        return len(self.file_ids)
    
    def __getitem__(self, idx):
        raw = self.archive.read_file(self.file_ids[idx])
        img = Image.open(io.BytesIO(raw)).convert('RGB')
        return self.transform(img)

# One line — and you're training on 5 GB of data
dataset = Food101Stream(
    "https://huggingface.co/datasets/zevatov/nra-food101/resolve/main/food-101.nra"
)
loader = DataLoader(dataset, batch_size=32, num_workers=4, shuffle=True)

print(f"✅ {len(dataset)} images ready. No download. No disk usage. Training NOW!")

for i, batch in enumerate(loader):
    # batch shape: [32, 3, 224, 224] — ready for ResNet, ViT, etc.
    if i % 100 == 0:
        print(f"  Batch {i}: {batch.shape}")
    if i >= 300:
        break
```

---

## 🏗️ How It Works

```
Your Model → PyTorch DataLoader → NRA (Rust) → HTTP Range GET → HF CDN
                                                      ↓
                                        Only the 4MB block you need
                                                      ↓
                                        Zstd decompress in RAM
                                                      ↓
                                        PIL Image → GPU Tensor
```

1. **Manifest-first:** The NRA manifest (file index) sits at the beginning of the archive. One HTTP request fetches it — giving O(1) lookup for all 101,000 files.
2. **Surgical HTTP Range:** When you request `images/pizza/1001116.jpg`, NRA looks up the exact byte offset in the manifest and fetches only the compressed 4MB block containing that file.
3. **Smart LRU Cache:** Fetched blocks are cached in RAM. Adjacent files in the same block are served instantly — zero network latency.

---

## 📊 Dataset Details

| Field | Value |
|-------|-------|
| **Source** | Food-101 (Bossard et al., 2014) |
| **Format** | `.nra` (Neural Ready Archive v4.5) |
| **Images** | 101,000 (variable resolution, avg ~384×384) |
| **Categories** | 101 food classes |
| **Archive size** | 4.7 GB |
| **Compression** | Zstd (level 15) + Content-Defined Chunking |
| **NRA SDK** | `pip install nra==1.0.3` |

---

## 📚 Learn More

- 🏠 **[GitHub Repository](https://github.com/zevatov/NRA)** — Full source code, benchmarks, whitepapers
- 📦 **[PyPI Package](https://pypi.org/project/nra/)** — `pip install nra`
- 🔬 **[CIFAR-10 NRA (170 MB)](https://huggingface.co/datasets/zevatov/nra-cifar10)** — Smaller demo dataset for quick testing
- 📄 **[Technical Whitepaper](https://github.com/zevatov/NRA/blob/main/docs/nra_whitepaper.md)** — Architecture deep-dive with benchmarks

## License

This dataset and the NRA format are released under the **MIT License**.
