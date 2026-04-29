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

[![PyPI](https://img.shields.io/badge/pip_install_nra-1.0.0-blue)](https://pypi.org/project/nra/1.0.0/)
[![GitHub](https://img.shields.io/badge/GitHub-zevatov%2FNRA-black?logo=github)](https://github.com/zevatov/NRA)
[![License](https://img.shields.io/badge/license-MIT-green)](https://opensource.org/licenses/MIT)

</div>

This dataset contains **CIFAR-10** (60,000 images) packaged in the **NRA (Neural Ready Archive)** format — a next-generation binary format built in Rust for the AI era.

## 🚀 Why This Matters

**You DO NOT need to download this dataset.** NRA streams data directly into your PyTorch `DataLoader` via HTTP Range requests. Only the exact 4MB blocks your model needs are fetched on-the-fly.

| Metric | Traditional (tar.gz) | NRA (this dataset) |
|--------|---------------------|-------------------|
| Time to first batch | ~30 min (download + unpack) | **150 ms** |
| Local disk space | 170 MB | **0 bytes** |
| Random file access | Impossible | **O(1) instant** |

---

## ⚡ Quick Start: Train in 30 Seconds

### Google Colab / Jupyter / Local

```bash
pip install nra==1.0.0 torch
```

```python
import nra
import torch
from torch.utils.data import Dataset, DataLoader

class NraStreamDataset(Dataset):
    def __init__(self, url):
        self.url = url
        # The manifest downloads in ~150ms. The archive stays on Hugging Face!
        self.file_ids = nra.CloudArchive(url).file_ids()
        self._archive = None
        
    def __len__(self):
        return len(self.file_ids)
        
    def __getitem__(self, idx):
        if self._archive is None:
            self._archive = nra.CloudArchive(self.url)
            
        file_id = self.file_ids[idx]
        
        # NRA fetches only the exact chunk via HTTP Range.
        # The GIL is released; Rust streams data at max speed.
        raw_bytes = self._archive.read_file(file_id)
        
        # For real training: decode the image
        # img = Image.open(io.BytesIO(raw_bytes))
        # tensor = transforms.ToTensor()(img)
        return torch.tensor([len(raw_bytes)], dtype=torch.float32)

# Point directly to the .nra file in this repository
dataset = NraStreamDataset(
    "https://huggingface.co/datasets/zevatov/nra-cifar10/resolve/main/cifar10.nra"
)
loader = DataLoader(dataset, batch_size=256, num_workers=4)

print(f"✅ Loaded {len(dataset)} items. Training starts NOW — zero bytes on your SSD!")

for batch in loader:
    # Your model trains immediately. No waiting, no downloading.
    pass
```

---

## 🛠️ CLI: Inspect, Stream, or Mount

If you prefer working from the terminal:

```bash
# Install the Rust CLI
cargo install nra-cli
```

```bash
# Stream a single file without downloading the archive
nra-cli stream-beta \
  --url https://huggingface.co/datasets/zevatov/nra-cifar10/resolve/main/cifar10.nra \
  --file-id image_001.png \
  --out ./image_001.png

# Mount the remote archive as a local folder (Mac/Linux FUSE)
nra-cli mount \
  --input https://huggingface.co/datasets/zevatov/nra-cifar10/resolve/main/cifar10.nra \
  --mountpoint ./virtual_dataset

# Your files appear as a regular folder — but they're streaming from Hugging Face!
ls ./virtual_dataset/
```

---

## 🏗️ How It Works

```
PyTorch DataLoader → NRA Core (Rust) → HTTP Range GET → Hugging Face CDN
                                                              ↓
                                              Only the 4MB block you need
                                                              ↓
                                              Zstd decompress in RAM
                                                              ↓
                                              Raw bytes → GPU tensor
```

NRA uses:
- **B+ Tree Manifest** for O(1) file lookups (no scanning)
- **4MB Solid Blocks** with Zstd compression
- **HTTP Range Requests** to fetch only the exact bytes needed
- **Content-Defined Chunking (CDC)** for automatic deduplication

---

## 🔄 Convert Your Own Datasets

Have a `tar.gz` or `zip` dataset? Convert it to NRA in seconds:

```bash
# Unpack and repack as NRA
nra-cli pack-beta --input ./your_dataset/ --output your_dataset.nra --dictionary --zstd-level 15

# Upload to your own HF dataset
# Then use the same streaming code above with your URL!
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
| **NRA SDK** | `pip install nra==1.0.0` |

---

## 📚 Learn More

- 🏠 **[GitHub Repository](https://github.com/zevatov/NRA)** — Full source code, benchmarks, whitepapers
- 📦 **[PyPI Package](https://pypi.org/project/nra/1.0.0/)** — `pip install nra==1.0.0`
- 📄 **[Technical Whitepaper](https://github.com/zevatov/NRA/blob/main/docs/nra_whitepaper.md)** — Architecture deep-dive with 8 benchmark charts

## License

This dataset and the NRA format are released under the **MIT License**.
