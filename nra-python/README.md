<div align="center">
  <h1>🧬 NRA (Neural Ready Archive)</h1>
  <p><b>The 21st Century Data Format for the AI Era. Forget about <code>tar.gz</code> and <code>zip</code>.</b></p>

  [![PyPI version](https://img.shields.io/pypi/v/nra.svg)](https://pypi.org/project/nra/)
  [![Rust](https://img.shields.io/badge/rust-1.80+-blue.svg)](https://www.rust-lang.org)
  [![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
  [![HuggingFace](https://img.shields.io/badge/🤗_HuggingFace-Dataset-yellow)](https://huggingface.co/datasets/zevatov/nra-cifar10)
</div>

<br/>

Traditional archiving formats (`ZIP`, `Tar.gz`) were designed in the 90s for floppy disks. Today, they are the main **bottleneck** of IT infrastructure. They force you to download entire 500GB datasets, cannot stream individual files from the cloud, and cause extremely expensive GPUs to sit idle waiting for data.

**NRA (Neural Ready Archive)** is a next-generation binary format. It combines enterprise-grade deduplication, ultra-fast Zstd compression, and B+ Tree indexing so you can train neural networks directly from the public cloud.

---

## ⚡ Why Legacy Formats Are Dead (Our Benchmarks)

We ran a stress test on 60,000 small files (CIFAR-10):

| Format | Packing Time | "Cold Start" Speed (Streaming) |
| :--- | :---: | :---: |
| 🛑 **Tar.gz** | 38.0 seconds | ~30 minutes (Requires full download) |
| 🛑 **ZIP** | 13.4 seconds | Impossible (Requires full download) |
| 🏆 **NRA** | **3.3 seconds** (11.5x faster) | **150 milliseconds** (Zero-Download) |

NRA extracts 100% of your CPU's multi-core power (thanks to Rust Rayon) and glues files into 4MB Solid blocks, guaranteeing instant O(1) random access.

---

## 🚀 Try It Now: Train Online Without Downloading

### Use our ready-made dataset on Hugging Face

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
        # The manifest downloads in 150ms. The archive itself stays in the cloud!
        self.file_ids = nra.CloudArchive(url).file_ids()
        self._archive = None
        
    def __len__(self):
        return len(self.file_ids)
        
    def __getitem__(self, idx):
        if self._archive is None:
            self._archive = nra.CloudArchive(self.url)
        raw_bytes = self._archive.read_file(self.file_ids[idx])
        return torch.tensor([len(raw_bytes)], dtype=torch.float32)

# 🤗 Our ready-made dataset on Hugging Face (NRA format)
dataset = NraStreamDataset(
    "https://huggingface.co/datasets/zevatov/nra-cifar10/resolve/main/cifar10.nra"
)
loader = DataLoader(dataset, batch_size=256, num_workers=4)

for batch in loader:
    # Training starts at second 0. Zero bytes on your SSD!
    pass
```

> 🤗 **[Open the dataset on Hugging Face →](https://huggingface.co/datasets/zevatov/nra-cifar10)**

### Convert ANY existing dataset on-the-fly

Already have a `tar.gz` or `zip` dataset? NRA can **convert it live** — still faster than downloading:

```bash
# Convert tar.gz → NRA in 1-3 seconds (Zero-Disk I/O, pure RAM)
nra-cli convert --input dataset.tar.gz --output dataset.nra
```

> Converting `Tar.gz → NRA` takes **0.71 seconds** vs **8.35 seconds** to unpack to SSD. **11x faster.**

---

## 🏗️ How Cloud Architecture Works (Zero-Disk I/O)

```mermaid
sequenceDiagram
    participant PT as PyTorch (Python)
    participant Core as NRA Core (Rust)
    participant S3 as HuggingFace / S3
    
    PT->>Core: Give me "image_99.png"
    Note over Core: B+ Tree manifest lookup (O(1))
    Core->>S3: HTTP GET (Range: bytes=4000-8000)
    S3-->>Core: Returns compressed 4MB Solid Block
    Note over Core: Zstd Decompression in RAM
    Core-->>PT: Returns raw bytes (Zero-Disk I/O)
    Note over PT: GPU instantly receives the tensor
```

---

## 🛠️ The NRA Ecosystem

1. **Python SDK (`pip install nra==1.0.0`):** Integration into PyTorch and TensorFlow.
2. **NRA CLI (`cargo install nra-cli`):** Console utility for servers.
3. **NRA GUI:** Desktop application for visual archive management. *(In development: [zevatov/nra-manager-pro](https://github.com/zevatov/nra-manager-pro))*
4. **FUSE Mount:** Mount `.nra` archives as virtual drives (`nra-cli mount`).
5. **🤗 HuggingFace:** [zevatov/nra-cifar10](https://huggingface.co/datasets/zevatov/nra-cifar10) — ready-to-use NRA dataset.

---

## 🗺️ Roadmap

| Milestone | Status |
|-----------|--------|
| **1.0** Core Engine (Zstd/LZ4, B+ Tree, CDC, AES-256) | ✅ Released |
| **1.0** Python SDK + PyTorch DataLoader | ✅ Released |
| **1.0** CLI (pack, extract, convert, stream, mount) | ✅ Released |
| **1.1** NRA Manager Pro (GUI) | 🔧 In Progress |
| **1.2** Delta Updates (append without rebuild) | 📋 Planned |
| **1.3** NRA CDN (edge-caching proxy) | 📋 Planned |
| **1.4** NRA Registry (private dataset hub) | 📋 Planned |
| **1.5** Streaming Converter (remote tar.gz → NRA live) | 📋 Planned |
| **2.0** Multi-platform PyPI Wheels | 📋 Planned |

---

## 📚 Deep Documentation

For full architectural documentation, whitepapers, and archiving benchmarks, visit our [Official GitHub Repository](https://github.com/zevatov/NRA).

## License
The `nra-core`, `nra-cli`, and `nra-python` components are distributed under the **MIT** license.
