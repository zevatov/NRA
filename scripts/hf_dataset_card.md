---
license: mit
task_categories:
  - image-classification
size_categories:
  - 10K<n<100K
---

# CIFAR-10 (NRA Format)

This is the classic CIFAR-10 dataset packed in [Neural Ready Archive](https://github.com/nra-team/nra) format.

## Quick Start

```bash
pip install nra
```

```python
import nra_datasets

ds = nra_datasets.load("nra-team/cifar10-nra")
for item in ds:
    img = item["bytes"]  # raw PNG bytes
    label = item["file_id"].split("/")[0]  # "airplane", "automobile", etc.
```

## Why NRA?
- ⚡ **Smaller** than tar.gz thanks to CDC deduplication and solid Zstd blocks
- ☁️ **Zero-download** — start training instantly via HTTP Range streaming without downloading the dataset first
- 🔒 **AES-256 encryption** support for sensitive enterprise data
