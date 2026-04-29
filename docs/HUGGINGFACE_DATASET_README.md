---
license: mit
tags:
- nra
- neural-ready-archive
- streaming
- zero-download
---

# 🧬 Neural Ready Archive (NRA) Dataset

This dataset is packaged in the **NRA (Neural Ready Archive)** format.
NRA is a next-generation binary format optimized for modern AI infrastructure. It supports **Zero-Download HTTP Streaming**, O(1) random access, and Content-Defined Deduplication.

**You DO NOT need to download this dataset to your hard drive to train your model!**

---

## ⚡ Quick Start: Train Online (Google Colab / Local)

You can train your PyTorch models directly from this Hugging Face repository. NRA will stream only the required 4MB blocks on-the-fly, bypassing your local SSD completely.

**1. Install the NRA Python SDK:**
```bash
pip install nra torch
```

**2. Load the dataset into your PyTorch DataLoader:**
```python
import nra
import torch
from torch.utils.data import Dataset, DataLoader

class HuggingFaceStreamDataset(Dataset):
    def __init__(self, url):
        self.url = url
        # The manifest downloads instantly. The archive itself stays on Hugging Face!
        self.file_ids = nra.CloudArchive(url).file_ids()
        self._archive = None
        
    def __len__(self):
        return len(self.file_ids)
        
    def __getitem__(self, idx):
        if self._archive is None:
            # Lazy initialization for DataLoader workers
            self._archive = nra.CloudArchive(self.url)
            
        file_id = self.file_ids[idx]
        
        # NRA downloads only the exact chunk needed via HTTP Range requests.
        # The Python GIL is released; Rust streams the data in the background.
        raw_bytes = self._archive.read_file(file_id)
        
        # Decode your raw_bytes (e.g., Image.open(io.BytesIO(raw_bytes)))
        # and return the tensor
        return torch.tensor([len(raw_bytes)], dtype=torch.float32)

# Important: Use the "resolve" URL to point directly to the raw .nra file
# Example: "https://huggingface.co/datasets/your-username/your-dataset/resolve/main/dataset.nra"
dataset_url = "REPLACE_WITH_YOUR_HUGGINGFACE_RESOLVE_URL"

dataset = HuggingFaceStreamDataset(dataset_url)
loader = DataLoader(dataset, batch_size=256, num_workers=4)

print(f"Loaded dataset with {len(dataset)} items.")

for batch in loader:
    # Training starts immediately! Zero bytes are stored on your local SSD!
    pass
```

---

## 🛠️ Inspect or Download Locally (CLI)

If you prefer to download the archive or inspect it locally, use the NRA CLI built in Rust.

**Install the CLI:**
```bash
cargo install nra-cli
```

**Stream a single file without downloading the archive:**
```bash
nra-cli stream-beta \
  --url REPLACE_WITH_YOUR_HUGGINGFACE_RESOLVE_URL \
  --file-id image_001.png \
  --out ./image_001.png
```

**Mount the remote archive as a local folder (Mac/Linux FUSE):**
```bash
nra-cli mount --input REPLACE_WITH_YOUR_HUGGINGFACE_RESOLVE_URL --mountpoint ./virtual_dataset
```

---

## 📚 About NRA

NRA guarantees:
- **Zero GPU Idle Time:** Your GPUs no longer wait for the hard drive. Data is fed directly into memory at CPU speeds.
- **O(1) Random Access:** Unlike `tar.gz` or `WebDataset`, you can instantly access any file globally without scanning the archive.
- **Deduplication:** Repeated files are deduplicated automatically, reducing dataset sizes by up to 80%.

To learn more about the NRA format, visit the [Official GitHub Repository](https://github.com/zevatov/NRA) or the [PyPI package page](https://pypi.org/project/nra/).
