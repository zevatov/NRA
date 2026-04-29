"""
NRA Cloud Streaming Example — Train PyTorch models without downloading datasets.

This script demonstrates how to use NRA's Zero-Download architecture
to stream training data directly from a remote .nra archive (e.g., on HuggingFace)
into a PyTorch DataLoader. No local disk space is consumed.

Requirements:
    pip install nra torch pillow

Usage:
    python stream_from_cloud.py --url https://huggingface.co/.../resolve/main/dataset.nra
"""

import argparse
import io
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

try:
    from PIL import Image
    import torchvision.transforms as transforms
    HAS_VISION = True
except ImportError:
    HAS_VISION = False

import nra


class NraCloudDataset(Dataset):
    """PyTorch Dataset that streams files from a remote .nra archive."""

    def __init__(self, url: str, transform=None):
        self.url = url
        self.transform = transform

        # The manifest downloads in ~150ms. The archive itself stays in the cloud.
        print(f"[NRA] Fetching manifest from {url} ...")
        t0 = time.time()
        self.file_ids = nra.CloudArchive(url).file_ids()
        print(f"[NRA] Manifest loaded: {len(self.file_ids)} files in {time.time() - t0:.2f}s")

        # Lazy archive handle (initialized per-worker in DataLoader)
        self._archive = None

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        # Lazy init: safe for multiprocessing DataLoader workers
        if self._archive is None:
            self._archive = nra.CloudArchive(self.url)

        file_id = self.file_ids[idx]

        # NRA downloads only the exact 4MB chunk via HTTP Range.
        # The Python GIL is released; Rust streams data in the background.
        raw_bytes = self._archive.read_file(file_id)

        if self.transform and HAS_VISION:
            image = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
            return self.transform(image), 0  # label placeholder
        else:
            return torch.tensor([len(raw_bytes)], dtype=torch.float32), 0


def main():
    parser = argparse.ArgumentParser(description="NRA Cloud Streaming Demo")
    parser.add_argument("--url", type=str, required=True,
                        help="URL to a remote .nra archive (e.g. HuggingFace resolve link)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=10,
                        help="Stop after N batches (for demo purposes)")
    args = parser.parse_args()

    # Build transform if torchvision is available
    transform = None
    if HAS_VISION:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    # === Zero-Download Dataset ===
    dataset = NraCloudDataset(args.url, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        num_workers=args.num_workers, shuffle=True)

    # Tiny demo model
    model = nn.Sequential(
        nn.Flatten(),
        nn.LazyLinear(128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\n[TRAIN] Starting training loop (max {args.max_batches} batches)...")
    t_start = time.time()

    for batch_idx, (data, target) in enumerate(loader):
        if batch_idx >= args.max_batches:
            break

        optimizer.zero_grad()
        output = model(data)
        target = torch.zeros(data.size(0), dtype=torch.long)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        elapsed = time.time() - t_start
        print(f"  Batch {batch_idx + 1}/{args.max_batches} | "
              f"Loss: {loss.item():.4f} | "
              f"Time: {elapsed:.2f}s | "
              f"Throughput: {(batch_idx + 1) * args.batch_size / elapsed:.0f} samples/s")

    total = time.time() - t_start
    print(f"\n[DONE] Processed {min(args.max_batches, batch_idx + 1)} batches in {total:.2f}s")
    print(f"[DONE] Zero bytes were written to your local SSD!")


if __name__ == "__main__":
    main()
