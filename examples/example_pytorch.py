import nra
import torch
from torch.utils.data import Dataset, DataLoader
import time
import os
import shutil
import random

class NRADataset(Dataset):
    """
    PyTorch Dataset that reads directly from a .nra archive.
    
    IMPORTANT: Uses lazy archive initialization so that each DataLoader worker
    (when num_workers > 0) opens its own file descriptor. This avoids corrupted
    reads caused by multiple processes sharing a single seek position after fork().
    """
    def __init__(self, archive_path):
        self.archive_path = archive_path
        # Open once in the main process just to read the file list (manifest only)
        temp_archive = nra.Archive(archive_path)
        self.file_ids = temp_archive.file_ids()
        self._archive = None  # Lazy: each worker will open its own handle
        
    def _get_archive(self):
        """Lazy init: ensures each forked worker gets its own file descriptor."""
        if self._archive is None:
            self._archive = nra.Archive(self.archive_path)
        return self._archive
        
    def __len__(self):
        return len(self.file_ids)
        
    def __getitem__(self, idx):
        archive = self._get_archive()
        file_id = self.file_ids[idx]
        raw_bytes = archive.read_file(file_id)
        
        # Here you would typically decode an image or parse JSON.
        # For this example, we just return the length of the data and a dummy tensor.
        return torch.tensor([len(raw_bytes)], dtype=torch.float32)

def main():
    # 1. Create a dummy dataset directory if it doesn't exist
    os.makedirs("/tmp/nra_dummy_data", exist_ok=True)
    print("Generating dummy data...")
    for i in range(100):
        with open(f"/tmp/nra_dummy_data/file_{i}.txt", "w") as f:
            f.write("A" * random.randint(1000, 5000))
            
    # 2. Pack it into .nra using our CLI tool (we'll just use the pre-built bench_tool archive if it exists, 
    # but for simplicity let's assume /tmp/nra_bench.nra exists from our previous benchmark)
    archive_path = "/tmp/nra_bench.nra"
    
    if not os.path.exists(archive_path):
        print(f"Error: Could not find {archive_path}. Please run the benchmark first.")
        return

    print(f"\n🚀 Initializing NRADataset from {archive_path}...")
    dataset = NRADataset(archive_path)
    print(f"Dataset contains {len(dataset)} files.")
    
    # 3. Create a PyTorch DataLoader
    # Safe with num_workers > 0: each worker lazily opens its own file descriptor
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0)
    
    print("\nTraining Loop Simulation:")
    start_time = time.time()
    
    batches = 0
    for batch_idx, data in enumerate(dataloader):
        # 'data' is a batch of tensors returned by __getitem__
        if batch_idx == 0:
            print(f"  First batch shape: {data.shape} (batch_size, 1)")
            
        # Simulate a fast training step
        time.sleep(0.01)
        batches += 1
        
    end_time = time.time()
    print(f"\n✅ Finished processing {len(dataset)} files ({batches} batches).")
    print(f"Total time: {end_time - start_time:.2f} seconds.")
    print("Zero unpacking was needed! Everything streamed straight from the .nra archive.")

if __name__ == "__main__":
    main()
