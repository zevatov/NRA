import nra
import torch
from torch.utils.data import Dataset, DataLoader
import time
import os

class NRABetaDataset(Dataset):
    """
    PyTorch Dataset that reads directly from a CDC-deduplicated .nra BETA archive.
    
    IMPORTANT: Uses lazy archive initialization so that each DataLoader worker
    (when num_workers > 0) opens its own file descriptor. This avoids corrupted
    reads caused by multiple processes sharing a single seek position after fork().
    """
    def __init__(self, archive_path):
        self.archive_path = archive_path
        # Open once in the main process just to read the manifest
        temp_archive = nra.BetaArchive(archive_path)
        self.file_ids = temp_archive.file_ids()
        self._archive = None  # Lazy: each worker will open its own handle
        
    def _get_archive(self):
        if self._archive is None:
            self._archive = nra.BetaArchive(self.archive_path)
        return self._archive
        
    def __len__(self):
        return len(self.file_ids)
        
    def __getitem__(self, idx):
        archive = self._get_archive()
        file_id = self.file_ids[idx]
        raw_bytes = archive.read_file(file_id)
        
        # Simulating parsing/decoding step:
        # We just return the length of the file as a tensor
        return torch.tensor([len(raw_bytes)], dtype=torch.float32)

def main():
    archive_path = "/tmp/heavy_beta.nra" # Created in our previous benchmark
    
    if not os.path.exists(archive_path):
        print(f"Error: Archive {archive_path} not found.")
        return

    print(f"\n🚀 Initializing PyTorch Dataset from BETA Archive: {archive_path}")
    dataset = NRABetaDataset(archive_path)
    print(f"Dataset contains {len(dataset)} versioned configs/files.")
    
    # Use multiple workers to prove lazy init is thread/process safe
    workers = 4
    batch_size = 256
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=workers,
        prefetch_factor=2 if workers > 0 else None
    )
    
    print(f"\n🔥 Starting High-Speed Training Loop Simulation")
    print(f"Workers: {workers} | Batch Size: {batch_size}")
    
    start_time = time.time()
    total_files = 0
    
    # We iterate over the entire dataset twice to measure epoch time
    for epoch in range(2):
        epoch_start = time.time()
        batches = 0
        
        for batch_idx, data in enumerate(dataloader):
            # 'data' is the tensor returned by __getitem__
            batches += 1
            total_files += data.shape[0]
            
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1}: Processed {batches} batches ({total_files} files) in {epoch_time:.2f}s")
        print(f"  Speed: {total_files / epoch_time:.0f} files/sec")
        total_files = 0 # reset for epoch 2
        
    total_time = time.time() - start_time
    print(f"\n✅ Finished processing!")
    print(f"Everything streamed straight from the 35 MB compressed .nra archive.")
    print(f"Total time for 2 epochs: {total_time:.2f} seconds.")

if __name__ == "__main__":
    main()
