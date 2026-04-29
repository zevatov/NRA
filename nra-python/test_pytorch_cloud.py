import nra
import torch
from torch.utils.data import Dataset, DataLoader
import time

class CloudDataset(Dataset):
    """
    PyTorch Dataset that streams files directly from an AWS S3 / HTTP server 
    using Zero-Download Async streaming (BETA format).
    """
    def __init__(self, url):
        self.url = url
        # Main process fetches manifest once
        temp_archive = nra.CloudArchive(url)
        self.file_ids = temp_archive.file_ids()
        self._archive = None
        
    def _get_archive(self):
        # Lazy initialization for each PyTorch worker thread/process
        if self._archive is None:
            self._archive = nra.CloudArchive(self.url)
        return self._archive
        
    def __len__(self):
        return len(self.file_ids)
        
    def __getitem__(self, idx):
        archive = self._get_archive()
        file_id = self.file_ids[idx]
        # This seamlessly drops the GIL and runs tokio under the hood to fetch the HTTP Range!
        raw_bytes = archive.read_file(file_id)
        
        return torch.tensor([len(raw_bytes)], dtype=torch.float32)

def main():
    url = "http://localhost:8000/heavy_beta.nra"
    
    print(f"\n☁️ Initializing PyTorch Cloud Dataset from: {url}")
    try:
        dataset = CloudDataset(url)
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you are running 'npx serve /tmp -p 8000' in the background!")
        return
        
    print(f"📦 Dataset contains {len(dataset)} files.")
    
    workers = 4
    batch_size = 256
    
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=workers,
        prefetch_factor=2 if workers > 0 else None
    )
    
    print(f"\n🚀 Starting Distributed Cloud Training Simulation")
    print(f"Workers: {workers} | Batch Size: {batch_size}")
    
    start_time = time.time()
    total_files = 0
    
    for epoch in range(1):
        epoch_start = time.time()
        batches = 0
        
        for batch_idx, data in enumerate(dataloader):
            batches += 1
            total_files += data.shape[0]
            if batch_idx % 5 == 0:
                print(f"  Downloaded {batches} batches ({total_files} files) over HTTP...")
            
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1} finished: Processed {total_files} files in {epoch_time:.2f}s")
        print(f"  Cloud Streaming Speed: {total_files / epoch_time:.0f} files/sec")
        
    print(f"\n✅ Finished processing!")
    print(f"Zero files were downloaded to the hard drive. Everything streamed directly into RAM.")

if __name__ == "__main__":
    main()
