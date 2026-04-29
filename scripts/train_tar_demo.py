import os
import time
import requests
import tarfile
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import io
import sys

class RawDiskDataset(Dataset):
    def __init__(self, directory):
        self.directory = directory
        self.files = [f for f in os.listdir(directory) if f.endswith('.png') and not f.startswith('._')]
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        path = os.path.join(self.directory, file_name)
        image = Image.open(path).convert('RGB')
        tensor = self.transform(image)
        # Parse label from name (e.g. 3_00001.png)
        label = int(file_name.split("_")[0])
        return tensor, label

def main():
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "http://127.0.0.1:8000/cifar10.tar.gz"

    print(f"=== TAR.GZ CLASSIC DOWNLOAD DEMO ===")
    print(f"Target: {url}")
    
    start_time = time.time()
    
    # 1. Download the archive
    archive_path = "temp_cifar10.tar.gz"
    print("Downloading tar.gz archive...")
    r = requests.get(url, stream=True)
    with open(archive_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    
    download_time = time.time() - start_time
    print(f"Download complete in {download_time:.3f}s")
    
    # 2. Extract the archive
    extract_dir = "temp_cifar10_raw"
    os.makedirs(extract_dir, exist_ok=True)
    print("Extracting tar.gz archive to disk...")
    extract_start = time.time()
    with tarfile.open(archive_path, 'r:gz') as tar:
        tar.extractall(path=extract_dir)
        
    extract_time = time.time() - extract_start
    print(f"Extraction complete in {extract_time:.3f}s")
    
    # Folder inside extraction
    data_dir = os.path.join(extract_dir, "cifar10_raw")
    if not os.path.exists(data_dir):
        data_dir = extract_dir
        
    dataset = RawDiskDataset(data_dir)
    loader = DataLoader(dataset, batch_size=256, num_workers=4, shuffle=True)
    
    ttfb = time.time() - start_time
    print(f"Found {len(dataset)} images.")
    print(f"Time to First Batch (TTFB) Setup Complete: {ttfb:.3f}s")
    
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting epoch 1 on {device}...")
    total_images = 0
    epoch_start = time.time()
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_images += data.size(0)
        
        if batch_idx % 10 == 0:
            elapsed = time.time() - epoch_start
            throughput = total_images / elapsed if elapsed > 0 else 0
            print(f"Batch {batch_idx:03d} | Loss: {loss.item():.4f} | Throughput: {throughput:.1f} img/s")

if __name__ == "__main__":
    main()
