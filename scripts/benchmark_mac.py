import os
import time
import shutil
import urllib.request
import tarfile
import threading
import http.server
import socketserver
from pathlib import Path
import RangeHTTPServer
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

import nra
import nra_datasets

# Configuration
DATA_DIR = Path(__file__).resolve().parent.parent / ".benchmark_data"
TAR_FILE = DATA_DIR / "food-101.tar.gz"
EXTRACT_DIR = DATA_DIR / "food-101-extracted"
NRA_FILE = DATA_DIR / "food-101.nra"
URL = "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
HTTP_PORT = 8081

def download_with_progress(url, dest_path):
    if dest_path.exists():
        print(f"✅ {dest_path.name} already exists.")
        return
        
    print(f"⬇️ Downloading {url} (~5GB)...")
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=dest_path, reporthook=t.update_to)
    print("✅ Download complete.")

def serve_directory_in_background(directory, port):
    import subprocess
    print(f"🌐 Starting RangeHTTPServer in subprocess on port {port}...")
    proc = subprocess.Popen(["python3", "-m", "RangeHTTPServer", str(port)], cwd=str(directory), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    time.sleep(1) # wait for server to start
    return proc



def train_epoch(dataloader, model, device, name):
    print(f"\n🚀 Starting 1 Epoch Training: {name}")
    model.to(device)
    model.train()
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    start_time = time.time()
    first_batch_time = None
    
    # We will only run a few batches to measure data loading / MPS speed, no need to train 101k images fully
    MAX_BATCHES = 100
    
    pbar = tqdm(total=MAX_BATCHES, desc=f"Training {name}")
    
    for batch_idx, (data, target) in enumerate(dataloader):
        if first_batch_time is None:
            first_batch_time = time.time() - start_time
            print(f"\n⏱️ TTFB (Time To First Batch): {first_batch_time:.4f} seconds")
            
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        pbar.update(1)
        if batch_idx >= MAX_BATCHES - 1:
            break
            
    pbar.close()
    epoch_time = time.time() - start_time
    
    # Calculate images per second
    total_images = MAX_BATCHES * dataloader.batch_size
    throughput = total_images / epoch_time
    
    print(f"✅ Finished {name} - Total Time: {epoch_time:.2f}s | Throughput: {throughput:.2f} img/sec")
    return first_batch_time, epoch_time, throughput

class CustomImageDatasetWrapper(torch.utils.data.Dataset):
    """Wraps NRA BetaArchive to mimic ImageFolder format for our test"""
    def __init__(self, archive, transform=None):
        self.archive = archive
        self.transform = transform
        
        # NRA archive contains raw bytes, we need to decode them.
        # Filter only image files
        self.files = [f for f in self.archive.file_ids() if f.endswith(('.jpg', '.png', '.jpeg'))]
        
        # Extract classes from paths (assuming 'images/class_name/file.jpg')
        classes = sorted(list(set([f.split('/')[-2] for f in self.files if '/' in f])))
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_id = self.files[idx]
        raw_bytes_list = self.archive.read_file(file_id)
        raw_bytes = bytes(raw_bytes_list)
        
        import io
        from PIL import Image
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
            
        class_name = file_id.split('/')[-2]
        label = self.class_to_idx[class_name]
        
        return img, label

def main():
    print("==================================================")
    print("  NRA vs Tarball - macOS M-Series Benchmark")
    print("==================================================")
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"💻 PyTorch Device: {device}")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_with_progress(URL, TAR_FILE)
    
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    model = torch.nn.Sequential(
        torch.nn.Flatten(),
        torch.nn.Linear(224*224*3, 101) # Simple model to keep compute low and I/O high
    )

    results = {}

    # ----------------------------------------------------
    # METHOD 1: Legacy Tarball
    # ----------------------------------------------------
    print("\n--- [Method 1] Legacy Tarball ---")
    if not EXTRACT_DIR.exists():
        print("📦 Extracting 101,000 files from tar.gz (this is the pain point)...")
        start_extract = time.time()
        with tarfile.open(TAR_FILE, "r:gz") as tar:
            tar.extractall(path=EXTRACT_DIR)
        extract_time = time.time() - start_extract
        print(f"⏱️ Extraction took: {extract_time:.2f} seconds")
    else:
        print("✅ Already extracted.")
        extract_time = 0

    # Locate the images folder inside extracted content
    img_dir = list(EXTRACT_DIR.rglob("images"))
    if img_dir:
        img_dir = img_dir[0]
    else:
        img_dir = EXTRACT_DIR

    legacy_dataset = datasets.ImageFolder(img_dir, transform=transform)
    legacy_loader = DataLoader(legacy_dataset, batch_size=64, shuffle=True, num_workers=0)
    
    ttfb1, time1, tp1 = train_epoch(legacy_loader, model, device, "Legacy ImageFolder")
    results['Legacy'] = {'Extract': extract_time, 'TTFB': ttfb1, 'Epoch': time1, 'Throughput': tp1}


    # ----------------------------------------------------
    # METHOD 2: NRA Converter
    # ----------------------------------------------------
    print("\n--- [Method 2] NRA Convert ---")
    if not NRA_FILE.exists():
        print("📦 Converting tar.gz directly to .nra...")
        start_convert = time.time()
        # Call nra-cli to convert
        os.system(f"cd {Path(__file__).resolve().parent.parent / 'nra-cli'} && cargo run --release -- convert --input {TAR_FILE} --output {NRA_FILE}")
        convert_time = time.time() - start_convert
        print(f"⏱️ Conversion took: {convert_time:.2f} seconds")
    else:
        print("✅ Already converted.")
        convert_time = 0

    # Load NRA
    nra_local = nra.BetaArchive(str(NRA_FILE))
    nra_local_dataset = CustomImageDatasetWrapper(nra_local, transform=transform)
    nra_local_loader = DataLoader(nra_local_dataset, batch_size=64, shuffle=True, num_workers=0)

    ttfb2, time2, tp2 = train_epoch(nra_local_loader, model, device, "NRA Local Read")
    results['NRA Convert'] = {'Convert': convert_time, 'TTFB': ttfb2, 'Epoch': time2, 'Throughput': tp2}

    # ----------------------------------------------------
    # METHOD 3: NRA Cloud Streaming
    # ----------------------------------------------------
    print("\n--- [Method 3] NRA Cloud Streaming ---")
    httpd = serve_directory_in_background(DATA_DIR, HTTP_PORT)
    
    # Load via CloudArchive (simulating zero-download S3 streaming)
    url = f"http://127.0.0.1:{HTTP_PORT}/{NRA_FILE.name}"
    nra_cloud = nra.CloudArchive(url)
    nra_cloud_dataset = CustomImageDatasetWrapper(nra_cloud, transform=transform)
    nra_cloud_loader = DataLoader(nra_cloud_dataset, batch_size=64, shuffle=True, num_workers=0)

    ttfb3, time3, tp3 = train_epoch(nra_cloud_loader, model, device, "NRA Cloud Stream")
    results['NRA Stream'] = {'Download': 0, 'TTFB': ttfb3, 'Epoch': time3, 'Throughput': tp3}
    
    if httpd:
        httpd.terminate()

    # ----------------------------------------------------
    # REPORTING
    # ----------------------------------------------------
    print("\n==================================================")
    print(" 🏆 FINAL RESULTS")
    print("==================================================")
    print(f"Legacy Tarball : Extract={results['Legacy']['Extract']:.2f}s | TTFB={results['Legacy']['TTFB']:.4f}s | TP={results['Legacy']['Throughput']:.1f} img/s")
    print(f"NRA Local      : Convert={results['NRA Convert']['Convert']:.2f}s | TTFB={results['NRA Convert']['TTFB']:.4f}s | TP={results['NRA Convert']['Throughput']:.1f} img/s")
    print(f"NRA Streaming  : Prep=0.00s | TTFB={results['NRA Stream']['TTFB']:.4f}s | TP={results['NRA Stream']['Throughput']:.1f} img/s")

if __name__ == "__main__":
    main()
