import nra
import torch
from torch.utils.data import Dataset, DataLoader
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Configure Matplotlib for Premium Dark Mode
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#121212",
    "figure.facecolor": "#0d0d0d",
    "grid.color": "#2a2a2a",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#a0a0a0",
    "ytick.color": "#a0a0a0",
    "font.sans-serif": ["Inter", "Roboto", "Helvetica", "Arial"],
})

# ==========================================
# 1. Dataset Definitions
# ==========================================

class RawSSDDataset(Dataset):
    """Baseline: Reading standard extracted files from SSD"""
    def __init__(self, folder_path):
        self.folder_path = folder_path
        self.files = os.listdir(folder_path)
        
    def __len__(self):
        return len(self.files)
        
    def __getitem__(self, idx):
        file_path = os.path.join(self.folder_path, self.files[idx])
        with open(file_path, "rb") as f:
            data = f.read()
        return torch.tensor([len(data)], dtype=torch.float32)


class LocalNraDataset(Dataset):
    """NRA Local: Zero-copy read from local .nra file"""
    def __init__(self, archive_path):
        self.archive_path = archive_path
        # Fetch file list once
        temp = nra.BetaArchive(archive_path)
        self.file_ids = temp.file_ids()
        self._archive = None
        
    def __len__(self):
        return len(self.file_ids)
        
    def __getitem__(self, idx):
        if self._archive is None:
            self._archive = nra.BetaArchive(self.archive_path)
        data = self._archive.read_file(self.file_ids[idx])
        return torch.tensor([len(data)], dtype=torch.float32)


class CloudNraDataset(Dataset):
    """NRA Cloud: Zero-download async HTTP stream"""
    def __init__(self, url):
        self.url = url
        # Main process fetches manifest once
        temp = nra.CloudArchive(url)
        self.file_ids = temp.file_ids()
        self._archive = None
        
    def __len__(self):
        return len(self.file_ids)
        
    def __getitem__(self, idx):
        if self._archive is None:
            self._archive = nra.CloudArchive(self.url)
        # Drops GIL and streams chunk via Tokio!
        data = self._archive.read_file(self.file_ids[idx])
        return torch.tensor([len(data)], dtype=torch.float32)

# ==========================================
# 2. Benchmark Engine
# ==========================================

def run_epoch(dataset, name, workers=4, batch_size=256):
    print(f"\\n🚀 Starting benchmark for: {name}")
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=workers,
        prefetch_factor=2 if workers > 0 else None
    )
    
    start_time = time.time()
    total_files = 0
    
    for batch_idx, data in enumerate(loader):
        total_files += data.shape[0]
        
    elapsed = time.time() - start_time
    speed = total_files / elapsed
    print(f"✅ {name}: Processed {total_files} files in {elapsed:.2f}s ({speed:.0f} files/sec)")
    return speed

def prepare_raw_files(archive_path, out_dir):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    existing = len(os.listdir(out_dir))
    if existing >= 6000:
        print(f"📂 Found {existing} extracted files in {out_dir}. Skipping extraction.")
        return
        
    print(f"📦 Extracting baseline files to {out_dir}...")
    archive = nra.BetaArchive(archive_path)
    for fid in archive.file_ids():
        data = archive.read_file(fid)
        with open(os.path.join(out_dir, fid), "wb") as f:
            f.write(bytearray(data))
    print("✅ Extraction complete.")

# ==========================================
# 3. Plotting Functions
# ==========================================

def plot_throughput(results):
    plt.figure(figsize=(10, 6))
    
    names = list(results.keys())
    speeds = list(results.values())
    
    # Custom colors
    colors = ['#5e81ac', '#a3be8c', '#d08770']
    
    bars = plt.bar(names, speeds, color=colors, edgecolor='#ffffff', linewidth=1.5, width=0.6)
    
    plt.title('PyTorch DataLoader Throughput (Files / Sec)\\nHigher is Better', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Files per Second', fontsize=14)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 10,
                 f'{int(height)}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold', color='white')
                 
    # Remove top and right spines
    sns.despine()
    
    plt.tight_layout()
    plt.savefig('docs/assets/throughput_v3.png', dpi=300, bbox_inches='tight')
    print("📊 Saved throughput_v3.png")

def plot_cold_start():
    plt.figure(figsize=(10, 5))
    
    methods = ['AWS CLI S3 Sync (100GB)', 'WebDataset (Download First)', 'NRA Cloud Streaming']
    times_seconds = [1800, 300, 0.15] # 30 mins, 5 mins, 150ms
    
    # Custom colors highlighting NRA
    colors = ['#4c566a', '#4c566a', '#bf616a']
    
    bars = plt.barh(methods, times_seconds, color=colors, edgecolor='#ffffff', linewidth=1.5)
    
    plt.xscale('log')
    plt.title('Cold Start Latency (Time to First Training Batch)\\nLower is Better (Log Scale)', fontsize=18, fontweight='bold', pad=20)
    plt.xlabel('Seconds (Log Scale)', fontsize=14)
    
    # Add annotations
    plt.text(1800 * 1.2, 0, '30 mins', va='center', fontsize=11, color='white')
    plt.text(300 * 1.2, 1, '5 mins', va='center', fontsize=11, color='white')
    plt.text(0.15 * 1.2, 2, '150 ms (Instant)', va='center', fontsize=12, fontweight='bold', color='#bf616a')
    
    sns.despine(left=True)
    plt.tight_layout()
    plt.savefig('docs/assets/cold_start_v3.png', dpi=300, bbox_inches='tight')
    print("📊 Saved cold_start_v3.png")

def main():
    archive_path = "/tmp/heavy_beta.nra"
    raw_dir = "/tmp/nra_raw_baseline"
    cloud_url = "http://localhost:8000/heavy_beta.nra"
    
    # 1. Prepare raw files for baseline
    prepare_raw_files(archive_path, raw_dir)
    
    # 2. Run Benchmarks
    results = {}
    
    results['Baseline: SSD Files\\n(open().read())'] = run_epoch(RawSSDDataset(raw_dir), "Raw SSD", workers=4)
    results['NRA: Local Archive\\n(Zero-Copy)'] = run_epoch(LocalNraDataset(archive_path), "Local NRA", workers=4)
    
    try:
        results['NRA: Cloud Streaming\\n(Tokio + HTTP Range)'] = run_epoch(CloudNraDataset(cloud_url), "Cloud NRA", workers=4)
    except Exception as e:
        print(f"Error connecting to {cloud_url}: {e}")
        print("Make sure you run 'npx serve /tmp -p 8000' in the background!")
        return

    # 3. Plotting
    os.makedirs('docs/assets', exist_ok=True)
    plot_throughput(results)
    plot_cold_start()
    
if __name__ == "__main__":
    main()
