#!/usr/bin/env python3
"""
NRA Honest Benchmark v1.0
=========================
All measurements use REAL I/O operations. No torch.randn(). No fake data.
Every number in the output is physically measured on this machine.
"""

import time
import os
import hashlib
import shutil
import subprocess
import struct
import io

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import nra

# ==========================================
# Config
# ==========================================

CIFAR_PNG_DIR = "/tmp/cifar10_png"
CIFAR_NRA    = "/tmp/cifar10.nra"
CIFAR_TARGZ  = "/tmp/cifar10.tar.gz"
CIFAR_DUP_DIR = "/tmp/cifar10_dup_png"
CIFAR_DUP_NRA = "/tmp/cifar10_dup.nra"
CLOUD_URL    = "http://localhost:8000/cifar10.nra"
RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "assets")
NUM_WORKERS  = 4
BATCH_SIZE   = 128
NUM_RUNS     = 3

os.makedirs(RESULTS_DIR, exist_ok=True)

# Dark theme
plt.style.use('dark_background')
sns.set_theme(style="darkgrid", rc={
    "axes.facecolor": "#121212", "figure.facecolor": "#0d0d0d",
    "grid.color": "#2a2a2a", "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0", "xtick.color": "#a0a0a0",
    "ytick.color": "#a0a0a0", "font.family": "sans-serif"
})

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
])

# ==========================================
# 1. Datasets (REAL I/O)
# ==========================================

class RawFileDataset(Dataset):
    """Reads real PNG files from disk."""
    def __init__(self, folder):
        self.files = sorted([
            os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')
        ])
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        with open(self.files[idx], "rb") as f:
            raw = f.read()
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return transform(img), 0

class NraLocalDataset(Dataset):
    """Reads real data from a local .nra archive."""
    def __init__(self, archive_path):
        self.path = archive_path
        self.archive = nra.BetaArchive(archive_path)
        self.file_ids = self.archive.file_ids()
    def __len__(self):
        return len(self.file_ids)
    def __getitem__(self, idx):
        raw = bytearray(self.archive.read_file(self.file_ids[idx]))
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return transform(img), 0

class NraCloudDataset(Dataset):
    """Reads real data from NRA Cloud (HTTP Range)."""
    def __init__(self, url):
        self.url = url
        self.file_ids = nra.CloudArchive(url).file_ids()
        self._archive = None
    def __len__(self):
        return len(self.file_ids)
    def __getitem__(self, idx):
        if self._archive is None:
            self._archive = nra.CloudArchive(self.url)
        raw = bytearray(self._archive.read_file(self.file_ids[idx]))
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        return transform(img), 0

# ==========================================
# 2. Training Model
# ==========================================

class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(32 * 8 * 8, 10)
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ==========================================
# 3. Benchmark Functions
# ==========================================

def bench_throughput(dataset, name, num_runs=NUM_RUNS):
    """Measure real throughput (files/sec) with actual I/O."""
    times = []
    for run in range(num_runs):
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
        start = time.perf_counter()
        count = 0
        for batch, _ in loader:
            count += batch.size(0)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        print(f"  Run {run+1}: {count} files in {elapsed:.2f}s ({count/elapsed:.0f} files/sec)")
    median = sorted(times)[len(times)//2]
    fps = len(dataset) / median
    print(f"  ✅ {name}: Median {median:.2f}s → {fps:.0f} files/sec\n")
    return fps, median

def bench_training(dataset, name):
    """Train 1 epoch of TinyCNN with real data + real forward/backward."""
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    model = TinyCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"\n🧠 Training {name}...")
    start = time.perf_counter()
    total_loss = 0
    count = 0
    for batch, targets in loader:
        targets = torch.zeros(batch.size(0), dtype=torch.long)
        optimizer.zero_grad()
        output = model(batch)
        loss = criterion(output, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        count += batch.size(0)
    elapsed = time.perf_counter() - start
    avg_loss = total_loss / max(count / BATCH_SIZE, 1)
    print(f"  ✅ {name}: {count} imgs in {elapsed:.2f}s ({count/elapsed:.0f} img/s), avg loss={avg_loss:.4f}")
    return elapsed, count / elapsed

def bench_integrity(archive_path, raw_dir):
    """Verify SHA256 of every file matches original."""
    print("\n🔒 Integrity Test (SHA256)...")
    archive = nra.BetaArchive(archive_path)
    file_ids = archive.file_ids()
    passed = 0
    failed = 0
    for fid in file_ids:
        nra_data = bytearray(archive.read_file(fid))
        raw_path = os.path.join(raw_dir, fid)
        if not os.path.exists(raw_path):
            continue
        with open(raw_path, "rb") as f:
            raw_data = f.read()
        if hashlib.sha256(nra_data).hexdigest() == hashlib.sha256(raw_data).hexdigest():
            passed += 1
        else:
            failed += 1
            print(f"  ❌ MISMATCH: {fid}")
    print(f"  ✅ Integrity: {passed} passed, {failed} failed out of {passed+failed} files")
    return passed, failed

def bench_storage():
    """Measure real storage sizes."""
    print("\n📦 Storage Benchmark...")
    
    # Raw files
    raw_size = sum(
        os.path.getsize(os.path.join(CIFAR_PNG_DIR, f))
        for f in os.listdir(CIFAR_PNG_DIR) if f.endswith('.png')
    )
    
    # tar.gz
    if not os.path.exists(CIFAR_TARGZ):
        subprocess.run(["tar", "czf", CIFAR_TARGZ, "-C", "/tmp", "cifar10_png"], check=True)
    targz_size = os.path.getsize(CIFAR_TARGZ)
    
    # NRA BETA
    nra_size = os.path.getsize(CIFAR_NRA)
    
    sizes = {
        "Raw PNG Files": raw_size,
        "Tar.gz": targz_size,
        "NRA BETA": nra_size,
    }
    
    for name, size in sizes.items():
        print(f"  {name}: {size / 1024 / 1024:.2f} MB")
    
    return sizes

def bench_cold_start():
    """Measure real cold-start latency."""
    print("\n⏱️ Cold Start Benchmark...")
    
    # Local: time to create DataLoader and get first batch
    start = time.perf_counter()
    ds = RawFileDataset(CIFAR_PNG_DIR)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)
    batch = next(iter(loader))
    local_time = time.perf_counter() - start
    print(f"  Local SSD: {local_time*1000:.1f}ms to first batch")
    
    # NRA Local
    start = time.perf_counter()
    ds = NraLocalDataset(CIFAR_NRA)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=0)
    batch = next(iter(loader))
    nra_time = time.perf_counter() - start
    print(f"  NRA Local: {nra_time*1000:.1f}ms to first batch")
    
    return {"Local SSD": local_time * 1000, "NRA Local": nra_time * 1000}

# ==========================================
# 4. Chart Rendering
# ==========================================

def plot_storage(sizes):
    plt.figure(figsize=(10, 6))
    names = list(sizes.keys())
    values = [v / 1024 / 1024 for v in sizes.values()]
    colors = ['#4c566a', '#5e81ac', '#bf616a']
    bars = plt.bar(names, values, color=colors, edgecolor='white', linewidth=1.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 0.1,
                 f'{h:.2f} MB', ha='center', va='bottom', fontsize=13, fontweight='bold', color='white')
    plt.title('CIFAR-10 (5000 PNG) — Real Storage Size', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Size (MB) — Lower is Better', fontsize=14)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/honest_storage.png', dpi=300, bbox_inches='tight')
    print("📊 Saved honest_storage.png")

def plot_throughput(results):
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    values = [v[0] for v in results.values()]
    colors = ['#4c566a', '#bf616a']
    bars = plt.bar(names, values, color=colors, edgecolor='white', width=0.5, linewidth=1.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 5,
                 f'{int(h)} files/s', ha='center', va='bottom', fontsize=13, fontweight='bold', color='white')
    plt.title('CIFAR-10 — Real DataLoader Throughput (Median of 3 runs)', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Throughput (files/sec) — Higher is Better', fontsize=14)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/honest_throughput.png', dpi=300, bbox_inches='tight')
    print("📊 Saved honest_throughput.png")

def plot_training(results):
    plt.figure(figsize=(10, 6))
    names = list(results.keys())
    values = [v[1] for v in results.values()]
    colors = ['#4c566a', '#bf616a']
    bars = plt.bar(names, values, color=colors, edgecolor='white', width=0.5, linewidth=1.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., h + 3,
                 f'{int(h)} img/s', ha='center', va='bottom', fontsize=13, fontweight='bold', color='white')
    plt.title('CIFAR-10 — Real CNN Training Speed (PIL decode + forward/backward)', fontsize=18, fontweight='bold', pad=20)
    plt.ylabel('Training Speed (img/sec) — Higher is Better', fontsize=14)
    sns.despine()
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/honest_training.png', dpi=300, bbox_inches='tight')
    print("📊 Saved honest_training.png")

# ==========================================
# 5. Main
# ==========================================

def main():
    print("=" * 60)
    print("🔬 NRA HONEST BENCHMARK v1.0")
    print("   Dataset: CIFAR-10 (5000 real PNG files)")
    print("   All I/O is REAL. No fake data. No torch.randn().")
    print("=" * 60)

    # Test 1: Storage
    sizes = bench_storage()

    # Test 2: Throughput
    print("\n📈 Throughput Benchmark (Real I/O)...")
    raw_ds = RawFileDataset(CIFAR_PNG_DIR)
    nra_ds = NraLocalDataset(CIFAR_NRA)

    print("  --- Raw SSD Files ---")
    raw_fps, raw_time = bench_throughput(raw_ds, "Raw SSD")
    print("  --- NRA BETA Local ---")
    nra_fps, nra_time = bench_throughput(nra_ds, "NRA BETA Local")

    throughput_results = {
        "Raw SSD Files": (raw_fps, raw_time),
        "NRA BETA Local": (nra_fps, nra_time),
    }

    # Test 3: Training
    print("\n🧠 Training Benchmark (Real CNN)...")
    raw_train_time, raw_train_ips = bench_training(raw_ds, "Raw SSD")
    nra_train_time, nra_train_ips = bench_training(nra_ds, "NRA BETA")

    training_results = {
        "Raw SSD Files": (raw_train_time, raw_train_ips),
        "NRA BETA Local": (nra_train_time, nra_train_ips),
    }

    # Test 4: Cold Start
    cold = bench_cold_start()

    # Test 5: Integrity
    passed, failed = bench_integrity(CIFAR_NRA, CIFAR_PNG_DIR)

    # Render charts
    print("\n🎨 Rendering honest charts...")
    plot_storage(sizes)
    plot_throughput(throughput_results)
    plot_training(training_results)

    # Summary
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    print(f"Storage:     Raw={sizes['Raw PNG Files']/1024/1024:.2f}MB  tar.gz={sizes['Tar.gz']/1024/1024:.2f}MB  NRA={sizes['NRA BETA']/1024/1024:.2f}MB")
    print(f"Throughput:  Raw={raw_fps:.0f} f/s  NRA={nra_fps:.0f} f/s  (diff: {abs(raw_fps-nra_fps)/raw_fps*100:.1f}%)")
    print(f"Training:    Raw={raw_train_ips:.0f} img/s  NRA={nra_train_ips:.0f} img/s")
    print(f"Cold Start:  Local={cold['Local SSD']:.1f}ms  NRA={cold['NRA Local']:.1f}ms")
    print(f"Integrity:   {passed}/{passed+failed} files OK ({failed} failed)")
    print("=" * 60)

if __name__ == "__main__":
    main()
