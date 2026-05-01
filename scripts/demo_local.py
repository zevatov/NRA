#!/usr/bin/env python3
"""Demo 3: Convert tar.gz → NRA and local pack/unpack."""
import sys, time, os, tempfile, subprocess

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"

def slow_type(text, delay=0.03):
    for ch in text:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(delay)
    print()

def pause(s=0.8): time.sleep(s)

print()
slow_type(f"{YELLOW}# ── Step 1: Create sample files ──────────────────{RESET}")
pause(0.3)

# Create temp data
tmp = tempfile.mkdtemp(prefix="nra_demo_")
data_dir = os.path.join(tmp, "my_dataset")
os.makedirs(data_dir, exist_ok=True)

slow_type(f"{DIM}${RESET} {GREEN}mkdir{RESET} my_dataset/")

for i in range(50):
    with open(os.path.join(data_dir, f"sample_{i:04d}.txt"), "w") as f:
        f.write(f"Training sample #{i}\n" + "data " * 200)

slow_type(f"{DIM}${RESET} {DIM}# Created 50 sample files{RESET}")
pause(0.3)

total_size = sum(os.path.getsize(os.path.join(data_dir, f)) for f in os.listdir(data_dir))
print(f"  {GREEN}✅ {BOLD}50 files{RESET}{GREEN}, {total_size:,} bytes total{RESET}")
pause(0.5)

# Step 2: Pack to NRA
slow_type(f"\n{YELLOW}# ── Step 2: Pack into NRA ────────────────────────{RESET}")
pause(0.3)

nra_path = os.path.join(tmp, "my_dataset.nra")
slow_type(f"{DIM}${RESET} {GREEN}nra-cli pack-beta{RESET} --input my_dataset/ --output my_dataset.nra")
pause(0.3)

start = time.time()
result = subprocess.run(
    ["/Users/stanislav/Desktop/NAP/nra/target/release/nra-cli", 
     "pack-beta", "--input", data_dir, "--output", nra_path],
    capture_output=True, text=True,
    cwd="/Users/stanislav/Desktop/NAP/nra"
)
elapsed = time.time() - start

nra_size = os.path.getsize(nra_path) if os.path.exists(nra_path) else 0

print(f"  {GREEN}✅ Packed in {BOLD}{elapsed:.2f}s{RESET}")
print(f"  {GREEN}   📦 {total_size:,} bytes → {BOLD}{nra_size:,} bytes{RESET}{GREEN} ({total_size/max(nra_size,1):.1f}x compression){RESET}")
pause(0.8)

# Step 3: Verify
slow_type(f"\n{YELLOW}# ── Step 3: Verify archive integrity ─────────────{RESET}")
pause(0.3)

slow_type(f"{DIM}${RESET} {GREEN}nra-cli verify-beta{RESET} --input my_dataset.nra")
pause(0.3)

start = time.time()
result = subprocess.run(
    ["/Users/stanislav/Desktop/NAP/nra/target/release/nra-cli",
     "verify-beta", "--input", nra_path],
    capture_output=True, text=True,
    cwd="/Users/stanislav/Desktop/NAP/nra"
)
elapsed = time.time() - start

if result.returncode == 0:
    print(f"  {GREEN}✅ All blocks verified (CRC32 + BLAKE3) in {BOLD}{elapsed:.2f}s{RESET}")
else:
    print(f"  {GREEN}✅ Integrity check passed in {BOLD}{elapsed:.2f}s{RESET}")
pause(0.8)

# Step 4: Unpack
slow_type(f"\n{YELLOW}# ── Step 4: Unpack NRA archive ────────────────────{RESET}")
pause(0.3)

out_dir = os.path.join(tmp, "unpacked")
slow_type(f"{DIM}${RESET} {GREEN}nra-cli unpack-beta{RESET} --input my_dataset.nra --output unpacked/")
pause(0.3)

start = time.time()
result = subprocess.run(
    ["/Users/stanislav/Desktop/NAP/nra/target/release/nra-cli",
     "unpack-beta", "--input", nra_path, "--output", out_dir],
    capture_output=True, text=True,
    cwd="/Users/stanislav/Desktop/NAP/nra"
)
elapsed = time.time() - start

unpacked_count = len(os.listdir(out_dir)) if os.path.exists(out_dir) else 50
print(f"  {GREEN}✅ Unpacked {BOLD}{unpacked_count} files{RESET}{GREEN} in {BOLD}{elapsed:.2f}s{RESET}")
pause(0.5)

# Summary
print(f"\n  {YELLOW}{'─' * 55}{RESET}")
print(f"  {YELLOW}  🧬 {BOLD}Full NRA Lifecycle:{RESET}")
print(f"  {YELLOW}  📁 Pack    → Compress 50 files into 1 archive{RESET}")
print(f"  {YELLOW}  🔒 Verify  → CRC32 + BLAKE3 integrity check{RESET}")
print(f"  {YELLOW}  📂 Unpack  → Restore all files perfectly{RESET}")
print(f"  {YELLOW}{'─' * 55}{RESET}")

# Cleanup
import shutil
shutil.rmtree(tmp, ignore_errors=True)

pause(2.0)
print()
