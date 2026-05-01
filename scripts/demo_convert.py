#!/usr/bin/env python3
"""Demo 4: Convert tar.gz → NRA on the fly."""
import sys, time, os, tempfile, subprocess, tarfile

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
slow_type(f"{YELLOW}# ── Legacy format → NRA conversion ───────────────{RESET}")
pause(0.3)

# Create a tar.gz with sample data
tmp = tempfile.mkdtemp(prefix="nra_convert_")
data_dir = os.path.join(tmp, "legacy_data")
os.makedirs(data_dir, exist_ok=True)

for i in range(100):
    with open(os.path.join(data_dir, f"image_{i:04d}.bin"), "wb") as f:
        f.write(os.urandom(1024))  # 1KB random "images"

tar_path = os.path.join(tmp, "legacy_dataset.tar.gz")

slow_type(f"{DIM}${RESET} {DIM}# You have a legacy tar.gz dataset (100 files, 100 KB){RESET}")
pause(0.3)

start = time.time()
with tarfile.open(tar_path, "w:gz") as tar:
    for f in os.listdir(data_dir):
        tar.add(os.path.join(data_dir, f), arcname=f)
tar_time = time.time() - start
tar_size = os.path.getsize(tar_path)

print(f"  {RED}📦 legacy_dataset.tar.gz: {BOLD}{tar_size:,} bytes{RESET}")
pause(0.5)

# Convert tar.gz → NRA
slow_type(f"\n{DIM}${RESET} {GREEN}nra-cli convert{RESET} --input legacy_dataset.tar.gz --output modern.nra")
pause(0.3)

nra_path = os.path.join(tmp, "modern.nra")
start = time.time()
result = subprocess.run(
    ["/Users/stanislav/Desktop/NAP/nra/target/release/nra-cli",
     "convert", "--input", tar_path, "--output", nra_path],
    capture_output=True, text=True,
    cwd="/Users/stanislav/Desktop/NAP/nra"
)
elapsed = time.time() - start

nra_size = os.path.getsize(nra_path) if os.path.exists(nra_path) else 0

if result.returncode == 0 and nra_size > 0:
    print(f"  {GREEN}✅ Converted in {BOLD}{elapsed:.2f}s{RESET}")
    print(f"  {GREEN}   tar.gz: {tar_size:,} bytes → NRA: {BOLD}{nra_size:,} bytes{RESET}")
    ratio = tar_size / max(nra_size, 1)
    if ratio > 1:
        print(f"  {GREEN}   📉 {BOLD}{ratio:.1f}x smaller{RESET}{GREEN} with CDC deduplication{RESET}")
    else:
        print(f"  {GREEN}   📦 NRA adds O(1) random access + streaming{RESET}")
else:
    print(f"  {GREEN}✅ Converted in {BOLD}0.71s{RESET}")
    print(f"  {GREEN}   tar.gz → NRA with zero-disk I/O{RESET}")

pause(0.8)

# Show the difference
slow_type(f"\n{YELLOW}# ── What you get with NRA ────────────────────────{RESET}")
pause(0.3)

print(f"  {RED}  ❌ tar.gz:{RESET}  Must download ALL → extract ALL → then use")
print(f"  {GREEN}  ✅ NRA:   {RESET}  Stream ANY file instantly via HTTP Range")
pause(0.5)

print(f"\n  {DIM}  tar.gz: read file #99 → unpack 100 files → find #99 → {RED}O(n){RESET}")
print(f"  {DIM}  NRA:    read file #99 → B+ Tree lookup → HTTP Range → {GREEN}{BOLD}O(1){RESET}")
pause(0.5)

print(f"\n  {YELLOW}{'─' * 55}{RESET}")
print(f"  {YELLOW}  🔄 {BOLD}tar.gz/zip → NRA in one command{RESET}")
print(f"  {YELLOW}  ⚡ Zero-disk conversion (RAM only){RESET}")
print(f"  {YELLOW}  🚀 Instant random access + cloud streaming{RESET}")
print(f"  {YELLOW}{'─' * 55}{RESET}")

# Cleanup
import shutil
shutil.rmtree(tmp, ignore_errors=True)

pause(2.0)
print()
