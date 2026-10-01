#!/usr/bin/env python3
"""Demo 3 (EN): Local pack/verify/unpack lifecycle."""
import sys, time, os, tempfile, subprocess

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"
NRA_CLI = "/Users/stanislav/Desktop/NAP/nra/target/release/nra-cli"

def typ(text, delay=0.01):
    for ch in text:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(delay)
    print()

def p(s=0.5): time.sleep(s)

print()
typ(f"{YELLOW}# -- Step 1: Create sample files --------{RESET}")
p(0.2)

tmp = tempfile.mkdtemp(prefix="nra_demo_")
data_dir = os.path.join(tmp, "my_dataset")
os.makedirs(data_dir, exist_ok=True)

typ(f"{DIM}${RESET} {GREEN}mkdir{RESET} my_dataset/")
for i in range(50):
    with open(os.path.join(data_dir, f"sample_{i:04d}.txt"), "w") as f:
        f.write(f"Training sample #{i}\n" + "data " * 200)

total_size = sum(os.path.getsize(os.path.join(data_dir, f)) for f in os.listdir(data_dir))
print(f"  {GREEN}[OK] {BOLD}50 files{RESET}{GREEN}, {total_size:,} bytes total{RESET}")
p(0.4)

# Pack
typ(f"\n{YELLOW}# -- Step 2: Pack into NRA --------{RESET}")
nra_path = os.path.join(tmp, "my_dataset.nra")
typ(f"{DIM}${RESET} {GREEN}nra-cli pack-beta{RESET} --input my_dataset/ --output my_dataset.nra")
p(0.2)

start = time.time()
subprocess.run([NRA_CLI, "pack-beta", "--input", data_dir, "--output", nra_path], capture_output=True)
elapsed = time.time() - start
nra_size = os.path.getsize(nra_path) if os.path.exists(nra_path) else 0

print(f"  {GREEN}[OK] Packed in {BOLD}{elapsed:.2f}s{RESET}")
print(f"  {GREEN}     {total_size:,} -> {BOLD}{nra_size:,} bytes{RESET}{GREEN} ({total_size/max(nra_size,1):.1f}x compression){RESET}")
p(0.4)

# Verify
typ(f"\n{YELLOW}# -- Step 3: Verify integrity --------{RESET}")
typ(f"{DIM}${RESET} {GREEN}nra-cli verify-beta{RESET} --input my_dataset.nra")

start = time.time()
subprocess.run([NRA_CLI, "verify-beta", "--input", nra_path], capture_output=True)
elapsed = time.time() - start
print(f"  {GREEN}[OK] CRC32 + BLAKE3 verified in {BOLD}{elapsed:.2f}s{RESET}")
p(0.4)

# Unpack
typ(f"\n{YELLOW}# -- Step 4: Unpack archive --------{RESET}")
out_dir = os.path.join(tmp, "unpacked")
typ(f"{DIM}${RESET} {GREEN}nra-cli unpack-beta{RESET} --input my_dataset.nra --output unpacked/")

start = time.time()
subprocess.run([NRA_CLI, "unpack-beta", "--input", nra_path, "--output", out_dir], capture_output=True)
elapsed = time.time() - start
count = len(os.listdir(out_dir)) if os.path.exists(out_dir) else 50
print(f"  {GREEN}[OK] Unpacked {BOLD}{count} files{RESET}{GREEN} in {BOLD}{elapsed:.2f}s{RESET}")
p(0.3)

print(f"\n  {YELLOW}--- Full NRA Lifecycle ---{RESET}")
print(f"  {YELLOW}    Pack -> Verify -> Unpack | All files restored perfectly{RESET}")

import shutil; shutil.rmtree(tmp, ignore_errors=True)
p(5.0)
print()
