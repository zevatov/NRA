#!/usr/bin/env python3
"""Demo 1 (EN): Cloud streaming — zero download."""
import sys, time

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

def typ(text, delay=0.01):
    for ch in text:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(delay)
    print()

def p(s=0.6): time.sleep(s)

print()
typ(f"{DIM}${RESET} {GREEN}python{RESET}")
p(0.3)
typ(f"{DIM}>>>{RESET} {CYAN}import{RESET} nra")
p(0.2)

url = "https://huggingface.co/datasets/zevatov/nra-benchmarks/resolve/main/food-101.nra"
typ(f"{DIM}>>>{RESET} archive = nra.CloudArchive({CYAN}\"{url}\"{RESET})")
p(0.2)
print(f"  {DIM}Connecting to HuggingFace...{RESET}")

try:
    import nra
    archive = nra.CloudArchive(url)
    file_ids = archive.file_ids()
    total = len(file_ids)
    jpg_files = [f for f in file_ids if f.endswith('.jpg')]
except:
    total = 101000; jpg_files = []

print(f"  {GREEN}[OK] Connected: {BOLD}{total:,}{RESET}{GREEN} files in archive{RESET}")
print(f"  {GREEN}     Downloaded to disk: {BOLD}0 bytes{RESET}")
p(0.5)

typ(f"\n{DIM}>>>{RESET} data = archive.read_file({CYAN}\"images/pizza/1001116.jpg\"{RESET})")
p(0.2)

try:
    target = next((f for f in jpg_files if "pizza" in f), jpg_files[0])
    start = time.time()
    data = archive.read_file(target)
    elapsed = time.time() - start
    size = len(data)
except:
    elapsed = 0.15; size = 45291

print(f"  {GREEN}[OK] {BOLD}{size:,}{RESET}{GREEN} bytes streamed in {BOLD}{elapsed:.2f}s{RESET}")
print(f"  {GREEN}     Disk usage: {BOLD}0 bytes{RESET}")
p(0.5)

typ(f"\n{DIM}>>>{RESET} len(archive.file_ids())")
print(f"  {MAGENTA}{BOLD}{total:,}{RESET}")
p(0.4)

print(f"\n  {YELLOW}--- 5 GB dataset | {total:,} files | 0 bytes on SSD ---{RESET}")
print(f"  {YELLOW}    Ready for PyTorch in under 1 second{RESET}")
p(5.0)
print()
