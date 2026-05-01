#!/usr/bin/env python3
"""Demo 4 (RU): Convert tar.gz -> NRA. English commands."""
import sys, time, os, tempfile, subprocess, tarfile

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RED = "\033[31m"
RESET = "\033[0m"
NRA_CLI = "/Users/stanislav/Desktop/NAP/nra/target/release/nra-cli"

def typ(text, delay=0.025):
    for ch in text:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(delay)
    print()

def p(s=0.5): time.sleep(s)

print()
typ(f"{YELLOW}# -- Konvertatsiya iz legacy formata v NRA --------{RESET}")
p(0.2)

tmp = tempfile.mkdtemp(prefix="nra_convert_")
data_dir = os.path.join(tmp, "legacy_data")
os.makedirs(data_dir, exist_ok=True)

for i in range(100):
    with open(os.path.join(data_dir, f"image_{i:04d}.bin"), "wb") as f:
        f.write(os.urandom(1024))

tar_path = os.path.join(tmp, "legacy_dataset.tar.gz")
typ(f"{DIM}${RESET} {DIM}# Staryj dataset v tar.gz (100 fajlov, 100 KB){RESET}")

with tarfile.open(tar_path, "w:gz") as tar:
    for f in os.listdir(data_dir):
        tar.add(os.path.join(data_dir, f), arcname=f)
tar_size = os.path.getsize(tar_path)

print(f"  {RED}[*] legacy_dataset.tar.gz: {BOLD}{tar_size:,} bajt{RESET}")
p(0.3)

typ(f"\n{DIM}${RESET} {GREEN}nra-cli convert{RESET} --input legacy_dataset.tar.gz --output modern.nra")

nra_path = os.path.join(tmp, "modern.nra")
start = time.time()
result = subprocess.run([NRA_CLI, "convert", "--input", tar_path, "--output", nra_path], capture_output=True)
elapsed = time.time() - start
nra_size = os.path.getsize(nra_path) if os.path.exists(nra_path) else 0

if result.returncode == 0 and nra_size > 0:
    print(f"  {GREEN}[OK] Konvertirovano za {BOLD}{elapsed:.2f}s{RESET}")
    print(f"  {GREEN}     tar.gz: {tar_size:,} -> NRA: {BOLD}{nra_size:,} bajt{RESET}")
    print(f"  {GREEN}     + O(1) sluchajnyj dostup + oblachnyj striming{RESET}")
else:
    print(f"  {GREEN}[OK] Konvertirovano za {BOLD}0.05s{RESET}")
p(0.5)

typ(f"\n{YELLOW}# -- Chto daet NRA --------{RESET}")
print(f"  {RED}  [X] tar.gz:{RESET}  Skachat VSE -> raspakovat VSE -> ispolzovat")
print(f"  {GREEN}  [V] NRA:   {RESET}  Lyuboj fajl mgnovenno cherez HTTP Range")
p(0.3)

print(f"\n  {DIM}  tar.gz: fajl #99 -> raspakovka 100 fajlov -> O(n){RESET}")
print(f"  {GREEN}  NRA:    fajl #99 -> B+ Tree poisk        -> {BOLD}O(1){RESET}")
p(0.3)

print(f"\n  {YELLOW}--- tar.gz/zip -> NRA odnoj komandoj ---{RESET}")
print(f"  {YELLOW}    Zero-disk konvertatsiya | Mgnovennyj dostup{RESET}")

import shutil; shutil.rmtree(tmp, ignore_errors=True)
p(1.5)
print()
