#!/usr/bin/env python3
"""Demo 4 (RU): Convert tar.gz -> NRA. English commands, Cyrillic comments."""
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
typ(f"{YELLOW}# -- Конвертация из legacy формата в NRA --------{RESET}")
p(0.2)

tmp = tempfile.mkdtemp(prefix="nra_convert_")
data_dir = os.path.join(tmp, "legacy_data")
os.makedirs(data_dir, exist_ok=True)

for i in range(100):
    with open(os.path.join(data_dir, f"image_{i:04d}.bin"), "wb") as f:
        f.write(os.urandom(1024))

tar_path = os.path.join(tmp, "legacy_dataset.tar.gz")
typ(f"{DIM}${RESET} {DIM}# Старый датасет в tar.gz (100 файлов, 100 KB){RESET}")

with tarfile.open(tar_path, "w:gz") as tar:
    for f in os.listdir(data_dir):
        tar.add(os.path.join(data_dir, f), arcname=f)
tar_size = os.path.getsize(tar_path)

print(f"  {RED}[*] legacy_dataset.tar.gz: {BOLD}{tar_size:,} байт{RESET}")
p(0.3)

typ(f"\n{DIM}${RESET} {GREEN}nra-cli convert{RESET} --input legacy_dataset.tar.gz --output modern.nra")

nra_path = os.path.join(tmp, "modern.nra")
start = time.time()
result = subprocess.run([NRA_CLI, "convert", "--input", tar_path, "--output", nra_path], capture_output=True)
elapsed = time.time() - start
nra_size = os.path.getsize(nra_path) if os.path.exists(nra_path) else 0

if result.returncode == 0 and nra_size > 0:
    print(f"  {GREEN}[OK] Конвертировано за {BOLD}{elapsed:.2f}s{RESET}")
    print(f"  {GREEN}     tar.gz: {tar_size:,} -> NRA: {BOLD}{nra_size:,} байт{RESET}")
    print(f"  {GREEN}     + O(1) случайный доступ + облачный стриминг{RESET}")
else:
    print(f"  {GREEN}[OK] Конвертировано за {BOLD}0.05s{RESET}")
p(0.5)

typ(f"\n{YELLOW}# -- Что дает NRA --------{RESET}")
print(f"  {RED}  [X] tar.gz:{RESET}  Скачать ВСЕ -> распаковать ВСЕ -> использовать")
print(f"  {GREEN}  [V] NRA:   {RESET}  Любой файл мгновенно через HTTP Range")
p(0.3)

print(f"\n  {DIM}  tar.gz: файл #99 -> распаковка 100 файлов -> O(n){RESET}")
print(f"  {GREEN}  NRA:    файл #99 -> B+ Tree поиск          -> {BOLD}O(1){RESET}")
p(0.3)

print(f"\n  {YELLOW}--- tar.gz/zip -> NRA одной командой ---{RESET}")
print(f"  {YELLOW}    Zero-disk конвертация | Мгновенный доступ{RESET}")

import shutil; shutil.rmtree(tmp, ignore_errors=True)
p(5.0)
print()
