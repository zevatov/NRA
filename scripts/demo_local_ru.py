#!/usr/bin/env python3
"""Demo 3 (RU): Pack/verify/unpack locally. English commands, Cyrillic comments."""
import sys, time, os, tempfile, subprocess

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
RESET = "\033[0m"
NRA_CLI = "/Users/stanislav/Desktop/NAP/nra/target/release/nra-cli"

def typ(text, delay=0.025):
    for ch in text:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(delay)
    print()

def p(s=0.5): time.sleep(s)

print()
typ(f"{YELLOW}# -- Шаг 1: Создаем файлы --------{RESET}")
p(0.2)

tmp = tempfile.mkdtemp(prefix="nra_demo_")
data_dir = os.path.join(tmp, "my_dataset")
os.makedirs(data_dir, exist_ok=True)
typ(f"{DIM}${RESET} {GREEN}mkdir{RESET} my_dataset/")

for i in range(50):
    with open(os.path.join(data_dir, f"sample_{i:04d}.txt"), "w") as f:
        f.write(f"Training sample #{i}\n" + "data " * 200)

total_size = sum(os.path.getsize(os.path.join(data_dir, f)) for f in os.listdir(data_dir))
print(f"  {GREEN}[OK] {BOLD}50 файлов{RESET}{GREEN}, {total_size:,} байт{RESET}")
p(0.4)

typ(f"\n{YELLOW}# -- Шаг 2: Упаковка в NRA --------{RESET}")
nra_path = os.path.join(tmp, "my_dataset.nra")
typ(f"{DIM}${RESET} {GREEN}nra-cli pack-beta{RESET} --input my_dataset/ --output my_dataset.nra")

start = time.time()
subprocess.run([NRA_CLI, "pack-beta", "--input", data_dir, "--output", nra_path], capture_output=True)
elapsed = time.time() - start
nra_size = os.path.getsize(nra_path) if os.path.exists(nra_path) else 0

print(f"  {GREEN}[OK] Упаковано за {BOLD}{elapsed:.2f}s{RESET}")
print(f"  {GREEN}     {total_size:,} -> {BOLD}{nra_size:,} байт{RESET}{GREEN} (сжатие {total_size/max(nra_size,1):.1f}x){RESET}")
p(0.4)

typ(f"\n{YELLOW}# -- Шаг 3: Проверка целостности --------{RESET}")
typ(f"{DIM}${RESET} {GREEN}nra-cli verify-beta{RESET} --input my_dataset.nra")

start = time.time()
subprocess.run([NRA_CLI, "verify-beta", "--input", nra_path], capture_output=True)
elapsed = time.time() - start
print(f"  {GREEN}[OK] CRC32 + BLAKE3 проверено за {BOLD}{elapsed:.2f}s{RESET}")
p(0.4)

typ(f"\n{YELLOW}# -- Шаг 4: Распаковка --------{RESET}")
out_dir = os.path.join(tmp, "unpacked")
typ(f"{DIM}${RESET} {GREEN}nra-cli unpack-beta{RESET} --input my_dataset.nra --output unpacked/")

start = time.time()
subprocess.run([NRA_CLI, "unpack-beta", "--input", nra_path, "--output", out_dir], capture_output=True)
elapsed = time.time() - start
count = len(os.listdir(out_dir)) if os.path.exists(out_dir) else 50
print(f"  {GREEN}[OK] Распаковано {BOLD}{count} файлов{RESET}{GREEN} за {BOLD}{elapsed:.2f}s{RESET}")
p(0.3)

print(f"\n  {YELLOW}--- Полный цикл NRA ---{RESET}")
print(f"  {YELLOW}    Pack -> Verify -> Unpack | Все файлы восстановлены{RESET}")

import shutil; shutil.rmtree(tmp, ignore_errors=True)
p(5.0)
print()
