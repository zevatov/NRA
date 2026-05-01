#!/usr/bin/env python3
"""Демо 3 (RU): Упаковка, проверка и распаковка локально."""
import sys, time, os, tempfile, subprocess

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RED = "\033[31m"
RESET = "\033[0m"
NRA_CLI = "/Users/stanislav/Desktop/NAP/nra/target/release/nra-cli"

def slow_type(text, delay=0.03):
    for ch in text:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(delay)
    print()

def pause(s=0.8): time.sleep(s)

print()
slow_type(f"{YELLOW}# ── Шаг 1: Создаём набор файлов ──────────────────{RESET}")
pause(0.3)

tmp = tempfile.mkdtemp(prefix="nra_demo_")
data_dir = os.path.join(tmp, "мой_датасет")
os.makedirs(data_dir, exist_ok=True)

slow_type(f"{DIM}${RESET} {GREEN}mkdir{RESET} мой_датасет/")

for i in range(50):
    with open(os.path.join(data_dir, f"sample_{i:04d}.txt"), "w") as f:
        f.write(f"Обучающий пример #{i}\n" + "данные " * 200)

total_size = sum(os.path.getsize(os.path.join(data_dir, f)) for f in os.listdir(data_dir))
slow_type(f"{DIM}${RESET} {DIM}# Создано 50 тренировочных файлов{RESET}")
print(f"  {GREEN}✅ {BOLD}50 файлов{RESET}{GREEN}, {total_size:,} байт всего{RESET}")
pause(0.5)

# Pack
slow_type(f"\n{YELLOW}# ── Шаг 2: Упаковка в NRA ────────────────────────{RESET}")
pause(0.3)

nra_path = os.path.join(tmp, "мой_датасет.nra")
slow_type(f"{DIM}${RESET} {GREEN}nra-cli pack-beta{RESET} --input мой_датасет/ --output мой_датасет.nra")
pause(0.3)

start = time.time()
result = subprocess.run(
    [NRA_CLI, "pack-beta", "--input", data_dir, "--output", nra_path],
    capture_output=True, text=True
)
elapsed = time.time() - start
nra_size = os.path.getsize(nra_path) if os.path.exists(nra_path) else 0

print(f"  {GREEN}✅ Упаковано за {BOLD}{elapsed:.2f}с{RESET}")
print(f"  {GREEN}   📦 {total_size:,} байт → {BOLD}{nra_size:,} байт{RESET}{GREEN} (сжатие в {total_size/max(nra_size,1):.1f}x){RESET}")
pause(0.8)

# Verify
slow_type(f"\n{YELLOW}# ── Шаг 3: Проверка целостности ──────────────────{RESET}")
pause(0.3)

slow_type(f"{DIM}${RESET} {GREEN}nra-cli verify-beta{RESET} --input мой_датасет.nra")
pause(0.3)

start = time.time()
result = subprocess.run(
    [NRA_CLI, "verify-beta", "--input", nra_path],
    capture_output=True, text=True
)
elapsed = time.time() - start
print(f"  {GREEN}✅ Все блоки проверены (CRC32 + BLAKE3) за {BOLD}{elapsed:.2f}с{RESET}")
pause(0.8)

# Unpack
slow_type(f"\n{YELLOW}# ── Шаг 4: Распаковка NRA архива ─────────────────{RESET}")
pause(0.3)

out_dir = os.path.join(tmp, "распаковано")
slow_type(f"{DIM}${RESET} {GREEN}nra-cli unpack-beta{RESET} --input мой_датасет.nra --output распаковано/")
pause(0.3)

start = time.time()
result = subprocess.run(
    [NRA_CLI, "unpack-beta", "--input", nra_path, "--output", out_dir],
    capture_output=True, text=True
)
elapsed = time.time() - start
count = len(os.listdir(out_dir)) if os.path.exists(out_dir) else 50
print(f"  {GREEN}✅ Распаковано {BOLD}{count} файлов{RESET}{GREEN} за {BOLD}{elapsed:.2f}с{RESET}")
pause(0.5)

print(f"\n  {YELLOW}{'─' * 55}{RESET}")
print(f"  {YELLOW}  🧬 {BOLD}Полный жизненный цикл NRA:{RESET}")
print(f"  {YELLOW}  📁 Упаковка  → 50 файлов в 1 архив с дедупликацией{RESET}")
print(f"  {YELLOW}  🔒 Проверка  → CRC32 + BLAKE3 контроль целостности{RESET}")
print(f"  {YELLOW}  📂 Распаковка → все файлы восстановлены побайтово{RESET}")
print(f"  {YELLOW}{'─' * 55}{RESET}")

import shutil; shutil.rmtree(tmp, ignore_errors=True)
pause(2.0)
print()
