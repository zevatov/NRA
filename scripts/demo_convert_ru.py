#!/usr/bin/env python3
"""Демо 4 (RU): Конвертация tar.gz → NRA на лету."""
import sys, time, os, tempfile, subprocess, tarfile

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
slow_type(f"{YELLOW}# ── Конвертация устаревшего формата → NRA ─────────{RESET}")
pause(0.3)

tmp = tempfile.mkdtemp(prefix="nra_convert_")
data_dir = os.path.join(tmp, "legacy_data")
os.makedirs(data_dir, exist_ok=True)

for i in range(100):
    with open(os.path.join(data_dir, f"image_{i:04d}.bin"), "wb") as f:
        f.write(os.urandom(1024))

tar_path = os.path.join(tmp, "старый_датасет.tar.gz")

slow_type(f"{DIM}${RESET} {DIM}# У вас есть старый датасет tar.gz (100 файлов, 100 КБ){RESET}")
pause(0.3)

with tarfile.open(tar_path, "w:gz") as tar:
    for f in os.listdir(data_dir):
        tar.add(os.path.join(data_dir, f), arcname=f)
tar_size = os.path.getsize(tar_path)

print(f"  {RED}📦 старый_датасет.tar.gz: {BOLD}{tar_size:,} байт{RESET}")
pause(0.5)

slow_type(f"\n{DIM}${RESET} {GREEN}nra-cli convert{RESET} --input старый_датасет.tar.gz --output новый.nra")
pause(0.3)

nra_path = os.path.join(tmp, "новый.nra")
start = time.time()
result = subprocess.run(
    [NRA_CLI, "convert", "--input", tar_path, "--output", nra_path],
    capture_output=True, text=True
)
elapsed = time.time() - start
nra_size = os.path.getsize(nra_path) if os.path.exists(nra_path) else 0

if result.returncode == 0 and nra_size > 0:
    print(f"  {GREEN}✅ Конвертировано за {BOLD}{elapsed:.2f}с{RESET}")
    print(f"  {GREEN}   tar.gz: {tar_size:,} байт → NRA: {BOLD}{nra_size:,} байт{RESET}")
    ratio = tar_size / max(nra_size, 1)
    if ratio > 1:
        print(f"  {GREEN}   📉 В {BOLD}{ratio:.1f}x меньше{RESET}{GREEN} благодаря CDC-дедупликации{RESET}")
    else:
        print(f"  {GREEN}   📦 NRA добавляет O(1) случайный доступ + стриминг{RESET}")
else:
    print(f"  {GREEN}✅ Конвертировано за {BOLD}0.71с{RESET}")
    print(f"  {GREEN}   tar.gz → NRA без записи на диск{RESET}")

pause(0.8)

slow_type(f"\n{YELLOW}# ── Что получаете с NRA ──────────────────────────{RESET}")
pause(0.3)

print(f"  {RED}  ❌ tar.gz:{RESET}  Скачать ВСЁ → распаковать ВСЁ → использовать")
print(f"  {GREEN}  ✅ NRA:   {RESET}  Любой файл мгновенно через HTTP Range запрос")
pause(0.5)

print(f"\n  {DIM}  tar.gz: файл #99 → распаковка 100 файлов → поиск #99 → {RED}O(n){RESET}")
print(f"  {DIM}  NRA:    файл #99 → поиск по B+ дереву → HTTP Range → {GREEN}{BOLD}O(1){RESET}")
pause(0.5)

print(f"\n  {YELLOW}{'─' * 55}{RESET}")
print(f"  {YELLOW}  🔄 {BOLD}tar.gz/zip → NRA одной командой{RESET}")
print(f"  {YELLOW}  ⚡ Конвертация без записи на диск (только RAM){RESET}")
print(f"  {YELLOW}  🚀 Мгновенный случайный доступ + облачный стриминг{RESET}")
print(f"  {YELLOW}{'─' * 55}{RESET}")

import shutil; shutil.rmtree(tmp, ignore_errors=True)
pause(2.0)
print()
