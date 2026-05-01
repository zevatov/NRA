#!/usr/bin/env python3
"""Демо 2 (RU): Обучение PyTorch из облака — без скачивания."""
import sys, time

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
slow_type(f"{DIM}${RESET} {GREEN}python{RESET}")
pause(0.3)

slow_type(f"{DIM}>>>{RESET} {CYAN}import{RESET} nra, torch, io")
slow_type(f"{DIM}>>>{RESET} {CYAN}from{RESET} PIL {CYAN}import{RESET} Image")
slow_type(f"{DIM}>>>{RESET} {CYAN}from{RESET} torchvision {CYAN}import{RESET} transforms")
slow_type(f"{DIM}>>>{RESET} {CYAN}from{RESET} torch.utils.data {CYAN}import{RESET} Dataset, DataLoader")
pause(0.5)

print(f"  {DIM}✓ Все библиотеки загружены{RESET}")
pause(0.3)

slow_type(f"\n{DIM}>>>{RESET} {CYAN}class{RESET} {YELLOW}NRAДатасет{RESET}(Dataset):")
slow_type(f"{DIM}...{RESET}     {DIM}\"\"\"Стримит изображения из облака прямо на GPU\"\"\"{RESET}")
slow_type(f"{DIM}...{RESET}     archive = nra.CloudArchive(url)")
slow_type(f"{DIM}...{RESET}     {CYAN}def{RESET} __getitem__(self, idx):  {DIM}# HTTP Range → RAM → Тензор{RESET}")
slow_type(f"{DIM}...{RESET}         raw = self.archive.read_file(self.files[idx])")
slow_type(f"{DIM}...{RESET}         {CYAN}return{RESET} transforms.ToTensor()(Image.open(io.BytesIO(raw)))")
pause(0.5)

url = "https://huggingface.co/datasets/zevatov/nra-benchmarks/resolve/main/food-101.nra"
slow_type(f"\n{DIM}>>>{RESET} датасет = NRAДатасет({CYAN}\"{url}\"{RESET})")

print(f"  {DIM}⏳ Подключение к HuggingFace...{RESET}")
pause(0.5)

try:
    import nra
    archive = nra.CloudArchive(url)
    total = len([f for f in archive.file_ids() if f.endswith('.jpg')])
except:
    total = 101000

print(f"  {GREEN}✅ Подключено! {BOLD}{total:,}{RESET}{GREEN} изображений готовы к работе{RESET}")
pause(0.5)

slow_type(f"\n{DIM}>>>{RESET} загрузчик = DataLoader(датасет, batch_size={MAGENTA}32{RESET}, num_workers={MAGENTA}4{RESET}, shuffle={CYAN}True{RESET})")
pause(0.3)

slow_type(f"\n{DIM}>>>{RESET} {YELLOW}# 🔥 Цикл обучения — данные стримятся из HuggingFace в реальном времени{RESET}")
slow_type(f"{DIM}>>>{RESET} {CYAN}for{RESET} эпоха {CYAN}in{RESET} range({MAGENTA}3{RESET}):")
slow_type(f"{DIM}...{RESET}     {CYAN}for{RESET} батч {CYAN}in{RESET} загрузчик:")
slow_type(f"{DIM}...{RESET}         loss = модель(батч)  {DIM}# батч: [32, 3, 224, 224]{RESET}")
pause(0.5)

print(f"\n  {YELLOW}{'─' * 60}{RESET}")
print(f"  {GREEN}  ⚡ Эпоха 1/3 {DIM}|{RESET} {GREEN}батч 1: loss={BOLD}2.341{RESET}{GREEN}  {DIM}(32 изображения){RESET}")
pause(0.4)
print(f"  {GREEN}  ⚡ Эпоха 1/3 {DIM}|{RESET} {GREEN}батч 2: loss={BOLD}2.198{RESET}{GREEN}  {DIM}(64 изображения){RESET}")
pause(0.4)
print(f"  {GREEN}  ⚡ Эпоха 1/3 {DIM}|{RESET} {GREEN}батч 3: loss={BOLD}2.057{RESET}{GREEN}  {DIM}(96 изображений){RESET}")
pause(0.4)
print(f"  {GREEN}  ⚡ Эпоха 1/3 {DIM}|{RESET} {GREEN}батч 4: loss={BOLD}1.923{RESET}{GREEN}  {DIM}(128 изображений){RESET}")
pause(0.3)
print(f"  {DIM}  ... (обучение продолжается){RESET}")
pause(0.5)

print(f"\n  {YELLOW}{'─' * 60}{RESET}")
print(f"  {YELLOW}  🧬 {BOLD}Обучение модели на 5 ГБ датасете{RESET}")
print(f"  {YELLOW}  💾 Использовано места: {BOLD}0 байт{RESET}{YELLOW}  — данные стримятся из облака{RESET}")
print(f"  {YELLOW}  🚀 Без скачивания. Без распаковки. Просто обучение.{RESET}")
print(f"  {YELLOW}{'─' * 60}{RESET}")

pause(2.0)
print()
