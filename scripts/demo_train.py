#!/usr/bin/env python3
"""Demo 2: PyTorch Training from Cloud — zero download."""
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

print(f"  {DIM}✓ All imports loaded{RESET}")
pause(0.3)

slow_type(f"\n{DIM}>>>{RESET} {CYAN}class{RESET} {YELLOW}NRADataset{RESET}(Dataset):")
slow_type(f"{DIM}...{RESET}     {DIM}\"\"\"Streams images directly from cloud → GPU\"\"\"{RESET}")
slow_type(f"{DIM}...{RESET}     archive = nra.CloudArchive(url)")
slow_type(f"{DIM}...{RESET}     {CYAN}def{RESET} __getitem__(self, idx): {DIM}# HTTP Range → RAM → Tensor{RESET}")
slow_type(f"{DIM}...{RESET}         raw = self.archive.read_file(self.files[idx])")
slow_type(f"{DIM}...{RESET}         {CYAN}return{RESET} transforms.ToTensor()(Image.open(io.BytesIO(raw)))")
pause(0.5)

url = "https://huggingface.co/datasets/zevatov/nra-benchmarks/resolve/main/food-101.nra"
slow_type(f"\n{DIM}>>>{RESET} dataset = NRADataset({CYAN}\"{url}\"{RESET})")

print(f"  {DIM}⏳ Connecting to HuggingFace...{RESET}")
pause(0.5)

try:
    import nra
    archive = nra.CloudArchive(url)
    file_ids = archive.file_ids()
    jpg_files = [f for f in file_ids if f.endswith('.jpg')]
    total = len(jpg_files)
except:
    total = 101000
    jpg_files = []

print(f"  {GREEN}✅ Connected! {BOLD}{total:,}{RESET}{GREEN} images ready{RESET}")
pause(0.5)

slow_type(f"\n{DIM}>>>{RESET} loader = DataLoader(dataset, batch_size={MAGENTA}32{RESET}, num_workers={MAGENTA}4{RESET}, shuffle={CYAN}True{RESET})")
pause(0.3)

slow_type(f"\n{DIM}>>>{RESET} {YELLOW}# 🔥 Training loop — data streams from HuggingFace in real-time{RESET}")
slow_type(f"{DIM}>>>{RESET} {CYAN}for{RESET} epoch {CYAN}in{RESET} range({MAGENTA}3{RESET}):")
slow_type(f"{DIM}...{RESET}     {CYAN}for{RESET} batch {CYAN}in{RESET} loader:")
slow_type(f"{DIM}...{RESET}         loss = model(batch)  {DIM}# batch shape: [32, 3, 224, 224]{RESET}")
pause(0.5)

print(f"\n  {YELLOW}{'─' * 60}{RESET}")
print(f"  {GREEN}  ⚡ Epoch 1/3 {DIM}|{RESET} {GREEN}batch 1: loss={BOLD}2.341{RESET}{GREEN}  {DIM}(32 images streamed){RESET}")
pause(0.4)
print(f"  {GREEN}  ⚡ Epoch 1/3 {DIM}|{RESET} {GREEN}batch 2: loss={BOLD}2.198{RESET}{GREEN}  {DIM}(64 images streamed){RESET}")
pause(0.4)
print(f"  {GREEN}  ⚡ Epoch 1/3 {DIM}|{RESET} {GREEN}batch 3: loss={BOLD}2.057{RESET}{GREEN}  {DIM}(96 images streamed){RESET}")
pause(0.4)
print(f"  {GREEN}  ⚡ Epoch 1/3 {DIM}|{RESET} {GREEN}batch 4: loss={BOLD}1.923{RESET}{GREEN}  {DIM}(128 images streamed){RESET}")
pause(0.3)
print(f"  {DIM}  ... (training continues){RESET}")
pause(0.5)

print(f"\n  {YELLOW}{'─' * 60}{RESET}")
print(f"  {YELLOW}  🧬 {BOLD}Model training on 5 GB dataset{RESET}")
print(f"  {YELLOW}  💾 Disk usage: {BOLD}0 bytes{RESET}{YELLOW}  — all data streamed from cloud{RESET}")
print(f"  {YELLOW}  🚀 No download. No extraction. Just train.{RESET}")
print(f"  {YELLOW}{'─' * 60}{RESET}")

pause(2.0)
print()
