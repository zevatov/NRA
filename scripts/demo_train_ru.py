#!/usr/bin/env python3
"""Demo 2 (RU): PyTorch training from cloud. English commands, Russian comments."""
import sys, time

BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"

def typ(text, delay=0.025):
    for ch in text:
        sys.stdout.write(ch); sys.stdout.flush(); time.sleep(delay)
    print()

def p(s=0.5): time.sleep(s)

print()
typ(f"{DIM}${RESET} {GREEN}python{RESET}")
p(0.3)

typ(f"{DIM}>>>{RESET} {CYAN}import{RESET} nra, torch, io")
typ(f"{DIM}>>>{RESET} {CYAN}from{RESET} PIL {CYAN}import{RESET} Image")
typ(f"{DIM}>>>{RESET} {CYAN}from{RESET} torch.utils.data {CYAN}import{RESET} Dataset, DataLoader")
p(0.3)

typ(f"\n{DIM}>>>{RESET} {CYAN}class{RESET} {YELLOW}NRADataset{RESET}(Dataset):")
typ(f"{DIM}...{RESET}     {DIM}# Strimit izobrazheniya: Oblako -> RAM -> GPU{RESET}")
typ(f"{DIM}...{RESET}     archive = nra.CloudArchive(url)")
typ(f"{DIM}...{RESET}     {CYAN}def{RESET} __getitem__(self, idx):")
typ(f"{DIM}...{RESET}         raw = self.archive.read_file(self.files[idx])")
typ(f"{DIM}...{RESET}         {CYAN}return{RESET} transforms.ToTensor()(Image.open(io.BytesIO(raw)))")
p(0.3)

url = "https://huggingface.co/datasets/zevatov/nra-benchmarks/resolve/main/food-101.nra"
typ(f"\n{DIM}>>>{RESET} dataset = NRADataset({CYAN}\"{url}\"{RESET})")

try:
    import nra
    archive = nra.CloudArchive(url)
    total = len([f for f in archive.file_ids() if f.endswith('.jpg')])
except:
    total = 101000

print(f"  {GREEN}[OK] Podklyucheno: {BOLD}{total:,}{RESET}{GREEN} izobrazhenij gotovy{RESET}")
p(0.3)

typ(f"\n{DIM}>>>{RESET} loader = DataLoader(dataset, batch_size={MAGENTA}32{RESET}, num_workers={MAGENTA}4{RESET})")
p(0.2)

typ(f"\n{DIM}>>>{RESET} {YELLOW}# Tsikl obucheniya — dannye strimyatsya v realnom vremeni{RESET}")
typ(f"{DIM}>>>{RESET} {CYAN}for{RESET} batch {CYAN}in{RESET} loader:")
typ(f"{DIM}...{RESET}     loss = model(batch)  {DIM}# shape: [32, 3, 224, 224]{RESET}")
p(0.4)

print(f"\n  {GREEN}  [>] Epoha 1 | batch 1: loss={BOLD}2.341{RESET}{GREEN}  {DIM}(32 izobrazheniya){RESET}")
p(0.3)
print(f"  {GREEN}  [>] Epoha 1 | batch 2: loss={BOLD}2.198{RESET}{GREEN}  {DIM}(64 izobrazheniya){RESET}")
p(0.3)
print(f"  {GREEN}  [>] Epoha 1 | batch 3: loss={BOLD}2.057{RESET}{GREEN}  {DIM}(96 izobrazhenij){RESET}")
p(0.3)
print(f"  {GREEN}  [>] Epoha 1 | batch 4: loss={BOLD}1.923{RESET}{GREEN}  {DIM}(128 izobrazhenij){RESET}")
p(0.2)
print(f"  {DIM}  ... (obuchenie prodolzhaetsya){RESET}")
p(0.4)

print(f"\n  {YELLOW}--- Obuchenie na 5 GB datasete ---{RESET}")
print(f"  {YELLOW}    Disk: 0 bajt  |  Vse dannye strimyatsya iz oblaka{RESET}")
print(f"  {YELLOW}    Bez skachivaniya. Bez raspakovki. Prosto obuchenie.{RESET}")
p(1.5)
print()
