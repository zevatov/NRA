import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import nra
from PIL import Image
import io
import time
import sys

# Ссылка на локальный файл или удаленный HF_URL
# Для демо на HF мы используем прямую ссылку на твой загруженный файл
ARCHIVE_PATH = "https://huggingface.co/datasets/zevatov/nra-cifar10/resolve/main/cifar10.nra"

class NraCloudDataset(Dataset):
    def __init__(self, url):
        self.url = url
        print("Fetching manifest (Zero-Download initialization)...")
        start = time.time()
        
        # Определяем, локальный файл или CloudArchive
        if url.startswith("http"):
            archive = nra.CloudArchive(url)
        else:
            archive = nra.BetaArchive(url)
            
        self.file_ids = archive.file_ids()
        # Только картинки (избегаем чтения других файлов и Mac OS dotfiles)
        self.file_ids = [f for f in self.file_ids if f.endswith(".png") and not f.split("/")[-1].startswith("._")]
        print(f"Found {len(self.file_ids)} images in {time.time() - start:.3f}s")
        
        self._archive = None
        
        # Стандартные трансформации CIFAR-10
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        # Ленивая инициализация для multiprocessing
        if self._archive is None:
            if self.url.startswith("http"):
                self._archive = nra.CloudArchive(self.url)
            else:
                self._archive = nra.BetaArchive(self.url)
            
        file_id = self.file_ids[idx]
        
        # Это мгновенно вытащит конкретный файл (асинхронно или mmap)
        raw_bytes = self._archive.read_file(file_id)
        
        # Декодирование PNG
        image = Image.open(io.BytesIO(bytearray(raw_bytes))).convert('RGB')
        tensor = self.transform(image)
        
        # Парсим лейбл из имени файла (например "train/3_00001.png")
        basename = file_id.split("/")[-1]
        label = int(basename.split("_")[0])
        
        return tensor, label

def main():
    if len(sys.argv) > 1:
        archive_url = sys.argv[1]
    else:
        archive_url = ARCHIVE_PATH

    print(f"=== NRA ZERO-DOWNLOAD DEMO ===")
    print(f"Target: {archive_url}")
    
    start_time = time.time()
    
    dataset = NraCloudDataset(archive_url)
    
    # 8 воркеров для демонстрации того, как GIL освобождается
    loader = DataLoader(dataset, batch_size=256, num_workers=4, shuffle=True)
    
    print(f"Time to First Batch (TTFB) Setup Complete: {time.time() - start_time:.3f}s")
    
    # Простая CNN для CIFAR-10
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64 * 8 * 8, 512),
        nn.ReLU(),
        nn.Linear(512, 10)
    )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    print(f"Starting epoch 1 on {device}...")
    
    # Замеряем throughput
    total_images = 0
    epoch_start = time.time()
    
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        
        # Обучение
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_images += data.size(0)
        
        if batch_idx % 10 == 0:
            elapsed = time.time() - epoch_start
            throughput = total_images / elapsed if elapsed > 0 else 0
            print(f"Batch {batch_idx:03d} | Loss: {loss.item():.4f} | Throughput: {throughput:.1f} img/s")

if __name__ == "__main__":
    main()
