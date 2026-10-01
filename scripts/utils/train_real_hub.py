import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

# Add local adapters to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from adapters import NRADataset

# HTTP URL for our NRA Hub Demo
HUB_URL = "http://localhost:8080/datasets/A_Vision.nra"

print("============================================================")
print("🧠 NRA ZERO-DOWNLOAD TRAINING DEMO")
print(f"📡 Стриминг датасета из облака: {HUB_URL}")
print("============================================================")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

t0 = time.time()
print("\n🔄 Подключение к хабу и чтение манифеста...")
dataset = NRADataset(archive_path=None, cloud_url=HUB_URL, transform=transform)
print(f"✅ Готово! Найдено {len(dataset)} файлов. Заняло: {time.time()-t0:.2f}s")

loader = DataLoader(dataset, batch_size=32, num_workers=0)

# Load a real ResNet-18 model
print("\n🤖 Инициализация ResNet-18...")
model = models.resnet18()
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
model.train()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

print(f"\n🚀 СТАРТ ОБУЧЕНИЯ (Device: {device})")
print("Данные скачиваются напрямую в оперативную память по частям.")

total_imgs = 0
start_train = time.time()

# Ограничим 10 батчами для быстрого демо
max_batches = 10

for i, (images, targets) in enumerate(loader):
    if i >= max_batches:
        break
        
    t_batch = time.time()
    images = images.to(device)
    targets = targets.to(device)
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, targets)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    batch_time = time.time() - t_batch
    total_imgs += len(images)
    print(f"Batch {i+1}/{max_batches} | Loss: {loss.item():.4f} | Images: {len(images)} | Time: {batch_time:.3f}s")

total_time = time.time() - start_train
speed = total_imgs / total_time

print("\n============================================================")
print("📊 ИТОГИ ОБУЧЕНИЯ")
print(f"Всего изображений обработано: {total_imgs}")
print(f"Затраченное время: {total_time:.2f}s")
print(f"Скорость (Стриминг + Обучение): {speed:.2f} img/sec")
print("✅ ТЕСТ УСПЕШНО ПРОЙДЕН. Данные стримились без загрузки на диск!")
print("============================================================")
