import os
import json
import random
import shutil
from tqdm import tqdm

DATA_DIR = "/tmp/nra_ultimate_data"
os.makedirs(DATA_DIR, exist_ok=True)

print("🚀 СТАРТ: Генерация данных для Ultimate Benchmark")

# ============================================================
# Dataset A: Real Vision (High-Res Images)
# ============================================================
DIR_A = os.path.join(DATA_DIR, "dataset_a_vision")
if not os.path.exists(DIR_A):
    print("\n📦 Создание Dataset A (Vision - настоящие картинки)...")
    os.makedirs(DIR_A)
    # Используем библиотеку datasets для загрузки реальных фоток еды (Food-101) или кошек/собак
    try:
        from datasets import load_dataset
        ds = load_dataset("Bingsu/Cat_and_Dog", split="train[:2000]") # 2000 реальных фото
        for i, item in enumerate(tqdm(ds, desc="Сохранение фото JPEG")):
            img = item['image']
            img.save(os.path.join(DIR_A, f"img_{i:05d}.jpg"), "JPEG")
    except Exception as e:
        print(f"⚠️ Ошибка загрузки из HF (нет интернета/прав), генерируем фейковые 'тяжелые' картинки: {e}")
        from PIL import Image
        for i in tqdm(range(2000), desc="Генерация заглушек"):
            img = Image.new('RGB', (512, 512), color=(random.randint(0,255), 100, 100))
            img.save(os.path.join(DIR_A, f"img_{i:05d}.jpg"), "JPEG")
else:
    print("\n✅ Dataset A уже существует.")

# ============================================================
# Dataset B: Heavy Duplication (LLM JSON Logs)
# ============================================================
DIR_B = os.path.join(DATA_DIR, "dataset_b_duplication")
if not os.path.exists(DIR_B):
    print("\n📦 Создание Dataset B (LLM Logs с 40% дубликатов)...")
    os.makedirs(DIR_B)
    
    # Создаем "уникальные" шаблоны для логов
    templates = [
        {"system": "You are a helpful assistant.", "user": "How to learn Rust?", "assistant": "Start with the Book."},
        {"system": "You are a coder.", "user": "Write hello world in Python", "assistant": "print('hello world')"},
        {"system": "Analyze this data.", "data": "A"*5000, "result": "Lots of A's"} # Тяжелый блок текста
    ]
    
    # 5000 файлов: 3000 уникальных, 2000 - полные копии старых (для проверки дедупликации)
    for i in tqdm(range(5000), desc="Генерация JSON"):
        if i < 3000:
            # Уникальные
            data = random.choice(templates).copy()
            data["timestamp"] = i
            data["noise"] = "X" * random.randint(10, 500)
        else:
            # Дубликаты (копируем файл 0-1999 в точности)
            with open(os.path.join(DIR_B, f"log_{i-3000:05d}.json"), "r") as f:
                data = json.load(f)
        
        with open(os.path.join(DIR_B, f"log_{i:05d}.json"), "w") as f:
            json.dump(data, f)
else:
    print("✅ Dataset B уже существует.")

# ============================================================
# Dataset C: Multimodal Chaos
# ============================================================
DIR_C = os.path.join(DATA_DIR, "dataset_c_multimodal")
if not os.path.exists(DIR_C):
    print("\n📦 Создание Dataset C (Хаос: картинки + длинный текст + json)...")
    os.makedirs(DIR_C)
    
    # Берем немного картинок из A
    files_a = os.listdir(DIR_A)[:500]
    for f in tqdm(files_a, desc="Микс картинок"):
        shutil.copy(os.path.join(DIR_A, f), os.path.join(DIR_C, f))
    
    # Добавляем длинные тексты
    for i in tqdm(range(500), desc="Генерация длинных текстов"):
        with open(os.path.join(DIR_C, f"doc_{i:04d}.txt"), "w") as f:
            f.write("СЛОВО " * 2000) # Длинный текст (отлично жмется)
            
    # Добавляем маленькие метаданные
    for i in tqdm(range(500), desc="Генерация мета-тегов"):
        with open(os.path.join(DIR_C, f"meta_{i:04d}.json"), "w") as f:
            json.dump({"id": i, "tags": ["cat", "cute", "fluffy"]}, f)
else:
    print("✅ Dataset C уже существует.")

print("\n🎉 Все датасеты готовы в /tmp/nra_ultimate_data/")
