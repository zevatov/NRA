import os
import torchvision
from PIL import Image
from tqdm import tqdm

def dump_cifar10(root_dir, train=True):
    dataset = torchvision.datasets.CIFAR10(root='./data', train=train, download=True)
    prefix = "train" if train else "test"
    out_dir = os.path.join(root_dir, prefix)
    os.makedirs(out_dir, exist_ok=True)
    
    print(f"Dumping {prefix} split...")
    for idx in tqdm(range(len(dataset))):
        img, label = dataset[idx]
        # Format: label_index.png (e.g., 3_00001.png)
        filename = f"{label}_{idx:05d}.png"
        img.save(os.path.join(out_dir, filename))

if __name__ == "__main__":
    out_root = "./cifar10_raw"
    os.makedirs(out_root, exist_ok=True)
    dump_cifar10(out_root, train=True)
    dump_cifar10(out_root, train=False)
    print("Done dumping CIFAR-10!")
