import os
import shutil
import tempfile
import subprocess
try:
    import torchvision
    from PIL import Image
except ImportError:
    print("Please install torchvision and Pillow: pip install torchvision Pillow")
    exit(1)

def main():
    print("📥 Downloading CIFAR-10 dataset...")
    # We use a temporary directory for the raw dataset
    dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True)
    classes = dataset.classes

    # Create a temporary directory to unpack the images
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"📦 Extracting images to {tmpdir}...")
        for i, (img, label_idx) in enumerate(dataset):
            label = classes[label_idx]
            label_dir = os.path.join(tmpdir, label)
            os.makedirs(label_dir, exist_ok=True)
            
            img_path = os.path.join(label_dir, f"{i:05d}.png")
            img.save(img_path)

        out_nra = os.path.abspath("cifar10.nra")
        if os.path.exists(out_nra):
            os.remove(out_nra)

        print(f"🚀 Packing dataset into {out_nra} using NRA-CLI...")
        # Make sure nra-cli is built
        cargo_build_cmd = ["cargo", "build", "--release", "-p", "nra-cli"]
        subprocess.run(cargo_build_cmd, check=True, cwd=os.path.join(os.path.dirname(__file__), ".."))

        cli_path = os.path.join(os.path.dirname(__file__), "..", "target", "release", "nra-cli")
        
        pack_cmd = [
            cli_path,
            "pack-beta",
            "--input", tmpdir,
            "--output", out_nra,
            "--name", "CIFAR-10 NRA",
            "--dictionary"  # Use zstd dictionary for better compression of small images
        ]
        
        subprocess.run(pack_cmd, check=True)
        
        print("\n📊 Archive Information:")
        info_cmd = [cli_path, "info-beta", "--input", out_nra]
        subprocess.run(info_cmd, check=True)
        
        print("\n✅ Verification (Roundtrip)...")
        with tempfile.TemporaryDirectory() as verify_dir:
            unpack_cmd = [cli_path, "unpack-beta", "--input", out_nra, "--output", verify_dir]
            subprocess.run(unpack_cmd, check=True)
            
            # Simple check: count files
            unpacked_count = 0
            for root, _, files in os.walk(verify_dir):
                unpacked_count += len(files)
                
            print(f"✅ Unpacked {unpacked_count} files successfully.")
            if unpacked_count != len(dataset):
                print(f"❌ Verification failed! Expected {len(dataset)}, got {unpacked_count}")
                exit(1)
                
        print("\n🎉 CIFAR-10 converted to NRA successfully!")

if __name__ == "__main__":
    main()
