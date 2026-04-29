#!/usr/bin/env python3
"""
Integration test: NRA HuggingFace Datasets loader.
Requires: pip install nra datasets
"""
import sys
import os
import tempfile
import subprocess

def test_hf_loader():
    # 1. Generate test files
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(10):
            path = os.path.join(tmpdir, f"file_{i:04d}.txt")
            with open(path, "w") as f:
                f.write(f"Hello from file {i}\n" * 10)

        # 2. Pack via nra-cli
        archive_path = os.path.join(tmpdir, "test.nra")
        project_root = os.path.dirname(os.path.dirname(__file__))
        
        result = subprocess.run(
            ["cargo", "run", "--release", "-p", "nra-cli", "--", "pack-beta",
             "--input", tmpdir, "--output", archive_path],
            capture_output=True, text=True, cwd=project_root
        )
        assert result.returncode == 0, f"pack-beta failed: {result.stderr}"

        # 3. Load via nra_datasets
        try:
            import nra_datasets
            import datasets
            
            # Use datasets module just to confirm it's available
            _ = datasets.__version__
            
            ds = nra_datasets.load(archive_path)
            
            items = list(ds)
            assert len(items) == 10, f"Expected 10 items, got {len(items)}"
            
            for item in items:
                assert "file_id" in item, "Missing file_id key"
                assert "bytes" in item, "Missing bytes key"
                assert len(item["bytes"]) > 0, "Empty bytes"
            
            print(f"✅ HF Loader test passed: {len(items)} items loaded")
            sys.exit(0)
            
        except ImportError as e:
            print(f"⚠️  Skipped HF test (missing dependency): {e}")
            sys.exit(0) # Graceful skip

if __name__ == "__main__":
    test_hf_loader()
