"""
NRA Multi-Framework Adapters
============================
Drop-in integrations for PyTorch, TensorFlow, and HuggingFace Datasets.

Usage:
    # PyTorch
    from nra.adapters import NRADataset
    ds = NRADataset("/path/to/archive.nra", transform=my_transform)
    loader = DataLoader(ds, batch_size=32)

    # TensorFlow
    from nra.adapters import nra_tf_dataset
    tf_ds = nra_tf_dataset("/path/to/archive.nra")

    # HuggingFace
    from nra.adapters import NRAHuggingFaceDataset
    hf_ds = NRAHuggingFaceDataset("/path/to/archive.nra")
"""

import io


# ============================================================
# PyTorch Adapter
# ============================================================

def NRADataset(archive_path, transform=None, cloud_url=None):
    """
    Create a PyTorch Dataset backed by an NRA archive.
    
    Args:
        archive_path: Path to local .nra file (or None if using cloud_url)
        transform: Optional torchvision transform
        cloud_url: Optional HTTP URL for cloud streaming
    """
    import torch.utils.data as data
    import nra as nra_lib
    
    class _NRADataset(data.Dataset):
        def __init__(self):
            if cloud_url:
                self._archive = nra_lib.CloudArchive(cloud_url)
            else:
                self._archive = nra_lib.BetaArchive(archive_path)
            self._file_ids = self._archive.file_ids()
            self._transform = transform
        
        def __len__(self):
            return len(self._file_ids)
        
        def __getitem__(self, idx):
            raw = bytearray(self._archive.read_file(self._file_ids[idx]))
            
            if self._transform:
                from PIL import Image
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                return self._transform(img), 0
            
            return raw, self._file_ids[idx]
    
    return _NRADataset()


# ============================================================
# PyTorch Distributed Adapter (with Deterministic Sampler)
# ============================================================

def NRADistributedDataset(archive_path, seed=42, transform=None):
    """
    PyTorch Dataset with built-in Elastic Determinism.
    
    Usage with DistributedDataParallel:
        ds = NRADistributedDataset("data.nra", seed=42)
        sampler = ds.get_sampler(epoch=0, rank=rank, world_size=world_size)
        loader = DataLoader(ds, sampler=sampler)
    """
    import torch.utils.data as data
    import nra as nra_lib
    
    class _NRADistDataset(data.Dataset):
        def __init__(self):
            self._archive = nra_lib.BetaArchive(archive_path)
            self._file_ids = self._archive.file_ids()
            self._transform = transform
            self._sampler = nra_lib.DeterministicSampler(seed, len(self._file_ids))
        
        def __len__(self):
            return len(self._file_ids)
        
        def __getitem__(self, idx):
            raw = bytearray(self._archive.read_file(self._file_ids[idx]))
            if self._transform:
                from PIL import Image
                img = Image.open(io.BytesIO(raw)).convert("RGB")
                return self._transform(img), 0
            return raw, self._file_ids[idx]
        
        def get_sampler(self, epoch, rank, world_size):
            """Get a deterministic sampler shard for this worker."""
            indices = self._sampler.shard(epoch, rank, world_size)
            return data.SubsetRandomSampler(indices)
    
    return _NRADistDataset()


# ============================================================
# TensorFlow Adapter
# ============================================================

def nra_tf_dataset(archive_path, output_signature=None):
    """
    Create a tf.data.Dataset from an NRA archive.
    
    Args:
        archive_path: Path to local .nra file
        output_signature: Optional tf.TensorSpec (defaults to variable-length bytes)
    
    Returns:
        tf.data.Dataset yielding raw bytes for each file
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow is required. Install with: pip install tensorflow")
    
    import nra as nra_lib
    
    archive = nra_lib.BetaArchive(archive_path)
    file_ids = archive.file_ids()
    
    def generator():
        a = nra_lib.BetaArchive(archive_path)
        for fid in file_ids:
            raw = bytes(bytearray(a.read_file(fid)))
            yield raw
    
    if output_signature is None:
        output_signature = tf.TensorSpec(shape=(), dtype=tf.string)
    
    return tf.data.Dataset.from_generator(generator, output_signature=output_signature)


# ============================================================
# HuggingFace Datasets Adapter
# ============================================================

class NRAHuggingFaceDataset:
    """
    A HuggingFace-compatible dataset backed by NRA.
    
    Usage:
        ds = NRAHuggingFaceDataset("data.nra")
        for item in ds:
            print(item["file_id"], len(item["bytes"]))
    """
    
    def __init__(self, archive_path):
        import nra as nra_lib
        self._archive = nra_lib.BetaArchive(archive_path)
        self._file_ids = self._archive.file_ids()
    
    def __len__(self):
        return len(self._file_ids)
    
    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return [self[i] for i in range(*idx.indices(len(self)))]
        fid = self._file_ids[idx]
        raw = bytes(bytearray(self._archive.read_file(fid)))
        return {"file_id": fid, "bytes": raw}
    
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]
    
    def select(self, indices):
        """Return a subset (HuggingFace-style)."""
        class Subset:
            def __init__(self_inner, parent, indices):
                self_inner._parent = parent
                self_inner._indices = indices
            def __len__(self_inner):
                return len(self_inner._indices)
            def __getitem__(self_inner, idx):
                return self_inner._parent[self_inner._indices[idx]]
            def __iter__(self_inner):
                for i in self_inner._indices:
                    yield self_inner._parent[i]
        return Subset(self, indices)

# ============================================================
# HuggingFace Native Loader Shortcut
# ============================================================
def load_hf_dataset(archive_path, split="train", streaming=False):
    import nra_datasets
    return nra_datasets.load(archive_path, split=split, streaming=streaming)
