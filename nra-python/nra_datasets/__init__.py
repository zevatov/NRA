"""
HuggingFace Datasets integration for Neural Ready Archive (NRA).
"""

__version__ = "0.1.0"

def load(archive_path, split="train", streaming=False):
    """
    Load NRA archive as HuggingFace Dataset.
    
    Usage:
        import nra_datasets
        ds = nra_datasets.load("/path/to/data.nra")
        for item in ds:
            print(item["file_id"], len(item["bytes"]))
    """
    import datasets
    import os
    
    script_path = os.path.join(os.path.dirname(__file__), "nra_loader.py")
    
    return datasets.load_dataset(
        script_path,
        data_files=archive_path,
        split=split,
        streaming=streaming,
    )
