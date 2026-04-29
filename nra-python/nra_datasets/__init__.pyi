from typing import Optional, Any

__version__: str

def load(
    archive_path: str,
    split: str = "train",
    streaming: bool = False,
) -> Any:
    """
    Load an NRA archive as a HuggingFace Dataset.
    
    Args:
        archive_path: Path to the .nra archive file or HTTP URL.
        split: Dataset split name (default: "train").
        streaming: If True, return an IterableDataset for lazy loading.
    
    Returns:
        A HuggingFace datasets.Dataset or datasets.IterableDataset.
    
    Example:
        >>> import nra_datasets
        >>> ds = nra_datasets.load("dataset.nra")
        >>> for item in ds:
        ...     print(item["file_id"], len(item["bytes"]))
    """
    ...
