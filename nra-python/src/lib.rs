#![allow(unsafe_op_in_unsafe_fn)]

use nra_core::{NraReader, beta_reader::BetaReader};
use pyo3::exceptions::{PyIOError, PyKeyError, PyRuntimeError};
use pyo3::prelude::*;
use std::sync::{Arc, Mutex};

/// Neural Ready Archive (NRA) Python Bindings.
/// Allows zero-copy (or low-copy) access to datasets straight from the archive.
#[pyclass]
pub struct Archive {
    // We wrap reader in Arc<Mutex> because PyO3 might share it across threads,
    // though realistically a DataLoader handles concurrent reads via multiprocessing (forking).
    // In actual production, we might want a lock-free reader using mmap or pread,
    // but Mutex<NraReader> is sufficient for the MVP.
    inner: Arc<Mutex<NraReader>>,
}

#[pymethods]
impl Archive {
    /// Open an .nra archive from a given path.
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let reader = NraReader::open(path)
            .map_err(|e| PyIOError::new_err(format!("Failed to open archive: {}", e)))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(reader)),
        })
    }

    /// Get a list of all file IDs in the archive.
    pub fn file_ids(&self) -> PyResult<Vec<String>> {
        let reader = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Poisoned lock"))?;
        Ok(reader
            .file_ids()
            .into_iter()
            .map(|s| s.to_string())
            .collect())
    }

    /// Read a file by its ID. Returns the raw bytes.
    pub fn read_file(&self, file_id: &str) -> PyResult<Vec<u8>> {
        let mut reader = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Poisoned lock"))?;
        match reader.read_file(file_id) {
            Ok(data) => Ok(data),
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Err(PyKeyError::new_err(format!(
                        "File '{}' not found in archive",
                        file_id
                    )))
                } else {
                    Err(PyIOError::new_err(format!("Failed to read file: {}", e)))
                }
            }
        }
    }

    /// Total number of files in the archive.
    pub fn len(&self) -> PyResult<usize> {
        let reader = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Poisoned lock"))?;
        Ok(reader.manifest().summary.total_files as usize)
    }

    /// Magic method to allow len(archive)
    pub fn __len__(&self) -> PyResult<usize> {
        self.len()
    }

    /// Magic method to allow archive["file_id"]
    pub fn __getitem__(&self, file_id: &str) -> PyResult<Vec<u8>> {
        self.read_file(file_id)
    }
}

/// Neural Ready Archive (NRA) BETA Python Bindings.
/// Supports reading CDC-deduplicated and Solid-compressed archives.
#[pyclass]
pub struct BetaArchive {
    inner: Arc<Mutex<BetaReader>>,
}

#[pymethods]
impl BetaArchive {
    #[new]
    pub fn new(path: &str) -> PyResult<Self> {
        let reader = BetaReader::open(path)
            .map_err(|e| PyIOError::new_err(format!("Failed to open BETA archive: {}", e)))?;
        Ok(Self {
            inner: Arc::new(Mutex::new(reader)),
        })
    }

    pub fn file_ids(&self) -> PyResult<Vec<String>> {
        let reader = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Poisoned lock"))?;
        Ok(reader
            .file_ids()
            .iter()
            .map(|s| s.to_string())
            .collect())
    }

    pub fn read_file(&self, file_id: &str) -> PyResult<Vec<u8>> {
        let mut reader = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Poisoned lock"))?;
        match reader.read_file(file_id) {
            Ok(data) => Ok(data),
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Err(PyKeyError::new_err(format!(
                        "File '{}' not found in BETA archive",
                        file_id
                    )))
                } else {
                    Err(PyIOError::new_err(format!("Failed to read file: {}", e)))
                }
            }
        }
    }

    pub fn len(&self) -> PyResult<usize> {
        let reader = self
            .inner
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Poisoned lock"))?;
        Ok(reader.manifest().summary.total_files as usize)
    }

    pub fn __len__(&self) -> PyResult<usize> {
        self.len()
    }

    pub fn __getitem__(&self, file_id: &str) -> PyResult<Vec<u8>> {
        self.read_file(file_id)
    }
}



use nra_core::AsyncBetaReader;
use nra_registry::HttpRandomAccess;

/// Cloud Streaming Archive.
/// Connects directly to an HTTP URL and fetches only requested byte ranges.
///
/// # ⚠️ PyTorch DataLoader Safety
///
/// Do NOT create a `CloudArchive` inside `__init__` of a `torch.utils.data.Dataset`
/// if you plan to use `DataLoader(num_workers > 0)`. PyTorch forks worker processes
/// via `os.fork()`, and Tokio Runtime is NOT fork-safe.
///
/// **Correct pattern** (lazy initialization):
/// ```python
/// class MyDataset(Dataset):
///     def __init__(self, url):
///         self.url = url
///         self._archive = None  # Do NOT create CloudArchive here
///
///     def __getitem__(self, idx):
///         if self._archive is None:
///             self._archive = nra.CloudArchive(self.url)  # Created per-worker
///         return self._archive.read_file(...)
/// ```
#[pyclass]
pub struct CloudArchive {
    // СЕРЬЁЗНО-2 fix: `rt` is listed first so it is dropped LAST.
    // Rust drops fields top-to-bottom. `inner` (which holds Tokio Mutex,
    // reqwest::Client, etc.) must be dropped while the Runtime is still alive.
    rt: Arc<tokio::runtime::Runtime>,
    inner: Arc<AsyncBetaReader<HttpRandomAccess>>,
}

#[pymethods]
impl CloudArchive {
    #[new]
    pub fn new(url: &str) -> PyResult<Self> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to start tokio runtime: {}", e)))?;
        
        let url_clone = url.to_string();
        let reader = rt.block_on(async move {
            let streamer = HttpRandomAccess::new(&url_clone);
            AsyncBetaReader::open(streamer).await
        }).map_err(|e| PyIOError::new_err(format!("Failed to open cloud archive: {}", e)))?;

        Ok(Self {
            inner: Arc::new(reader),
            rt: Arc::new(rt),
        })
    }

    pub fn file_ids(&self) -> PyResult<Vec<String>> {
        Ok(self.inner.file_ids().into_iter().map(|s| s.to_string()).collect())
    }

    pub fn len(&self) -> PyResult<usize> {
        Ok(self.inner.manifest().summary.total_files as usize)
    }

    pub fn __len__(&self) -> PyResult<usize> {
        self.len()
    }

    pub fn read_file(&self, py: Python<'_>, file_id: &str) -> PyResult<Vec<u8>> {
        let inner = self.inner.clone();
        let rt = self.rt.clone();
        let file_id_str = file_id.to_string();

        let result = py.allow_threads(move || {
            rt.block_on(async move {
                inner.read_file(&file_id_str).await
            })
        });

        match result {
            Ok(data) => Ok(data),
            Err(e) => {
                if e.kind() == std::io::ErrorKind::NotFound {
                    Err(PyKeyError::new_err(format!("File '{}' not found in cloud archive", file_id)))
                } else {
                    Err(PyIOError::new_err(format!("Failed to read file from cloud: {}", e)))
                }
            }
        }
    }

    pub fn __getitem__(&self, py: Python<'_>, file_id: &str) -> PyResult<Vec<u8>> {
        self.read_file(py, file_id)
    }
}

// ============================================================
// Stage 3-4: New Python Bindings
// ============================================================

use nra_core::sampler;
use nra_core::vector_index;

/// Deterministic sampler for distributed training.
/// Guarantees identical sample ordering across any GPU/worker count.
#[pyclass]
pub struct DeterministicSampler {
    inner: sampler::DeterministicSampler,
}

#[pymethods]
impl DeterministicSampler {
    #[new]
    pub fn new(seed: u64, dataset_size: usize) -> Self {
        Self {
            inner: sampler::DeterministicSampler::new(seed, dataset_size),
        }
    }

    /// Get the full shuffled permutation for an epoch.
    pub fn permutation(&self, epoch: u64) -> Vec<usize> {
        self.inner.permutation(epoch)
    }

    /// Get this worker's shard of indices.
    pub fn shard(&self, epoch: u64, rank: usize, world_size: usize) -> Vec<usize> {
        self.inner.shard(epoch, rank, world_size)
    }
}

/// Checkpoint state for mid-epoch resumption.
#[pyclass]
pub struct DataLoaderCheckpoint {
    inner: sampler::DataLoaderCheckpoint,
}

#[pymethods]
impl DataLoaderCheckpoint {
    #[new]
    pub fn new(seed: u64, dataset_size: usize, epoch: u64, batch_index: u64, world_size: usize, rank: usize) -> Self {
        Self {
            inner: sampler::DataLoaderCheckpoint::new(seed, dataset_size, epoch, batch_index, world_size, rank),
        }
    }

    /// Save checkpoint to JSON bytes.
    pub fn save(&self) -> PyResult<Vec<u8>> {
        self.inner.save().map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Load checkpoint from JSON bytes.
    #[staticmethod]
    pub fn load(data: &[u8]) -> PyResult<Self> {
        let inner = sampler::DataLoaderCheckpoint::load(data)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Get remaining indices after skipping processed batches.
    pub fn remaining_indices(&self, batch_size: usize) -> Vec<usize> {
        self.inner.remaining_indices(batch_size)
    }

    #[getter]
    pub fn epoch(&self) -> u64 { self.inner.epoch }
    #[getter]
    pub fn batch_index(&self) -> u64 { self.inner.batch_index }
    #[getter]
    pub fn rank(&self) -> usize { self.inner.rank }
}

/// In-memory vector index for ANN/RAG search.
#[pyclass]
pub struct VectorIndex {
    inner: vector_index::VectorIndex,
}

#[pymethods]
impl VectorIndex {
    #[new]
    pub fn new(dimension: usize) -> Self {
        Self {
            inner: vector_index::VectorIndex::new(dimension),
        }
    }

    /// Add an embedding for a file_id.
    pub fn insert(&mut self, file_id: &str, vector: Vec<f32>) {
        self.inner.insert(file_id, vector);
    }

    /// Search for top_k nearest neighbors by cosine similarity.
    pub fn search(&self, query: Vec<f32>, top_k: usize) -> Vec<(String, f32)> {
        self.inner.search(&query, top_k)
            .into_iter()
            .map(|r| (r.file_id, r.score))
            .collect()
    }

    pub fn __len__(&self) -> usize {
        self.inner.len()
    }

    /// Serialize index to JSON bytes.
    pub fn serialize(&self) -> PyResult<Vec<u8>> {
        self.inner.serialize().map_err(|e| PyIOError::new_err(e.to_string()))
    }

    /// Load index from JSON bytes.
    #[staticmethod]
    pub fn deserialize(data: &[u8], dimension: usize) -> PyResult<Self> {
        let inner = vector_index::VectorIndex::deserialize(data, dimension)
            .map_err(|e| PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn nra(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Archive>()?;
    m.add_class::<BetaArchive>()?;
    m.add_class::<CloudArchive>()?;
    m.add_class::<DeterministicSampler>()?;
    m.add_class::<DataLoaderCheckpoint>()?;
    m.add_class::<VectorIndex>()?;
    Ok(())
}
