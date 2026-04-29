//! NRA Tensor: Zero-Copy Loading and GPUDirect Storage Placeholder.
//!
//! Provides two tiers of GPU integration:
//! - **Local Maximum:** `MmapTensorReader` for zero-copy SafeTensors loading via `memmap2`.
//!   Works on all platforms (macOS, Linux, Windows).
//! - **Enterprise GPUDirect:** Placeholder for NVIDIA GDS / kvikio integration.
//!   Requires Linux + datacenter GPUs (A100/H100).

use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

/// Information about a tensor parsed directly from the disk.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    /// (byte_offset_in_file, byte_length)
    pub data_offset: (usize, usize),
}

/// A zero-copy reader for SafeTensors that maps the file directly to memory.
/// This completely bypasses RAM allocation for the tensor data.
///
/// # Safety Contract (for callers)
///
/// - The file **MUST NOT** be truncated or deleted while this reader exists.
/// - On macOS, file deletion during active mmap causes **SIGBUS** (process abort).
///   On Linux, the data remains accessible until the last fd is closed.
/// - **Not recommended** for files on network filesystems (NFS, CIFS) due to
///   potential stale-data reads without notification.
/// - Memory alignment: SafeTensors format does not guarantee specific alignment
///   of tensor data. For GPU transfers, PyTorch handles re-alignment internally
///   when calling `.to("cuda")` or `.to("mps")`.
pub struct MmapTensorReader {
    mmap: memmap2::Mmap,
    /// Cached tensor metadata parsed once at open() time.
    /// Maps tensor name → (byte_offset_in_mmap, byte_length).
    tensor_offsets: HashMap<String, TensorInfo>,
    /// Keep file handle alive to prevent unlink-while-mapped on Linux.
    _file: File,
}

impl MmapTensorReader {
    /// Memory map a SafeTensors file from disk.
    ///
    /// # Safety
    ///
    /// Uses `unsafe` for memory mapping. The caller must ensure that:
    /// - The file is not modified, truncated, or deleted externally while
    ///   this `MmapTensorReader` is alive.
    /// - The file is on a local filesystem (not NFS/CIFS).
    pub fn open<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        // Safety: The file must not be modified externally while it's memory mapped.
        // We keep `_file` alive to hold the fd open, which on Linux prevents
        // actual data deletion even if the file is unlinked.
        let mmap = unsafe { MmapOptions::new().map(&file)? };

        // РЕКОМЕНДАЦИЯ-1 fix: parse header ONCE and cache all tensor metadata.
        let tensor_offsets = Self::parse_tensors(&mmap)?;

        Ok(Self {
            mmap,
            tensor_offsets,
            _file: file,
        })
    }

    /// Parse all tensor metadata from the SafeTensors header (done once).
    fn parse_tensors(mmap: &[u8]) -> std::io::Result<HashMap<String, TensorInfo>> {
        let st = SafeTensors::deserialize(mmap)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{}", e)))?;

        let base_ptr = mmap.as_ptr() as usize;
        let mut map = HashMap::new();

        for name in st.names() {
            let tensor_view = st.tensor(name)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, format!("{}", e)))?;
            let offset = tensor_view.data().as_ptr() as usize - base_ptr;
            let length = tensor_view.data().len();
            map.insert(
                name.to_string(),
                TensorInfo {
                    name: name.to_string(),
                    shape: tensor_view.shape().to_vec(),
                    dtype: format!("{:?}", tensor_view.dtype()),
                    data_offset: (offset, length),
                },
            );
        }

        Ok(map)
    }

    /// Return cached metadata for all tensors. O(1) after open().
    pub fn inspect(&self) -> Vec<&TensorInfo> {
        self.tensor_offsets.values().collect()
    }

    /// Get the number of tensors in the file.
    pub fn len(&self) -> usize {
        self.tensor_offsets.len()
    }

    /// Check if the file contains no tensors.
    pub fn is_empty(&self) -> bool {
        self.tensor_offsets.is_empty()
    }

    /// Get a raw byte slice for a specific tensor.
    /// This is an O(1) operation because offsets are cached at open() time.
    /// PyTorch can use this slice to send data directly to the GPU
    /// via `.to("cuda")` or `.to("mps")`.
    pub fn get_tensor_bytes(&self, name: &str) -> std::io::Result<&[u8]> {
        let info = self.tensor_offsets.get(name).ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::NotFound,
                format!("Tensor '{}' not found", name),
            )
        })?;

        let (offset, length) = info.data_offset;
        Ok(&self.mmap[offset..offset + length])
    }
}

/// NVIDIA GPUDirect Storage (GDS) / kvikio integration stub.
/// To be implemented for AWS/Lambda Labs Enterprise instances.
///
/// When available, this module will provide:
/// - Direct NVMe → GPU VRAM transfers via PCIe (bypassing CPU RAM entirely)
/// - Requires: Linux, NVIDIA datacenter GPUs (A100/H100), MLNX_OFED drivers
///
/// See STARTUP_ROADMAP.md §2 "Enterprise: GPU Direct Storage" for roadmap.
pub mod gpudirect {
    pub struct GpuDirectStorage;

    impl GpuDirectStorage {
        pub fn load_tensor_to_vram(_nvme_offset: u64, _size: usize, _gpu_addr: usize) {
            unimplemented!(
                "GPUDirect Storage requires Linux, NVIDIA Datacenter GPUs (A100/H100), \
                 and specific NVMe drivers. See STARTUP_ROADMAP.md for Enterprise availability."
            );
        }
    }
}
