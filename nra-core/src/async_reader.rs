//! NRA Async Reader: Zero-download streaming for Cloud Native workloads.
//!
//! This module provides the core abstraction for reading NRA BETA archives
//! from any async random-access source (HTTP Range, S3, local async file, etc.)

use crate::checksum::calc_crc32;

use crate::format::{HEADER_SIZE, NraHeader};
use crate::manifest::BetaManifest;
use std::collections::HashMap;
use std::io;
use std::sync::Arc;
use tokio::sync::{Mutex, OnceCell};

/// Maximum manifest size we will accept from an untrusted source (1 GiB).
/// Protects against malicious servers advertising absurdly large manifests.
const MAX_MANIFEST_SIZE: u64 = 1 << 30;

/// Abstract trait for stateless, asynchronous random access (e.g. HTTP Range).
pub trait AsyncRandomAccess: Send + Sync {
    /// Read exactly `length` bytes at `offset`.
    fn read_at(&self, offset: u64, length: usize) -> impl std::future::Future<Output = io::Result<Vec<u8>>> + Send;

    /// Get the total size of the resource, if known.
    fn size(&self) -> impl std::future::Future<Output = io::Result<u64>> + Send;
}

/// Maximum number of decompressed blocks to keep in the async cache.
/// At 4 MB per block, 64 entries ≈ 256 MB max cache footprint.
const MAX_CACHED_BLOCKS: usize = 64;

/// Asynchronous reader for NRA BETA archives (Cloud Streaming).
pub struct AsyncBetaReader<R: AsyncRandomAccess> {
    source: R,
    #[allow(dead_code)]
    header: NraHeader,
    manifest: BetaManifest,
    /// Fast lookup: chunk hash (hex) → index into chunk_table
    chunk_index: HashMap<String, usize>,
    /// Thread-safe block cache using single-flight pattern (OnceCell).
    /// Prevents the Thundering Herd problem: if 10 tasks request the same
    /// block concurrently, only the first one fetches it; the rest await.
    /// Bounded to MAX_CACHED_BLOCKS entries to prevent OOM.
    block_cache: Mutex<BlockCache>,
    /// Decoded Zstd dictionary for decompression (if archive was built with one).
    dictionary: Option<Vec<u8>>,
}

/// LRU-bounded block cache for async reader.
struct BlockCache {
    entries: HashMap<u64, Arc<OnceCell<Vec<u8>>>>,
    order: Vec<u64>,
}

impl<R: AsyncRandomAccess> AsyncBetaReader<R> {
    /// Open an archive from an async random access source (e.g. HTTP server).
    pub async fn open(source: R) -> io::Result<Self> {
        // 1. Read 32-byte header
        let header_buf = source.read_at(0, HEADER_SIZE).await?;
        let header_array: [u8; HEADER_SIZE] = header_buf
            .try_into()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid header length"))?;
        let header = NraHeader::from_bytes(&header_array)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // 2. Validate manifest size against DoS limit (КРИТИЧНО-2 fix)
        if header.manifest_size > MAX_MANIFEST_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Manifest size {} bytes exceeds safety limit of {} bytes",
                    header.manifest_size, MAX_MANIFEST_SIZE
                ),
            ));
        }

        // 3. Read manifest bytes
        let manifest_buf = source
            .read_at(header.manifest_offset, header.manifest_size as usize)
            .await?;

        // 4. Deserialize manifest
        let manifest = BetaManifest::deserialize(&manifest_buf)?;

        // 5. Decode Zstd dictionary if present (CRIT-1 fix)
        let dictionary = if let Some(dict_b64) = &manifest.dictionary {
            use base64::{Engine as _, engine::general_purpose::STANDARD};
            Some(STANDARD.decode(dict_b64).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Dictionary decode error: {}", e))
            })?)
        } else {
            None
        };

        let chunk_index: HashMap<String, usize> = manifest
            .chunk_table
            .iter()
            .enumerate()
            .map(|(i, c)| (c.hash.clone(), i))
            .collect();

        Ok(Self {
            source,
            header,
            manifest,
            chunk_index,
            block_cache: Mutex::new(BlockCache {
                entries: HashMap::new(),
                order: Vec::new(),
            }),
            dictionary,
        })
    }

    pub fn manifest(&self) -> &BetaManifest {
        &self.manifest
    }

    pub fn file_ids(&self) -> Vec<&str> {
        self.manifest.files.iter().map(|f| f.id.as_str()).collect()
    }

    /// Read and reconstruct a file asynchronously.
    pub async fn read_file(&self, file_id: &str) -> io::Result<Vec<u8>> {
        let file_record = self
            .manifest
            .files
            .iter()
            .find(|f| f.id == file_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "File not found in BETA manifest"))?;

        let expected_size = file_record.original_size as usize;

        let mut result = Vec::with_capacity(expected_size);

        for hash_hex in &file_record.chunks {
            let chunk_data = self.read_chunk(hash_hex).await?;
            result.extend_from_slice(&chunk_data);
        }

        if result.len() != expected_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Reconstructed size mismatch for '{}': expected {}, got {}",
                    file_id, expected_size, result.len()
                ),
            ));
        }

        Ok(result)
    }

    /// Fetch and decompress a specific chunk asynchronously.
    ///
    /// Uses a single-flight pattern (OnceCell) to prevent the Thundering Herd
    /// problem: if N tasks request chunks from the same block concurrently,
    /// only the first one performs the HTTP fetch + Zstd decompression.
    /// The remaining N-1 tasks await the same OnceCell future.
    async fn read_chunk(&self, hash_hex: &str) -> io::Result<Vec<u8>> {
        let idx = *self
            .chunk_index
            .get(hash_hex)
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::NotFound,
                    format!("Chunk not found: {}", hash_hex),
                )
            })?;

        let record = &self.manifest.chunk_table[idx];
        let block_offset = record.offset;
        let compressed_size = record.compressed_size as usize;
        let inner_offset = record.inner_offset as usize;
        let original_size = record.original_size as usize;
        let expected_crc = record.crc32;

        // Single-flight pattern via OnceCell.
        // Acquire lock only briefly to get-or-insert the OnceCell, then drop it.
        let cell = {
            let mut cache = self.block_cache.lock().await;

            // WARN-4 fix: Evict oldest block if cache is full
            #[allow(clippy::collapsible_if)]
            if cache.entries.len() >= MAX_CACHED_BLOCKS
                && !cache.entries.contains_key(&block_offset)
            {
                if let Some(oldest) = cache.order.first().copied() {
                    cache.entries.remove(&oldest);
                    cache.order.remove(0);
                }
            }

            let cell = cache
                .entries
                .entry(block_offset)
                .or_insert_with(|| Arc::new(OnceCell::new()))
                .clone();

            if !cache.order.contains(&block_offset) {
                cache.order.push(block_offset);
            }

            cell
        }; // Mutex released here — no lock held during I/O.

        // Only the first caller to reach this point will execute the closure.
        // All other concurrent callers for the same block_offset will await.
        let dict_ref = self.dictionary.as_deref();
        let block_data = cell
            .get_or_try_init(|| async {
                let buf = self.source.read_at(block_offset, compressed_size).await?;

                // Verify CRC32 before decompression
                let computed_crc = calc_crc32(&buf);
                if computed_crc != expected_crc {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "CRC32 mismatch for block at offset {}: expected 0x{:08X}, got 0x{:08X}",
                            block_offset, expected_crc, computed_crc
                        ),
                    ));
                }

                // CRIT-1 fix: pass dictionary to decompressor
                crate::codec::decompress(&buf, dict_ref)
            })
            .await?;

        let end = inner_offset + original_size;

        if end > block_data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Chunk inner offset out of bounds: {}..{} > {}",
                    inner_offset, end, block_data.len()
                ),
            ));
        }

        // Clone the specific chunk bytes out of the cached block
        Ok(block_data[inner_offset..end].to_vec())
    }
}
