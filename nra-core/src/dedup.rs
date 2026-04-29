//! Content-Defined Deduplication Engine for NRA BETA.
//!
//! Uses FastCDC (Content-Defined Chunking) to split files into variable-size
//! chunks and BLAKE3 to hash them. Identical chunks across files are stored
//! only once, achieving massive deduplication on real ML datasets.

use fastcdc::v2020::FastCDC;
use std::collections::HashMap;

/// A content-defined chunk with its hash and data.
#[derive(Debug, Clone)]
pub struct Chunk {
    pub hash: [u8; 32],
    pub data: Vec<u8>,
}

/// The "recipe" for reconstructing a file: an ordered list of chunk hashes.
#[derive(Debug, Clone)]
pub struct FileRecipe {
    pub file_id: String,
    pub original_size: u64,
    pub chunk_hashes: Vec<[u8; 32]>,
}

/// Global chunk store that deduplicates chunks across files.
pub struct ChunkStore {
    /// hash → compressed chunk data
    chunks: HashMap<[u8; 32], Vec<u8>>,
    /// Insertion order for deterministic output
    order: Vec<[u8; 32]>,
    /// hash → insertion index for O(1) lookup
    index_map: HashMap<[u8; 32], usize>,
    /// Total bytes before dedup
    pub total_input_bytes: u64,
    /// Total unique bytes after dedup
    pub total_unique_bytes: u64,
}

/// CDC tuning parameters.
/// min_size / avg_size / max_size control the chunk granularity.
/// Smaller chunks = better dedup ratio but more metadata overhead.
const CDC_MIN_SIZE: u32 = 2 * 1024;     // 2 KB
const CDC_AVG_SIZE: u32 = 8 * 1024;     // 8 KB
const CDC_MAX_SIZE: u32 = 64 * 1024;    // 64 KB

impl Default for ChunkStore {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkStore {
    pub fn new() -> Self {
        Self {
            chunks: HashMap::new(),
            order: Vec::new(),
            index_map: HashMap::new(),
            total_input_bytes: 0,
            total_unique_bytes: 0,
        }
    }

    /// Ingest a file: split into CDC chunks, hash, and deduplicate.
    /// Returns a recipe (list of chunk hashes) for reconstructing the file.
    pub fn ingest(&mut self, file_id: &str, data: &[u8]) -> FileRecipe {
        let (recipe, chunks) = chunk_data(file_id, data);
        self.ingest_chunks(&recipe, chunks);
        recipe
    }

    pub fn ingest_chunks(&mut self, recipe: &FileRecipe, chunks: Vec<Chunk>) {
        self.total_input_bytes += recipe.original_size;

        for chunk in chunks {
            if !self.chunks.contains_key(&chunk.hash) {
                self.total_unique_bytes += chunk.data.len() as u64;
                let idx = self.order.len();
                self.index_map.insert(chunk.hash, idx);
                self.chunks.insert(chunk.hash, chunk.data);
                self.order.push(chunk.hash);
            }
        }
    }
}

pub fn chunk_data(file_id: &str, data: &[u8]) -> (FileRecipe, Vec<Chunk>) {
    let mut chunk_hashes = Vec::new();
    let mut chunks = Vec::new();

    if data.is_empty() {
        return (
            FileRecipe {
                file_id: file_id.to_string(),
                original_size: 0,
                chunk_hashes,
            },
            chunks,
        );
    }

    let chunker = FastCDC::new(data, CDC_MIN_SIZE, CDC_AVG_SIZE, CDC_MAX_SIZE);

    for chunk_entry in chunker {
        let chunk_data = &data[chunk_entry.offset..chunk_entry.offset + chunk_entry.length];
        let hash: [u8; 32] = blake3::hash(chunk_data).into();
        
        chunk_hashes.push(hash);
        chunks.push(Chunk {
            hash,
            data: chunk_data.to_vec(),
        });
    }

    (
        FileRecipe {
            file_id: file_id.to_string(),
            original_size: data.len() as u64,
            chunk_hashes,
        },
        chunks,
    )
}

impl ChunkStore {
    /// Get a chunk by hash.
    pub fn get(&self, hash: &[u8; 32]) -> Option<&[u8]> {
        self.chunks.get(hash).map(|v| v.as_slice())
    }

    /// Iterate over all unique chunks in insertion order.
    pub fn iter_ordered(&self) -> impl Iterator<Item = (&[u8; 32], &[u8])> {
        self.order.iter().map(move |h| (h, self.chunks[h].as_slice()))
    }

    /// Total number of unique chunks stored.
    pub fn unique_chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Deduplication ratio (e.g., 3.0 means 3x compression from dedup alone).
    pub fn dedup_ratio(&self) -> f64 {
        if self.total_unique_bytes == 0 {
            return 1.0;
        }
        self.total_input_bytes as f64 / self.total_unique_bytes as f64
    }

    /// Get chunk data by hash (returns None if chunk was already seen = dedup hit).
    /// This is used by StreamBetaWriter to only write truly new chunks.
    pub fn get_chunk_data(&self, hash: &[u8; 32]) -> Option<&[u8]> {
        self.chunks.get(hash).map(|v| v.as_slice())
    }

    /// Get the global insertion index of a chunk (position in `order` vec). O(1) via HashMap.
    pub fn chunk_global_index(&self, hash: &[u8; 32]) -> Option<usize> {
        self.index_map.get(hash).copied()
    }
}

/// Reconstruct a file from its recipe and a chunk lookup function.
pub fn reconstruct_file<F>(recipe: &FileRecipe, chunk_getter: F) -> Result<Vec<u8>, String>
where
    F: Fn(&[u8; 32]) -> Option<Vec<u8>>,
{
    let mut result = Vec::with_capacity(recipe.original_size as usize);

    for hash in &recipe.chunk_hashes {
        let chunk_data = chunk_getter(hash)
            .ok_or_else(|| format!("Missing chunk: {}", hash_to_hex(hash)))?;
        result.extend_from_slice(&chunk_data);
    }

    if result.len() != recipe.original_size as usize {
        return Err(format!(
            "Size mismatch: expected {}, got {}",
            recipe.original_size,
            result.len()
        ));
    }

    Ok(result)
}

/// Convert a 32-byte hash to a hex string (for serialization).
/// Uses a lookup table instead of format! to avoid 32 intermediate allocations per call.
pub fn hash_to_hex(hash: &[u8; 32]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for &b in hash {
        s.push(HEX[(b >> 4) as usize] as char);
        s.push(HEX[(b & 0xf) as usize] as char);
    }
    s
}

/// Parse a hex string back to a 32-byte hash.
pub fn hex_to_hash(hex: &str) -> Result<[u8; 32], String> {
    if hex.len() != 64 {
        return Err(format!("Invalid hash length: {}", hex.len()));
    }
    let mut hash = [0u8; 32];
    for i in 0..32 {
        hash[i] = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16)
            .map_err(|e| format!("Invalid hex: {}", e))?;
    }
    Ok(hash)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_dedup() {
        let mut store = ChunkStore::new();

        // Two identical files should produce 100% dedup
        let data = vec![0x42u8; 16384]; // 16 KB of identical bytes
        let r1 = store.ingest("file1.bin", &data);
        let r2 = store.ingest("file2.bin", &data);

        assert_eq!(r1.chunk_hashes, r2.chunk_hashes);
        assert!(store.dedup_ratio() >= 1.9); // Should be ~2.0x
    }

    #[test]
    fn partial_dedup() {
        let mut store = ChunkStore::new();

        // 256 KB of pseudo-random varied data — large enough for CDC to produce multiple chunks
        let data1: Vec<u8> = (0..262144u32).map(|i| ((i.wrapping_mul(2654435761)) % 256) as u8).collect();
        let mut data2 = data1.clone();
        // Modify the last 64KB — CDC should still share the first ~192KB of chunks
        for item in data2[196608..262144].iter_mut() {
            *item = item.wrapping_add(1);
        }

        let r1 = store.ingest("file1.bin", &data1);
        let r2 = store.ingest("file2.bin", &data2);

        // The files should produce multiple chunks, sharing most of them
        assert!(r1.chunk_hashes.len() > 1, "Need multiple chunks, got {}", r1.chunk_hashes.len());
        let common: usize = r1
            .chunk_hashes
            .iter()
            .filter(|h| r2.chunk_hashes.contains(h))
            .count();
        assert!(common > 0, "Should share at least some chunks (shared {} of {})", common, r1.chunk_hashes.len());
        assert!(store.dedup_ratio() > 1.0, "Should have some dedup");
    }

    #[test]
    fn reconstruct_roundtrip() {
        let mut store = ChunkStore::new();
        let data = b"Hello, NRA BETA! This is a test of content-defined deduplication. \
                      We need enough data to trigger at least one chunk boundary.";
        let data_repeated = data.repeat(100); // ~6.7 KB

        let recipe = store.ingest("test.txt", &data_repeated);

        let reconstructed = reconstruct_file(&recipe, |hash| {
            store.get(hash).map(|s| s.to_vec())
        })
        .unwrap();

        assert_eq!(reconstructed, data_repeated);
    }

    #[test]
    fn hex_roundtrip() {
        let hash: [u8; 32] = blake3::hash(b"test data").into();
        let hex = hash_to_hex(&hash);
        let parsed = hex_to_hash(&hex).unwrap();
        assert_eq!(hash, parsed);
    }
}
