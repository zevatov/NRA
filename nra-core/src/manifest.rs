use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Compression {
    None = 0,
    Zstd = 1,
    Lz4 = 2,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct EmbeddingSpace {
    pub id: String,
    pub model_uri: String,
    pub dimension: u16,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FileVector {
    pub space_id: String,
    pub data: Vec<u8>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct FileRecord {
    pub id: String,
    pub offset: u64,
    #[serde(default)]
    pub inner_offset: u64, // Used for Chunked Solid Compression. 0 for Speed mode.
    pub compressed_size: u64,
    pub original_size: u64,
    pub crc32: u32,
    pub compression: Compression,
    pub vectors: Vec<FileVector>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct SummaryHeader {
    pub name: String,
    pub total_files: u64,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Manifest {
    pub version: String,
    pub summary: SummaryHeader,
    pub spaces: Vec<EmbeddingSpace>,
    pub files: Vec<FileRecord>,
}

impl Default for Manifest {
    fn default() -> Self {
        Self::new()
    }
}

/// Macro implementing MessagePack serialization with NRA magic bytes and JSON fallback.
macro_rules! impl_manifest_serde {
    ($type:ty) => {
        impl $type {
            pub fn serialize(&self) -> Result<Vec<u8>, std::io::Error> {
                let mut buf = Vec::new();
                buf.extend_from_slice(b"NRA\x01"); // magic bytes: NRA + format version 1
                rmp_serde::encode::write(&mut buf, self)
                    .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
                Ok(buf)
            }

            pub fn deserialize(data: &[u8]) -> Result<Self, std::io::Error> {
                if data.starts_with(b"NRA\x01") {
                    // Binary manifest (v4.5+)
                    rmp_serde::from_slice(&data[4..])
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                } else if data.first() == Some(&b'{') {
                    // Legacy JSON manifest (v1-v4.0)
                    serde_json::from_slice(data)
                        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
                } else {
                    Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "Unknown manifest format"))
                }
            }
        }
    }
}

impl Manifest {
    pub fn new() -> Self {
        Self {
            version: "2.0.0".to_string(),
            summary: SummaryHeader {
                name: "NRA Dataset".to_string(),
                total_files: 0,
            },
            spaces: Vec::new(),
            files: Vec::new(),
        }
    }
}

impl_manifest_serde!(Manifest);

// ============================================================
// NRA BETA: Dedup-aware Manifest Structures
// ============================================================

/// A unique chunk stored in the archive's data section.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BetaChunkRecord {
    /// BLAKE3 hash of the chunk (hex-encoded, 64 chars)
    pub hash: String,
    /// Absolute byte offset of the compressed block in the archive
    pub offset: u64,
    /// Size of the compressed block data (shared by all chunks in the same block)
    pub compressed_size: u64,
    /// Size of the original (uncompressed) chunk data
    pub original_size: u64,
    /// Offset of this chunk within the decompressed block
    #[serde(default)]
    pub inner_offset: u64,
    /// CRC32 of the compressed block data (for integrity verification)
    pub crc32: u32,
}

/// A file described as a "recipe" of chunk references.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BetaFileRecord {
    /// File identifier (e.g., "sample_00001.png")
    pub id: String,
    /// Total original (uncompressed) size of the file
    pub original_size: u64,
    /// Ordered list of chunk hashes that compose this file
    pub chunks: Vec<String>,
}

/// BETA Manifest: stores chunk table + file recipes.
#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BetaManifest {
    pub version: String,
    pub summary: BetaSummary,
    pub chunk_table: Vec<BetaChunkRecord>,
    pub files: Vec<BetaFileRecord>,
    #[serde(default)]
    pub dictionary: Option<String>,
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct BetaSummary {
    pub name: String,
    pub total_files: u64,
    pub total_chunks: u64,
    pub dedup_ratio: f64,
    pub total_original_bytes: u64,
    pub total_stored_bytes: u64,
    #[serde(default)]
    pub encrypted: bool,
}

impl Default for BetaManifest {
    fn default() -> Self {
        Self::new()
    }
}

impl BetaManifest {
    pub fn new() -> Self {
        Self {
            version: "3.0.0-beta".to_string(),
            summary: BetaSummary {
                name: "NRA BETA Dataset".to_string(),
                total_files: 0,
                total_chunks: 0,
                dedup_ratio: 1.0,
                total_original_bytes: 0,
                total_stored_bytes: 0,
                encrypted: false,
            },
            chunk_table: Vec::new(),
            files: Vec::new(),
            dictionary: None,
        }
    }
}

impl_manifest_serde!(BetaManifest);

