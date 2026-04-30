pub mod checksum;
pub mod compression;
pub mod format;
pub mod manifest;
pub mod reader;
pub mod writer;

// NRA BETA: Content-Defined Deduplication
pub mod dedup;
pub mod beta_writer;
pub mod beta_reader;
pub mod async_reader;
#[cfg(feature = "fuse")]
pub mod fuse;

// Stage 3: Advanced Features
pub mod codec;    // Dual codec: Zstd + LZ4
pub mod crypto;   // AES-256-GCM block encryption
pub mod metrics;  // Lock-free performance counters
pub mod stream_writer;  // Streaming single-pass writer
pub mod delta;    // Delta updates (append without rebuild)

// Stage 4: Revenue-Critical Features
pub mod sampler;        // Elastic Determinism + Mid-Epoch Resumption
pub mod vector_index;   // Embedded ANN for RAG pipelines

pub use format::{FORMAT_VERSION, HEADER_SIZE, MAGIC_BYTES, NraHeader};
pub use manifest::{Compression, EmbeddingSpace, FileRecord, FileVector, Manifest};
pub use manifest::{BetaManifest, BetaChunkRecord, BetaFileRecord, BetaSummary};
pub use reader::NraReader;
pub use writer::{NraWriter, OptimizationMode};
pub use beta_writer::BetaWriter;
pub use beta_reader::BetaReader;
pub use async_reader::{AsyncRandomAccess, AsyncBetaReader};
pub use codec::Codec;
pub use crypto::{encrypt_block, decrypt_block, key_from_env};
pub use metrics::{Metrics, MetricsSnapshot};
