//! Streaming Writer: builds .nra archives in a single pass.
//!
//! Unlike `BetaWriter` which buffers all files in memory, `StreamBetaWriter`
//! writes compressed blocks to the output as soon as they fill up (~4MB).
//! The manifest is written as a footer at the end.
//!
//! Archive layout (streaming):
//! ```text
//! [Header 32B] [Block 0] [Block 1] ... [Block N] [Manifest JSON] [Footer 8B: manifest_offset]
//! ```
//!
//! The header's `manifest_offset` points to the footer, which stores the
//! actual manifest position. This allows single-pass writes.

use crate::checksum::calc_crc32;
use crate::codec::{Codec, compress};
use crate::crypto;
use crate::dedup::{hash_to_hex, ChunkStore, FileRecipe};
use crate::format::{HEADER_SIZE, NraHeader};
use crate::manifest::{BetaChunkRecord, BetaFileRecord, BetaManifest};
use std::io::{self, BufWriter, Seek, SeekFrom, Write};

/// Target size for solid blocks before compression.
const SOLID_BLOCK_SIZE: usize = 4 * 1024 * 1024; // 4 MB

/// A streaming NRA BETA writer that flushes blocks to disk incrementally.
pub struct StreamBetaWriter<W: Write + Seek> {
    writer: BufWriter<W>,
    store: ChunkStore,
    recipes: Vec<FileRecipe>,
    codec: Codec,
    encryption_key: Option<[u8; 32]>,

    // Current solid block accumulator
    current_block_raw: Vec<u8>,
    current_block_chunks: Vec<PendingChunk>,

    // Written blocks metadata
    written_blocks: Vec<WrittenBlock>,
    data_bytes_written: u64,
}

struct PendingChunk {
    global_index: usize,
    inner_offset: u64,
    original_size: u64,
}

struct WrittenBlock {
    offset: u64,
    compressed_size: u64,
    crc32: u32,
    chunks: Vec<PendingChunk>,
}

impl<W: Write + Seek> StreamBetaWriter<W> {
    pub fn new(inner: W) -> io::Result<Self> {
        let mut writer = BufWriter::new(inner);

        // Write a placeholder header (will be overwritten at finalize)
        let placeholder_header = [0u8; HEADER_SIZE];
        writer.write_all(&placeholder_header)?;

        Ok(Self {
            writer,
            store: ChunkStore::new(),
            recipes: Vec::new(),
            codec: Codec::Zstd,
            encryption_key: None,
            current_block_raw: Vec::new(),
            current_block_chunks: Vec::new(),
            written_blocks: Vec::new(),
            data_bytes_written: 0,
        })
    }

    /// Set compression codec (Zstd or LZ4). Default: Zstd.
    pub fn set_codec(&mut self, codec: Codec) {
        self.codec = codec;
    }

    /// Enable AES-256-GCM encryption for all blocks.
    pub fn set_encryption_key(&mut self, key: [u8; 32]) {
        self.encryption_key = Some(key);
    }

    /// Add a file to the archive. Blocks are flushed to disk when they reach ~4MB.
    pub fn add_file(&mut self, id: &str, data: &[u8]) -> io::Result<()> {
        let recipe = self.store.ingest(id, data);

        // For each chunk in the recipe, if it's a new unique chunk, add to current block
        for hash in &recipe.chunk_hashes {
            if let Some(chunk_data) = self.store.get_chunk_data(hash) {
                let global_index = self.store.chunk_global_index(hash).unwrap_or(0);

                let inner_offset = self.current_block_raw.len() as u64;
                let original_size = chunk_data.len() as u64;

                self.current_block_raw.extend_from_slice(chunk_data);
                self.current_block_chunks.push(PendingChunk {
                    global_index,
                    inner_offset,
                    original_size,
                });

                // Flush block if it's big enough
                if self.current_block_raw.len() >= SOLID_BLOCK_SIZE {
                    self.flush_current_block()?;
                }
            }
            // If chunk already seen (dedup hit), skip — it's in a previous block
        }

        self.recipes.push(recipe);
        Ok(())
    }

    /// Flush the current solid block to disk.
    fn flush_current_block(&mut self) -> io::Result<()> {
        if self.current_block_raw.is_empty() {
            return Ok(());
        }

        let mut compressed = compress(&self.current_block_raw, self.codec, 3, None)?;

        // Encrypt if key is set
        if let Some(ref key) = self.encryption_key {
            compressed = crypto::encrypt_block(&compressed, key)?;
        }

        let crc32 = calc_crc32(&compressed);
        let offset = HEADER_SIZE as u64 + self.data_bytes_written;

        self.writer.write_all(&compressed)?;
        self.data_bytes_written += compressed.len() as u64;

        let chunks = std::mem::take(&mut self.current_block_chunks);
        self.written_blocks.push(WrittenBlock {
            offset,
            compressed_size: compressed.len() as u64,
            crc32,
            chunks,
        });

        self.current_block_raw.clear();
        Ok(())
    }

    /// Finalize the archive: flush remaining data, write manifest footer, fix header.
    pub fn finalize(mut self) -> io::Result<()> {
        // Flush any remaining data in the current block
        self.flush_current_block()?;

        // Build manifest
        let mut manifest = BetaManifest::new();
        manifest.summary.name = "NRA BETA Streaming Archive".to_string();
        manifest.summary.total_files = self.recipes.len() as u64;
        manifest.summary.total_chunks = self.store.unique_chunk_count() as u64;
        manifest.summary.dedup_ratio = self.store.dedup_ratio();
        manifest.summary.total_original_bytes = self.store.total_input_bytes;
        manifest.summary.total_stored_bytes = self.data_bytes_written;

        // Build chunk_table from written blocks
        // We need a lookup: global_index → (block_idx in written_blocks)
        let mut chunk_lookup: std::collections::HashMap<usize, (usize, u64, u64)> =
            std::collections::HashMap::new();
        for (bi, block) in self.written_blocks.iter().enumerate() {
            for chunk in &block.chunks {
                chunk_lookup.insert(chunk.global_index, (bi, chunk.inner_offset, chunk.original_size));
            }
        }

        // Emit chunk_table in order
        for (hash, _data) in self.store.iter_ordered() {
            let idx = self.store.chunk_global_index(hash).unwrap_or(0);
            let hex = hash_to_hex(hash);
            if let Some(&(bi, inner_offset, original_size)) = chunk_lookup.get(&idx) {
                let block = &self.written_blocks[bi];
                manifest.chunk_table.push(BetaChunkRecord {
                    hash: hex,
                    offset: block.offset,
                    compressed_size: block.compressed_size,
                    original_size,
                    inner_offset,
                    crc32: block.crc32,
                });
            }
        }

        // Build file records
        for recipe in &self.recipes {
            manifest.files.push(BetaFileRecord {
                id: recipe.file_id.clone(),
                original_size: recipe.original_size,
                chunks: recipe.chunk_hashes.iter().map(hash_to_hex).collect(),
            });
        }

        // Write manifest as footer
        let manifest_bytes = manifest.serialize()?;
        let manifest_offset = HEADER_SIZE as u64 + self.data_bytes_written;
        let manifest_size = manifest_bytes.len() as u64;
        self.writer.write_all(&manifest_bytes)?;

        // Rewrite header with correct offsets
        let header = NraHeader::new(manifest_size, HEADER_SIZE as u64);
        // Override manifest_offset to point to footer
        let mut header_bytes = header.to_bytes();
        // Patch manifest_offset (bytes 8..16)
        header_bytes[8..16].copy_from_slice(&manifest_offset.to_le_bytes());
        // Patch manifest_size (bytes 16..24)
        header_bytes[16..24].copy_from_slice(&manifest_size.to_le_bytes());
        // Patch data_section_offset (bytes 24..32)
        let data_start = HEADER_SIZE as u64;
        header_bytes[24..32].copy_from_slice(&data_start.to_le_bytes());

        self.writer.seek(SeekFrom::Start(0))?;
        self.writer.write_all(&header_bytes)?;
        self.writer.flush()?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_reader::BetaReader;
    use std::io::Cursor;

    #[test]
    fn stream_write_roundtrip() {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut sw = StreamBetaWriter::new(&mut buf).unwrap();
            sw.add_file("hello.txt", b"Hello from streaming writer!").unwrap();
            sw.add_file("data.bin", &[0xAB; 8192]).unwrap();
            sw.finalize().unwrap();
        }

        // Write to temp file for BetaReader
        let path = std::env::temp_dir().join("nra_stream_test.nra");
        std::fs::write(&path, buf.into_inner()).unwrap();

        let mut reader = BetaReader::open(&path).unwrap();
        let hello = reader.read_file("hello.txt").unwrap();
        assert_eq!(&hello, b"Hello from streaming writer!");

        let data = reader.read_file("data.bin").unwrap();
        assert_eq!(data, vec![0xAB; 8192]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn stream_write_lz4() {
        let mut buf = Cursor::new(Vec::new());
        {
            let mut sw = StreamBetaWriter::new(&mut buf).unwrap();
            sw.set_codec(Codec::Lz4);
            sw.add_file("fast.bin", &[0x42; 16384]).unwrap();
            sw.finalize().unwrap();
        }

        let path = std::env::temp_dir().join("nra_stream_lz4.nra");
        std::fs::write(&path, buf.into_inner()).unwrap();

        let mut reader = BetaReader::open(&path).unwrap();
        let data = reader.read_file("fast.bin").unwrap();
        assert_eq!(data, vec![0x42; 16384]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn stream_write_dedup() {
        let shared_data = vec![0xFFu8; 65536]; // 64KB identical

        let mut buf = Cursor::new(Vec::new());
        {
            let mut sw = StreamBetaWriter::new(&mut buf).unwrap();
            for i in 0..5 {
                sw.add_file(&format!("dup_{}.bin", i), &shared_data).unwrap();
            }
            sw.finalize().unwrap();
        }

        let raw_size = 5 * 65536;
        let archive_size = buf.get_ref().len();
        eprintln!("Stream dedup: {} bytes archived from {} raw (ratio: {:.1}x)",
            archive_size, raw_size, raw_size as f64 / archive_size as f64);
        assert!(archive_size < raw_size / 3, "Dedup should reduce size significantly");
    }
}
