//! NRA BETA Reader: Reconstructs files from deduplicated solid-compressed chunks.

use crate::checksum::calc_crc32;
use crate::codec;
use crate::format::{HEADER_SIZE, NraHeader};
use crate::manifest::BetaManifest;
use std::collections::HashMap;
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;
use memmap2::Mmap;
use crate::crypto;
use crate::metrics::{Metrics, MetricsSnapshot};

pub struct BetaReader {
    mmap: Mmap,
    #[allow(dead_code)]
    header: NraHeader,
    manifest: BetaManifest,
    /// Fast lookup: chunk hash (hex) → index into chunk_table
    chunk_index: HashMap<String, usize>,
    /// Bounded cache of decompressed blocks: block_offset → decompressed data.
    /// Limited to MAX_CACHED_BLOCKS entries to prevent OOM on large archives.
    block_cache: HashMap<u64, Vec<u8>>,
    /// Insertion order for LRU-style eviction
    block_cache_order: Vec<u64>,
    decryption_key: Option<[u8; 32]>,
    metrics: Metrics,
    dictionary: Option<Vec<u8>>,
    /// Cached file IDs for zero-alloc access
    file_id_cache: Vec<String>,
}

/// Maximum number of decompressed blocks to keep in cache.
/// At 4 MB per block, 64 entries ≈ 256 MB max cache footprint.
const MAX_CACHED_BLOCKS: usize = 64;

impl BetaReader {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path)?;

        let mut header_buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_buf)?;
        let header = NraHeader::from_bytes(&header_buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        let mut manifest_buf = vec![0u8; header.manifest_size as usize];
        file.seek(SeekFrom::Start(header.manifest_offset))?;
        file.read_exact(&mut manifest_buf)?;

        let manifest = BetaManifest::deserialize(&manifest_buf)?;

        let chunk_index: HashMap<String, usize> = manifest
            .chunk_table
            .iter()
            .enumerate()
            .map(|(i, c)| (c.hash.clone(), i))
            .collect();

        let mmap = unsafe { Mmap::map(&file)? };

        let dictionary = if let Some(dict_b64) = &manifest.dictionary {
            use base64::{Engine as _, engine::general_purpose::STANDARD};
            Some(STANDARD.decode(dict_b64).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("Failed to decode dictionary: {}", e))
            })?)
        } else {
            None
        };

        let file_id_cache: Vec<String> = manifest.files.iter().map(|f| f.id.clone()).collect();

        Ok(Self {
            mmap,
            header,
            manifest,
            chunk_index,
            block_cache: HashMap::new(),
            block_cache_order: Vec::new(),
            decryption_key: crypto::key_from_env()?,
            metrics: Metrics::new(),
            dictionary,
            file_id_cache,
        })
    }

    pub fn set_decryption_key(&mut self, key: [u8; 32]) {
        self.decryption_key = Some(key);
    }

    pub fn metrics_snapshot(&self) -> MetricsSnapshot {
        self.metrics.snapshot()
    }

    pub fn manifest(&self) -> &BetaManifest {
        &self.manifest
    }

    /// Returns cached list of file IDs (zero-alloc on repeated calls).
    pub fn file_ids(&self) -> &[String] {
        &self.file_id_cache
    }

    pub fn file_size(&self, id: &str) -> Option<u64> {
        self.manifest.files.iter().find(|f| f.id == id).map(|f| f.original_size)
    }

    /// Read and reconstruct a file from its chunk recipe.
    pub fn read_file(&mut self, file_id: &str) -> io::Result<Vec<u8>> {
        let file_record = self
            .manifest
            .files
            .iter()
            .find(|f| f.id == file_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "File not found in BETA manifest"))?;

        let chunk_hashes = file_record.chunks.clone();
        let expected_size = file_record.original_size as usize;

        let mut result = Vec::with_capacity(expected_size);

        for hash_hex in &chunk_hashes {
            let chunk_data = self.read_chunk(hash_hex)?;
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

    /// Read a single chunk by its hash, using block cache for solid blocks.
    fn read_chunk(&mut self, hash_hex: &str) -> io::Result<Vec<u8>> {
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

        // Check if this block is already decompressed in cache
        if !self.block_cache.contains_key(&block_offset) {
            self.metrics.record_cache_miss();
            
            // Fetch the whole block from memory map
            let end_offset = block_offset as usize + compressed_size;
            if end_offset > self.mmap.len() {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Block read past end of file",
                ));
            }
            let buf = &self.mmap[block_offset as usize..end_offset];

            // Verify CRC32
            let computed_crc = calc_crc32(buf);
            if computed_crc != expected_crc {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "CRC32 mismatch for block at offset {}: expected 0x{:08X}, got 0x{:08X}",
                        block_offset, expected_crc, computed_crc
                    ),
                ));
            }

            // Decrypt if necessary, otherwise use the raw mmap slice directly
            let decompression_input: std::borrow::Cow<[u8]> = if self.manifest.summary.encrypted {
                let key = self.decryption_key.ok_or_else(|| {
                    io::Error::new(io::ErrorKind::PermissionDenied, "Archive is encrypted, but no key provided")
                })?;
                std::borrow::Cow::Owned(crypto::decrypt_block(buf, &key)?)
            } else {
                std::borrow::Cow::Borrowed(buf)
            };

            let decompressed = codec::decompress(
                &decompression_input,
                self.dictionary.as_deref()
            )?;

            // Evict oldest block if cache is full
            if self.block_cache.len() >= MAX_CACHED_BLOCKS
                && let Some(oldest_offset) = self.block_cache_order.first().copied() {
                    self.block_cache.remove(&oldest_offset);
                    self.block_cache_order.remove(0);
                }

            self.block_cache_order.push(block_offset);
            self.block_cache.insert(block_offset, decompressed);
        } else {
            self.metrics.record_cache_hit();
        }

        // Slice out this chunk from the decompressed block
        let block_data = &self.block_cache[&block_offset];
        let end = inner_offset + original_size;

        if end > block_data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Chunk inner offset out of bounds: {}..{} > {}", inner_offset, end, block_data.len()),
            ));
        }

        self.metrics.record_chunk_read(original_size as u64);

        Ok(block_data[inner_offset..end].to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_writer::BetaWriter;

    #[test]
    fn beta_roundtrip() {
        let dir = std::env::temp_dir().join("nra_beta_roundtrip.nra");

        let mut writer = BetaWriter::new();
        writer.set_name("Beta Test Archive");
        writer.add_file("hello.txt", b"Hello, NRA BETA! This is a dedup test.");
        writer.add_file("data.bin", &[0xDE, 0xAD, 0xBE, 0xEF].repeat(2048));
        writer.save(&dir).unwrap();

        let mut reader = BetaReader::open(&dir).unwrap();
        assert_eq!(reader.manifest().summary.total_files, 2);

        let hello = reader.read_file("hello.txt").unwrap();
        assert_eq!(&hello, b"Hello, NRA BETA! This is a dedup test.");

        let data = reader.read_file("data.bin").unwrap();
        assert_eq!(data, [0xDE, 0xAD, 0xBE, 0xEF].repeat(2048));

        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn beta_dedup_identical_files() {
        let dir = std::env::temp_dir().join("nra_beta_dedup.nra");

        let data = vec![0x42u8; 65536]; // 64 KB

        let mut writer = BetaWriter::new();
        for i in 0..10 {
            writer.add_file(&format!("copy_{}.bin", i), &data);
        }
        writer.print_stats();
        writer.save(&dir).unwrap();

        let archive_size = std::fs::metadata(&dir).unwrap().len();
        let raw_size = 10 * 65536;
        eprintln!("Archive: {} bytes, Raw: {} bytes, Ratio: {:.1}x",
            archive_size, raw_size, raw_size as f64 / archive_size as f64);
        assert!(archive_size < raw_size / 5, "Dedup should reduce size significantly");

        let mut reader = BetaReader::open(&dir).unwrap();
        for i in 0..10 {
            let result = reader.read_file(&format!("copy_{}.bin", i)).unwrap();
            assert_eq!(result, data);
        }

        std::fs::remove_file(&dir).ok();
    }
    
    #[test]
    fn crypto_roundtrip_beta() {
        let dir = std::env::temp_dir().join("nra_beta_crypto.nra");
        let key = [0x42u8; 32];

        let mut writer = BetaWriter::new();
        writer.set_encryption_key(key);
        writer.add_file("secret.txt", b"Top secret data");
        writer.save(&dir).unwrap();

        let mut reader = BetaReader::open(&dir).unwrap();
        reader.set_decryption_key(key);
        let data = reader.read_file("secret.txt").unwrap();
        assert_eq!(&data, b"Top secret data");
        
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn wrong_key_fails_beta() {
        let dir = std::env::temp_dir().join("nra_beta_wrong_key.nra");
        let key = [0x42u8; 32];
        let wrong_key = [0x00u8; 32];

        let mut writer = BetaWriter::new();
        writer.set_encryption_key(key);
        writer.add_file("secret.txt", b"Top secret data");
        writer.save(&dir).unwrap();

        let mut reader = BetaReader::open(&dir).unwrap();
        reader.set_decryption_key(wrong_key);
        let result = reader.read_file("secret.txt");
        assert!(result.is_err());
        
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn pack_beta_lz4_roundtrip() {
        let dir = std::env::temp_dir().join("nra_beta_lz4.nra");

        let mut writer = BetaWriter::new();
        writer.set_codec(crate::codec::Codec::Lz4);
        writer.add_file("fast.txt", b"LZ4 goes brrrrrr");
        writer.save(&dir).unwrap();

        let mut reader = BetaReader::open(&dir).unwrap();
        let data = reader.read_file("fast.txt").unwrap();
        assert_eq!(&data, b"LZ4 goes brrrrrr");
        
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn beta_many_small_files() {
        let dir = std::env::temp_dir().join("nra_beta_small.nra");

        let mut writer = BetaWriter::new();
        // 100 JSON files with common schema — should compress extremely well in solid blocks
        for i in 0..100 {
            let json = format!(
                r#"{{"id":{},"label":"class_{}","score":0.{:04},"metadata":{{"width":224,"height":224,"format":"JPEG"}}}}"#,
                i, i % 10, i * 7
            );
            writer.add_file(&format!("meta_{:04}.json", i), json.as_bytes());
        }
        writer.save(&dir).unwrap();

        let mut reader = BetaReader::open(&dir).unwrap();
        assert_eq!(reader.manifest().summary.total_files, 100);

        // Verify first and last file
        let first = reader.read_file("meta_0000.json").unwrap();
        assert!(String::from_utf8_lossy(&first).contains(r#""id":0"#));

        let last = reader.read_file("meta_0099.json").unwrap();
        assert!(String::from_utf8_lossy(&last).contains(r#""id":99"#));

        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn dictionary_roundtrip() {
        let dir = std::env::temp_dir().join("nra_beta_dictionary.nra");

        let mut writer = BetaWriter::new();
        // Add a bunch of small, highly repetitive JSON files
        for i in 0..200 {
            let json = format!(
                r#"{{"id":{},"label":"class_{}","score":0.{:04},"metadata":{{"width":224,"height":224,"format":"JPEG"}}}}"#,
                i, i % 10, i * 7
            );
            writer.add_file(&format!("meta_{:04}.json", i), json.as_bytes());
        }

        // Train dictionary
        writer.train_dictionary(110 * 1024).unwrap();
        writer.save(&dir).unwrap();

        let mut reader = BetaReader::open(&dir).unwrap();
        assert!(reader.manifest().dictionary.is_some());

        // Verify decompression still works with the dictionary
        let first = reader.read_file("meta_0000.json").unwrap();
        assert!(String::from_utf8_lossy(&first).contains(r#""id":0"#));

        std::fs::remove_file(&dir).ok();
    }
    #[test]
    fn test_read_legacy_json_manifest() {
        // Assume legacy_v4.nra was created by a previous version
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/legacy_v4.nra");
        let mut reader = BetaReader::open(path).expect("Failed to open legacy archive");
        let data = reader.read_file("hello.txt").expect("Failed to read file");
        assert_eq!(&data, b"Legacy compatibility test");
    }
}
