//! NRA BETA Writer: CDC Dedup + Solid Block Compression.
//!
//! 1. Split each file into CDC chunks (FastCDC)
//! 2. Deduplicate identical chunks globally (BLAKE3)
//! 3. Group unique chunks into solid blocks (~4 MB each)
//! 4. Compress each block with Zstd (massive compression from solid context)
//! 5. Each chunk is addressable by (block_offset, inner_offset, size)

use crate::checksum::calc_crc32;
use crate::dedup::{hash_to_hex, ChunkStore, FileRecipe};
use crate::format::{HEADER_SIZE, NraHeader};
use crate::manifest::{BetaChunkRecord, BetaFileRecord, BetaManifest};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

/// Target size for solid blocks before compression.
const SOLID_BLOCK_SIZE: usize = 4 * 1024 * 1024; // 4 MB

use crate::crypto;
use crate::codec::{self, Codec};
use crate::metrics::Metrics;

pub struct BetaWriter {
    store: ChunkStore,
    recipes: Vec<FileRecipe>,
    name: String,
    encryption_key: Option<[u8; 32]>,
    codec: Codec,
    zstd_level: i32,
    pub metrics: Metrics,
    dictionary: Option<Vec<u8>>,
}

impl Default for BetaWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl BetaWriter {
    pub fn new() -> Self {
        Self {
            store: ChunkStore::new(),
            recipes: Vec::new(),
            name: "NRA BETA Dataset".to_string(),
            encryption_key: None,
            codec: Codec::Zstd,
            zstd_level: 3,
            metrics: Metrics::new(),
            dictionary: None,
        }
    }

    pub fn set_encryption_key(&mut self, key: [u8; 32]) {
        self.encryption_key = Some(key);
    }

    pub fn set_codec(&mut self, codec: Codec) {
        self.codec = codec;
    }

    pub fn set_zstd_level(&mut self, level: i32) {
        self.zstd_level = level;
    }

    pub fn train_dictionary(&mut self, max_dict_size: usize) -> Result<(), std::io::Error> {
        let mut samples = Vec::new();
        let mut total_size = 0;

        // Collect up to 1 MB of unique chunks as training data
        for (_, data) in self.store.iter_ordered() {
            samples.push(data.as_ref());
            total_size += data.len();
            if total_size > 1024 * 1024 {
                break;
            }
        }

        if samples.is_empty() {
            return Ok(());
        }

        let dict = zstd::dict::from_samples(&samples, max_dict_size)
            .map_err(std::io::Error::other)?;
            
        self.dictionary = Some(dict);
        Ok(())
    }

    pub fn set_name(&mut self, name: &str) {
        self.name = name.to_string();
    }

    pub fn add_file(&mut self, id: &str, data: &[u8]) {
        let recipe = self.store.ingest(id, data);
        self.recipes.push(recipe);
    }

    pub fn add_prechunked(&mut self, recipe: FileRecipe, chunks: Vec<crate::dedup::Chunk>) {
        self.store.ingest_chunks(&recipe, chunks);
        self.recipes.push(recipe);
    }

    pub fn print_stats(&self) {
        let ratio = self.store.dedup_ratio();
        let unique = self.store.unique_chunk_count();
        let total_chunks: usize = self.recipes.iter().map(|r| r.chunk_hashes.len()).sum();
        eprintln!("📊 NRA BETA Dedup Stats:");
        eprintln!("   Files:         {}", self.recipes.len());
        eprintln!("   Total chunks:  {}", total_chunks);
        eprintln!("   Unique chunks: {} ({:.1}% dedup)", unique,
            (1.0 - unique as f64 / total_chunks.max(1) as f64) * 100.0);
        eprintln!("   Input bytes:   {:.2} MB", self.store.total_input_bytes as f64 / 1e6);
        eprintln!("   Unique bytes:  {:.2} MB", self.store.total_unique_bytes as f64 / 1e6);
        eprintln!("   Dedup ratio:   {:.2}x", ratio);
    }

    #[must_use]
    pub fn save<P: AsRef<Path>>(self, path: P) -> io::Result<()> {
        let mut manifest = BetaManifest::new();
        manifest.summary.name = self.name.clone();
        manifest.summary.total_files = self.recipes.len() as u64;
        manifest.summary.total_chunks = self.store.unique_chunk_count() as u64;
        manifest.summary.dedup_ratio = self.store.dedup_ratio();
        manifest.summary.total_original_bytes = self.store.total_input_bytes;
        manifest.summary.encrypted = self.encryption_key.is_some();

        // === Phase 1: Group chunks into solid blocks ===
        // Collect all unique chunks in order (zero-copy references to ChunkStore)
        let all_chunks: Vec<(&[u8; 32], &[u8])> = self
            .store
            .iter_ordered()
            .collect();

        // Group into blocks of ~SOLID_BLOCK_SIZE
        struct BlockPlan {
            chunk_indices: Vec<usize>, // indices into all_chunks
        }

        let mut block_plans: Vec<BlockPlan> = Vec::new();
        let mut current = BlockPlan { chunk_indices: Vec::new() };
        let mut current_size: usize = 0;

        for (i, (_, data)) in all_chunks.iter().enumerate() {
            current_size += data.len();
            current.chunk_indices.push(i);
            if current_size >= SOLID_BLOCK_SIZE {
                block_plans.push(current);
                current = BlockPlan { chunk_indices: Vec::new() };
                current_size = 0;
            }
        }
        if !current.chunk_indices.is_empty() {
            block_plans.push(current);
        }

        eprintln!("   Solid blocks:  {}", block_plans.len());

        // === Phase 2: Compress each block, record inner offsets ===
        // For each block, concatenate raw chunks, compress, and track inner offsets.
        struct CompressedBlock {
            data: Vec<u8>,
            crc32: u32,
            // (chunk_global_idx, inner_offset_in_block, raw_chunk_size)
            entries: Vec<(usize, u64, u64)>,
        }

        use rayon::prelude::*;

        let codec = self.codec;
        let zstd_level = self.zstd_level;
        let encryption_key = self.encryption_key;
        let dictionary = self.dictionary.as_deref();

        let compressed_blocks: Vec<CompressedBlock> = block_plans
            .par_iter()
            .map(|plan| {
                let mut raw_buffer = Vec::new();
                let mut entries = Vec::new();

                for &ci in &plan.chunk_indices {
                    let inner_offset = raw_buffer.len() as u64;
                    let chunk_data = &all_chunks[ci].1;
                    raw_buffer.extend_from_slice(chunk_data);
                    entries.push((ci, inner_offset, chunk_data.len() as u64));
                }

                let compressed = codec::compress(&raw_buffer, codec, zstd_level, dictionary).expect("Compression failed");
                
                let data = if let Some(key) = encryption_key {
                    crypto::encrypt_block(&compressed, &key).expect("Encryption failed")
                } else {
                    compressed
                };

                let crc32 = calc_crc32(&data);

                CompressedBlock {
                    data,
                    crc32,
                    entries,
                }
            })
            .collect();

        let total_stored: u64 = compressed_blocks.iter().map(|b| b.data.len() as u64).sum();

        manifest.summary.total_stored_bytes = total_stored;
        eprintln!("   Stored bytes:  {:.2} MB (after dedup + solid Zstd)", total_stored as f64 / 1e6);

        // === Phase 3: Build file records ===
        for recipe in &self.recipes {
            manifest.files.push(BetaFileRecord {
                id: recipe.file_id.clone(),
                original_size: recipe.original_size,
                chunks: recipe.chunk_hashes.iter().map(hash_to_hex).collect(),
            });
        }

        if let Some(dict) = &self.dictionary {
            use base64::{Engine as _, engine::general_purpose::STANDARD};
            manifest.dictionary = Some(STANDARD.encode(dict));
        }

        // === Phase 4: Build chunk_table ===
        // Each chunk stores: block_offset (placeholder 0), inner_offset, compressed_size (of whole block), original_size (of this chunk)
        // We need to emit chunks in the same global order as all_chunks.
        // Build a lookup: global_chunk_idx → (block_idx, inner_offset)
        let mut chunk_meta: Vec<(usize, u64)> = vec![(0, 0); all_chunks.len()]; // (block_idx, inner_offset)
        for (block_idx, block) in compressed_blocks.iter().enumerate() {
            for &(ci, inner_offset, _) in &block.entries {
                chunk_meta[ci] = (block_idx, inner_offset);
            }
        }

        for (ci, (hash, raw_data)) in all_chunks.iter().enumerate() {
            let (block_idx, inner_offset) = chunk_meta[ci];
            let block = &compressed_blocks[block_idx];
            manifest.chunk_table.push(BetaChunkRecord {
                hash: hash_to_hex(hash),
                offset: 0, // placeholder — set during stabilization
                compressed_size: block.data.len() as u64,
                original_size: raw_data.len() as u64,
                crc32: block.crc32,
                inner_offset,
            });
        }

        // === Phase 5: Offset stabilization ===
        // All chunks in the same block must share the same offset value.
        // We track which block each chunk belongs to.
        let block_sizes: Vec<u64> = compressed_blocks.iter().map(|b| b.data.len() as u64).collect();
        let chunk_to_block: Vec<usize> = chunk_meta.iter().map(|(bi, _)| *bi).collect();

        let manifest_bytes = stabilize_beta_offsets(&mut manifest, &block_sizes, &chunk_to_block)?;
        let manifest_size = manifest_bytes.len() as u64;
        let data_section_offset = HEADER_SIZE as u64 + manifest_size;

        let header = NraHeader::new(manifest_size, data_section_offset);

        // === Phase 6: Write ===
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        writer.write_all(&header.to_bytes())?;
        writer.write_all(&manifest_bytes)?;
        for block in &compressed_blocks {
            writer.write_all(&block.data)?;
            self.metrics.record_bytes_written(block.data.len());
        }
        writer.flush()?;
        
        eprintln!("   Metrics: {:?}", self.metrics.snapshot());
        
        Ok(())
    }
}

fn stabilize_beta_offsets(
    manifest: &mut BetaManifest,
    block_sizes: &[u64],
    chunk_to_block: &[usize],
) -> io::Result<Vec<u8>> {
    const MAX_ITERATIONS: usize = 8;

    for _ in 0..MAX_ITERATIONS {
        let trial = manifest.serialize()?;
        let manifest_size = trial.len() as u64;
        let data_start = HEADER_SIZE as u64 + manifest_size;

        // Compute block offsets
        let mut block_offsets = vec![0u64; block_sizes.len()];
        let mut offset = data_start;
        for (i, &size) in block_sizes.iter().enumerate() {
            block_offsets[i] = offset;
            offset += size;
        }

        let mut changed = false;
        for (ci, record) in manifest.chunk_table.iter_mut().enumerate() {
            let block_offset = block_offsets[chunk_to_block[ci]];
            if record.offset != block_offset {
                record.offset = block_offset;
                changed = true;
            }
        }

        if !changed {
            return Ok(trial);
        }
    }

    Err(io::Error::other(
        "BETA manifest offset stabilization did not converge",
    ))
}
