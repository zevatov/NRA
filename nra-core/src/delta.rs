//! Delta Packer: append new files to an existing .nra archive without full rebuild.
//!
//! Strategy:
//! 1. Read the existing archive's manifest.
//! 2. Build a set of known chunk hashes (already stored).
//! 3. CDC-chunk the new files, skip chunks that already exist.
//! 4. Compress only truly new chunks into new blocks.
//! 5. Append new blocks after the old data.
//! 6. Write a merged manifest (old files + new files, old chunks + new chunks).
//! 7. Overwrite the header to point to the new manifest.
//!
//! This is an O(new_data) operation, not O(total_archive).

use crate::beta_reader::BetaReader;
use crate::checksum::calc_crc32;
use crate::codec;
use crate::dedup::{self, hash_to_hex, ChunkStore};
use crate::format::{HEADER_SIZE, NraHeader};
use crate::manifest::{BetaChunkRecord, BetaFileRecord};
use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{self, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

/// Target size for solid blocks.
const SOLID_BLOCK_SIZE: usize = 4 * 1024 * 1024;

/// Append new files to an existing NRA BETA archive.
///
/// # Arguments
/// * `archive_path` - Path to the existing .nra file (modified in-place).
/// * `new_files` - Slice of (file_id, file_data) pairs to add.
///
/// # Returns
/// Number of new unique chunks added (0 if everything was deduplicated).
pub fn delta_pack(
    archive_path: &Path,
    new_files: &[(&str, &[u8])],
) -> io::Result<usize> {
    // Phase 1: Read existing manifest
    let reader = BetaReader::open(archive_path)?;
    let old_manifest = reader.manifest().clone();

    // Build set of known chunk hashes
    let known_hashes: HashSet<String> = old_manifest
        .chunk_table
        .iter()
        .map(|c| c.hash.clone())
        .collect();

    // Phase 2: Ingest new files, identify truly new chunks
    use rayon::prelude::*;

    // We can't share a single ChunkStore mutably across threads easily,
    // so we will chunk the files in parallel and then insert them into a single ChunkStore.
    let prechunked: Vec<_> = new_files
        .par_iter()
        .map(|&(id, data)| dedup::chunk_data(id, data))
        .collect();

    let mut store = ChunkStore::new();
    let mut recipes = Vec::new();

    // Now ingest the prechunked results sequentially into the store
    // This allows us to find unique chunks without duplicate work.
    for (recipe, chunks) in prechunked {
        store.ingest_chunks(&recipe, chunks);
        recipes.push(recipe);
    }

    // Filter to only chunks not already in the archive
    let mut new_chunks: Vec<([u8; 32], Vec<u8>)> = Vec::new();
    for (hash, data) in store.iter_ordered() {
        let hex = hash_to_hex(hash);
        if !known_hashes.contains(&hex) {
            new_chunks.push((*hash, data.to_vec()));
        }
    }

    let new_chunk_count = new_chunks.len();

    // Phase 3: Group new chunks into solid blocks and compress
    struct CompressedBlock {
        data: Vec<u8>,
        crc32: u32,
        entries: Vec<(String, u64, u64)>, // (hash_hex, inner_offset, original_size)
    }

    let mut new_blocks: Vec<CompressedBlock> = Vec::new();
    let mut raw_buffer = Vec::new();
    let mut entries = Vec::new();

    for (hash, chunk_data) in &new_chunks {
        let inner_offset = raw_buffer.len() as u64;
        raw_buffer.extend_from_slice(chunk_data);
        entries.push((hash_to_hex(hash), inner_offset, chunk_data.len() as u64));

        if raw_buffer.len() >= SOLID_BLOCK_SIZE {
            let compressed = codec::compress(&raw_buffer, codec::Codec::Zstd, 3, None)?;
            let crc32 = calc_crc32(&compressed);
            new_blocks.push(CompressedBlock {
                data: compressed,
                crc32,
                entries: std::mem::take(&mut entries),
            });
            raw_buffer.clear();
        }
    }
    if !raw_buffer.is_empty() {
        let compressed = codec::compress(&raw_buffer, codec::Codec::Zstd, 3, None)?;
        let crc32 = calc_crc32(&compressed);
        new_blocks.push(CompressedBlock {
            data: compressed,
            crc32,
            entries,
        });
    }

    // Phase 4: Open archive for appending
    let mut file = OpenOptions::new().read(true).write(true).open(archive_path)?;

    // Find the end of existing data blocks (= old manifest offset, since manifest is after data)
    let mut header_buf = [0u8; HEADER_SIZE];
    file.read_exact(&mut header_buf)?;
    let old_header = NraHeader::from_bytes(&header_buf)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    // Append position: after existing data, before old manifest
    // In the current format: [Header][Manifest][Data...]
    // We need to figure out where data ends. The manifest is at manifest_offset.
    // Data starts at data_section_offset.
    // For existing archives, the manifest comes first, then data.
    // So the end of data = file_size (or manifest_offset if manifest is at the end).
    let file_len = file.seek(SeekFrom::End(0))?;
    let append_offset = file_len;

    // Phase 5: Append new blocks
    file.seek(SeekFrom::Start(append_offset))?;
    let mut writer = BufWriter::new(&mut file);

    let mut block_offsets: Vec<u64> = Vec::new();
    let mut current_offset = append_offset;

    for block in &new_blocks {
        writer.write_all(&block.data)?;
        block_offsets.push(current_offset);
        current_offset += block.data.len() as u64;
    }

    // Phase 6: Build merged manifest
    let mut merged_manifest = old_manifest;

    // Add new chunk records
    for (bi, block) in new_blocks.iter().enumerate() {
        for (hash_hex, inner_offset, original_size) in &block.entries {
            merged_manifest.chunk_table.push(BetaChunkRecord {
                hash: hash_hex.clone(),
                offset: block_offsets[bi],
                compressed_size: block.data.len() as u64,
                original_size: *original_size,
                inner_offset: *inner_offset,
                crc32: block.crc32,
            });
        }
    }

    // Add new file records
    for recipe in &recipes {
        merged_manifest.files.push(BetaFileRecord {
            id: recipe.file_id.clone(),
            original_size: recipe.original_size,
            chunks: recipe.chunk_hashes.iter().map(hash_to_hex).collect(),
        });
    }

    merged_manifest.summary.total_files = merged_manifest.files.len() as u64;
    merged_manifest.summary.total_chunks = merged_manifest.chunk_table.len() as u64;

    // Recalculate summary stats that delta invalidated
    merged_manifest.summary.total_original_bytes = merged_manifest
        .files
        .iter()
        .map(|f| f.original_size)
        .sum();
    merged_manifest.summary.total_stored_bytes = {
        // Sum of unique compressed block sizes (deduplicate by offset to avoid double-counting)
        let mut seen_offsets = std::collections::HashSet::new();
        let mut total = 0u64;
        for chunk in &merged_manifest.chunk_table {
            if seen_offsets.insert(chunk.offset) {
                total += chunk.compressed_size;
            }
        }
        total
    };
    merged_manifest.summary.dedup_ratio = if merged_manifest.summary.total_stored_bytes > 0 {
        merged_manifest.summary.total_original_bytes as f64
            / merged_manifest.summary.total_stored_bytes as f64
    } else {
        1.0
    };

    // Write new manifest after the new data blocks
    let manifest_bytes = merged_manifest.serialize()?;
    let new_manifest_offset = current_offset;
    let new_manifest_size = manifest_bytes.len() as u64;
    writer.write_all(&manifest_bytes)?;
    writer.flush()?;

    // Phase 7: Overwrite header
    drop(writer);
    file.seek(SeekFrom::Start(0))?;
    let mut new_header_bytes = [0u8; HEADER_SIZE];
    new_header_bytes[0..4].copy_from_slice(&crate::format::MAGIC_BYTES);
    new_header_bytes[4..6].copy_from_slice(&crate::format::FORMAT_VERSION.to_le_bytes());
    new_header_bytes[6..8].copy_from_slice(&0u16.to_le_bytes()); // flags
    new_header_bytes[8..16].copy_from_slice(&new_manifest_offset.to_le_bytes());
    new_header_bytes[16..24].copy_from_slice(&new_manifest_size.to_le_bytes());
    new_header_bytes[24..32].copy_from_slice(&old_header.data_section_offset.to_le_bytes());
    file.write_all(&new_header_bytes)?;
    file.flush()?;

    Ok(new_chunk_count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::beta_writer::BetaWriter;
    use crate::beta_reader::BetaReader;

    #[test]
    fn delta_pack_adds_files() {
        let path = std::env::temp_dir().join("nra_delta_test.nra");

        // Create initial archive with 2 files
        let mut writer = BetaWriter::new();
        writer.add_file("original_1.txt", b"This is the first file.");
        writer.add_file("original_2.txt", b"This is the second file.");
        writer.save(&path).unwrap();

        // Verify initial state
        let reader = BetaReader::open(&path).unwrap();
        assert_eq!(reader.file_ids().len(), 2);

        // Delta-pack 2 more files
        let new_files: Vec<(&str, &[u8])> = vec![
            ("delta_3.txt", b"Third file added via delta!"),
            ("delta_4.txt", b"Fourth file, also new."),
        ];
        let new_chunks = delta_pack(&path, &new_files).unwrap();
        eprintln!("Delta added {} new chunks", new_chunks);

        // Verify merged state
        let mut reader = BetaReader::open(&path).unwrap();
        assert_eq!(reader.file_ids().len(), 4);

        // All files readable
        let f1 = reader.read_file("original_1.txt").unwrap();
        assert_eq!(&f1, b"This is the first file.");

        let f3 = reader.read_file("delta_3.txt").unwrap();
        assert_eq!(&f3, b"Third file added via delta!");

        let f4 = reader.read_file("delta_4.txt").unwrap();
        assert_eq!(&f4, b"Fourth file, also new.");

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn delta_pack_dedup_existing_chunks() {
        let path = std::env::temp_dir().join("nra_delta_dedup.nra");
        let shared = vec![0xABu8; 32768]; // 32KB

        // Create initial archive
        let mut writer = BetaWriter::new();
        writer.add_file("v1.bin", &shared);
        writer.save(&path).unwrap();

        let size_before = std::fs::metadata(&path).unwrap().len();

        // Delta-pack the same data under a different name
        let new_files: Vec<(&str, &[u8])> = vec![("v2.bin", &shared)];
        let new_chunks = delta_pack(&path, &new_files).unwrap();

        let size_after = std::fs::metadata(&path).unwrap().len();
        eprintln!("Delta dedup: before={} after={} new_chunks={}", size_before, size_after, new_chunks);

        // Should have 0 new chunks (all deduplicated)
        assert_eq!(new_chunks, 0, "Identical data should be fully deduplicated");

        // But the file should still be readable
        let mut reader = BetaReader::open(&path).unwrap();
        assert_eq!(reader.file_ids().len(), 2);
        let v2 = reader.read_file("v2.bin").unwrap();
        assert_eq!(v2, shared);

        std::fs::remove_file(&path).ok();
    }
}
