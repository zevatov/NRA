use crate::checksum::calc_crc32;
use crate::compression::compress_zstd;
use crate::format::{HEADER_SIZE, NraHeader};
use crate::manifest::{Compression, FileRecord, FileVector, Manifest};
use sha2::{Digest, Sha256};
use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizationMode {
    Speed,
    Size,
}

pub struct NraWriter {
    manifest: Manifest,
    blocks: Vec<Vec<u8>>,
    mode: OptimizationMode,

    // Chunking state for OptimizationMode::Size
    chunk_buffer: Vec<u8>,
    chunk_files: Vec<usize>,            // indices into manifest.files
    block_assignments: Vec<Vec<usize>>, // block_index -> [manifest_file_indices]
}

impl Default for NraWriter {
    fn default() -> Self {
        Self::new()
    }
}

impl NraWriter {
    pub fn new() -> Self {
        Self {
            manifest: Manifest::new(),
            blocks: Vec::new(),
            mode: OptimizationMode::Speed,
            chunk_buffer: Vec::new(),
            chunk_files: Vec::new(),
            block_assignments: Vec::new(),
        }
    }

    pub fn set_mode(&mut self, mode: OptimizationMode) {
        self.mode = mode;
    }

    pub fn set_name(&mut self, name: &str) {
        self.manifest.summary.name = name.to_string();
    }

    pub fn add_file(&mut self, id: &str, data: &[u8]) -> io::Result<()> {
        match self.mode {
            OptimizationMode::Speed => {
                let compressed = compress_zstd(data, 3)?;
                let crc32 = calc_crc32(&compressed);

                let record = FileRecord {
                    id: id.to_string(),
                    offset: 0,
                    inner_offset: 0,
                    compressed_size: compressed.len() as u64,
                    original_size: data.len() as u64,
                    crc32,
                    compression: Compression::Zstd,
                    vectors: Vec::new(),
                };

                self.manifest.files.push(record);
                self.blocks.push(compressed);
                self.block_assignments
                    .push(vec![self.manifest.files.len() - 1]);
                self.manifest.summary.total_files += 1;
            }
            OptimizationMode::Size => {
                let inner_offset = self.chunk_buffer.len() as u64;
                self.chunk_buffer.extend_from_slice(data);

                let record = FileRecord {
                    id: id.to_string(),
                    offset: 0,
                    inner_offset,
                    compressed_size: 0, // Will be updated on flush
                    original_size: data.len() as u64,
                    crc32: 0, // Will be updated on flush
                    compression: Compression::Zstd,
                    vectors: Vec::new(),
                };

                self.manifest.files.push(record);
                self.chunk_files.push(self.manifest.files.len() - 1);
                self.manifest.summary.total_files += 1;

                // Flush if chunk exceeds 64 MB
                if self.chunk_buffer.len() > 64 * 1024 * 1024 {
                    self.flush_chunk()?;
                }
            }
        }
        Ok(())
    }

    fn flush_chunk(&mut self) -> io::Result<()> {
        if self.chunk_buffer.is_empty() {
            return Ok(());
        }

        let compressed = compress_zstd(&self.chunk_buffer, 3)?;
        let crc32 = calc_crc32(&compressed);
        let compressed_size = compressed.len() as u64;

        for &file_idx in &self.chunk_files {
            self.manifest.files[file_idx].compressed_size = compressed_size;
            self.manifest.files[file_idx].crc32 = crc32;
        }

        self.blocks.push(compressed);
        self.block_assignments.push(self.chunk_files.clone());

        self.chunk_buffer.clear();
        self.chunk_files.clear();
        Ok(())
    }

    pub fn add_vector(
        &mut self,
        file_id: &str,
        space_id: &str,
        vector_data: &[u8],
    ) -> Result<(), &'static str> {
        if let Some(file) = self.manifest.files.iter_mut().find(|f| f.id == file_id) {
            file.vectors.push(FileVector {
                space_id: space_id.to_string(),
                data: vector_data.to_vec(),
            });
            Ok(())
        } else {
            Err("File ID not found in manifest")
        }
    }

    /// Finalize and write the .nra archive to disk.
    ///
    /// Strategy to solve the "manifest size changes after offsets are written" problem:
    /// 1. Serialize manifest with placeholder offsets (all 0).
    /// 2. Measure the serialized size — this is the UPPER BOUND because
    ///    real offsets (positive integers) are always >= the placeholder size.
    ///    Wait — that's wrong, real offsets are LARGER numbers than 0.
    ///    So we use a two-pass approach:
    ///    Pass 1: serialize with max-width placeholder offsets to get stable size.
    ///    Pass 2: serialize with real offsets, pad to the same size.
    ///
    /// Actually, the simplest correct approach: serialize once with real offsets,
    /// then use THAT size to recompute. If the size changed, iterate until stable.
    pub fn save<P: AsRef<Path>>(mut self, path: P) -> io::Result<()> {
        self.flush_chunk()?;

        // Iterative offset stabilization (converges in 1-2 iterations for JSON)
        let manifest_bytes = self.stabilize_offsets()?;
        let manifest_size = manifest_bytes.len() as u64;
        let data_section_offset = HEADER_SIZE as u64 + manifest_size;

        let header = NraHeader::new(manifest_size, data_section_offset);
        let header_bytes = header.to_bytes();

        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);
        let mut hasher = Sha256::new();

        // 1. Header (32 bytes)
        writer.write_all(&header_bytes)?;
        hasher.update(header_bytes);

        // 2. Manifest
        writer.write_all(&manifest_bytes)?;
        hasher.update(&manifest_bytes);

        // 3. Data blocks
        for block in &self.blocks {
            writer.write_all(block)?;
            hasher.update(block);
        }

        // 4. Footer: SHA-256(header+manifest+data) + manifest_crc32 + magic
        let archive_hash = hasher.finalize();
        let manifest_crc32 = calc_crc32(&manifest_bytes);

        writer.write_all(&archive_hash)?;
        writer.write_all(&manifest_crc32.to_le_bytes())?;
        writer.write_all(&crate::format::MAGIC_BYTES)?;
        writer.flush()?;

        Ok(())
    }

    /// Iteratively serialize the manifest until the byte size stabilizes.
    /// This is needed because offset values change the JSON size,
    /// which in turn changes the offsets. Converges in 1-2 rounds.
    fn stabilize_offsets(&mut self) -> io::Result<Vec<u8>> {
        const MAX_ITERATIONS: usize = 8;

        for _ in 0..MAX_ITERATIONS {
            let trial = self.manifest.serialize()?;
            let manifest_size = trial.len() as u64;
            let data_start = HEADER_SIZE as u64 + manifest_size;

            let mut offset = data_start;
            let mut changed = false;

            for (block_idx, block) in self.blocks.iter().enumerate() {
                let block_size = block.len() as u64;
                for &file_idx in &self.block_assignments[block_idx] {
                    if self.manifest.files[file_idx].offset != offset {
                        self.manifest.files[file_idx].offset = offset;
                        changed = true;
                    }
                }
                offset += block_size;
            }

            if !changed {
                return Ok(trial);
            }
        }

        Err(io::Error::other(
            "Manifest offset stabilization did not converge",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::NraReader;

    #[test]
    fn write_and_read_roundtrip() {
        let dir = std::env::temp_dir().join("nra_test_roundtrip.nra");

        let mut writer = NraWriter::new();
        writer.set_name("Test Archive");
        writer.add_file("hello.txt", b"Hello, NRA!").unwrap();
        writer
            .add_file("data.bin", &[0xDE, 0xAD, 0xBE, 0xEF])
            .unwrap();
        writer.save(&dir).unwrap();

        let mut reader = NraReader::open(&dir).unwrap();
        assert_eq!(reader.manifest().summary.total_files, 2);

        let hello = reader.read_file("hello.txt").unwrap();
        assert_eq!(&hello, b"Hello, NRA!");

        let data = reader.read_file("data.bin").unwrap();
        assert_eq!(&data, &[0xDE, 0xAD, 0xBE, 0xEF]);

        assert!(reader.read_file("nope.txt").is_err());
        std::fs::remove_file(&dir).ok();
    }

    #[test]
    fn write_and_read_chunked_size_mode() {
        let dir = std::env::temp_dir().join("nra_test_size_mode.nra");

        let mut writer = NraWriter::new();
        writer.set_mode(OptimizationMode::Size);
        writer.set_name("Test Archive Chunks");

        // Add 3 tiny files that should go into the same chunk
        writer.add_file("file1.txt", b"First file data").unwrap();
        writer
            .add_file("file2.txt", b"Second file data goes here")
            .unwrap();
        writer.add_file("file3.txt", b"Third one").unwrap();

        writer.save(&dir).unwrap();

        let mut reader = NraReader::open(&dir).unwrap();
        assert_eq!(reader.manifest().summary.total_files, 3);

        // Verify they all share the same offset (they are in the same block)
        let r1 = reader.get_file_record("file1.txt").unwrap();
        let r2 = reader.get_file_record("file2.txt").unwrap();
        assert_eq!(r1.offset, r2.offset);
        assert_ne!(r1.inner_offset, r2.inner_offset);

        // Verify content
        assert_eq!(&reader.read_file("file1.txt").unwrap(), b"First file data");
        assert_eq!(
            &reader.read_file("file2.txt").unwrap(),
            b"Second file data goes here"
        );
        assert_eq!(&reader.read_file("file3.txt").unwrap(), b"Third one");

        std::fs::remove_file(&dir).ok();
    }
}
