use crate::checksum::calc_crc32;
use crate::compression::decompress_zstd;
use crate::format::{HEADER_SIZE, NraHeader};
use crate::manifest::{Compression, FileRecord, Manifest};
use std::fs::File;
use std::io::{self, Read, Seek, SeekFrom};
use std::path::Path;

pub struct NraReader {
    file: File,
    #[allow(dead_code)] // Retained for future footer verification and version checks
    header: NraHeader,
    manifest: Manifest,
}

impl NraReader {
    pub fn open<P: AsRef<Path>>(path: P) -> io::Result<Self> {
        let mut file = File::open(path)?;

        // Read and validate 32-byte header
        let mut header_buf = [0u8; HEADER_SIZE];
        file.read_exact(&mut header_buf)?;
        let header = NraHeader::from_bytes(&header_buf)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

        // Sanity check: don't allocate absurd amounts of memory
        if header.manifest_size > 1_073_741_824 {
            // 1 GiB limit for manifest
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Manifest size exceeds 1 GiB safety limit",
            ));
        }

        // Read manifest
        let mut manifest_buf = vec![0u8; header.manifest_size as usize];
        file.seek(SeekFrom::Start(header.manifest_offset))?;
        file.read_exact(&mut manifest_buf)?;

        let manifest = Manifest::deserialize(&manifest_buf)?;

        Ok(Self {
            file,
            header,
            manifest,
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn file_ids(&self) -> Vec<&str> {
        self.manifest.files.iter().map(|f| f.id.as_str()).collect()
    }

    pub fn get_file_record(&self, file_id: &str) -> Option<&FileRecord> {
        self.manifest.files.iter().find(|f| f.id == file_id)
    }

    pub fn read_file(&mut self, file_id: &str) -> io::Result<Vec<u8>> {
        let record = self
            .get_file_record(file_id)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "File not found in manifest"))?;

        // Copy fields to release the immutable borrow on self before mutating self.file
        let offset = record.offset;
        let inner_offset = record.inner_offset;
        let compressed_size = record.compressed_size as usize;
        let original_size = record.original_size as usize;
        let expected_crc = record.crc32;
        let compression = record.compression;

        // Safety: cap decompressed size to prevent zip-bomb style attacks
        if original_size > 4_294_967_296 {
            // 4 GiB per file
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "File original_size exceeds 4 GiB safety limit",
            ));
        }

        // Safety: cap compressed block size to prevent OOM from malicious manifests
        const MAX_COMPRESSED_BLOCK: usize = 512 * 1024 * 1024; // 512 MiB
        if compressed_size > MAX_COMPRESSED_BLOCK {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Compressed block size {} exceeds {} MiB safety limit",
                    compressed_size,
                    MAX_COMPRESSED_BLOCK / (1024 * 1024)
                ),
            ));
        }

        let mut buf = vec![0u8; compressed_size];
        self.file.seek(SeekFrom::Start(offset))?;
        self.file.read_exact(&mut buf)?;

        // Verify CRC32 BEFORE decompression (protects against corrupted input)
        let computed_crc = calc_crc32(&buf);
        if computed_crc != expected_crc {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "CRC32 mismatch for '{}': expected 0x{:08X}, got 0x{:08X}",
                    file_id, expected_crc, computed_crc
                ),
            ));
        }

        // Decompress based on per-file compression field
        let decompressed = match compression {
            Compression::Zstd => decompress_zstd(&buf)?,
            Compression::None => buf,
            Compression::Lz4 => {
                return Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    "LZ4 decompression not implemented yet",
                ));
            }
        };

        // Extract the specific file from the chunk using inner_offset
        let start = inner_offset as usize;
        let end = start + original_size;

        if end > decompressed.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Inner offset out of bounds",
            ));
        }

        Ok(decompressed[start..end].to_vec())
    }
}
