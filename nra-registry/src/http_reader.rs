use nra_core::{
    compression::decompress_zstd, Manifest, NraHeader, HEADER_SIZE,
};
use reqwest::blocking::Client;
use reqwest::header::RANGE;
use std::io::{self, Error, ErrorKind};

pub struct HttpReader {
    client: Client,
    url: String,
    #[allow(dead_code)] // Retained for future integrity checks
    header: NraHeader,
    manifest: Manifest,
}

impl HttpReader {
    /// Open an .nra archive directly from an HTTP(S) URL.
    pub fn open(url: &str) -> io::Result<Self> {
        let client = Client::new();
        
        // 1. Fetch Header (first 32 bytes)
        let header_bytes = Self::fetch_range(&client, url, 0, (HEADER_SIZE - 1) as u64)?;
        if header_bytes.len() != HEADER_SIZE {
            return Err(Error::new(ErrorKind::InvalidData, "Incomplete header received"));
        }
        let mut buf = [0u8; 32];
        buf.copy_from_slice(&header_bytes);
        
        let header = NraHeader::from_bytes(&buf)
            .map_err(|e| Error::new(ErrorKind::InvalidData, e))?;
            
        // 2. Fetch Manifest
        let manifest_size = header.manifest_size;
        if manifest_size > 1024 * 1024 * 1024 { // 1 GiB safety cap
            return Err(Error::new(ErrorKind::InvalidData, "Manifest size exceeds safety cap (1 GiB)"));
        }
        
        let manifest_bytes = Self::fetch_range(
            &client, 
            url, 
            header.manifest_offset, 
            header.manifest_offset + manifest_size - 1
        )?;
        
        let manifest: Manifest = serde_json::from_slice(&manifest_bytes)
            .map_err(|e| Error::new(ErrorKind::InvalidData, format!("Failed to parse manifest: {}", e)))?;
            
        Ok(Self {
            client,
            url: url.to_string(),
            header,
            manifest,
        })
    }

    /// Read a specific file from the cloud archive
    pub fn read_file(&self, file_id: &str) -> io::Result<Vec<u8>> {
        let record = self.manifest.files.iter()
            .find(|f| f.id == file_id)
            .ok_or_else(|| Error::new(ErrorKind::NotFound, "File not found in archive"))?;
            
        // Safety cap for compressed block (prevent OOM before fetching)
        if record.compressed_size > 512 * 1024 * 1024 {
            return Err(Error::new(ErrorKind::InvalidData, "Compressed block exceeds safety cap (512 MiB)"));
        }
        
        // Calculate the exact byte range for this file's compressed chunk (record.offset is absolute)
        let start = record.offset;
        let end = start + record.compressed_size - 1;
        
        // Fetch the compressed chunk via HTTP Range request
        let compressed_data = Self::fetch_range(&self.client, &self.url, start, end)?;
        
        // Decompress based on the compression algorithm
        let raw_decompressed = match record.compression {
            nra_core::Compression::None => compressed_data,
            nra_core::Compression::Zstd => {
                decompress_zstd(&compressed_data)
                    .map_err(|e| Error::new(ErrorKind::InvalidData, format!("Decompression failed: {}", e)))?
            }
            nra_core::Compression::Lz4 => {
                return Err(Error::new(ErrorKind::Unsupported, "LZ4 decompression not implemented yet"));
            }
        };
        
        // If this is a chunked archive (Size mode), we slice out the exact inner file using inner_offset
        let inner_start = record.inner_offset as usize;
        let inner_end = inner_start + record.original_size as usize;
        
        if inner_end > raw_decompressed.len() {
            return Err(Error::new(ErrorKind::InvalidData, "Inner offset out of bounds"));
        }
        
        Ok(raw_decompressed[inner_start..inner_end].to_vec())
    }
    
    /// Get the list of all file IDs
    pub fn file_ids(&self) -> Vec<&String> {
        self.manifest.files.iter().map(|f| &f.id).collect()
    }

    fn fetch_range(client: &Client, url: &str, start: u64, end: u64) -> io::Result<Vec<u8>> {
        let range_header = format!("bytes={}-{}", start, end);
        let resp = client.get(url)
            .header(RANGE, range_header)
            .send()
            .map_err(|e| Error::other(format!("HTTP request failed: {}", e)))?;
            
        if !resp.status().is_success() {
            return Err(Error::other(format!("HTTP error: {}", resp.status())));
        }
        
        let bytes = resp.bytes()
            .map_err(|e| Error::other(format!("Failed to read body: {}", e)))?;
            
        Ok(bytes.to_vec())
    }
}
