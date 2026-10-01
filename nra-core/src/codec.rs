//! Codec abstraction: Zstd (best ratio) or LZ4 (fastest decompression).
//!
//! Each compressed block stores a 1-byte codec tag before the payload:
//!   0x01 = Zstd
//!   0x02 = LZ4
//!
//! On read, the tag is auto-detected so old and new archives coexist.

use std::io;

/// Compression codec selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum Codec {
    /// Zstd: Best compression ratio. Default for archival workloads.
    Zstd = 0x01,
    /// LZ4: ~3x faster decompression. Use for real-time inference pipelines.
    Lz4 = 0x02,
}

impl Codec {
    pub fn from_tag(tag: u8) -> io::Result<Self> {
        match tag {
            0x01 => Ok(Codec::Zstd),
            0x02 => Ok(Codec::Lz4),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("Unknown codec tag: 0x{:02X}", tag),
            )),
        }
    }

    pub fn tag(self) -> u8 {
        self as u8
    }
}

/// Compress data with the specified codec. Prepends a 1-byte codec tag.
pub fn compress(data: &[u8], codec: Codec, zstd_level: i32, dictionary: Option<&[u8]>) -> io::Result<Vec<u8>> {
    let compressed = match codec {
        Codec::Zstd => {
            if let Some(dict) = dictionary {
                let mut encoder = zstd::stream::Encoder::with_dictionary(Vec::new(), zstd_level, dict)?;
                std::io::Write::write_all(&mut encoder, data)?;
                encoder.finish()?
            } else {
                zstd::encode_all(data, zstd_level)?
            }
        },
        Codec::Lz4 => lz4_flex::compress_prepend_size(data).to_vec(),
    };
    let mut out = Vec::with_capacity(1 + compressed.len());
    out.push(codec.tag());
    out.extend_from_slice(&compressed);
    Ok(out)
}

/// Decompress data. Auto-detects codec from the 1-byte tag prefix.
/// Falls back to raw Zstd if the first byte is not a valid tag
/// (backwards compatibility with pre-codec archives).
pub fn decompress(data: &[u8], dictionary: Option<&[u8]>) -> io::Result<Vec<u8>> {
    if data.is_empty() {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "Empty block"));
    }

    // Try to detect codec tag
    match Codec::from_tag(data[0]) {
        Ok(Codec::Zstd) => decompress_zstd_safe(&data[1..], dictionary),
        Ok(Codec::Lz4) => decompress_lz4_safe(&data[1..]),
        Err(_) => {
            // No tag — legacy Zstd block (pre-codec era). Decompress as raw Zstd.
            decompress_zstd_safe(data, dictionary)
        }
    }
}

fn decompress_zstd_safe(data: &[u8], dictionary: Option<&[u8]>) -> io::Result<Vec<u8>> {
    use std::io::Read;
    const MAX_DECOMPRESSED: usize = 512 * 1024 * 1024; // 512 MiB

    let decoder = io::Cursor::new(data);
    let mut output = Vec::new();
    
    // We cannot easily use the same type variable for Decoder because lifetimes differ,
    // so we just branch the entire read loop.
    if let Some(dict) = dictionary {
        let mut zstd_reader = zstd::stream::Decoder::with_dictionary(decoder, dict)?;
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = zstd_reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            if output.len() + n > MAX_DECOMPRESSED {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Decompressed output exceeds {} MiB safety limit", MAX_DECOMPRESSED / (1024 * 1024)),
                ));
            }
            output.extend_from_slice(&buf[..n]);
        }
    } else {
        let mut zstd_reader = zstd::stream::Decoder::new(decoder)?;
        let mut buf = [0u8; 64 * 1024];
        loop {
            let n = zstd_reader.read(&mut buf)?;
            if n == 0 {
                break;
            }
            if output.len() + n > MAX_DECOMPRESSED {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("Decompressed output exceeds {} MiB safety limit", MAX_DECOMPRESSED / (1024 * 1024)),
                ));
            }
            output.extend_from_slice(&buf[..n]);
        }
    }
    
    Ok(output)
}

fn decompress_lz4_safe(data: &[u8]) -> io::Result<Vec<u8>> {
    const MAX_DECOMPRESSED: usize = 512 * 1024 * 1024; // 512 MiB

    lz4_flex::decompress_size_prepended(data).map_err(|e| {
        io::Error::new(io::ErrorKind::InvalidData, format!("LZ4 decompression failed: {}", e))
    }).and_then(|out| {
        if out.len() > MAX_DECOMPRESSED {
            Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "LZ4 decompressed output exceeds safety limit",
            ))
        } else {
            Ok(out)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zstd_roundtrip() {
        let original = b"Hello NRA! ".repeat(1000);
        let compressed = compress(&original, Codec::Zstd, 3, None).unwrap();
        assert_eq!(compressed[0], 0x01);
        let decompressed = decompress(&compressed, None).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn lz4_roundtrip() {
        let original = b"Fast LZ4 path! ".repeat(1000);
        let compressed = compress(&original, Codec::Lz4, 0, None).unwrap();
        assert_eq!(compressed[0], 0x02);
        let decompressed = decompress(&compressed, None).unwrap();
        assert_eq!(decompressed, original);
    }

    #[test]
    fn legacy_zstd_compat() {
        // Simulate a legacy block (no tag, raw Zstd)
        let original = b"Legacy data".to_vec();
        let raw_zstd = zstd::encode_all(original.as_slice(), 3).unwrap();
        // Legacy blocks don't have a tag, decompress should still work
        let decompressed = decompress(&raw_zstd, None).unwrap();
        assert_eq!(decompressed, original);
    }
}
