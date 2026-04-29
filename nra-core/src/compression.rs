use std::io::{self, Read};

pub fn compress_zstd(data: &[u8], level: i32) -> io::Result<Vec<u8>> {
    zstd::encode_all(data, level)
}

pub fn decompress_zstd(data: &[u8]) -> io::Result<Vec<u8>> {
    // Safety: cap decompressed output to prevent zip-bomb attacks.
    // In Chunked mode, a malicious 10 MB compressed block could expand to 100 GB.
    const MAX_DECOMPRESSED: usize = 512 * 1024 * 1024; // 512 MiB

    let mut decoder = io::Cursor::new(data);
    let mut output = Vec::new();
    let mut zstd_reader = zstd::Decoder::new(&mut decoder)?;

    // Read in chunks to detect oversized output early without pre-allocating
    let mut buf = [0u8; 64 * 1024]; // 64 KiB read buffer
    loop {
        let n = zstd_reader.read(&mut buf)?;
        if n == 0 {
            break;
        }
        if output.len() + n > MAX_DECOMPRESSED {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "Decompressed output exceeds {} MiB safety limit (zip-bomb protection)",
                    MAX_DECOMPRESSED / (1024 * 1024)
                ),
            ));
        }
        output.extend_from_slice(&buf[..n]);
    }
    Ok(output)
}
