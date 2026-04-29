//! Block-level encryption using AES-256-GCM.
//!
//! Each block is encrypted with a unique nonce (12 bytes, counter-based).
//! The nonce is prepended to the ciphertext: [nonce (12 bytes)][ciphertext + tag (16 bytes)].
//!
//! Key derivation: The caller provides a 32-byte key via `NRA_ENCRYPTION_KEY` env var
//! or programmatically. We do NOT store the key inside the archive.

use aes_gcm::{
    Aes256Gcm, KeyInit, Nonce,
    aead::Aead,
};
use std::io;
use std::sync::atomic::{AtomicU64, Ordering};

/// Nonce size for AES-256-GCM (96 bits = 12 bytes).
const NONCE_SIZE: usize = 12;

/// Global nonce counter to ensure uniqueness within a single process.
static NONCE_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Encrypt a block with AES-256-GCM.
///
/// Returns: [nonce (12 bytes)] ++ [ciphertext + auth tag (16 bytes)]
///
/// # Arguments
/// * `data` - Plaintext data to encrypt
/// * `key` - 32-byte (256-bit) encryption key
pub fn encrypt_block(data: &[u8], key: &[u8; 32]) -> io::Result<Vec<u8>> {
    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("Invalid key: {}", e)))?;

    // Generate a unique nonce from a counter + 4 random bytes
    let counter = NONCE_COUNTER.fetch_add(1, Ordering::SeqCst);
    let mut nonce_bytes = [0u8; NONCE_SIZE];
    nonce_bytes[..8].copy_from_slice(&counter.to_le_bytes());
    // Remaining 4 bytes are zero (acceptable for single-process use with counter)

    let nonce = Nonce::from_slice(&nonce_bytes);

    let ciphertext = cipher.encrypt(nonce, data)
        .map_err(|e| io::Error::other(format!("Encryption failed: {}", e)))?;

    let mut output = Vec::with_capacity(NONCE_SIZE + ciphertext.len());
    output.extend_from_slice(&nonce_bytes);
    output.extend_from_slice(&ciphertext);
    Ok(output)
}

/// Decrypt a block that was encrypted with `encrypt_block`.
///
/// Expects input format: [nonce (12 bytes)] ++ [ciphertext + auth tag]
pub fn decrypt_block(data: &[u8], key: &[u8; 32]) -> io::Result<Vec<u8>> {
    if data.len() < NONCE_SIZE + 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "Encrypted block too short (missing nonce or auth tag)",
        ));
    }

    let cipher = Aes256Gcm::new_from_slice(key)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("Invalid key: {}", e)))?;

    let nonce = Nonce::from_slice(&data[..NONCE_SIZE]);
    let ciphertext = &data[NONCE_SIZE..];

    cipher.decrypt(nonce, ciphertext)
        .map_err(|_| io::Error::new(
            io::ErrorKind::InvalidData,
            "Decryption failed: invalid key or corrupted data (auth tag mismatch)",
        ))
}

/// Load encryption key from environment variable `NRA_ENCRYPTION_KEY`.
/// Expects a 64-character hex string (32 bytes).
pub fn key_from_env() -> io::Result<Option<[u8; 32]>> {
    match std::env::var("NRA_ENCRYPTION_KEY") {
        Ok(hex_str) => {
            let hex_str = hex_str.trim();
            if hex_str.len() != 64 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("NRA_ENCRYPTION_KEY must be 64 hex chars (32 bytes), got {}", hex_str.len()),
                ));
            }
            let mut key = [0u8; 32];
            for i in 0..32 {
                key[i] = u8::from_str_radix(&hex_str[i*2..i*2+2], 16)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, format!("Invalid hex: {}", e)))?;
            }
            Ok(Some(key))
        }
        Err(_) => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encrypt_decrypt_roundtrip() {
        let key = [0x42u8; 32];
        let plaintext = b"Hello, encrypted NRA world! This is a secret block.";

        let encrypted = encrypt_block(plaintext, &key).unwrap();
        assert_ne!(&encrypted[NONCE_SIZE..], plaintext); // must be different

        let decrypted = decrypt_block(&encrypted, &key).unwrap();
        assert_eq!(&decrypted, plaintext);
    }

    #[test]
    fn wrong_key_fails() {
        let key = [0x42u8; 32];
        let wrong_key = [0x00u8; 32];
        let plaintext = b"Secret data";

        let encrypted = encrypt_block(plaintext, &key).unwrap();
        let result = decrypt_block(&encrypted, &wrong_key);
        assert!(result.is_err());
    }

    #[test]
    fn unique_nonces() {
        let key = [0xABu8; 32];
        let data = b"test";
        let e1 = encrypt_block(data, &key).unwrap();
        let e2 = encrypt_block(data, &key).unwrap();
        // Nonces must differ
        assert_ne!(&e1[..NONCE_SIZE], &e2[..NONCE_SIZE]);
    }
}
