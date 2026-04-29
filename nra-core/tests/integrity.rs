use nra_core::{NraReader, NraWriter};
use std::fs;
use std::path::PathBuf;

fn create_test_archive(path: &PathBuf) {
    let mut writer = NraWriter::new();
    writer.set_name("Integrity Test");
    writer
        .add_file("secret.txt", b"This is a top secret AI dataset")
        .unwrap();
    writer
        .add_file("data.bin", &[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        .unwrap();
    writer.save(path).unwrap();
}

#[test]
fn test_archive_integrity_valid() {
    let path = std::env::temp_dir().join("valid_archive.nra");
    create_test_archive(&path);

    // Reader should open it without errors
    let mut reader = NraReader::open(&path).expect("Failed to open valid archive");

    // File reading should succeed
    let data = reader
        .read_file("secret.txt")
        .expect("Failed to read valid file");
    assert_eq!(&data, b"This is a top secret AI dataset");

    fs::remove_file(&path).ok();
}

#[test]
fn test_archive_integrity_corrupted_data_block() {
    let path = std::env::temp_dir().join("corrupted_data.nra");
    create_test_archive(&path);

    // Corrupt the data
    // We know the manifest comes after the 32-byte header.
    // Find the exact offset of the data block by parsing it once
    let offset = {
        let reader = NraReader::open(&path).unwrap();
        reader.get_file_record("secret.txt").unwrap().offset as usize
    };

    let mut raw_bytes = fs::read(&path).unwrap();

    // Corrupt the first byte of the Zstd compressed data block
    raw_bytes[offset] ^= 0xFF;

    fs::write(&path, &raw_bytes).unwrap();

    let mut reader = NraReader::open(&path).unwrap();

    // CRC32 check should catch this when we try to read!
    let res1 = reader.read_file("secret.txt");

    assert!(
        res1.is_err(),
        "Corrupted data was not caught by CRC32 or Zstd!"
    );

    fs::remove_file(&path).ok();
}

#[test]
fn test_archive_integrity_corrupted_header_magic() {
    let path = std::env::temp_dir().join("corrupted_magic.nra");
    create_test_archive(&path);

    let mut raw_bytes = fs::read(&path).unwrap();
    // Corrupt MAGIC BYTES "NRA\0"
    raw_bytes[0] = b'X';
    fs::write(&path, &raw_bytes).unwrap();

    // Reader should immediately reject the file
    let res = NraReader::open(&path);
    assert!(
        res.is_err(),
        "Parser accepted a file with invalid magic bytes!"
    );

    fs::remove_file(&path).ok();
}

#[test]
fn test_archive_integrity_corrupted_manifest() {
    let path = std::env::temp_dir().join("corrupted_manifest.nra");
    create_test_archive(&path);

    let mut raw_bytes = fs::read(&path).unwrap();
    // Corrupt manifest (starts at byte 32)
    raw_bytes[35] ^= 0xFF;
    fs::write(&path, &raw_bytes).unwrap();

    // The manifest JSON is now invalid, so it should fail during parsing
    let res = NraReader::open(&path);
    assert!(res.is_err(), "Parser accepted a corrupted manifest!");

    fs::remove_file(&path).ok();
}
