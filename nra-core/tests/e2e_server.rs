use nra_core::{beta_reader::BetaReader, beta_writer::BetaWriter, stream_writer::StreamBetaWriter};


#[test]
fn test_stream_writer_full_pipeline() {
    // 1. Generate 50 test files (text + binary)
    let files: Vec<(String, Vec<u8>)> = (0..50)
        .map(|i| {
            let name = format!("file_{:04}.bin", i);
            let size = 1024 + (i * 100);
            let mut data = Vec::with_capacity(size);
            for j in 0..size {
                data.push(((i + j) % 256) as u8);
            }
            (name, data)
        })
        .collect();

    let tmp = tempfile::NamedTempFile::new().expect("Failed to create temp file");

    // 2. Pack using StreamBetaWriter (simulating what nra-registry-server does)
    let file = tmp.reopen().expect("Failed to reopen temp file");
    let mut writer = StreamBetaWriter::new(file).expect("Failed to create StreamBetaWriter");

    for (name, data) in &files {
        writer.add_file(name, data).expect("Failed to add file");
    }
    writer.finalize().expect("Failed to finalize stream writer");

    // 3. Read using BetaReader
    let mut reader = BetaReader::open(tmp.path()).expect("Failed to open with BetaReader");
    let ids = reader.file_ids().to_vec();

    // 4. Check number of files
    assert_eq!(ids.len(), 50, "Expected exactly 50 files in archive");

    // 5. Check exact byte match
    for (name, expected) in &files {
        let actual = reader.read_file(name).unwrap_or_else(|_| panic!("Failed to read {}", name));
        assert_eq!(&actual, expected, "Data mismatch for file {}", name);
    }
}

#[test]
fn test_dictionary_full_pipeline() {
    // 1. Generate 100 small JSON files (ideal for dictionary compression)
    let files: Vec<(String, Vec<u8>)> = (0..100)
        .map(|i| {
            let json = format!(
                r#"{{"id": {}, "label": "cat", "score": 0.95, "metadata": {{"source": "imagenet"}}}}"#,
                i
            );
            (format!("sample_{:04}.json", i), json.into_bytes())
        })
        .collect();

    let tmp = tempfile::NamedTempFile::new().expect("Failed to create temp file");

    let mut writer = BetaWriter::new();
    
    for (name, data) in &files {
        writer.add_file(name, data);
    }

    // Train dictionary on incoming chunks automatically
    writer.train_dictionary(110 * 1024).expect("Failed to train dictionary");
    writer.save(tmp.path()).expect("Failed to finalize writer");

    // 3. Open with BetaReader
    let mut reader = BetaReader::open(tmp.path()).expect("Failed to open with BetaReader");

    // Verify dictionary was actually recorded in manifest
    assert!(
        reader.manifest().dictionary.is_some(),
        "Dictionary was not saved in manifest"
    );

    // 4. Exact byte match verify
    for (name, expected) in &files {
        let actual = reader.read_file(name).unwrap_or_else(|_| panic!("Failed to read {}", name));
        assert_eq!(&actual, expected, "Dictionary data mismatch for file {}", name);
    }
}
