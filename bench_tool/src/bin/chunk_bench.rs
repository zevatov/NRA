use rand::Rng;
use std::io::Write;

fn main() {
    let mut rng = rand::thread_rng();
    let num_files = 10000;

    // Create some common vocabulary to simulate NLP/JSON datasets
    let vocab = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "user_id",
        "email",
        "timestamp",
        "payload",
        "status",
        "success",
    ];

    let mut all_files = Vec::new();
    let mut total_raw_size = 0;

    println!("Generating {} tiny text files...", num_files);
    for _ in 0..num_files {
        let mut content = String::new();
        for _ in 0..50 {
            let word = vocab[rng.gen_range(0..vocab.len())];
            content.push_str(word);
            content.push(' ');
        }
        let bytes = content.into_bytes();
        total_raw_size += bytes.len();
        all_files.push(bytes);
    }

    println!("Total Raw Size: {} KB", total_raw_size / 1024);

    // 1. Per-File Compression (Our current method)
    let mut total_per_file_compressed_size = 0;
    for file in &all_files {
        // Compress using zstd level 3
        let compressed = zstd::stream::encode_all(file.as_slice(), 3).unwrap();
        total_per_file_compressed_size += compressed.len();
    }
    println!(
        "Per-File Zstd Size (No Chunks): {} KB",
        total_per_file_compressed_size / 1024
    );

    // 2. Chunked Compression (All files concatenated, then compressed)
    let mut solid_buffer = Vec::new();
    for file in &all_files {
        solid_buffer.write_all(file).unwrap();
    }
    let solid_compressed = zstd::stream::encode_all(solid_buffer.as_slice(), 3).unwrap();
    println!(
        "Solid Zstd Size (With Chunks): {} KB",
        solid_compressed.len() / 1024
    );

    println!("-------------------------------");
    println!(
        "Compression Ratio (Raw -> No Chunks): {:.2}x",
        total_raw_size as f64 / total_per_file_compressed_size as f64
    );
    println!(
        "Compression Ratio (Raw -> With Chunks): {:.2}x",
        total_raw_size as f64 / solid_compressed.len() as f64
    );
    println!(
        "Difference: Chunked is {:.2}x smaller than Per-File!",
        total_per_file_compressed_size as f64 / solid_compressed.len() as f64
    );
}
