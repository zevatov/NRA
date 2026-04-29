use nra_tensor::MmapTensorReader;
use std::time::Instant;

fn main() {
    let path = "/tmp/dummy.safetensors";

    if !std::path::Path::new(path).exists() {
        println!("Please create {} using PyTorch/safetensors first.", path);
        return;
    }

    println!("📂 Mapping SafeTensors file: {}", path);

    let start = Instant::now();
    let reader = MmapTensorReader::open(path).expect("Failed to map file");

    let infos = reader.inspect();
    let parse_time = start.elapsed();

    println!("✅ Parsed header in {:?} ({} tensors cached)", parse_time, reader.len());
    println!("📊 Tensors:");

    for info in &infos {
        println!(
            "  - {}: shape={:?}, dtype={}, offset={}, size={} bytes",
            info.name, info.shape, info.dtype,
            info.data_offset.0, info.data_offset.1
        );

        // Verify O(1) access to raw bytes (no re-parsing)
        let bytes_start = Instant::now();
        let raw_bytes = reader.get_tensor_bytes(&info.name).unwrap();
        let access_time = bytes_start.elapsed();

        println!(
            "    ⚡ Zero-copy mapped {} bytes in {:?} (Ready for PyTorch .to('cuda'))",
            raw_bytes.len(),
            access_time
        );
    }
}
