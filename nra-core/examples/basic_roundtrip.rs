use nra_core::{NraReader, NraWriter};

fn main() -> std::io::Result<()> {
    println!("=== NRA Roundtrip Example ===\n");

    // 1. Pack
    let path = "/tmp/example.nra";
    let mut writer = NraWriter::new();
    writer.set_name("Example Dataset");
    writer.add_file("hello.txt", b"Hello, Neural Ready Archive!")?;
    writer.add_file("data.bin", &vec![0xAA; 1024])?;
    writer.save(path)?;
    println!("✅ Packed archive → {}", path);

    // 2. Unpack
    let mut reader = NraReader::open(path)?;
    println!(
        "📦 Archive: '{}', {} files",
        reader.manifest().summary.name,
        reader.manifest().summary.total_files
    );
    for id in reader.file_ids() {
        println!("  - {}", id);
    }

    let content = reader.read_file("hello.txt")?;
    println!(
        "\n📄 hello.txt contents: {}",
        String::from_utf8_lossy(&content)
    );

    Ok(())
}
