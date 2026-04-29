use indicatif::{ProgressBar, ProgressStyle};
use nra_registry::HttpReader;
use std::error::Error;
use std::time::Instant;

fn main() -> Result<(), Box<dyn Error>> {
    let url = "http://127.0.0.1:3000/archives/demo.nra";
    println!("🚀 Starting NRA Cloud Streaming Demo!");
    println!("🌐 Target: {}\n", url);
    
    // 1. Instantly read the manifest via HTTP Range
    let start_time = Instant::now();
    let reader = HttpReader::open(url)?;
    println!("✅ Connected & Manifest loaded in {:?}", start_time.elapsed());
    
    let files = reader.file_ids();
    println!("📦 Archive contains {} files", files.len());
    println!("☁️  Streaming files over HTTP Range Requests (Zero local storage)...\n");
    
    let pb = ProgressBar::new(files.len() as u64);
    pb.set_style(ProgressStyle::default_bar()
        .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} files ({eta})")
        .unwrap()
        .progress_chars("#>-"));

    let fetch_start = Instant::now();
    let mut total_bytes = 0;
    
    for file_id in files.iter() {
        // Fetch only the required compressed block via HTTP Range
        let data = reader.read_file(file_id)?;
        total_bytes += data.len();
        pb.inc(1);
    }
    
    pb.finish_with_message("Done!");
    
    let duration = fetch_start.elapsed();
    println!("\n🎉 Streaming Complete!");
    println!("📊 Total Decompressed Data: {} bytes", total_bytes);
    println!("⏱️  Total Time: {:?}", duration);
    println!("⚡ Throughput: {:.2} files/sec", files.len() as f64 / duration.as_secs_f64());
    
    Ok(())
}
