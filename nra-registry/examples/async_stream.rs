use nra_core::AsyncBetaReader;
use nra_registry::HttpRandomAccess;
use std::time::Instant;

#[tokio::main]
async fn main() {
    let url = "http://localhost:8000/heavy_beta.nra";
    println!("📡 Connecting to {}", url);

    let streamer = HttpRandomAccess::new(url);
    
    let start_time = Instant::now();
    let reader_result = AsyncBetaReader::open(streamer).await;
    
    match reader_result {
        Ok(reader) => {
            let manifest_time = start_time.elapsed();
            println!("✅ Successfully fetched and parsed Manifest in {:?}", manifest_time);
            
            let files = reader.file_ids();
            println!("📦 Archive contains {} files", files.len());
            
            if !files.is_empty() {
                // Fetch a file from the middle of the archive
                let target_file = files[files.len() / 2];
                println!("🎯 Fetching file: {}", target_file);
                
                let fetch_start = Instant::now();
                let data = reader.read_file(target_file).await.expect("Failed to read file");
                let fetch_time = fetch_start.elapsed();
                
                println!("🚀 Successfully fetched {} bytes in {:?}", data.len(), fetch_time);
                println!("👀 Preview: {:?}", String::from_utf8_lossy(&data[..data.len().min(50)]));
            }
        }
        Err(e) => {
            eprintln!("❌ Failed to open archive: {}", e);
        }
    }
}
