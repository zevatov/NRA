use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use nra_core::{NraReader, NraWriter, BetaWriter};
use std::fs;
use std::path::{Path, PathBuf};

mod converter;

#[derive(Parser)]
#[command(author, version, about = "Neural Ready Archive (NRA) CLI", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Pack a directory into an .nra archive
    Pack {
        /// Input directory containing files to pack
        #[arg(short, long)]
        input: PathBuf,

        /// Output .nra file path
        #[arg(short, long)]
        output: PathBuf,

        /// Name of the dataset/archive
        #[arg(short, long, default_value = "NRA Dataset")]
        name: String,

        /// Optimization mode: "speed" (default, per-file) or "size" (chunked solid compression)
        #[arg(long, default_value = "speed")]
        optimize_for: String,
    },
    /// Unpack an .nra archive into a directory
    Unpack {
        /// Input .nra archive
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory to extract files into
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Display information about an .nra archive
    Info {
        /// Input .nra archive
        #[arg(short, long)]
        input: PathBuf,
    },
    /// Convert an existing dataset archive (.tar, .zip) into NRA format directly
    Convert {
        /// Input archive (.tar.gz, .zip)
        #[arg(short, long)]
        input: PathBuf,

        /// Output .nra file path
        #[arg(short, long)]
        output: PathBuf,

        /// Name of the dataset/archive
        #[arg(short, long, default_value = "Converted Dataset")]
        name: String,

        /// Compression codec to use: "zstd" or "lz4"
        #[arg(long, default_value = "zstd")]
        codec: String,

        /// Zstd compression level (1-22), only used if codec is zstd
        #[arg(long, default_value = "3")]
        zstd_level: i32,
    },
    /// Mount an .nra archive as a local filesystem (requires FUSE)
    Mount {
        /// Input .nra archive
        #[arg(short, long)]
        input: PathBuf,

        /// Mount point (directory)
        #[arg(short, long)]
        mountpoint: PathBuf,
    },
    /// Append new files to an existing .nra BETA archive
    Append {
        /// Input directory containing new files to append
        #[arg(short, long)]
        input: PathBuf,

        /// Target .nra archive to modify in-place
        #[arg(short, long)]
        archive: PathBuf,

        /// Encrypt the archive blocks (uses NRA_ENCRYPTION_KEY env var)
        #[arg(long)]
        encrypt: bool,
    },
    /// Pack a directory into an .nra BETA archive (with Content-Defined Deduplication)
    PackBeta {
        /// Input directory containing files to pack
        #[arg(short, long)]
        input: PathBuf,

        /// Output .nra file path
        #[arg(short, long)]
        output: PathBuf,

        /// Name of the dataset/archive
        #[arg(short, long, default_value = "NRA BETA Dataset")]
        name: String,

        /// Encrypt the archive blocks (uses NRA_ENCRYPTION_KEY env var)
        #[arg(long)]
        encrypt: bool,

        /// Compression codec to use: "zstd" or "lz4"
        #[arg(long, default_value = "zstd")]
        codec: String,

        /// Zstd compression level (1-22), only used if codec is zstd
        #[arg(long, default_value = "3")]
        zstd_level: i32,

        /// Train a Zstd dictionary for better compression of small files
        #[arg(long)]
        dictionary: bool,
    },
    /// Stream a specific file from a remote BETA archive over HTTP
    StreamBeta {
        /// HTTP URL to the .nra archive
        #[arg(short, long)]
        url: String,

        /// File ID to extract
        #[arg(short, long)]
        file_id: String,

        /// Optional output path to save the file
        #[arg(short, long)]
        output: Option<PathBuf>,
    },
    /// Unpack an NRA BETA archive into a directory
    UnpackBeta {
        /// Input .nra BETA archive
        #[arg(short, long)]
        input: PathBuf,

        /// Output directory to extract files into
        #[arg(short, long)]
        output: PathBuf,
    },
    /// Display detailed information about an NRA BETA archive
    InfoBeta {
        /// Input .nra BETA archive
        #[arg(short, long)]
        input: PathBuf,

        /// Show verbose list of all files
        #[arg(long)]
        verbose: bool,
    },
    /// Push a directory to a remote NRA Registry server via tar streaming
    Push {
        /// Input directory containing files to pack
        #[arg(short, long)]
        input: PathBuf,

        /// Server URL for the upload endpoint (e.g. http://127.0.0.1:3000/api/v1/upload/my_dataset)
        #[arg(short, long)]
        url: String,
    },
}

fn pack_dir(input: &Path, output: &Path, name: &str, optimize_for: &str) -> Result<()> {
    if !input.is_dir() {
        anyhow::bail!("Input must be a directory");
    }

    let mode = match optimize_for.to_lowercase().as_str() {
        "size" => nra_core::OptimizationMode::Size,
        "speed" => nra_core::OptimizationMode::Speed,
        _ => anyhow::bail!("Invalid optimization mode. Use 'speed' or 'size'."),
    };

    println!(
        "📦 Packing directory: {} (Mode: {:?})",
        input.display(),
        mode
    );
    let mut writer = NraWriter::new();
    writer.set_mode(mode);
    writer.set_name(name);

    let mut count = 0;
    for entry in fs::read_dir(input)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let file_name = path.file_name().unwrap().to_string_lossy().to_string();
            let data = fs::read(&path)?;
            writer.add_file(&file_name, &data)?;
            count += 1;
        }
    }

    writer.save(output)?;
    println!(
        "✅ Successfully packed {} files into {}",
        count,
        output.display()
    );
    Ok(())
}

fn unpack_archive(input: &Path, output: &Path) -> Result<()> {
    println!("📦 Unpacking archive: {}", input.display());

    if !output.exists() {
        fs::create_dir_all(output)?;
    } else if !output.is_dir() {
        anyhow::bail!("Output must be a directory");
    }

    let mut reader = NraReader::open(input)?;
    let mut count = 0;

    let file_ids: Vec<String> = reader
        .file_ids()
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    for file_id in file_ids {
        let data = reader.read_file(&file_id)?;
        let out_path = output.join(file_id);
        fs::write(&out_path, data)?;
        count += 1;
    }

    println!(
        "✅ Successfully unpacked {} files to {}",
        count,
        output.display()
    );
    Ok(())
}

fn info_archive(input: &Path) -> Result<()> {
    let reader = NraReader::open(input).context("Failed to open archive")?;
    let manifest = reader.manifest();

    println!("=== NRA Archive Info ===");
    println!("Name: {}", manifest.summary.name);
    println!("Version: {}", manifest.version);
    println!("Total files: {}", manifest.summary.total_files);

    println!("\nFiles:");
    for file in &manifest.files {
        println!(
            "  - {} (Compressed: {} bytes, Original: {} bytes, CRC32: {:08X})",
            file.id, file.compressed_size, file.original_size, file.crc32
        );
    }

    Ok(())
}

fn pack_beta(input: &Path, output: &Path, name: &str, encrypt: bool, codec_str: &str, zstd_level: i32, use_dictionary: bool) -> Result<()> {
    if !input.is_dir() {
        anyhow::bail!("Input must be a directory");
    }

    println!("🧬 Packing directory with NRA BETA (CDC Dedup): {}", input.display());

    let mut writer = BetaWriter::new();
    writer.set_name(name);
    
    let codec = match codec_str.to_lowercase().as_str() {
        "lz4" => nra_core::codec::Codec::Lz4,
        "zstd" => nra_core::codec::Codec::Zstd,
        _ => anyhow::bail!("Invalid codec: {}", codec_str),
    };
    writer.set_codec(codec);
    writer.set_zstd_level(zstd_level);

    if encrypt {
        if let Some(key) = nra_core::crypto::key_from_env()? {
            writer.set_encryption_key(key);
            println!("🔒 Encryption enabled.");
        } else {
            anyhow::bail!("--encrypt flag passed, but NRA_ENCRYPTION_KEY environment variable is not set.");
        }
    }

    let mut count = 0;
    let mut paths = Vec::new();
    for entry in fs::read_dir(input)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            paths.push(path);
        }
    }

    use rayon::prelude::*;

    // Process in batches of 1000 to prevent OOM on massive datasets
    for chunk in paths.chunks(1000) {
        let results: Vec<_> = chunk
            .par_iter()
            .map(|path| {
                let file_name = path.file_name().unwrap().to_string_lossy().to_string();
                let data = fs::read(path).unwrap_or_default();
                nra_core::dedup::chunk_data(&file_name, &data)
            })
            .collect();

        for (recipe, chunks) in results {
            writer.add_prechunked(recipe, chunks);
            count += 1;
        }
    }

    writer.print_stats();
    
    if use_dictionary {
        println!("🧠 Training Zstd dictionary...");
        writer.train_dictionary(110 * 1024).unwrap_or_else(|e| println!("⚠️ Failed to train dictionary: {}", e));
    }

    writer.save(output)?;
    println!(
        "✅ Successfully packed {} files into {} (BETA mode)",
        count,
        output.display()
    );
    Ok(())
}

fn stream_beta(url: &str, file_id: &str, output: Option<PathBuf>) -> Result<()> {
    println!("☁️  Streaming file '{}' from: {}", file_id, url);

    let rt = tokio::runtime::Runtime::new()?;
    rt.block_on(async {
        use nra_core::AsyncBetaReader;
        use nra_registry::HttpRandomAccess;
        use std::time::Instant;

        let start_time = Instant::now();
        let streamer = HttpRandomAccess::new(url);
        
        let reader = AsyncBetaReader::open(streamer).await
            .context("Failed to open remote archive. Ensure the URL supports HTTP Range requests.")?;
        
        let manifest_time = start_time.elapsed();
        println!("✅ Manifest fetched in {:?}", manifest_time);

        let data = reader.read_file(file_id).await
            .with_context(|| format!("Failed to read file '{}'", file_id))?;
            
        let fetch_time = start_time.elapsed() - manifest_time;
        println!("🚀 Successfully fetched {} bytes in {:?}", data.len(), fetch_time);

        if let Some(out_path) = output {
            fs::write(&out_path, &data)?;
            println!("💾 Saved to: {}", out_path.display());
        } else {
            println!("👀 Content Preview:\n{}", String::from_utf8_lossy(&data[..data.len().min(500)]));
        }

        Ok::<(), anyhow::Error>(())
    })?;

    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Pack {
            input,
            output,
            name,
            optimize_for,
        } => pack_dir(&input, &output, &name, &optimize_for)?,
        Commands::Unpack { input, output } => unpack_archive(&input, &output)?,
        Commands::Info { input } => info_archive(&input)?,
        Commands::Convert { input, output, name, codec, zstd_level } => {
            converter::convert_archive(&input, &output, &name, &codec, zstd_level)?
        },
        Commands::Mount { input, mountpoint } => mount_archive(&input, &mountpoint)?,
        Commands::Append { input, archive, encrypt } => append_archive(&input, &archive, encrypt)?,
        Commands::PackBeta { input, output, name, encrypt, codec, zstd_level, dictionary } => {
            pack_beta(&input, &output, &name, encrypt, &codec, zstd_level, dictionary)?
        },
        Commands::StreamBeta { url, file_id, output } => stream_beta(&url, &file_id, output)?,
        Commands::UnpackBeta { input, output } => unpack_beta(&input, &output)?,
        Commands::InfoBeta { input, verbose } => info_beta(&input, verbose)?,
        Commands::Push { input, url } => push_directory(&input, &url)?,
    }

    Ok(())
}

fn unpack_beta(input: &Path, output: &Path) -> Result<()> {
    println!("📦 Unpacking BETA archive: {}", input.display());

    if !output.exists() {
        fs::create_dir_all(output).context("Failed to create output directory")?;
    } else if !output.is_dir() {
        anyhow::bail!("Output must be a directory");
    }

    use nra_core::beta_reader::BetaReader;
    let mut reader = BetaReader::open(input).context("Failed to open BETA archive")?;
    let mut count = 0;

    let file_ids = reader.file_ids().to_vec();
    for file_id in file_ids {
        let data = reader.read_file(&file_id).with_context(|| format!("Failed to read file {}", file_id))?;
        let out_path = output.join(&file_id);
        if let Some(parent) = out_path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("Failed to create directory {}", parent.display()))?;
        }
        fs::write(&out_path, data).with_context(|| format!("Failed to write file {}", out_path.display()))?;
        count += 1;
    }

    println!(
        "✅ Successfully unpacked {} files to {}",
        count,
        output.display()
    );
    Ok(())
}

fn info_beta(input: &Path, verbose: bool) -> Result<()> {
    use nra_core::beta_reader::BetaReader;
    let reader = BetaReader::open(input).context("Failed to open BETA archive")?;
    let manifest = reader.manifest();

    println!("=== NRA BETA Archive Info ===");
    println!("Name: {}", manifest.summary.name);
    println!("Version: {}", manifest.version);
    println!("Total files: {}", manifest.files.len());
    
    let total_original: u64 = manifest.files.iter().map(|f| f.original_size).sum();
    let total_stored: u64 = manifest.chunk_table.iter().map(|c| c.compressed_size).sum();
    
    println!("Total chunks: {}", manifest.chunk_table.len());
    println!("Original size: {:.2} MB", total_original as f64 / 1e6);
    println!("Stored size: {:.2} MB", total_stored as f64 / 1e6);
    
    if total_original > 0 {
        let ratio = total_original as f64 / total_stored as f64;
        println!("Compression ratio: {:.2}x", ratio);
    }
    
    println!("Dictionary: {}", if manifest.dictionary.is_some() { "Yes" } else { "No" });
    println!("Encrypted: {}", if manifest.summary.encrypted { "Yes" } else { "No" });
    
    if verbose {
        println!("\nFiles:");
        for file in &manifest.files {
            println!(
                "  - {} (Original: {} bytes, Chunks: {})",
                file.id, file.original_size, file.chunks.len()
            );
        }
    }

    Ok(())
}

fn append_archive(input: &Path, archive: &Path, encrypt: bool) -> Result<()> {
    if !input.is_dir() {
        anyhow::bail!("Input must be a directory");
    }
    if !archive.is_file() {
        anyhow::bail!("Target archive must exist to append to it");
    }

    if encrypt {
        anyhow::bail!("Delta updates with encryption are not yet fully supported through the CLI. Wait for v4.1.");
    }

    println!("➕ Appending new files from {:?} into existing archive {:?}", input, archive);

    let mut count = 0;
    let mut new_files_data = Vec::new();
    
    // Read all files into memory for simplicity (in a real scenario we'd batch this)
    for entry in fs::read_dir(input)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            let file_name = path.file_name().unwrap().to_string_lossy().to_string();
            let data = fs::read(&path).unwrap_or_default();
            new_files_data.push((file_name, data));
            count += 1;
        }
    }

    let slices: Vec<(&str, &[u8])> = new_files_data
        .iter()
        .map(|(id, data)| (id.as_str(), data.as_slice()))
        .collect();

    let new_chunks = nra_core::delta::delta_pack(archive, &slices)?;

    println!(
        "✅ Successfully appended {} files (added {} new unique chunks) to {:?}",
        count, new_chunks, archive
    );
    
    Ok(())
}

fn mount_archive(input: &Path, mountpoint: &Path) -> Result<()> {
    use nra_core::beta_reader::BetaReader;
    use nra_core::fuse::NraFuse;

    if !input.is_file() {
        anyhow::bail!("Input must be a valid .nra file");
    }

    if !mountpoint.exists() {
        fs::create_dir_all(mountpoint)?;
    } else if !mountpoint.is_dir() {
        anyhow::bail!("Mountpoint must be a directory");
    }

    println!("🚀 Mounting NRA archive: {:?}", input);
    println!("📂 Mountpoint: {:?}", mountpoint);
    println!("💡 Press Ctrl+C to unmount and exit.");

    let reader = BetaReader::open(input)?;
    let fuse_fs = NraFuse::new(reader);

    let options = vec![
        fuser::MountOption::RO,
        fuser::MountOption::FSName("nra".to_string()),
        fuser::MountOption::AutoUnmount,
    ];

    fuser::mount2(fuse_fs, mountpoint, &options)?;

    Ok(())
}

fn push_directory(input: &Path, url: &str) -> Result<()> {
    if !input.is_dir() {
        anyhow::bail!("Input must be a directory");
    }
    
    println!("🚀 Pushing directory to {url}...");

    let temp_tar = std::env::current_dir()?.join(format!("nra_push_{}.tar", std::process::id()));
    println!("📦 Creating temporary tar archive at {:?}", temp_tar);
    let tar_file = fs::File::create(&temp_tar)?;
    let mut builder = tar::Builder::new(tar_file);
    builder.append_dir_all(".", input)?;
    builder.into_inner()?;
    println!("✅ Tar archive created.");

    let client = reqwest::blocking::Client::new();
    let tar_data = fs::File::open(&temp_tar)?;
    
    println!("📡 Sending POST request...");
    let res = client.post(url)
        .header("Content-Type", "application/x-tar")
        .body(tar_data)
        .send()?;

    if res.status().is_success() {
        println!("✅ Successfully pushed dataset.");
    } else {
        println!("❌ Server returned error: {}", res.status());
        println!("{}", res.text().unwrap_or_default());
    }

    let _ = fs::remove_file(temp_tar);

    Ok(())
}
