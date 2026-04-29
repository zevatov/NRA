use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use nra_core::dedup::chunk_data;
use nra_core::BetaWriter;
use rayon::prelude::*;
use std::fs::File;
use std::io::Read;
use std::path::Path;

pub fn convert_archive(input: &Path, output: &Path, archive_name: &str, codec_str: &str, zstd_level: i32) -> Result<()> {
    let mut writer = BetaWriter::new();
    writer.set_name(archive_name);
    
    let codec = match codec_str.to_lowercase().as_str() {
        "lz4" => nra_core::codec::Codec::Lz4,
        "zstd" => nra_core::codec::Codec::Zstd,
        _ => anyhow::bail!("Invalid codec: {}", codec_str),
    };
    writer.set_codec(codec);
    writer.set_zstd_level(zstd_level);

    println!("🔄 Converting archive: {:?}", input);
    println!("📦 Output NRA: {:?}", output);

    let name = input.to_string_lossy().to_lowercase();

    if name.ends_with(".tar.gz") || name.ends_with(".tgz") || name.ends_with(".tar") {
        convert_tar(input, &mut writer)?;
    } else if name.ends_with(".zip") {
        convert_zip(input, &mut writer)?;
    } else {
        anyhow::bail!("Unsupported archive format. Supported: .tar, .tar.gz, .tgz, .zip");
    }

    // Finalize the archive (compress solid blocks and write manifest)
    writer.print_stats();
    writer.save(output)?;
    
    println!("✅ Successfully converted into {:?} (BETA mode)", output);
    Ok(())
}

fn convert_tar(input: &Path, writer: &mut BetaWriter) -> Result<()> {
    let file = File::open(input).context("Failed to open input archive")?;
    
    let is_gz = input.extension().unwrap_or_default() == "gz" || input.extension().unwrap_or_default() == "tgz";
    
    let mut archive = if is_gz {
        let decoder = GzDecoder::new(file);
        tar::Archive::new(Box::new(decoder) as Box<dyn Read>)
    } else {
        tar::Archive::new(Box::new(file) as Box<dyn Read>)
    };

    let mut batch: Vec<(String, Vec<u8>)> = Vec::new();
    let batch_size = 1000;

    for entry in archive.entries()? {
        let mut entry = entry?;
        if entry.header().entry_type().is_file() {
            let path = entry.path()?.to_string_lossy().to_string();
            let mut data = Vec::new();
            entry.read_to_end(&mut data)?;
            
            batch.push((path, data));

            if batch.len() >= batch_size {
                process_batch(writer, &mut batch);
            }
        }
    }

    if !batch.is_empty() {
        process_batch(writer, &mut batch);
    }

    Ok(())
}

fn convert_zip(input: &Path, writer: &mut BetaWriter) -> Result<()> {
    let file = File::open(input).context("Failed to open input zip")?;
    let mut archive = zip::ZipArchive::new(file)?;

    let mut batch: Vec<(String, Vec<u8>)> = Vec::new();
    let batch_size = 1000;

    for i in 0..archive.len() {
        let mut entry = archive.by_index(i)?;
        if entry.is_file() {
            let path = entry.name().to_string();
            let mut data = Vec::new();
            entry.read_to_end(&mut data)?;

            batch.push((path, data));

            if batch.len() >= batch_size {
                process_batch(writer, &mut batch);
            }
        }
    }

    if !batch.is_empty() {
        process_batch(writer, &mut batch);
    }

    Ok(())
}

fn process_batch(writer: &mut BetaWriter, batch: &mut Vec<(String, Vec<u8>)>) {
    // 1. Process chunks in parallel using Rayon
    let prechunked: Vec<_> = batch.par_iter()
        .map(|(path, data)| {
            chunk_data(path, data)
        })
        .collect();

    // 2. Ingest sequentially into BetaWriter to preserve determinism
    for (recipe, chunks) in prechunked {
        writer.add_prechunked(recipe, chunks);
    }

    batch.clear();
}
