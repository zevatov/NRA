use flate2::Compression;
use flate2::write::GzEncoder;
use nra_core::{NraReader, NraWriter};
use rand::{Rng, distributions::Alphanumeric};
use std::fs::{self, File};
use std::io::{Read, Write};
use std::path::Path;
use std::time::Instant;
use zip::write::SimpleFileOptions;

const NUM_FILES: usize = 2000;
const FILE_SIZE: usize = 4096; // 4 KB each
const READS: usize = 100;

fn generate_data(dir: &Path) {
    println!(
        "Generating {} files ({} KB each) in {}...",
        NUM_FILES,
        FILE_SIZE / 1024,
        dir.display()
    );
    fs::create_dir_all(dir).unwrap();

    let mut rng = rand::thread_rng();
    for i in 0..NUM_FILES {
        let content: String = (0..FILE_SIZE)
            .map(|_| rng.sample(Alphanumeric) as char)
            .collect();
        fs::write(dir.join(format!("file_{:04}.txt", i)), content).unwrap();
    }
}

fn build_zip(in_dir: &Path, out_file: &Path) {
    let file = File::create(out_file).unwrap();
    let mut zip = zip::ZipWriter::new(file);
    let options = SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);

    for i in 0..NUM_FILES {
        let name = format!("file_{:04}.txt", i);
        zip.start_file(&name, options).unwrap();
        let data = fs::read(in_dir.join(&name)).unwrap();
        zip.write_all(&data).unwrap();
    }
    zip.finish().unwrap();
}

fn build_tar_gz(in_dir: &Path, out_file: &Path) {
    let file = File::create(out_file).unwrap();
    let enc = GzEncoder::new(file, Compression::default());
    let mut tar = tar::Builder::new(enc);

    for i in 0..NUM_FILES {
        let name = format!("file_{:04}.txt", i);
        tar.append_path_with_name(in_dir.join(&name), &name)
            .unwrap();
    }
    tar.finish().unwrap();
}

fn build_nra(in_dir: &Path, out_file: &Path) {
    let mut writer = NraWriter::new();
    writer.set_name("Benchmark");
    for i in 0..NUM_FILES {
        let name = format!("file_{:04}.txt", i);
        let data = fs::read(in_dir.join(&name)).unwrap();
        writer.add_file(&name, &data).unwrap();
    }
    writer.save(out_file).unwrap();
}

fn bench_fs(dir: &Path, targets: &[String]) -> u128 {
    let start = Instant::now();
    for name in targets {
        let _data = fs::read(dir.join(name)).unwrap();
    }
    start.elapsed().as_millis()
}

fn bench_zip(file: &Path, targets: &[String]) -> u128 {
    let start = Instant::now();
    let f = File::open(file).unwrap();
    let mut archive = zip::ZipArchive::new(f).unwrap();
    for name in targets {
        let mut file = archive.by_name(name).unwrap();
        let mut buf = Vec::new();
        file.read_to_end(&mut buf).unwrap();
    }
    start.elapsed().as_millis()
}

fn bench_tar_gz(file: &Path, targets: &[String]) -> u128 {
    let start = Instant::now();
    // tar.gz has NO random access. We have to parse from the beginning for EVERY file,
    // or parse once and keep it in memory (but that's unpacking the whole thing).
    // Let's simulate a naive random access (open, scan, read)
    for name in targets {
        let f = File::open(file).unwrap();
        let dec = flate2::read::GzDecoder::new(f);
        let mut archive = tar::Archive::new(dec);
        for entry in archive.entries().unwrap() {
            let mut e = entry.unwrap();
            if e.path().unwrap().to_string_lossy() == *name {
                let mut buf = Vec::new();
                e.read_to_end(&mut buf).unwrap();
                break;
            }
        }
    }
    start.elapsed().as_millis()
}

fn bench_nra(file: &Path, targets: &[String]) -> u128 {
    let start = Instant::now();
    let mut reader = NraReader::open(file).unwrap();
    for name in targets {
        let _data = reader.read_file(name).unwrap();
    }
    start.elapsed().as_millis()
}

fn main() {
    let dir = Path::new("/tmp/nra_bench_data");
    let zip_file = Path::new("/tmp/nra_bench.zip");
    let tar_file = Path::new("/tmp/nra_bench.tar.gz");
    let nra_file = Path::new("/tmp/nra_bench.nra");

    if !dir.exists() {
        generate_data(dir);
        println!("Building ZIP...");
        build_zip(dir, zip_file);
        println!("Building TAR.GZ...");
        build_tar_gz(dir, tar_file);
        println!("Building NRA...");
        build_nra(dir, nra_file);
    }

    println!("Original Dir Size: {} bytes", NUM_FILES * FILE_SIZE);
    println!("ZIP Size: {} bytes", fs::metadata(zip_file).unwrap().len());
    println!(
        "TAR.GZ Size: {} bytes",
        fs::metadata(tar_file).unwrap().len()
    );
    println!("NRA Size: {} bytes", fs::metadata(nra_file).unwrap().len());

    let mut rng = rand::thread_rng();
    let targets: Vec<String> = (0..READS)
        .map(|_| format!("file_{:04}.txt", rng.gen_range(0..NUM_FILES)))
        .collect();

    println!("\n=== BENCHMARK: Randomly reading {} files ===", READS);

    let fs_time = bench_fs(dir, &targets);
    println!("Raw Filesystem: {} ms", fs_time);

    let zip_time = bench_zip(zip_file, &targets);
    println!("ZIP Archive: {} ms", zip_time);

    let nra_time = bench_nra(nra_file, &targets);
    println!("NRA Archive: {} ms", nra_time);

    let tar_time = bench_tar_gz(tar_file, &targets);
    println!("TAR.GZ Archive: {} ms (O(N) scan)", tar_time);

    // Save to JSON
    let json = serde_json::json!({
        "reads": READS,
        "fs_ms": fs_time,
        "zip_ms": zip_time,
        "nra_ms": nra_time,
        "tar_ms": tar_time,
        "nra_size": fs::metadata(nra_file).unwrap().len(),
        "zip_size": fs::metadata(zip_file).unwrap().len(),
        "tar_size": fs::metadata(tar_file).unwrap().len(),
    });
    fs::write("/tmp/nra_bench_results.json", json.to_string()).unwrap();
}
