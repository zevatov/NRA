import os
import time
import tarfile
import pyarrow as pa
import pyarrow.parquet as pq
import tensorflow as tf
import nra
from tqdm import tqdm

DATA_DIR = "/tmp/nra_ultimate_data"
OUT_DIR = "/tmp/nra_ultimate_benchmarks"
os.makedirs(OUT_DIR, exist_ok=True)

DATASETS = {
    "A_Vision": os.path.join(DATA_DIR, "dataset_a_vision"),
    "B_Dedup": os.path.join(DATA_DIR, "dataset_b_duplication"),
    "C_Multi": os.path.join(DATA_DIR, "dataset_c_multimodal"),
}

results = []

def get_dir_size(path):
    total = 0
    for root, dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return total

def pack_tar(name, in_dir):
    out_file = os.path.join(OUT_DIR, f"{name}.tar.gz")
    t0 = time.time()
    with tarfile.open(out_file, "w:gz") as tar:
        tar.add(in_dir, arcname=".")
    return time.time() - t0, os.path.getsize(out_file)

def pack_tfrecord(name, in_dir):
    out_file = os.path.join(OUT_DIR, f"{name}.tfrecord")
    t0 = time.time()
    with tf.io.TFRecordWriter(out_file) as writer:
        for root, dirs, files in os.walk(in_dir):
            for f in sorted(files):
                with open(os.path.join(root, f), "rb") as fp:
                    data = fp.read()
                feature = {
                    'filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[f.encode('utf-8')])),
                    'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))
                }
                example = tf.train.Example(features=tf.train.Features(feature=feature))
                writer.write(example.SerializeToString())
    return time.time() - t0, os.path.getsize(out_file)

def pack_parquet(name, in_dir):
    out_file = os.path.join(OUT_DIR, f"{name}.parquet")
    t0 = time.time()
    names, data_col = [], []
    for root, dirs, files in os.walk(in_dir):
        for f in sorted(files):
            names.append(f)
            with open(os.path.join(root, f), "rb") as fp:
                data_col.append(fp.read())
                
    table = pa.Table.from_arrays([pa.array(names), pa.array(data_col)], names=['filename', 'data'])
    pq.write_table(table, out_file, compression='snappy')
    return time.time() - t0, os.path.getsize(out_file)

def pack_nra(name, in_dir):
    out_file = os.path.join(OUT_DIR, f"{name}.nra")
    t0 = time.time()
    # Call the Rust CLI
    import subprocess
    cmd = ["cargo", "run", "--release", "--manifest-path", "/Users/stanislav/Desktop/NAP/nra/nra-cli/Cargo.toml", "--", "pack-beta", "--input", in_dir, "--output", out_file]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return time.time() - t0, os.path.getsize(out_file)

print("🏆 Начинаем упаковку форматов (Tar.gz, TFRecord, Parquet, NRA)")

for ds_name, ds_path in DATASETS.items():
    print(f"\n📦 Обработка: {ds_name} (Raw size: {get_dir_size(ds_path)/1e6:.2f} MB)")
    
    time_tar, size_tar = pack_tar(ds_name, ds_path)
    print(f"  Tar.gz:   {size_tar/1e6:7.2f} MB | {time_tar:.2f} sec")
    
    time_tf, size_tf = pack_tfrecord(ds_name, ds_path)
    print(f"  TFRecord: {size_tf/1e6:7.2f} MB | {time_tf:.2f} sec")
    
    time_pq, size_pq = pack_parquet(ds_name, ds_path)
    print(f"  Parquet:  {size_pq/1e6:7.2f} MB | {time_pq:.2f} sec")
    
    time_nra, size_nra = pack_nra(ds_name, ds_path)
    print(f"  NRA Beta: {size_nra/1e6:7.2f} MB | {time_nra:.2f} sec")
    
    results.append({
        "dataset": ds_name,
        "raw_size": get_dir_size(ds_path),
        "tar": {"size": size_tar, "time": time_tar},
        "tfrecord": {"size": size_tf, "time": time_tf},
        "parquet": {"size": size_pq, "time": time_pq},
        "nra": {"size": size_nra, "time": time_nra},
    })

import json
with open(os.path.join(OUT_DIR, "packing_results.json"), "w") as f:
    json.dump(results, f, indent=2)

print("\n🎉 Упаковка завершена! Результаты сохранены в packing_results.json")
