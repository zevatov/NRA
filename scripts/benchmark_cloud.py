import argparse
import time
import statistics
try:
    import nra
except ImportError:
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), "..", "nra-python"))
    import nra

def main():
    parser = argparse.ArgumentParser(description="Benchmark NRA Cloud Streaming")
    parser.add_argument("--url", required=True, help="HTTP URL to .nra archive")
    parser.add_argument("--num-files", type=int, default=100, help="Number of files to stream for throughput test")
    args = parser.parse_args()

    print(f"🚀 Benchmarking NRA Cloud Streaming against {args.url}")
    print("-" * 50)

    # 1. Measure TTFB (Time To First Byte)
    start_time = time.time()
    archive = nra.CloudArchive(args.url)
    file_ids = archive.file_ids()
    ttfb = time.time() - start_time
    print(f"⏱️  TTFB (Manifest Download & Parse): {ttfb * 1000:.2f} ms")
    print(f"📦 Total files in archive: {len(file_ids)}")

    # 2. Measure Throughput
    num_to_test = min(args.num_files, len(file_ids))
    if num_to_test == 0:
        print("Archive is empty!")
        return

    print(f"\n🏃 Streaming {num_to_test} files...")
    start_time = time.time()
    latencies = []
    total_bytes = 0

    for i in range(num_to_test):
        file_id = file_ids[i]
        t0 = time.time()
        data = archive.read_file(file_id)
        t1 = time.time()
        
        latencies.append((t1 - t0) * 1000)
        total_bytes += len(data)

    total_time = time.time() - start_time
    files_per_sec = num_to_test / total_time
    mb_per_sec = (total_bytes / 1024 / 1024) / total_time

    print("\n📊 Results:")
    print(f"Throughput: {files_per_sec:.2f} files/sec ({mb_per_sec:.2f} MB/s)")
    print(f"Average Latency: {statistics.mean(latencies):.2f} ms/file")
    print(f"p50 Latency: {statistics.median(latencies):.2f} ms/file")
    if len(latencies) >= 100:
        p99 = sorted(latencies)[int(len(latencies) * 0.99)]
        print(f"p99 Latency: {p99:.2f} ms/file")

    print("\n🆚 Baseline Test (Raw HTTP Download of the whole file)")
    print("Skipping raw baseline to save bandwidth, but you can use 'curl' to compare raw throughput.")

if __name__ == "__main__":
    main()
