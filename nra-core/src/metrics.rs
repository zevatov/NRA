//! Runtime metrics for NRA operations.
//!
//! All counters are lock-free `AtomicU64` — safe for multi-threaded
//! PyTorch DataLoader workers without any mutex overhead.

use std::sync::atomic::{AtomicU64, Ordering};

/// Lock-free performance counters for NRA archive operations.
#[derive(Debug)]
pub struct Metrics {
    pub chunks_read: AtomicU64,
    pub bytes_streamed: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub blocks_decompressed: AtomicU64,
    pub files_read: AtomicU64,
    pub bytes_written: AtomicU64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            chunks_read: AtomicU64::new(0),
            bytes_streamed: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            blocks_decompressed: AtomicU64::new(0),
            files_read: AtomicU64::new(0),
            bytes_written: AtomicU64::new(0),
        }
    }

    pub fn record_chunk_read(&self, bytes: u64) {
        self.chunks_read.fetch_add(1, Ordering::Relaxed);
        self.bytes_streamed.fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn record_cache_hit(&self) {
        self.cache_hits.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_cache_miss(&self) {
        self.cache_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_block_decompressed(&self) {
        self.blocks_decompressed.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_file_read(&self) {
        self.files_read.fetch_add(1, Ordering::Relaxed);
    }

    pub fn record_bytes_written(&self, bytes: usize) {
        self.bytes_written.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// Take a snapshot of all counters for export (e.g., to Python dict or Prometheus).
    pub fn snapshot(&self) -> MetricsSnapshot {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let total = hits + misses;
        MetricsSnapshot {
            chunks_read: self.chunks_read.load(Ordering::Relaxed),
            bytes_streamed: self.bytes_streamed.load(Ordering::Relaxed),
            cache_hits: hits,
            cache_misses: misses,
            cache_hit_rate: if total > 0 { hits as f64 / total as f64 } else { 0.0 },
            blocks_decompressed: self.blocks_decompressed.load(Ordering::Relaxed),
            files_read: self.files_read.load(Ordering::Relaxed),
            bytes_written: self.bytes_written.load(Ordering::Relaxed),
        }
    }
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

/// Immutable snapshot of metrics at a point in time.
#[derive(Debug, Clone)]
pub struct MetricsSnapshot {
    pub chunks_read: u64,
    pub bytes_streamed: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub cache_hit_rate: f64,
    pub blocks_decompressed: u64,
    pub files_read: u64,
    pub bytes_written: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn metrics_basic() {
        let m = Metrics::new();
        m.record_chunk_read(4096);
        m.record_chunk_read(8192);
        m.record_cache_hit();
        m.record_cache_hit();
        m.record_cache_miss();
        m.record_file_read();

        let s = m.snapshot();
        assert_eq!(s.chunks_read, 2);
        assert_eq!(s.bytes_streamed, 4096 + 8192);
        assert_eq!(s.cache_hits, 2);
        assert_eq!(s.cache_misses, 1);
        assert!((s.cache_hit_rate - 0.6667).abs() < 0.01);
        assert_eq!(s.files_read, 1);
    }
}
