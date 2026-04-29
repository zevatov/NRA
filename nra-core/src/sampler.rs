//! Deterministic Sampler for Distributed Training.
//!
//! Guarantees mathematically identical sample ordering across any number
//! of GPUs/nodes/workers. This is critical for:
//! - Reproducibility of experiments
//! - Debugging loss spikes
//! - Mid-epoch resumption
//!
//! Implements the same algorithm as MosaicML Streaming's Elastic Determinism,
//! but integrated with NRA's CDC-deduplicated chunk-addressed storage.

/// A deterministic sampler that produces the same permutation regardless
/// of the number of workers.
///
/// Algorithm: Fisher-Yates shuffle with a seeded PRNG (SplitMix64).
/// Given (seed, epoch), the permutation is fully determined.
pub struct DeterministicSampler {
    seed: u64,
    dataset_size: usize,
}

impl DeterministicSampler {
    pub fn new(seed: u64, dataset_size: usize) -> Self {
        Self { seed, dataset_size }
    }

    /// Generate the full permutation for a given epoch.
    /// The result is deterministic: same (seed, epoch, dataset_size) → same order.
    pub fn permutation(&self, epoch: u64) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.dataset_size).collect();
        let mut rng = SplitMix64::new(self.seed.wrapping_add(epoch.wrapping_mul(0x9E3779B97F4A7C15)));

        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = (rng.next() as usize) % (i + 1);
            indices.swap(i, j);
        }

        indices
    }

    /// Get the shard of indices for a specific worker in distributed training.
    ///
    /// # Arguments
    /// * `epoch` - Current epoch number
    /// * `rank` - This worker's rank (0-indexed)
    /// * `world_size` - Total number of workers
    ///
    /// Each worker gets a non-overlapping slice. No sample is processed twice.
    pub fn shard(&self, epoch: u64, rank: usize, world_size: usize) -> Vec<usize> {
        let perm = self.permutation(epoch);
        perm.into_iter()
            .skip(rank)
            .step_by(world_size)
            .collect()
    }
}

/// SplitMix64: A fast, high-quality 64-bit PRNG.
/// Used because it's deterministic, portable, and has no platform-dependent behavior.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

/// Checkpoint state for mid-epoch resumption.
///
/// Serializable to JSON. Store this alongside your model checkpoint.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DataLoaderCheckpoint {
    pub epoch: u64,
    pub batch_index: u64,
    pub seed: u64,
    pub dataset_size: usize,
    pub world_size: usize,
    pub rank: usize,
}

impl DataLoaderCheckpoint {
    pub fn new(seed: u64, dataset_size: usize, epoch: u64, batch_index: u64, world_size: usize, rank: usize) -> Self {
        Self { epoch, batch_index, seed, dataset_size, world_size, rank }
    }

    /// Serialize to JSON bytes.
    pub fn save(&self) -> Result<Vec<u8>, std::io::Error> {
        serde_json::to_vec_pretty(self)
            .map_err(std::io::Error::other)
    }

    /// Deserialize from JSON bytes.
    pub fn load(data: &[u8]) -> Result<Self, std::io::Error> {
        serde_json::from_slice(data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
    }

    /// Get the remaining indices for this worker, skipping already-processed batches.
    pub fn remaining_indices(&self, batch_size: usize) -> Vec<usize> {
        let sampler = DeterministicSampler::new(self.seed, self.dataset_size);
        let shard = sampler.shard(self.epoch, self.rank, self.world_size);
        let skip = (self.batch_index as usize) * batch_size;
        shard.into_iter().skip(skip).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_across_calls() {
        let s = DeterministicSampler::new(42, 1000);
        let p1 = s.permutation(0);
        let p2 = s.permutation(0);
        assert_eq!(p1, p2, "Same seed+epoch must produce identical permutation");
    }

    #[test]
    fn different_epochs() {
        let s = DeterministicSampler::new(42, 1000);
        let p0 = s.permutation(0);
        let p1 = s.permutation(1);
        assert_ne!(p0, p1, "Different epochs should produce different permutations");
    }

    #[test]
    fn shards_cover_all_indices() {
        let s = DeterministicSampler::new(42, 100);
        let world_size = 4;
        let mut all: Vec<usize> = Vec::new();
        for rank in 0..world_size {
            all.extend(s.shard(0, rank, world_size));
        }
        all.sort();
        let expected: Vec<usize> = (0..100).collect();
        assert_eq!(all, expected, "All shards combined must cover every index exactly once");
    }

    #[test]
    fn checkpoint_roundtrip() {
        let ckpt = DataLoaderCheckpoint::new(42, 1000, 5, 37, 4, 2);
        let bytes = ckpt.save().unwrap();
        let loaded = DataLoaderCheckpoint::load(&bytes).unwrap();
        assert_eq!(loaded.epoch, 5);
        assert_eq!(loaded.batch_index, 37);
        assert_eq!(loaded.rank, 2);
    }

    #[test]
    fn resumption_skips_processed() {
        let ckpt = DataLoaderCheckpoint::new(42, 100, 0, 5, 1, 0);
        let remaining = ckpt.remaining_indices(10); // 5 batches of 10 = 50 processed
        assert_eq!(remaining.len(), 50, "Should have 50 remaining after skipping 50");
    }
}
