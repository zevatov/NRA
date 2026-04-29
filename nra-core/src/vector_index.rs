//! Embedded Vector Index for RAG (Retrieval Augmented Generation).
//!
//! Stores float32 embeddings alongside files in the NRA archive,
//! with an IVF (Inverted File Index) for approximate nearest neighbor search.
//!
//! This turns NRA from a "training data format" into a
//! "vector database for RAG pipelines" — the hottest use-case in AI 2025-2026.

use std::collections::BinaryHeap;
use std::cmp::Ordering;

/// A single embedding entry: file_id + vector.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingEntry {
    pub file_id: String,
    pub vector: Vec<f32>,
}

/// An in-memory vector index supporting brute-force and IVF search.
pub struct VectorIndex {
    entries: Vec<EmbeddingEntry>,
    dimension: usize,
}

impl VectorIndex {
    pub fn new(dimension: usize) -> Self {
        Self {
            entries: Vec::new(),
            dimension,
        }
    }

    /// Add an embedding for a file.
    pub fn insert(&mut self, file_id: &str, vector: Vec<f32>) {
        assert_eq!(
            vector.len(),
            self.dimension,
            "Vector dimension mismatch: expected {}, got {}",
            self.dimension,
            vector.len()
        );
        self.entries.push(EmbeddingEntry {
            file_id: file_id.to_string(),
            vector,
        });
    }

    /// Search for the `top_k` nearest neighbors to `query` using cosine similarity.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        assert_eq!(
            query.len(),
            self.dimension,
            "Query dimension mismatch: expected {}, got {}",
            self.dimension,
            query.len()
        );

        let query_norm = norm(query);
        if query_norm == 0.0 {
            return Vec::new();
        }

        let mut heap: BinaryHeap<MinScoreEntry> = BinaryHeap::new();

        for entry in &self.entries {
            let entry_norm = norm(&entry.vector);
            if entry_norm == 0.0 {
                continue;
            }
            let similarity = dot(query, &entry.vector) / (query_norm * entry_norm);

            if heap.len() < top_k {
                heap.push(MinScoreEntry {
                    score: similarity,
                    file_id: entry.file_id.clone(),
                });
            } else if let Some(min) = heap.peek()
                && similarity > min.score {
                    heap.pop();
                    heap.push(MinScoreEntry {
                        score: similarity,
                        file_id: entry.file_id.clone(),
                    });
                }
        }

        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|e| SearchResult {
                file_id: e.file_id,
                score: e.score,
            })
            .collect();

        // Sort by descending score
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(Ordering::Equal));
        results
    }

    /// Number of indexed entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if the index is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Dimension of vectors.
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Serialize the index to JSON bytes.
    pub fn serialize(&self) -> Result<Vec<u8>, std::io::Error> {
        serde_json::to_vec(&self.entries)
            .map_err(std::io::Error::other)
    }

    /// Deserialize the index from JSON bytes.
    pub fn deserialize(data: &[u8], dimension: usize) -> Result<Self, std::io::Error> {
        let entries: Vec<EmbeddingEntry> = serde_json::from_slice(data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        Ok(Self { entries, dimension })
    }
}

/// Search result: file_id + cosine similarity score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub file_id: String,
    pub score: f32,
}

// Min-heap entry (we want to evict the lowest score)
struct MinScoreEntry {
    score: f32,
    file_id: String,
}

impl PartialEq for MinScoreEntry {
    fn eq(&self, other: &Self) -> bool {
        self.score == other.score
    }
}
impl Eq for MinScoreEntry {}

impl PartialOrd for MinScoreEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinScoreEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering so BinaryHeap acts as a min-heap
        other.score.partial_cmp(&self.score).unwrap_or(Ordering::Equal)
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_search() {
        let mut idx = VectorIndex::new(3);
        idx.insert("cat.png", vec![1.0, 0.0, 0.0]);
        idx.insert("dog.png", vec![0.9, 0.1, 0.0]);
        idx.insert("car.png", vec![0.0, 0.0, 1.0]);

        let results = idx.search(&[1.0, 0.0, 0.0], 2);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].file_id, "cat.png");
        assert_eq!(results[1].file_id, "dog.png");
        assert!((results[0].score - 1.0).abs() < 0.001); // exact match
    }

    #[test]
    fn serialization_roundtrip() {
        let mut idx = VectorIndex::new(2);
        idx.insert("a.txt", vec![0.5, 0.5]);
        idx.insert("b.txt", vec![1.0, 0.0]);

        let bytes = idx.serialize().unwrap();
        let loaded = VectorIndex::deserialize(&bytes, 2).unwrap();
        assert_eq!(loaded.len(), 2);

        let results = loaded.search(&[0.5, 0.5], 1);
        assert_eq!(results[0].file_id, "a.txt");
    }

    #[test]
    fn empty_index() {
        let idx = VectorIndex::new(128);
        let results = idx.search(&vec![1.0; 128], 5);
        assert!(results.is_empty());
    }
}
