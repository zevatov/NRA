//! HTTP Range Streaming implementation for NRA.
//!
//! Provides `HttpRandomAccess`, which translates `read_at(offset, len)` calls
//! into HTTP Range requests (`Range: bytes=X-Y`).

use nra_core::AsyncRandomAccess;
use reqwest::header::{RANGE, CONTENT_LENGTH};
use reqwest::Client;
use std::io;

/// Maximum number of retry attempts for transient HTTP errors (503, timeouts).
const MAX_RETRIES: u32 = 3;

/// Initial backoff duration in milliseconds. Doubles on each retry.
const INITIAL_BACKOFF_MS: u64 = 100;

/// An asynchronous reader that fetches byte ranges over HTTP/2.
pub struct HttpRandomAccess {
    client: Client,
    url: String,
}

impl HttpRandomAccess {
    /// Create a new HTTP streamer.
    pub fn new(url: &str) -> Self {
        Self {
            client: Client::new(),
            url: url.to_string(),
        }
    }
}

impl AsyncRandomAccess for HttpRandomAccess {
    async fn read_at(&self, offset: u64, length: usize) -> io::Result<Vec<u8>> {
        // РЕКОМЕНДАЦИЯ-4 fix: guard against length=0 underflow
        if length == 0 {
            return Ok(Vec::new());
        }

        let end = offset + (length as u64) - 1;
        let range_val = format!("bytes={}-{}", offset, end);

        // РЕКОМЕНДАЦИЯ-3 fix: exponential backoff retry for transient errors
        let mut last_err = None;
        for attempt in 0..MAX_RETRIES {
            if attempt > 0 {
                let backoff = INITIAL_BACKOFF_MS * (1 << (attempt - 1));
                tokio::time::sleep(std::time::Duration::from_millis(backoff)).await;
            }

            let res = match self
                .client
                .get(&self.url)
                .header(RANGE, &range_val)
                .send()
                .await
            {
                Ok(r) => r,
                Err(e) => {
                    last_err = Some(io::Error::other(e.to_string()));
                    continue; // retry on connection errors
                }
            };

            let status = res.status();

            // Retry on 503 Service Unavailable (S3 throttling)
            if status == reqwest::StatusCode::SERVICE_UNAVAILABLE {
                last_err = Some(io::Error::other(
                    format!("HTTP 503 (attempt {}/{})", attempt + 1, MAX_RETRIES),
                ));
                continue;
            }

            if status != reqwest::StatusCode::PARTIAL_CONTENT {
                return Err(io::Error::other(
                    format!("HTTP error: expected 206 Partial Content, got {}", status),
                ));
            }

            let bytes = res
                .bytes()
                .await
                .map_err(|e| io::Error::other(e.to_string()))?;

            if bytes.len() != length {
                return Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!(
                        "Incomplete HTTP Range read: expected {}, got {}",
                        length,
                        bytes.len()
                    ),
                ));
            }

            return Ok(bytes.to_vec());
        }

        Err(last_err.unwrap_or_else(|| {
            io::Error::other("HTTP request failed after all retries")
        }))
    }

    async fn size(&self) -> io::Result<u64> {
        let res = self
            .client
            .head(&self.url)
            .send()
            .await
            .map_err(|e| io::Error::other(e.to_string()))?;

        if !res.status().is_success() {
            return Err(io::Error::other(
                format!("HTTP HEAD error: {}", res.status()),
            ));
        }

        let len_str = res
            .headers()
            .get(CONTENT_LENGTH)
            .and_then(|val| val.to_str().ok())
            .ok_or_else(|| {
                io::Error::other("Missing Content-Length header")
            })?;

        len_str
            .parse::<u64>()
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Invalid Content-Length"))
    }
}
