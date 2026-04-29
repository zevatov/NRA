# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | ✅ Active support  |
| < 1.0   | ❌ Not supported   |

## Reporting a Vulnerability

If you discover a security vulnerability in NRA, please report it responsibly:

1. **DO NOT** open a public GitHub Issue
2. Email: **[security contact — add your email here]**
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

We will acknowledge receipt within **48 hours** and aim to release a fix within **7 days** for critical vulnerabilities.

## Security Features

NRA includes built-in security mechanisms:

- **AES-256-GCM** encryption for archive contents
- **SHA-256** checksums for data integrity verification
- **Content-Defined Chunking (CDC)** with cryptographic hashing for deduplication
