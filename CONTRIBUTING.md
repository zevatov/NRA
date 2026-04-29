# Contributing to NRA

Thank you for your interest in contributing to NRA! 🧬

## Getting Started

### Prerequisites

- **Rust 1.80+** — [Install Rust](https://rustup.rs/)
- **Python 3.10+** — For the Python SDK
- **libfuse-dev** (Linux) or **macFUSE** (macOS) — For FUSE mount support

### Building from Source

```bash
git clone https://github.com/zevatov/NRA.git
cd NRA

# Build everything
cargo build --release

# Run tests
cargo test --workspace --exclude nra-python --exclude nra-gui

# Build the CLI
cargo build --release -p nra-cli
```

### Building the Python SDK

```bash
cd nra-python
pip install maturin
maturin develop --release
```

## Project Structure

```
NRA/
├── nra-spec/          # Binary format specification (FlatBuffers)
├── nra-core/          # Core library: compression, dedup, manifest, crypto
├── nra-cli/           # CLI tool: pack, extract, convert, stream, mount
├── nra-python/        # Python SDK (PyO3 bindings)
├── nra-registry/      # HTTP streaming client for cloud archives
├── nra-tensor/        # Tensor/SafeTensors integration
├── bench_tool/        # Benchmarking tools
├── docs/              # Whitepapers, reports, asset images
└── examples/          # Example scripts
```

## How to Contribute

### Bug Reports & Feature Requests

Use [GitHub Issues](https://github.com/zevatov/NRA/issues) with our templates.

### Code Contributions

1. **Fork** the repository
2. Create a **feature branch** (`git checkout -b feat/my-feature`)
3. Make your changes
4. Run tests: `cargo test --workspace --exclude nra-python --exclude nra-gui`
5. Submit a **Pull Request**

### Areas We Need Help With

- 🐧 **Linux/Windows testing** — We develop primarily on macOS
- 📊 **Benchmarks** — More datasets, more comparisons
- 🌍 **Documentation** — English translations, tutorials
- 🔌 **Integrations** — TensorFlow, JAX, HuggingFace `datasets` library

## Code Style

- Rust: Follow `rustfmt` defaults (`cargo fmt`)
- Python: Follow PEP 8
- Commits: Use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`)

## License

By contributing, you agree that your contributions will be licensed under the **MIT License**.
