# sonr

`sonr` is a high-performance semantic search tool for local codebases. It consists of a background daemon that manages LLM inference and a CLI for performing fast, context-aware searches.

## Features

- **Semantic Search**: Find code by meaning, not just keywords.
- **GPU Accelerated**: Uses `llama.cpp` (via `llama-server`) for fast embedding and reranking.
- **Automatic Model Management**: Downloads required GGUF models from Hugging Face automatically.
- **Persistent Caching**: Embeddings are cached to disk to make subsequent searches near-instant.
- **Robustness**: Implements a "death pact" (PDEATHSIG) to ensure background processes are cleaned up if the daemon exits.

## Architecture

- **sonr-daemon**: A background service that manages two `llama-server` instances (one for embeddings, one for reranking) and provides a REST API.
- **sonr-cli**: A lightweight proxy that sends search queries to the daemon and displays results.

## Installation

### Prerequisites

- `llama-server` must be installed and available in your `PATH`.
- A C compiler and `libc` development headers (for the death-pact implementation).

### Building

```bash
cargo build --release
```

## Usage

### 1. Start the Daemon

```bash
./target/release/sonr-daemon --cache-file ./embeddings.json
```

The daemon will download the default models (Qwen3-0.6B based) and start listening on port 3000.

### 2. Search via CLI

```bash
./target/release/sonr "how does the chunking logic work?" ./crates/sonr-daemon/src
```

### CLI Options

- `sonr <query> [paths...]`: Search for `<query>` in the specified paths.
- `--limit <N>`: Number of results to return (default: 10).
- `--json`: Output results in JSON format.
- `--url <URL>`: Connect to a custom daemon URL (default: `http://localhost:3000`).

## Configuration

The daemon supports several flags:
- `--port`: API port (default: 3000).
- `--embedding-hf-repo`: HF repository for the embedding model.
- `--reranker-hf-repo`: HF repository for the reranker model.
- `--gpu-layers`: Number of layers to offload to GPU (default: 99).
- `--cache-file`: Path to persist the embedding cache.