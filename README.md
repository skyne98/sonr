# sonr

`sonr` is a high-performance semantic search tool for local codebases. It consists of a background daemon that manages LLM inference and a CLI for performing fast, context-aware searches.

## Features

- **Semantic Search**: Find code by meaning, not just keywords.
- **GPU Accelerated**: Uses `llama.cpp` (via `llama-server`) for fast embedding and reranking.
- **Automatic Model Management**: Downloads required GGUF models from Hugging Face automatically.
- **Persistent Caching**: Embeddings are cached to disk to make subsequent searches near-instant.
- **MCP Server**: Expose semantic search as a tool via the Model Context Protocol.

## Architecture

- **sonr-daemon**: A background service that lazily starts two `llama-server` instances per search (one for embeddings, one for reranking), then shuts them down after the request while providing a REST API.
- **sonr**: A lightweight CLI that sends search queries to the daemon and displays results.

## Installation

### Prerequisites

- `llama-server` must be installed and available in your `PATH`.

### Install via Cargo

```bash
cargo install sonr sonr-daemon
```

## Usage

### 1. Start the Daemon

```bash
sonr-daemon
```

The daemon will download the default models (Qwen3-0.6B based) and start listening on an available port. `llama-server` helpers are launched lazily for each search and shut down when the request completes. It writes the daemon port to a discovery file so the CLI can find it automatically.

### 2. Search via CLI

```bash
sonr "how does the chunking logic work?" ./src
```

### CLI Options

- `sonr <query> [paths...]`: Search for `<query>` in the specified paths.
- `--limit <N>`: Number of results to return (default: 5).
- `--json`: Output results in JSON format.
- `--url <URL>`: Connect to a custom daemon URL (overrides automatic discovery).
- `--port-file <PATH>`: Path to the daemon's port discovery file.

### MCP Server

The daemon exposes an MCP (Model Context Protocol) server at `/mcp`. You can use the CLI as a stdio-to-HTTP bridge:

```bash
sonr mcp stdio
```

This allows MCP clients to use the `semantic_search` tool with parameters:
- `query`: Natural language search query
- `root_directory`: Path to search in
- `limit`: Maximum number of results (optional)

## Configuration

The daemon supports several flags:
- `--port`: API port (default: 0 for OS-assigned).
- `--port-file`: Path to write the assigned port for CLI discovery.
- `--embedding-hf-repo`: HF repository for the embedding model.
- `--reranker-hf-repo`: HF repository for the reranker model.
- `--gpu-layers`: Number of layers to offload to GPU (optional; leave unset to let llama.cpp decide when used with `--fit`).
- `--fit`: Enable llama.cpp automatic GPU fit for the internal embedding and reranker servers.
- `--cache-file`: Path to persist the embedding cache.

Run `sonr-daemon --help` for full details.