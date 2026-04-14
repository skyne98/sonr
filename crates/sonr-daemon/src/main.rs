use anyhow::{Context, Result};
use axum::{
    Json as AxumJson, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use clap::Parser;
use directories::ProjectDirs;
use ignore::WalkBuilder;
use memvdb::{CacheDB, Distance, Embedding};
use reqwest::Client;
use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::*,
    tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        session::local::LocalSessionManager,
        tower::{StreamableHttpServerConfig, StreamableHttpService},
    },
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::os::unix::process::CommandExt;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use tokio::process::{Child, Command};
use tokio::sync::{Mutex, RwLock, Semaphore};
use tokio::time::{Duration, sleep};
use tracing::{error, info};

#[derive(Parser, Debug)]
#[command(
    name = "sonr-daemon",
    author = "fox",
    version = env!("CARGO_PKG_VERSION"),
    about = "Background inference engine for sonr",
    long_about = "sonr-daemon manages local LLM instances for embedding and reranking.
It automatically downloads models from Hugging Face and provides a REST API for the sonr CLI.
It handles persistent caching of embeddings to ensure fast subsequent searches.",
    after_help = "NOTES:
    - Requires `llama-server` to be available in your PATH.
    - Models are cached in the standard Hugging Face cache directory.
    - By default, embeddings are stored in your OS-specific user cache directory.",
    help_template = "{about-section}

USAGE:
    {usage}

OPTIONS:
{options}

{after-help}"
)]
struct Args {
    /// Port to listen on for the daemon API
    #[arg(
        long,
        default_value = "3000",
        help = "The port the daemon API will listen on. Use 0 for OS-assigned port."
    )]
    port: u16,

    /// Path to a file where the daemon will write its port number
    #[arg(
        long,
        help = "Path to a file where the daemon will write its port number"
    )]
    port_file: Option<PathBuf>,

    /// Hugging Face repo for the embedding model
    #[arg(
        long,
        default_value = "yomir/Qwen3-Embedding-0.6B-GGUF",
        help = "HF repository for the embedding model (GGUF)"
    )]
    embedding_hf_repo: String,

    /// Hugging Face file for the embedding model
    #[arg(
        long,
        default_value = "Qwen3-Embedding-0.6B-Q8_0.gguf",
        help = "Specific GGUF file to use from the embedding repo"
    )]
    embedding_hf_file: String,

    /// Port for the internal llama-server embedding instance
    #[arg(
        long,
        default_value = "8081",
        help = "Internal port for the embedding llama-server"
    )]
    llama_port: u16,

    /// Hugging Face repo for the reranker model
    #[arg(
        long,
        default_value = "ggml-org/Qwen3-Reranker-0.6B-Q8_0-GGUF",
        help = "HF repository for the reranker model (GGUF)"
    )]
    reranker_hf_repo: String,

    /// Port for the internal llama-server reranker instance
    #[arg(
        long,
        default_value = "8082",
        help = "Internal port for the reranker llama-server"
    )]
    llama_reranker_port: u16,

    /// Context size for the models
    #[arg(
        long,
        default_value = "8192",
        help = "Maximum context size (tokens) for the models"
    )]
    ctx_size: u32,

    /// Number of GPU layers to offload
    #[arg(
        long,
        help = "Number of model layers to offload to GPU (leave unset to let llama.cpp auto-fit)"
    )]
    gpu_layers: Option<i32>,

    /// Whether to enable llama.cpp auto-fit for device memory
    #[arg(
        long,
        default_value_t = false,
        help = "Pass --fit on to the internal llama-server instances"
    )]
    fit: bool,

    /// Path to the embedding cache file
    #[arg(
        long,
        help = "Custom path to persist the embedding cache (defaults to OS cache dir)"
    )]
    cache_file: Option<PathBuf>,
}

struct LlamaProcess {
    child: Child,
}

impl LlamaProcess {
    async fn spawn(
        hf_repo: &str,
        hf_file: Option<&str>,
        port: u16,
        ctx_size: u32,
        gpu_layers: Option<i32>,
        is_reranker: bool,
        fit: bool,
    ) -> Result<Self> {
        let mode_str = if is_reranker { "reranker" } else { "embedding" };
        info!("Starting llama-server for {} on port {}", mode_str, port);

        let mut cmd = Command::new("llama-server");
        cmd.arg("--hf-repo").arg(hf_repo);
        if let Some(file) = hf_file {
            cmd.arg("--hf-file").arg(file);
        }

        cmd.arg("--port")
            .arg(port.to_string())
            .arg("--ctx-size")
            .arg(ctx_size.to_string())
            .arg("--batch-size")
            .arg("2048")
            .arg("--ubatch-size")
            .arg("512");

        if let Some(gpu_layers) = gpu_layers {
            cmd.arg("--n-gpu-layers").arg(gpu_layers.to_string());
        }

        if fit {
            cmd.arg("--fit").arg("on");
        }

        if is_reranker {
            cmd.arg("--reranking");
        } else {
            cmd.arg("--embedding").arg("--pooling").arg("mean");
        }

        unsafe {
            cmd.as_std_mut().pre_exec(|| {
                libc::prctl(
                    libc::PR_SET_PDEATHSIG,
                    libc::SIGTERM as libc::c_ulong,
                    0,
                    0,
                    0,
                );
                if libc::getppid() == 1 {
                    libc::exit(1);
                }
                Ok(())
            });
        }

        let child = cmd
            .stdout(Stdio::inherit())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .context("Failed to spawn llama-server")?;

        let client = Client::new();
        let health_url = format!("http://127.0.0.1:{}/health", port);

        let mut attempts = 0;
        loop {
            if attempts > 150 {
                return Err(anyhow::anyhow!("llama-server failed to start"));
            }
            match client.get(&health_url).send().await {
                Ok(resp) if resp.status().is_success() => {
                    info!("llama-server on port {} is ready!", port);
                    break;
                }
                _ => {
                    sleep(Duration::from_secs(2)).await;
                    attempts += 1;
                }
            }
        }

        Ok(Self { child })
    }

    async fn terminate(mut self) {
        if let Err(e) = self.child.start_kill() {
            if e.kind() != std::io::ErrorKind::InvalidInput {
                error!("Failed to signal llama-server termination: {}", e);
            }
        }

        if let Err(e) = self.child.wait().await {
            error!("Failed to wait for llama-server termination: {}", e);
        }
    }
}

#[derive(Clone)]
struct LlamaRuntimeConfig {
    embedding_hf_repo: String,
    embedding_hf_file: String,
    llama_port: u16,
    reranker_hf_repo: String,
    llama_reranker_port: u16,
    ctx_size: u32,
    gpu_layers: Option<i32>,
    fit: bool,
}

struct LlamaRuntime {
    config: LlamaRuntimeConfig,
    embed_proc: Option<LlamaProcess>,
    rerank_proc: Option<LlamaProcess>,
    active_searches: usize,
}

impl LlamaRuntime {
    fn new(config: LlamaRuntimeConfig) -> Self {
        Self {
            config,
            embed_proc: None,
            rerank_proc: None,
            active_searches: 0,
        }
    }

    async fn ensure_running(&mut self) -> Result<()> {
        if self.embed_proc.is_some() && self.rerank_proc.is_some() {
            return Ok(());
        }

        if let Some(proc) = self.rerank_proc.take() {
            proc.terminate().await;
        }
        if let Some(proc) = self.embed_proc.take() {
            proc.terminate().await;
        }

        info!("Starting llama-server helpers for semantic search");
        let config = self.config.clone();
        let embed_proc = LlamaProcess::spawn(
            &config.embedding_hf_repo,
            Some(&config.embedding_hf_file),
            config.llama_port,
            config.ctx_size,
            config.gpu_layers,
            false,
            config.fit,
        )
        .await?;

        let rerank_proc = match LlamaProcess::spawn(
            &config.reranker_hf_repo,
            None,
            config.llama_reranker_port,
            config.ctx_size,
            config.gpu_layers,
            true,
            config.fit,
        )
        .await
        {
            Ok(proc) => proc,
            Err(err) => {
                embed_proc.terminate().await;
                return Err(err);
            }
        };

        self.embed_proc = Some(embed_proc);
        self.rerank_proc = Some(rerank_proc);
        Ok(())
    }
}

struct LlamaRuntimeLease {
    runtime: Arc<Mutex<LlamaRuntime>>,
}

impl LlamaRuntimeLease {
    async fn release(self) {
        let (embed_proc, rerank_proc, remaining_searches) = {
            let mut runtime = self.runtime.lock().await;
            runtime.active_searches = runtime.active_searches.saturating_sub(1);
            let remaining_searches = runtime.active_searches;
            if remaining_searches == 0 {
                (
                    runtime.embed_proc.take(),
                    runtime.rerank_proc.take(),
                    remaining_searches,
                )
            } else {
                (None, None, remaining_searches)
            }
        };

        if remaining_searches != 0 {
            return;
        }

        info!("Stopping llama-server helpers after semantic search");
        if let Some(proc) = rerank_proc {
            proc.terminate().await;
        }
        if let Some(proc) = embed_proc {
            proc.terminate().await;
        }
    }
}

#[derive(Clone)]
struct AppState {
    client: Client,
    llama_embed_url: String,
    llama_rerank_url: String,
    llama_runtime: Arc<Mutex<LlamaRuntime>>,
    embedding_cache: Arc<RwLock<HashMap<String, Vec<f32>>>>,
    cache_dirty: Arc<AtomicBool>,
    cache_file: Option<PathBuf>,
    model_id: String,
    tool_router: ToolRouter<Self>,
}

#[derive(Deserialize, rmcp::schemars::JsonSchema)]
struct SemanticSearchToolRequest {
    #[schemars(description = "The natural language query")]
    query: String,
    #[schemars(description = "The root directory to search in")]
    root_directory: String,
    #[schemars(description = "Maximum number of results to return")]
    limit: Option<usize>,
}

#[derive(Serialize, rmcp::schemars::JsonSchema)]
struct SemanticSearchToolResponse {
    #[schemars(description = "List of search results ordered by relevance")]
    results: Vec<SearchResult>,
}

#[tool_router]
impl AppState {
    #[tool(description = "Search codebases using natural language semantic search")]
    async fn semantic_search(
        &self,
        params: Parameters<SemanticSearchToolRequest>,
    ) -> Result<rmcp::Json<SemanticSearchToolResponse>, McpError> {
        let payload = params.0;
        let req = SearchRequest {
            query: payload.query,
            paths: vec![PathBuf::from(payload.root_directory)],
            limit: payload.limit,
        };

        match perform_semantic_search(self, req).await {
            Ok(results) => Ok(rmcp::Json(SemanticSearchToolResponse { results })),
            Err(e) => Err(McpError::new(ErrorCode(-32603), e.1, None)),
        }
    }
}

impl AppState {
    async fn acquire_llama_runtime(&self) -> Result<LlamaRuntimeLease, (StatusCode, String)> {
        let mut runtime = self.llama_runtime.lock().await;
        runtime
            .ensure_running()
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
        runtime.active_searches += 1;
        Ok(LlamaRuntimeLease {
            runtime: self.llama_runtime.clone(),
        })
    }
}

#[tool_handler]
impl ServerHandler for AppState {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "sonr-daemon".into(),
                version: env!("CARGO_PKG_VERSION").into(),
                title: Some("sonr-daemon".into()),
                icons: None,
                website_url: None,
            },
            instructions: Some("Use this tool to find relevant code snippets in a codebase using semantic search. It is much more effective than grep for conceptual queries.".into()),
        }
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();

    let proj_dirs = ProjectDirs::from("com", "sonr", "sonr");

    let cache_file = if let Some(path) = args.cache_file {
        Some(path)
    } else {
        proj_dirs.as_ref().map(|proj_dirs| {
            let cache_dir = proj_dirs.cache_dir();
            let _ = std::fs::create_dir_all(cache_dir);
            cache_dir.join("embeddings.json")
        })
    };

    let mut initial_cache = HashMap::new();
    if let Some(ref path) = cache_file {
        if path.exists() {
            if let Ok(data) = std::fs::read(path) {
                if let Ok(decoded) = serde_json::from_slice::<HashMap<String, Vec<f32>>>(&data) {
                    initial_cache = decoded;
                    info!("Loaded {} embeddings from cache", initial_cache.len());
                }
            }
        }
    }

    let runtime_config = LlamaRuntimeConfig {
        embedding_hf_repo: args.embedding_hf_repo.clone(),
        embedding_hf_file: args.embedding_hf_file.clone(),
        llama_port: args.llama_port,
        reranker_hf_repo: args.reranker_hf_repo.clone(),
        llama_reranker_port: args.llama_reranker_port,
        ctx_size: args.ctx_size,
        gpu_layers: args.gpu_layers,
        fit: args.fit,
    };

    let state = AppState {
        client: Client::new(),
        llama_embed_url: format!("http://127.0.0.1:{}", runtime_config.llama_port),
        llama_rerank_url: format!("http://127.0.0.1:{}", runtime_config.llama_reranker_port),
        llama_runtime: Arc::new(Mutex::new(LlamaRuntime::new(runtime_config))),
        embedding_cache: Arc::new(RwLock::new(initial_cache)),
        cache_dirty: Arc::new(AtomicBool::new(false)),
        cache_file,
        model_id: format!("{}:{}", args.embedding_hf_repo, args.embedding_hf_file),
        tool_router: AppState::tool_router(),
    };

    // Background cache saver
    if let Some(cache_path) = state.cache_file.clone() {
        let cache = state.embedding_cache.clone();
        let dirty = state.cache_dirty.clone();
        tokio::spawn(async move {
            loop {
                sleep(Duration::from_secs(30)).await;
                if dirty.swap(false, Ordering::SeqCst) {
                    info!("Saving embedding cache to disk...");
                    let data = {
                        let read_cache = cache.read().await;
                        serde_json::to_vec(&*read_cache).ok()
                    };
                    if let Some(bytes) = data {
                        if let Err(e) = tokio::fs::write(&cache_path, bytes).await {
                            error!("Failed to save cache: {}", e);
                            dirty.store(true, Ordering::SeqCst);
                        }
                    }
                }
            }
        });
    }

    let mcp_service = StreamableHttpService::new(
        {
            let state = state.clone();
            move || Ok(state.clone())
        },
        Arc::new(LocalSessionManager::default()),
        StreamableHttpServerConfig::default(),
    );

    let app = Router::new()
        .route("/health", get(health_check))
        .route("/search", post(semantic_search_handler))
        .nest_service("/mcp", mcp_service)
        .with_state(state.clone());

    let addr = SocketAddr::from(([127, 0, 0, 1], args.port));
    let listener = tokio::net::TcpListener::bind(addr).await?;
    let local_addr = listener.local_addr()?;
    info!("sonr-daemon listening on {}", local_addr);

    let port_file = if let Some(path) = args.port_file {
        Some(path)
    } else {
        proj_dirs.as_ref().map(|proj_dirs| {
            let run_dir = proj_dirs
                .runtime_dir()
                .unwrap_or_else(|| proj_dirs.cache_dir());
            let _ = std::fs::create_dir_all(run_dir);
            run_dir.join("daemon.port")
        })
    };

    if let Some(ref path) = port_file {
        if let Err(e) = std::fs::write(path, local_addr.port().to_string()) {
            error!("Failed to write port file {:?}: {}", path, e);
        } else {
            info!("Port written to {:?}", path);
        }
    }

    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            if let Some(path) = port_file {
                let _ = std::fs::remove_file(path);
            }
        })
        .await?;

    // Final save on shutdown
    if state.cache_dirty.load(Ordering::SeqCst) {
        if let Some(ref path) = state.cache_file {
            info!("Saving embedding cache on shutdown...");
            let cache = state.embedding_cache.read().await;
            if let Ok(data) = serde_json::to_vec(&*cache) {
                let _ = tokio::fs::write(path, data).await;
            }
        }
    }

    Ok(())
}

async fn health_check() -> StatusCode {
    StatusCode::OK
}

async fn shutdown_signal() {
    let ctrl_c = async {
        tokio::signal::ctrl_c()
            .await
            .expect("failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate())
            .expect("failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    paths: Vec<PathBuf>,
    limit: Option<usize>,
}

#[derive(Serialize, rmcp::schemars::JsonSchema)]
struct SearchResult {
    content: String,
    file: String,
    line_start: i64,
    line_end: i64,
    score: f32,
}

async fn get_embedding(state: &AppState, text: &str) -> Result<Vec<f32>, (StatusCode, String)> {
    let hash = format!("{:x}", md5::compute(text));
    let cache_key = format!("{}:{}", state.model_id, hash);

    {
        let read_cache = state.embedding_cache.read().await;
        if let Some(vec) = read_cache.get(&cache_key) {
            return Ok(vec.clone());
        }
    }

    let body = serde_json::json!({ "input": text, "model": "default" });
    let resp = state
        .client
        .post(&format!("{}/v1/embeddings", state.llama_embed_url))
        .json(&body)
        .send()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let status = resp.status();
    if !status.is_success() {
        let error_text = resp.text().await.unwrap_or_default();
        error!("llama-server error ({}): {}", status, error_text);
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("llama-server error ({}): {}", status, error_text),
        ));
    }

    let json: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    let embedding: Vec<f32> = json["data"][0]["embedding"]
        .as_array()
        .ok_or((
            StatusCode::INTERNAL_SERVER_ERROR,
            "Invalid embedding format".into(),
        ))?
        .iter()
        .map(|v| v.as_f64().unwrap_or(0.0) as f32)
        .collect();

    {
        let mut write_cache = state.embedding_cache.write().await;
        write_cache.insert(cache_key, embedding.clone());
        state.cache_dirty.store(true, Ordering::SeqCst);
    }

    Ok(embedding)
}

struct Chunk {
    content: String,
    line_start: i64,
    line_end: i64,
}

fn chunk_text(text: &str, max_chars: usize) -> Vec<Chunk> {
    let mut chunks = Vec::new();
    let lines: Vec<&str> = text.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        // Skip empty lines at start of chunk
        while i < lines.len() && lines[i].trim().is_empty() {
            i += 1;
        }
        if i >= lines.len() {
            break;
        }

        let start_line = i + 1;
        let mut current_chunk = String::new();
        let mut last_non_empty = i;

        while i < lines.len() {
            let line = lines[i];
            if current_chunk.len() + line.len() > max_chars && !current_chunk.is_empty() {
                break;
            }

            current_chunk.push_str(line);
            current_chunk.push('\n');

            if !line.trim().is_empty() {
                last_non_empty = i;
            }

            i += 1;

            // If we hit a double newline (empty line), and we have enough content, we can stop here
            if i < lines.len() && lines[i].trim().is_empty() && current_chunk.len() > 500 {
                break;
            }
        }

        chunks.push(Chunk {
            content: current_chunk.trim().to_string(),
            line_start: start_line as i64,
            line_end: (last_non_empty + 1) as i64,
        });
    }
    chunks
}

async fn semantic_search_handler(
    State(state): State<AppState>,
    AxumJson(payload): AxumJson<SearchRequest>,
) -> Result<AxumJson<Vec<SearchResult>>, (StatusCode, String)> {
    perform_semantic_search(&state, payload).await.map(AxumJson)
}

async fn perform_semantic_search(
    state: &AppState,
    payload: SearchRequest,
) -> Result<Vec<SearchResult>, (StatusCode, String)> {
    let runtime_lease = state.acquire_llama_runtime().await?;
    let result = async {
        let query_vec = get_embedding(state, &payload.query).await?;

        let mut db = CacheDB::new();
        let collection_name = "temp_search";
        db.create_collection(
            collection_name.to_string(),
            query_vec.len(),
            Distance::Cosine,
        )
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let mut join_set = tokio::task::JoinSet::new();
        let semaphore = Arc::new(Semaphore::new(32));

        for root in payload.paths {
            let walker = WalkBuilder::new(root).hidden(true).git_ignore(true).build();

            for result in walker {
                if let Ok(entry) = result {
                    if entry.file_type().map(|ft| ft.is_file()).unwrap_or(false) {
                        let path = entry.path().to_path_buf();
                        let state = state.clone();
                        let sem = semaphore.clone();

                        join_set.spawn(async move {
                            let content = match tokio::fs::read_to_string(&path).await {
                                Ok(c) => c,
                                Err(_) => return Vec::new(),
                            };

                            let chunks = chunk_text(&content, 4000);
                            let mut file_embeddings = Vec::new();

                            for chunk in chunks {
                                let _permit = sem.acquire().await.unwrap();
                                let vector = match get_embedding(&state, &chunk.content).await {
                                    Ok(v) => v,
                                    Err(_) => continue,
                                };

                                let mut id = HashMap::new();
                                id.insert("hash".into(), format!("{:x}", md5::compute(&chunk.content)));

                                let mut metadata = HashMap::new();
                                metadata.insert("content".into(), chunk.content);
                                metadata.insert("file".into(), path.display().to_string());
                                metadata.insert("line_start".into(), chunk.line_start.to_string());
                                metadata.insert("line_end".into(), chunk.line_end.to_string());

                                file_embeddings.push(Embedding {
                                    id,
                                    vector,
                                    metadata: Some(metadata),
                                });
                            }
                            file_embeddings
                        });
                    }
                }
            }
        }

        while let Some(res) = join_set.join_next().await {
            if let Ok(embeddings) = res {
                for emb in embeddings {
                    let _ = db.insert_into_collection(collection_name, emb);
                }
            }
        }

        let collection = db
            .get_collection(collection_name)
            .ok_or((StatusCode::INTERNAL_SERVER_ERROR, "DB error".into()))?;

        let limit = payload.limit.unwrap_or(10);
        let initial_results = collection.get_similarity(&query_vec, limit * 2);

        if initial_results.is_empty() {
            return Ok(vec![]);
        }

        let mut docs_to_rerank = Vec::new();
        let mut metadata_list = Vec::new();
        for res in initial_results {
            let meta = res.embedding.metadata.clone().unwrap_or_default();
            docs_to_rerank.push(meta.get("content").cloned().unwrap_or_default());
            metadata_list.push(meta);
        }

        let rerank_body = serde_json::json!({
            "model": "default",
            "query": payload.query,
            "documents": docs_to_rerank
        });

        let resp = state
            .client
            .post(&format!("{}/v1/rerank", state.llama_rerank_url))
            .json(&rerank_body)
            .send()
            .await
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

        let status = resp.status();
        if status.is_success() {
            let json: serde_json::Value = resp
                .json()
                .await
                .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

            if let Some(reranked) = json["results"].as_array() {
                let mut final_results = Vec::new();
                for item in reranked.iter().take(limit) {
                    let idx = item["index"].as_u64().unwrap() as usize;
                    let score = item["relevance_score"].as_f64().unwrap() as f32;
                    let meta = &metadata_list[idx];

                    final_results.push(SearchResult {
                        content: docs_to_rerank[idx].clone(),
                        file: meta.get("file").cloned().unwrap_or_default(),
                        line_start: meta
                            .get("line_start")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0),
                        line_end: meta
                            .get("line_end")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0),
                        score,
                    });
                }
                Ok(final_results)
            } else {
                let final_results = docs_to_rerank
                    .into_iter()
                    .zip(metadata_list.into_iter())
                    .take(limit)
                    .map(|(content, meta)| SearchResult {
                        content,
                        file: meta.get("file").cloned().unwrap_or_default(),
                        line_start: meta
                            .get("line_start")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0),
                        line_end: meta
                            .get("line_end")
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(0),
                        score: 0.0,
                    })
                    .collect();
                Ok(final_results)
            }
        } else {
            let error_text = resp.text().await.unwrap_or_default();
            error!("llama-server rerank error ({}): {}", status, error_text);

            let final_results = docs_to_rerank
                .into_iter()
                .zip(metadata_list.into_iter())
                .take(limit)
                .map(|(content, meta)| SearchResult {
                    content,
                    file: meta.get("file").cloned().unwrap_or_default(),
                    line_start: meta
                        .get("line_start")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0),
                    line_end: meta
                        .get("line_end")
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0),
                    score: 0.0,
                })
                .collect();
            Ok(final_results)
        }
    }.await;

    runtime_lease.release().await;
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunk_text_basic() {
        let text = "line1\nline2";
        let chunks = chunk_text(text, 100);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "line1\nline2");
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[0].line_end, 2);
    }

    #[test]
    fn test_chunk_text_multiple() {
        let text = "line1\nline2";
        let chunks = chunk_text(text, 5);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].content, "line1");
        assert_eq!(chunks[1].content, "line2");
    }

    #[test]
    fn test_chunk_text_empty_lines() {
        let text = "\n\nline1\n\n\nline2\n\n";
        let chunks = chunk_text(text, 100);
        // Small chunks are merged into one
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].content, "line1\n\n\nline2");
        assert_eq!(chunks[0].line_start, 3);
        assert_eq!(chunks[0].line_end, 6);
    }

    #[tokio::test]
    async fn test_health_check() {
        let status = health_check().await;
        assert_eq!(status, StatusCode::OK);
    }
}
