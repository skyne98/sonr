use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use directories::ProjectDirs;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "sonr",
    author = "fox",
    version = env!("CARGO_PKG_VERSION"),
    about = "Semantic grep for your codebase",
    long_about = "sonr is a high-performance semantic search tool that understands your code.
It uses local LLM embeddings and reranking to find relevant snippets based on meaning rather than just keywords.

Requires a running `sonr-daemon` to perform inference.",
    after_help = "EXAMPLES:
    Search for logic in the current directory:
      $ sonr \"how are user sessions handled?\"

    Search in specific directories with a limit:
      $ sonr \"database connection pool\" ./src ./lib --limit 10

    Run as an MCP stdio bridge:
      $ sonr mcp stdio",
    help_template = "{about-section}

USAGE:
    {usage}

ARGUMENTS:
{positionals}

OPTIONS:
{options}

{after-help}"
)]
struct Args {
    #[command(subcommand)]
    command: Option<Commands>,

    /// The semantic query or question about the code
    #[arg(
        value_name = "QUERY",
        help = "The natural language query to search for"
    )]
    query: Option<String>,

    /// Files or directories to index and search
    #[arg(
        value_name = "PATHS",
        help = "Paths to search in (files or directories)"
    )]
    paths: Vec<PathBuf>,

    /// Maximum number of results to return
    #[arg(short, long, help = "Limit the number of search results")]
    limit: Option<usize>,

    /// URL of the sonr-daemon
    #[arg(
        long,
        env = "SONR_URL",
        help = "The address of the running sonr-daemon (overrides port file discovery)"
    )]
    url: Option<String>,

    /// Path to the daemon's port file
    #[arg(long, help = "Path to the daemon's port file for automatic discovery")]
    port_file: Option<PathBuf>,

    /// Output raw JSON instead of formatted text
    #[arg(long, help = "Format output as JSON for machine consumption")]
    json: bool,
}

#[derive(Subcommand)]
enum Commands {
    /// MCP related commands
    Mcp {
        #[command(subcommand)]
        command: McpCommands,
    },
}

#[derive(Subcommand)]
enum McpCommands {
    /// Run an MCP bridge over stdio
    Stdio,
}

#[derive(Serialize)]
struct SearchRequest {
    query: String,
    paths: Vec<PathBuf>,
    limit: Option<usize>,
}

#[derive(Deserialize, Debug, Serialize)]
struct SearchResult {
    content: String,
    file: String,
    line_start: usize,
    line_end: usize,
    score: f32,
}

async fn get_base_url(args: &Args) -> String {
    if let Some(url) = &args.url {
        url.clone()
    } else {
        let port_file = args.port_file.clone().or_else(|| {
            ProjectDirs::from("com", "sonr", "sonr").map(|proj_dirs| {
                proj_dirs
                    .runtime_dir()
                    .unwrap_or_else(|| proj_dirs.cache_dir())
                    .join("daemon.port")
            })
        });

        let mut discovered_url = "http://localhost:3000".to_string();
        if let Some(path) = port_file {
            if path.exists() {
                if let Ok(port_str) = std::fs::read_to_string(&path) {
                    if let Ok(port) = port_str.trim().parse::<u16>() {
                        discovered_url = format!("http://127.0.0.1:{}", port);
                    }
                }
            }
        }
        discovered_url
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();

    match &args.command {
        Some(Commands::Mcp {
            command: McpCommands::Stdio,
        }) => run_mcp_stdio(&args).await,
        None => {
            let query = args
                .query
                .clone()
                .context("No query provided. Use 'sonr --help' for usage.")?;
            let paths = if args.paths.is_empty() {
                vec![PathBuf::from(".")]
            } else {
                args.paths.clone()
            };
            run_search(&args, query, paths).await
        }
    }
}

async fn run_search(args: &Args, query: String, paths: Vec<PathBuf>) -> Result<()> {
    let client = Client::new();
    let base_url = get_base_url(args).await;

    // Canonicalize paths to ensure the daemon can find them if they are local
    let mut absolute_paths = Vec::new();
    for path in paths {
        if let Ok(abs) = std::fs::canonicalize(&path) {
            absolute_paths.push(abs);
        } else {
            absolute_paths.push(path);
        }
    }

    let request = SearchRequest {
        query,
        paths: absolute_paths,
        limit: Some(args.limit.unwrap_or(5)),
    };

    let url = format!("{}/search", base_url.trim_end_matches('/'));
    let resp = client
        .post(&url)
        .json(&request)
        .send()
        .await
        .context("Failed to connect to sonr-daemon. Is it running?")?;

    if !resp.status().is_success() {
        let err_text = resp
            .text()
            .await
            .unwrap_or_else(|_| "Unknown error".to_string());
        eprintln!("\x1b[1;31mError from daemon:\x1b[0m {}", err_text);
        std::process::exit(1);
    }

    let results: Vec<SearchResult> = resp
        .json()
        .await
        .context("Failed to parse search results from daemon")?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&results)?);
        return Ok(());
    }

    if results.is_empty() {
        println!("No semantically relevant matches found.");
    } else {
        for (i, res) in results.iter().enumerate() {
            println!(
                "\x1b[1;32mMatch {} \x1b[0m\x1b[2m(Relevance: {:.4})\x1b[0m",
                i + 1,
                res.score
            );
            println!(
                "\x1b[1;34m--> {}:{}-{}\x1b[0m",
                res.file, res.line_start, res.line_end
            );
            for line in res.content.trim().lines() {
                println!("  {}", line);
            }
            println!("\x1b[2m{}\x1b[0m", "─".repeat(60));
        }
    }

    Ok(())
}

async fn run_mcp_stdio(args: &Args) -> Result<()> {
    use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

    let base_url = get_base_url(args).await;
    let mcp_url = format!("{}/mcp", base_url.trim_end_matches('/'));
    let client = Client::new();

    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin);
    let mut line = String::new();
    let mut session_id: Option<String> = None;

    loop {
        line.clear();
        let n = reader.read_line(&mut line).await?;
        if n == 0 {
            break; // EOF
        }

        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        // Build request with proper MCP headers
        let mut req = client
            .post(&mcp_url)
            .header("Content-Type", "application/json")
            .header("Accept", "application/json, text/event-stream");

        if let Some(ref sid) = session_id {
            req = req.header("mcp-session-id", sid);
        }

        let resp = req.body(trimmed.to_string()).send().await;

        match resp {
            Ok(response) => {
                // Extract session ID from response headers
                if let Some(sid) = response.headers().get("mcp-session-id") {
                    if let Ok(s) = sid.to_str() {
                        session_id = Some(s.to_string());
                    }
                }

                let body = response.text().await.unwrap_or_default();
                // Parse SSE format: extract JSON from "data: {...}" lines
                for line in body.lines() {
                    let line = line.trim();
                    if line.starts_with("data:") {
                        let json_part = line.strip_prefix("data:").unwrap_or("").trim();
                        if !json_part.is_empty() {
                            stdout.write_all(json_part.as_bytes()).await?;
                            stdout.write_all(b"\n").await?;
                            stdout.flush().await?;
                        }
                    }
                }
            }
            Err(e) => {
                let err_response = serde_json::json!({
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": format!("Failed to connect to daemon: {}", e)
                    },
                    "id": null
                });
                stdout
                    .write_all(err_response.to_string().as_bytes())
                    .await?;
                stdout.write_all(b"\n").await?;
                stdout.flush().await?;
            }
        }
    }

    Ok(())
}
