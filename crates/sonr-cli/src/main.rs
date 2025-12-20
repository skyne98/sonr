use anyhow::{Context, Result};
use clap::Parser;
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

    Get raw JSON for piping into other tools:
      $ sonr \"auth middleware\" --json | jq .",
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
    /// The semantic query or question about the code
    #[arg(
        value_name = "QUERY",
        help = "The natural language query to search for"
    )]
    query: String,

    /// Files or directories to index and search
    #[arg(
        value_name = "PATHS",
        default_value = ".",
        help = "Paths to search in (files or directories)"
    )]
    paths: Vec<PathBuf>,

    /// Maximum number of results to return
    #[arg(
        short,
        long,
        default_value = "5",
        help = "Limit the number of search results"
    )]
    limit: usize,

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

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let client = Client::new();

    let base_url = if let Some(url) = args.url {
        url
    } else {
        let port_file = args.port_file.or_else(|| {
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
    };

    // Canonicalize paths to ensure the daemon can find them if they are local
    let mut absolute_paths = Vec::new();
    for path in args.paths {
        if let Ok(abs) = std::fs::canonicalize(&path) {
            absolute_paths.push(abs);
        } else {
            // If it doesn't exist yet or can't be canonicalized, use as is
            absolute_paths.push(path);
        }
    }

    let request = SearchRequest {
        query: args.query,
        paths: absolute_paths,
        limit: Some(args.limit),
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
            // Header with match number and score
            println!(
                "\x1b[1;32mMatch {} \x1b[0m\x1b[2m(Relevance: {:.4})\x1b[0m",
                i + 1,
                res.score
            );

            // File path and line numbers
            println!(
                "\x1b[1;34m--> {}:{}-{}\x1b[0m",
                res.file, res.line_start, res.line_end
            );

            // Content with indentation
            for line in res.content.trim().lines() {
                println!("  {}", line);
            }

            // Separator
            println!("\x1b[2m{}\x1b[0m", "─".repeat(60));
        }
    }

    Ok(())
}
