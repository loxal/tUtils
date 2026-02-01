// Copyright 2026 Alexander Orlov <alexander.orlov@loxal.net>

mod convert;

use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "hurl-to-har-to-hurl-converter",
    version,
    about = "Convert between HURL and HAR file formats"
)]
struct Args {
    /// Input file (.hurl or .har)
    input: PathBuf,

    /// Output directory (defaults to current directory)
    #[arg(short, long, default_value = ".")]
    output_dir: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args = Args::parse();

    let ext = args
        .input
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    match ext {
        "hurl" => {
            let out = convert::hurl_to_har(&args.input, &args.output_dir)?;
            println!("Converted {} -> {}", args.input.display(), out.display());
        }
        "har" => {
            let out = convert::har_to_hurl(&args.input, &args.output_dir)?;
            println!("Converted {} -> {}", args.input.display(), out.display());
        }
        _ => {
            return Err(format!(
                "Cannot determine conversion direction for '{}'. Expected .hurl or .har extension.",
                args.input.display()
            )
            .into());
        }
    }

    Ok(())
}
