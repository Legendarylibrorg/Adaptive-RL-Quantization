//! CLI entrypoint — JSON stdin/stdout, same contract as Python subprocess hooks.

use adaptive_rl_sim::{evaluate_episode, EvaluateRequest};
use std::io::{self, Read, Write};

fn main() {
    if let Err(err) = run(std::env::args().skip(1).collect()) {
        eprintln!("adaptive-rl-quant-rust: {err}");
        std::process::exit(1);
    }
}

fn run(args: Vec<String>) -> Result<(), String> {
    let command = args.first().map(String::as_str).unwrap_or("help");
    match command {
        "sim-eval" => cmd_sim_eval(),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        _ => Err(format!("unknown command {command:?}; try 'sim-eval' or 'help'")),
    }
}

fn cmd_sim_eval() -> Result<(), String> {
    let mut input = String::new();
    io::stdin()
        .read_to_string(&mut input)
        .map_err(|e| e.to_string())?;
    let req: EvaluateRequest = serde_json::from_str(&input).map_err(|e| e.to_string())?;
    let metrics = evaluate_episode(&req);
    let json = serde_json::to_string(&metrics).map_err(|e| e.to_string())?;
    io::stdout()
        .write_all(json.as_bytes())
        .map_err(|e| e.to_string())?;
    io::stdout().write_all(b"\n").map_err(|e| e.to_string())?;
    Ok(())
}

fn print_help() {
    println!(
        "adaptive-rl-quant-rust — optional Rust accelerators for adaptive-rl-quant\n\
         \n\
         Commands:\n\
           sim-eval   Read EvaluateRequest JSON from stdin; write metrics JSON to stdout\n\
           help       Show this message\n\
         \n\
         Python orchestrator stays canonical; build with scripts/build_rust.sh"
    );
}
